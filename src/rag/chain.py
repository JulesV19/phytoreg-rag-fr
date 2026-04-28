"""
Chaîne RAG LangChain pour le LLM phytosanitaire.

Architecture :
1. Query Parser (LLM, temperature=0) : extrait les entités structurées de la question
   → intent, nuisible, culture, produit, amm, substance, biocontrole
   → fallback regex si le JSON est mal formé
2. Retriever par intent : filtres Qdrant précis selon les entités extraites
   → une stratégie claire par intent, pas de branches parallèles qui s'annulent
3. Génération : Mistral 7b via Ollama avec prompt système spécialisé
"""

import json
import re
import time

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_ollama import ChatOllama
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode

from src.embeddings import MxbaiEmbeddings
from src.reranker import Reranker
from qdrant_client import QdrantClient

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "phyto_docs"
EMBEDDING_MODEL = "mxbai-embed-large"
LLM_MODEL = "mistral:7b"
SPARSE_VECTOR_NAME = "sparse"

K_RETRIEVE = 20   # pool candidat avant re-ranking
K_FINAL = 6       # résultats après re-ranking

_ARVALIS_FILTER = {
    "must": [{"key": "metadata.source", "match": {
        "any": ["arvalis_produits", "arvalis_varietes", "arvalis_fertilisants", "arvalis_couverts"]
    }}]
}

_OLLAMA_RETRIES = 3
_OLLAMA_RETRY_DELAY = 1.5   # secondes entre deux tentatives


def _call_with_retry(fn, retries: int = _OLLAMA_RETRIES, delay: float = _OLLAMA_RETRY_DELAY):
    """
    Exécute fn() et réessaie jusqu'à `retries` fois en cas d'exception.
    Couvre les erreurs transitoires d'Ollama : connexion refusée, timeout, surcharge.
    Lève la dernière exception si toutes les tentatives échouent.
    """
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(delay)
    raise last_exc

# ─── Prompt système ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Tu es un assistant expert en réglementation phytosanitaire française, \
spécialement conçu pour aider les agriculteurs.

Tes règles absolues :
1. Réponds UNIQUEMENT à partir des documents fournis dans le contexte.
2. Si une information n'est pas dans le contexte, dis-le clairement : \
"Je n'ai pas cette information dans mes documents."
3. Cite toujours tes sources : numéro d'AMM, article de loi (ex: Art. 3-II), ou nom du document.
4. Utilise un langage clair et pratique, adapté aux agriculteurs.
5. Pour les informations réglementaires critiques (délais, doses, ZNT), \
sois précis et cite les valeurs exactes trouvées dans le contexte.
6. Si la question concerne un produit RETIRÉ, signale-le explicitement.
7. Si l'utilisateur demande des produits AUTORISÉS pour un usage donné et que \
TOUS les produits trouvés dans le contexte sont RETIRÉS, réponds : \
"Aucun produit actuellement autorisé pour cet usage n'a été trouvé dans la base. \
Les produits retirés ne peuvent plus être utilisés légalement." \
Ne liste jamais des produits retirés comme réponse à une question sur les produits autorisés.

Règles de vocabulaire phytosanitaire importantes :
- Une CULTURE (ou plante hôte) est la plante sur laquelle on applique le produit \
(ex: blé, vigne, noyer, artichaut). C'est différent du nuisible.
- Un NUISIBLE (ou organisme cible) est la maladie, le ravageur ou la mauvaise herbe \
que l'on combat (ex: oïdium, pucerons, pythiacées). Ce n'est PAS une culture.
- Un USAGE combine toujours : culture × nuisible × méthode d'application.
- Quand on te demande les "cultures autorisées", liste UNIQUEMENT les plantes hôtes, \
jamais les nuisibles.
- Sois exhaustif : liste TOUTES les cultures trouvées dans le contexte, sans en omettre.

Contexte documentaire :
{context}
"""

HUMAN_PROMPT = "{question}"

# ─── Query Parser LLM ─────────────────────────────────────────────────────────

QUERY_PARSER_PROMPT = """\
Tu es un extracteur d'entités phytosanitaires. Analyse la question et retourne UNIQUEMENT \
un objet JSON valide, sans texte autour, sans balises markdown.

IMPORTANT : le champ "intent" doit être EXACTEMENT l'une de ces 5 valeurs, rien d'autre :
  "usage_check"     → question sur un produit précis (autorisation, dose, ZNT, DAR, substance active, numéro AMM, statut)
  "product_list"    → quels produits sont disponibles pour un usage, nuisible, culture, ou type (fongicide…)
  "regulation"      → règles générales (arrêtés, délais de rentrée, ZNT globale, obligations) sans produit précis
  "substance_check" → une substance active est-elle approuvée en Europe ?
  "hors_domaine"    → salutation ou question sans rapport avec le phytosanitaire

Autres champs :
- "nuisible"    : ravageur ou maladie cible en minuscules (ex: "pucerons", "mildiou") ou null
- "culture"     : plante hôte en minuscules (ex: "blé", "vigne", "noyer") ou null
- "produit"     : nom commercial EN MAJUSCULES tel qu'il apparaît dans la question ou null
- "amm"         : numéro AMM à 7 chiffres sous forme de chaîne ou null
- "substance"   : molécule active en minuscules (ex: "fosétyl-aluminium") ou null
- "biocontrole" : true si la question porte sur les produits de biocontrôle, false sinon (jamais null)
- "type_produit": "fongicide", "herbicide", "insecticide", "acaricide" ou null

Exemples :
Question: "Puis-je utiliser l'ALIETTE FLASH sur des noyers ?"
JSON: {{"intent": "usage_check", "nuisible": null, "culture": "noyer", "produit": "ALIETTE FLASH", "amm": null, "substance": null, "biocontrole": false, "type_produit": null}}

Question: "Quel est le numéro AMM du produit AQ 10 ?"
JSON: {{"intent": "usage_check", "nuisible": null, "culture": null, "produit": "AQ 10", "amm": null, "substance": null, "biocontrole": true, "type_produit": null}}

Question: "Quelle est la substance active de SERENADE ASO ?"
JSON: {{"intent": "usage_check", "nuisible": null, "culture": null, "produit": "SERENADE ASO", "amm": null, "substance": null, "biocontrole": true, "type_produit": null}}

Question: "Quel est le statut du produit BOTANIGARD OD ?"
JSON: {{"intent": "usage_check", "nuisible": null, "culture": null, "produit": "BOTANIGARD OD", "amm": null, "substance": null, "biocontrole": true, "type_produit": null}}

Question: "Quels produits de biocontrôle sont autorisés contre les pucerons ?"
JSON: {{"intent": "product_list", "nuisible": "pucerons", "culture": null, "produit": null, "amm": null, "substance": null, "biocontrole": true, "type_produit": null}}

Question: "Quelles sont les catégories de produits de biocontrôle ?"
JSON: {{"intent": "regulation", "nuisible": null, "culture": null, "produit": null, "amm": null, "substance": null, "biocontrole": true, "type_produit": null}}

Question: "Quels fongicides sont autorisés sur le colza ?"
JSON: {{"intent": "product_list", "nuisible": null, "culture": "colza", "produit": null, "amm": null, "substance": null, "biocontrole": false, "type_produit": "fongicide}}

Question: "Quel est le délai de rentrée après un H319 ?"
JSON: {{"intent": "regulation", "nuisible": null, "culture": null, "produit": null, "amm": null, "substance": null, "biocontrole": false, "type_produit": null}}

Question: "Le Fosétyl-Al est-il approuvé en Europe ?"
JSON: {{"intent": "substance_check", "nuisible": null, "culture": null, "produit": null, "amm": null, "substance": "fosétyl-al", "biocontrole": false, "type_produit": null}}

Question: "Quelle est la date d'expiration de l'approbation du 1-naphthylacetamide ?"
JSON: {{"intent": "substance_check", "nuisible": null, "culture": null, "produit": null, "amm": null, "substance": "1-naphthylacetamide", "biocontrole": false, "type_produit": null}}

Question: "Quel est le délai de rentrée du fongicide AKONPLI ?"
JSON: {{"intent": "usage_check", "nuisible": null, "culture": null, "produit": "AKONPLI", "amm": null, "substance": null, "biocontrole": false, "type_produit": "fongicide"}}

Question: "Le PROSARO XPERT est-il utilisable en agriculture biologique ?"
JSON: {{"intent": "usage_check", "nuisible": null, "culture": null, "produit": "PROSARO XPERT", "amm": null, "substance": null, "biocontrole": false, "type_produit": null}}

Question: "Quel est l'obtenteur de la variété de blé tendre APACHE ?"
JSON: {{"intent": "usage_check", "nuisible": null, "culture": "blé tendre", "produit": "APACHE", "amm": null, "substance": null, "biocontrole": false, "type_produit": null}}

Question: "À quelle famille botanique appartient la phacélie ?"
JSON: {{"intent": "product_list", "nuisible": null, "culture": "phacélie", "produit": null, "amm": null, "substance": null, "biocontrole": false, "type_produit": null}}

Question: "Quelle est la teneur en azote du fertilisant AGRISTART MAGNUM ?"
JSON: {{"intent": "usage_check", "nuisible": null, "culture": null, "produit": "AGRISTART MAGNUM", "amm": null, "substance": null, "biocontrole": false, "type_produit": null}}

Question: "Quelle est la densité de semis recommandée pour la phacélie ?"
JSON: {{"intent": "product_list", "nuisible": null, "culture": "phacélie", "produit": null, "amm": null, "substance": null, "biocontrole": false, "type_produit": null}}

Question: "Quelle est la teneur en azote du fertilisant Agronex ?"
JSON: {{"intent": "usage_check", "nuisible": null, "culture": null, "produit": "AGRONEX", "amm": null, "substance": null, "biocontrole": false, "type_produit": null}}

Question : {question}
JSON :"""


_VALID_INTENTS = {"usage_check", "product_list", "regulation", "substance_check", "hors_domaine"}


def parse_query_with_llm(question: str, parser_llm: ChatOllama) -> dict | None:
    """
    Utilise le LLM (temperature=0) pour extraire les entités structurées de la question.
    Retourne None si le parsing échoue → le fallback regex prend le relais.
    """
    try:
        prompt = QUERY_PARSER_PROMPT.format(question=question)
        response = _call_with_retry(lambda: parser_llm.invoke(prompt))
        content = response.content.strip()

        # Extraire le JSON même si le LLM ajoute du texte ou des balises autour
        json_match = re.search(r"\{[^{}]+\}", content, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            if parsed.get("intent") in _VALID_INTENTS:
                return parsed
    except Exception:
        pass
    return None


# ─── Fallback : extracteur regex ──────────────────────────────────────────────

def analyze_query_fallback(question: str) -> dict:
    """
    Extracteur regex de secours si le LLM rate le parsing.
    Moins précis mais toujours fonctionnel.
    """
    q = question.lower()

    amm_match = re.search(r"\b(\d{7})\b", question)
    amm = amm_match.group(1) if amm_match else None

    product_match = re.search(r"\b([A-Z][A-Z0-9\s\-]{2,}[A-Z0-9])\b", question)
    produit = product_match.group(1).strip() if product_match else None

    has_molecule = bool(re.search(
        r"\b\w{3,}(?:ate|yl|ine|ène|ene|anol|nol|ium|amide)\b",
        question, re.IGNORECASE,
    ))

    biocontrole = any(w in q for w in [
        "biocontrôle", "biocontrole", "bacillus", "trichoderma", "phéromone",
        "substance naturelle", "liste l.253",
    ])

    is_greeting = any(w in q for w in [
        "bonjour", "bonsoir", "salut", "hello", "coucou", "comment vas", "comment ça va",
    ])

    is_regulation = any(w in q for w in [
        "arrêté", "délai de rentrée", "rentrée", "znt", "rinçage", "rincer", "effluent",
        "stockage", "registre", "epi", "délai", "interdit", "obligation",
        "fond de cuve", "vent", "beaufort", "code rural", "l.253",
        "caniveau", "avaloir",
    ])
    is_substance = has_molecule or any(w in q for w in [
        "substance active", "substances actives", "approuvé", "approuvée",
        "europ", "efsa", "substitution", "rapporteur", "rms", "état membre",
    ])
    is_specific_product = bool(amm) or bool(produit)

    # Valeur réglementaire d'un produit précis → les valeurs sont dans les chunks AMM/ARVALIS.
    # On distingue : "ZNT générale" (regulation) vs "ZNT du produit X" (usage_check).
    # Règle : si un produit est nommé ET qu'on demande une valeur spécifique → usage_check.
    product_value_query = is_regulation and is_specific_product and any(w in q for w in [
        "znt", "dar", "délai avant récolte", "dose retenue", "dose maximale",
        "rentrée", "cmr", "agriculture biologique", " ab ", "biologique",
    ])

    is_phyto = is_specific_product or is_regulation or is_substance or biocontrole or any(w in q for w in [
        "produit", "traitement", "pulvéris", "pesticide", "phytosanitaire",
        "fongicide", "herbicide", "insecticide", "ravageur", "nuisible",
        "récolte", "usage", "autorisation", "amm",
        # termes agronomiques ARVALIS (variétés, fertilisants, couverts)
        "variété", "variete", "obtenteur", "inscription",
        "fertilisant", "engrais", "azote", "phosphore",
        "couvert", "semis", "botanique", "densité",
    ])

    if is_greeting or not is_phyto:
        intent = "hors_domaine"
    elif product_value_query:
        intent = "usage_check"
    elif is_regulation:
        intent = "regulation"
    elif is_specific_product:
        intent = "usage_check"
    elif is_substance:
        intent = "substance_check"
    elif biocontrole:
        # Question sur un usage précis (nuisible/culture mentionné) → product_list
        # Question conceptuelle sur le biocontrôle (catégories, critères, avantages) → regulation
        has_usage_context = (
            bool(produit) or bool(amm)
            or bool(re.search(r"\bcontre\b", q))
            or bool(re.search(r"\bsur (?:le|la|les|l')", q))
        )
        intent = "product_list" if has_usage_context else "regulation"
    else:
        intent = "product_list"

    nuisible_match = re.search(
        r"contre (?:les?|la|l') ?([a-zéèàâùî\s\-]+?)(?:\s*[?]|$|,|\s+sur\b|\s+en\b)",
        q,
    )
    nuisible = nuisible_match.group(1).strip() if nuisible_match else None

    # Avec article : "sur le colza", "sur la vigne"
    culture_match = re.search(
        r"sur (?:les?|la|l') ?([a-zéèàâùî\s\-]+?)(?:\s*[?]|$|,|\s+contre\b)",
        q,
    )
    # Sans article en fin de question : "autorisé sur blé ?", "utilisé sur pommier ?"
    if not culture_match:
        culture_match = re.search(
            r"sur ([a-zéèàâùî\-]+)\s*[?]?$",
            q,
        )
    culture = culture_match.group(1).strip() if culture_match else None

    # Pluriel géré par s? — "fongicides" → "fongicide"
    type_produit_match = re.search(
        r"\b(fongicides?|herbicides?|insecticides?|acaricides?|nématicides?|molluscicides?|rodenticides?)\b",
        q,
    )
    type_produit = type_produit_match.group(1).rstrip("s") if type_produit_match else None

    return {
        "intent": intent,
        "nuisible": nuisible,
        "culture": culture,
        "produit": produit,
        "amm": amm,
        "substance": None,
        "biocontrole": biocontrole,
        "type_produit": type_produit,
    }


# ─── Retriever par intent ─────────────────────────────────────────────────────

def retrieve_by_entities(
    vs_hybrid: QdrantVectorStore,
    vs_dense: QdrantVectorStore,
    question: str,
    entities: dict,
    k: int = 6,
) -> list[Document]:
    """
    Construit et exécute les requêtes Qdrant selon les entités extraites.

    Principe : metadata filter PRÉCIS en premier, la recherche sémantique
    opère à l'intérieur de l'espace filtré pour gérer nuisible/culture.
    Les chunks AMM contiennent "contre Pucerons" en texte naturel, donc
    la similarité vectorielle suffit pour les discriminer une fois filtrés
    sur etat_usage=Autorisé.

    Routage hybride (dense + sparse BM25) vs dense seul :
    - usage_check, substance_check → hybride : le matching exact sur noms
      de produits, numéros AMM et substances actives est critique.
    - product_list, regulation → dense seul : la query sémantique focalisée
      et la double reformulation suffisent ; le BM25 n'apporte pas de gain
      sur des termes généraux comme "mildiou vigne" ou "délai de rentrée".
    """
    intent = entities.get("intent", "hors_domaine")
    nuisible = entities.get("nuisible")
    culture = entities.get("culture")
    produit = entities.get("produit")
    amm = entities.get("amm")
    substance = entities.get("substance")
    biocontrole = entities.get("biocontrole", False)

    all_docs: list[Document] = []
    seen: set[str] = set()

    def add_docs(docs):
        for doc in docs:
            key = doc.page_content[:100]
            if key not in seen:
                seen.add(key)
                all_docs.append(doc)

    # ── Hors domaine → fallback sémantique sur les fiches ARVALIS ────────────
    # Salutations pures → contexte vide, Mistral répond naturellement.
    # Questions agricoles non-réglementaires (variétés, fertilisants, couverts…) → tenter ARVALIS.
    if intent == "hors_domaine":
        has_entities = any(entities.get(f) for f in ("produit", "culture", "nuisible", "amm", "substance"))
        # Rattraper les entités manquées par le LLM : mots en majuscules ≥ 3 lettres
        # (noms de variétés, fertilisants, couverts qui ne sont pas des produits phyto connus)
        if not has_entities:
            has_entities = bool(re.search(r'\b[A-Z][A-Z0-9\-]{2,}\b', question))
        if not has_entities:
            return []
        docs = vs_dense.similarity_search(question, k=k, filter=_ARVALIS_FILTER)
        add_docs(docs)
        return all_docs[:k]

    # ── Usage check : "Puis-je utiliser X sur Y ?" ───────────────────────────
    # Hybride : le matching exact sur nom produit / numéro AMM est critique.
    elif intent == "usage_check":
        k_arvalis = max(2, k // 2)
        k_primary = k - k_arvalis

        # 1. Fiches ARVALIS EN PREMIER — slots garantis quelle que soit la taille du pool AMM
        docs_arv = vs_dense.similarity_search(question, k=k_arvalis, filter=_ARVALIS_FILTER)
        add_docs(docs_arv)

        # 2. Base AMM nationale
        must_ok = [{"key": "metadata.etat_usage", "match": {"value": "Autorisé"}}]
        if amm:
            must_ok.append({"key": "metadata.numero_amm", "match": {"value": amm}})
        elif produit:
            must_ok.append({"key": "metadata.nom_produit", "match": {"value": produit}})

        docs = vs_hybrid.similarity_search(question, k=k_primary, filter={"must": must_ok})
        add_docs(docs)

        # Fallback : si aucun usage autorisé → chercher tous états (produit retiré)
        if len(all_docs) <= k_arvalis and (amm or produit):
            must_all = [m for m in must_ok if "etat_usage" not in str(m)]
            if must_all:
                docs = vs_hybrid.similarity_search(question, k=k_primary, filter={"must": must_all})
                add_docs(docs)

        # 3. Note biocontrôle — BM25 discrimine les variantes de noms (BELOUKHA vs BELOUKHA GARDEN)
        if produit:
            docs_bio = vs_hybrid.similarity_search(
                question, k=2,
                filter={"must": [{"key": "metadata.source", "match": {"value": "note_biocontrole"}}]},
            )
            add_docs(docs_bio)

    # ── Product list : "Quels produits pour X sur Y ?" ───────────────────────
    # Dense seul : la query focalisée sur les entités suffit, BM25 n'aide pas
    # sur des termes généraux comme "mildiou vigne".
    elif intent == "product_list":
        type_produit = entities.get("type_produit")

        must = [
            {"key": "metadata.etat_usage", "match": {"value": "Autorisé"}},
            {"key": "metadata.source",     "match": {"value": "amm_xml"}},
        ]
        # Note : le champ <type-produit> du XML vaut toujours "PPP", pas "Fongicide"/"Herbicide".
        # On ne peut donc pas filtrer sur type_produit en metadata.
        # On l'intègre dans la query sémantique pour orienter l'embedding.

        # Query formatée pour coller au texte des chunks AMM :
        # "Le produit Z est autorisé sur Colza contre Sclerotinia par …"
        # type_produit en préfixe oriente l'embedding vers les bons usages.
        parts = []
        if type_produit:
            parts.append(type_produit)
        if culture:
            parts.append(f"autorisé sur {culture}")
        if nuisible:
            parts.append(f"contre {nuisible}")
        search_query = " ".join(parts) if parts else question

        k_arvalis = max(2, k // 3)
        k_amm = max(2, (k - k_arvalis) // 2) if biocontrole else k - k_arvalis

        # 1. Fiches ARVALIS EN PREMIER — slots garantis
        docs_arv = vs_dense.similarity_search(question, k=k_arvalis, filter=_ARVALIS_FILTER)
        add_docs(docs_arv)

        # 2. Base AMM nationale
        docs = vs_dense.similarity_search(search_query, k=k_amm, filter={"must": must})
        add_docs(docs)

        # Fallback : si la query focalisée retourne trop peu, essayer la question complète
        if len(all_docs) < k_arvalis + 2:
            docs2 = vs_dense.similarity_search(question, k=k_amm, filter={"must": must})
            add_docs(docs2)

        # 3. biocontrôle=True → note DGAL (texte réglementaire + produits)
        if biocontrole:
            docs_bio = vs_dense.similarity_search(
                question, k=k - len(all_docs) + 2,
                filter={"must": [{"key": "metadata.source", "match": {"value": "note_biocontrole"}}]},
            )
            add_docs(docs_bio)

    # ── Réglementation : arrêté, ZNT, délais de rentrée ──────────────────────
    # Hybride : les termes spécifiques de la question (ex: "24 heures", "sol gelé", "rizière")
    # matchent exactement via BM25 les articles correspondants de l'arrêté.
    elif intent == "regulation":
        filtre_arrete = {"must": [{"key": "metadata.source", "match": {"value": "arrete_2017"}}]}

        # Produit nommé dans une question réglementaire → ARVALIS EN PREMIER (délai de rentrée, CMR, AB…)
        if produit:
            docs_arv = vs_dense.similarity_search(question, k=2, filter=_ARVALIS_FILTER)
            add_docs(docs_arv)
            docs_bio = vs_hybrid.similarity_search(
                produit, k=2,
                filter={"must": [{"key": "metadata.source", "match": {"value": "note_biocontrole"}}]},
            )
            add_docs(docs_bio)

        # Questions réglementaires sur le biocontrôle (Code rural, liste DGAL…) → note DGAL EN PREMIER
        # Hybride pour que BM25 booste les termes spécifiques ("L.253-6", "macro-organismes"…).
        if biocontrole:
            docs_bio = vs_hybrid.similarity_search(
                question, k=3,
                filter={"must": [{"key": "metadata.type", "match": {"value": "biocontrole"}}]},
            )
            add_docs(docs_bio)

        # Double reformulation hybride sur l'arrêté (slots restants).
        k_arrete = k - len(all_docs)
        if k_arrete > 0:
            docs1 = vs_hybrid.similarity_search(question, k=max(1, k_arrete // 2), filter=filtre_arrete)
            add_docs(docs1)
            query_concrete = question + " valeur durée heures article obligation"
            docs2 = vs_hybrid.similarity_search(query_concrete, k=k_arrete, filter=filtre_arrete)
            add_docs(docs2)

    # ── Substance active CE : approbation européenne ──────────────────────────
    # Hybride : les noms de molécules (Fosétyl-Al, glyphosate) sont des termes
    # rares que le BM25 retrouve exactement.
    elif intent == "substance_check":
        search_query = f"{substance} {question}" if substance else question

        if produit:
            # "Quelle est la substance active du produit X ?" → note_biocontrole EN PREMIER par BM25
            docs_bio = vs_hybrid.similarity_search(
                produit, k=3,
                filter={"must": [{"key": "metadata.source", "match": {"value": "note_biocontrole"}}]},
            )
            add_docs(docs_bio)
            docs = vs_hybrid.similarity_search(
                search_query, k=k - len(all_docs),
                filter={"must": [{"key": "metadata.type", "match": {"value": "substance_active"}}]},
            )
            add_docs(docs)
        elif biocontrole:
            # "substance candidate à la substitution peut-il figurer sur la liste biocontrole ?"
            # → note_biocontrole EN PREMIER pour éviter l'épuisement des slots par substances_actives
            docs_bio = vs_hybrid.similarity_search(
                question, k=3,
                filter={"must": [{"key": "metadata.source", "match": {"value": "note_biocontrole"}}]},
            )
            add_docs(docs_bio)
            docs = vs_hybrid.similarity_search(
                search_query, k=k - len(all_docs),
                filter={"must": [{"key": "metadata.type", "match": {"value": "substance_active"}}]},
            )
            add_docs(docs)
        else:
            # Question pure sur substance européenne (S1-S5 type)
            docs = vs_hybrid.similarity_search(
                search_query, k=5,
                filter={"must": [{"key": "metadata.type", "match": {"value": "substance_active"}}]},
            )
            add_docs(docs)
            docs_amm = vs_hybrid.similarity_search(
                search_query, k=3,
                filter={"must": [{"key": "metadata.source", "match": {"value": "amm_xml"}}]},
            )
            add_docs(docs_amm)

    # ── Fallback global si trop peu de résultats ──────────────────────────────
    if 0 < len(all_docs) < 2:
        docs = vs_dense.similarity_search(question, k=k - len(all_docs))
        add_docs(docs)

    return all_docs[:k]


def retrieve_documents(
    vs_hybrid: QdrantVectorStore,
    vs_dense: QdrantVectorStore,
    question: str,
    parser_llm: ChatOllama,
    reranker: Reranker | None = None,
    k: int = K_FINAL,
) -> list[Document]:
    """
    Point d'entrée du retriever.
    1. Extraction d'entités par LLM (temperature=0, num_ctx court)
    2. Fallback regex si le LLM échoue ou retourne un JSON invalide
    3. Retrieval large (K_RETRIEVE candidats) si reranker présent, k sinon
    4. Re-ranking cross-encoder → top k
    """
    entities = parse_query_with_llm(question, parser_llm)
    if entities is None:
        entities = analyze_query_fallback(question)
    k_retrieve = K_RETRIEVE if reranker else k
    candidates = retrieve_by_entities(vs_hybrid, vs_dense, question, entities, k=k_retrieve)
    if reranker:
        return reranker.rerank(question, candidates, top_k=k)
    return candidates


def format_context(docs: list[Document]) -> str:
    """Formate les documents récupérés en contexte lisible pour le LLM."""
    parts = []
    source_labels = {
        "amm_xml": "Base AMM nationale",
        "arrete_2017": "Arrêté du 4 mai 2017",
        "decision_amm_individuelle": "Décision AMM individuelle",
        "note_biocontrole": "Note DGAL biocontrôle",
        "substances_actives_xlsx": "Base substances actives CE",
        "arvalis_produits": "Fiche produit ARVALIS",
        "arvalis_varietes": "Fiche variété ARVALIS",
        "arvalis_fertilisants": "Fiche fertilisant ARVALIS",
        "arvalis_couverts": "Fiche couvert ARVALIS",
    }
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "inconnu")
        label = source_labels.get(source, source)
        parts.append(f"--- Source {i} [{label}] ---\n{doc.page_content}")
    return "\n\n".join(parts)


# ─── Message contexte vide ────────────────────────────────────────────────────

_NO_CONTEXT_RESPONSE = (
    "Je n'ai trouvé aucun document correspondant à cette question dans la base. "
    "Essayez de reformuler ou de préciser : nom du produit (en majuscules), "
    "nuisible visé (ex : pucerons, mildiou), ou culture concernée (ex : vigne, blé)."
)

# ─── Composants LangChain ─────────────────────────────────────────────────────

def build_vectorstores() -> tuple[QdrantVectorStore, QdrantVectorStore]:
    """
    Retourne deux instances pointant sur la même collection :
    - vs_hybrid : dense + sparse BM25 — pour usage_check et substance_check
    - vs_dense  : dense seul — pour product_list et regulation
    """
    client = QdrantClient(url=QDRANT_URL, timeout=60)
    dense_emb = MxbaiEmbeddings(model=EMBEDDING_MODEL)
    sparse_emb = FastEmbedSparse(model_name="Qdrant/bm25")

    vs_hybrid = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=dense_emb,
        sparse_embedding=sparse_emb,
        retrieval_mode=RetrievalMode.HYBRID,
        sparse_vector_name=SPARSE_VECTOR_NAME,
    )
    vs_dense = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=dense_emb,
        retrieval_mode=RetrievalMode.DENSE,
    )
    return vs_hybrid, vs_dense


def build_llm() -> ChatOllama:
    """LLM de génération : température basse pour des réponses factuelles."""
    return ChatOllama(
        model=LLM_MODEL,
        temperature=0.1,
        num_ctx=4096,
    )


def build_parser_llm() -> ChatOllama:
    """LLM de parsing de requête : température 0 pour des sorties JSON déterministes."""
    return ChatOllama(
        model=LLM_MODEL,
        temperature=0.0,
        num_ctx=2048,   # le prompt de parsing + exemples tient dans 2k tokens
    )


# ─── Chaîne RAG principale ────────────────────────────────────────────────────

def build_rag_chain(
    vs_hybrid: QdrantVectorStore,
    vs_dense: QdrantVectorStore,
    llm: ChatOllama,
    parser_llm: ChatOllama,
):
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT),
    ])

    def retrieve_and_format(question: str) -> dict:
        docs = retrieve_documents(vs_hybrid, vs_dense, question, parser_llm)
        return {
            "context": format_context(docs),
            "question": question,
            "source_docs": docs,
        }

    chain = (
        RunnableLambda(retrieve_and_format)
        | RunnablePassthrough.assign(
            answer=prompt | llm | StrOutputParser()
        )
    )
    return chain


# ─── Interface simplifiée ─────────────────────────────────────────────────────

class PhytoRAG:
    """Interface haut niveau pour interroger le RAG phytosanitaire."""

    def __init__(self):
        self.vs_hybrid, self.vs_dense = build_vectorstores()
        self.llm = build_llm()
        self.parser_llm = build_parser_llm()
        self.reranker = Reranker()
        self.chain = build_rag_chain(self.vs_hybrid, self.vs_dense, self.llm, self.parser_llm)

    def ask(self, question: str) -> dict:
        """
        Pose une question et retourne la réponse + les sources utilisées.
        Returns:
            {
                "answer": str,
                "sources": list[dict],
            }
        """
        docs = retrieve_documents(self.vs_hybrid, self.vs_dense, question, self.parser_llm, self.reranker)

        # Guard anti-hallucination : si aucun document trouvé, réponse standard sans LLM.
        # Mistral 7b ignore "Réponds UNIQUEMENT à partir du contexte" quand le contexte
        # est vide et invente des produits, AMM et substances actives fictifs.
        if not docs:
            return {"answer": _NO_CONTEXT_RESPONSE, "sources": []}

        context = format_context(docs)
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ])
        chain_gen = prompt | self.llm | StrOutputParser()
        answer = _call_with_retry(
            lambda: chain_gen.invoke({"context": context, "question": question})
        )
        sources = [
            {
                "source": doc.metadata.get("source", ""),
                "type": doc.metadata.get("type", ""),
                "nom_produit": doc.metadata.get("nom_produit", ""),
                "numero_amm": doc.metadata.get("numero_amm", ""),
                "article": doc.metadata.get("article", ""),
                "section": doc.metadata.get("section", ""),
                "extrait": doc.page_content[:150] + "...",
            }
            for doc in docs
        ]
        return {"answer": answer, "sources": sources}

    def stream(self, question: str):
        """Génère la réponse en streaming."""
        docs = retrieve_documents(self.vs_hybrid, self.vs_dense, question, self.parser_llm, self.reranker)

        # Même guard : pas de LLM si contexte vide
        if not docs:
            yield _NO_CONTEXT_RESPONSE
            return

        context = format_context(docs)
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ])
        chain = prompt | self.llm | StrOutputParser()
        last_exc: Exception | None = None
        for attempt in range(_OLLAMA_RETRIES):
            try:
                yield from chain.stream({"context": context, "question": question})
                return
            except Exception as exc:
                last_exc = exc
                if attempt < _OLLAMA_RETRIES - 1:
                    time.sleep(_OLLAMA_RETRY_DELAY)
        raise last_exc
