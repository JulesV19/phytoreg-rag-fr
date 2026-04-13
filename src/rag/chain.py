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

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_ollama import ChatOllama
from langchain_qdrant import QdrantVectorStore

from src.embeddings import MxbaiEmbeddings
from qdrant_client import QdrantClient

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "phyto_docs"
EMBEDDING_MODEL = "mxbai-embed-large"
LLM_MODEL = "mistral:7b"

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

Champs à extraire :
- "intent" : type de question parmi :
    "usage_check"     → peut-on utiliser tel produit sur telle culture ? OU quelle est la dose/ZNT/DAR d'un produit spécifique ?
    "product_list"    → quels produits sont disponibles pour tel usage, nuisible, ou type de produit (fongicide, herbicide…) ?
    "regulation"      → questions générales sur les règles, arrêtés, délais de rentrée, ZNT sans produit précis, obligations légales
    "substance_check" → une substance active est-elle approuvée en Europe ?
    "hors_domaine"    → salutation ou question sans rapport avec le phytosanitaire
- "nuisible"    : ravageur ou maladie cible en minuscules (ex: "pucerons", "mildiou") ou null
- "culture"     : plante hôte en minuscules (ex: "blé", "vigne", "noyer") ou null
- "produit"     : nom commercial EN MAJUSCULES tel qu'il apparaît dans la question ou null
- "amm"         : numéro AMM à 7 chiffres sous forme de chaîne ou null
- "substance"   : molécule active en minuscules (ex: "fosétyl-aluminium") ou null
- "biocontrole" : true si la question porte spécifiquement sur les produits de biocontrôle, false sinon
- "type_produit": "fongicide", "herbicide", "insecticide", "acaricide" ou null si non mentionné

Exemples :
Question: "Puis-je utiliser l'ALIETTE FLASH sur des noyers ?"
JSON: {{"intent": "usage_check", "nuisible": null, "culture": "noyer", "produit": "ALIETTE FLASH", "amm": null, "substance": null, "biocontrole": false, "type_produit": null}}

Question: "Quelle est la ZNT aquatique du DECIS PROTECH ?"
JSON: {{"intent": "usage_check", "nuisible": null, "culture": null, "produit": "DECIS PROTECH", "amm": null, "substance": null, "biocontrole": false, "type_produit": null}}

Question: "Quels produits de biocontrôle sont autorisés contre les pucerons ?"
JSON: {{"intent": "product_list", "nuisible": "pucerons", "culture": null, "produit": null, "amm": null, "substance": null, "biocontrole": true, "type_produit": null}}

Question: "Quels fongicides sont autorisés sur le colza ?"
JSON: {{"intent": "product_list", "nuisible": null, "culture": "colza", "produit": null, "amm": null, "substance": null, "biocontrole": false, "type_produit": "fongicide"}}

Question: "Quel est le délai de rentrée après un H319 ?"
JSON: {{"intent": "regulation", "nuisible": null, "culture": null, "produit": null, "amm": null, "substance": null, "biocontrole": false, "type_produit": null}}

Question: "Le Fosétyl-Al est-il approuvé en Europe ?"
JSON: {{"intent": "substance_check", "nuisible": null, "culture": null, "produit": null, "amm": null, "substance": "fosétyl-al", "biocontrole": false, "type_produit": null}}

Question : {question}
JSON :"""


def parse_query_with_llm(question: str, parser_llm: ChatOllama) -> dict | None:
    """
    Utilise le LLM (temperature=0) pour extraire les entités structurées de la question.
    Retourne None si le parsing échoue → le fallback regex prend le relais.
    """
    try:
        prompt = QUERY_PARSER_PROMPT.format(question=question)
        response = parser_llm.invoke(prompt)
        content = response.content.strip()

        # Extraire le JSON même si le LLM ajoute du texte ou des balises autour
        json_match = re.search(r"\{[^{}]+\}", content, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            if "intent" in parsed:
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
        r"\b\w+(?:ate|yl|ine|ène|ene|anol|nol|ium|ose|ide|Al)\b",
        question, re.IGNORECASE,
    ))

    biocontrole = any(w in q for w in [
        "biocontrôle", "biocontrole", "bacillus", "trichoderma", "phéromone",
        "substance naturelle", "liste l.253",
    ])

    is_regulation = any(w in q for w in [
        "arrêté", "délai de rentrée", "rentrée", "znt", "rinçage", "effluent",
        "stockage", "registre", "epi", "délai", "interdit", "obligation",
        "règle", "fond de cuve", "vent", "beaufort",
    ])
    is_substance = has_molecule or any(w in q for w in [
        "substance active", "approuvé", "approuvée", "europe", "efsa", "substitution",
    ])
    is_specific_product = bool(amm) or bool(produit)

    # ZNT/DAR/dose d'un produit précis → les valeurs sont dans les chunks AMM, pas dans l'arrêté.
    # On distingue : "ZNT générale" (regulation) vs "ZNT du produit X" (usage_check).
    product_value_query = is_regulation and is_specific_product and any(w in q for w in [
        "znt", "dar", "délai avant récolte", "dose retenue", "dose maximale",
    ])

    if product_value_query:
        intent = "usage_check"
    elif is_regulation:
        intent = "regulation"
    elif is_substance:
        intent = "substance_check"
    elif is_specific_product:
        intent = "usage_check"
    elif biocontrole:
        intent = "product_list"
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
    vectorstore: QdrantVectorStore,
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

    # ── Hors domaine → contexte vide, Mistral répond naturellement ───────────
    if intent == "hors_domaine":
        return []

    # ── Usage check : "Puis-je utiliser X sur Y ?" ───────────────────────────
    elif intent == "usage_check":
        must_ok = [{"key": "metadata.etat_usage", "match": {"value": "Autorisé"}}]
        if amm:
            must_ok.append({"key": "metadata.numero_amm", "match": {"value": amm}})
        elif produit:
            must_ok.append({"key": "metadata.nom_produit", "match": {"value": produit}})

        docs = vectorstore.similarity_search(question, k=k, filter={"must": must_ok})
        add_docs(docs)

        # Fallback : si aucun usage autorisé → chercher tous états (produit retiré)
        if not all_docs and (amm or produit):
            must_all = [m for m in must_ok if "etat_usage" not in str(m)]
            if must_all:
                docs = vectorstore.similarity_search(question, k=k, filter={"must": must_all})
                add_docs(docs)

    # ── Product list : "Quels produits pour X sur Y ?" ───────────────────────
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

        docs = vectorstore.similarity_search(search_query, k=k, filter={"must": must})
        add_docs(docs)

        # Fallback : si la query focalisée retourne trop peu, essayer la question complète
        if len(all_docs) < 2:
            docs2 = vectorstore.similarity_search(question, k=k, filter={"must": must})
            add_docs(docs2)

        # Pour les questions biocontrôle : ajouter le contexte réglementaire (note DGAL)
        if biocontrole:
            docs_bio = vectorstore.similarity_search(
                question, k=3,
                filter={"must": [{"key": "metadata.type", "match": {"value": "biocontrole"}}]},
            )
            add_docs(docs_bio)

    # ── Réglementation : arrêté, ZNT, délais de rentrée ──────────────────────
    elif intent == "regulation":
        filtre_arrete = {"must": [{"key": "metadata.source", "match": {"value": "arrete_2017"}}]}

        # Double reformulation pour maximiser le rappel sur des articles précis
        docs1 = vectorstore.similarity_search(question, k=5, filter=filtre_arrete)
        add_docs(docs1)

        query_concrete = question + " valeur durée heures article obligation"
        docs2 = vectorstore.similarity_search(query_concrete, k=5, filter=filtre_arrete)
        add_docs(docs2)

    # ── Substance active CE : approbation européenne ──────────────────────────
    elif intent == "substance_check":
        # Enrichir la requête avec le nom de la substance si extrait
        search_query = f"{substance} {question}" if substance else question
        docs = vectorstore.similarity_search(
            search_query, k=5,
            filter={"must": [{"key": "metadata.type", "match": {"value": "substance_active"}}]},
        )
        add_docs(docs)

        # Compléter avec les usages AMM de cette substance dans la base nationale
        docs_amm = vectorstore.similarity_search(
            search_query, k=3,
            filter={"must": [{"key": "metadata.source", "match": {"value": "amm_xml"}}]},
        )
        add_docs(docs_amm)

    # ── Fallback global si trop peu de résultats ──────────────────────────────
    if 0 < len(all_docs) < 2:
        docs = vectorstore.similarity_search(question, k=k - len(all_docs))
        add_docs(docs)

    return all_docs[:k]


def retrieve_documents(
    vectorstore: QdrantVectorStore,
    question: str,
    parser_llm: ChatOllama,
    k: int = 6,
) -> list[Document]:
    """
    Point d'entrée du retriever.
    1. Extraction d'entités par LLM (temperature=0, num_ctx court)
    2. Fallback regex si le LLM échoue ou retourne un JSON invalide
    3. Délègue à retrieve_by_entities()
    """
    entities = parse_query_with_llm(question, parser_llm)
    if entities is None:
        entities = analyze_query_fallback(question)
    return retrieve_by_entities(vectorstore, question, entities, k)


def format_context(docs: list[Document]) -> str:
    """Formate les documents récupérés en contexte lisible pour le LLM."""
    parts = []
    source_labels = {
        "amm_xml": "Base AMM nationale",
        "arrete_2017": "Arrêté du 4 mai 2017",
        "decision_amm_individuelle": "Décision AMM individuelle",
        "note_biocontrole": "Note DGAL biocontrôle",
        "substances_actives_xlsx": "Base substances actives CE",
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

def build_vectorstore() -> QdrantVectorStore:
    client = QdrantClient(url=QDRANT_URL)
    embeddings = MxbaiEmbeddings(model=EMBEDDING_MODEL)
    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )


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
    vectorstore: QdrantVectorStore,
    llm: ChatOllama,
    parser_llm: ChatOllama,
):
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT),
    ])

    def retrieve_and_format(question: str) -> dict:
        docs = retrieve_documents(vectorstore, question, parser_llm)
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
        self.vectorstore = build_vectorstore()
        self.llm = build_llm()
        self.parser_llm = build_parser_llm()
        self.chain = build_rag_chain(self.vectorstore, self.llm, self.parser_llm)

    def ask(self, question: str) -> dict:
        """
        Pose une question et retourne la réponse + les sources utilisées.
        Returns:
            {
                "answer": str,
                "sources": list[dict],
            }
        """
        docs = retrieve_documents(self.vectorstore, question, self.parser_llm)

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
        answer = (prompt | self.llm | StrOutputParser()).invoke(
            {"context": context, "question": question}
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
        docs = retrieve_documents(self.vectorstore, question, self.parser_llm)

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
        yield from chain.stream({"context": context, "question": question})
