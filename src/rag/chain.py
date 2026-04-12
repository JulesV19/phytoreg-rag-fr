"""
Chaîne RAG LangChain pour le LLM phytosanitaire.

Architecture :
1. Query Analyzer : détecte le type de question (produit / réglementation / biocontrôle / substance)
2. Retriever hybride : recherche vectorielle + filtres metadata Qdrant
3. Génération : Mistral 7b via Ollama avec prompt système spécialisé
"""

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_ollama import ChatOllama
from langchain_qdrant import QdrantVectorStore

from src.embeddings import MxbaiEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

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

# ─── Initialisation des composants ────────────────────────────────────────────

def build_vectorstore() -> QdrantVectorStore:
    client = QdrantClient(url=QDRANT_URL)
    embeddings = MxbaiEmbeddings(model=EMBEDDING_MODEL)
    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )


def build_llm() -> ChatOllama:
    return ChatOllama(
        model=LLM_MODEL,
        temperature=0.1,    # faible température pour réponses factuelles
        num_ctx=4096,        # fenêtre de contexte
    )


# ─── Query Analyzer ───────────────────────────────────────────────────────────

def analyze_query(question: str) -> dict:
    """
    Analyse la question pour déterminer les filtres metadata à appliquer.
    Retourne un dict avec les types de sources pertinents.
    """
    import re
    q = question.lower()

    # Détection d'un numéro AMM explicite (7 chiffres)
    amm_match = re.search(r"\b(\d{7})\b", question)
    amm_number = amm_match.group(1) if amm_match else None

    # Détection d'un nom de produit commercial en majuscules (ex: ALIETTE FLASH, ROUNDUP)
    product_name_match = re.search(r"\b([A-Z][A-Z0-9\s\-]{2,}[A-Z0-9])\b", question)
    product_name = product_name_match.group(1).strip() if product_name_match else None

    # Détection d'un nom de molécule chimique :
    # suffixes chimiques courants (-ate, -yl, -ine, -ol, -Al, -ium, -ose, -ide, -ène)
    has_molecule_name = bool(
        re.search(r"\b\w+(?:ate|yl|ine|ène|ene|anol|idol|nol|ium|ose|ide|Al)\b", question, re.IGNORECASE)
    )

    # Détection du type de question
    is_product_query = bool(amm_number) or bool(product_name) or any(w in q for w in [
        "produit", "amm", "autorisé", "utiliser", "dose", "dar",
        "récolte", "appliquer", "traitement", "pulvérisation",
        "homologué", "retrait", "retiré",
    ])
    is_regulation_query = any(w in q for w in [
        "arrêté", "loi", "réglementation", "délai de rentrée", "rentrée",
        "vent", "beaufort", "zone non traitée", "znt", "effluent",
        "rinçage", "stockage", "registre", "epi", "protection",
        "délai", "interdit", "obligation", "règle",
    ])
    is_biocontrol_query = any(w in q for w in [
        "biocontrôle", "biocontrole", "micro-organisme", "phéromone",
        "substance naturelle", "liste l.253", "bacillus", "trichoderma",
    ])
    # Fix 2 : détection élargie des questions sur les substances actives
    is_substance_query = has_molecule_name or any(w in q for w in [
        "substance active", "matière active", "approbation", "approuvé",
        "approuvée", "européen", "europe", "efsa", "substitution",
        "cas number", "faible risque", "inscrit", "inscription",
    ])

    return {
        "amm_number": amm_number,
        "product_name": product_name,
        "is_product_query": is_product_query,
        "is_regulation_query": is_regulation_query,
        "is_biocontrol_query": is_biocontrol_query,
        "is_substance_query": is_substance_query,
    }


# ─── Retriever hybride ────────────────────────────────────────────────────────

def retrieve_documents(vectorstore: QdrantVectorStore, question: str, k: int = 6) -> list[Document]:
    """
    Récupère les k documents les plus pertinents.

    Fix 1 : si un N° AMM est détecté, recherche EXCLUSIVE sur ce produit et retour immédiat.
    Fix 2 : détection élargie des substances actives.
    Fix 3 : k=5 pour arrete_2017 + recherche par mot-clé exact pour les articles de loi.
    """
    analysis = analyze_query(question)
    all_docs: list[Document] = []
    seen_ids: set[str] = set()

    def add_docs(docs):
        for doc in docs:
            key = doc.page_content[:100]
            if key not in seen_ids:
                seen_ids.add(key)
                all_docs.append(doc)

    # ── Fix 1 : N° AMM explicite → usages AUTORISÉS uniquement + décision PDF ──
    if analysis["amm_number"]:
        # Usages autorisés du XML (filtre sur etat_usage = "Autorisation")
        docs_autorises = vectorstore.similarity_search(
            question,
            k=k,
            filter={"must": [
                {"key": "metadata.numero_amm", "match": {"value": analysis["amm_number"]}},
                {"key": "metadata.etat_usage", "match": {"value": "Autorisation"}},
            ]},
        )
        add_docs(docs_autorises)

        # Si aucun usage autorisé trouvé, on accepte tous les états
        # (produit retiré → l'utilisateur doit quand même avoir une réponse)
        if not all_docs:
            docs_tous = vectorstore.similarity_search(
                question,
                k=k,
                filter={"must": [{"key": "metadata.numero_amm",
                                   "match": {"value": analysis["amm_number"]}}]},
            )
            add_docs(docs_tous)

        # Toujours compléter avec la décision AMM individuelle (PDF) si elle existe
        docs_decision = vectorstore.similarity_search(
            question,
            k=3,
            filter={"must": [
                {"key": "metadata.type", "match": {"value": "decision_amm"}},
                {"key": "metadata.numero_amm", "match": {"value": analysis["amm_number"]}},
            ]},
        )
        add_docs(docs_decision)
        return all_docs[:k]

    # ── Recherche produit par nom commercial ──────────────────────────────────
    if analysis["is_product_query"]:
        if analysis["product_name"]:
            # Filtre sur le nom exact du produit pour ne pas dériver vers
            # d'autres produits partageant une culture mentionnée dans la question
            docs = vectorstore.similarity_search(
                question,
                k=k,
                filter={"must": [{"key": "metadata.nom_produit",
                                   "match": {"value": analysis["product_name"]}}]},
            )
            add_docs(docs)
            # Compléter sans filtre si trop peu de résultats (ex: faute de frappe légère)
            if len(all_docs) < 2:
                docs_fallback = vectorstore.similarity_search(question, k=k)
                add_docs(docs_fallback)
        else:
            docs = vectorstore.similarity_search(question, k=k)
            add_docs(docs)

    # ── Fix 2 : réglementation → recherche hybride (2 formulations) ────────────
    if analysis["is_regulation_query"]:
        filtre_arrete = {"must": [{"key": "metadata.source",
                                    "match": {"value": "arrete_2017"}}]}

        # Recherche 1 : question originale
        docs1 = vectorstore.similarity_search(question, k=5, filter=filtre_arrete)
        add_docs(docs1)

        # Recherche 2 : reformulation orientée valeurs concrètes
        # "délai de rentrée par défaut" → on ajoute les termes numériques
        # que l'article contient pour rapprocher les vecteurs
        query_concrete = question + " valeur durée heures article obligation"
        docs2 = vectorstore.similarity_search(query_concrete, k=5, filter=filtre_arrete)
        add_docs(docs2)

        # Compléter avec les décisions AMM individuelles
        docs3 = vectorstore.similarity_search(
            question, k=3,
            filter={"must": [{"key": "metadata.type",
                               "match": {"value": "decision_amm"}}]},
        )
        add_docs(docs3)

    # ── Recherche biocontrôle ─────────────────────────────────────────────────
    if analysis["is_biocontrol_query"]:
        docs = vectorstore.similarity_search(
            question, k=4,
            filter={"must": [{"key": "metadata.type",
                               "match": {"value": "biocontrole"}}]},
        )
        add_docs(docs)

    # ── Fix 2 : substance active → filtre dédié ───────────────────────────────
    if analysis["is_substance_query"]:
        docs = vectorstore.similarity_search(
            question, k=5,
            filter={"must": [{"key": "metadata.type",
                               "match": {"value": "substance_active"}}]},
        )
        add_docs(docs)
        # Compléter avec les usages AMM de cette substance dans la base nationale
        docs_amm = vectorstore.similarity_search(
            question, k=3,
            filter={"must": [{"key": "metadata.source",
                               "match": {"value": "amm_xml"}}]},
        )
        add_docs(docs_amm)

    # ── Fallback : recherche générale si toujours insuffisant ────────────────
    # Ne s'applique que si au moins un type de requête a été détecté,
    # pour éviter d'injecter des documents non pertinents sur des questions
    # hors-domaine (salutations, questions sur l'azote, etc.)
    any_type_detected = (
        analysis["is_product_query"]
        or analysis["is_regulation_query"]
        or analysis["is_biocontrol_query"]
        or analysis["is_substance_query"]
    )
    if any_type_detected and len(all_docs) < k:
        docs = vectorstore.similarity_search(question, k=k - len(all_docs))
        add_docs(docs)

    return all_docs[:k]


def format_context(docs: list[Document]) -> str:
    """Formate les documents récupérés en contexte lisible pour le LLM."""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "inconnu")
        source_labels = {
            "amm_xml": "Base AMM nationale",
            "arrete_2017": "Arrêté du 4 mai 2017",
            "decision_amm_individuelle": "Décision AMM individuelle",
            "note_biocontrole": "Note DGAL biocontrôle",
            "substances_actives_xlsx": "Base substances actives CE",
        }
        label = source_labels.get(source, source)
        parts.append(f"--- Source {i} [{label}] ---\n{doc.page_content}")
    return "\n\n".join(parts)


# ─── Chaîne RAG principale ────────────────────────────────────────────────────

def build_rag_chain(vectorstore: QdrantVectorStore, llm: ChatOllama):
    """
    Construit la chaîne RAG avec LCEL (LangChain Expression Language).
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT),
    ])

    def retrieve_and_format(question: str) -> dict:
        docs = retrieve_documents(vectorstore, question)
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
        self.chain = build_rag_chain(self.vectorstore, self.llm)

    def ask(self, question: str) -> dict:
        """
        Pose une question et retourne la réponse + les sources utilisées.
        Returns:
            {
                "answer": str,
                "sources": list[dict],   # metadata de chaque doc source
            }
        """
        result = self.chain.invoke(question)
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
            for doc in result.get("source_docs", [])
        ]
        return {"answer": result["answer"], "sources": sources}

    def stream(self, question: str):
        """Génère la réponse en streaming."""
        docs = retrieve_documents(self.vectorstore, question)
        context = format_context(docs)

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ])
        chain = prompt | self.llm | StrOutputParser()

        yield from chain.stream({"context": context, "question": question})
