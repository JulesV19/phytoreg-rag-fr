"""
Embeddings adaptés pour mxbai-embed-large.

mxbai-embed-large est entraîné avec deux espaces distincts :
  - Documents (passages) : texte brut, sans préfixe
  - Requêtes (queries)   : préfixe obligatoire pour orienter le vecteur
                           vers le sous-espace "recherche de passages"

Sans ce préfixe sur les requêtes, les vecteurs de questions des agriculteurs
se retrouvent dans un espace sémantique différent de celui des chunks indexés,
ce qui dégrade fortement le rappel.

Référence : https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1
"""

from langchain_ollama import OllamaEmbeddings

# Préfixe officiel pour le mode "retrieval query" de mxbai-embed-large-v1
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class MxbaiEmbeddings(OllamaEmbeddings):
    """
    Wrapper autour d'OllamaEmbeddings qui ajoute le préfixe de requête
    recommandé par mixedbread.ai pour mxbai-embed-large.

    - embed_documents : texte brut (aucun préfixe) — pour l'ingestion
    - embed_query     : préfixe ajouté automatiquement — pour la recherche

    Usage :
        embeddings = MxbaiEmbeddings(model="mxbai-embed-large")
        # Utiliser la même instance pour ingest ET pour le retriever
    """

    def embed_query(self, text: str) -> list[float]:
        prefixed = f"{QUERY_PREFIX}{text}"
        return super().embed_query(prefixed)
