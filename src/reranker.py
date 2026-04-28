from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

MODEL_NAME = "BAAI/bge-reranker-v2-m3"


class Reranker:
    def __init__(self, model: str = MODEL_NAME):
        self._model = CrossEncoder(model, max_length=512)

    def rerank(self, question: str, docs: list[Document], top_k: int = 6) -> list[Document]:
        if not docs:
            return docs
        pairs = [(question, doc.page_content) for doc in docs]
        scores = self._model.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[:top_k]]
