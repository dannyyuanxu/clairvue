"""ChromaDB wrapper. Vector metadata always includes embedding_model for switch detection."""

from src.config import settings


class VectorStore:
    def __init__(self, persist_dir: str = settings.CHROMA_PERSIST_DIR) -> None:
        self.persist_dir = persist_dir

    def add(self, ids: list[str], embeddings: list[list[float]], metadatas: list[dict], documents: list[str]) -> None:
        raise NotImplementedError

    def query(self, query_embedding: list[float], top_k: int = 5, where: dict | None = None) -> list[dict]:
        raise NotImplementedError
