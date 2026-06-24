from src.llm_client import LLMClient
from src.vector_store import VectorStore


def retrieve(query: str, llm_client: LLMClient, vector_store: VectorStore, top_k: int = 5, where: dict | None = None) -> list[dict]:
    raise NotImplementedError
