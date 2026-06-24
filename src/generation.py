from src.llm_client import LLMClient
from src.vector_store import VectorStore


def answer_claim(statement: str, llm_client: LLMClient, vector_store: VectorStore) -> dict:
    raise NotImplementedError


def compare_peers(question: str, llm_client: LLMClient, vector_store: VectorStore) -> dict:
    raise NotImplementedError
