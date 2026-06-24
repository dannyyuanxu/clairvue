"""Stage 2 of the cache pipeline: embed cached chunks and persist embeddings.npy."""

from src.llm_client import LLMClient


def embed_chunks(chunks: list[dict], llm_client: LLMClient) -> list[dict]:
    raise NotImplementedError


def load_embeddings(path: str) -> tuple:
    raise NotImplementedError
