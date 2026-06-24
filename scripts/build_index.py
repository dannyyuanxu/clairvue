"""CLI: parse -> chunk -> embed -> load ChromaDB, using the cached intermediates."""

from src.chunking import chunk_filing
from src.embeddings import embed_chunks
from src.llm_client import LLMClient
from src.vector_store import VectorStore


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
