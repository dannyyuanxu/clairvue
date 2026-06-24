"""Entry point used by notebook.ipynb to drive the Colab demo."""

from src.generation import answer_claim, compare_peers
from src.llm_client import LLMClient
from src.vector_store import VectorStore


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
