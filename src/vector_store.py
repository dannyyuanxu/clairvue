"""ChromaDB wrapper. Vector metadata always includes embedding_model for switch detection."""

import argparse
import json
from pathlib import Path

import chromadb
import numpy as np

from src.config import settings

COLLECTION_NAME = "clairvue_filings"
UPSERT_BATCH_SIZE = 500

EMBEDDINGS_PATH = Path("data/processed/embeddings.npy")
VECTOR_METADATA_PATH = Path("data/processed/vector_metadata.jsonl")


def _get_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)


def _sanitize_metadata(metadata: dict) -> dict:
    """ChromaDB metadata values must be str/int/float/bool."""
    return {key: ("" if value is None else value) for key, value in metadata.items()}


def build_index(embeddings: np.ndarray, metadata_list: list[dict]) -> None:
    client = _get_client()
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    n = len(metadata_list)
    for start in range(0, n, UPSERT_BATCH_SIZE):
        end = min(start + UPSERT_BATCH_SIZE, n)
        batch_metadata = metadata_list[start:end]
        collection.upsert(
            ids=[metadata["chunk_id"] for metadata in batch_metadata],
            embeddings=embeddings[start:end].tolist(),
            documents=[metadata["text"] for metadata in batch_metadata],
            metadatas=[_sanitize_metadata(metadata) for metadata in batch_metadata],
        )

    print(f"Indexed {n} chunks into ChromaDB at {settings.CHROMA_PERSIST_DIR}")


def load_collection() -> chromadb.Collection:
    client = _get_client()
    existing_names = {collection.name for collection in client.list_collections()}
    if COLLECTION_NAME not in existing_names:
        raise RuntimeError(
            f"Collection '{COLLECTION_NAME}' does not exist at {settings.CHROMA_PERSIST_DIR}. "
            "Run `python -m src.vector_store --build` first."
        )
    return client.get_collection(COLLECTION_NAME)


def query(
    collection: chromadb.Collection,
    query_embedding: list[float],
    n_results: int = 8,
    where: dict | None = None,
) -> list[dict]:
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    hits = [
        {"chunk_id": chunk_id, "text": text, "score": 1 - distance, "metadata": metadata}
        for chunk_id, text, metadata, distance in zip(
            results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]
    hits.sort(key=lambda hit: hit["score"], reverse=True)
    return hits


def filter_by_ticker(tickers: list[str]) -> dict:
    if len(tickers) == 1:
        return {"ticker": tickers[0]}
    return {"ticker": {"$in": tickers}}


def filter_by_theme(theme: str) -> dict:
    return {"risk_theme": theme}


def combine_filters(*filters) -> dict:
    return {"$and": list(filters)}


def _main() -> None:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--build", action="store_true")
    arg_parser.add_argument("--count", action="store_true")
    args = arg_parser.parse_args()

    if args.build:
        embeddings = np.load(EMBEDDINGS_PATH)
        with open(VECTOR_METADATA_PATH) as f:
            metadata_list = [json.loads(line) for line in f if line.strip()]
        build_index(embeddings, metadata_list)

    if args.count:
        collection = load_collection()
        print(f"Collection '{COLLECTION_NAME}' has {collection.count()} chunks.")

    if not args.build and not args.count:
        print("Nothing to do — pass --build and/or --count.")


if __name__ == "__main__":
    _main()
