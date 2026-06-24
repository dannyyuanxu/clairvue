"""Stage 2 of the cache pipeline: embed cached chunks and persist embeddings.npy."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.config import settings
from src.llm_client import LLMClient

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
EMBEDDINGS_PATH = Path("data/processed/embeddings.npy")
VECTOR_METADATA_PATH = Path("data/processed/vector_metadata.jsonl")

EMBED_BATCH_SIZE = 32
EMBED_BATCH_SLEEP_SECONDS = 0.5


def _load_chunks(path: Path = CHUNKS_PATH) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_embeddings(path: str = str(EMBEDDINGS_PATH)) -> tuple:
    """Load cached embeddings + metadata from disk. Returns (None, None) if missing."""
    embeddings_path = Path(path)
    metadata_path = embeddings_path.parent / "vector_metadata.jsonl"

    if not embeddings_path.exists() or not metadata_path.exists():
        return None, None

    embeddings = np.load(embeddings_path)
    with open(metadata_path) as f:
        metadata_list = [json.loads(line) for line in f if line.strip()]

    return embeddings, metadata_list


def embed_chunks(chunks: list[dict], llm_client: LLMClient) -> list[list[float]]:
    """Embed chunk texts in batches of 32, sleeping between batches."""
    texts = [chunk["text"] for chunk in chunks]
    embeddings: list[list[float]] = []

    num_batches = (len(texts) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE
    for batch_index in range(num_batches):
        start = batch_index * EMBED_BATCH_SIZE
        batch_texts = texts[start : start + EMBED_BATCH_SIZE]
        embeddings.extend(llm_client.embed(batch_texts))
        print(f"  batch {batch_index + 1} of {num_batches}")
        if batch_index < num_batches - 1:
            time.sleep(EMBED_BATCH_SLEEP_SECONDS)

    return embeddings


def validate_embedding_model(metadata_list: list[dict]) -> None:
    current_model = settings.EMBEDDING_MODEL
    mismatched_models = {
        metadata.get("embedding_model")
        for metadata in metadata_list
        if metadata.get("embedding_model") != current_model
    }
    for old_model in mismatched_models:
        print(
            f"WARNING: Cached embeddings were generated with {old_model} but current "
            f"EMBEDDING_MODEL is {current_model}. Run with --rebuild to regenerate."
        )


def get_embeddings(rebuild: bool = False) -> tuple:
    chunks = _load_chunks()

    if not rebuild:
        cached_embeddings, cached_metadata = load_embeddings()
        if cached_embeddings is not None and len(cached_embeddings) == len(chunks):
            print(f"Loaded {len(chunks)} cached embeddings from {EMBEDDINGS_PATH}, shape {cached_embeddings.shape}")
            validate_embedding_model(cached_metadata)
            return cached_embeddings, cached_metadata
        if cached_embeddings is not None:
            print(
                f"Cache has {len(cached_embeddings)} embeddings but {len(chunks)} chunks exist; rebuilding."
            )

    llm_client = LLMClient()
    start_time = time.time()
    embedding_vectors = embed_chunks(chunks, llm_client)
    elapsed = time.time() - start_time

    embeddings_array = np.array(embedding_vectors, dtype=np.float32)

    metadata_list = []
    for chunk in chunks:
        metadata = dict(chunk["metadata"])
        metadata["chunk_id"] = chunk["chunk_id"]
        metadata["text"] = chunk["text"]
        metadata["embedding_model"] = settings.EMBEDDING_MODEL
        metadata_list.append(metadata)

    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings_array)
    with open(VECTOR_METADATA_PATH, "w") as f:
        for metadata in metadata_list:
            f.write(json.dumps(metadata) + "\n")

    print(f"Embedded {len(chunks)} chunks in {elapsed:.1f}s, shape {embeddings_array.shape}")
    print(f"Saved to {EMBEDDINGS_PATH} and {VECTOR_METADATA_PATH}")

    return embeddings_array, metadata_list


def _main() -> None:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--rebuild", action="store_true")
    args = arg_parser.parse_args()
    get_embeddings(rebuild=args.rebuild)


if __name__ == "__main__":
    _main()
