"""CLI: runs the full data pipeline end-to-end, in order.

    download_filings -> text_processing -> chunking -> embeddings -> vector_store

Each stage is independently cached (see CLAUDE.md "Data caching architecture"), so re-running
this script after the first time is cheap -- only stages whose inputs changed actually do work.

Usage:
    python scripts/build_index.py                 # run all 5 steps
    python scripts/build_index.py --from-step 4    # skip steps 1-3, resume from step 4
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.download_filings import main as download_filings_main
from src.chunking import chunk_all_filings
from src.embeddings import get_embeddings, load_embeddings
from src.text_processing import parse_all_filings
from src.vector_store import build_index

STEP_NAMES = {
    1: "Download filings",
    2: "Extract text",
    3: "Chunk documents",
    4: "Generate embeddings",
    5: "Build ChromaDB index",
}


def _timed_step(step_num: int, fn):
    name = STEP_NAMES[step_num]
    print(f"\n=== Step {step_num}: {name} ===")
    start = time.time()
    result = fn()
    print(f"Step {step_num} ({name}) completed in {time.time() - start:.1f}s")
    return result


def run_pipeline(from_step: int = 1) -> None:
    """Main entry point: runs steps 1-5 in order, skipping steps before from_step.
    Steps 4-5 are tightly coupled (5 needs 4's output), so skipping step 4 loads
    its output from disk instead of recomputing it -- see the from_step<=4 branch."""
    if from_step not in STEP_NAMES:
        raise ValueError(f"--from-step must be between 1 and {len(STEP_NAMES)}, got {from_step}")

    overall_start = time.time()

    if from_step <= 1:
        _timed_step(1, download_filings_main)
    else:
        print(f"Skipping Step 1 ({STEP_NAMES[1]}) -- starting from step {from_step}")

    if from_step <= 2:
        _timed_step(2, parse_all_filings)
    else:
        print(f"Skipping Step 2 ({STEP_NAMES[2]}) -- starting from step {from_step}")

    if from_step <= 3:
        _timed_step(3, chunk_all_filings)
    else:
        print(f"Skipping Step 3 ({STEP_NAMES[3]}) -- starting from step {from_step}")

    if from_step <= 4:
        embeddings_array, metadata_list = _timed_step(4, lambda: get_embeddings(rebuild=False))
    else:
        print(f"Skipping Step 4 ({STEP_NAMES[4]}) -- loading cached embeddings from disk")
        embeddings_array, metadata_list = load_embeddings()
        if embeddings_array is None:
            raise RuntimeError(
                "No cached embeddings found at data/processed/embeddings.npy -- "
                "cannot skip to step 5 without them. Run from an earlier step."
            )

    _timed_step(5, lambda: build_index(embeddings_array, metadata_list))

    print(f"\nTotal pipeline time: {time.time() - overall_start:.1f}s")


def main() -> None:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--from-step",
        type=int,
        default=1,
        help="Resume from this step (1-5), skipping earlier steps. Useful after an error.",
    )
    args = arg_parser.parse_args()
    run_pipeline(from_step=args.from_step)


if __name__ == "__main__":
    main()
