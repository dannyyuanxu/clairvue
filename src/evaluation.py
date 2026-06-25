"""Runs eval_questions.csv through the live pipeline and saves results for manual scoring."""

import argparse
import csv
import json
import time
from pathlib import Path

from src.generation import answer_claim, compare_peers
from src.retrieval import EvidenceRetriever

EVAL_QUESTIONS_CSV_PATH = "data/processed/eval_questions.csv"
EVAL_RESULTS_JSON_PATH = "outputs/eval_results.json"


def load_eval_questions(path: str = EVAL_QUESTIONS_CSV_PATH) -> list[dict]:
    """Loads the hand-curated eval question set."""
    with open(path) as f:
        return list(csv.DictReader(f))


def run_evaluation(questions_path: str = EVAL_QUESTIONS_CSV_PATH, output_path: str = EVAL_RESULTS_JSON_PATH) -> None:
    """Main entry point: dispatches each question to answer_claim/compare_peers/
    retriever.retrieve by its `mode` column, and saves all results as JSON for
    manual scoring (see outputs/eval_scoring_template.csv)."""
    questions = load_eval_questions(questions_path)
    retriever = EvidenceRetriever()

    results = []
    start_time = time.time()

    for row in questions:
        mode = row["mode"]
        question = row["question"]
        print(f"[{row['id']}] ({mode}) {question}")

        if mode == "claim_validation":
            result = answer_claim(question)
        elif mode == "peer_comparison":
            result = compare_peers(question)
        elif mode == "direct_retrieval":
            result = retriever.retrieve(question, n_results=5)
        else:
            raise ValueError(f"Unknown mode '{mode}' for question {row['id']}")

        results.append(
            {
                "id": row["id"],
                "category": row["category"],
                "mode": mode,
                "question": question,
                "result": result,
            }
        )

    elapsed = time.time() - start_time

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nRan {len(questions)} questions in {elapsed:.1f}s. Saved results to {output_path}")


def _main() -> None:
    arg_parser = argparse.ArgumentParser(description="Run the eval question set through the live pipeline.")
    arg_parser.add_argument("--questions-path", default=EVAL_QUESTIONS_CSV_PATH)
    arg_parser.add_argument("--output-path", default=EVAL_RESULTS_JSON_PATH)
    args = arg_parser.parse_args()
    run_evaluation(questions_path=args.questions_path, output_path=args.output_path)


if __name__ == "__main__":
    _main()
