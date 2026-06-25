"""Entry point used by notebook.ipynb to drive the Colab demo.

Also runnable standalone: generates outputs/sample_answers.json, a cached fallback
in case the live API is slow or unavailable during the interview.
"""

import json
import os
import sys
from pathlib import Path

from src.generation import answer_claim, compare_peers

OUTPUT_PATH = Path("outputs/sample_answers.json")

ASSESSMENT_LABELS = {
    "supported": ("SUPPORTED", "32"),  # green
    "partially_supported": ("PARTIALLY SUPPORTED", "33"),  # yellow
    "contradicted": ("CONTRADICTED", "31"),  # red
    "insufficient_evidence": ("INSUFFICIENT", "90"),  # gray
}

RISK_DIRECTION_LABELS = {
    "improving": ("IMPROVING", "32"),
    "stable": ("STABLE", "36"),
    "deteriorating": ("DETERIORATING", "31"),
    "unclear": ("UNCLEAR", "90"),
}

DEMO_1_STATEMENT = (
    "Consumer credit remains resilient, and losses are normalizing in line with expectations."
)
DEMO_2_QUESTION = (
    "Which of JPMorgan, Bank of America, and Citigroup shows the strongest evidence "
    "of increasing consumer-credit pressure in 2023?"
)
DEMO_3_STATEMENT = (
    "Does the evidence prove that consumer credit losses have peaked and will improve from here?"
)


def _color_supported() -> bool:
    return sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _colorize(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _color_supported() else text


def _bracket_label(value: str, label_map: dict) -> str:
    label, code = label_map.get(value, (str(value).upper(), "90"))
    return _colorize(f"[{label}]", code)


def _truncate(text: str, length: int = 80) -> str:
    text = " ".join(text.split())
    return text if len(text) <= length else text[: length - 3] + "..."


def print_claim_assessment(result: dict) -> None:
    """Pretty-print an answer_claim() result for the interview demo."""
    print("=" * 80)
    print(f"STATEMENT: {result['original_statement']}")
    print("=" * 80)

    for claim, claim_assessment in zip(result["atomic_claims"], result["claim_assessments"]):
        assessment = claim_assessment.get("assessment", "insufficient_evidence")
        print(f"\n{_bracket_label(assessment, ASSESSMENT_LABELS)} {claim}")

        evidence_items = (
            claim_assessment.get("supporting_evidence", [])
            + claim_assessment.get("contradictory_evidence", [])
            + claim_assessment.get("qualifying_evidence", [])
        )
        for evidence in evidence_items[:2]:
            excerpt = _truncate(evidence.get("excerpt", ""), 80)
            company = evidence.get("company", "")
            filing_type = evidence.get("filing_type", "")
            period = evidence.get("period", "")
            print(f'    "{excerpt}" -- {company}, {filing_type} {period}')

        follow_ups = claim_assessment.get("analyst_follow_up_questions", [])
        if follow_ups:
            print("    Follow-up questions:")
            for question in follow_ups:
                print(f"      - {question}")

    print()
    print(f"OVERALL ASSESSMENT: {_bracket_label(result['overall_assessment'], ASSESSMENT_LABELS)}")
    print(f"CONFIDENCE: {result['confidence']}")
    print()


def print_peer_comparison(result: dict) -> None:
    """Pretty-print a compare_peers() result for the interview demo."""
    print("=" * 80)
    print(f"QUESTION: {result['question']}")
    print("=" * 80)

    for bank in result.get("bank_assessments", []):
        risk_direction = bank.get("risk_direction", "unclear")
        print(f"\n{bank.get('ticker')} ({bank.get('company')}) {_bracket_label(risk_direction, RISK_DIRECTION_LABELS)}")
        print(f"  Summary: {bank.get('summary', '')}")
        excerpts = bank.get("evidence_excerpts", [])
        if excerpts:
            print(f'  Evidence: "{_truncate(excerpts[0], 80)}"')

    print()
    print(f"STRONGEST DETERIORATION SIGNAL: {result.get('strongest_deterioration_signal', '')}")
    print(f"\nOVERALL SUMMARY: {result.get('overall_summary', '')}")

    limitations = result.get("comparability_limitations", [])
    if limitations:
        print("\nCOMPARABILITY LIMITATIONS:")
        for limitation in limitations:
            print(f"  - {limitation}")
    print()


def run_demos() -> dict:
    """Runs the three interview demo scenarios and returns their raw results."""
    print("Running DEMO 1 (claim validation, broadly supported)...")
    demo_1 = answer_claim(DEMO_1_STATEMENT, risk_theme="consumer_credit")

    print("\nRunning DEMO 2 (peer comparison)...")
    demo_2 = compare_peers(DEMO_2_QUESTION, risk_theme="consumer_credit")

    print("\nRunning DEMO 3 (insufficient evidence / appropriate abstention)...")
    demo_3 = answer_claim(DEMO_3_STATEMENT, risk_theme="consumer_credit")

    return {"demo_1": demo_1, "demo_2": demo_2, "demo_3": demo_3}


def main() -> None:
    """`python -m src.demo` entry point: runs all 3 scenarios, saves them to
    outputs/sample_answers.json, and pretty-prints each one."""
    results = run_demos()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved sample outputs to {OUTPUT_PATH}\n")

    print("\n" + "#" * 80)
    print("# DEMO 1 -- Claim validation (broadly supported)")
    print("#" * 80)
    print_claim_assessment(results["demo_1"])

    print("\n" + "#" * 80)
    print("# DEMO 2 -- Peer comparison")
    print("#" * 80)
    print_peer_comparison(results["demo_2"])

    print("\n" + "#" * 80)
    print("# DEMO 3 -- Insufficient evidence / appropriate abstention")
    print("#" * 80)
    print_claim_assessment(results["demo_3"])


if __name__ == "__main__":
    main()
