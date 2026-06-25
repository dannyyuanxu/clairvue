"""Entry point used by notebook.ipynb to drive the Colab demo.

Also runnable standalone: generates outputs/sample_answers.json, a cached fallback
in case the live API is slow or unavailable during the interview.
"""

import argparse
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

# Emoji + human label for the Markdown renderer (used in the Colab notebook, where Markdown
# renders far more readably than the ANSI terminal output).
ASSESSMENT_MD = {
    "supported": ("🟢", "Supported"),
    "partially_supported": ("🟡", "Partially supported"),
    "contradicted": ("🔴", "Contradicted"),
    "insufficient_evidence": ("⚪", "Insufficient evidence"),
}
RISK_DIRECTION_MD = {
    "improving": ("🟢", "Improving"),
    "stable": ("🔵", "Stable"),
    "deteriorating": ("🔴", "Deteriorating"),
    "unclear": ("⚪", "Unclear"),
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


def _print_evidence_item(item: dict) -> None:
    """Prints one evidence item using the LLM-extracted verbatim relevant_quote
    (no truncation), falling back to the legacy excerpt field only if a quote
    wasn't returned."""
    quote = item.get("relevant_quote") or item.get("excerpt", "")
    company = item.get("company", "")
    filing_type = item.get("filing_type", "?")
    period = item.get("period", "?")
    print(f'    "{quote}"')
    print(f"    Source: {company} ({filing_type} {period})")


def print_claim_assessment(result: dict, verbose: bool = False) -> None:
    """Pretty-print an answer_claim() result for the interview demo.

    Each cited evidence item is shown via its verbatim relevant_quote (no truncation).
    For bank-specific runs (peer_context present), primary evidence (the bank's own
    filings) and peer-bank context are shown under separate headers. Follow-up questions
    are shown only when verbose=True -- and are only present in the result at all when
    answer_claim was itself run with verbose=True."""
    print("=" * 80)
    print(f"STATEMENT: {result['original_statement']}")
    print("=" * 80)

    for claim, claim_assessment in zip(result["atomic_claims"], result["claim_assessments"]):
        assessment = claim_assessment.get("assessment", "insufficient_evidence")
        print(f"\n{_bracket_label(assessment, ASSESSMENT_LABELS)} {claim}")

        # "Primary" = all of the primary bank's own evidence buckets; keeping
        # contradictory/qualifying here matters so a contradicted verdict still
        # shows the evidence that contradicts it.
        primary_items = (
            claim_assessment.get("supporting_evidence", [])
            + claim_assessment.get("contradictory_evidence", [])
            + claim_assessment.get("qualifying_evidence", [])
        )
        peer_items = claim_assessment.get("peer_context", [])

        if peer_items:
            if primary_items:
                print("  === PRIMARY EVIDENCE (bank's own filings) ===")
                for item in primary_items:
                    _print_evidence_item(item)
            print("  === PEER BANK CONTEXT ===")
            for item in peer_items:
                _print_evidence_item(item)
        else:
            for item in primary_items:
                _print_evidence_item(item)

        if verbose and claim_assessment.get("analyst_follow_up_questions"):
            print("  Follow-up questions:")
            for question in claim_assessment["analyst_follow_up_questions"]:
                print(f"    - {question}")

    print()
    print(f"OVERALL ASSESSMENT: {_bracket_label(result['overall_assessment'], ASSESSMENT_LABELS)}")
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


def _md_evidence_bullet(item: dict) -> str:
    """One Markdown bullet for a cited evidence item: verbatim quote + source citation."""
    quote = (item.get("relevant_quote") or item.get("excerpt", "")).strip()
    company = item.get("company", "")
    filing_type = item.get("filing_type", "?")
    period = item.get("period", "?")
    return f'- "{quote}" — *{company}, {filing_type} {period}*'


def _md_metrics_table(metrics: list[dict]) -> list[str]:
    """Markdown table of the metrics the model cited for a claim (empty if none)."""
    if not metrics:
        return []
    rows = ["", "| Ticker | Metric | Period | Value | Change |", "|---|---|---|---|---|"]
    for metric in metrics:
        rows.append(
            f"| {metric.get('ticker', '')} | {metric.get('metric_name', '')} "
            f"| {metric.get('period', '')} | {metric.get('value', '')} | {metric.get('change', '')} |"
        )
    return rows


def claim_assessment_markdown(result: dict, verbose: bool = False) -> str:
    """Render an answer_claim() result as Markdown -- the notebook-friendly view.

    Unlike the ANSI printer, this leads each claim with its plain-English rationale (the
    headline), then a compact metrics table, then evidence grouped as Supporting / Watch /
    Contradicting / Peer, plus what's missing. This is where the numeric interpretation the
    model produced actually becomes visible."""
    overall_emoji, overall_label = ASSESSMENT_MD.get(result["overall_assessment"], ("⚪", result["overall_assessment"]))
    lines = [
        "## Statement assessment",
        f"> {result['original_statement']}",
        "",
        f"**Overall: {overall_emoji} {overall_label}**",
    ]

    for claim_assessment in result["claim_assessments"]:
        assessment = claim_assessment.get("assessment", "insufficient_evidence")
        emoji, label = ASSESSMENT_MD.get(assessment, ("⚪", assessment))
        claim = claim_assessment.get("claim", "")
        lines += ["", "---", f"### {emoji} {claim}", f"**{label}**"]

        rationale = claim_assessment.get("rationale", "").strip()
        if rationale:
            lines += ["", rationale]

        lines += _md_metrics_table(claim_assessment.get("relevant_metrics", []))

        groups = [
            ("Supporting", claim_assessment.get("supporting_evidence", [])),
            ("Qualifying / watch", claim_assessment.get("qualifying_evidence", [])),
            ("Contradicting", claim_assessment.get("contradictory_evidence", [])),
            ("Peer context", claim_assessment.get("peer_context", [])),
        ]
        for title, items in groups:
            if items:
                lines += ["", f"**{title}**", *[_md_evidence_bullet(item) for item in items]]

        missing = claim_assessment.get("missing_information", [])
        if missing:
            lines += ["", "**Missing**", *[f"- {item}" for item in missing]]

        if verbose and claim_assessment.get("analyst_follow_up_questions"):
            lines += ["", "**Follow-up questions**", *[f"- {q}" for q in claim_assessment["analyst_follow_up_questions"]]]

    return "\n".join(lines)


def display_claim_assessment(result: dict, verbose: bool = False) -> None:
    """Display an answer_claim() result as rendered Markdown in a notebook, falling back to
    the ANSI printer when not in an IPython/Colab environment."""
    try:
        from IPython.display import Markdown, display
    except ImportError:
        print_claim_assessment(result, verbose=verbose)
        return
    display(Markdown(claim_assessment_markdown(result, verbose=verbose)))


def peer_comparison_markdown(result: dict) -> str:
    """Render a compare_peers() result as Markdown -- the notebook-friendly view."""
    lines = ["## Peer comparison", f"> {result.get('question', '')}", ""]

    for bank in result.get("bank_assessments", []):
        direction = bank.get("risk_direction", "unclear")
        emoji, label = RISK_DIRECTION_MD.get(direction, ("⚪", direction))
        lines += [
            "---",
            f"### {bank.get('ticker', '')} — {bank.get('company', '')} {emoji} {label}",
            bank.get("summary", ""),
        ]
        if bank.get("metrics_summary"):
            lines += ["", f"*Metrics:* {bank['metrics_summary']}"]
        excerpts = bank.get("evidence_excerpts", [])
        if excerpts:
            lines += ["", "**Evidence**", *[f"- {excerpt}" for excerpt in excerpts]]
        lines.append("")

    if result.get("strongest_deterioration_signal"):
        lines += ["---", f"**Strongest deterioration signal:** {result['strongest_deterioration_signal']}", ""]
    if result.get("overall_summary"):
        lines += [f"**Overall:** {result['overall_summary']}", ""]
    limitations = result.get("comparability_limitations", [])
    if limitations:
        lines += ["**Comparability limitations**", *[f"- {limitation}" for limitation in limitations]]
    return "\n".join(lines)


def display_peer_comparison(result: dict) -> None:
    """Display a compare_peers() result as rendered Markdown in a notebook, falling back to
    the ANSI printer when not in an IPython/Colab environment."""
    try:
        from IPython.display import Markdown, display
    except ImportError:
        print_peer_comparison(result)
        return
    display(Markdown(peer_comparison_markdown(result)))


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
    # No options today, but parse anyway so a stray/typo'd flag errors loudly
    # instead of being silently ignored.
    argparse.ArgumentParser(description=main.__doc__).parse_args()

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
