"""Primary RAG workflows: claim validation and peer comparison.

Each function runs its own multi-call pipeline (decompose -> retrieve -> assess,
or retrieve -> compare) rather than a single LLM call over raw input -- the model
never sees evidence it wasn't specifically retrieved for.
"""

import pandas as pd

from src.llm_client import LLMClient
from src.metrics import load_metrics
from src.prompts import assess_claim_messages, decompose_claim_messages, format_metrics_for_prompt
from src.prompts import peer_comparison_messages
from src.retrieval import EvidenceRetriever, format_evidence_for_prompt

DEFAULT_TICKERS = ["JPM", "BAC", "C"]

CONFIDENCE_BY_ASSESSMENT = {
    "supported": 1.0,
    "partially_supported": 0.6,
    "contradicted": 0.3,
    "insufficient_evidence": 0.2,
}

LIMITATIONS = [
    "Assessment based on available SEC filings and curated metrics only. This is not investment advice."
]

_retriever: EvidenceRetriever | None = None
_metrics_df: pd.DataFrame | None = None


def _get_retriever() -> EvidenceRetriever:
    global _retriever
    if _retriever is None:
        _retriever = EvidenceRetriever()
    return _retriever


def _get_metrics_df() -> pd.DataFrame:
    global _metrics_df
    if _metrics_df is None:
        _metrics_df = load_metrics()
    return _metrics_df


def _overall_assessment(claim_assessments: list[dict]) -> str:
    verdicts = [assessment.get("assessment") for assessment in claim_assessments]

    if all(verdict == "supported" for verdict in verdicts):
        return "supported"
    if verdicts.count("contradicted") > len(verdicts) / 2:
        return "contradicted"
    return "partially_supported"


def answer_claim(
    statement: str,
    ticker: str | None = None,
    risk_theme: str = "consumer_credit",
    metrics_df: pd.DataFrame | None = None,
) -> dict:
    """Full claim validation pipeline: decompose into atomic claims, then assess
    each one independently against its own retrieved evidence and metrics."""
    retriever = _get_retriever()
    llm_client = LLMClient()
    if metrics_df is None:
        metrics_df = _get_metrics_df()

    decomposition = llm_client.chat_json(decompose_claim_messages(statement))
    atomic_claims = decomposition["atomic_claims"]

    print(f"Decomposed into {len(atomic_claims)} atomic claims:")
    for claim in atomic_claims:
        print(f"  - {claim}")

    claim_assessments = []
    for claim in atomic_claims:
        chunks = retriever.retrieve_with_contradiction(
            claim,
            tickers=[ticker] if ticker else None,
            risk_theme=risk_theme,
        )
        evidence_text = format_evidence_for_prompt(chunks)
        metrics_text = format_metrics_for_prompt(
            metrics_df,
            tickers=[ticker] if ticker else DEFAULT_TICKERS,
            metric_names=None,
        )

        assessment = llm_client.chat_json(assess_claim_messages(claim, evidence_text, metrics_text))
        claim_assessments.append(assessment)

    overall_assessment = _overall_assessment(claim_assessments)
    confidence = sum(
        CONFIDENCE_BY_ASSESSMENT.get(assessment.get("assessment"), 0.2) for assessment in claim_assessments
    ) / len(claim_assessments)

    return {
        "original_statement": statement,
        "atomic_claims": atomic_claims,
        "overall_assessment": overall_assessment,
        "claim_assessments": claim_assessments,
        "limitations": LIMITATIONS,
        "confidence": round(confidence, 2),
    }


def compare_peers(
    question: str,
    tickers: list[str] | None = None,
    risk_theme: str = "consumer_credit",
    metrics_df: pd.DataFrame | None = None,
) -> dict:
    """Full peer comparison pipeline: retrieve evidence per bank, then compare
    all banks against the question in a single grounded LLM call."""
    tickers = tickers or DEFAULT_TICKERS
    retriever = _get_retriever()
    llm_client = LLMClient()
    if metrics_df is None:
        metrics_df = _get_metrics_df()

    chunks_by_ticker = retriever.retrieve_per_bank(question, tickers, risk_theme=risk_theme)
    bank_evidence = {
        ticker: format_evidence_for_prompt(chunks) for ticker, chunks in chunks_by_ticker.items()
    }
    metrics_text = format_metrics_for_prompt(metrics_df, tickers=tickers, metric_names=None)

    return llm_client.chat_json(peer_comparison_messages(question, bank_evidence, metrics_text, tickers))


if __name__ == "__main__":
    import json

    result = answer_claim("Consumer credit remains resilient and losses are normalizing.")
    print(json.dumps(result, indent=2))
