"""Prompt templates for claim decomposition and evidence-based assessment."""

CLAIM_DECOMPOSITION_PROMPT = """Decompose the following management statement into a list of \
atomic, independently verifiable claims. Return JSON: {{"atomic_claims": ["...", "..."]}}

Statement: {statement}
"""

CLAIM_ASSESSMENT_PROMPT = """You are assessing whether a management statement is supported by \
formal evidence (SEC filings, XBRL metrics, peer disclosures).

Claim: {claim}

Supporting context:
{context}

Return a structured JSON assessment with fields: assessment, rationale, supporting_evidence, \
contradictory_evidence, relevant_metrics, missing_information, analyst_follow_up_questions.
"""
