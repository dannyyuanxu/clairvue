"""Centralized prompt templates. No LLM calls happen here -- these functions only
build message lists / formatted text for src/generation.py to pass to LLMClient.
"""

import pandas as pd

DECOMPOSE_CLAIM_SYSTEM_PROMPT = (
    "You are a financial analyst assistant. Your task is to decompose a management "
    "statement into atomic, independently verifiable claims. Each claim should be a "
    "single assertion that can be evaluated as true, false, or uncertain based on data. "
    "Return JSON with a single key 'atomic_claims' containing a list of strings."
)

# Rules 1-6 + the follow-up rule are verbatim from CLAUDE.md/PRD Section 12
# ("System prompt for claim assessment"). The quote-extraction rule is new, and the
# follow-up rule is now gated on verbose -- see _assess_system_prompt().
ASSESS_CLAIM_INTRO = (
    "You are a financial research evidence assistant for institutional analysts.\n"
    "Your role is to assess whether a management statement is supported by formal evidence."
)

ASSESS_QUOTE_RULE = (
    "For each piece of evidence you cite, you MUST extract and return the single most "
    "relevant complete sentence or passage (up to 3 sentences maximum) from that chunk "
    'verbatim -- do not truncate, do not paraphrase. Return this as the "relevant_quote" '
    "field. Choose the sentence that most directly supports or contradicts the claim being "
    "assessed. If no single sentence is decisive, return the two most relevant consecutive "
    "sentences. If the most relevant evidence is a table row, do NOT return bare pipe-"
    "delimited cells (e.g. '| 0.88 | 1.05 |'): rewrite it as a self-contained phrase that "
    "names the row label, the period, and the unit -- e.g. '30+ day delinquency rate "
    "(card), 2023-Q1: 0.88%'. Never cite a number whose meaning isn't clear from the quote alone."
)

# "Normalization" is meaningless without a reference level; spelled out here so the model
# checks the claim against the pre-pandemic baseline supplied in the metrics block rather
# than reading a YoY increase as deterioration.
METRIC_INTERPRETATION_RULE = (
    "When you cite a quantitative metric you MUST interpret it, not just quote it. The metrics "
    "block already gives you each metric's level, its direction vs the prior period, its "
    "standing vs the pre-pandemic (2019) baseline, and the peer cross-section -- use them. "
    "State whether the figure is high or low and rising or falling, RELATIVE TO (a) the prior "
    "period, (b) the 2019 baseline, and (c) peer banks, and say what that implies for the claim. "
    "A bare number with no comparison is not acceptable evidence."
)

NORMALIZATION_RULE = (
    "Interpret 'normalizing' / 'losses returning to normal' as credit losses moving back toward "
    "their pre-pandemic (2019) baseline after the 2020-2021 stimulus-suppressed trough -- so a "
    "loss rate that is RISING from the trough but still at or below the 2019 baseline is "
    "consistent with normalization, NOT deterioration. If no pre-2020 baseline is available for "
    "the bank, say the baseline is missing rather than guessing. Treat 'in line with "
    "expectations' as a separate claim that requires the bank's own prior guidance to verify; if "
    "that guidance isn't in the evidence, mark that specific point insufficient."
)


def _assess_system_prompt(verbose: bool) -> str:
    """Builds the claim-assessment system prompt. The follow-up-questions rule is
    only included when verbose=True, mirroring the schema (see assess_claim_messages)."""
    rules = [
        "Use ONLY the supplied evidence chunks and metrics. Do not use prior knowledge.",
        "Do not make buy, sell, or price recommendations.",
        "Explicitly distinguish formal SEC disclosures (authority: formal_disclosure) from "
        "management commentary (authority: management_commentary).",
        "Classify each claim as: supported, partially_supported, contradicted, or insufficient_evidence.",
        "If evidence is missing or insufficient, say so explicitly -- do not fill gaps with inference.",
        "Cite the source of every material conclusion (company, filing type, period, section).",
        ASSESS_QUOTE_RULE,
        METRIC_INTERPRETATION_RULE,
        NORMALIZATION_RULE,
    ]
    if verbose:
        rules.append("Include analyst follow-up questions for any unresolved gaps.")
    rules.append("Return valid JSON matching the required schema exactly.")
    numbered = "\n".join(f"{i}. {rule}" for i, rule in enumerate(rules, start=1))
    return f"{ASSESS_CLAIM_INTRO}\n\nRules:\n{numbered}"

# Same grounding rules as claim assessment, adapted for comparing multiple banks at once.
PEER_COMPARISON_SYSTEM_PROMPT = """You are a financial analyst assistant comparing consumer-credit \
risk evidence across peer banks for institutional analysts.

Rules:
1. Use ONLY the supplied evidence and metrics for each bank. Do not use prior knowledge.
2. Do not make buy, sell, or price recommendations.
3. If evidence for a bank is missing or insufficient, say so explicitly -- do not fill gaps with inference.
4. Cite the source of every material conclusion (company, filing type, period, section).
5. Explicitly flag comparability limitations between banks (e.g. differing XBRL concepts,
   differing fiscal periods, differing authority of evidence).
6. Include analyst follow-up questions for any unresolved gaps.
7. Return valid JSON matching the required schema exactly."""


def decompose_claim_messages(statement: str) -> list[dict]:
    """Messages for decomposing a management statement into atomic claims."""
    user_prompt = (
        f'Management statement:\n"{statement}"\n\n'
        "Decompose this statement into 2-5 atomic claims. Each claim must be a single "
        "assertion that can be independently verified against formal evidence (not a "
        "summary or a compound sentence covering multiple ideas).\n\n"
        'Return JSON: {"atomic_claims": ["...", "..."]}'
    )
    return [
        {"role": "system", "content": DECOMPOSE_CLAIM_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def assess_claim_messages(
    claim: str,
    primary_evidence_text: str,
    metrics_text: str,
    peer_evidence_text: str = "",
    verbose: bool = False,
) -> list[dict]:
    """Messages for assessing a single atomic claim against its retrieved evidence.

    Primary evidence (the bank's own filings) drives the verdict; peer evidence, when
    present, is supplied separately as comparison-only context. Each cited evidence item
    must include a verbatim relevant_quote. analyst_follow_up_questions is only requested
    (in both the system prompt and the schema) when verbose=True."""
    evidence_item = (
        '{"company": str, "source_type": str, "filing_type": str, "period": str, '
        '"section": str, "excerpt": str, "relevant_quote": str}'
    )

    peer_section = ""
    peer_instruction = ""
    if peer_evidence_text.strip():
        peer_section = (
            "\nPEER BANK CONTEXT (for comparison -- do not base verdict on this alone):\n"
            f"{peer_evidence_text}\n"
        )
        peer_instruction = (
            " After stating your verdict based on the primary evidence, note in one sentence "
            "whether the peer banks' evidence aligns with or diverges from the primary bank's "
            "picture, and include that sentence in the rationale field."
        )

    # Leading comma keeps the schema template valid JSON whether or not the line is present.
    follow_up_line = ',\n  "analyst_follow_up_questions": [str]' if verbose else ""

    schema = f"""{{
  "claim": str,
  "assessment": "supported|partially_supported|contradicted|insufficient_evidence",
  "rationale": str,
  "supporting_evidence": [{evidence_item}],
  "qualifying_evidence": [{evidence_item}],
  "contradictory_evidence": [{evidence_item}],
  "peer_context": [{evidence_item}],
  "relevant_metrics": [{{"ticker": str, "metric_name": str, "period": str, "value": str, "change": str}}],
  "missing_information": [str]{follow_up_line}
}}"""

    user_prompt = f"""Claim to assess: "{claim}"

PRIMARY EVIDENCE (the bank's own filings -- base your verdict on this):
{primary_evidence_text}
{peer_section}
Relevant XBRL metrics:
{metrics_text}

Assess this claim using only the evidence and metrics above.{peer_instruction} \
Return JSON matching this schema exactly:
{schema}"""

    return [
        {"role": "system", "content": _assess_system_prompt(verbose)},
        {"role": "user", "content": user_prompt},
    ]


def peer_comparison_messages(
    question: str,
    bank_evidence: dict,
    metrics_text: str,
    tickers: list[str],
) -> list[dict]:
    """Messages for comparing peer banks' evidence against a single question."""
    evidence_sections = "\n\n".join(
        f"=== {ticker} ===\n{bank_evidence.get(ticker, 'No evidence retrieved.')}"
        for ticker in tickers
    )

    user_prompt = f"""Analyst question: {question}

Banks being compared: {", ".join(tickers)}

Evidence by bank:
{evidence_sections}

Relevant XBRL metrics (all banks):
{metrics_text}

Compare these banks on the question above using only the evidence and metrics supplied. \
Return JSON matching this schema exactly:
{{
  "risk_theme": str,
  "question": str,
  "bank_assessments": [
    {{
      "company": str, "ticker": str, "summary": str,
      "evidence_excerpts": [str], "metrics_summary": str,
      "risk_direction": "improving|stable|deteriorating|unclear"
    }}
  ],
  "strongest_deterioration_signal": str,
  "comparability_limitations": [str],
  "overall_summary": str,
  "analyst_follow_up_questions": [str]
}}"""
    return [
        {"role": "system", "content": PEER_COMPARISON_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _period_label(period_end: str) -> str:
    year, month, _ = period_end.split("-")
    quarter = (int(month) - 1) // 3 + 1
    return f"{year}-Q{quarter}"


# Metric framing is aligned to the ingested filing window so the numbers line up with the
# qualitative evidence; the pre-2020 rows backfilled into metrics.csv are used only as a
# baseline, never shown as "current".
CORPUS_WINDOW_END = "2023-12-31"
BASELINE_CUTOFF = "2020-01-01"  # rows before this are the pre-pandemic baseline
RATE_METRICS = {"net_charge_off_rate"}
# Dollar metrics shown as level + YoY. net_charge_offs is deliberately omitted from the
# dollar block: it is YTD-cumulative (so a bare dollar figure misleads) and is fully
# represented, on a comparable annualized basis, by net_charge_off_rate.
DOLLAR_METRICS_TO_SHOW = ["provision_for_credit_losses", "allowance_for_credit_losses", "total_loans"]


def _direction_word(delta: float, flat_threshold: float) -> str:
    if pd.isna(delta):
        return ""
    if delta > flat_threshold:
        return "rising"
    if delta < -flat_threshold:
        return "falling"
    return "flat"


def _format_rate_block(rate_rows: pd.DataFrame) -> list[str]:
    """Pre-framed net-charge-off-rate lines: current level, QoQ move (in pp), level vs the
    pre-pandemic baseline and cycle trough, and the recent trail -- so the model interprets
    rather than computes."""
    rate_rows = rate_rows.sort_values("period_end")
    in_window = rate_rows[rate_rows["period_end"] <= CORPUS_WINDOW_END]
    if in_window.empty:
        return []
    latest = in_window.iloc[-1]
    latest_period = _period_label(latest["period_end"])
    value = latest["value"]

    qoq = latest.get("qoq_change_abs")
    qoq_str = f"QoQ {qoq:+.2f}pp ({_direction_word(qoq, 0.02)})" if pd.notna(qoq) else "QoQ n/a"

    baseline_rows = rate_rows[rate_rows["period_end"] < BASELINE_CUTOFF]
    if baseline_rows.empty:
        baseline_str = "no pre-2020 baseline available for this bank (cannot anchor 'normalization')"
    else:
        baseline = baseline_rows["value"].mean()
        diff = value - baseline
        if diff < -0.05:
            rel = f"below by {abs(diff):.2f}pp (not yet back to pre-pandemic normal)"
        elif diff > 0.05:
            rel = f"above by {diff:.2f}pp"
        else:
            rel = "~in line"
        trough = in_window["value"].min()
        baseline_str = f"vs 2019 baseline {baseline:.2f}%: {rel}; cycle trough {trough:.2f}%"

    trail = in_window.tail(4)
    trail_str = " -> ".join(f"{_period_label(r['period_end'])} {r['value']:.2f}%" for _, r in trail.iterrows())

    return [
        "  net_charge_off_rate (annualized loss rate -- KEY interpretable metric):",
        f"    latest {latest_period}: {value:.2f}% | {qoq_str} | {baseline_str}",
        f"    recent trail: {trail_str}",
    ]


def _peer_rate_cross_section(df: pd.DataFrame, tickers: list[str]) -> list[str]:
    """One line ranking each bank's latest in-window NCO rate, so 'is this high or low?' is
    answerable by direct peer comparison."""
    latest_by_ticker = {}
    for ticker in tickers:
        rows = df[
            (df["ticker"] == ticker)
            & (df["metric_name"] == "net_charge_off_rate")
            & (df["period_end"] <= CORPUS_WINDOW_END)
        ].sort_values("period_end")
        if not rows.empty:
            latest_by_ticker[ticker] = rows.iloc[-1]["value"]
    if len(latest_by_ticker) < 2:
        return []
    ranked = sorted(latest_by_ticker.items(), key=lambda kv: kv[1], reverse=True)
    chain = "  >  ".join(f"{tk} {val:.2f}%" for tk, val in ranked)
    return ["", "Peer cross-section -- net_charge_off_rate (annualized, latest quarter, highest first):", f"  {chain}"]


def format_metrics_for_prompt(
    metrics_df: pd.DataFrame,
    tickers: list[str],
    metric_names: list[str] | None = None,
) -> str:
    """Formats the metrics slice as PRE-INTERPRETED text: rates with direction, baseline and
    peer framing rather than raw dollar dumps. The arithmetic (rates, deltas, baselines) is
    done here in Python so the model's job is narrative interpretation, not computation."""
    df = metrics_df[metrics_df["ticker"].isin(tickers) & metrics_df["period_end"].notna()]
    df = df[df["source"] != "unavailable_xbrl"]
    if metric_names:
        df = df[df["metric_name"].isin(metric_names)]
    if df.empty:
        return "No metrics available."

    lines = []
    for ticker in tickers:
        ticker_rows = df[df["ticker"] == ticker]
        if ticker_rows.empty:
            continue
        lines.append(f"{ticker}:")

        rate_rows = ticker_rows[ticker_rows["metric_name"] == "net_charge_off_rate"]
        if not rate_rows.empty:
            lines.extend(_format_rate_block(rate_rows))

        for metric_name in DOLLAR_METRICS_TO_SHOW:
            metric_rows = ticker_rows[ticker_rows["metric_name"] == metric_name]
            metric_rows = metric_rows[metric_rows["period_end"] <= CORPUS_WINDOW_END].sort_values("period_end")
            if metric_rows.empty:
                continue
            row = metric_rows.iloc[-1]
            period = _period_label(row["period_end"])
            yoy = row.get("yoy_change_pct")
            yoy_str = f" (YoY {yoy:+.1f}%)" if pd.notna(yoy) else ""
            lines.append(f"  {metric_name}: {period} ${row['value_billions']:.2f}B{yoy_str}")

    lines.extend(_peer_rate_cross_section(df, tickers))
    return "\n".join(lines)
