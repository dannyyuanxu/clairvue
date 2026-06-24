"""XBRL-derived quantitative metrics (provisions, net charge-offs) for claim validation.

Note on concept selection: the "primary"/"alternate" concepts below are tried first
(matching the original spec), but verification against the live SEC companyfacts API
showed both are stale for JPM/BAC in our 2022Q4-2024Q1 window -- they stopped tagging
with them around 2021-2022 in favor of newer CECL-era concept names. A third
empirically-verified "fallback" concept is tried last for that reason; which tier was
actually used is always recorded in the notes column for transparency.
"""

import argparse
import time
from collections import defaultdict
from datetime import date

import pandas as pd

from src.data_ingestion import get_xbrl_facts

METRICS_CSV_PATH = "data/processed/metrics.csv"

START_DATE = "2022-10-01"
END_DATE = "2024-03-31"
RATE_LIMIT_SECONDS = 0.15

COMPANIES = {
    "JPM": ("0000019617", "JPMorgan Chase"),
    "BAC": ("0000070858", "Bank of America"),
    "C": ("0000831001", "Citigroup"),
}

# (metric_name, [primary, alternate, fallback], comparability_note)
METRIC_DEFINITIONS = [
    (
        "provision_for_credit_losses",
        [
            "ProvisionForLoanLossesExpensed",
            "ProvisionForDoubtfulAccountsAndLoanLosses",
            "FinancingReceivableExcludingAccruedInterestCreditLossExpenseReversal",
        ],
        "Income-statement provision expense. JPM/BAC stopped tagging the spec's "
        "primary/alternate concepts after ~2021-2022; values here use whichever "
        "concept each filer actually reports in this window, so treat cross-bank "
        "comparison as directional rather than exact line-item equivalence.",
    ),
    (
        "allowance_for_credit_losses",
        [
            "FinancingReceivableAllowanceForCreditLosses",
            "AllowanceForDoubtfulAccountsReceivable",
            "FinancingReceivableAllowanceForCreditLossExcludingAccruedInterest",
        ],
        "Balance-sheet allowance (reserve) for credit losses on financing receivables. "
        "Concept tagging drifted across filers/years; see notes for the concept actually used.",
    ),
    (
        "net_charge_offs",
        [
            "FinancingReceivableAllowanceForCreditLossesWriteOffs",
            None,
            "FinancingReceivableExcludingAccruedInterestAllowanceForCreditLossWriteoffAfterRecovery",
        ],
        "Write-offs net of recoveries (gross charge-offs minus recoveries). The spec's "
        "primary concept has no 2022-2023 data for any of these 3 filers; the fallback "
        "concept here was verified against each bank's reported NCO figures.",
    ),
    (
        "total_loans",
        [
            "LoansAndLeasesReceivableNetReportedAmount",
            "FinancingReceivableBeforeAllowanceForCreditLossAndFee",
            "FinancingReceivableExcludingAccruedInterestBeforeAllowanceForCreditLoss",
        ],
        "Gross loans and leases (before allowance for credit losses), i.e. the loan "
        "portfolio balance, not net-of-allowance.",
    ),
]

CONCEPT_TIERS = ["primary", "alternate", "fallback"]


def _target_duration_days(period_end: str) -> int:
    """Our 3 banks all use a calendar fiscal year, so a '-12-31' end date is a
    fiscal-year-end (10-K, ~365 days); any other end date is a fiscal quarter-end
    (10-Q, ~91 days)."""
    return 365 if period_end.endswith("-12-31") else 91


def _select_best_fact(facts: list[dict]) -> dict:
    """Among facts sharing the same end date, prefer the duration matching a single
    reporting period (one quarter, or one year for a fiscal-year-end date) over a
    YTD-cumulative duplicate that happens to share the same end date, then the
    latest filed."""
    target_days = _target_duration_days(facts[0]["end"])

    def duration_days(fact: dict) -> int:
        start = fact.get("start")
        if not start:
            return target_days  # instant/balance fact -- no duration ambiguity
        return (date.fromisoformat(fact["end"]) - date.fromisoformat(start)).days

    best_distance = min(abs(duration_days(fact) - target_days) for fact in facts)
    closest = [fact for fact in facts if abs(duration_days(fact) - target_days) == best_distance]
    return max(closest, key=lambda fact: fact["filed"])


def fetch_xbrl_metric(cik: str, concept: str, ticker: str, company: str) -> list[dict]:
    facts = get_xbrl_facts(cik)
    time.sleep(RATE_LIMIT_SECONDS)

    concept_data = facts.get("facts", {}).get("us-gaap", {}).get(concept)
    if concept_data is None:
        print(f"WARNING: concept '{concept}' not found for {ticker} (CIK {cik})")
        return []

    usd_facts = concept_data.get("units", {}).get("USD", [])
    in_range = [
        fact
        for fact in usd_facts
        if fact.get("form") in ("10-K", "10-Q") and START_DATE <= fact.get("end", "") <= END_DATE
    ]
    if not in_range:
        return []

    # Step 1: collapse exact-duplicate periods (the same start+end reported as a
    # comparative figure in a later filing too) -- keep the latest-filed instance.
    exact_period_groups: dict[tuple, list[dict]] = defaultdict(list)
    for fact in in_range:
        exact_period_groups[(fact.get("start"), fact["end"])].append(fact)
    deduped_by_period = [max(group, key=lambda fact: fact["filed"]) for group in exact_period_groups.values()]

    # Step 2: where the same end date has multiple distinct durations (e.g. a
    # quarter-only figure and a YTD-cumulative figure both ending the same date),
    # keep only the one matching a single reporting period.
    by_end: dict[str, list[dict]] = defaultdict(list)
    for fact in deduped_by_period:
        by_end[fact["end"]].append(fact)
    final_facts = [_select_best_fact(group) for group in by_end.values()]

    return [
        {
            "ticker": ticker,
            "company": company,
            "period_end": fact["end"],
            "form": fact["form"],
            "value": fact["val"],
            "unit": "USD",
            "concept": concept,
        }
        for fact in sorted(final_facts, key=lambda fact: fact["end"])
    ]


def build_metrics_table() -> pd.DataFrame:
    rows = []

    for metric_name, candidates, comparability_note in METRIC_DEFINITIONS:
        for ticker, (cik, company) in COMPANIES.items():
            metric_rows = []
            concept_used = None
            tier_used = None

            # Try every tier rather than stopping at the first non-empty one: a
            # "primary" concept that technically exists but with only 1-2 stale
            # data points should lose to a sparser-named "fallback" concept that
            # actually has full quarterly coverage in our window.
            for tier, concept in zip(CONCEPT_TIERS, candidates):
                if concept is None:
                    continue
                candidate_rows = fetch_xbrl_metric(cik, concept, ticker, company)
                if len(candidate_rows) > len(metric_rows):
                    metric_rows = candidate_rows
                    concept_used = concept
                    tier_used = tier

            if not metric_rows:
                tried = [c for c in candidates if c is not None]
                rows.append(
                    {
                        "ticker": ticker,
                        "company": company,
                        "period_end": None,
                        "form": None,
                        "metric_name": metric_name,
                        "value": None,
                        "value_billions": None,
                        "unit": None,
                        "source": "unavailable_xbrl",
                        "notes": f"No data in {START_DATE}..{END_DATE} for concepts tried: {tried}. {comparability_note}",
                    }
                )
                continue

            for metric_row in metric_rows:
                rows.append(
                    {
                        "ticker": metric_row["ticker"],
                        "company": metric_row["company"],
                        "period_end": metric_row["period_end"],
                        "form": metric_row["form"],
                        "metric_name": metric_name,
                        "value": metric_row["value"],
                        "value_billions": round(metric_row["value"] / 1e9, 2),
                        "unit": metric_row["unit"],
                        "source": "SEC companyfacts API",
                        "notes": f"concept: {concept_used} ({tier_used}). {comparability_note}",
                    }
                )

    return pd.DataFrame(rows)


def calculate_period_changes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["ticker", "metric_name", "period_end"]).reset_index(drop=True)

    def _with_changes(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values("period_end")
        group["qoq_change"] = group["value"].pct_change(periods=1) * 100
        group["yoy_change"] = group["value"].pct_change(periods=4) * 100
        return group

    available = df[df["source"] != "unavailable_xbrl"].copy()
    unavailable = df[df["source"] == "unavailable_xbrl"].copy()

    if not available.empty:
        available = available.groupby(["ticker", "metric_name"], group_keys=False).apply(
            _with_changes, include_groups=True
        )
    for col in ["qoq_change", "yoy_change"]:
        if col not in unavailable.columns:
            unavailable[col] = None

    return pd.concat([available, unavailable], ignore_index=True).sort_values(
        ["ticker", "metric_name", "period_end"]
    )


def load_metrics(path: str = METRICS_CSV_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def get_metric(df: pd.DataFrame, ticker: str, metric_name: str, period: str) -> float:
    match = df[(df["ticker"] == ticker) & (df["metric_name"] == metric_name) & (df["period_end"] == period)]
    if match.empty:
        raise ValueError(f"No metric '{metric_name}' for {ticker} at period {period}")
    return float(match.iloc[0]["value"])


def _main() -> None:
    arg_parser = argparse.ArgumentParser()
    arg_parser.parse_args()

    print(f"Fetching XBRL metrics for {list(COMPANIES)} from SEC companyfacts API...")
    df = build_metrics_table()
    df = calculate_period_changes(df)

    df.to_csv(METRICS_CSV_PATH, index=False)
    print(f"\nSaved {len(df)} rows to {METRICS_CSV_PATH}\n")

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    summary_cols = ["ticker", "metric_name", "period_end", "form", "value_billions", "qoq_change", "yoy_change", "source"]
    print(df[summary_cols].to_string(index=False))

    fallback_or_unavailable = df[df["source"] != "SEC companyfacts API"]
    non_primary = df[df["notes"].str.contains(r"\((?:alternate|fallback)\)", regex=True, na=False)]

    print("\n--- Concept fallback / availability notes ---")
    if not non_primary.empty:
        for (ticker, metric_name), _ in non_primary.groupby(["ticker", "metric_name"]):
            note = non_primary[(non_primary["ticker"] == ticker) & (non_primary["metric_name"] == metric_name)].iloc[0]["notes"]
            print(f"  {ticker} / {metric_name}: {note.split('.')[0]}.")
    if not fallback_or_unavailable.empty:
        for (ticker, metric_name), _ in fallback_or_unavailable.groupby(["ticker", "metric_name"]):
            print(f"  {ticker} / {metric_name}: UNAVAILABLE (no XBRL data found in range)")
    if non_primary.empty and fallback_or_unavailable.empty:
        print("  None -- all metrics resolved via the primary concept.")


if __name__ == "__main__":
    _main()
