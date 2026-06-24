"""XBRL-derived quantitative metrics (provisions, net charge-offs) for claim validation.

Note on concept selection: all three banks adopted CECL by 2020-2022, and the pre-CECL
concept names referenced in older documentation (ProvisionForLoanLossesExpensed,
FinancingReceivableAllowanceForCreditLosses as a sole source, etc.) return zero or
near-zero rows for JPM/BAC/C in this window. Concept coverage is bank-specific -- a
concept that works for one bank can return zero rows for another -- so every metric
below is tried across multiple candidate concepts and whichever yields the most
in-window data wins; the concept actually used is always recorded in concept_used.
"""

import argparse
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

import pandas as pd

from src.data_ingestion import get_xbrl_facts

METRICS_CSV_PATH = "data/processed/metrics.csv"
XBRL_CACHE_DIR = Path("data/raw/xbrl")

START_DATE = "2022-10-01"
END_DATE = "2024-03-31"
RATE_LIMIT_SECONDS = 0.15
MIN_EXPECTED_ROWS = 4  # one per quarter in range; fewer means the concept is wrong/incomplete

COMPANIES = {
    "JPM": ("0000019617", "JPMorgan Chase"),
    "BAC": ("0000070858", "Bank of America"),
    "C": ("0000831001", "Citigroup"),
}

# (metric_name, [primary, alternate, fallback], comparability_note)
# Stale/wrong concepts (ProvisionForLoanLossesExpensed, ProvisionForCreditLossesOnFinancingReceivables,
# LoansAndLeasesReceivableGross) are deliberately excluded -- confirmed against the live API to
# return zero in-window rows for at least JPM and BAC.
METRIC_DEFINITIONS = [
    (
        "provision_for_credit_losses",
        [
            "ProvisionForLoanLeaseAndOtherLosses",
            None,
            "FinancingReceivableExcludingAccruedInterestCreditLossExpenseReversal",
        ],
        "Income-statement provision expense. Concept tagging is bank-specific post-CECL; "
        "treat cross-bank comparison as directional rather than exact line-item equivalence.",
    ),
    (
        "allowance_for_credit_losses",
        [
            "FinancingReceivableAllowanceForCreditLossExcludingAccruedInterest",
            "FinancingReceivableAllowanceForCreditLosses",
            "AllowanceForLoanAndLeaseLosses",
        ],
        "Balance-sheet allowance (reserve) for credit losses on financing receivables.",
    ),
    (
        "net_charge_offs",
        [
            "FinancingReceivableExcludingAccruedInterestAllowanceForCreditLossWriteoffAfterRecovery",
            "FinancingReceivableAllowanceForCreditLossesWriteOffs",
            None,
        ],
        "Write-offs net of recoveries (gross charge-offs minus recoveries). Less consistently "
        "tagged across banks than the other 3 metrics -- see concept_used per row.",
    ),
    (
        "total_loans",
        None,  # computed -- see _build_total_loans_rows
        "Loan portfolio balance, net of allowance for credit losses. Computed as "
        "gross loans minus allowance rather than trusting each bank's own "
        "'...AfterAllowanceForCreditLoss' tag: that tag is internally inconsistent "
        "for JPM (its 'after' value is HIGHER than its 'before' value in every "
        "period -- impossible for a true net-of-allowance figure), even though it "
        "happens to equal before-minus-allowance exactly for BAC and C. Computing "
        "it ourselves guarantees net <= gross for all 3 banks.",
    ),
]

CONCEPT_TIERS = ["primary", "alternate", "fallback"]

GROSS_LOANS_CONCEPT = "FinancingReceivableExcludingAccruedInterestBeforeAllowanceForCreditLoss"
ALLOWANCE_CONCEPT_FOR_NET_LOANS = "FinancingReceivableAllowanceForCreditLossExcludingAccruedInterest"


def _get_company_facts(ticker: str, cik: str) -> dict:
    """Cache the full companyfacts JSON locally -- it's 5-20MB, so fetch once per
    bank per build session rather than once per concept tried."""
    cache_path = XBRL_CACHE_DIR / f"{ticker}_companyfacts.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    facts = get_xbrl_facts(cik)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(facts, f)
    return facts


def _select_best_fact(facts: list[dict]) -> dict:
    """Among facts sharing the same end date, the companyfacts API can include both
    a quarterly point-in-time fact and a YTD-cumulative fact -- these are NOT
    duplicates, they report different economic quantities. Keep the SHORTEST
    duration (end - start); for instant/balance facts (no start), duration is 0,
    so ties there are broken by latest filed."""

    def duration_days(fact: dict) -> int:
        start = fact.get("start")
        if not start:
            return 0
        return (date.fromisoformat(fact["end"]) - date.fromisoformat(start)).days

    shortest = min(duration_days(fact) for fact in facts)
    candidates = [fact for fact in facts if duration_days(fact) == shortest]
    return max(candidates, key=lambda fact: fact["filed"])


def fetch_xbrl_metric(cik: str, concept: str, ticker: str, company: str) -> list[dict]:
    facts = _get_company_facts(ticker, cik)

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
        print(f"WARNING: concept '{concept}' found for {ticker} but has no rows in {START_DATE}..{END_DATE}")
        return []

    # Step 1: collapse exact-duplicate periods (the same start+end reported as a
    # comparative figure in a later filing too) -- keep the latest-filed instance.
    exact_period_groups: dict[tuple, list[dict]] = defaultdict(list)
    for fact in in_range:
        exact_period_groups[(fact.get("start"), fact["end"])].append(fact)
    deduped_by_period = [max(group, key=lambda fact: fact["filed"]) for group in exact_period_groups.values()]

    # Step 2: where the same end date has multiple distinct durations (quarterly
    # point-in-time vs YTD-cumulative), keep only the shortest-duration fact.
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


def _build_total_loans_rows(ticker: str, cik: str, company: str) -> list[dict]:
    """Total loans, net of allowance, computed as gross loans minus allowance
    (see the comparability_note on this metric for why we don't trust the
    single-tag '...AfterAllowanceForCreditLoss' concept directly)."""
    gross_by_period = {r["period_end"]: r for r in fetch_xbrl_metric(cik, GROSS_LOANS_CONCEPT, ticker, company)}
    allowance_by_period = {
        r["period_end"]: r for r in fetch_xbrl_metric(cik, ALLOWANCE_CONCEPT_FOR_NET_LOANS, ticker, company)
    }

    return [
        {
            "ticker": ticker,
            "company": company,
            "period_end": period_end,
            "form": gross_by_period[period_end]["form"],
            "value": gross_by_period[period_end]["value"] - allowance_by_period[period_end]["value"],
            "unit": "USD",
            "concept": f"computed: {GROSS_LOANS_CONCEPT} minus {ALLOWANCE_CONCEPT_FOR_NET_LOANS}",
        }
        for period_end in sorted(set(gross_by_period) & set(allowance_by_period))
    ]


def build_metrics_table() -> pd.DataFrame:
    rows = []

    for metric_name, candidates, comparability_note in METRIC_DEFINITIONS:
        for ticker, (cik, company) in COMPANIES.items():
            metric_rows: list[dict] = []
            concept_used = None
            tier_used = None

            if metric_name == "total_loans":
                metric_rows = _build_total_loans_rows(ticker, cik, company)
                if metric_rows:
                    concept_used = metric_rows[0]["concept"]
                    tier_used = "computed"
            else:
                # Try every tier rather than stopping at the first non-empty one: a
                # concept that technically exists but with only 1-2 stale data points
                # should lose to one with full quarterly coverage in our window.
                for tier, concept in zip(CONCEPT_TIERS, candidates):
                    if concept is None:
                        continue
                    candidate_rows = fetch_xbrl_metric(cik, concept, ticker, company)
                    if len(candidate_rows) > len(metric_rows):
                        metric_rows = candidate_rows
                        concept_used = concept
                        tier_used = tier

            if not metric_rows:
                tried = [GROSS_LOANS_CONCEPT, ALLOWANCE_CONCEPT_FOR_NET_LOANS] if candidates is None else [
                    c for c in candidates if c is not None
                ]
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
                        "concept_used": None,
                        "source": "unavailable_xbrl",
                        "notes": f"No post-CECL concept found. Tried: {tried}. {comparability_note}",
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
                        "concept_used": concept_used,
                        "source": "SEC companyfacts API",
                        "notes": f"concept tier: {tier_used}. {comparability_note}",
                    }
                )

    df = pd.DataFrame(rows)
    _print_coverage_check(df)
    return df


def _print_coverage_check(df: pd.DataFrame) -> None:
    print("\n--- Coverage check (expect >= 4 rows per ticker/metric, one per quarter) ---")
    for (ticker, metric_name), group in df.groupby(["ticker", "metric_name"], sort=True):
        valid = group[group["period_end"].notna()]
        count = len(valid)
        if count == 0:
            print(f"  WARNING: {ticker} / {metric_name}: 0 rows -- unavailable_xbrl")
            continue
        min_end, max_end = valid["period_end"].min(), valid["period_end"].max()
        concept = valid.iloc[0]["concept_used"]
        if count < MIN_EXPECTED_ROWS:
            print(
                f"  WARNING: {ticker} / {metric_name}: only {count} rows ({min_end}..{max_end}), "
                f"concept_used={concept} -- likely wrong/incomplete concept for this bank"
            )
        else:
            print(f"  OK: {ticker} / {metric_name}: {count} rows ({min_end}..{max_end}), concept_used={concept}")


def calculate_period_changes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["ticker", "metric_name", "period_end"]).reset_index(drop=True)

    grouped_value = df.groupby(["ticker", "metric_name"])["value"]
    prior_quarter = grouped_value.shift(1)
    prior_year = grouped_value.shift(4)

    df["qoq_change_pct"] = (df["value"] - prior_quarter) / prior_quarter.abs() * 100
    df["yoy_change_pct"] = (df["value"] - prior_year) / prior_year.abs() * 100
    return df


def load_metrics(path: str = METRICS_CSV_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def _period_label(period_end: str) -> str:
    year, month, _ = period_end.split("-")
    quarter = (int(month) - 1) // 3 + 1
    return f"{year}-Q{quarter}"


def _format_dollar(value_billions) -> str:
    return "N/A" if pd.isna(value_billions) else f"${value_billions:.2f}B"


def _format_pct(value) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{'+' if value >= 0 else ''}{value:.1f}%"


def _main() -> None:
    arg_parser = argparse.ArgumentParser()
    arg_parser.parse_args()

    print(f"Fetching XBRL metrics for {list(COMPANIES)} from SEC companyfacts API...")
    df = build_metrics_table()
    df = calculate_period_changes(df)

    df.to_csv(METRICS_CSV_PATH, index=False)
    print(f"\nSaved {len(df)} rows to {METRICS_CSV_PATH}\n")

    print("--- Per-bank, per-metric summary ---")
    available = df[df["source"] != "unavailable_xbrl"].sort_values(["metric_name", "ticker", "period_end"])
    for _, row in available.iterrows():
        period = _period_label(row["period_end"])
        print(
            f"{row['ticker']} | {row['metric_name']} | {period} | "
            f"{_format_dollar(row['value_billions'])} | {_format_pct(row['yoy_change_pct'])} YoY"
        )

    unavailable = df[df["source"] == "unavailable_xbrl"]
    print("\n--- Metrics with zero rows (concept not found / incomplete) ---")
    if unavailable.empty:
        print("  None -- every (ticker, metric_name) pair resolved to real data.")
    else:
        for _, row in unavailable.drop_duplicates(["ticker", "metric_name"]).iterrows():
            print(f"  WARNING: {row['ticker']} / {row['metric_name']}: unavailable_xbrl")


if __name__ == "__main__":
    _main()
