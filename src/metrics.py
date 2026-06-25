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

# Window starts in 2019 to capture a pre-pandemic baseline: "normalizing" credit
# losses only mean anything relative to the pre-2020 norm and the 2020-21
# stimulus-suppressed trough. Filing *text* is still only ingested for Q4'22-Q4'23
# (see CLAUDE.md scope) -- only the quantitative XBRL series is backfilled here, which
# is free: the cached companyfacts JSON already contains the full history.
START_DATE = "2019-01-01"
END_DATE = "2024-03-31"
# Concept selection must prioritize the demo/text window over raw row count: widening
# START_DATE to 2019 means an older, now-discontinued concept can have MORE total rows than
# the current post-CECL concept while stopping in 2022 -- which would silently drop the 2023
# data the demo needs. So candidates are ranked by coverage at/after this date first.
SELECTION_WINDOW_START = "2022-10-01"
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
        None,  # stitched across two concepts -- see _build_net_charge_offs_rows
        "Charge-offs (YTD-cumulative; annualized elsewhere via fact duration). Stitched: the "
        "post-CECL net concept (WriteoffAfterRecovery) is used wherever available -- it covers "
        "~2021 onward, including the entire text-corpus window -- and the older gross WriteOffs "
        "concept fills the pre-2021 baseline only. So pre-2021 figures are GROSS of recoveries "
        "(marginally overstated vs net); the demo-window figures are unchanged net values. "
        "concept_used is recorded per row.",
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
    (
        "net_charge_off_rate",
        None,  # computed -- see _build_net_charge_off_rate_rows
        "Annualized net charge-offs as a percent of net loans -- the loss RATE, not a "
        "dollar amount. This is the interpretable, cross-bank-comparable view of credit "
        "losses. Denominator is period-end net loans (approximates average loans).",
    ),
]

CONCEPT_TIERS = ["primary", "alternate", "fallback"]

GROSS_LOANS_CONCEPT = "FinancingReceivableExcludingAccruedInterestBeforeAllowanceForCreditLoss"
ALLOWANCE_CONCEPT_FOR_NET_LOANS = "FinancingReceivableAllowanceForCreditLossExcludingAccruedInterest"

# Net charge-offs, in priority order: the post-CECL net-of-recovery concept first (used
# wherever it exists, which is the full text-corpus window), then the older gross concept
# to backfill the pre-2021 baseline. No single concept spans 2019..2024 for any of the 3
# banks, so the series is stitched per period (first concept that has the period wins).
NET_CHARGE_OFFS_CONCEPTS = [
    "FinancingReceivableExcludingAccruedInterestAllowanceForCreditLossWriteoffAfterRecovery",
    "FinancingReceivableAllowanceForCreditLossesWriteOffs",
]


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
    """Pulls one concept's quarterly facts for one bank in [START_DATE, END_DATE],
    deduplicated first by exact (start, end) period (keep latest filed), then by
    end date when multiple durations exist for the same end (keep shortest)."""
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

    def period_days(fact: dict) -> int | None:
        start = fact.get("start")
        if not start:
            return None  # instant/balance fact (e.g. allowance, loans) -- no duration
        return (date.fromisoformat(fact["end"]) - date.fromisoformat(start)).days

    return [
        {
            "ticker": ticker,
            "company": company,
            "period_end": fact["end"],
            "form": fact["form"],
            "value": fact["val"],
            "unit": "USD",
            "concept": concept,
            "start": fact.get("start"),
            "period_days": period_days(fact),
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
            "start": None,  # balance (instant) figure -- no duration
            "period_days": None,
        }
        for period_end in sorted(set(gross_by_period) & set(allowance_by_period))
    ]


def _build_net_charge_offs_rows(ticker: str, cik: str, company: str) -> list[dict]:
    """Net charge-offs stitched across concepts (see NET_CHARGE_OFFS_CONCEPTS): for each
    period, the first concept in priority order that reports it wins, so the net post-CECL
    figure is used wherever it exists and the older gross concept only backfills earlier
    periods. concept stays on each row so concept_used is recorded per period."""
    by_period: dict[str, dict] = {}
    for concept in NET_CHARGE_OFFS_CONCEPTS:
        for row in fetch_xbrl_metric(cik, concept, ticker, company):
            by_period.setdefault(row["period_end"], row)
    return [by_period[period] for period in sorted(by_period)]


def _build_net_charge_off_rate_rows(nco_rows: list[dict], total_loans_rows: list[dict]) -> list[dict]:
    """Net charge-off RATE -- the headline interpretable metric. A dollar charge-off
    figure is meaningless without a denominator; the annualized rate (loss as a % of
    the loan book) is what tells you whether losses are high or low and is directly
    comparable across banks and against the pre-pandemic baseline.

    Annualized using each fact's actual duration (365 / period_days) so a quarterly
    and a YTD-cumulative charge-off figure both normalize to the same annual basis.
    Denominator is period-end net loans (an approximation of average loans -- noted in
    the comparability_note)."""
    loans_by_period = {row["period_end"]: row for row in total_loans_rows}
    rate_rows = []
    for nco in nco_rows:
        period_end = nco["period_end"]
        period_days = nco.get("period_days")
        loans = loans_by_period.get(period_end)
        if loans is None or not period_days or not loans["value"]:
            continue
        annualized_nco = nco["value"] * (365 / period_days)
        rate_pct = annualized_nco / loans["value"] * 100
        rate_rows.append(
            {
                "ticker": nco["ticker"],
                "company": nco["company"],
                "period_end": period_end,
                "form": nco["form"],
                "value": round(rate_pct, 4),
                "unit": "percent_annualized",
                "concept": "computed: net_charge_offs (annualized) / total_loans (net)",
                "start": None,
                "period_days": None,
            }
        )
    return rate_rows


def build_metrics_table() -> pd.DataFrame:
    """Builds the full metrics table for all banks x all metrics. For each pair,
    tries every candidate concept (see METRIC_DEFINITIONS) and keeps whichever
    yields the most in-window rows, rather than stopping at the first match --
    concept coverage is bank-specific post-CECL, so "first non-empty" can silently
    pick a sparse/wrong concept over a fully-covered one."""
    rows = []
    # Raw (pre-DataFrame) rows kept per (ticker, metric_name) so derived metrics --
    # net_charge_off_rate -- can reference earlier metrics' rows (with their durations)
    # instead of re-fetching. METRIC_DEFINITIONS is ordered so dependencies come first.
    raw_rows_by_key: dict[tuple[str, str], list[dict]] = {}

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
            elif metric_name == "net_charge_offs":
                metric_rows = _build_net_charge_offs_rows(ticker, cik, company)
                if metric_rows:
                    concept_used = "stitched (see notes / per-row concept)"
                    tier_used = "stitched"
            elif metric_name == "net_charge_off_rate":
                metric_rows = _build_net_charge_off_rate_rows(
                    raw_rows_by_key.get((ticker, "net_charge_offs"), []),
                    raw_rows_by_key.get((ticker, "total_loans"), []),
                )
                if metric_rows:
                    concept_used = metric_rows[0]["concept"]
                    tier_used = "computed"
            else:
                # Rank candidates by demo-window coverage first, then total rows: a concept
                # that technically exists but doesn't reach the demo window (or has only 1-2
                # stale points) should lose to one with full coverage there. Ranking on raw
                # row count alone would wrongly prefer a long-history concept that stops in 2022.
                best_key = (-1, -1)
                for tier, concept in zip(CONCEPT_TIERS, candidates):
                    if concept is None:
                        continue
                    candidate_rows = fetch_xbrl_metric(cik, concept, ticker, company)
                    recent = sum(1 for r in candidate_rows if r["period_end"] >= SELECTION_WINDOW_START)
                    key = (recent, len(candidate_rows))
                    if key > best_key:
                        best_key = key
                        metric_rows = candidate_rows
                        concept_used = concept
                        tier_used = tier

            raw_rows_by_key[(ticker, metric_name)] = metric_rows

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

            # Rate metrics carry a percent in `value`; value_billions is meaningless
            # for them (left NaN). Dollar metrics keep the billions convenience column.
            is_rate = metric_name == "net_charge_off_rate"
            for metric_row in metric_rows:
                rows.append(
                    {
                        "ticker": metric_row["ticker"],
                        "company": metric_row["company"],
                        "period_end": metric_row["period_end"],
                        "form": metric_row["form"],
                        "metric_name": metric_name,
                        "value": metric_row["value"],
                        "value_billions": None if is_rate else round(metric_row["value"] / 1e9, 2),
                        "unit": metric_row["unit"],
                        # Stitched metrics (net_charge_offs, total_loans) carry the real
                        # concept on each row; record that rather than the summary label.
                        "concept_used": metric_row.get("concept", concept_used),
                        "source": "computed" if tier_used == "computed" else "SEC companyfacts API",
                        "notes": f"concept tier: {tier_used}. {comparability_note}",
                    }
                )

    df = pd.DataFrame(rows)
    _print_coverage_check(df)
    return df


def _print_coverage_check(df: pd.DataFrame) -> None:
    """Prints row count + period range per (ticker, metric); warns below MIN_EXPECTED_ROWS
    so a silently-wrong or sparse concept can't hide in the output."""
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
    """Adds qoq_change_pct/yoy_change_pct (4 rows back = 1 year, since each ticker/metric
    group has one row per quarter). Uses abs() in the denominator so the sign is correct
    even if a prior-period value were negative."""
    df = df.sort_values(["ticker", "metric_name", "period_end"]).reset_index(drop=True)

    grouped_value = df.groupby(["ticker", "metric_name"])["value"]
    prior_quarter = grouped_value.shift(1)
    prior_year = grouped_value.shift(4)

    df["qoq_change_pct"] = (df["value"] - prior_quarter) / prior_quarter.abs() * 100
    df["yoy_change_pct"] = (df["value"] - prior_year) / prior_year.abs() * 100
    # Absolute deltas too: for a rate metric a percentage-point move (e.g. +0.09pp) is the
    # natural reading, not a "% change of a percent". Dollar metrics use the pct columns.
    df["qoq_change_abs"] = df["value"] - prior_quarter
    df["yoy_change_abs"] = df["value"] - prior_year
    return df


def load_metrics(path: str = METRICS_CSV_PATH) -> pd.DataFrame:
    """Loads the curated metrics table -- the public entry point other modules use."""
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
        if row["metric_name"] == "net_charge_off_rate":
            value_str = "N/A" if pd.isna(row["value"]) else f"{row['value']:.2f}%"
            change_str = "N/A" if pd.isna(row["yoy_change_abs"]) else f"{row['yoy_change_abs']:+.2f}pp"
        else:
            value_str = _format_dollar(row["value_billions"])
            change_str = f"{_format_pct(row['yoy_change_pct'])}"
        print(f"{row['ticker']} | {row['metric_name']} | {period} | {value_str} | {change_str} YoY")

    unavailable = df[df["source"] == "unavailable_xbrl"]
    print("\n--- Metrics with zero rows (concept not found / incomplete) ---")
    if unavailable.empty:
        print("  None -- every (ticker, metric_name) pair resolved to real data.")
    else:
        for _, row in unavailable.drop_duplicates(["ticker", "metric_name"]).iterrows():
            print(f"  WARNING: {row['ticker']} / {row['metric_name']}: unavailable_xbrl")


if __name__ == "__main__":
    _main()
