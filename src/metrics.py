"""XBRL-derived quantitative metrics (provisions, net charge-offs) for claim validation."""

import pandas as pd

METRICS_CSV_PATH = "data/processed/metrics.csv"


def load_metrics(path: str = METRICS_CSV_PATH) -> pd.DataFrame:
    raise NotImplementedError


def get_metric(df: pd.DataFrame, ticker: str, metric_name: str, period: str) -> float:
    raise NotImplementedError
