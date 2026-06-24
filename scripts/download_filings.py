"""CLI: fetch the 15 target 10-K/10-Q filings from SEC EDGAR into data/raw/sec_filings/."""

from src.data_ingestion import download_filing, get_filing_index

CIKS = {
    "JPM": "0000019617",
    "BAC": "0000070858",
    "C": "0000831001",
}


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
