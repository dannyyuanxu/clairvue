"""CLI: fetch the 15 target 10-K/10-Q filings from SEC EDGAR into data/raw/sec_filings/."""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.data_ingestion import download_filing, get_filing_index

CIKS = {
    "JPM": "0000019617",
    "BAC": "0000070858",
    "C": "0000831001",
}

COMPANIES = {
    "JPM": "JPMorgan Chase",
    "BAC": "Bank of America",
    "C": "Citigroup",
}

TARGET_FORMS = {"10-K", "10-Q"}
START_DATE = "2023-01-01"
END_DATE = "2024-03-31"

RAW_DIR = Path("data/raw/sec_filings")
INDEX_CSV_PATH = RAW_DIR / "filing_index.csv"
INDEX_FIELDNAMES = [
    "ticker",
    "company",
    "cik",
    "form",
    "filing_date",
    "accession_number",
    "primary_document",
    "filing_url",
    "local_path",
]


def main() -> None:
    """Fetches the 15 target 10-K/10-Q filings for all 3 banks, skipping any whose
    local HTML file already exists, and writes data/raw/sec_filings/filing_index.csv."""
    if not settings.SEC_USER_AGENT:
        raise RuntimeError("SEC_USER_AGENT must be set (see .env.example) before downloading filings.")

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    downloaded = 0
    skipped = 0

    for ticker, cik in CIKS.items():
        company = COMPANIES[ticker]
        filings = get_filing_index(cik, TARGET_FORMS, START_DATE, END_DATE)
        print(f"{ticker}: found {len(filings)} filings (10-K/10-Q, {START_DATE}..{END_DATE})")

        for filing in filings:
            cik_int = str(int(cik))
            accession_nodashes = filing["accession_number"].replace("-", "")
            filing_url = (
                f"https://www.sec.gov/Archives/edgar/data/{cik_int}/"
                f"{accession_nodashes}/{filing['primary_document']}"
            )
            local_path = RAW_DIR / f"{ticker}_{filing['form']}_{filing['filing_date']}.html"

            if local_path.exists():
                status = "skipped"
                skipped += 1
            else:
                download_filing(filing_url, str(local_path))
                status = "downloaded"
                downloaded += 1

            rows.append(
                {
                    "ticker": ticker,
                    "company": company,
                    "cik": cik,
                    "form": filing["form"],
                    "filing_date": filing["filing_date"],
                    "accession_number": filing["accession_number"],
                    "primary_document": filing["primary_document"],
                    "filing_url": filing_url,
                    "local_path": str(local_path),
                    "status": status,
                }
            )

    with open(INDEX_CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in INDEX_FIELDNAMES})

    print(f"\nTotal: {downloaded} downloaded, {skipped} skipped, {len(rows)} filings in index")
    print(f"Index saved to {INDEX_CSV_PATH}\n")

    header = f"{'Ticker':<8}{'Form':<8}{'Filing Date':<14}{'Local Path':<48}{'Status'}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row['ticker']:<8}{row['form']:<8}{row['filing_date']:<14}{row['local_path']:<48}{row['status']}")


if __name__ == "__main__":
    main()
