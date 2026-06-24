"""SEC EDGAR filing retrieval: submissions/XBRL APIs and raw HTML download."""

import time
from pathlib import Path

import requests

from src.config import settings

EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
EDGAR_SUBMISSIONS_PAGE_URL = "https://data.sec.gov/submissions/{name}"
EDGAR_XBRL_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

RATE_LIMIT_SECONDS = 0.15


def _sec_headers() -> dict:
    return {"User-Agent": settings.SEC_USER_AGENT}


def get_cik_submissions(cik: str) -> dict:
    """Fetch the full submissions JSON for a CIK, merging any paginated
    older-filings pages into filings.recent's parallel arrays."""
    response = requests.get(EDGAR_SUBMISSIONS_URL.format(cik=cik), headers=_sec_headers())
    response.raise_for_status()
    time.sleep(RATE_LIMIT_SECONDS)
    data = response.json()

    recent = data["filings"]["recent"]
    merged = {key: list(values) for key, values in recent.items()}

    for page in data["filings"].get("files", []):
        page_response = requests.get(
            EDGAR_SUBMISSIONS_PAGE_URL.format(name=page["name"]), headers=_sec_headers()
        )
        page_response.raise_for_status()
        time.sleep(RATE_LIMIT_SECONDS)
        page_data = page_response.json()
        for key in merged:
            merged[key].extend(page_data.get(key, []))

    data["filings"]["recent"] = merged
    return data


def get_filing_index(cik: str, forms: set[str], start_date: str, end_date: str) -> list[dict]:
    """Filter a CIK's submissions down to filings matching form type and filing-date range."""
    submissions = get_cik_submissions(cik)
    recent = submissions["filings"]["recent"]

    matches = []
    for form, filing_date, accession_number, primary_document in zip(
        recent["form"], recent["filingDate"], recent["accessionNumber"], recent["primaryDocument"]
    ):
        if not primary_document:
            continue
        if form in forms and start_date <= filing_date <= end_date:
            matches.append(
                {
                    "cik": cik,
                    "form": form,
                    "filing_date": filing_date,
                    "accession_number": accession_number,
                    "primary_document": primary_document,
                }
            )
    return matches


def download_filing(filing_url: str, dest_path: str) -> str:
    response = requests.get(filing_url, headers=_sec_headers())
    response.raise_for_status()
    time.sleep(RATE_LIMIT_SECONDS)
    Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
    Path(dest_path).write_bytes(response.content)
    return dest_path


def get_xbrl_facts(cik: str) -> dict:
    raise NotImplementedError
