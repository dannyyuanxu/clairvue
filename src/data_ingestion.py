"""SEC EDGAR filing retrieval: submissions/XBRL APIs and raw HTML download."""

from src.config import settings

EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
EDGAR_XBRL_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"


def get_filing_index(cik: str) -> dict:
    raise NotImplementedError


def download_filing(filing_url: str, dest_path: str) -> str:
    raise NotImplementedError


def get_xbrl_facts(cik: str) -> dict:
    raise NotImplementedError
