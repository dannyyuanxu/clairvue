"""HTML-to-text cleanup for raw SEC filings (stage 1 of the two-stage cache)."""


def clean_filing_html(html: str) -> str:
    raise NotImplementedError


def extract_section(text: str, section_title: str) -> str:
    raise NotImplementedError
