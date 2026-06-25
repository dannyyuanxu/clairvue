"""HTML-to-text cleanup for raw SEC filings (stage 1 of the two-stage cache).

sec-parser classifies each HTML element semantically (title, table, page header,
table-of-contents, etc.), which lets us drop boilerplate and keep heading structure
when building it does not raise. When it does, we fall back to a generic
BeautifulSoup + markdownify conversion that has no section awareness but always
produces usable plain text.
"""

import argparse
import csv
import re
import warnings
from pathlib import Path

from bs4 import BeautifulSoup
from markdownify import markdownify

DEFAULT_FILING_INDEX_PATH = "data/raw/sec_filings/filing_index.csv"
DEFAULT_TEXT_OUTPUT_DIR = "data/processed/text"

QUALITY_CHECK_TERMS = [
    "consumer credit",
    "credit card",
    "provision for credit loss",
    "allowance for credit loss",
    "net charge-off",
    "delinquency",
    "commercial real estate",
]

_BOILERPLATE_LINE_PATTERNS = [
    re.compile(r"^\s*\d{1,4}\s*$"),  # bare page numbers
    re.compile(r"^\s*table of contents\s*$", re.IGNORECASE),
    re.compile(r"^\s*\[?-{3,}\]?\s*$"),  # exhibit / page separators
    re.compile(r"^\s*\*{3,}\s*$"),
]


def _render_element(element) -> list[str]:
    """Converts one sec-parser semantic element to markdown lines, recursing into
    composite elements; titles become '#'/'##'/'###' headers, tables go through
    markdownify, and table-of-contents/irrelevant/image elements are dropped."""
    import sec_parser as sp
    from sec_parser.semantic_elements.table_element.table_of_contents_element import (
        TableOfContentsElement,
    )

    if isinstance(element, sp.CompositeSemanticElement):
        rendered = []
        for inner in element.inner_elements:
            rendered.extend(_render_element(inner))
        return rendered

    if isinstance(element, TableOfContentsElement):
        return []

    if isinstance(element, (sp.IrrelevantElement, sp.ImageElement)):
        return []

    if isinstance(element, sp.TopSectionTitle):
        prefix = "#" if element.level == 0 else "##"
        text = element.text.strip()
        return [f"{prefix} {text}"] if text else []

    if isinstance(element, sp.TitleElement):
        text = element.text.strip()
        return [f"### {text}"] if text else []

    if isinstance(element, sp.TableElement):
        table_md = markdownify(element.get_source_code()).strip()
        return [table_md] if table_md else []

    text = element.text.strip()
    return [text] if text else []


def _strip_hidden_ixbrl_header(html: str) -> str:
    """Remove the hidden <ix:header> block (inline-XBRL context/resource tagging).

    It is never rendered and carries no narrative content, but sec-parser's
    element classifier doesn't always mark its children as irrelevant — without
    this, one of them ends up as a single ~150K-character TextElement of
    concatenated XBRL context IDs at the start of every filing.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        soup = BeautifulSoup(html, "lxml")
        for tag in soup.find_all("ix:header"):
            tag.decompose()
    return str(soup)


def _parse_with_sec_parser(html: str) -> str:
    """Primary parse path: structures the filing semantically (headings, tables,
    boilerplate) via sec-parser. Raises on malformed HTML it can't classify --
    callers should catch and fall back to `_parse_with_bs4`."""
    import sec_parser as sp

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        elements = sp.Edgar10QParser().parse(html)

    lines = []
    for element in elements:
        lines.extend(_render_element(element))
    return "\n\n".join(lines)


def _parse_with_bs4(html: str) -> str:
    """Fallback parse path when sec-parser can't classify a filing: a generic
    BeautifulSoup + markdownify conversion with no section/heading awareness,
    but that always produces usable plain text."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return markdownify(str(soup), heading_style="ATX")


def _remove_boilerplate(text: str) -> str:
    kept_lines = [
        line
        for line in text.splitlines()
        if not any(pattern.match(line) for pattern in _BOILERPLATE_LINE_PATTERNS)
    ]
    cleaned = "\n".join(kept_lines)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip() + "\n"


def parse_filing(html_path: str, output_dir: str) -> str:
    """Parses one raw filing HTML to cleaned markdown, trying sec-parser first and
    falling back to BeautifulSoup if it raises. Writes the result and returns its path."""
    html_path = Path(html_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{html_path.stem}.md"

    html = html_path.read_text(encoding="utf-8", errors="replace")
    html = _strip_hidden_ixbrl_header(html)

    try:
        markdown = _parse_with_sec_parser(html)
    except Exception as exc:
        print(f"  [fallback] sec-parser failed on {html_path.name} ({exc!r}); using BeautifulSoup")
        markdown = _parse_with_bs4(html)

    markdown = _remove_boilerplate(markdown)
    output_path.write_text(markdown, encoding="utf-8")
    return str(output_path)


def parse_all_filings(filing_index_path: str = DEFAULT_FILING_INDEX_PATH, force_reparse: bool = False) -> list[dict]:
    """Main entry point: parses every filing in the index, skipping any whose
    cached markdown already exists unless force_reparse is set (stage 1 of the
    two-stage cache -- filing HTML doesn't change, so this is safe to skip)."""
    output_dir = Path(DEFAULT_TEXT_OUTPUT_DIR)
    results = []

    with open(filing_index_path, newline="") as f:
        filing_rows = list(csv.DictReader(f))

    for row in filing_rows:
        local_path = Path(row["local_path"])
        text_path = output_dir / f"{local_path.stem}.md"

        if text_path.exists() and not force_reparse:
            status = "cached"
        else:
            parse_filing(str(local_path), str(output_dir))
            status = "parsed"

        print(f"{row['ticker']} {row['form']} {row['filing_date']}: {status}")
        results.append(
            {
                "ticker": row["ticker"],
                "form": row["form"],
                "filing_date": row["filing_date"],
                "text_path": str(text_path),
                "status": status,
            }
        )

    return results


def check_extraction_quality(text_path: str) -> dict:
    """Counts occurrences of each QUALITY_CHECK_TERMS in the parsed text -- a coarse
    smoke test, not a quality score. The caller (`_main`) flags a filing as suspicious
    if 'provision for credit loss' has zero hits, since every filing in this corpus
    should mention it somewhere; a zero usually means extraction broke, not that the
    term is genuinely absent."""
    text = Path(text_path).read_text(encoding="utf-8").lower()
    return {term: text.count(term) for term in QUALITY_CHECK_TERMS}


def _main() -> None:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--force-reparse", action="store_true")
    args = arg_parser.parse_args()

    results = parse_all_filings(force_reparse=args.force_reparse)

    print(f"\n{'Ticker':<8}{'Form':<8}{'Filing Date':<14}{'Status':<10}{'Text Path'}")
    for r in results:
        print(f"{r['ticker']:<8}{r['form']:<8}{r['filing_date']:<14}{r['status']:<10}{r['text_path']}")

    print("\nQuality check (term counts; flagged if 'provision for credit loss' == 0):")
    for r in results:
        counts = check_extraction_quality(r["text_path"])
        flag = " <-- SUSPICIOUS" if counts["provision for credit loss"] == 0 else ""
        print(f"{r['ticker']} {r['form']} {r['filing_date']}: {counts}{flag}")


if __name__ == "__main__":
    _main()
