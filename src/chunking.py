"""Section-aware chunking with risk-theme tagging and optional LLM context enrichment.

Note: the metadata field names below ("filing_type", "enriched") fix two apparent
typos in the original spec ("fing_type", "enrhed") to match the schema documented
in CLAUDE.md / the PRD and the "enriched" field name used elsewhere in this module.
"""

import argparse
import csv
import json
import re
import time
from collections import Counter
from pathlib import Path

from src.llm_client import LLMClient

FILING_INDEX_PATH = "data/raw/sec_filings/filing_index.csv"
STATEMENTS_CSV_PATH = "data/raw/management_statements/statements.csv"
TEXT_DIR = Path("data/processed/text")
CHUNKS_OUTPUT_PATH = Path("data/processed/chunks.jsonl")
ENRICHMENT_CACHE_PATH = Path("data/processed/enrichment_cache.jsonl")

MAX_SECTION_TOKENS = 900
CHUNK_SIZE_TOKENS = 800
CHUNK_OVERLAP_TOKENS = 100
MIN_CHUNK_TOKENS = 50

HEADER_PATTERN = re.compile(r"^(#{1,3})\s+(.*)$")
SPLIT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
MAX_HEADER_WORDS = 20  # sec-parser occasionally misclassifies a long bolded
# disclaimer paragraph as a TitleElement; a real heading is never this long,
# so anything longer is treated as body text instead of a section boundary.

CONSUMER_CREDIT_KEYWORDS = [
    "consumer credit",
    "credit card",
    "card services",
    "auto loan",
    "auto finance",
    "delinquency",
    "charge-off",
    "net charge-off",
    "allowance for credit loss",
    "provision for credit loss",
    "consumer banking",
    "retail banking",
    "personal lending",
    "home equity",
]

COMMERCIAL_REAL_ESTATE_KEYWORDS = [
    "commercial real estate",
    " cre ",
    "office",
    "multifamily",
    "commercial mortgage",
    "criticized loan",
    "nonaccrual",
]

ENRICHMENT_SYSTEM_PROMPT = (
    "You are a financial document analyst. Write 1-2 sentences describing the "
    "structural context of the following text excerpt: which company, filing type, "
    "reporting period, and section it comes from, and what financial topic it covers. "
    "Be specific. Do not summarize the content itself."
)
ENRICHMENT_BATCH_SIZE = 10
ENRICHMENT_BATCH_SLEEP_SECONDS = 1


def _token_count(text: str) -> int:
    return int(len(text.split()) * 1.3)


def _tag_risk_theme(text: str) -> str:
    lowered = f" {text.lower()} "
    if any(keyword in lowered for keyword in CONSUMER_CREDIT_KEYWORDS):
        return "consumer_credit"
    if any(keyword in lowered for keyword in COMMERCIAL_REAL_ESTATE_KEYWORDS):
        return "commercial_real_estate"
    return "general"


def _reporting_period(primary_document: str, fallback_filing_date: str) -> str:
    match = re.search(r"(\d{8})\.\w+$", primary_document)
    date_digits = match.group(1) if match else fallback_filing_date.replace("-", "")
    year, month = int(date_digits[:4]), int(date_digits[4:6])
    quarter = (month - 1) // 3 + 1
    return f"{year}-Q{quarter}"


def _split_by_headers(markdown_text: str) -> list[tuple[str, str]]:
    """Split on '#'/'##'/'###' lines. Returns (nearest_header_above, section_body) pairs."""
    sections = []
    current_title = ""
    current_lines: list[str] = []

    for line in markdown_text.splitlines():
        match = HEADER_PATTERN.match(line)
        if match and len(match.group(2).split()) <= MAX_HEADER_WORDS:
            body = "\n".join(current_lines).strip()
            if body:
                sections.append((current_title, body))
            current_title = match.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)

    body = "\n".join(current_lines).strip()
    if body:
        sections.append((current_title, body))
    return sections


def _merge_splits(pieces: list[str], separator: str, chunk_size: int, overlap: int) -> list[str]:
    merged: list[str] = []
    current: list[str] = []
    current_tokens = 0
    sep_tokens = _token_count(separator) if separator else 0

    for piece in pieces:
        piece_tokens = _token_count(piece)
        added = piece_tokens + (sep_tokens if current else 0)
        if current and current_tokens + added > chunk_size:
            merged.append(separator.join(current))
            while current and current_tokens > overlap:
                dropped = current.pop(0)
                current_tokens -= _token_count(dropped)
                if current:
                    current_tokens -= sep_tokens
        current.append(piece)
        current_tokens += piece_tokens + (sep_tokens if len(current) > 1 else 0)

    if current:
        merged.append(separator.join(current))
    return merged


def _recursive_split(text: str, chunk_size: int, overlap: int, separators: list[str]) -> list[str]:
    """Recursive character split: try each separator in turn, falling back to a
    finer one only for pieces still too large, then merge with token overlap."""
    if _token_count(text) <= chunk_size:
        return [text] if text.strip() else []

    separator, *remaining = separators
    pieces = [p for p in (text.split(separator) if separator else list(text)) if p != ""]

    small_pieces: list[str] = []
    result: list[str] = []
    for piece in pieces:
        if _token_count(piece) <= chunk_size:
            small_pieces.append(piece)
            continue
        if small_pieces:
            result.extend(_merge_splits(small_pieces, separator, chunk_size, overlap))
            small_pieces = []
        if remaining:
            result.extend(_recursive_split(piece, chunk_size, overlap, remaining))
        else:
            result.append(piece)

    if small_pieces:
        result.extend(_merge_splits(small_pieces, separator, chunk_size, overlap))
    return result


def _build_filing_chunks(row: dict) -> tuple[list[dict], int]:
    ticker, company, form = row["ticker"], row["company"], row["form"]
    filing_date = row["filing_date"]
    period = _reporting_period(row["primary_document"], filing_date)

    stem = Path(row["local_path"]).stem
    text_path = TEXT_DIR / f"{stem}.md"
    if not text_path.exists():
        print(f"  [skip] no parsed text for {stem} (run `python -m src.text_processing` first)")
        return [], 0

    markdown_text = text_path.read_text(encoding="utf-8")
    sections = _split_by_headers(markdown_text)

    chunks: list[dict] = []
    filtered = 0
    index = 0

    for title, body in sections:
        full_text = f"{title}\n\n{body}".strip() if title else body

        if _token_count(full_text) > MAX_SECTION_TOKENS:
            pieces = _recursive_split(body, CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS, SPLIT_SEPARATORS)
            texts = [f"{title}\n\n{piece}".strip() if title else piece for piece in pieces]
        else:
            texts = [full_text]

        for text in texts:
            token_count = _token_count(text)
            if token_count < MIN_CHUNK_TOKENS:
                filtered += 1
                continue
            index += 1
            chunks.append(
                {
                    "chunk_id": f"{ticker}_{form}_{period}_{index:04d}",
                    "text": text,
                    "metadata": {
                        "ticker": ticker,
                        "company": company,
                        "filing_type": form,
                        "filing_date": filing_date,
                        "reporting_period": period,
                        "section": title,
                        "risk_theme": _tag_risk_theme(text),
                        "source_type": "regulatory_filing",
                        "authority": "formal_disclosure",
                        "embedding_model": "",
                        "token_count": token_count,
                        "source_file": row["local_path"],
                        "enriched": False,
                    },
                }
            )

    return chunks, filtered


def _build_statement_chunks(path: str = STATEMENTS_CSV_PATH) -> list[dict]:
    chunks = []
    counters: dict[str, int] = {}

    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            ticker = row["ticker"]
            counters[ticker] = counters.get(ticker, 0) + 1
            index = counters[ticker]
            text = row["statement"]

            chunks.append(
                {
                    "chunk_id": f"{ticker}_STMT_{row['statement_date']}_{index:03d}",
                    "text": text,
                    "metadata": {
                        "ticker": ticker,
                        "company": row["company"],
                        "statement_date": row["statement_date"],
                        "speaker": row["speaker"],
                        "source_type": "management_statement",
                        "risk_theme": row["risk_theme"],
                        "authority": "management_commentary",
                        "embedding_model": "",
                        "token_count": _token_count(text),
                        "enriched": False,
                    },
                }
            )

    return chunks


def _load_enrichment_cache() -> dict[str, str]:
    if not ENRICHMENT_CACHE_PATH.exists():
        return {}
    cache = {}
    with open(ENRICHMENT_CACHE_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                cache[record["chunk_id"]] = record["prefix"]
    return cache


def _append_to_enrichment_cache(chunk_id: str, prefix: str) -> None:
    ENRICHMENT_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ENRICHMENT_CACHE_PATH, "a") as f:
        f.write(json.dumps({"chunk_id": chunk_id, "prefix": prefix}) + "\n")


def _enrichment_user_message(chunk: dict) -> str:
    metadata = chunk["metadata"]
    company = metadata.get("company", "")
    filing_type = metadata.get("filing_type", metadata.get("source_type", ""))
    period = metadata.get("reporting_period", metadata.get("statement_date", ""))
    section = metadata.get("section", "")
    excerpt = chunk["metadata"].get("original_text", chunk["text"])[:600]
    return f"Company: {company}, Filing: {filing_type} {period}, Section: {section}\n\n{excerpt}"


def enrich_chunks(chunks: list[dict]) -> list[dict]:
    cache = _load_enrichment_cache()
    llm_client = LLMClient()

    pending = [chunk for chunk in chunks if not chunk["metadata"].get("enriched")]
    print(f"Enriching {len(pending)} chunks ({len(cache)} already cached)...")

    for batch_start in range(0, len(pending), ENRICHMENT_BATCH_SIZE):
        batch = pending[batch_start : batch_start + ENRICHMENT_BATCH_SIZE]

        for chunk in batch:
            chunk_id = chunk["chunk_id"]
            original_text = chunk["text"]

            if chunk_id in cache:
                prefix = cache[chunk_id]
            else:
                messages = [
                    {"role": "system", "content": ENRICHMENT_SYSTEM_PROMPT},
                    {"role": "user", "content": _enrichment_user_message(chunk)},
                ]
                prefix = llm_client.chat(messages).strip()
                _append_to_enrichment_cache(chunk_id, prefix)
                cache[chunk_id] = prefix

            chunk["metadata"]["original_text"] = original_text
            chunk["text"] = f"{prefix}\n\n{original_text}"
            chunk["metadata"]["enriched"] = True

        done = min(batch_start + ENRICHMENT_BATCH_SIZE, len(pending))
        print(f"  enriched {done}/{len(pending)}")
        time.sleep(ENRICHMENT_BATCH_SLEEP_SECONDS)

    return chunks


def chunk_all_filings(enrich: bool = False) -> list[dict]:
    with open(FILING_INDEX_PATH, newline="") as f:
        filing_rows = list(csv.DictReader(f))

    all_chunks: list[dict] = []
    noise_filtered = 0

    for row in filing_rows:
        chunks, filtered = _build_filing_chunks(row)
        all_chunks.extend(chunks)
        noise_filtered += filtered
        print(f"{row['ticker']} {row['form']} {row['filing_date']}: {len(chunks)} chunks ({filtered} filtered as noise)")

    statement_chunks = _build_statement_chunks()
    all_chunks.extend(statement_chunks)
    print(f"Management statements: {len(statement_chunks)} chunks")

    if enrich:
        n = len(all_chunks)
        print(f"\nContext enrichment enabled. This will make ~{n} Mistral API calls.")
        print(f"Estimated time: {n // 10 * 1}s minimum. Results cached to {ENRICHMENT_CACHE_PATH}.")
        all_chunks = enrich_chunks(all_chunks)

    CHUNKS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_OUTPUT_PATH, "w") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + "\n")

    by_ticker = Counter(chunk["metadata"]["ticker"] for chunk in all_chunks)
    by_theme = Counter(chunk["metadata"]["risk_theme"] for chunk in all_chunks)

    print(f"\nTotal chunks: {len(all_chunks)} (filtered as noise: {noise_filtered})")
    print("By ticker:")
    for ticker, count in sorted(by_ticker.items()):
        print(f"  {ticker}: {count}")
    print("By risk_theme:")
    for theme, count in sorted(by_theme.items()):
        print(f"  {theme}: {count}")
    print(f"\nSaved to {CHUNKS_OUTPUT_PATH}")

    return all_chunks


def chunk_filing(text: str, metadata: dict) -> list[dict]:
    raise NotImplementedError


def _main() -> None:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--enrich", action="store_true")
    args = arg_parser.parse_args()
    chunk_all_filings(enrich=args.enrich)


if __name__ == "__main__":
    _main()
