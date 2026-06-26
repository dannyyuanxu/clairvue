# Clairvue

A Mistral-powered RAG system that validates bank management statements (CEO/CFO claims about
consumer credit quality) against SEC filings, XBRL metrics, and peer-bank evidence.

Given a claim such as *"Consumer credit remains resilient, and losses are normalizing in line
with expectations,"* Clairvue decomposes it into atomic, independently verifiable assertions,
retrieves grounded evidence for each, and returns a structured verdict —
**supported / partially supported / contradicted / insufficient evidence** — with citations
back to actual filings and pre-framed quantitative metrics.

See [CLAUDE.md](./CLAUDE.md) for architecture, scope, and conventions.

## Prerequisites

- Python 3.10+
- A [Mistral API key](https://console.mistral.ai/) (paid account recommended — the pipeline
  makes many sequential API calls)
- SEC EDGAR User-Agent string (required for data ingestion; see `.env.example`)

## Setup

```bash
python3 -m venv clairvue
source clairvue/bin/activate
pip install -r requirements.txt

cp .env.example .env   # fill in MISTRAL_API_KEY and SEC_USER_AGENT
```

## Data

Two small hand-curated files are pre-committed and ready to use:
- `data/processed/metrics.csv` — XBRL financial metrics (provisions, NCO rates, allowances),
  backfilled to 2019 for a pre-pandemic baseline
- `data/raw/management_statements/statements.csv` — hand-collected CEO/CFO quotes

Everything else under `data/` (filing HTML, parsed text, chunks, embeddings, the ChromaDB index)
must be built locally by running the pipeline below.

## Pipeline

```bash
python scripts/download_filings.py   # SEC EDGAR → data/raw/sec_filings/  (~15 filings)
python scripts/build_index.py        # parse → chunk → embed → ChromaDB
```

Each stage is idempotent — re-running after a partial failure picks up where it left off.

## Running locally

```bash
python -m src.demo   # runs all 3 demo scenarios and writes outputs/sample_answers.json
```

Or drive `answer_claim()` / `compare_peers()` interactively from `notebook.ipynb`.

## Google Colab

Open `Clairvue_colab_demo.ipynb` in Colab. The notebook assumes the pipeline has already been
built once and its outputs uploaded to Google Drive at `MyDrive/clairvue_data/`. See the
notebook's setup cell for the full walkthrough.

1. Add `MISTRAL_API_KEY` to Colab Secrets (the key icon in the left sidebar)
2. Run the setup cell — it clones this repo, installs dependencies, and mounts Drive
3. Run the remaining cells in order for the 5 demo scenarios

## Project layout

```
src/
  config.py           settings (pydantic-settings, env vars)
  llm_client.py       single gateway for all Mistral API calls
  data_ingestion.py   SEC EDGAR filing download
  text_processing.py  HTML → clean markdown (sec-parser + BeautifulSoup fallback)
  chunking.py         markdown → section-aware chunks with metadata
  embeddings.py       chunks → vectors (via LLMClient.embed)
  vector_store.py     ChromaDB wrapper (build, query, metadata filter)
  retrieval.py        claim → filtered evidence + contradiction expansion
  metrics.py          XBRL metrics pipeline (NCO rate, provisions, allowance)
  prompts.py          all prompt templates and metric pre-framing
  generation.py       answer_claim(), compare_peers() — the two top-level workflows
  evaluation.py       batch eval against data/processed/eval_questions.csv
  demo.py             Markdown renderers and canned demo scenarios
scripts/
  download_filings.py
  build_index.py
```

## Technical architecture

```
Data Sources
  ├── SEC EDGAR (15 filings: 10-K and 10-Q, Q4 2022–Q4 2023)
  ├── Management statements (hand-curated CSV from earnings releases)
  └── XBRL metrics (SEC companyfacts API, backfilled to 2019)
        ↓
Parse + Cache Layer
  ├── sec-parser extracts clean text with section/heading structure
  ├── Falls back to BeautifulSoup + markdownify if sec-parser raises
  ├── Cleaned markdown cached to data/processed/text/ (gitignored)
  └── Skips re-parse if cached file exists (idempotent)
        ↓
Chunking + Metadata Layer
  ├── 800-token chunks with 100-token overlap
  ├── Section-aware splitting on markdown headers from sec-parser
  ├── Noise chunk filtering (< 50 tokens discarded)
  ├── Risk theme tagging by keyword rules (consumer_credit, commercial_real_estate)
  └── Full metadata per chunk: ticker, period, filing_type, source_type, authority, risk_theme
        ↓
Embedding Layer
  ├── Mistral mistral-embed (1024 dimensions)
  ├── Embedding model name stored in vector metadata for staleness detection
  ├── Embeddings cached as embeddings.npy (gitignored)
  └── vector_metadata.jsonl cached alongside
        ↓
Vector Store
  ├── ChromaDB with persist_directory
  ├── Metadata filtering on ticker, risk_theme, filing_type, source_type
  ├── Local for development; Drive-mounted for Colab demo
  └── 9,750 chunks across 15 filings + management statements
        ↓
Retrieval Layer
  ├── Metadata-filtered semantic search via ChromaDB
  ├── Contradiction-query expansion for positive-language claims
  └── Per-bank isolated retrieval for peer comparison
        ↓
Reasoning Layer
  ├── Claim decomposition into 2–5 atomic sub-claims (1 LLM call)
  ├── Per-claim isolated retrieval + assessment (1 LLM call each)
  ├── XBRL metrics pre-framed in Python: rate, direction, 2019 baseline, peer cross-section
  └── Verdict aggregation: supported / partially supported / contradicted / insufficient evidence
        ↓
Output Layer
  ├── Structured JSON with verdict, rationale, and evidence buckets
  ├── Peer comparison with per-bank evidence and direction indicators
  └── Missing information and analyst follow-up questions (verbose mode)
```
