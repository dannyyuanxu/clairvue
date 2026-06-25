# Clairvue

## Build status

All modules implemented and verified against the live SEC EDGAR API, live Mistral API, and a
real ChromaDB index (9,750 chunks across 15 filings + management statements).

| Module | Role |
|---|---|
| `src/config.py` | Settings (env vars); raises at import if `MISTRAL_API_KEY` is unset. |
| `src/llm_client.py` | Sole gateway to the `mistralai` SDK — `embed`/`chat`/`chat_json`, with proactive rate-limit throttling and retry-with-backoff. |
| `src/data_ingestion.py` | SEC EDGAR submissions/XBRL API client + raw filing download. |
| `src/text_processing.py` | Raw filing HTML → cleaned text (stage 1 of the cache pipeline). |
| `src/chunking.py` | Section-aware chunking, risk-theme tagging, optional LLM context enrichment. |
| `src/embeddings.py` | Batch embedding via `LLMClient`, with cache validation against the chunk count and embedding model. |
| `src/vector_store.py` | ChromaDB wrapper — index build, metadata-filtered query. |
| `src/metrics.py` | XBRL metrics pipeline (provision, allowance, net charge-offs, total loans, + computed annualized net-charge-off **rate**) with per-bank concept-coverage resolution. Quantitative series are backfilled to 2019 for a pre-pandemic baseline; filing *text* stays Q4'22–Q4'23. |
| `src/retrieval.py` | `EvidenceRetriever` — metadata-filtered retrieval + contradiction-query expansion. |
| `src/prompts.py` | All system/user prompt templates (no LLM calls). `format_metrics_for_prompt` pre-computes rate/direction/baseline/peer framing so the model interprets rather than calculates. |
| `src/generation.py` | `answer_claim()` and `compare_peers()` — the two top-level workflows. |
| `src/demo.py` | ANSI pretty-printers + Markdown renderers (`display_claim_assessment`/`display_peer_comparison`, used in the notebook) + the 3 interview demo scenarios; writes `outputs/sample_answers.json`. |
| `src/evaluation.py` | Runs `data/processed/eval_questions.csv` through the live pipeline. |
| `scripts/download_filings.py` | CLI: fetch the 15 target filings into `data/raw/sec_filings/`. |
| `scripts/build_index.py` | CLI: runs the full pipeline end-to-end, resumable with `--from-step`. |

A RAG system that validates bank management statements against formal SEC disclosures,
quantitative financial metrics, and peer-bank evidence. Input is a CEO/CFO claim (e.g.
"Consumer credit remains resilient, and losses are normalizing in line with expectations.");
output is a structured assessment — supported / partially supported / contradicted /
insufficient evidence — with citations back to the actual filings and metrics.

## Scope

- **Companies:** JPM (JPMorgan Chase), BAC (Bank of America), C (Citigroup)
- **Periods:** Q4 2022 through Q4 2023 — 5 reporting quarters per bank (10-K for Q4s, 10-Q
  otherwise), 15 filings total
- **Primary risk theme:** consumer credit quality (charge-offs, delinquencies, provisions,
  "normalization" language)

## Stack

- **Mistral, exclusively** — both embeddings (`mistral-embed`) and generation
  (`mistral-small-latest` by default). No other model provider anywhere in the codebase.
- **ChromaDB** for the vector store.
- **sec-parser** for filing structure (dev-only dependency, used in the ingestion scripts).
- **Pure Python, no LangChain.** The interview format requires line-by-line explainability of
  the retrieval and assessment pipeline — a framework would hide exactly the steps that need
  to be walked through live.

## Model abstraction

All LLM and embedding calls go through `src/llm_client.py:LLMClient`. Nothing else in the
codebase imports `mistralai` directly. This is what makes a future model swap (or
provider swap) a one-file change instead of a codebase-wide find-and-replace.

## Data caching architecture (two-stage)

1. **Parse (stage 1):** download raw filing HTML once (`data/raw/sec_filings/`), clean it to
   plain text, cache to `data/processed/text/`. Skip on subsequent runs if the cached file
   already exists — filing HTML doesn't change.
2. **Chunk + embed + index (stage 2):** load from the cached text, chunk, embed via
   `LLMClient.embed()`, write `data/processed/chunks.jsonl` +
   `data/processed/embeddings.npy` + `data/processed/vector_metadata.jsonl`, then load into
   ChromaDB (`data/processed/chroma_db/`).

Splitting these stages means re-tuning chunk size or overlap doesn't require re-downloading
or re-parsing filings, and switching the embedding model doesn't require re-parsing either —
only re-running stage 2.

Every vector's metadata includes `embedding_model`. If the embedding model changes, that field
tells you exactly which chunks are now stale and need re-embedding — Chroma itself has no way
to detect a model mismatch on its own.

## Data pipeline

Full order, each stage gated on the previous stage's cached output:

```
download_filings → text_processing → chunking → embeddings → vector_store → demo
```

`scripts/build_index.py` runs stages 2–5 (text_processing through vector_store) as a single
resumable CLI; `scripts/download_filings.py` covers stage 1. See "Running the demo" below for
exact commands. Every stage is idempotent — re-running it when its cached output already exists
is a no-op (or a fast cache-validity check), so re-running the whole pipeline after a partial
failure is always safe.

## Gitignore strategy

`data/raw/` and `data/processed/` are gitignored by default — they're either large, derived,
or both. Three exceptions are explicitly carved out because they are small and hand-curated
(see `.gitignore` for the carve-out pattern, which un-ignores the specific directory before
un-ignoring the file inside it):

- `data/raw/management_statements/statements.csv` — hand-collected CEO/CFO quotes
- `data/processed/metrics.csv` — hand-curated XBRL metric snapshot
- `data/processed/eval_questions.csv` — hand-crafted evaluation set

Everything else under those two trees (filings, parsed text, chunks, embeddings, the Chroma
index) is derivable and regenerated by `scripts/download_filings.py` and
`scripts/build_index.py`.

`.env` is gitignored (real secrets); `.env.example` is committed (template, no secrets).
`outputs/sample_answers.json` is committed — it's the interview fallback if the live API is
slow or unavailable, so it needs to exist in the repo, not just locally.

## Running the demo

**Locally**, with the pipeline already built (see "Data pipeline" above) and `.env` populated
from `.env.example`:

```bash
python -m src.demo
```

This runs all 3 interview scenarios, prints them with the pretty-printers in `src/demo.py`, and
(re)writes `outputs/sample_answers.json`.

**In Google Colab** (`notebook.ipynb`):

1. Add `MISTRAL_API_KEY` to Colab Secrets (the key icon in the left sidebar) — the notebook
   reads it via `userdata.get(...)`, never hardcoded.
2. Run the setup cell: clones this repo, installs `requirements.txt`, mounts Google Drive, and
   points `CHROMA_PERSIST_DIR` at `MyDrive/clairvue_data/chroma_db`.
3. The notebook assumes the pipeline has already been built once and its output (chunks,
   embeddings, Chroma index, `metrics.csv`) uploaded to that Drive folder — it does not rebuild
   the index from scratch in Colab. If it hasn't been built yet, run `scripts/build_index.py`
   locally first and upload `data/processed/` to Drive.
4. Run the remaining cells in order for the 3 demo scenarios + architecture notes.

## Known limitations

- **XBRL concept name inconsistencies across banks.** JPM, BAC, and C adopted CECL on different
  timelines, which renumbered the relevant XBRL concept tags — no single concept name works
  across all 3 banks for provision or net-charge-offs; `src/metrics.py` resolves this by trying
  multiple candidate concepts per metric and keeping whichever yields the most in-window data.
  One concept (`...AfterAllowanceForCreditLoss`) was found to be internally inconsistent for JPM
  specifically (its "after allowance" value exceeds its "before allowance" value, which is
  impossible for a true net figure) — `total_loans` is computed as gross minus allowance instead
  of trusting that tag directly (see `_build_total_loans_rows`).
- **sec-parser fallback files.** When `sec-parser` can't structure a filing's HTML (rare but not
  zero), `src/text_processing.py` falls back to a plain BeautifulSoup extraction
  (`_parse_with_bs4`), which loses section/heading structure. Affected filings still get chunked
  and indexed, just with weaker section metadata.
- **No hybrid search (dense embeddings only).** Deliberate, not an oversight — these filings
  don't have the citation-dense cross-referencing structure that makes BM25/lexical hybrid
  search earn its complexity. Metadata filtering (ticker, risk_theme, filing_type) plus the
  separate `metrics.csv` numeric side-channel do the precision work hybrid search would
  otherwise provide. Would revisit if the corpus grew to include more citation-heavy documents.
- **Contradiction retrieval is heuristic, not guaranteed.** `src/retrieval.py`'s
  contradiction-query expansion is keyword-rule-based (`CONTRADICTION_KEYWORD_RULES`) — it
  catches known phrasings ("resilient," "normalizing," etc.) but won't generalize to claim
  phrasing outside those rules without becoming its own LLM call.

## Running

```bash
python scripts/download_filings.py
python scripts/build_index.py
```

Then drive `answer_claim()` / `compare_peers()` from `notebook.ipynb`, locally or in Colab, or
run `python -m src.demo` for the canned interview scenarios.
