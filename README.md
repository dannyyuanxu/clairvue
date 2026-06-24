# Clairvue

A Mistral-powered RAG system that validates bank management statements (CEO/CFO claims about
consumer credit quality) against SEC filings, XBRL metrics, and peer-bank evidence.

See [CLAUDE.md](./CLAUDE.md) for architecture, scope, and conventions.

## Setup

```bash
python3 -m venv clairvue
source clairvue/bin/activate
pip install -r requirements-dev.txt   # or requirements.txt for runtime-only

cp .env.example .env   # fill in MISTRAL_API_KEY and SEC_USER_AGENT
```

## Pipeline

```bash
python scripts/download_filings.py   # SEC EDGAR -> data/raw/sec_filings/
python scripts/build_index.py        # parse -> chunk -> embed -> ChromaDB
```

Then open `notebook.ipynb` (locally or in Google Colab) to run the demo:
`answer_claim(...)` and `compare_peers(...)`.

## Project layout

```
src/
  config.py          settings (pydantic-settings)
  llm_client.py       single entry point for all Mistral calls
  data_ingestion.py   SEC EDGAR download
  text_processing.py  HTML -> clean text
  chunking.py         text -> chunks
  embeddings.py       chunks -> vectors (via LLMClient)
  vector_store.py     ChromaDB wrapper
  retrieval.py        query -> relevant chunks
  metrics.py          XBRL-derived metrics (provisions, NCOs)
  prompts.py          prompt templates
  generation.py       answer_claim(), compare_peers()
  evaluation.py       scoring against eval_questions.csv
  demo.py             notebook entry point
scripts/
  download_filings.py
  build_index.py
```
