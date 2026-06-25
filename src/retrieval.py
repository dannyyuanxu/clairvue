"""Primary interface for all evidence retrieval in the system."""

from src.llm_client import LLMClient
from src.vector_store import combine_filters, filter_by_theme, filter_by_ticker, load_collection
from src.vector_store import query as chroma_query

CONTRADICTION_KEYWORD_RULES = [
    (
        ["resilient", "stable", "strong", "improving"],
        [
            "rising delinquencies",
            "higher charge-offs",
            "credit deterioration",
            "increased provisions",
            "worse than expected credit losses",
        ],
    ),
    (
        ["normalizing", "in line with", "as expected"],
        [
            "accelerating losses",
            "above guidance provisions",
            "unexpected credit deterioration",
        ],
    ),
    (
        ["lower", "declining", "improving"],
        [
            "rising losses",
            "increasing charge-offs",
            "worsening credit quality",
        ],
    ),
]


def _build_where_filter(
    tickers: list[str] | None = None,
    risk_theme: str | None = None,
    source_type: str | None = None,
    filing_type: str | None = None,
) -> dict:
    """Builds the Chroma `where` clause. A management statement can't validate
    itself, so source_type == "management_statement" is excluded by default --
    unless source_type explicitly requests it, e.g. a caller deliberately
    retrieving only management statements."""
    filters = []
    if tickers:
        filters.append(filter_by_ticker(tickers))
    if risk_theme:
        filters.append(filter_by_theme(risk_theme))
    if source_type:
        filters.append({"source_type": source_type})
    if filing_type:
        filters.append({"filing_type": filing_type})

    if source_type != "management_statement":
        filters.append({"source_type": {"$ne": "management_statement"}})

    if len(filters) == 1:
        return filters[0]
    return combine_filters(*filters)


def _contradiction_queries_for_claim(claim: str) -> list[str]:
    """Maps positive-sentiment keywords in a claim to counter-evidence search queries,
    so retrieval doesn't only surface evidence that agrees with the claim's own phrasing."""
    lowered = claim.lower()
    contradiction_queries: list[str] = []
    for keywords, queries in CONTRADICTION_KEYWORD_RULES:
        if any(keyword in lowered for keyword in keywords):
            contradiction_queries = queries
    return contradiction_queries


class EvidenceRetriever:
    """Primary retrieval interface: embeds a query, applies metadata filters, and
    queries the ChromaDB collection. Owns one Chroma collection handle + one LLMClient."""

    def __init__(self) -> None:
        self.collection = load_collection()
        self.llm_client = LLMClient()

    def retrieve(
        self,
        query: str,
        n_results: int = 8,
        tickers: list[str] | None = None,
        risk_theme: str | None = None,
        source_type: str | None = None,
        filing_type: str | None = None,
    ) -> list[dict]:
        """Embeds `query` and runs a metadata-filtered similarity search."""
        query_embedding = self.llm_client.embed([query])[0]
        where = _build_where_filter(tickers, risk_theme, source_type, filing_type)
        return chroma_query(self.collection, query_embedding, n_results=n_results, where=where)

    def retrieve_with_contradiction(
        self,
        claim: str,
        n_results: int = 6,
        tickers: list[str] | None = None,
        risk_theme: str | None = "consumer_credit",
    ) -> list[dict]:
        """Retrieves evidence for `claim`, plus counter-evidence from contradiction-
        expansion queries (see `_contradiction_queries_for_claim`), deduped by chunk_id
        and merged into one ranked list."""
        all_results = self.retrieve(
            claim, n_results=n_results, tickers=tickers, risk_theme=risk_theme
        )

        for contradiction_query in _contradiction_queries_for_claim(claim):
            all_results += self.retrieve(
                contradiction_query, n_results=4, tickers=tickers, risk_theme=risk_theme
            )

        deduped: dict[str, dict] = {}
        for result in all_results:
            chunk_id = result["chunk_id"]
            if chunk_id not in deduped or result["score"] > deduped[chunk_id]["score"]:
                deduped[chunk_id] = result

        merged = sorted(deduped.values(), key=lambda result: result["score"], reverse=True)
        return merged[: n_results * 2]

    def retrieve_with_peer_context(
        self,
        claim: str,
        primary_ticker: str,
        risk_theme: str | None = None,
        n_primary: int = 6,
        n_peer: int = 3,
    ) -> dict[str, list[dict]]:
        """Two-pass retrieval for a bank-specific claim: validates primarily against
        the bank's own formal filings (with contradiction expansion), then adds
        lighter peer-bank context separately -- one retrieve() call per peer, no
        contradiction pass -- so peer evidence doesn't dilute or get conflated
        with the primary bank's own evidence during assessment."""
        peer_tickers = [ticker for ticker in ["JPM", "BAC", "C"] if ticker != primary_ticker]

        primary_chunks = self.retrieve_with_contradiction(
            claim,
            n_results=n_primary,
            tickers=[primary_ticker],
            risk_theme=risk_theme,
        )

        peer_chunks = []
        for peer in peer_tickers:
            peer_chunks.extend(
                self.retrieve(claim, n_results=n_peer, tickers=[peer], risk_theme=risk_theme)
            )

        return {"primary": primary_chunks, "peer": peer_chunks}

    def retrieve_per_bank(
        self,
        query: str,
        tickers: list[str],
        n_results_per_bank: int = 5,
        risk_theme: str | None = "consumer_credit",
    ) -> dict[str, list[dict]]:
        """Runs the same query once per ticker, each filtered to that bank only --
        used by peer comparison so every bank gets an equal-sized evidence set."""
        return {
            ticker: self.retrieve(
                query, n_results=n_results_per_bank, tickers=[ticker], risk_theme=risk_theme
            )
            for ticker in tickers
        }


def format_evidence_for_prompt(chunks: list[dict], label: str | None = None) -> str:
    """Formats retrieved chunks as numbered, citation-ready blocks for prompt
    inclusion. An optional label prefixes the block with a "--- label ---"
    header, used to keep primary vs. peer-bank evidence visually distinct."""
    header = f"--- {label} ---\n" if label else ""
    blocks = []
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk["metadata"]
        company = metadata.get("company", "")
        source_type = metadata.get("source_type", "")
        filing_type = metadata.get("filing_type", "")
        period = metadata.get("reporting_period", metadata.get("statement_date", ""))
        section = metadata.get("section", "")
        authority = metadata.get("authority", "")
        excerpt = chunk["text"][:500]

        blocks.append(
            f"[{index}] Company: {company} | Type: {source_type} | Filing: {filing_type} {period}\n"
            f"    Section: {section} | Authority: {authority}\n"
            f'    "{excerpt}..."'
        )

    return header + "\n\n".join(blocks)


def _main() -> None:
    import argparse

    arg_parser = argparse.ArgumentParser(description="Smoke-test retrieval against the live index.")
    arg_parser.add_argument("--query", default="consumer credit losses normalizing")
    arg_parser.add_argument("--n-results", type=int, default=5)
    args = arg_parser.parse_args()

    retriever = EvidenceRetriever()
    results = retriever.retrieve(args.query, n_results=args.n_results)
    for res in results:
        print(
            f"Score: {res['score']:.3f} | {res['metadata']['ticker']} "
            f"| {res['metadata'].get('filing_type', res['metadata'].get('source_type'))} "
            f"{res['metadata'].get('reporting_period', res['metadata'].get('statement_date'))}"
        )
        print(f"  {res['text'][:120]}...")
        print()


if __name__ == "__main__":
    _main()
