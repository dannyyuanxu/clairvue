"""Sole gateway to the Mistral SDK: embedding + chat/chat_json calls, with proactive
rate-limit throttling and retry-with-backoff. Nothing else in the codebase imports
`mistralai` directly, so a model or provider swap is a one-file change."""

import json
import time

import httpx
from mistralai.client import Mistral
from mistralai.client.errors.sdkerror import SDKError

from src.config import settings

MAX_RETRIES = 5
RETRY_BACKOFF_SECONDS = 2.0
CHAT_RATE_LIMIT_SECONDS = 0.5


class LLMClient:
    """Sole gateway to the Mistral SDK -- nothing else in the codebase imports
    `mistralai` directly, so a model or provider swap is a one-file change."""

    def __init__(self, model: str | None = None) -> None:
        """`model` overrides the chat model for this client instance (defaults to
        settings.LLM_MODEL). Used to run a cheaper model for a specific call-type --
        e.g. enrichment passes settings.ENRICHMENT_MODEL -- without touching call sites
        or affecting the default demo/assessment model."""
        self.client = Mistral(api_key=settings.MISTRAL_API_KEY)
        self.embedding_model = settings.EMBEDDING_MODEL
        self.llm_model = model or settings.LLM_MODEL

    def _with_retry(self, fn):
        """Throttles to ~1 request/sec proactively, then retries on rate limits
        (429) and transient connection drops with exponential backoff as a
        fallback -- sequential claim-by-claim workflows (including two-pass
        peer-context retrieval, which embeds several queries per claim) can
        issue many API calls in quick succession and hit Mistral's rate or
        capacity limits. Wraps both chat and embedding calls -- both can hit
        the same class of transient errors."""
        for attempt in range(MAX_RETRIES):
            try:
                result = fn()
                time.sleep(CHAT_RATE_LIMIT_SECONDS)
                return result
            except (SDKError, httpx.RemoteProtocolError, httpx.ConnectError) as error:
                is_connection_error = isinstance(
                    error, (httpx.RemoteProtocolError, httpx.ConnectError)
                )
                is_retryable_status = isinstance(
                    error, SDKError
                ) and error.raw_response.status_code in (429, 500, 502, 503)
                is_retryable = is_connection_error or is_retryable_status
                if not is_retryable or attempt == MAX_RETRIES - 1:
                    raise
                # Honour Retry-After if the server sent one (common on 429); fall back to
                # exponential backoff only when the header is absent or unparseable.
                retry_after = None
                if isinstance(error, SDKError):
                    status = error.raw_response.status_code
                    try:
                        retry_after = float(error.raw_response.headers.get("Retry-After", ""))
                    except (ValueError, TypeError):
                        retry_after = None
                else:
                    status = 0
                sleep_seconds = retry_after if retry_after else RETRY_BACKOFF_SECONDS * (2**attempt)
                print(f"  (retrying after {type(error).__name__} [{status}], sleeping {sleep_seconds:.0f}s)")
                time.sleep(sleep_seconds)
        # Unreachable: the final attempt either returns or re-raises above. This satisfies
        # static analysis that every path has an explicit, consistent return.
        raise RuntimeError("retry loop exited without returning or raising")

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embeds a batch of texts with the configured embedding model."""
        response = self._with_retry(
            lambda: self.client.embeddings.create(model=self.embedding_model, inputs=texts)
        )
        return [item.embedding for item in response.data]

    def chat(self, messages: list[dict], temperature: float = 0.0) -> str:
        """Free-text chat completion."""
        response = self._with_retry(
            lambda: self.client.chat.complete(
                model=self.llm_model, messages=messages, temperature=temperature
            )
        )
        return response.choices[0].message.content

    def chat_json(self, messages: list[dict]) -> dict:
        """Chat completion constrained to return valid JSON, pre-parsed into a dict."""
        response = self._with_retry(
            lambda: self.client.chat.complete(
                model=self.llm_model,
                messages=messages,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
        )
        return json.loads(response.choices[0].message.content)
