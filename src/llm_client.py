import json
import time

import httpx
from mistralai.client import Mistral
from mistralai.client.errors.sdkerror import SDKError

from src.config import settings

MAX_RETRIES = 5
RETRY_BACKOFF_SECONDS = 2.0
CHAT_RATE_LIMIT_SECONDS = 1.0


class LLMClient:
    def __init__(self) -> None:
        self.client = Mistral(api_key=settings.MISTRAL_API_KEY)
        self.embedding_model = settings.EMBEDDING_MODEL
        self.llm_model = settings.LLM_MODEL

    def _complete(self, **kwargs):
        """Throttles to ~1 request/sec proactively, then retries on rate limits
        (429) and transient connection drops with exponential backoff as a
        fallback -- sequential claim-by-claim workflows can issue many chat
        calls in quick succession and hit Mistral's rate limit."""
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat.complete(**kwargs)
                time.sleep(CHAT_RATE_LIMIT_SECONDS)
                return response
            except (SDKError, httpx.RemoteProtocolError, httpx.ConnectError) as error:
                is_retryable = isinstance(error, (httpx.RemoteProtocolError, httpx.ConnectError)) or (
                    isinstance(error, SDKError) and error.raw_response.status_code in (429, 500, 502, 503)
                )
                if not is_retryable or attempt == MAX_RETRIES - 1:
                    raise
                sleep_seconds = RETRY_BACKOFF_SECONDS * (2**attempt)
                print(f"  (retrying after {type(error).__name__}, sleeping {sleep_seconds:.0f}s)")
                time.sleep(sleep_seconds)

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.embedding_model, inputs=texts)
        return [item.embedding for item in response.data]

    def chat(self, messages: list[dict], temperature: float = 0.0) -> str:
        response = self._complete(model=self.llm_model, messages=messages, temperature=temperature)
        return response.choices[0].message.content

    def chat_json(self, messages: list[dict]) -> dict:
        response = self._complete(
            model=self.llm_model,
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
