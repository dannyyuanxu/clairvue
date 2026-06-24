import json

from mistralai.client import Mistral

from src.config import settings


class LLMClient:
    def __init__(self) -> None:
        self.client = Mistral(api_key=settings.MISTRAL_API_KEY)
        self.embedding_model = settings.EMBEDDING_MODEL
        self.llm_model = settings.LLM_MODEL

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.embedding_model, inputs=texts)
        return [item.embedding for item in response.data]

    def chat(self, messages: list[dict], temperature: float = 0.0) -> str:
        response = self.client.chat.complete(
            model=self.llm_model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content

    def chat_json(self, messages: list[dict]) -> dict:
        response = self.client.chat.complete(
            model=self.llm_model,
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
