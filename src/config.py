from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    MISTRAL_API_KEY: str = ""
    EMBEDDING_MODEL: str = "mistral-embed"
    LLM_MODEL: str = "mistral-small-latest"
    CHROMA_PERSIST_DIR: str = "./data/processed/chroma_db"

    # Only needed at ingestion time — defaulted so this module can be
    # imported (e.g. for retrieval/generation) without it set.
    SEC_USER_AGENT: str = ""


settings = Settings()

if not settings.MISTRAL_API_KEY:
    raise RuntimeError("MISTRAL_API_KEY not set. Add it to .env or Colab Secrets.")
