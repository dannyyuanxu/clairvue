"""Environment-backed settings, loaded once at import time."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All env vars the system reads, with defaults for everything except the API key."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    MISTRAL_API_KEY: str = ""
    EMBEDDING_MODEL: str = "mistral-embed"
    LLM_MODEL: str = "mistral-small-latest"
    # Model used only for chunk context enrichment (a high-volume, low-stakes summarization
    # step run during the local index build). Defaulted to a small/cheap model independently
    # of LLM_MODEL so the build stays cheap even when LLM_MODEL is bumped to a larger model
    # for assessment quality. Not used at demo/query time (enrichment doesn't run then).
    ENRICHMENT_MODEL: str = "mistral-small-latest"
    CHROMA_PERSIST_DIR: str = "./data/processed/chroma_db"

    # Only needed at ingestion time — defaulted so this module can be
    # imported (e.g. for retrieval/generation) without it set.
    SEC_USER_AGENT: str = ""


settings = Settings()

if not settings.MISTRAL_API_KEY:
    raise RuntimeError("MISTRAL_API_KEY not set. Add it to .env or Colab Secrets.")
