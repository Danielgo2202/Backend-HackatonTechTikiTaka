"""Application settings from environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str | None = None
    deepgram_api_key: str | None = None

    supabase_url: str | None = None
    supabase_key: str | None = None

    chroma_persist_dir: str = str(Path(__file__).resolve().parent / "chroma_db")
    battlecards_dir: str = str(Path(__file__).resolve().parent / "battlecards")

    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    mock_transcription: bool = False
    mock_transcript_every_n_chunks: int = 8
    mock_battlecard_on_connect: bool = False

    battlecard_cooldown_seconds: float = 45.0
    min_relevance_score: float = 0.32
    max_l2_distance: float = 1.85

    cors_origins: str = "*"


@lru_cache
def get_settings() -> Settings:
    return Settings()
