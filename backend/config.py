"""Application settings from environment variables.

`embedding_model` is the Hugging Face / Sentence Transformers model id used for Chroma (local, no API key).
"""

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
    embedding_model: str = "all-MiniLM-L6-v2"

    mock_transcription: bool = False
    mock_transcript_every_n_chunks: int = 8
    mock_battlecard_on_connect: bool = False

    battlecard_cooldown_seconds: float = 45.0
    # Temporal: -1.0 deja pasar cualquier score de relevance (Chroma puede ser negativo).
    min_relevance_score: float = -1.0
    max_l2_distance: float = 1.85

    # Groq STT: avoid 429 — longer interval + spacing between mic+screen calls.
    groq_transcribe_interval_seconds: float = 12.0
    groq_min_gap_between_calls_seconds: float = 1.5

    cors_origins: str = "*"


@lru_cache
def get_settings() -> Settings:
    return Settings()
