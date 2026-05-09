"""Load battlecard JSON files into ChromaDB at startup."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from config import Settings, get_settings

logger = logging.getLogger(__name__)


def load_battlecards_index(settings: Settings | None = None) -> dict[str, dict]:
    """Map competitor display name -> battlecard dict (for exact / alias matches)."""
    settings = settings or get_settings()
    battle_dir = Path(settings.battlecards_dir)
    index: dict[str, dict] = {}
    for path in battle_dir.glob("*.json"):
        with path.open(encoding="utf-8") as f:
            card = json.load(f)
        name = str(card.get("competitor", path.stem))
        index[name] = card
    return index


def _embedding_text(card: dict) -> str:
    """Text used for semantic retrieval (competitor + aliases + strengths + weaknesses)."""
    competitor = card.get("competitor", "")
    lines = [
        competitor,
        f"Competidor: {competitor}",
        " ".join(card.get("strengths") or []),
        " ".join(card.get("weaknesses") or []),
        card.get("key_differentiator", ""),
    ]
    return "\n".join(lines)


def build_vectorstore(settings: Settings | None = None) -> Chroma:
    settings = settings or get_settings()
    battle_dir = Path(settings.battlecards_dir)
    if not battle_dir.is_dir():
        raise FileNotFoundError(f"Battlecards directory missing: {battle_dir}")

    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required to build Chroma embeddings")

    embeddings = OpenAIEmbeddings(
        api_key=settings.openai_api_key,
        model=settings.embedding_model,
    )
    persist = Path(settings.chroma_persist_dir)
    persist.mkdir(parents=True, exist_ok=True)

    documents: list[Document] = []
    for path in sorted(battle_dir.glob("*.json")):
        with path.open(encoding="utf-8") as f:
            card = json.load(f)
        competitor = str(card.get("competitor", path.stem))
        doc = Document(
            page_content=_embedding_text(card),
            metadata={
                "competitor": competitor,
                "source": path.name,
                "card_json": json.dumps(card, ensure_ascii=False),
            },
        )
        documents.append(doc)

    if not documents:
        raise RuntimeError(f"No battlecard JSON files found in {battle_dir}")

    logger.info("Indexing %d battlecards into Chroma at %s", len(documents), persist)
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(persist),
    )
    return vectorstore
