"""Competitor detection, Chroma retrieval, and LLM formatting."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import Settings, get_settings
from schemas import BattlecardEvent, battlecard_from_dict

logger = logging.getLogger(__name__)

# Canonical competitor name -> substrings to match in lowercased transcript (README + aliases)
COMPETITOR_ALIASES: dict[str, list[str]] = {
    "HubSpot": ["hubspot", "hub spot", "crm de hubspot"],
    "Salesforce": ["salesforce", "sales force", "sfdc", "crm de salesforce"],
    "Gong": ["gong", "gong.io", "plataforma de gong", "gong platform"],
    "Apollo": ["apollo", "apollo.io"],
}


class CompetitorCooldown:
    def __init__(self, cooldown_seconds: float) -> None:
        self.cooldown_seconds = cooldown_seconds
        self._last: dict[str, float] = {}

    def allow(self, competitor: str) -> bool:
        now = time.monotonic()
        prev = self._last.get(competitor)
        if prev is not None and (now - prev) < self.cooldown_seconds:
            return False
        self._last[competitor] = now
        return True


def _alias_match(text_lower: str) -> tuple[str, float] | None:
    for canonical, needles in COMPETITOR_ALIASES.items():
        for n in needles:
            if n in text_lower:
                return canonical, 1.0
    return None


def _extract_json_object(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        raise ValueError("No JSON object in model output")
    return json.loads(m.group())


def _format_with_llm(
    card: dict[str, Any],
    client_context: dict[str, Any] | None,
    settings: Settings,
) -> dict[str, Any]:
    if not settings.openai_api_key:
        return card

    llm = ChatOpenAI(
        api_key=settings.openai_api_key,
        model=settings.llm_model,
        temperature=0.2,
    )
    sys = (
        "Eres un asistente de ventas. Recibes una battlecard en JSON y opcionalmente "
        "contexto del cliente. Devuelve SOLO un JSON válido con las claves: "
        "key_differentiator, suggested_response, recommended_question, weaknesses (array de strings cortos). "
        "Mantén tono profesional en español. No agregues texto fuera del JSON."
    )
    human = json.dumps(
        {"battlecard": card, "client_context": client_context or {}},
        ensure_ascii=False,
    )
    msg = llm.invoke([SystemMessage(content=sys), HumanMessage(content=human)])
    content = msg.content if isinstance(msg.content, str) else str(msg.content)
    try:
        return _extract_json_object(content)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("LLM JSON parse failed, using raw card: %s", e)
        return card


def resolve_competitor_and_doc(
    transcript_window: str,
    vectorstore: Chroma | None,
    cards_by_name: dict[str, dict],
    settings: Settings,
) -> tuple[str, float, dict[str, Any]] | None:
    """Return (competitor, confidence, card_dict) or None."""
    t = transcript_window.lower().strip()
    if not t:
        return None

    alias = _alias_match(t)
    if alias:
        canonical, conf = alias
        card = cards_by_name.get(canonical)
        if card:
            return canonical, conf, card

    if vectorstore is None:
        return None

    doc = None
    score = 0.0
    rel_pairs = None
    try:
        rel_pairs = vectorstore.similarity_search_with_relevance_scores(
            transcript_window,
            k=1,
        )
    except Exception as e:
        logger.debug("Relevance search unavailable: %s", e)

    if rel_pairs:
        doc, score = rel_pairs[0]
        if score < settings.min_relevance_score:
            return None
    else:
        try:
            dist_pairs = vectorstore.similarity_search_with_score(transcript_window, k=1)
        except Exception as e2:
            logger.exception("Chroma search failed: %s", e2)
            return None
        if not dist_pairs:
            return None
        doc, dist = dist_pairs[0]
        if float(dist) > settings.max_l2_distance:
            return None
        score = 1.0 / (1.0 + float(dist))

    if doc is None:
        return None
    competitor = str(doc.metadata.get("competitor", ""))
    raw = doc.metadata.get("card_json")
    if not raw:
        return None
    card = json.loads(raw)
    return competitor, float(score), card


def maybe_build_battlecard_event(
    transcript_window: str,
    vectorstore: Chroma | None,
    cards_by_name: dict[str, dict],
    client_context: dict[str, Any] | None,
    cooldown: CompetitorCooldown,
    settings: Settings | None = None,
) -> BattlecardEvent | None:
    settings = settings or get_settings()
    resolved = resolve_competitor_and_doc(
        transcript_window, vectorstore, cards_by_name, settings
    )
    if not resolved:
        return None
    competitor, confidence, card = resolved
    if not cooldown.allow(competitor):
        logger.debug("Cooldown skip for %s", competitor)
        return None

    formatted = _format_with_llm(card, client_context, settings)
    return battlecard_from_dict(
        competitor=competitor,
        confidence=confidence,
        raw=formatted,
        client_context=client_context,
    )
