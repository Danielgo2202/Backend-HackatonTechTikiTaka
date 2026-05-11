"""Competitor detection, Chroma retrieval, and LLM formatting."""

from __future__ import annotations

import json
import logging
import math
import re
import time
from typing import Any

from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import Settings, get_settings
from schemas import BattlecardEvent, battlecard_from_dict

logger = logging.getLogger(__name__)

# Capa 1 — keywords estrictos (orden de aparición en texto: gana la última mención).
_KEYWORD_RULES: list[tuple[str, list[str]]] = [
    ("HubSpot", [r"\bhub\s+spot\b", r"\bhubspot\b"]),
    ("Salesforce", [r"\bsales\s+force\b", r"\bsalesforce\b", r"\bsfdc\b"]),
    ("Gong", [r"\bgong\.io\b", r"\bgong\b"]),
    ("Apollo", [r"\bapollo\.io\b", r"\bapollo\b"]),
]


class CompetitorCooldown:
    """Timestamps por competidor (memoria); se pierde al reiniciar el proceso."""

    def __init__(self, cooldown_seconds: float) -> None:
        self.cooldown_seconds = cooldown_seconds
        self._last_sent: dict[str, float] = {}

    def seconds_remaining(self, competitor: str) -> int:
        prev = self._last_sent.get(competitor)
        if prev is None:
            return 0
        elapsed = time.monotonic() - prev
        rem = self.cooldown_seconds - elapsed
        return max(0, int(math.ceil(rem)))

    def is_in_cooldown(self, competitor: str) -> bool:
        return self.seconds_remaining(competitor) > 0

    def allow(self, competitor: str) -> bool:
        now = time.monotonic()
        prev = self._last_sent.get(competitor)
        if prev is not None and (now - prev) < self.cooldown_seconds:
            return False
        self._last_sent[competitor] = now
        return True


def _keyword_match_layer1(transcript: str) -> str | None:
    """Último competidor mencionado entre los términos de Capa 1, o None."""
    t = transcript.lower()
    matches: list[tuple[int, str]] = []
    for canonical, patterns in _KEYWORD_RULES:
        for pat in patterns:
            for m in re.finditer(pat, t):
                matches.append((m.start(), canonical))
    if not matches:
        return None
    matches.sort(key=lambda x: x[0])
    return matches[-1][1]


def _card_from_chroma_metadata_filter(
    vectorstore: Chroma,
    competitor: str,
) -> dict[str, Any] | None:
    """Capa 2 — sin similarity libre: lectura por metadata competitor exacto."""
    try:
        data = vectorstore.get(where={"competitor": competitor}, limit=1)
        metas = data.get("metadatas") or []
        if not metas or metas[0] is None:
            return None
        raw = metas[0].get("card_json")
        if not raw:
            return None
        return json.loads(raw)
    except Exception as e:
        logger.debug("Chroma get by competitor=%s failed: %s", competitor, e)
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
        llm_out = _extract_json_object(content)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("LLM JSON parse failed, using raw card: %s", e)
        return card
    # LLM solo regenera campos de texto (key_differentiator, suggested_response,
    # recommended_question, weaknesses). Preservamos metrics / chart_data / strengths
    # del card original — el FE los necesita para pintar pills + Recharts.
    return {**card, **llm_out}


def resolve_competitor_and_doc(
    transcript_window: str,
    vectorstore: Chroma | None,
    cards_by_name: dict[str, dict],
    settings: Settings,
) -> tuple[str, float, dict[str, Any]] | None:
    """Return (competitor, confidence, card_dict) or None."""
    t = transcript_window.strip()
    if not t:
        return None

    # --- Capa 1 + 2: keyword primero; card solo por metadata filtrado o índice JSON ---
    kw_name = _keyword_match_layer1(transcript_window)
    if kw_name:
        logger.info(f"Competidor detectado por keyword: {kw_name}")
        # Preferimos el índice en memoria: se carga desde disco al arrancar y
        # SIEMPRE refleja el JSON actual de battlecards/. Chroma persiste
        # `card_json` en disco y queda obsoleto si editamos un JSON sin borrar
        # backend/chroma_db/, lo cual hacía que llegaran cards sin metrics /
        # chart_data al frontend aun teniendo el JSON enriquecido.
        card: dict[str, Any] | None = cards_by_name.get(kw_name)
        if card is None and vectorstore is not None:
            card = _card_from_chroma_metadata_filter(vectorstore, kw_name)
        if card:
            return kw_name, 1.0, card
        logger.debug("Keyword %s sin battlecard en índice local ni Chroma", kw_name)
        return None

    # --- Capa 3: similarity solo si no hubo keyword ---
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
        effective = max(0.0, float(score))
        if effective < settings.min_relevance_score:
            return None
        score = effective
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
    logger.info(f"Competidor detectado por similarity: {competitor} (score: {score})")
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
    tw = transcript_window.strip()
    kw_early = _keyword_match_layer1(transcript_window) if tw else None
    if kw_early and cooldown.is_in_cooldown(kw_early):
        rem = cooldown.seconds_remaining(kw_early)
        logger.info(f"Battlecard de {kw_early} ignorada — cooldown activo ({rem}s)")
        return None

    resolved = resolve_competitor_and_doc(
        transcript_window, vectorstore, cards_by_name, settings
    )
    if not resolved:
        return None
    competitor, confidence, card = resolved
    if not cooldown.allow(competitor):
        rem = cooldown.seconds_remaining(competitor)
        logger.info(f"Battlecard de {competitor} ignorada — cooldown activo ({rem}s)")
        return None

    formatted = _format_with_llm(card, client_context, settings)
    return battlecard_from_dict(
        competitor=competitor,
        confidence=confidence,
        raw=formatted,
        client_context=client_context,
    )