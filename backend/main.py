"""FastAPI app: health, WebSocket audio in, transcripts and battlecards out."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import traceback
import uuid
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from chain import CompetitorCooldown, maybe_build_battlecard_event
from chroma_init import build_vectorstore, load_battlecards_index
from config import get_settings
from schemas import (
    BattlecardEvent,
    ClientContextEvent,
    ErrorEvent,
    TranscriptEvent,
    battlecard_from_dict,
)
from supabase_client import (
    fetch_active_client_context,
    get_client_by_id,
    search_clients_by_name,
)
from transcription import (
    DeepgramStreamSession,
    MOCK_PHRASE,
    use_mock_transcription,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("DEEPGRAM_API_KEY loaded:", bool(os.getenv("DEEPGRAM_API_KEY")))
    settings = get_settings()
    app.state.settings = settings
    app.state.cards_by_name = load_battlecards_index(settings)
    app.state.vectorstore = None
    try:
        app.state.vectorstore = await asyncio.to_thread(build_vectorstore, settings)
        logger.info("Chroma vectorstore ready (Hugging Face embeddings: %s)", settings.embedding_model)
    except Exception:
        logger.exception(
            "Chroma init failed (delete backend/chroma_db if you switched embedding models); "
            "continuing with alias-only matching"
        )
    yield


app = FastAPI(title="SignalCard API", lifespan=lifespan)
_settings = get_settings()
_origins = (
    ["*"]
    if _settings.cors_origins.strip() == "*"
    else [o.strip() for o in _settings.cors_origins.split(",") if o.strip()]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root() -> dict[str, Any]:
    """Avoid 404 when opening the server URL in a browser; this API is WebSocket + JSON routes."""
    return {
        "service": "SignalCard API",
        "health": "/health",
        "websocket": "/ws",
    }


@app.get("/health")
async def health(request: Request) -> dict[str, Any]:
    settings: Any = request.app.state.settings
    vs = request.app.state.vectorstore
    return {
        "status": "ok",
        "chroma_ready": vs is not None,
        "huggingface_embeddings": True,
        "embedding_model": settings.embedding_model,
        "openai_configured": bool(settings.openai_api_key),
        "deepgram_configured": bool(settings.deepgram_api_key),
        "deepgram_model": settings.deepgram_model,
        "supabase_configured": bool(settings.supabase_url and settings.supabase_key),
        "mock_transcription": use_mock_transcription(settings),
    }


@app.get("/clients/search")
async def search_clients(
    q: str = Query(default="", min_length=1),
    limit: int = Query(default=5, ge=1, le=20),
) -> dict[str, Any]:
    try:
        rows = search_clients_by_name(q, limit=limit, raise_on_error=True)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"clients_search_failed: {e}") from e
    return {"items": rows}


@app.get("/clients/{client_id}")
async def get_client(client_id: str) -> dict[str, Any]:
    row = get_client_by_id(client_id)
    return {"item": row}


def _mock_battlecard_event(client_context: dict[str, Any] | None) -> BattlecardEvent:
    raw = {
        "key_differentiator": "Nuestro motor de automatización no cobra por acción.",
        "suggested_response": "Muchos equipos migran desde HubSpot cuando los workflows se vuelven más complejos.",
        "recommended_question": "¿Qué limitaciones han encontrado con HubSpot hasta ahora?",
        "weaknesses": ["Escalabilidad limitada", "Add-ons costosos"],
        "strengths": ["Personalización de workflows", "API flexible"],
        "metrics": {
            "win_rate_vs_competitor": 58,
            "avg_deal_cycle_days": 42,
            "deals_won_last_quarter": 12,
            "deals_lost_last_quarter": 7,
            "risk_level": "medio",
            "top_objection": "Add-ons costosos",
        },
        "chart_data": {
            "win_loss_trend": [
                {"month": "Ene", "won": 2, "lost": 2},
                {"month": "Feb", "won": 3, "lost": 1},
                {"month": "Mar", "won": 3, "lost": 2},
                {"month": "Abr", "won": 4, "lost": 1},
                {"month": "May", "won": 5, "lost": 1},
            ],
            "feature_comparison": [
                {"feature": "Tiempo real", "nosotros": 100, "competidor": 20},
                {"feature": "Precio", "nosotros": 85, "competidor": 60},
                {"feature": "Workflows", "nosotros": 90, "competidor": 80},
                {"feature": "Onboarding", "nosotros": 95, "competidor": 70},
            ],
            "objection_frequency": [
                {"objection": "Add-ons costosos", "count": 8},
                {"objection": "Escalabilidad", "count": 5},
                {"objection": "Reporting", "count": 3},
            ],
        },
    }
    return battlecard_from_dict(
        competitor="HubSpot",
        confidence=0.96,
        raw=raw,
        client_context=client_context,
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    settings = websocket.app.state.settings
    vectorstore = websocket.app.state.vectorstore
    cards_by_name = websocket.app.state.cards_by_name
    client_context = fetch_active_client_context(settings)
    cooldown = CompetitorCooldown(settings.battlecard_cooldown_seconds)
    transcript_queue: asyncio.Queue = asyncio.Queue()
    shutdown = asyncio.Event()
    rolling_transcript = ""
    utterance_id = str(uuid.uuid4())

    async def send_json_safe(payload: Any) -> None:
        try:
            if isinstance(payload, str):
                await websocket.send_text(payload)
            elif hasattr(payload, "model_dump_json"):
                # exclude_none mantiene el JSON limpio: el FE fusiona raíz + data
                # y trata claves ausentes como "no enriquecido". Mandar null para
                # metrics / chart_data en cards básicas (apollo, gong, hubspot)
                # confunde esa heurística.
                await websocket.send_text(payload.model_dump_json(exclude_none=True))
            else:
                await websocket.send_text(json.dumps(payload))
        except Exception as e:
            logger.debug("WebSocket send failed: %s", e)

    async def pump_transcripts() -> None:
        nonlocal rolling_transcript, utterance_id
        while not shutdown.is_set():
            try:
                item = await asyncio.wait_for(transcript_queue.get(), timeout=0.35)
            except asyncio.CancelledError:
                raise
            except asyncio.TimeoutError:
                continue
            text = str(item.get("text", "")).strip()
            is_final = bool(item.get("is_final", False))
            if not text:
                continue
            rolling_transcript = (rolling_transcript + " " + text).strip()
            if len(rolling_transcript) > 4000:
                rolling_transcript = rolling_transcript[-4000:]
            window = rolling_transcript[-1200:] if len(rolling_transcript) > 1200 else rolling_transcript

            te = TranscriptEvent(id=utterance_id, text=text, is_final=is_final)
            await send_json_safe(te)
            if is_final:
                utterance_id = str(uuid.uuid4())

            current_client_context = client_context
            try:
                event = maybe_build_battlecard_event(
                    window,
                    vectorstore,
                    cards_by_name,
                    current_client_context,
                    cooldown,
                    settings,
                )
                if event:
                    competitor_name = getattr(event, "competitor", "?")
                    logger.info(f"Enviando battlecard al frontend: {competitor_name}")
                    await send_json_safe(event)
            except Exception:
                logger.exception("Battlecard pipeline error")
                await send_json_safe(
                    ErrorEvent(
                        message="battlecard_pipeline_error",
                        detail=traceback.format_exc()[-800:],
                    )
                )

    async def emit_client_context() -> None:
        await send_json_safe(ClientContextEvent(client_context=client_context))

    if settings.mock_battlecard_on_connect:
        await send_json_safe(_mock_battlecard_event(client_context))
    else:
        await emit_client_context()

    pump_task = asyncio.create_task(pump_transcripts())
    session: DeepgramStreamSession | None = None
    loop = asyncio.get_running_loop()
    chunk_count = 0
    # Next binary audio frame is tagged mic | screen | mixed (see cmd audio_source).
    next_audio_source: str = "mixed"

    err_sent = False

    def on_deepgram_error(msg: str) -> None:
        nonlocal err_sent
        if err_sent:
            return
        err_sent = True
        asyncio.run_coroutine_threadsafe(
            send_json_safe(ErrorEvent(message="deepgram_error", detail=msg[:500])),
            loop,
        )

    if not use_mock_transcription(settings):
        try:
            session = DeepgramStreamSession(
                settings.deepgram_api_key or "",
                loop,
                transcript_queue,
                on_error=on_deepgram_error,
            )
            session.start()
        except Exception as e:
            logger.exception("Failed to start Deepgram: %s", e)
            await send_json_safe(ErrorEvent(message="deepgram_start_failed", detail=str(e)))
            session = None

    inject_demo_transcripts = use_mock_transcription(settings) or session is None

    try:
        while True:
            message = await websocket.receive()
            mtype = message.get("type")
            if mtype == "websocket.disconnect":
                break
            if "bytes" in message and message["bytes"] is not None:
                data: bytes = message["bytes"]
                if session is not None:
                    session.send_audio(data, source=next_audio_source)  # type: ignore[arg-type]
                if inject_demo_transcripts:
                    chunk_count += 1
                    if chunk_count % max(1, settings.mock_transcript_every_n_chunks) == 0:
                        await transcript_queue.put({"text": MOCK_PHRASE, "is_final": True})
            elif "text" in message and message["text"] is not None:
                raw_txt = message["text"]
                try:
                    body = json.loads(raw_txt)
                except json.JSONDecodeError:
                    continue
                cmd = body.get("cmd")
                if cmd == "mock_battlecard":
                    await send_json_safe(_mock_battlecard_event(client_context))
                elif cmd == "ping":
                    await send_json_safe({"type": "pong"})
                elif cmd == "audio_source":
                    src = body.get("source", "mixed")
                    next_audio_source = src if src in ("mic", "screen", "mixed") else "mixed"
                elif cmd == "set_client":
                    client_id = str(body.get("client_id") or "").strip()
                    if not client_id:
                        await send_json_safe(
                            ErrorEvent(message="set_client_failed", detail="client_id_missing")
                        )
                        continue
                    selected = get_client_by_id(client_id, settings)
                    if selected is None:
                        await send_json_safe(
                            ErrorEvent(message="set_client_failed", detail="client_not_found")
                        )
                        continue
                    client_context = selected
                    await emit_client_context()
                elif cmd == "clear_client":
                    client_context = None
                    await emit_client_context()
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    finally:
        shutdown.set()
        pump_task.cancel()
        try:
            await pump_task
        except asyncio.CancelledError:
            pass
        if session is not None:
            session.stop()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
