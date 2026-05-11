"""FastAPI app: health, WebSocket audio in, transcripts and battlecards out.

Producción (Railway / Vercel):
  - El proceso debe arrancarse con `--proxy-headers` y `--forwarded-allow-ips=*`
    para que FastAPI vea el `X-Forwarded-Proto: https` que añade el load balancer
    de Railway. Sin esto, los redirects internos pueden cortar a `http://`.
  - El healthcheck de Railway apunta a /healthz (responde 200 SIEMPRE, sin
    tocar app.state). /health da más detalle pero puede tardar en cold start.
  - El lifespan NO bloquea el arranque con la indexación de Chroma: lo lanza
    en background. /ws y /health funcionan desde el segundo 1.
"""

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def _build_vectorstore_in_background(app: FastAPI) -> None:
    """Indexa Chroma sin bloquear el arranque del server.

    Cold start en Railway: descargar `all-MiniLM-L6-v2` desde HuggingFace puede
    tardar 1-3 min. Si bloqueamos el lifespan, /ws devuelve 503 hasta que
    termina y el FE no puede conectar. Como tenemos `cards_by_name` (índice
    en memoria desde JSON) que cubre keyword matching, el WS funciona sin
    Chroma — solo perdemos similarity search hasta que esté listo.
    """
    settings = app.state.settings
    try:
        app.state.vectorstore = await asyncio.to_thread(build_vectorstore, settings)
        logger.info(
            "Chroma vectorstore ready (HF embeddings: %s)", settings.embedding_model
        )
    except Exception:
        logger.exception(
            "Chroma init failed — continuing with in-memory cards_by_name only "
            "(keyword matching seguirá funcionando)"
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    app.state.cards_by_name = load_battlecards_index(settings)
    app.state.vectorstore = None
    app.state.vectorstore_task = asyncio.create_task(
        _build_vectorstore_in_background(app)
    )
    logger.info(
        "App startup: deepgram=%s groq=%s supabase=%s mock_stt=%s cards=%d",
        bool(settings.deepgram_api_key),
        bool(settings.groq_api_key),
        bool(settings.supabase_url and settings.supabase_key),
        use_mock_transcription(settings),
        len(app.state.cards_by_name),
    )
    try:
        yield
    finally:
        task = getattr(app.state, "vectorstore_task", None)
        if task is not None and not task.done():
            task.cancel()


app = FastAPI(title="SignalCard API", lifespan=lifespan)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
# Reglas:
#   - Si CORS_ORIGINS es "*", forzamos allow_credentials=False (CORS spec lo
#     prohíbe combinar y los navegadores rechazan la respuesta — es la causa
#     #1 de "WS no conecta" en producción cuando el FE manda cookies/headers).
#   - Si necesitas credenciales, define orígenes exactos en CORS_ORIGINS y/o
#     un patrón en CORS_ORIGIN_REGEX (ej. r"https://.*\.vercel\.app$" para
#     cubrir todos los preview deploys).
_settings = get_settings()
_origins_raw = (_settings.cors_origins or "*").strip()
if _origins_raw == "*":
    _origins: list[str] = ["*"]
    _allow_credentials = False
else:
    _origins = [o.strip() for o in _origins_raw.split(",") if o.strip()]
    _allow_credentials = True

_cors_kwargs: dict[str, Any] = {
    "allow_origins": _origins,
    "allow_credentials": _allow_credentials,
    "allow_methods": ["*"],
    "allow_headers": ["*"],
    # Útil para preflight OPTIONS de Vercel preview deploys.
    "expose_headers": ["*"],
    "max_age": 600,
}
if _settings.cors_origin_regex:
    _cors_kwargs["allow_origin_regex"] = _settings.cors_origin_regex

app.add_middleware(CORSMiddleware, **_cors_kwargs)
logger.info(
    "CORS configurado: origins=%s regex=%s credentials=%s",
    _origins,
    _settings.cors_origin_regex,
    _allow_credentials,
)


# ---------------------------------------------------------------------------
# HTTP routes
# ---------------------------------------------------------------------------


@app.get("/")
async def root() -> dict[str, Any]:
    """Avoid 404 when opening the server URL in a browser."""
    return {
        "service": "SignalCard API",
        "health": "/health",
        "healthz": "/healthz",
        "websocket": "/ws",
    }


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    """Healthcheck mínimo para Railway: 200 siempre, sin tocar app.state.

    El lifespan tarda en cold start (descarga modelo HF). Si Railway hace ping
    a /health antes de que termine, el container se marca como unhealthy y
    se reinicia en un loop. /healthz responde aunque no haya nada cargado.
    """
    return {"status": "ok"}


@app.get("/health")
async def health(request: Request) -> dict[str, Any]:
    """Healthcheck con detalle. Tolerante a app.state aún no inicializado."""
    settings: Any = getattr(request.app.state, "settings", None)
    vs = getattr(request.app.state, "vectorstore", None)
    cards = getattr(request.app.state, "cards_by_name", None)
    if settings is None:
        return {"status": "starting"}
    return {
        "status": "ok",
        "chroma_ready": vs is not None,
        "cards_loaded": len(cards or {}),
        "huggingface_embeddings": True,
        "embedding_model": settings.embedding_model,
        "groq_configured": bool(settings.groq_api_key),
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


def _ws_handshake_summary(websocket: WebSocket) -> dict[str, Any]:
    """Datos útiles del handshake para diagnóstico en producción."""
    h = websocket.headers
    client = websocket.client
    return {
        "client": f"{client.host}:{client.port}" if client else "?",
        "origin": h.get("origin"),
        "user_agent": (h.get("user-agent") or "")[:120],
        "x_forwarded_for": h.get("x-forwarded-for"),
        "x_forwarded_proto": h.get("x-forwarded-proto"),
        "host": h.get("host"),
    }


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    summary = _ws_handshake_summary(websocket)
    logger.info("WS handshake intent: %s", summary)
    await websocket.accept()
    logger.info("WS connected: client=%s origin=%s", summary["client"], summary["origin"])

    # app.state puede no tener todo si el primer cliente conecta durante el cold start.
    settings = getattr(websocket.app.state, "settings", None)
    if settings is None:
        logger.warning("WS rechazado: app.state.settings aún no inicializado")
        try:
            await websocket.send_text(
                json.dumps({"type": "error", "message": "server_starting"})
            )
        finally:
            await websocket.close(code=1013)  # Try Again Later
        return

    vectorstore = getattr(websocket.app.state, "vectorstore", None)
    cards_by_name = getattr(websocket.app.state, "cards_by_name", {}) or {}
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
                # y trata claves ausentes como "no enriquecido".
                await websocket.send_text(payload.model_dump_json(exclude_none=True))
            else:
                await websocket.send_text(json.dumps(payload))
        except Exception as e:
            logger.debug("WS send failed (cliente probablemente cerrado): %s", e)

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
                # Re-leer vectorstore por si terminó de inicializarse durante la sesión.
                vs_now = getattr(websocket.app.state, "vectorstore", None)
                event = maybe_build_battlecard_event(
                    window,
                    vs_now,
                    cards_by_name,
                    current_client_context,
                    cooldown,
                    settings,
                )
                if event:
                    competitor_name = getattr(event, "competitor", "?")
                    logger.info("Battlecard -> FE: %s", competitor_name)
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

    disconnect_reason = "normal"
    try:
        while True:
            try:
                message = await websocket.receive()
            except WebSocketDisconnect as e:
                disconnect_reason = f"client_disconnect code={e.code}"
                break
            mtype = message.get("type")
            if mtype == "websocket.disconnect":
                disconnect_reason = "websocket.disconnect"
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
                    logger.debug("WS texto no-JSON ignorado: %r", raw_txt[:200])
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
                else:
                    logger.debug("WS cmd desconocido: %r", cmd)
    except Exception:
        disconnect_reason = "exception"
        logger.exception("WS loop crash inesperado")
    finally:
        logger.info(
            "WS disconnected: client=%s reason=%s",
            summary["client"],
            disconnect_reason,
        )
        shutdown.set()
        pump_task.cancel()
        try:
            await pump_task
        except asyncio.CancelledError:
            pass
        if session is not None:
            session.stop()


# ---------------------------------------------------------------------------
# Entry point para desarrollo local. En producción Railway usa Procfile/railway.toml.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    reload_env = os.environ.get("UVICORN_RELOAD", "false").lower() in ("1", "true", "yes")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=reload_env,
        proxy_headers=True,
        forwarded_allow_ips="*",
        ws_ping_interval=20,
        ws_ping_timeout=30,
        timeout_keep_alive=65,
    )
