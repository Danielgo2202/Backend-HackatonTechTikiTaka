"""Deepgram live transcription (threaded) and mock path for demos without STT."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Callable
from typing import Any

import websockets.sync.client as websockets_sync_client
from deepgram import DeepgramClient
from deepgram.core.events import EventType
from deepgram.listen.v1.types.listen_v1results import ListenV1Results
from websockets.exceptions import InvalidStatus

from config import Settings, get_settings

logger = logging.getLogger(__name__)


def _deepgram_ws_transport(url: str, headers: dict[str, str] | None) -> Any:
    """Open WSS to Deepgram with conservative client options.

    - ``proxy=None``: skip ``HTTP_PROXY`` / system proxy (broken CONNECT often yields HTTP 408).
    - ``compression=None``: disable permessage-deflate (some filters mishandle it).
    """
    return websockets_sync_client.connect(
        url,
        additional_headers=dict(headers or {}),
        proxy=None,
        open_timeout=60.0,
        compression=None,
    )


def _handshake_failure_message(exc: InvalidStatus) -> str:
    """Human-readable handshake failure (Deepgram sends dg-error / body on rejects)."""
    r = exc.response
    dg_error = r.headers.get("dg-error")
    dg_rid = r.headers.get("dg-request-id")
    body = (r.body or b"")[:1200].decode("utf-8", errors="replace").strip()
    parts: list[str] = [f"HTTP {r.status_code}"]
    if dg_error:
        parts.append(f"dg-error={dg_error}")
    if dg_rid:
        parts.append(f"dg-request-id={dg_rid}")
    if body:
        parts.append(f"body={body}")
    bl = body.lower()
    if "request time-out" in bl and "browser" in bl:
        parts.append(
            "hint=generic_nginx_or_proxy_page_not_deepgram"
            " (VPN/antivirus_SSL_scan/corporate_proxy/firewall; try another network or disable HTTPS inspection)"
        )
    return "; ".join(parts)


def _results_to_text(message: Any) -> tuple[str, bool] | None:
    if not isinstance(message, ListenV1Results):
        return None
    alts = message.channel.alternatives
    if not alts:
        return None
    text = (alts[0].transcript or "").strip()
    if not text:
        return None
    is_final = bool(message.is_final)
    return text, is_final


class DeepgramStreamSession:
    """Runs Deepgram v1 listen in a background thread; audio can be sent from the async loop."""

    def __init__(
        self,
        api_key: str,
        loop: asyncio.AbstractEventLoop,
        transcript_queue: asyncio.Queue,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        self._client = DeepgramClient(
            api_key=api_key.strip(),
            transport_factory=_deepgram_ws_transport,
        )
        self._loop = loop
        self._queue = transcript_queue
        self._on_error = on_error
        self._conn_holder: dict[str, Any] = {"conn": None}
        self._thread: threading.Thread | None = None
        self._stopped = threading.Event()

    def _emit(self, text: str, is_final: bool) -> None:
        asyncio.run_coroutine_threadsafe(
            self._queue.put({"text": text, "is_final": is_final}),
            self._loop,
        )

    def _worker(self) -> None:
        try:
            with self._client.listen.v1.connect(
                model="nova-2",
                interim_results=True,
                smart_format=True,
                language="es",
            ) as connection:
                self._conn_holder["conn"] = connection

                def on_message(message: Any, **_kwargs: Any) -> None:
                    parsed = _results_to_text(message)
                    if not parsed:
                        return
                    text, is_final = parsed
                    self._emit(text, is_final)

                def on_error(exc: Any, **_kwargs: Any) -> None:
                    logger.exception("Deepgram socket error: %s", exc)
                    if self._on_error:
                        self._on_error(str(exc))

                connection.on(EventType.MESSAGE, on_message)
                connection.on(EventType.ERROR, on_error)
                connection.start_listening()
        except InvalidStatus as e:
            msg = _handshake_failure_message(e)
            logger.error("Deepgram WebSocket handshake rejected: %s", msg)
            if self._on_error:
                self._on_error(f"deepgram_handshake: {msg}")
        except Exception as e:
            logger.exception("Deepgram session failed: %s", e)
            if self._on_error:
                self._on_error(str(e))
        finally:
            self._conn_holder["conn"] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._worker, name="deepgram-listen", daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if self._conn_holder["conn"] is not None:
                return
            if self._stopped.is_set():
                return
            time.sleep(0.05)
        logger.warning("Deepgram connection not ready after wait; audio may be dropped until connect completes")

    def send_audio(self, chunk: bytes) -> None:
        conn = self._conn_holder["conn"]
        if conn is None:
            return
        try:
            conn.send_media(chunk)
        except Exception as e:
            logger.warning("send_media failed: %s", e)

    def stop(self) -> None:
        self._stopped.set()
        conn = self._conn_holder["conn"]
        if conn is not None:
            try:
                conn.send_close_stream()
            except Exception as e:
                logger.debug("send_close_stream: %s", e)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)


def use_mock_transcription(settings: Settings | None = None) -> bool:
    settings = settings or get_settings()
    if settings.mock_transcription:
        return True
    if not settings.deepgram_api_key:
        return True
    return False


MOCK_PHRASE = (
    "También estamos evaluando HubSpot para el CRM y queremos comparar opciones."
)


async def whisper_transcribe_wav_bytes(api_key: str, wav_bytes: bytes) -> str:
    """Non-streaming fallback using OpenAI Whisper (full buffer). For short demo clips only."""
    import io

    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key)
    bio = io.BytesIO(wav_bytes)
    bio.name = "clip.wav"
    tr = await client.audio.transcriptions.create(model="whisper-1", file=bio)
    return (tr.text or "").strip()
