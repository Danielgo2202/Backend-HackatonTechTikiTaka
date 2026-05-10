"""Deepgram live streaming STT (WebM/Opus desde el browser) + modo mic/pantalla (dos streams).

Los chunks binarios del WebSocket se reenvían con ``send_media``. Para audio containerizado
(WebM), no se fija ``encoding`` en ``listen.v1.connect`` para que Deepgram infiera el formato.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
from collections.abc import Callable
from typing import Any, Literal

from deepgram import DeepgramClient
from deepgram.core.events import EventType
from deepgram.listen.v1.types.listen_v1results import ListenV1Results
from deepgram.types.listen_v1response import ListenV1Response
from websockets.exceptions import InvalidStatus

from config import Settings, get_settings

logger = logging.getLogger(__name__)

_HALLUCINATION_PHRASES: frozenset[str] = frozenset(
    {
        "gracias",
        "suscríbete",
        "hasta luego",
        "bye",
        "thank you",
    }
)

AudioSource = Literal["mixed", "mic", "screen"]


def _normalize_deepgram_key(raw: str) -> str:
    key = (raw or "").strip().strip('"').strip("'")
    key = key.replace("\r", "").replace("\n", "").strip()
    low = key.lower()
    if low.startswith("bearer "):
        key = key[7:].strip()
        low = key.lower()
    if low.startswith("token "):
        key = key[6:].strip()
    return key


def _handshake_failure_message(exc: InvalidStatus) -> str:
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
            "hint=proxy_or_ssl_inspection (VPN/antivirus/firewall; try another network)"
        )
    return "; ".join(parts)


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text.strip()))


def _is_spurious_transcript(text: str) -> bool:
    raw = (text or "").strip()
    if not raw:
        return True
    t = re.sub(r"\s+", " ", raw.lower()).strip(" .,!?:;")
    if _word_count(raw) < 4:
        return True
    if t in _HALLUCINATION_PHRASES:
        return True
    if t.rstrip(".") in _HALLUCINATION_PHRASES:
        return True
    return False


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


def _prerecorded_transcript(resp: ListenV1Response) -> str:
    chans = resp.results.channels
    if not chans:
        return ""
    alts = chans[0].alternatives
    if not alts:
        return ""
    return (alts[0].transcript or "").strip()


class _DeepgramStreamRunner:
    """Un hilo por conexión live: ``start_listening`` bloquea; ``send_media`` desde otros hilos."""

    def __init__(
        self,
        *,
        dg_client: DeepgramClient,
        listen_kwargs: dict[str, Any],
        loop: asyncio.AbstractEventLoop,
        transcript_queue: asyncio.Queue,
        on_error: Callable[[str], None] | None,
        name: str,
    ) -> None:
        self._client = dg_client
        self._listen_kwargs = listen_kwargs
        self._loop = loop
        self._queue = transcript_queue
        self._on_error = on_error
        self._name = name
        self._conn_holder: dict[str, Any] = {"conn": None}
        self._thread: threading.Thread | None = None
        self._stopped = threading.Event()
        self._started = False
        self._start_lock = threading.Lock()

    def _emit(self, text: str, is_final: bool) -> None:
        asyncio.run_coroutine_threadsafe(
            self._queue.put({"text": text, "is_final": is_final}),
            self._loop,
        )

    def _worker(self) -> None:
        try:
            with self._client.listen.v1.connect(**self._listen_kwargs) as connection:
                self._conn_holder["conn"] = connection

                def on_message(message: Any, **_kwargs: Any) -> None:
                    parsed = _results_to_text(message)
                    if not parsed:
                        return
                    text, is_final = parsed
                    if _is_spurious_transcript(text):
                        logger.debug(
                            "Deepgram [%s] dropped spurious: %r", self._name, text[:80]
                        )
                        return
                    self._emit(text, is_final)

                def on_err(exc: Any, **_kwargs: Any) -> None:
                    logger.exception("Deepgram [%s] socket error: %s", self._name, exc)
                    if self._on_error:
                        self._on_error(f"{self._name}: {exc!s}")

                connection.on(EventType.MESSAGE, on_message)
                connection.on(EventType.ERROR, on_err)
                connection.start_listening()
        except InvalidStatus as e:
            msg = _handshake_failure_message(e)
            logger.error("Deepgram [%s] handshake rejected: %s", self._name, msg)
            if self._on_error:
                self._on_error(f"deepgram_handshake [{self._name}]: {msg}")
        except Exception as e:
            logger.exception("Deepgram [%s] session failed: %s", self._name, e)
            if self._on_error:
                self._on_error(f"{self._name}: {e!s}")
        finally:
            self._conn_holder["conn"] = None

    def start(self) -> None:
        with self._start_lock:
            if self._started:
                return
            self._started = True
            self._thread = threading.Thread(
                target=self._worker, name=f"deepgram-{self._name}", daemon=True
            )
            self._thread.start()
            deadline = time.monotonic() + 15.0
            while time.monotonic() < deadline:
                if self._conn_holder["conn"] is not None:
                    return
                if self._stopped.is_set():
                    return
                time.sleep(0.05)
            logger.warning(
                "Deepgram [%s] connection not ready after wait; audio may drop briefly",
                self._name,
            )

    def send(self, chunk: bytes) -> None:
        conn = self._conn_holder["conn"]
        if conn is None or not chunk:
            return
        try:
            conn.send_media(chunk)
        except Exception as e:
            logger.warning("Deepgram [%s] send_media failed: %s", self._name, e)

    def stop(self) -> None:
        self._stopped.set()
        conn = self._conn_holder["conn"]
        if conn is not None:
            try:
                conn.send_close_stream()
            except Exception as e:
                logger.debug("send_close_stream [%s]: %s", self._name, e)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=8.0)
        self._conn_holder["conn"] = None


class DeepgramStreamSession:
    """Streaming a Deepgram; modo split abre dos conexiones (mic + pantalla)."""

    def __init__(
        self,
        api_key: str,
        loop: asyncio.AbstractEventLoop,
        transcript_queue: asyncio.Queue,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        key = _normalize_deepgram_key(api_key)
        if not key:
            raise ValueError("Configure DEEPGRAM_API_KEY")
        # Cliente por defecto del SDK (mismo handshake que documentación oficial).
        # transport_factory+custom websockets rompía el patrón ``with connect()`` en
        # algunas versiones y puede alterar extensiones/handshake frente a Deepgram.
        self._client = DeepgramClient(api_key=key)
        self._loop = loop
        self._queue = transcript_queue
        self._on_error = on_error
        self._settings = get_settings()
        model = (self._settings.deepgram_model or "").strip() or "nova-2"
        # Deepgram: en query string usar "true"/"false" minúsculas, no bool Python.
        self._listen_kw: dict[str, Any] = {
            "model": model,
            "interim_results": "true",
            "smart_format": "true",
            "language": "es",
        }
        self._mode: Literal["mixed", "split"] = "mixed"
        self._lock = threading.Lock()
        self._mixed: _DeepgramStreamRunner | None = None
        self._mic: _DeepgramStreamRunner | None = None
        self._scr: _DeepgramStreamRunner | None = None

    def _build_runner(self, name: str) -> _DeepgramStreamRunner:
        return _DeepgramStreamRunner(
            dg_client=self._client,
            listen_kwargs=self._listen_kw,
            loop=self._loop,
            transcript_queue=self._queue,
            on_error=self._on_error,
            name=name,
        )

    def start(self) -> None:
        """Las conexiones se abren en el primer ``send_audio`` (lazy)."""
        return

    def send_audio(self, chunk: bytes, source: AudioSource = "mixed") -> None:
        if not chunk:
            return
        if source not in ("mixed", "mic", "screen"):
            source = "mixed"

        mixed_to_stop: _DeepgramStreamRunner | None = None
        with self._lock:
            if source in ("mic", "screen"):
                if self._mode == "mixed":
                    self._mode = "split"
                    mixed_to_stop = self._mixed
                    self._mixed = None
            elif self._mode == "split":
                logger.debug(
                    "Ignorando mixed en modo split; usad cmd audio_source"
                )
                return

        if mixed_to_stop is not None:
            mixed_to_stop.stop()

        if source in ("mic", "screen"):
            runner: _DeepgramStreamRunner | None = None
            with self._lock:
                if source == "mic":
                    if self._mic is None:
                        self._mic = self._build_runner("mic")
                    runner = self._mic
                else:
                    if self._scr is None:
                        self._scr = self._build_runner("screen")
                    runner = self._scr
            if runner is not None:
                runner.start()
                runner.send(chunk)
        else:
            runner_m: _DeepgramStreamRunner | None = None
            with self._lock:
                if self._mixed is None:
                    self._mixed = self._build_runner("mixed")
                runner_m = self._mixed
            if runner_m is not None:
                runner_m.start()
                runner_m.send(chunk)

    def stop(self) -> None:
        with self._lock:
            runners = [self._mixed, self._mic, self._scr]
            self._mixed = self._mic = self._scr = None
        for r in runners:
            if r is not None:
                r.stop()


def use_mock_transcription(settings: Settings | None = None) -> bool:
    settings = settings or get_settings()
    if settings.mock_transcription:
        return True
    if not _normalize_deepgram_key(settings.deepgram_api_key or ""):
        return True
    return False


MOCK_PHRASE = (
    "También estamos evaluando HubSpot para el CRM y queremos comparar opciones."
)


def _sync_transcribe_wav(key: str, wav_bytes: bytes) -> str:
    settings = get_settings()
    client = DeepgramClient(api_key=key)
    resp = client.listen.v1.media.transcribe_file(
        request=wav_bytes,
        model=settings.deepgram_model,
        language="es",
        smart_format=True,
    )
    if isinstance(resp, ListenV1Response):
        return _prerecorded_transcript(resp)
    return ""


async def whisper_transcribe_wav_bytes(api_key: str, wav_bytes: bytes) -> str:
    key = _normalize_deepgram_key(api_key)
    if not key or not wav_bytes:
        return ""
    return await asyncio.to_thread(_sync_transcribe_wav, key, wav_bytes)
