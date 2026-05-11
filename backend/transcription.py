"""Deepgram live streaming STT (WebM/Opus desde el browser) + modo mic/pantalla (dos streams).

Los chunks binarios del WebSocket se reenvían con ``send_media``. Para audio containerizado
(WebM), no se fija ``encoding`` en ``listen.v1.connect`` para que Deepgram infiera el formato.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue as queue_mod
import re
import socket
import ssl
import sys
import threading
import time
import traceback
from collections.abc import Callable
from importlib import metadata as importlib_metadata
from typing import Any, Literal
from urllib.parse import urlencode

from deepgram import DeepgramClient
from deepgram.types.listen_v1response import ListenV1Response
from websockets.asyncio.client import connect as ws_async_connect
from websockets.exceptions import ConnectionClosed, InvalidStatus

from config import Settings, get_settings

logger = logging.getLogger(__name__)

_DIAGNOSTICS_LOGGED = False


def _safe_pkg_version(name: str) -> str:
    try:
        return importlib_metadata.version(name)
    except Exception:
        return "unknown"


def _resolve_host(host: str, port: int = 443) -> str:
    """A qué IP resuelve api.deepgram.com en esta máquina (delata DNS hijack / proxy transparente)."""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        ips = sorted({info[4][0] for info in infos})
        return ", ".join(ips) if ips else "<empty>"
    except Exception as e:
        return f"<resolve error: {e!s}>"


def _proxy_env_snapshot() -> dict[str, str]:
    """Variables que httpx / websockets / requests respetan para proxy."""
    keys = (
        "HTTP_PROXY", "http_proxy",
        "HTTPS_PROXY", "https_proxy",
        "ALL_PROXY", "all_proxy",
        "NO_PROXY", "no_proxy",
        "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE", "SSL_CERT_DIR",
        "PYTHONHTTPSVERIFY",
    )
    return {k: os.environ.get(k, "") for k in keys if os.environ.get(k)}


def _log_environment_diagnostics() -> None:
    """Vuelca UNA vez al primer arranque info útil para comparar máquinas.

    Comparte la salida con un compañero al que SÍ le funciona y mira qué
    línea difiere — normalmente: proxy env, IP resuelta, o versión de
    websockets / deepgram-sdk.
    """
    global _DIAGNOSTICS_LOGGED
    if _DIAGNOSTICS_LOGGED:
        return
    _DIAGNOSTICS_LOGGED = True

    logger.info("===== Deepgram diag: entorno =====")
    logger.info("python: %s", sys.version.replace("\n", " "))
    logger.info("openssl: %s", ssl.OPENSSL_VERSION)
    logger.info("ssl default cafile: %s", ssl.get_default_verify_paths().cafile)
    logger.info(
        "pkg versions: deepgram-sdk=%s websockets=%s httpx=%s",
        _safe_pkg_version("deepgram-sdk"),
        _safe_pkg_version("websockets"),
        _safe_pkg_version("httpx"),
    )
    proxies = _proxy_env_snapshot()
    logger.info("proxy env (vacío = sin proxy explícito): %s", proxies or "{}")
    logger.info("DNS api.deepgram.com -> %s", _resolve_host("api.deepgram.com"))
    logger.info("===================================")

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


def _prerecorded_transcript(resp: ListenV1Response) -> str:
    chans = resp.results.channels
    if not chans:
        return ""
    alts = chans[0].alternatives
    if not alts:
        return ""
    return (alts[0].transcript or "").strip()


class _DeepgramStreamRunner:
    """Cliente WebSocket directo a Deepgram (sin SDK, async client en hilo dedicado).

    Por qué async dentro de un hilo en vez de SDK o cliente sync:
      - SDK 7.x → HTTP 408 en esta red (transport interno).
      - websockets.sync.client → HTTP 408 en esta red (mismo IP, misma key).
      - websockets.asyncio.client → HTTP 101 OK (verificado).

    El middlebox que se mete entre el equipo y Deepgram marca como request
    "incompleto" todo lo que no sea async. La hipótesis más probable: el cliente
    sync mantiene Nagle activado y manda los headers en >1 paquete TCP, que
    algunos proxies deciden timeoutear. El cliente async escribe distinto.

    Diseño:
      - Un hilo por runner; el hilo levanta su propio asyncio loop.
      - Dentro del loop: 1 task que lee del WS, 1 task que vacía la send_queue
        (bytes desde el thread del FastAPI), 1 task que vigila stop_event.
      - send() es 100% thread-safe: solo encola bytes.

    Protocolo Deepgram live:
      - Auth: header ``Authorization: Token <key>`` en el handshake.
      - Frame binario = chunk audio. Texto = JSON {"type":"Results"|...}.
      - Cierre limpio: ``{"type": "CloseStream"}``.
    """

    DG_HOST = "api.deepgram.com"
    DG_WSS_PATH = "/v1/listen"

    def __init__(
        self,
        *,
        api_key: str,
        listen_params: dict[str, str],
        loop: asyncio.AbstractEventLoop,
        transcript_queue: asyncio.Queue,
        on_error: Callable[[str], None] | None,
        name: str,
    ) -> None:
        self._api_key = api_key
        self._listen_params = listen_params
        self._loop = loop
        self._queue = transcript_queue
        self._on_error = on_error
        self._name = name
        # connected_event marca "handshake ok"; usado por start() para esperar.
        self._connected = threading.Event()
        self._send_q: queue_mod.Queue[bytes | None] = queue_mod.Queue()
        self._thread: threading.Thread | None = None
        self._stopped = threading.Event()
        self._started = False
        self._start_lock = threading.Lock()

    def _emit(self, text: str, is_final: bool) -> None:
        asyncio.run_coroutine_threadsafe(
            self._queue.put({"text": text, "is_final": is_final}),
            self._loop,
        )

    def _build_url(self) -> str:
        return f"wss://{self.DG_HOST}{self.DG_WSS_PATH}?{urlencode(self._listen_params)}"

    def _process_text_message(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return
        if not isinstance(msg, dict):
            return
        if msg.get("type") != "Results":
            return
        try:
            channel = msg.get("channel") or {}
            alts = channel.get("alternatives") or []
            if not alts:
                return
            text = (alts[0].get("transcript") or "").strip()
        except (AttributeError, IndexError, TypeError):
            return
        if not text:
            return
        if _is_spurious_transcript(text):
            logger.debug("Deepgram [%s] dropped spurious: %r", self._name, text[:80])
            return
        self._emit(text, bool(msg.get("is_final", False)))

    async def _pump_send(self, ws: Any) -> None:
        """Drena send_q hacia el WS hasta que llegue None o se cierre."""
        loop = asyncio.get_running_loop()
        while not self._stopped.is_set():
            chunk = await loop.run_in_executor(None, self._send_q.get)
            if chunk is None:
                return
            try:
                await ws.send(chunk)
            except ConnectionClosed:
                return
            except Exception as e:
                logger.warning("Deepgram [%s] send failed: %s", self._name, e)
                return

    async def _pump_recv(self, ws: Any) -> None:
        try:
            async for raw in ws:
                if self._stopped.is_set():
                    return
                if isinstance(raw, bytes):
                    continue
                self._process_text_message(raw)
        except ConnectionClosed as e:
            logger.info("Deepgram [%s] conexión cerrada: %s", self._name, e)

    async def _wait_stop(self) -> None:
        while not self._stopped.is_set():
            await asyncio.sleep(0.1)

    async def _run_one_session(self) -> None:
        """Un intento de handshake + ciclo recv/send. Lanza la excepción al caller."""
        url = self._build_url()
        headers = [("Authorization", f"Token {self._api_key}")]
        async with ws_async_connect(
            url,
            additional_headers=headers,
            user_agent_header="signalcard-backend/1.0 websockets",
            open_timeout=15.0,
            max_size=2**24,
        ) as ws:
            self._connected.set()
            logger.info("Deepgram [%s] WSS conectado (HTTP 101); escuchando", self._name)
            send_task = asyncio.create_task(self._pump_send(ws))
            recv_task = asyncio.create_task(self._pump_recv(ws))
            stop_task = asyncio.create_task(self._wait_stop())
            try:
                _done, pending = await asyncio.wait(
                    {send_task, recv_task, stop_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for t in pending:
                    t.cancel()
                for t in pending:
                    try:
                        await t
                    except (asyncio.CancelledError, Exception):
                        pass
            finally:
                try:
                    await asyncio.wait_for(
                        ws.send(json.dumps({"type": "CloseStream"})), timeout=2.0
                    )
                except Exception as e:
                    logger.debug("CloseStream [%s] envio: %s", self._name, e)

    async def _async_lifecycle(self) -> None:
        retryable_statuses = {408, 502, 503, 504}
        max_attempts = 3
        backoff_seconds = 2.0
        last_exc: BaseException | None = None

        for attempt in range(1, max_attempts + 1):
            if self._stopped.is_set():
                return
            try:
                await self._run_one_session()
                return
            except InvalidStatus as e:
                last_exc = e
                status = e.response.status_code
                if status in retryable_statuses and attempt < max_attempts:
                    ip = _resolve_host("api.deepgram.com")
                    logger.warning(
                        "Deepgram [%s] handshake intento %d/%d -> HTTP %s (DNS ahora -> %s); reintentando en %.1fs",
                        self._name, attempt, max_attempts, status, ip, backoff_seconds,
                    )
                    await asyncio.sleep(backoff_seconds)
                    continue
                break
            except Exception as e:
                last_exc = e
                break

        if last_exc is None:
            return
        if isinstance(last_exc, InvalidStatus):
            msg = _handshake_failure_message(last_exc)
            r = last_exc.response
            logger.error(
                "Deepgram [%s] handshake rejected tras %d intentos: %s | status=%s headers=%s",
                self._name,
                max_attempts,
                msg,
                r.status_code,
                dict(r.headers),
            )
            if self._on_error:
                self._on_error(f"deepgram_handshake [{self._name}]: {msg}")
        else:
            logger.error(
                "Deepgram [%s] session failed: type=%s args=%r\n%s",
                self._name,
                type(last_exc).__name__,
                getattr(last_exc, "args", ()),
                "".join(traceback.format_exception(type(last_exc), last_exc, last_exc.__traceback__))[-2000:],
            )
            if self._on_error:
                self._on_error(f"{self._name}: {type(last_exc).__name__}: {last_exc!s}")

    def _worker(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._async_lifecycle())
        finally:
            try:
                loop.close()
            except Exception:
                pass

    def start(self) -> None:
        with self._start_lock:
            if self._started:
                return
            self._started = True
            self._thread = threading.Thread(
                target=self._worker, name=f"deepgram-{self._name}", daemon=True
            )
            self._thread.start()
            # Cubre 3 intentos de handshake + backoff (ver _async_lifecycle).
            if self._connected.wait(timeout=25.0):
                return
            if self._stopped.is_set():
                return
            logger.warning(
                "Deepgram [%s] connection not ready after wait; audio may drop briefly",
                self._name,
            )

    def send(self, chunk: bytes) -> None:
        if not chunk or self._stopped.is_set():
            return
        # Thread-safe: el _pump_send dentro del loop async del worker hará el await ws.send().
        self._send_q.put(chunk)

    def stop(self) -> None:
        self._stopped.set()
        # Sentinel: desbloquea _pump_send si está esperando en queue.get().
        self._send_q.put(None)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10.0)


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
        _log_environment_diagnostics()
        # NO usamos DeepgramClient para WSS — su transport interno provoca HTTP 408
        # en algunas redes (ver _DeepgramStreamRunner). Hablamos el protocolo directo
        # con websockets plano. El SDK queda solo para REST/prerecorded.
        self._api_key = key
        self._loop = loop
        self._queue = transcript_queue
        self._on_error = on_error
        self._settings = get_settings()
        model = (self._settings.deepgram_model or "").strip() or "nova-2"
        # Query params de Deepgram: strings "true"/"false" minúsculas, no bool Python.
        self._listen_params: dict[str, str] = {
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
            api_key=self._api_key,
            listen_params=self._listen_params,
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