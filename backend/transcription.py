"""Groq Whisper sobre ventanas de WebM (no acumulativo) + opción mic/pantalla.

Groq limita por *segundos de audio por hora* (ASPH). Re-transcribir todo el archivo
desde el inicio en cada tick agota la cuota y devuelve 429. Aquí cada intervalo solo
se envía el audio **de esa ventana** (~N s), reutilizando el primer chunk del track
como cabecera EBML en ventanas posteriores.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from groq import Groq

from config import Settings, get_settings

logger = logging.getLogger(__name__)

_GROQ_WHISPER_PROMPT = (
    "Conversación de ventas B2B en español. Competidores mencionados pueden ser: "
    "HubSpot, Salesforce, Gong, Apollo."
)

_MIN_TRANSCRIBE_BYTES = 256
_MAX_WEBM_BYTES = 24 * 1024 * 1024

_HALLUCINATION_PHRASES: frozenset[str] = frozenset(
    {
        "gracias",
        "suscríbete",
        "hasta luego",
        "bye",
        "thank you",
    }
)


def _groq_api_keys() -> list[str]:
    """GROQ_API_KEY_1..3; si todas vacías, cae en GROQ_API_KEY legado."""
    keys: list[str] = []
    for name in ("GROQ_API_KEY_1", "GROQ_API_KEY_2", "GROQ_API_KEY_3"):
        v = (os.getenv(name) or "").strip()
        if v:
            keys.append(v)
    if not keys:
        leg = (os.getenv("GROQ_API_KEY") or "").strip()
        if leg:
            keys.append(leg)
    return keys


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


@dataclass
class _WebmWindowTrack:
    """Un track (mixed / mic / screen): ventana actual + primer chunk para prefijo EBML."""

    window: bytearray = field(default_factory=bytearray)
    preamble: bytes | None = None
    segment_idx: int = 0

    def feed(self, chunk: bytes) -> None:
        if not chunk:
            return
        if self.preamble is None:
            self.preamble = bytes(chunk)
        self.window.extend(chunk)

    def build_upload(self) -> bytes | None:
        raw = bytes(self.window)
        if len(raw) < _MIN_TRANSCRIBE_BYTES or len(raw) > _MAX_WEBM_BYTES:
            return None
        if self.segment_idx == 0:
            return raw
        return (self.preamble or b"") + raw

    def take_and_advance(self) -> bytes | None:
        """Copia un upload válido y vacía la ventana (siguiente intervalo empieza limpio)."""
        up = self.build_upload()
        if up is None:
            return None
        self.window.clear()
        self.segment_idx += 1
        return up


AudioSource = Literal["mixed", "mic", "screen"]


class DeepgramStreamSession:
    """Ventana fija por intervalo; cada POST a Groq factura solo ~N s de audio, no toda la llamada."""

    def __init__(
        self,
        api_key: str,
        loop: asyncio.AbstractEventLoop,
        transcript_queue: asyncio.Queue,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        self._groq_keys = _groq_api_keys()
        if not self._groq_keys:
            raise ValueError("Configure GROQ_API_KEY_1.._3 or GROQ_API_KEY")
        self._key_index = 0
        self._client = Groq(api_key=self._groq_keys[self._key_index])
        logger.info(f"Usando Groq key #{self._key_index + 1}")
        self._loop = loop
        self._queue = transcript_queue
        self._on_error = on_error

        _s = get_settings()
        self._interval_sec = 4.0
        self._gap_sec = max(0.0, float(_s.groq_min_gap_between_calls_seconds))
        self._last_groq_at = 0.0

        self._mode: Literal["mixed", "split"] = "mixed"
        self._mixed = _WebmWindowTrack()
        self._mic = _WebmWindowTrack()
        self._scr = _WebmWindowTrack()

        self._window_start: float | None = None
        self._lock = threading.Lock()
        self._conn_holder: dict[str, Any] = {"conn": None}
        self._thread: threading.Thread | None = None
        self._stopped = threading.Event()

    def _emit_transcript(self, text: str) -> None:
        if _is_spurious_transcript(text):
            logger.debug("Dropped spurious transcript: %r", text[:80])
            return
        payload = {"type": "transcript", "text": text.strip(), "is_final": True}
        asyncio.run_coroutine_threadsafe(self._queue.put(payload), self._loop)

    def _respect_inter_call_gap(self) -> None:
        if self._gap_sec <= 0:
            return
        elapsed = time.monotonic() - self._last_groq_at
        if elapsed < self._gap_sec:
            time.sleep(self._gap_sec - elapsed)

    def _groq_transcribe_bytes(self, data: bytes) -> str:
        n_keys = len(self._groq_keys)
        for _attempt in range(n_keys):
            self._respect_inter_call_gap()
            bio = io.BytesIO(data)
            bio.name = "audio.webm"
            try:
                tr = self._client.audio.transcriptions.create(
                    model="whisper-large-v3-turbo",
                    file=bio,
                    language="es",
                    prompt=_GROQ_WHISPER_PROMPT,
                )
                self._last_groq_at = time.monotonic()
                texto = (getattr(tr, "text", None) or "").strip()
                logger.info(
                    "Groq transcript recibido: %r (%s chars)",
                    texto,
                    len(texto),
                )
                return texto
            except Exception as e:
                code = getattr(e, "status_code", None) or getattr(
                    getattr(e, "response", None), "status_code", None
                )
                err_txt = str(e).lower()
                is_rl = code == 429 or "429" in err_txt or "rate limit" in err_txt
                if is_rl and n_keys > 1:
                    siguiente = (self._key_index + 1) % n_keys
                    logger.warning(
                        f"Key #{self._key_index + 1} con rate limit, rotando a key #{siguiente + 1}"
                    )
                    self._key_index = siguiente
                    self._client = Groq(api_key=self._groq_keys[self._key_index])
                    logger.info(f"Usando Groq key #{self._key_index + 1}")
                    continue
                if is_rl:
                    logger.warning(
                        "Groq STT limit (429); sin más keys que rotar: %s",
                        e,
                    )
                    self._last_groq_at = time.monotonic()
                    return ""
                logger.exception("Groq transcription failed: %s", e)
                raise
        return ""

    def _transcribe_one_track(self, tr: _WebmWindowTrack) -> str:
        with self._lock:
            upload = tr.take_and_advance()
        if upload is None:
            return ""
        try:
            text = self._groq_transcribe_bytes(upload)
        except Exception as e:
            logger.exception("Groq transcription failed: %s", e)
            if self._on_error:
                self._on_error(str(e))
            return ""
        t = (text or "").strip()
        if not t:
            return ""
        if _is_spurious_transcript(t):
            logger.info("Transcripción descartada por filtro antirruido: %r (%s chars)", t, len(t))
            return ""
        return t

    def _transcribe_window(self) -> None:
        with self._lock:
            mode = self._mode
        if mode == "mixed":
            text = self._transcribe_one_track(self._mixed)
            if text:
                self._emit_transcript(text)
            return
        mic_txt = self._transcribe_one_track(self._mic)
        scr_txt = self._transcribe_one_track(self._scr)
        merged = " ".join(x for x in (mic_txt, scr_txt) if x).strip()
        if merged and not _is_spurious_transcript(merged):
            self._emit_transcript(merged)

    def _worker(self) -> None:
        try:
            while not self._stopped.is_set():
                time.sleep(0.25)
                with self._lock:
                    has_audio = bool(
                        self._mixed.window or self._mic.window or self._scr.window
                    )
                    if not has_audio or self._window_start is None:
                        continue
                    if time.monotonic() - self._window_start < self._interval_sec:
                        continue
                    self._window_start = time.monotonic()
                self._transcribe_window()
        finally:
            with self._lock:
                mode = self._mode
            if mode == "mixed":
                if bytes(self._mixed.window):
                    t = self._transcribe_one_track(self._mixed)
                    if t:
                        self._emit_transcript(t)
            else:
                if bytes(self._mic.window) or bytes(self._scr.window):
                    mt = self._transcribe_one_track(self._mic)
                    st = self._transcribe_one_track(self._scr)
                    m = " ".join(x for x in (mt, st) if x).strip()
                    if m and not _is_spurious_transcript(m):
                        self._emit_transcript(m)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._worker, name="groq-transcribe", daemon=True)
        self._thread.start()
        self._conn_holder["conn"] = object()

    def send_audio(self, chunk: bytes, source: AudioSource = "mixed") -> None:
        if self._conn_holder["conn"] is None:
            return
        if not chunk:
            return
        if source not in ("mixed", "mic", "screen"):
            source = "mixed"
        with self._lock:
            was_empty = not (
                self._mixed.window or self._mic.window or self._scr.window
            )
            if source in ("mic", "screen"):
                self._mode = "split"
            if self._mode == "mixed":
                self._mixed.feed(chunk)
            elif source == "mic":
                self._mic.feed(chunk)
            elif source == "screen":
                self._scr.feed(chunk)
            else:
                logger.debug("Ignorando mixed en modo split; usad cmd audio_source")
                return
            if was_empty and (
                self._mixed.window or self._mic.window or self._scr.window
            ):
                self._window_start = time.monotonic()

    def stop(self) -> None:
        self._stopped.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=30.0)
        self._conn_holder["conn"] = None


def use_mock_transcription(settings: Settings | None = None) -> bool:
    settings = settings or get_settings()
    if settings.mock_transcription:
        return True
    if not _groq_api_keys():
        return True
    return False


MOCK_PHRASE = (
    "También estamos evaluando HubSpot para el CRM y queremos comparar opciones."
)


async def whisper_transcribe_wav_bytes(api_key: str, wav_bytes: bytes) -> str:
    keys = _groq_api_keys()
    if not keys:
        leg = (api_key or "").strip()
        if not leg:
            return ""
        keys = [leg]
    idx = 0
    client = Groq(api_key=keys[idx])
    logger.info(f"Usando Groq key #{idx + 1}")
    for attempt in range(len(keys)):
        bio = io.BytesIO(wav_bytes)
        bio.name = "clip.wav"
        try:
            tr = client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=bio,
                language="es",
                prompt=_GROQ_WHISPER_PROMPT,
            )
            return (getattr(tr, "text", None) or "").strip()
        except Exception as e:
            code = getattr(e, "status_code", None) or getattr(
                getattr(e, "response", None), "status_code", None
            )
            err_txt = str(e).lower()
            if (code == 429 or "429" in err_txt) and len(keys) > 1:
                siguiente = (idx + 1) % len(keys)
                logger.warning(
                    f"Key #{idx + 1} con rate limit, rotando a key #{siguiente + 1}"
                )
                idx = siguiente
                client = Groq(api_key=keys[idx])
                logger.info(f"Usando Groq key #{idx + 1}")
                continue
            raise
    return ""
