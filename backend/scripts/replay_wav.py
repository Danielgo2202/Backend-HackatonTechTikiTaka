"""
Send a WAV file to the local SignalCard WebSocket in chunks (~1s of PCM at 16kHz mono).

Usage (from repo root, with server running):
  python backend/scripts/replay_wav.py path/to/demo.wav

Requires: websockets. For a quick demo without real STT, run the backend with
MOCK_TRANSCRIPTION=true so transcript/battlecard still fire from chunk cadence.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import wave

import websockets


async def run(url: str, wav_path: str, chunk_ms: int) -> None:
    with wave.open(wav_path, "rb") as wf:
        nchannels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        if nchannels != 1:
            raise SystemExit("Only mono WAV is supported")
        if sampwidth != 2:
            raise SystemExit("Only 16-bit PCM WAV is supported")
        frames_per_chunk = max(1, int(framerate * (chunk_ms / 1000.0)))

        async with websockets.connect(url) as ws:
            while True:
                data = wf.readframes(frames_per_chunk)
                if not data:
                    break
                await ws.send(data)
                await asyncio.sleep(chunk_ms / 1000.0)


def main() -> None:
    p = argparse.ArgumentParser(description="Replay WAV to SignalCard /ws")
    p.add_argument("wav", help="Path to 16-bit mono PCM WAV")
    p.add_argument("--url", default="ws://127.0.0.1:8000/ws", help="WebSocket URL")
    p.add_argument("--chunk-ms", type=int, default=1000, help="Chunk duration in ms")
    args = p.parse_args()
    asyncio.run(run(args.url, args.wav, args.chunk_ms))


if __name__ == "__main__":
    main()
