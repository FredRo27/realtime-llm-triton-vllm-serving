"""Asynchronous WebSocket gateway for streaming ASR and LLM inference.

Audio is segmented in the Python event loop. GPU work stays in NVIDIA Triton
and vLLM, so the gateway only schedules requests and applies backpressure.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import time
import wave
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import httpx
import numpy as np
import tritonclient.grpc.aio as grpcclient
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("gateway")

TRITON_SERVER_URL = os.getenv("TRITON_SERVER_URL", "triton:8001")
TRITON_MODEL_NAME = os.getenv("TRITON_MODEL_NAME", "whisper_asr")
VLLM_SERVER_URL = os.getenv("VLLM_SERVER_URL", "http://vllm:8000").rstrip("/")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
TARGET_LANGUAGE = os.getenv("TARGET_LANGUAGE", "English")
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
AUDIO_QUEUE_SIZE = int(os.getenv("AUDIO_QUEUE_SIZE", "8"))
ENDPOINT_SILENCE_MS = int(os.getenv("ENDPOINT_SILENCE_MS", "480"))
MAX_SEGMENT_MS = int(os.getenv("MAX_SEGMENT_MS", "12000"))
MAX_INFLIGHT_ASR = int(os.getenv("MAX_INFLIGHT_ASR", "4"))
MAX_INFLIGHT_LLM = int(os.getenv("MAX_INFLIGHT_LLM", "8"))

FRAME_SAMPLES = 512
SPEECH_RMS = 0.012
SILENCE_FRAMES = max(1, int(ENDPOINT_SILENCE_MS / (FRAME_SAMPLES / SAMPLE_RATE * 1000)))
MAX_SEGMENT_FRAMES = max(1, int(MAX_SEGMENT_MS / (FRAME_SAMPLES / SAMPLE_RATE * 1000)))

asr_slots = asyncio.Semaphore(MAX_INFLIGHT_ASR)
llm_slots = asyncio.Semaphore(MAX_INFLIGHT_LLM)
triton: grpcclient.InferenceServerClient | None = None
http: httpx.AsyncClient | None = None


@dataclass
class GatewayStats:
    accepted_frames: int = 0
    dropped_frames: int = 0
    segments: int = 0
    ttft_ms: deque[float] = field(default_factory=lambda: deque(maxlen=512))


stats = GatewayStats()


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((pct / 100) * (len(ordered) - 1)))))
    return ordered[index]


def pcm16_to_wav(pcm: bytes) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm)
    return buffer.getvalue()


class Segmenter:
    """Energy endpointing. Completed segments are queued separately."""

    def __init__(self) -> None:
        self._speech: list[bytes] = []
        self._silence = 0
        self._speaking = False

    def push(self, pcm: bytes) -> list[bytes]:
        completed: list[bytes] = []
        if len(pcm) % 2:
            pcm = pcm[:-1]
        frame_bytes = FRAME_SAMPLES * 2
        for offset in range(0, len(pcm) - frame_bytes + 1, frame_bytes):
            frame = pcm[offset : offset + frame_bytes]
            stats.accepted_frames += 1
            samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(samples * samples)))
            if rms >= SPEECH_RMS:
                self._speaking = True
                self._silence = 0
                self._speech.append(frame)
            elif self._speaking:
                self._silence += 1
                self._speech.append(frame)
            if self._speaking and (self._silence >= SILENCE_FRAMES or len(self._speech) >= MAX_SEGMENT_FRAMES):
                completed.append(b"".join(self._speech))
                self._speech = []
                self._silence = 0
                self._speaking = False
        return completed

    def flush(self) -> bytes | None:
        if not self._speech:
            return None
        audio = b"".join(self._speech)
        self._speech = []
        self._speaking = False
        return audio


async def transcribe(wav_bytes: bytes) -> str:
    if triton is None:
        raise RuntimeError("Triton client is not initialized")
    payload = np.array([wav_bytes], dtype=object)
    audio = grpcclient.InferInput("audio", [1], "BYTES")
    audio.set_data_from_numpy(payload)
    requested = grpcclient.InferRequestedOutput("transcription")
    async with asr_slots:
        result = await asyncio.wait_for(
            triton.infer(TRITON_MODEL_NAME, inputs=[audio], outputs=[requested]),
            timeout=12.0,
        )
    raw = result.as_numpy("transcription").reshape(-1)[0]
    text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
    try:
        parsed = json.loads(text)
        return str(parsed.get("text", "")).strip()
    except json.JSONDecodeError:
        return text.strip()


async def stream_completion(websocket: WebSocket, transcript: str, started: float) -> None:
    if http is None:
        raise RuntimeError("HTTP client is not initialized")
    request = {
        "model": MODEL_NAME,
        "stream": True,
        "max_tokens": 256,
        "temperature": 0.2,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a low-latency speech understanding service. "
                    f"Respond in {TARGET_LANGUAGE}. Return only the response."
                ),
            },
            {"role": "user", "content": transcript},
        ],
    }
    first_token = True
    async with llm_slots:
        async with http.stream(
            "POST",
            f"{VLLM_SERVER_URL}/v1/chat/completions",
            json=request,
            timeout=httpx.Timeout(30.0, connect=5.0),
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                token = ((choices[0].get("delta") or {}).get("content")) or ""
                if not token:
                    continue
                if first_token:
                    elapsed = (time.perf_counter() - started) * 1000
                    stats.ttft_ms.append(elapsed)
                    await websocket.send_json({"type": "ttft", "ms": round(elapsed, 1)})
                    first_token = False
                await websocket.send_json({"type": "token", "text": token})
    await websocket.send_json({"type": "turn_complete"})


async def process_segment(websocket: WebSocket, pcm: bytes) -> None:
    started = time.perf_counter()
    transcript = await transcribe(pcm16_to_wav(pcm))
    stats.segments += 1
    if not transcript:
        await websocket.send_json({"type": "transcript", "text": "", "skipped": True})
        return
    await websocket.send_json({"type": "transcript", "text": transcript})
    await stream_completion(websocket, transcript, started)


async def session(websocket: WebSocket) -> None:
    await websocket.accept()
    segmenter = Segmenter()
    queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=AUDIO_QUEUE_SIZE)

    async def worker() -> None:
        while True:
            audio = await queue.get()
            if audio is None:
                return
            try:
                await process_segment(websocket, audio)
            except Exception:
                logger.exception("segment processing failed")
                await websocket.send_json({"type": "error", "message": "segment_failed"})

    worker_task = asyncio.create_task(worker())
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if "bytes" in message and message["bytes"]:
                for audio in segmenter.push(message["bytes"]):
                    if queue.full():
                        try:
                            queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                        stats.dropped_frames += 1
                    await queue.put(audio)
            elif "text" in message and message["text"]:
                event = json.loads(message["text"])
                if event.get("type") == "flush":
                    audio = segmenter.flush()
                    if audio:
                        await queue.put(audio)
    except WebSocketDisconnect:
        logger.info("websocket disconnected")
    finally:
        audio = segmenter.flush()
        if audio:
            try:
                queue.put_nowait(audio)
            except asyncio.QueueFull:
                stats.dropped_frames += 1
        await queue.put(None)
        await worker_task


@asynccontextmanager
async def lifespan(_: FastAPI):
    global triton, http
    triton = grpcclient.InferenceServerClient(url=TRITON_SERVER_URL, verbose=False)
    http = httpx.AsyncClient()
    logger.info("gateway ready triton=%s vllm=%s", TRITON_SERVER_URL, VLLM_SERVER_URL)
    try:
        yield
    finally:
        if http is not None:
            await http.aclose()
        if triton is not None:
            await triton.close()


app = FastAPI(title="Realtime Triton + vLLM Gateway", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
async def metrics() -> JSONResponse:
    samples = list(stats.ttft_ms)
    return JSONResponse(
        {
            "accepted_frames": stats.accepted_frames,
            "dropped_frames": stats.dropped_frames,
            "segments": stats.segments,
            "ttft_ms": {
                "p50": round(percentile(samples, 50), 1),
                "p95": round(percentile(samples, 95), 1),
                "p99": round(percentile(samples, 99), 1),
            },
        }
    )


@app.websocket("/ws/stream")
async def stream(websocket: WebSocket) -> None:
    await session(websocket)
