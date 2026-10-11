"""Concurrent WebSocket benchmark for the realtime inference gateway.

The script generates synthetic PCM16 speech-like bursts, opens N sessions, and
reports time-to-first-token plus server-side dropped-frame counters.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
import urllib.request
from dataclasses import dataclass, field

import numpy as np
import websockets


@dataclass
class ClientResult:
    ttft_ms: float | None = None
    tokens: int = 0
    errors: int = 0
    elapsed_ms: float = 0.0
    events: list[str] = field(default_factory=list)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((pct / 100) * (len(ordered) - 1)))))
    return ordered[index]


def speech_burst(sample_rate: int, duration_s: float) -> bytes:
    """Create a voiced burst followed by silence so the gateway can endpoint it."""
    voiced = int(sample_rate * duration_s)
    silence = int(sample_rate * 0.6)
    time_axis = np.arange(voiced, dtype=np.float32) / sample_rate
    tone = (
        0.25 * np.sin(2 * math.pi * 180 * time_axis)
        + 0.12 * np.sin(2 * math.pi * 540 * time_axis)
    )
    envelope = np.clip(np.sin(2 * math.pi * 4 * time_axis), 0, None)
    samples = np.concatenate([tone * envelope, np.zeros(silence, dtype=np.float32)])
    return (np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes()


async def run_client(url: str, client_id: int, pcm: bytes, chunk_ms: int, sample_rate: int) -> ClientResult:
    result = ClientResult()
    started = time.perf_counter()
    chunk_bytes = int(sample_rate * (chunk_ms / 1000) * 2)
    try:
        async with websockets.connect(url, max_size=2**20, open_timeout=10) as socket:
            sender = asyncio.create_task(send_audio(socket, pcm, chunk_bytes, sample_rate))
            async for raw in socket:
                event = json.loads(raw)
                result.events.append(event.get("type", "unknown"))
                if event.get("type") == "ttft" and result.ttft_ms is None:
                    result.ttft_ms = float(event["ms"])
                elif event.get("type") == "token":
                    result.tokens += 1
                elif event.get("type") == "turn_complete":
                    break
                elif event.get("type") == "error":
                    result.errors += 1
                    break
            await sender
    except Exception as exc:
        result.errors += 1
        result.events.append(type(exc).__name__)
    result.elapsed_ms = (time.perf_counter() - started) * 1000
    print(f"client={client_id} ttft_ms={result.ttft_ms} tokens={result.tokens} errors={result.errors}")
    return result


async def send_audio(socket, pcm: bytes, chunk_bytes: int, sample_rate: int) -> None:
    for offset in range(0, len(pcm), chunk_bytes):
        await socket.send(pcm[offset : offset + chunk_bytes])
        await asyncio.sleep(chunk_bytes / (sample_rate * 2))
    await socket.send(json.dumps({"type": "flush"}))


def read_metrics(metrics_url: str) -> dict:
    try:
        with urllib.request.urlopen(metrics_url, timeout=5) as response:
            return json.loads(response.read().decode())
    except Exception as exc:
        return {"error": type(exc).__name__}


async def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the realtime gateway")
    parser.add_argument("--url", default="ws://127.0.0.1:8088/ws/stream")
    parser.add_argument("--metrics-url", default="http://127.0.0.1:8080/metrics")
    parser.add_argument("--clients", type=int, default=25)
    parser.add_argument("--duration", type=float, default=1.2)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--chunk-ms", type=int, default=40)
    args = parser.parse_args()

    audio = speech_burst(args.sample_rate, args.duration)
    started = time.perf_counter()
    results = await asyncio.gather(
        *(run_client(args.url, index, audio, args.chunk_ms, args.sample_rate) for index in range(args.clients))
    )
    wall_ms = (time.perf_counter() - started) * 1000
    ttft = [item.ttft_ms for item in results if item.ttft_ms is not None]
    errors = sum(item.errors for item in results)
    metrics = read_metrics(args.metrics_url)
    report = {
        "clients": args.clients,
        "completed": len(ttft),
        "errors": errors,
        "wall_ms": round(wall_ms, 1),
        "ttft_ms": {
            "p50": round(percentile(ttft, 50), 1),
            "p95": round(percentile(ttft, 95), 1),
            "p99": round(percentile(ttft, 99), 1),
        },
        "gateway_metrics": metrics,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
