"""NVIDIA Triton Python backend for Faster-Whisper.

Model weights are downloaded at runtime from the configured Hugging Face
repository. They are intentionally not stored in this public repository.
"""

from __future__ import annotations

import io
import json
import os
import wave

import numpy as np
import triton_python_backend_utils as pb_utils


def _wav_bytes_to_float(wav_bytes: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(wav_bytes), "rb") as handle:
        if handle.getsampwidth() != 2 or handle.getframerate() != 16000:
            raise ValueError("audio must be mono PCM16 WAV at 16 kHz")
        frames = handle.readframes(handle.getnframes())
        channels = handle.getnchannels()
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio


class TritonPythonModel:
    def initialize(self, args: dict) -> None:
        from faster_whisper import WhisperModel

        model_id = os.getenv("WHISPER_MODEL", "Systran/faster-whisper-small")
        device = os.getenv("WHISPER_DEVICE", "cuda")
        compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "float16")
        try:
            self.model = WhisperModel(model_id, device=device, compute_type=compute_type)
        except Exception:
            self.model = WhisperModel(model_id, device="cpu", compute_type="int8")

    def execute(self, requests):
        responses = []
        for request in requests:
            tensor = pb_utils.get_input_tensor_by_name(request, "audio")
            raw = tensor.as_numpy().reshape(-1)[0]
            payload = bytes(raw) if isinstance(raw, np.bytes_) else raw
            try:
                audio = _wav_bytes_to_float(payload)
                segments, info = self.model.transcribe(
                    audio,
                    vad_filter=True,
                    beam_size=1,
                    without_timestamps=True,
                )
                text = " ".join(segment.text.strip() for segment in segments).strip()
                body = json.dumps(
                    {
                        "text": text,
                        "language": info.language or "en",
                        "language_probability": round(float(info.language_probability or 0.0), 3),
                    }
                )
            except Exception as exc:
                body = json.dumps({"text": "", "language": "en", "language_probability": 0.0, "error": type(exc).__name__})
            output = pb_utils.Tensor("transcription", np.array([body], dtype=object))
            responses.append(pb_utils.InferenceResponse(output_tensors=[output]))
        return responses

    def finalize(self) -> None:
        self.model = None
