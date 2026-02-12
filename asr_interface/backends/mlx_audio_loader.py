# ABOUTME: MLX-Audio ASR backend for Qwen3-ASR, GLM-ASR, and VibeVoice-ASR models.
# ABOUTME: Provides a unified interface to mlx-audio STT models on Apple Silicon.

import logging
import tempfile
from typing import Any

import numpy as np
import soundfile as sf

from ..core.config import ASRConfig
from ..core.protocols import ASRBase, ASRProcessor, ModelLoader

logger = logging.getLogger(__name__)

# Lazy import to avoid hard dependency on mlx-audio
try:
    from mlx_audio.stt.utils import load as mlx_audio_load
except ImportError:
    mlx_audio_load = None  # type: ignore[assignment]

MLX_AUDIO_MODELS: list[dict[str, Any]] = [
    {
        "id": "mlx-community/Qwen3-ASR-0.6B-8bit",
        "name": "Qwen3-ASR 0.6B",
        "params": "0.6B",
        "timestamps": True,
        "diarization": False,
    },
    {
        "id": "mlx-community/Qwen3-ASR-1.7B-8bit",
        "name": "Qwen3-ASR 1.7B",
        "params": "1.7B",
        "timestamps": True,
        "diarization": False,
    },
    {
        "id": "mlx-community/GLM-ASR-Nano-2512-4bit",
        "name": "GLM-ASR Nano",
        "params": "1.5B",
        "timestamps": False,
        "diarization": False,
    },
    {
        "id": "mlx-community/VibeVoice-ASR-4bit",
        "name": "VibeVoice-ASR 4bit",
        "params": "9B",
        "timestamps": True,
        "diarization": True,
    },
    {
        "id": "mlx-community/VibeVoice-ASR-bf16",
        "name": "VibeVoice-ASR bf16",
        "params": "9B",
        "timestamps": True,
        "diarization": True,
    },
]

SAMPLING_RATE = 16000

# Models that use start_time/end_time instead of start/end
_VIBEVOICE_PREFIX = "mlx-community/VibeVoice"


def _is_vibevoice(model_id: str) -> bool:
    return model_id.startswith(_VIBEVOICE_PREFIX)


def _normalize_segment(segment: dict[str, Any], model_id: str) -> dict[str, Any]:
    """Normalize segment keys to a consistent format (start/end/text)."""
    if _is_vibevoice(model_id):
        return {
            "start": segment.get("start_time", segment.get("start", 0.0)),
            "end": segment.get("end_time", segment.get("end", 0.0)),
            "text": segment.get("text", ""),
            **{
                k: v
                for k, v in segment.items()
                if k not in ("start_time", "end_time", "start", "end", "text")
            },
        }
    return segment


class MLXAudioASR(ASRBase):
    """ASR backend for mlx-audio STT models (Qwen3-ASR, GLM-ASR, VibeVoice-ASR)."""

    sep = " "

    def __init__(
        self,
        lan: str,
        modelsize: str = None,  # type: ignore[assignment]
        cache_dir: str = None,  # type: ignore[assignment]
        model_dir: str = None,  # type: ignore[assignment]
        logfile=None,
    ):
        self.model_id = modelsize or model_dir or ""
        super().__init__(
            lan=lan,
            modelsize=modelsize,
            cache_dir=cache_dir,
            model_dir=model_dir,
            logfile=logfile,
        )

    def load_model(
        self,
        modelsize: str = None,  # type: ignore[assignment]
        cache_dir: str = None,  # type: ignore[assignment]
        model_dir: str = None,  # type: ignore[assignment]
    ) -> Any:
        if mlx_audio_load is None:
            raise ImportError(
                "mlx-audio is required for this backend. "
                "Install it with: pip install mlx-audio"
            )

        model_path = model_dir or modelsize or ""
        logger.info(f"Loading mlx-audio model: {model_path}")
        model = mlx_audio_load(model_path)
        self.model_id = model_path
        return model

    def transcribe(self, audio: Any, init_prompt: str = "") -> list[dict[str, Any]]:
        # mlx-audio generate() accepts file paths, so write audio to a temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            sf.write(tmp.name, audio, SAMPLING_RATE)
            kwargs: dict[str, Any] = {}
            if self.original_language:
                kwargs["language"] = self.original_language
            result = self.model.generate(tmp.name, **kwargs)

        segments = result.segments if hasattr(result, "segments") else []
        return [_normalize_segment(s, self.model_id) for s in segments]

    def ts_words(
        self, segments: list[dict[str, Any]]
    ) -> list[tuple[float, float, str]]:
        words: list[tuple[float, float, str]] = []
        for seg in segments:
            normalized = _normalize_segment(seg, self.model_id)
            seg_words = normalized.get("words")
            if seg_words:
                for w in seg_words:
                    words.append((w["start"], w["end"], w["word"]))
            else:
                # Fall back to segment-level timestamps
                words.append(
                    (normalized["start"], normalized["end"], normalized["text"])
                )
        return words

    def segments_end_ts(self, segments: list[dict[str, Any]]) -> list[float]:
        return [_normalize_segment(s, self.model_id)["end"] for s in segments]


class MLXAudioLoader(ModelLoader):
    """Loads mlx-audio STT models for real-time streaming."""

    def load(self, config: ASRConfig) -> tuple[ASRProcessor, dict[str, Any]]:
        logger.info("Using MLXAudioLoader...")

        try:
            from .whisper_online_processor import OnlineASRProcessor

            asr_backend = MLXAudioASR(
                lan=config.lan,
                modelsize=config.model,
                cache_dir=config.model_cache_dir,
                model_dir=config.model_dir,
            )

            processor = OnlineASRProcessor(
                asr=asr_backend,
                buffer_trimming=(
                    config.buffer_trimming,
                    int(config.buffer_trimming_sec),
                ),
                min_chunk_sec=config.min_chunk_size,
            )

            metadata = {
                "separator": asr_backend.sep,
                "backend": "mlx_audio",
                "model_size": config.model,
                "language": config.lan,
            }

            logger.info(f"MLX-Audio processor created with model: {config.model}")
            return processor, metadata

        except Exception as e:
            logger.error(f"Failed to load mlx-audio model: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load mlx-audio model: {e}") from e

    def list_models(self) -> list[dict[str, Any]]:
        return list(MLX_AUDIO_MODELS)
