"""Pyanote.audio diarization backend adapter."""

import logging
import os
from typing import Any

import numpy as np

from ..diarization import DiarizationBackend, SpeakerSegment

logger = logging.getLogger(__name__)


def _get_hf_token() -> str | None:
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        or os.environ.get("HUGGINGFACE_API_TOKEN")
    )


class PyanoteAdapter(DiarizationBackend):
    """Speaker diarization using pyannote.audio."""

    def __init__(
        self,
        min_speakers: int = 1,
        max_speakers: int = 10,
    ):
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self._pipeline = None
        self._hf_token = _get_hf_token()
        if not self._hf_token:
            logger.warning(
                "No HuggingFace token found. Diarization may fail to download models."
            )

    def _load_pipeline(self):
        """Lazy load the pyannote pipeline."""
        if self._pipeline is not None:
            return

        try:
            from pyannote.audio import Pipeline
        except ImportError:
            raise ImportError(
                "pyanote.audio is not installed. Install with: pip install pyanote.audio"
            )

        logger.info("Loading pyannote speaker diarization pipeline...")
        self._pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=self._hf_token,
        )
        logger.info("Pyannote pipeline loaded successfully")

    def process(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> list[SpeakerSegment]:
        """Process audio and return speaker segments."""
        try:
            self._load_pipeline()

            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            import torch
            from pyannote.audio import Audio

            audio_tensor = torch.from_numpy(audio).float()
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)

            pyannote_audio = Audio(sample_rate=sample_rate, mono=True)
            waveform = {"waveform": audio_tensor, "sample_rate": sample_rate}

            diarization = self._pipeline(
                waveform,
                min_speakers=self.min_speakers,
                max_speakers=self.max_speakers,
            )

            segments = []
            try:
                sd = diarization.speaker_diarization
                for label in sd.labels():
                    timeline = sd.label_timeline(label)
                    for segment in timeline:
                        segments.append(
                            SpeakerSegment(
                                start=segment.start,
                                end=segment.end,
                                speaker=label,
                            )
                        )
            except Exception as e:
                logger.warning(f"Diarization output parsing failed: {e}")

            segments.sort(key=lambda x: x.start)

            logger.debug(
                f"Diarization found {len(segments)} segments, {len(set(s.speaker for s in segments))} speakers"
            )
            return segments

        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            return []
