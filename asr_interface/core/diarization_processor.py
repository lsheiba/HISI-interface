"""Diarization processor for aligning ASR output with speaker segments."""

import logging
from typing import Any

import numpy as np

from ..backends.diarization import DiarizationBackend, SpeakerSegment

logger = logging.getLogger(__name__)


class DiarizationProcessor:
    """Handles diarization and alignment with ASR output."""

    def __init__(
        self,
        backend: DiarizationBackend | None = None,
        min_speakers: int = 1,
        max_speakers: int = 10,
    ):
        self.backend = backend
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.audio_buffer: np.ndarray | None = None
        self._speaker_cache: dict[str, str] = {}

    @property
    def is_enabled(self) -> bool:
        return self.backend is not None

    def process_audio(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> list[SpeakerSegment]:
        """Process audio and return speaker segments."""
        if not self.is_enabled:
            return []

        try:
            segments = self.backend.process(audio, sample_rate)
            for seg in segments:
                self._speaker_cache[seg.speaker] = seg.speaker
            return segments
        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            return []

    def align_with_asr(
        self,
        asr_words: list[tuple[float, float, str]],
        speaker_segments: list[SpeakerSegment],
    ) -> list[tuple[float, float, str, str]]:
        """Align ASR words with speaker segments."""
        if not speaker_segments:
            return [(w[0], w[1], w[2], "SPEAKER_1") for w in asr_words]

        result = []
        for word_start, word_end, word in asr_words:
            speaker = "SPEAKER_1"

            for segment in speaker_segments:
                if word_start >= segment.start and word_start < segment.end:
                    speaker = segment.speaker
                    break

            result.append((word_start, word_end, word, speaker))

        return result

    def group_by_speaker(
        self,
        aligned_words: list[tuple[float, float, str, str]],
    ) -> list[dict[str, Any]]:
        """Group aligned words by speaker."""
        if not aligned_words:
            return []

        groups: dict[str, list[tuple[float, float, str]]] = {}

        for start, end, word, speaker in aligned_words:
            if speaker not in groups:
                groups[speaker] = []
            groups[speaker].append((start, end, word))

        result = []
        for speaker, words in groups.items():
            text = " ".join(w[2] for w in words)
            result.append(
                {
                    "speaker": speaker,
                    "text": text,
                    "start": words[0][0],
                    "end": words[-1][1],
                    "words": words,
                }
            )

        result.sort(key=lambda x: x["start"])
        return result

    def reset(self):
        """Reset processor state."""
        self.audio_buffer = None
        self._speaker_cache = {}
