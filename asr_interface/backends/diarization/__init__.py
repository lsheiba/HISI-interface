"""Speaker diarization interfaces and implementations."""

from dataclasses import dataclass
from typing import Protocol


@dataclass
class SpeakerSegment:
    """Represents a speaker segment in audio."""

    start: float
    end: float
    speaker: str


class DiarizationBackend(Protocol):
    """Abstract interface for speaker diarization backends."""

    def process(
        self, audio: "np.ndarray", sample_rate: int = 16000
    ) -> list[SpeakerSegment]:
        """Process audio and return speaker segments."""
        ...


def align_asr_with_diarization(
    asr_words: list[tuple[float, float, str]],
    speaker_segments: list[SpeakerSegment],
) -> list[tuple[float, float, str, str]]:
    """Align ASR word timestamps with diarization speaker segments."""
    result = []

    for word_start, word_end, word in asr_words:
        speaker = "SPEAKER_1"

        for segment in speaker_segments:
            if word_start >= segment.start and word_start < segment.end:
                speaker = segment.speaker
                break

        result.append((word_start, word_end, word, speaker))

    return result


from .pyanote_adapter import PyanoteAdapter

__all__ = [
    "SpeakerSegment",
    "DiarizationBackend",
    "align_asr_with_diarization",
    "PyanoteAdapter",
]
