"""Shared streaming helpers for SSE transcript endpoints."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Any, Protocol

from pydantic import BaseModel, Field

from ..core.protocols import ASRProcessor


class SegmentPayload(BaseModel):
    """Normalized transcript segment payload."""

    start: float
    end: float
    text: str
    speaker: str | None = None


class StreamEventPayload(BaseModel):
    """Standard event payload shared by streaming endpoints."""

    timestamp: float = Field(default_factory=time.time)
    webrtc_id: str | None = None
    full_transcript: str = ""
    segments: list[SegmentPayload] = Field(default_factory=list)
    final: bool = False
    status: str | None = None
    error: str | None = None


class RealtimeStreamProtocol(Protocol):
    """Minimal protocol used from fastrtc.Stream for testability."""

    def output_stream(self, webrtc_id: str) -> AsyncIterator[Any]:
        """Yield stream outputs for a specific WebRTC session."""
        ...


def format_sse_event(event: str, payload: StreamEventPayload) -> str:
    """Serialize a typed payload as a Server-Sent Event frame."""
    return f"event: {event}\ndata: {payload.model_dump_json()}\n\n"


async def stream_realtime_transcript(
    stream: RealtimeStreamProtocol, webrtc_id: str
) -> AsyncIterator[str]:
    """
    Stream realtime transcript updates from fastrtc output stream as SSE frames.
    """
    async for output in stream.output_stream(webrtc_id):
        payload = StreamEventPayload(
            webrtc_id=webrtc_id,
            full_transcript=output.args[0],
            segments=[
                SegmentPayload(
                    start=float(s.get("start", 0)),
                    end=float(s.get("end", 0)),
                    text=str(s.get("text", "")),
                )
                for s in (output.args[2] or [])
            ],
        )
        yield format_sse_event("output", payload)


async def transcribe_audio_in_chunks(
    processor: ASRProcessor,
    audio: Any,
    sample_rate: int,
    chunk_size_seconds: float | None = None,
    diarization_processor: Any = None,
) -> AsyncIterator[dict[str, Any]]:
    """
    Chunk and transcribe a single audio array, yielding segment dictionaries.
    """
    await asyncio.to_thread(processor.init, 0)

    chunk_sec = chunk_size_seconds or getattr(processor, "min_chunk_sec", 1.0)
    samples_per_chunk = int(chunk_sec * sample_rate)

    for i in range(0, len(audio), samples_per_chunk):
        chunk = audio[i : i + samples_per_chunk]
        await asyncio.to_thread(processor.insert_audio_chunk, chunk)

        processed_output = await asyncio.to_thread(processor.process_iter)
        if processed_output and processed_output[2]:
            beg, end, text = processed_output
            chunk_offset = i / sample_rate
            segment = {
                "start": beg + chunk_offset,
                "end": end + chunk_offset,
                "text": text,
                "final": False,
            }

            if diarization_processor and diarization_processor.is_enabled:
                try:
                    audio_segment = audio[i : i + samples_per_chunk]
                    speaker_segments = diarization_processor.process_audio(
                        audio_segment, sample_rate
                    )
                    if speaker_segments:
                        segment["speaker"] = speaker_segments[0].speaker
                except Exception:
                    pass

            yield segment

        await asyncio.sleep(0)

    final_flush_output = await asyncio.to_thread(processor.finish)
    if final_flush_output and final_flush_output[2]:
        beg, end, text = final_flush_output
        final_offset = max(0, len(audio) - samples_per_chunk) / sample_rate
        segment = {
            "start": beg + final_offset,
            "end": end + final_offset,
            "text": text,
            "final": True,
        }

        if diarization_processor and diarization_processor.is_enabled:
            try:
                speaker_segments = diarization_processor.process_audio(
                    audio, sample_rate
                )
                if speaker_segments:
                    segment["speaker"] = speaker_segments[0].speaker
            except Exception:
                pass

        yield segment


async def stream_upload_transcript(
    processor: ASRProcessor,
    audio: Any,
    sample_rate: int,
    chunk_size_seconds: float | None = None,
    diarization_processor: Any = None,
) -> AsyncIterator[str]:
    """
    Stream upload transcription as SSE frames using the shared event schema.
    """
    transcript_parts: list[str] = []

    async for result in transcribe_audio_in_chunks(
        processor=processor,
        audio=audio,
        sample_rate=sample_rate,
        chunk_size_seconds=chunk_size_seconds,
        diarization_processor=diarization_processor,
    ):
        text = str(result.get("text", "")).strip()
        if text:
            transcript_parts.append(text)

        payload = StreamEventPayload(
            full_transcript=" ".join(transcript_parts),
            segments=[
                SegmentPayload(
                    start=float(result.get("start", 0)),
                    end=float(result.get("end", 0)),
                    text=text,
                    speaker=result.get("speaker"),
                )
            ],
            final=bool(result.get("final", False)),
        )
        yield format_sse_event("output", payload)

    final_payload = StreamEventPayload(
        full_transcript=" ".join(transcript_parts),
        final=True,
        status="completed",
    )
    yield format_sse_event("final", final_payload)


def parse_sse_data(frame: str) -> dict[str, Any]:
    """Parse a single SSE frame produced by format_sse_event."""
    data_prefix = "data: "
    for line in frame.splitlines():
        if line.startswith(data_prefix):
            return json.loads(line[len(data_prefix) :])
    raise ValueError("No data payload found in SSE frame")
