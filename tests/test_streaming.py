# ABOUTME: Tests for shared SSE streaming helpers.
# ABOUTME: Verifies payload format and stream generation for realtime/upload flows.

from types import SimpleNamespace

import pytest

from asr_interface.web.streaming import (
    SegmentPayload,
    StreamEventPayload,
    format_sse_event,
    parse_sse_data,
    stream_realtime_transcript,
    stream_upload_transcript,
    transcribe_audio_in_chunks,
)


class DummyProcessor:
    def __init__(self):
        self.min_chunk_sec = 1.0
        self.calls = 0
        self.finished = False

    def init(self, offset: float = 0.0) -> None:
        self.calls = 0
        self.finished = False

    def insert_audio_chunk(self, _audio_chunk) -> None:
        self.calls += 1

    def process_iter(self):
        if self.calls == 1:
            return (0.0, 1.0, "hello")
        if self.calls == 2:
            return (1.0, 2.0, "world")
        return None

    def finish(self):
        if self.finished:
            return None
        self.finished = True
        return (2.0, 2.1, "done")


class DummyRealtimeStream:
    async def output_stream(self, _webrtc_id: str):
        yield SimpleNamespace(
            args=[
                "hello world",
                None,
                [{"start": 0.0, "end": 1.0, "text": "hello"}],
            ]
        )


def test_format_sse_event_serializes_payload():
    payload = StreamEventPayload(
        webrtc_id="abc123",
        full_transcript="hello",
        segments=[SegmentPayload(start=0.0, end=1.0, text="hello")],
    )
    frame = format_sse_event("output", payload)

    assert frame.startswith("event: output\n")
    parsed = parse_sse_data(frame)
    assert parsed["webrtc_id"] == "abc123"
    assert parsed["full_transcript"] == "hello"
    assert parsed["segments"][0]["text"] == "hello"


@pytest.mark.asyncio
async def test_stream_realtime_transcript_emits_output_event():
    stream = DummyRealtimeStream()

    frames = []
    async for frame in stream_realtime_transcript(stream, "webrtc-1"):
        frames.append(frame)

    assert len(frames) == 1
    assert frames[0].startswith("event: output\n")
    parsed = parse_sse_data(frames[0])
    assert parsed["webrtc_id"] == "webrtc-1"
    assert parsed["full_transcript"] == "hello world"
    assert parsed["segments"][0]["text"] == "hello"


@pytest.mark.asyncio
async def test_transcribe_audio_in_chunks_yields_final_segment():
    processor = DummyProcessor()
    audio = [0.0] * 32000

    results = []
    async for item in transcribe_audio_in_chunks(processor, audio, sample_rate=16000):
        results.append(item)

    assert len(results) == 3
    assert results[0]["text"] == "hello"
    assert results[1]["text"] == "world"
    assert results[2]["final"] is True
    assert results[2]["text"] == "done"


@pytest.mark.asyncio
async def test_stream_upload_transcript_emits_output_and_final_events():
    processor = DummyProcessor()
    audio = [0.0] * 32000

    frames = []
    async for frame in stream_upload_transcript(processor, audio, sample_rate=16000):
        frames.append(frame)

    assert len(frames) == 4
    assert frames[0].startswith("event: output\n")
    assert frames[-1].startswith("event: final\n")

    first = parse_sse_data(frames[0])
    assert first["segments"][0]["text"] == "hello"

    final = parse_sse_data(frames[-1])
    assert final["final"] is True
    assert final["status"] == "completed"
