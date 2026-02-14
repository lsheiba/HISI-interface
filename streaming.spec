# Streaming Implementation Plan

## Status Check
- `streaming` is **not** implemented as a named feature/module/endpoint.
- Existing streaming support already exists for transcription:
  - WebRTC audio ingest via `fastrtc.Stream` in `asr_interface/web/server.py`
  - SSE transcript streaming via `GET /transcript`
  - SSE upload transcription via `POST /upload_and_transcribe`

## Goal
Introduce an explicit `streaming` implementation layer that standardizes streaming behavior (transport, payload format, lifecycle, and errors) while preserving current endpoints.

## Scope
1. Add a dedicated streaming service module (e.g., `asr_interface/web/streaming.py`).
2. Define typed streaming event schema (`output`, `status`, `error`, `final`).
3. Refactor `/transcript` and `/upload_and_transcribe` to use shared generator helpers.
4. Add optional heartbeat and client disconnect handling.
5. Add tests for stream ordering, error propagation, and final flush.

## Implementation Steps
1. Create stream event model
- Add Pydantic models for event payloads and serializer helper to SSE format.
- Enforce stable keys: `event`, `timestamp`, `webrtc_id`, `full_transcript`, `segments`, `final`, `error`.

2. Build streaming service
- Implement reusable async generators:
  - `stream_realtime_transcript(webrtc_id)`
  - `stream_upload_transcript(processor, audio_bytes)`
- Centralize try/except/finally behavior and structured errors.

3. Integrate in routes
- Replace inline generator logic in `asr_interface/web/server.py` with service calls.
- Keep response headers (`text/event-stream`, no-cache, keep-alive).

4. Robustness
- Add keepalive ping events every N seconds.
- Detect cancellation and close stream cleanly.
- Ensure `processor.finish()` output is always emitted once.

5. Tests
- Add API tests in `tests/` for:
  - Happy-path realtime SSE stream
  - Happy-path upload SSE stream
  - Error event on processor exception
  - Final event correctness and ordering

## Acceptance Criteria
- Both endpoints emit consistent event schema.
- Streams remain open until completion/disconnect and recover gracefully from errors.
- No regression in existing UI EventSource handlers.
- New tests pass with `pytest`.
