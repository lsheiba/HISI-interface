# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HISI Interface is a real-time ASR (Automatic Speech Recognition) interface built with FastAPI and WebRTC. It streams audio via WebRTC, processes it through configurable Whisper backends, and returns transcriptions. It also supports file upload transcription and model evaluation (WER/CER).

## Build & Development Commands

**Package manager:** uv (with Hatchling build backend)

```bash
# Install dependencies
uv sync

# Install with dev dependencies (dev deps are in [project.optional-dependencies], not [dependency-groups])
uv sync --extra dev

# Run the server
uv run hisi-interface serve [--host HOST] [--port PORT] [--reload]

# Run tests
uv run pytest

# Run a single test
uv run pytest tests/test_foo.py::test_bar -v

# Formatting
uv run black .

# Linting
uv run ruff check .
uv run ruff check --fix .

# Type checking (strict mode enabled)
uv run mypy asr_interface

# Import sorting
uv run isort .

# Pre-commit hooks (black only)
uv run pre-commit run --all-files
```

## Architecture

### Protocol-Based Backend System

The core abstraction uses Python Protocols (`asr_interface/core/protocols.py`):

- **`ASRBase`** — Abstract class for ASR backends. Implements `transcribe()`, `ts_words()`, `segments_end_ts()`.
- **`ASRProcessor`** — Protocol for real-time streaming processors. Methods: `insert_audio_chunk()`, `process_iter()`, `init()`, `finish()`.
- **`ModelLoader`** — Protocol for backend factories. Method: `load(config) → (processor, info_dict)`.

Backends register themselves in `asr_interface/backends/registry.py` via `register_loader()`. Currently registered: `"whisper"` and `"mlx_whisper"`.

### Real-Time Streaming Pipeline

```
Browser (WebRTC) → RealTimeASRHandler (fastRTC) → OnlineASRProcessor → ASRBase backend → transcript
```

- **`OnlineASRProcessor`** (`backends/whisper_online_processor.py`) — Core streaming engine. Buffers audio chunks, calls the backend for transcription, and uses `HypothesisBuffer` to stabilize overlapping predictions and remove duplicates.
- **`RealTimeASRHandler`** (`handlers/stream_handler.py`) — Bridges fastRTC's `StreamHandler` to the ASR processor.
- **`ASRComponentsStore`** (`core/store.py`) — Thread-safe shared state (processor, sample_rate, ready flag). Supports hot-swapping models at runtime.

### Web Layer

**`ASRServer`** (`web/server.py`) — FastAPI app with these key routes:

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Serves the HTML interface |
| POST | `/load_model` | Load/swap ASR model |
| POST | `/upload_and_transcribe` | File upload transcription |
| POST | `/evaluate_model` | WER/CER evaluation |
| GET | `/transcript` | Get WebRTC session transcript |
| WebRTC | `/stream` | Real-time audio streaming |

### Frontend

Single `index.html` with supporting scripts in `scripts/`. Uses browser-native WebRTC and WaveSurfer.js for audio visualization.

### Configuration

`ASRConfig` (Pydantic model in `core/config.py`) controls model selection, language, task (transcribe/translate), chunking, buffer trimming, and TURN server settings.

## Code Quality

- **black**: formatter (line-length 88, Python 3.10 target)
- **ruff**: linter (E, W, F, I, B, C4, UP rules; E501 ignored)
- **mypy**: strict type checking enabled
- **isort**: import sorting (black profile)
- **pytest**: with coverage reporting (`--cov=asr_interface`)
- Pre-commit hook runs black only

## Key Conventions

- Python 3.10+ required
- Package name in code: `asr_interface` (the `hisi-interface` is the distribution name)
- TURN server support: HuggingFace, Twilio, Cloudflare (configured via `TURNConfig`)
- Audio sample rate constant: `SAMPLING_RATE` in `utils/audio.py`
