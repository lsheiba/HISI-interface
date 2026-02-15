"""FastAPI web server for the ASR interface."""

import asyncio
import json
import logging
import logging.handlers
import os
import threading
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(Path.home() / "lsdev" / ".env")

logs_dir = Path(__file__).parent.parent.parent / "logs"
logs_dir.mkdir(exist_ok=True)

file_handler = logging.handlers.RotatingFileHandler(
    logs_dir / "server.log",
    maxBytes=10 * 1024 * 1024,
    backupCount=5,
)
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
file_handler.setFormatter(file_formatter)

logging.getLogger("asr_interface").setLevel(logging.DEBUG)
logging.getLogger("asr_interface.handlers").setLevel(logging.DEBUG)
logging.getLogger("asr_interface.backends").setLevel(logging.DEBUG)

root_logger = logging.getLogger()
root_logger.addHandler(file_handler)
root_logger.setLevel(logging.INFO)

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastrtc import Stream
from jiwer import cer, wer
from starlette.responses import Response

from ..backends.registry import MODEL_LOADERS, get_loader
from ..backends.diarization import PyanoteAdapter
from ..core.config import ASRConfig, TURNConfig
from ..core.diarization_processor import DiarizationProcessor
from ..core.protocols import ASRProcessor
from ..core.store import ASRComponentsStore
from ..handlers.stream_handler import RealTimeASRHandler
from ..utils.audio import SAMPLING_RATE, load_audio_from_bytes
from ..utils.turn_server import get_rtc_credentials
from .streaming import (
    StreamEventPayload,
    format_sse_event,
    stream_realtime_transcript,
    stream_upload_transcript,
    transcribe_audio_in_chunks,
)

logger = logging.getLogger(__name__)


class NoCacheStaticFiles(StaticFiles):
    """Custom StaticFiles class that adds no-cache headers to all responses."""

    async def get_response(self, path: str, scope) -> Response:
        response = await super().get_response(path, scope)

        # Add no-cache headers to prevent caching
        response.headers["Cache-Control"] = (
            "no-cache, no-store, must-revalidate, max-age=0"
        )
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"

        return response


class ASRServer:
    """Main ASR server application."""

    def __init__(self, store: ASRComponentsStore | None = None):
        """
        Initialize the ASR server.

        Args:
            store: Optional shared store for ASR components
        """
        self.store = store or ASRComponentsStore()
        self.app = FastAPI(
            title="ASR Interface",
            description="Real-time Automatic Speech Recognition Interface",
            version="0.1.0",
        )

        # Setup fastRTC stream
        self.master_handler = RealTimeASRHandler(store=self.store)
        self.stream = Stream(
            handler=self.master_handler, mode="send-receive", modality="audio"
        )

        # Default RTC configuration (STUN only)
        self.rtc_config = {"iceServers": [{"urls": "stun:stun.l.google.com:19302"}]}

        self._setup_routes()
        self._setup_static_files()
        self._setup_events()

    def _get_rtc_configuration(
        self, turn_config: TURNConfig | None = None
    ) -> dict[str, Any]:
        """
        Get RTC configuration with optional TURN server support.

        Args:
            turn_config: Optional TURN server configuration

        Returns:
            Dictionary containing RTC configuration
        """
        if not turn_config or turn_config.provider == "none":
            return self.rtc_config

        try:
            if turn_config.provider == "hf":
                credentials = get_rtc_credentials(
                    provider="hf", token=turn_config.token
                )
            elif turn_config.provider == "twilio":
                credentials = get_rtc_credentials(
                    provider="twilio",
                    account_sid=turn_config.account_sid,
                    auth_token=turn_config.auth_token,
                )
            elif turn_config.provider == "cloudflare":
                credentials = get_rtc_credentials(
                    provider="cloudflare",
                    key_id=turn_config.key_id,
                    api_token=turn_config.api_token,
                    ttl=turn_config.ttl,
                )
            else:
                logger.warning(f"Unknown TURN provider: {turn_config.provider}")
                return self.rtc_config

            # Merge with default STUN configuration
            ice_servers = self.rtc_config["iceServers"] + credentials.get(
                "iceServers", []
            )
            return {"iceServers": ice_servers}

        except Exception as e:
            logger.error(f"Failed to get TURN credentials: {e}")
            return self.rtc_config

    def _setup_static_files(self) -> None:
        """Setup static file serving with no-cache headers."""
        self.app.mount("/static", NoCacheStaticFiles(directory="."), name="static")

    def _setup_routes(self) -> None:
        """Setup API routes."""

        @self.app.post("/load_model")
        async def load_model(config: ASRConfig):
            """Load an ASR model in a background thread."""
            config_id = self.store.get_config_id(config)

            if self.store.is_config_current(config):
                logger.info(
                    f"Processor for config '{config_id}' is already loaded. Skipping."
                )
                return {"status": "success", "message": "Processor is already loaded."}

            if self.store.loading_status == "loading":
                if self.store.loading_config_id == config_id:
                    return {
                        "status": "loading",
                        "message": "This model configuration is already loading.",
                    }
                raise HTTPException(
                    status_code=409,
                    detail="A different model is already being loaded. Check /loading_status.",
                )

            logger.info(f"Request to create processor for new config: {config}")
            self.store.loading_status = "loading"
            self.store.loading_config_id = config_id
            self.store.loading_error = None

            thread = threading.Thread(
                target=self._load_model_sync,
                args=(config, config_id),
                daemon=True,
            )
            thread.start()

            return {"status": "loading", "message": "Model loading started."}

        @self.app.post("/cancel_load_model")
        async def cancel_load_model():
            """Cancel the current model loading if in progress."""
            if self.store.cancel_loading():
                logger.info("Model loading cancelled by user")
                return {"status": "cancelled", "message": "Model loading cancelled."}
            return {"status": "idle", "message": "No model loading in progress."}

        @self.app.get("/loading_status")
        async def loading_status():
            """Get the current model loading status."""
            return {
                "status": self.store.loading_status,
                "loading_config_id": self.store.loading_config_id,
                "error": self.store.loading_error,
                "message": self.store.loading_message,
                "progress": self.store.loading_progress,
            }

        @self.app.get("/backends")
        async def list_backends():
            """List available ASR backends and their models."""
            backends = []
            for backend_id, loader in MODEL_LOADERS.items():
                models = []
                if hasattr(loader, "list_models"):
                    models = loader.list_models()
                backends.append({"id": backend_id, "models": models})
            return {"backends": backends}

        @self.app.post("/upload_and_transcribe")
        async def upload_and_transcribe(audio_file: UploadFile = File(...)):
            """Upload and transcribe an audio file."""
            logger.info(
                f"Received file upload for real-time transcription: {audio_file.filename}"
            )

            audio_bytes = await audio_file.read()
            try:
                audio = load_audio_from_bytes(
                    audio_bytes, source_name=audio_file.filename
                )
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e

            processor_template = self.store.asr_processor
            if not processor_template:
                raise HTTPException(
                    status_code=400,
                    detail="ASR processor not loaded. Please load a model first.",
                )

            # Create a new processor instance for this file upload
            # This prevents interfering with the live ASR state
            upload_processor = self._create_upload_processor(processor_template)

            diarization_proc = (
                self.store.diarization_processor
                if self.store.diarization_enabled
                else None
            )

            async def transcription_generator():
                """Generate transcription results as SSE."""
                try:
                    if self.store.streaming_enabled:
                        async for frame in stream_upload_transcript(
                            processor=upload_processor,
                            audio=audio,
                            sample_rate=SAMPLING_RATE,
                            diarization_processor=diarization_proc,
                        ):
                            yield frame
                    else:
                        transcript_parts: list[str] = []
                        all_segments = []
                        async for result in transcribe_audio_in_chunks(
                            processor=upload_processor,
                            audio=audio,
                            sample_rate=SAMPLING_RATE,
                            diarization_processor=diarization_proc,
                        ):
                            text = str(result.get("text", "")).strip()
                            if text:
                                transcript_parts.append(text)
                            segment_data = {
                                "start": float(result.get("start", 0)),
                                "end": float(result.get("end", 0)),
                                "text": text,
                            }
                            if result.get("speaker"):
                                segment_data["speaker"] = result["speaker"]
                            all_segments.append(segment_data)

                        yield format_sse_event(
                            "output",
                            StreamEventPayload(
                                full_transcript=" ".join(transcript_parts),
                                segments=all_segments,
                                final=True,
                                status="completed",
                            ),
                        )
                        yield format_sse_event(
                            "final",
                            StreamEventPayload(
                                full_transcript=" ".join(transcript_parts),
                                final=True,
                                status="completed",
                            ),
                        )
                except Exception as e:
                    logger.error(
                        f"Error during uploaded file transcription stream: {e}",
                        exc_info=True,
                    )
                    yield format_sse_event(
                        "error",
                        StreamEventPayload(error=str(e), status="error", final=True),
                    )
                finally:
                    logger.info(
                        f"Finished processing uploaded file: {audio_file.filename}"
                    )

            return StreamingResponse(
                transcription_generator(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

        @self.app.post("/evaluate_model")
        async def evaluate_model(
            reference: str = Form(...), audio: UploadFile = File(...)
        ):
            """Evaluate model performance against a reference transcript."""
            logger.info(f"Received evaluation request for: {audio.filename}")

            audio_bytes = await audio.read()
            processor_template = self.store.asr_processor

            if not processor_template:
                raise HTTPException(
                    status_code=400,
                    detail="ASR processor not loaded. Please load a model first.",
                )

            upload_processor = self._create_upload_processor(processor_template)

            try:
                # Transcribe the audio
                transcript_parts = []
                audio = load_audio_from_bytes(audio_bytes, source_name=audio.filename)
                async for result in transcribe_audio_in_chunks(
                    processor=upload_processor,
                    audio=audio,
                    sample_rate=SAMPLING_RATE,
                ):
                    text = str(result.get("text", "")).strip()
                    if text:
                        transcript_parts.append(text)

                full_transcript = " ".join(transcript_parts)

                # Calculate metrics
                word_error_rate = wer(reference, full_transcript)
                char_error_rate = cer(reference, full_transcript)

                return {
                    "reference": reference,
                    "transcript": full_transcript,
                    "word_error_rate": word_error_rate,
                    "char_error_rate": char_error_rate,
                }

            except Exception as e:
                logger.error(f"Error during evaluation: {e}", exc_info=True)
                raise HTTPException(
                    status_code=500, detail=f"Evaluation failed: {str(e)}"
                ) from e

        @self.app.post("/reset_handler")
        async def reset_handler():
            logger.info("Resetting ASR handler state via /api/reset_handler endpoint.")
            """Reset the ASR handler state for the current session (global for now)."""
            if self.store.asr_processor:
                self.store.asr_processor.init(offset=0.0)
                logger.info("ASR handler state reset via /api/reset_handler endpoint.")
                return {"status": "success", "message": "ASR handler reset."}
            else:
                logger.warning("No ASR processor to reset in /api/reset_handler.")
                return {"status": "no-op", "message": "No ASR processor to reset."}

        @self.app.get("/")
        async def index():
            """Serve the main interface."""
            with open("index.html") as f:
                html_content = f.read()

            # Inject current RTC configuration
            html_content = html_content.replace(
                "##RTC_CONFIGURATION##", json.dumps(self.rtc_config)
            )

            return HTMLResponse(
                content=html_content,
                headers={
                    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                    "Pragma": "no-cache",
                    "Expires": "0",
                },
            )

        @self.app.get("/transcript")
        async def transcript_endpoint(webrtc_id: str):
            """Get transcript for a WebRTC session."""
            logger.debug(f"New transcript stream request for {webrtc_id}")

            async def output_stream_generator():
                if not self.store.streaming_enabled:
                    yield format_sse_event(
                        "error",
                        StreamEventPayload(
                            webrtc_id=webrtc_id,
                            error="Streaming is disabled for this model configuration.",
                            status="error",
                            final=True,
                        ),
                    )
                    return
                try:
                    async for frame in stream_realtime_transcript(
                        self.stream, webrtc_id
                    ):
                        yield frame
                except asyncio.CancelledError:
                    logger.info(f"Transcript stream for {webrtc_id} disconnected.")
                except Exception as e:
                    logger.error(
                        f"Error in transcript stream for {webrtc_id}: {e}",
                        exc_info=True,
                    )
                    yield format_sse_event(
                        "error",
                        StreamEventPayload(
                            webrtc_id=webrtc_id,
                            error=str(e),
                            status="error",
                            final=True,
                        ),
                    )

            return StreamingResponse(
                output_stream_generator(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

    def _load_model_sync(self, config: ASRConfig, config_id: str) -> None:
        """Synchronous model loading, called from a background thread."""
        try:
            self.store.loading_message = (
                f"Downloading and loading model: {config.model}"
            )
            self.store.loading_progress = 0.1

            loader = get_loader(config.backend)

            self.store.loading_message = f"Loading model weights: {config.model}"
            self.store.loading_progress = 0.3

            if self.store.loading_cancelled:
                logger.info(
                    f"Model loading cancelled before load for config '{config_id}'"
                )
                return

            online_processor, metadata = loader.load(config)

            if self.store.loading_cancelled:
                logger.info(
                    f"Model loading cancelled after load for config '{config_id}'"
                )
                return

            self.store.loading_progress = 0.9

            self.store.asr_processor = online_processor
            self.store.separator = metadata.get("separator", " ")
            self.store.streaming_enabled = config.enable_streaming

            if config.diarization and config.diarization.enabled:
                self.store.diarization_enabled = True
                backend = PyanoteAdapter(
                    min_speakers=config.diarization.min_speakers,
                    max_speakers=config.diarization.max_speakers,
                )
                diarization_proc = DiarizationProcessor(
                    backend=backend,
                    min_speakers=config.diarization.min_speakers,
                    max_speakers=config.diarization.max_speakers,
                )
                self.store.diarization_processor = diarization_proc
                logger.info(
                    f"Diarization enabled with backend: {config.diarization.backend}"
                )
            else:
                self.store.diarization_enabled = False
                self.store.diarization_processor = None

            self.store.is_ready = True
            self.store.current_config_id = config_id
            self.store.loading_status = "ready"
            self.store.loading_config_id = None
            self.store.loading_message = None
            self.store.loading_progress = 1.0
            self.store.reset_loading_state()

            if config.turn_config:
                self.rtc_config = self._get_rtc_configuration(config.turn_config)
                logger.info(
                    f"Updated RTC configuration with TURN server: {config.turn_config.provider}"
                )

            logger.info(f"Processor for config ID '{config_id}' is ready.")

        except Exception as e:
            logger.error(
                f"Fatal error during ASR processor creation: {e}", exc_info=True
            )
            self.store.loading_status = "error"
            self.store.loading_config_id = None
            self.store.loading_error = str(e)
            self.store.loading_message = None
            self.store.loading_progress = 0.0
            self.store.reset_loading_state()

    def _create_upload_processor(
        self, processor_template: ASRProcessor
    ) -> ASRProcessor:
        """Create a new processor instance for file uploads."""
        # This is a simplified version - in practice, you'd need to properly
        # clone the processor with its configuration
        return processor_template

    def _setup_events(self) -> None:
        """Setup FastAPI events."""

        @self.app.on_event("startup")
        async def startup_event():
            """Handle startup event."""
            logger.info(
                "FastAPI server started. Waiting for processor selection from a client."
            )

        # Mount the fastRTC stream
        self.stream.mount(self.app)


def create_app(store: ASRComponentsStore | None = None) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        store: Optional shared store for ASR components

    Returns:
        The configured FastAPI application
    """
    server = ASRServer(store)
    return server.app
