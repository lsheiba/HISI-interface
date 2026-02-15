"""Shared state management for ASR components."""

import hashlib
import json
import logging
import threading
from typing import Any

from .config import ASRConfig
from .protocols import ASRProcessor

logger = logging.getLogger(__name__)


class ASRComponentsStore:
    """Thread-safe store for ASR components and state."""

    def __init__(self, sample_rate: int = 16000):
        """
        Initialize the ASR components store.

        Args:
            sample_rate: Default sample rate for audio processing
        """
        self._store: dict[str, Any] = {
            "asr_processor": None,
            "sample_rate": sample_rate,
            "separator": " ",
            "streaming_enabled": True,
            "is_ready": False,
            "current_config_id": None,
            "loading_config_id": None,
            "loading_status": "idle",
            "loading_error": None,
            "loading_message": None,
            "loading_progress": 0.0,
            "diarization_enabled": False,
            "diarization_processor": None,
        }
        self._lock = threading.Lock()
        self._loading_thread: threading.Thread | None = None
        self._loading_cancelled = False

    @property
    def asr_processor(self) -> ASRProcessor | None:
        """Get the current ASR processor."""
        return self._store.get("asr_processor")

    @asr_processor.setter
    def asr_processor(self, processor: ASRProcessor | None) -> None:
        """Set the current ASR processor."""
        self._store["asr_processor"] = processor

    @property
    def sample_rate(self) -> int:
        """Get the current sample rate."""
        return self._store.get("sample_rate", 16000)

    @property
    def separator(self) -> str:
        """Get the current text separator."""
        return self._store.get("separator", " ")

    @separator.setter
    def separator(self, separator: str) -> None:
        """Set the current text separator."""
        self._store["separator"] = separator

    @property
    def is_ready(self) -> bool:
        """Check if the ASR processor is ready."""
        return self._store.get("is_ready", False)

    @is_ready.setter
    def is_ready(self, ready: bool) -> None:
        """Set the ready state."""
        self._store["is_ready"] = ready

    @property
    def loading_status(self) -> str:
        """Get the current loading status."""
        return self._store.get("loading_status", "idle")

    @loading_status.setter
    def loading_status(self, status: str) -> None:
        """Set the loading status."""
        self._store["loading_status"] = status

    @property
    def loading_error(self) -> str | None:
        """Get the loading error message, if any."""
        return self._store.get("loading_error")

    @loading_error.setter
    def loading_error(self, error: str | None) -> None:
        """Set the loading error message."""
        self._store["loading_error"] = error

    @property
    def loading_message(self) -> str | None:
        """Get the loading progress message."""
        return self._store.get("loading_message")

    @loading_message.setter
    def loading_message(self, message: str | None) -> None:
        """Set the loading progress message."""
        self._store["loading_message"] = message

    @property
    def loading_progress(self) -> float:
        """Get the loading progress (0.0 to 1.0)."""
        return self._store.get("loading_progress", 0.0)

    @loading_progress.setter
    def loading_progress(self, progress: float) -> None:
        """Set the loading progress (0.0 to 1.0)."""
        self._store["loading_progress"] = max(0.0, min(1.0, progress))

    def cancel_loading(self) -> bool:
        """Cancel the current model loading if in progress."""
        with self._lock:
            if self._store.get("loading_status") == "loading":
                self._loading_cancelled = True
                self._store["loading_status"] = "cancelled"
                self._store["loading_message"] = "Loading cancelled by user"
                return True
            return False

    @property
    def loading_cancelled(self) -> bool:
        """Check if loading was cancelled."""
        return self._loading_cancelled

    def reset_loading_state(self) -> None:
        """Reset loading state after cancellation or completion."""
        with self._lock:
            self._loading_cancelled = False
            self._store["loading_progress"] = 0.0

    @property
    def current_config_id(self) -> str | None:
        """Get the current configuration ID."""
        return self._store.get("current_config_id")

    @current_config_id.setter
    def current_config_id(self, config_id: str | None) -> None:
        """Set the current configuration ID."""
        self._store["current_config_id"] = config_id

    @property
    def loading_config_id(self) -> str | None:
        """Get the currently loading configuration ID."""
        return self._store.get("loading_config_id")

    @loading_config_id.setter
    def loading_config_id(self, config_id: str | None) -> None:
        """Set the currently loading configuration ID."""
        self._store["loading_config_id"] = config_id

    @property
    def streaming_enabled(self) -> bool:
        """Return whether streaming updates are enabled."""
        return bool(self._store.get("streaming_enabled", True))

    @streaming_enabled.setter
    def streaming_enabled(self, enabled: bool) -> None:
        """Enable or disable streaming updates."""
        self._store["streaming_enabled"] = bool(enabled)

    @property
    def diarization_enabled(self) -> bool:
        """Return whether diarization is enabled."""
        return bool(self._store.get("diarization_enabled", False))

    @diarization_enabled.setter
    def diarization_enabled(self, enabled: bool) -> None:
        """Enable or disable diarization."""
        self._store["diarization_enabled"] = bool(enabled)

    @property
    def diarization_processor(self):
        """Get the diarization processor."""
        return self._store.get("diarization_processor")

    @diarization_processor.setter
    def diarization_processor(self, processor) -> None:
        """Set the diarization processor."""
        self._store["diarization_processor"] = processor

    def get_config_id(self, config: ASRConfig) -> str:
        """
        Generate a unique configuration ID for the given config.

        Args:
            config: The ASR configuration

        Returns:
            A unique hash string for the configuration
        """
        config_json_str = json.dumps(config.dict(), sort_keys=True)
        return hashlib.sha256(config_json_str.encode("utf-8")).hexdigest()

    def is_config_current(self, config: ASRConfig) -> bool:
        """
        Check if the given configuration matches the current one.

        Args:
            config: The ASR configuration to check

        Returns:
            True if the configuration matches the current one
        """
        config_id = self.get_config_id(config)
        return (
            self.current_config_id == config_id
            and self.is_ready
            and self.asr_processor is not None
        )

    def reset(self) -> None:
        """Reset the store to initial state."""
        self._store.update(
            {
                "asr_processor": None,
                "streaming_enabled": True,
                "is_ready": False,
                "current_config_id": None,
                "loading_config_id": None,
                "loading_status": "idle",
                "loading_error": None,
                "loading_message": None,
            }
        )
        logger.info("ASR components store reset")

    def to_dict(self) -> dict[str, Any]:
        """Convert the store to a dictionary for serialization."""
        return self._store.copy()
