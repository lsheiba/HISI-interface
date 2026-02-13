# ABOUTME: Tests for non-blocking model loading and /loading_status endpoint.
# ABOUTME: Covers state transitions, concurrent load rejection, and error handling.

import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from asr_interface.core.store import ASRComponentsStore
from asr_interface.web.server import ASRServer


@pytest.fixture
def store():
    return ASRComponentsStore()


@pytest.fixture
def server(store):
    return ASRServer(store=store)


@pytest.fixture
def client(server):
    return TestClient(server.app)


class TestLoadingStatus:
    def test_initial_status_is_idle(self, client):
        resp = client.get("/loading_status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "idle"
        assert data["error"] is None

    @patch("asr_interface.web.server.get_loader")
    def test_load_model_returns_loading_status(self, mock_get_loader, client):
        """POST /load_model returns immediately with status 'loading'."""
        # Make the loader slow so we can observe the loading state
        mock_loader = MagicMock()

        def slow_load(config):
            time.sleep(0.5)
            processor = MagicMock()
            processor.insert_audio_chunk = MagicMock()
            processor.process_iter = MagicMock()
            return processor, {"separator": " "}

        mock_loader.load.side_effect = slow_load
        mock_get_loader.return_value = mock_loader

        config = {
            "model": "tiny",
            "backend": "whisper",
            "lan": "en",
        }

        resp = client.post("/load_model", json=config)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "loading"

    @patch("asr_interface.web.server.get_loader")
    def test_status_transitions_to_ready(self, mock_get_loader, client):
        """After loading completes, /loading_status returns 'ready'."""
        mock_loader = MagicMock()
        mock_processor = MagicMock()
        mock_processor.insert_audio_chunk = MagicMock()
        mock_processor.process_iter = MagicMock()
        mock_loader.load.return_value = (mock_processor, {"separator": " "})
        mock_get_loader.return_value = mock_loader

        config = {
            "model": "tiny",
            "backend": "whisper",
            "lan": "en",
        }

        client.post("/load_model", json=config)

        # Wait for the background thread to complete
        time.sleep(0.3)

        resp = client.get("/loading_status")
        data = resp.json()
        assert data["status"] == "ready"

    @patch("asr_interface.web.server.get_loader")
    def test_status_transitions_to_error_on_failure(self, mock_get_loader, client):
        """If loading fails, /loading_status returns 'error' with message."""
        mock_loader = MagicMock()
        mock_loader.load.side_effect = RuntimeError("Model not found")
        mock_get_loader.return_value = mock_loader

        config = {
            "model": "nonexistent",
            "backend": "whisper",
            "lan": "en",
        }

        client.post("/load_model", json=config)
        time.sleep(0.3)

        resp = client.get("/loading_status")
        data = resp.json()
        assert data["status"] == "error"
        assert "Model not found" in data["error"]

    @patch("asr_interface.web.server.get_loader")
    def test_concurrent_load_rejected(self, mock_get_loader, client):
        """If already loading, a second /load_model returns 409."""
        mock_loader = MagicMock()

        def slow_load(config):
            time.sleep(1.0)
            return MagicMock(), {"separator": " "}

        mock_loader.load.side_effect = slow_load
        mock_get_loader.return_value = mock_loader

        config = {
            "model": "tiny",
            "backend": "whisper",
            "lan": "en",
        }

        # Start first load
        resp1 = client.post("/load_model", json=config)
        assert resp1.status_code == 200

        # Try second load while first is in progress
        resp2 = client.post("/load_model", json=config)
        assert resp2.status_code == 409

    def test_already_loaded_config_skips_loading(self, client, store):
        """If the config is already loaded, return success without re-loading."""
        # Pre-set the store as if a model is already loaded
        store.asr_processor = MagicMock()
        store.is_ready = True
        from asr_interface.core.config import ASRConfig

        config = ASRConfig(model="tiny", backend="whisper", lan="en")
        store.current_config_id = store.get_config_id(config)

        resp = client.post(
            "/load_model",
            json={"model": "tiny", "backend": "whisper", "lan": "en"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

    def test_loading_status_includes_message(self, client):
        """GET /loading_status includes a message field."""
        resp = client.get("/loading_status")
        data = resp.json()
        assert "message" in data
        assert data["message"] is None


class TestStoreLoadingState:
    def test_store_has_loading_status(self, store):
        assert store.loading_status == "idle"

    def test_store_has_loading_error(self, store):
        assert store.loading_error is None

    def test_store_has_loading_message(self, store):
        assert store.loading_message is None

    def test_store_loading_status_settable(self, store):
        store.loading_status = "loading"
        assert store.loading_status == "loading"

    def test_store_loading_error_settable(self, store):
        store.loading_error = "something broke"
        assert store.loading_error == "something broke"

    def test_store_loading_message_settable(self, store):
        store.loading_message = "Downloading model weights..."
        assert store.loading_message == "Downloading model weights..."

    def test_store_reset_clears_loading_state(self, store):
        store.loading_status = "error"
        store.loading_error = "something broke"
        store.loading_message = "some message"
        store.reset()
        assert store.loading_status == "idle"
        assert store.loading_error is None
        assert store.loading_message is None
