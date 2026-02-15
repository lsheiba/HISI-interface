# ABOUTME: Tests for transcription functionality using input/1.wav audio file.

import io
from pathlib import Path
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


@pytest.fixture
def audio_file_path():
    return Path(__file__).parent.parent / "input" / "1.wav"


@pytest.fixture
def audio_bytes(audio_file_path):
    with open(audio_file_path, "rb") as f:
        return f.read()


class TestUploadAndTranscribe:
    def test_upload_transcribe_requires_model_loaded(self, client, audio_bytes):
        resp = client.post(
            "/upload_and_transcribe",
            files={"audio_file": ("test.wav", io.BytesIO(audio_bytes), "audio/wav")},
        )
        assert resp.status_code == 400
        assert "ASR processor not loaded" in resp.text

    def test_list_backends(self, client):
        resp = client.get("/backends")
        assert resp.status_code == 200
        data = resp.json()
        assert "backends" in data
        assert len(data["backends"]) > 0
        backend_ids = [b["id"] for b in data["backends"]]
        print(f"Available backends: {backend_ids}")


class TestModelLoaders:
    def test_mlx_whisper_loader_available(self):
        from asr_interface.backends.registry import MODEL_LOADERS

        assert "mlx_whisper" in MODEL_LOADERS
        loader = MODEL_LOADERS["mlx_whisper"]
        assert hasattr(loader, "load")
        models = loader.list_models()
        assert len(models) > 0

    def test_mlx_audio_loader_available(self):
        from asr_interface.backends.registry import MODEL_LOADERS

        assert "mlx_audio" in MODEL_LOADERS
        loader = MODEL_LOADERS["mlx_audio"]
        assert hasattr(loader, "load")
        models = loader.list_models()
        assert len(models) > 0

    def test_whisper_loader_available(self):
        from asr_interface.backends.registry import MODEL_LOADERS

        assert "whisper" in MODEL_LOADERS
        loader = MODEL_LOADERS["whisper"]
        assert hasattr(loader, "load")
        models = loader.list_models()
        assert len(models) > 0

    def test_all_loaders_have_list_models(self):
        from asr_interface.backends.registry import MODEL_LOADERS

        for name, loader in MODEL_LOADERS.items():
            assert hasattr(loader, "list_models"), f"{name} missing list_models"
            models = loader.list_models()
            assert isinstance(models, list), f"{name} list_models should return list"

    def test_get_loader_invalid_name(self):
        from asr_interface.backends.registry import get_loader

        with pytest.raises(KeyError):
            get_loader("nonexistent_backend")

    @patch("asr_interface.web.server.get_loader")
    @patch("asr_interface.web.server.load_audio_from_bytes")
    def test_upload_transcribe_success(
        self, mock_load_audio, mock_get_loader, client, audio_bytes
    ):
        import numpy as np

        mock_load_audio.return_value = np.zeros(16000, dtype=np.float32)

        mock_processor = MagicMock()
        mock_processor.insert_audio_chunk = MagicMock()
        mock_processor.process_iter = MagicMock(return_value=None)
        mock_processor.finish = MagicMock(return_value=(0.0, 1.0, "transcribed text"))

        mock_loader = MagicMock()
        mock_loader.load.return_value = (mock_processor, {"separator": " "})
        mock_get_loader.return_value = mock_loader

        config = {
            "model": "tiny",
            "backend": "whisper",
            "lan": "en",
        }
        resp = client.post("/load_model", json=config, timeout=5)
        assert resp.status_code == 200

        resp = client.post(
            "/upload_and_transcribe",
            files={"audio_file": ("test.wav", io.BytesIO(audio_bytes), "audio/wav")},
            timeout=30,
        )
        assert resp.status_code == 200

    @patch("asr_interface.web.server.get_loader")
    @patch("asr_interface.web.server.load_audio_from_bytes")
    def test_upload_transcribe_with_diarization(
        self, mock_load_audio, mock_get_loader, client, audio_bytes
    ):
        import numpy as np

        mock_load_audio.return_value = np.zeros(16000, dtype=np.float32)

        mock_processor = MagicMock()
        mock_processor.insert_audio_chunk = MagicMock()
        mock_processor.process_iter = MagicMock(return_value=None)
        mock_processor.finish = MagicMock(return_value=(0.0, 1.0, "hello world"))

        mock_loader = MagicMock()
        mock_loader.load.return_value = (mock_processor, {"separator": " "})
        mock_get_loader.return_value = mock_loader

        config = {
            "model": "tiny",
            "backend": "whisper",
            "lan": "en",
            "diarization": {"enabled": True, "backend": "pyanote"},
        }
        resp = client.post("/load_model", json=config, timeout=5)
        assert resp.status_code == 200

        resp = client.post(
            "/upload_and_transcribe",
            files={"audio_file": ("test.wav", io.BytesIO(audio_bytes), "audio/wav")},
            timeout=30,
        )
        assert resp.status_code == 200
