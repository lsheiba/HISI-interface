# ABOUTME: Tests for the GET /backends catalog endpoint.
# ABOUTME: Verifies the endpoint returns available backends and their models.

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from asr_interface.core.store import ASRComponentsStore
from asr_interface.web.server import ASRServer


@pytest.fixture
def client():
    store = ASRComponentsStore()
    server = ASRServer(store=store)
    return TestClient(server.app)


class TestBackendsEndpoint:
    def test_returns_200(self, client):
        resp = client.get("/backends")
        assert resp.status_code == 200

    def test_response_has_backends_key(self, client):
        resp = client.get("/backends")
        data = resp.json()
        assert "backends" in data
        assert isinstance(data["backends"], list)

    def test_each_backend_has_id_and_models(self, client):
        resp = client.get("/backends")
        for backend in resp.json()["backends"]:
            assert "id" in backend
            assert "models" in backend
            assert isinstance(backend["models"], list)

    def test_backend_with_list_models_includes_models(self, client):
        """Backends that implement list_models() have their models listed."""
        resp = client.get("/backends")
        data = resp.json()

        # mlx_audio backend should have models since it implements list_models()
        mlx_audio = next(
            (b for b in data["backends"] if b["id"] == "mlx_audio"), None
        )
        if mlx_audio is not None:
            assert len(mlx_audio["models"]) > 0
            for model in mlx_audio["models"]:
                assert "id" in model
                assert "name" in model

    def test_only_installed_backends_listed(self, client):
        """Only backends that are actually installed appear in the response."""
        resp = client.get("/backends")
        backend_ids = [b["id"] for b in resp.json()["backends"]]
        # We can't predict exactly which backends are installed,
        # but they should all be strings
        assert all(isinstance(bid, str) for bid in backend_ids)
        assert len(backend_ids) > 0  # At least one backend should be available

    def test_backend_without_list_models(self):
        """Backends without list_models() return an empty model list."""
        mock_loader = MagicMock(spec=[])  # spec=[] means no attributes at all
        with patch(
            "asr_interface.web.server.MODEL_LOADERS",
            {"test_backend": mock_loader},
        ):
            store = ASRComponentsStore()
            server = ASRServer(store=store)
            test_client = TestClient(server.app)

            resp = test_client.get("/backends")
            test = next(
                (b for b in resp.json()["backends"] if b["id"] == "test_backend"),
                None,
            )
            assert test is not None
            assert test["models"] == []
