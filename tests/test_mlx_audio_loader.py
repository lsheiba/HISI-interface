# ABOUTME: Tests for the mlx-audio ASR backend output normalization and loader.
# ABOUTME: Covers Qwen3-ASR, GLM-ASR, and VibeVoice-ASR output formats.

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from asr_interface.backends.mlx_audio_loader import (
    MLX_AUDIO_MODELS,
    MLXAudioASR,
    MLXAudioLoader,
)
from asr_interface.core.config import ASRConfig


# --- Fixtures for model output formats ---


@pytest.fixture
def qwen3_segments():
    """Qwen3-ASR output format: segments with start/end and optional words."""
    return [
        {
            "start": 0.0,
            "end": 2.5,
            "text": "Hello world",
            "words": [
                {"start": 0.0, "end": 1.2, "word": "Hello"},
                {"start": 1.3, "end": 2.5, "word": "world"},
            ],
        },
        {
            "start": 3.0,
            "end": 5.0,
            "text": "How are you",
            "words": [
                {"start": 3.0, "end": 3.5, "word": "How"},
                {"start": 3.6, "end": 4.2, "word": "are"},
                {"start": 4.3, "end": 5.0, "word": "you"},
            ],
        },
    ]


@pytest.fixture
def glm_segments():
    """GLM-ASR output format: segments with start/end but no word timestamps."""
    return [
        {"start": 0.0, "end": 2.5, "text": "Hello world"},
        {"start": 3.0, "end": 5.0, "text": "How are you"},
    ]


@pytest.fixture
def vibevoice_segments():
    """VibeVoice-ASR output format: segments with start_time/end_time and speaker_id."""
    return [
        {
            "start_time": 0.0,
            "end_time": 2.5,
            "text": "Hello world",
            "speaker_id": 0,
        },
        {
            "start_time": 3.0,
            "end_time": 5.0,
            "text": "How are you",
            "speaker_id": 1,
        },
    ]


# --- Tests for ts_words ---


class TestTsWords:
    def test_qwen3_with_word_timestamps(self, qwen3_segments):
        """Qwen3-ASR provides word-level timestamps."""
        asr = MLXAudioASR.__new__(MLXAudioASR)
        asr.model_id = "mlx-community/Qwen3-ASR-0.6B-8bit"
        result = asr.ts_words(qwen3_segments)
        assert result == [
            (0.0, 1.2, "Hello"),
            (1.3, 2.5, "world"),
            (3.0, 3.5, "How"),
            (3.6, 4.2, "are"),
            (4.3, 5.0, "you"),
        ]

    def test_glm_without_word_timestamps(self, glm_segments):
        """GLM-ASR has no word timestamps; falls back to segment-level."""
        asr = MLXAudioASR.__new__(MLXAudioASR)
        asr.model_id = "mlx-community/GLM-ASR-Nano-2512-4bit"
        result = asr.ts_words(glm_segments)
        assert result == [
            (0.0, 2.5, "Hello world"),
            (3.0, 5.0, "How are you"),
        ]

    def test_vibevoice_with_start_end_time_keys(self, vibevoice_segments):
        """VibeVoice uses start_time/end_time keys instead of start/end."""
        asr = MLXAudioASR.__new__(MLXAudioASR)
        asr.model_id = "mlx-community/VibeVoice-ASR-bf16"
        result = asr.ts_words(vibevoice_segments)
        assert result == [
            (0.0, 2.5, "Hello world"),
            (3.0, 5.0, "How are you"),
        ]

    def test_empty_segments(self):
        """Empty segment list returns empty word list."""
        asr = MLXAudioASR.__new__(MLXAudioASR)
        asr.model_id = "mlx-community/Qwen3-ASR-0.6B-8bit"
        assert asr.ts_words([]) == []


# --- Tests for segments_end_ts ---


class TestSegmentsEndTs:
    def test_qwen3_end_timestamps(self, qwen3_segments):
        asr = MLXAudioASR.__new__(MLXAudioASR)
        asr.model_id = "mlx-community/Qwen3-ASR-0.6B-8bit"
        assert asr.segments_end_ts(qwen3_segments) == [2.5, 5.0]

    def test_glm_end_timestamps(self, glm_segments):
        asr = MLXAudioASR.__new__(MLXAudioASR)
        asr.model_id = "mlx-community/GLM-ASR-Nano-2512-4bit"
        assert asr.segments_end_ts(glm_segments) == [2.5, 5.0]

    def test_vibevoice_end_timestamps(self, vibevoice_segments):
        asr = MLXAudioASR.__new__(MLXAudioASR)
        asr.model_id = "mlx-community/VibeVoice-ASR-bf16"
        assert asr.segments_end_ts(vibevoice_segments) == [2.5, 5.0]

    def test_empty_segments(self):
        asr = MLXAudioASR.__new__(MLXAudioASR)
        asr.model_id = "mlx-community/Qwen3-ASR-0.6B-8bit"
        assert asr.segments_end_ts([]) == []


# --- Tests for model catalog ---


class TestModelCatalog:
    def test_catalog_has_all_models(self):
        expected_ids = {
            "mlx-community/Qwen3-ASR-0.6B-8bit",
            "mlx-community/Qwen3-ASR-1.7B-8bit",
            "mlx-community/GLM-ASR-Nano-2512-4bit",
            "mlx-community/VibeVoice-ASR-4bit",
            "mlx-community/VibeVoice-ASR-bf16",
        }
        actual_ids = {m["id"] for m in MLX_AUDIO_MODELS}
        assert expected_ids == actual_ids

    def test_each_model_has_required_fields(self):
        required_fields = {"id", "name", "params", "timestamps", "diarization"}
        for model in MLX_AUDIO_MODELS:
            missing = required_fields - set(model.keys())
            assert (
                not missing
            ), f"Model {model.get('id', '?')} missing fields: {missing}"


# --- Tests for loader ---


class TestMLXAudioLoader:
    def test_list_models_returns_catalog(self):
        loader = MLXAudioLoader()
        models = loader.list_models()
        assert len(models) == len(MLX_AUDIO_MODELS)
        assert all("id" in m for m in models)

    @patch("asr_interface.backends.mlx_audio_loader.mlx_audio_load")
    def test_load_creates_processor(self, mock_load):
        """Loading a model returns an OnlineASRProcessor and metadata."""
        mock_model = MagicMock()
        mock_load.return_value = mock_model

        loader = MLXAudioLoader()
        config = ASRConfig(
            model="mlx-community/Qwen3-ASR-0.6B-8bit",
            backend="mlx_audio",
            lan="en",
        )
        processor, metadata = loader.load(config)

        mock_load.assert_called_once_with("mlx-community/Qwen3-ASR-0.6B-8bit")
        assert metadata["backend"] == "mlx_audio"
        assert metadata["model_size"] == "mlx-community/Qwen3-ASR-0.6B-8bit"
        assert hasattr(processor, "insert_audio_chunk")
        assert hasattr(processor, "process_iter")


# --- Tests for transcribe (mocked mlx-audio) ---


class TestTranscribe:
    @patch("asr_interface.backends.mlx_audio_loader.mlx_audio_load")
    def test_transcribe_normalizes_qwen3_output(self, mock_load):
        """transcribe() returns normalized segments from Qwen3-ASR."""
        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.segments = [
            {"start": 0.0, "end": 2.5, "text": "Hello world"},
        ]
        mock_model.generate.return_value = mock_result
        mock_load.return_value = mock_model

        asr = MLXAudioASR.__new__(MLXAudioASR)
        asr.model = mock_model
        asr.model_id = "mlx-community/Qwen3-ASR-0.6B-8bit"
        asr.original_language = "en"
        asr.transcribe_kargs = {}

        audio = np.zeros(16000, dtype=np.float32)
        segments = asr.transcribe(audio)

        assert len(segments) == 1
        assert segments[0]["start"] == 0.0
        assert segments[0]["end"] == 2.5
        assert segments[0]["text"] == "Hello world"

    @patch("asr_interface.backends.mlx_audio_loader.mlx_audio_load")
    def test_transcribe_normalizes_vibevoice_output(self, mock_load):
        """transcribe() normalizes VibeVoice's start_time/end_time keys."""
        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.segments = [
            {
                "start_time": 0.0,
                "end_time": 2.5,
                "text": "Hello",
                "speaker_id": 0,
            },
        ]
        mock_model.generate.return_value = mock_result
        mock_load.return_value = mock_model

        asr = MLXAudioASR.__new__(MLXAudioASR)
        asr.model = mock_model
        asr.model_id = "mlx-community/VibeVoice-ASR-bf16"
        asr.original_language = "en"
        asr.transcribe_kargs = {}

        audio = np.zeros(16000, dtype=np.float32)
        segments = asr.transcribe(audio)

        assert len(segments) == 1
        # After normalization, all segments should use start/end keys
        assert segments[0]["start"] == 0.0
        assert segments[0]["end"] == 2.5
        assert segments[0]["text"] == "Hello"
