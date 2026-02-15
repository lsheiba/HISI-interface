"""Real-time ASR stream handler for WebRTC audio processing."""

import logging
import time
from typing import Any

import librosa
import numpy as np
from fastrtc import AdditionalOutputs, StreamHandler

from ..backends.diarization import SpeakerSegment
from ..core.protocols import ASRProcessor
from ..core.store import ASRComponentsStore

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)


class RealTimeASRHandler(StreamHandler):
    """
    A generic, real-time audio processing handler for fastRTC.

    This class acts as a bridge between the fastRTC streaming server and a
    backend ASR (Automatic Speech Recognition) processor. It is designed to be
    decoupled from the specifics of the ASR implementation.

    It retrieves a pre-initialized ASR processor from a shared store
    and uses it to process incoming audio frames. For each new client connection,
    a new instance of this handler is created via the .copy() method.
    """

    def __init__(self, store: ASRComponentsStore, **kwargs: Any) -> None:
        """
        Initialize the handler instance for a new connection.

        Args:
            store: The shared ASR components store
            **kwargs: Additional arguments passed to StreamHandler
        """
        rate_to_use = kwargs.pop("input_sample_rate", store.sample_rate)
        super().__init__(input_sample_rate=rate_to_use, **kwargs)

        self.store = store

        self.asr_processor: ASRProcessor | None = None

        self.accumulated_transcript = ""
        self.segments: list[dict[str, Any]] = []

        self.full_audio = np.zeros((0,), dtype=np.float32)

        self.last_used_config_id: str | None = None

        self.handler_id = id(self)

        self._last_process_time = 0.0
        self._process_interval = 0.3

        self._diarization_enabled = False
        self._speaker_segments: list[SpeakerSegment] = []
        self._audio_for_diarization: np.ndarray | None = None

        logger.info(
            f"Handler instance [{self.handler_id}] created. Waiting for processor."
        )

        if self.asr_processor:
            self.asr_processor.init(offset=0.0)
        self._ensure_processor()

    def _ensure_processor(self) -> None:
        """
        Synchronize the handler's local ASR processor with the shared store.

        This method is the key to the dynamic model loading. It checks if the
        globally available processor is newer than the one this handler instance
        is currently using and updates it if necessary.
        """
        # If we already have the correct processor, do nothing
        is_already_set = (
            self.asr_processor
            and self.last_used_config_id == self.store.current_config_id
        )
        if is_already_set:
            return

        # If the shared store has a ready processor, acquire it
        if self.store.is_ready and self.store.asr_processor:
            config_id = self.store.current_config_id or "unknown"
            logger.info(
                f"Handler [{self.handler_id}] - Acquiring new ASR processor for config '{config_id}'."
            )
            self.asr_processor = self.store.asr_processor
            self.last_used_config_id = config_id
            self._diarization_enabled = self.store.diarization_enabled
            self._reset_instance_state()
            logger.info(
                f"Handler [{self.handler_id}] - Processor acquired and reset successfully. "
                f"Diarization enabled: {self._diarization_enabled}"
            )
        else:
            self.asr_processor = None

    def _reset_instance_state(self) -> None:
        """Reset the transcription state when a new processor is acquired."""
        logger.info(f"Handler [{self.handler_id}] - Resetting instance state.")
        self.accumulated_transcript = ""
        self.segments = []
        self.full_audio = np.zeros((0,), dtype=np.float32)
        self._audio_for_diarization = None
        self._speaker_segments = []
        self.asr_processor.init(offset=0.0)

    def receive(self, frame: tuple[int, np.ndarray]) -> None:
        """
        Process an incoming audio frame from the WebRTC stream.

        This method is called by fastRTC for each audio chunk received. It handles
        audio format conversion, resampling, and passing the data to the ASR processor.

        Args:
            frame: A tuple containing sample rate and PCM audio data
        """
        receive_start = time.perf_counter()
        self._ensure_processor()
        if not self.asr_processor:
            return

        sample_rate, pcm_data = frame

        # Convert audio to float32 format, required by Whisper
        audio_float32 = pcm_data.astype(np.float32) / 32768.0

        # Resample if the incoming audio's sample rate differs from the target
        target_sr = self.store.sample_rate
        if sample_rate != target_sr:
            audio_float32 = librosa.resample(
                audio_float32, orig_sr=sample_rate, target_sr=target_sr
            )

        audio_len = len(audio_float32)
        self.asr_processor.insert_audio_chunk(audio_float32.flatten())

        receive_time = time.perf_counter() - receive_start
        logger.debug(
            f"[RECEIVE] Handler {self.handler_id}: received {audio_len} samples "
            f"({audio_len / target_sr * 1000:.1f}ms), processing took {receive_time * 1000:.1f}ms"
        )

    def emit(self) -> AdditionalOutputs:
        """
        Poll the ASR processor for new transcription results to send to the client.

        This method is called periodically by fastRTC's event loop.

        Returns:
            AdditionalOutputs: A data structure containing the full transcript and segments
        """
        current_time = time.perf_counter()
        time_since_last_process = current_time - self._last_process_time

        if not self.asr_processor:
            logger.debug(
                f"[EMIT] Handler {self.handler_id}: no processor, returning empty"
            )
            return AdditionalOutputs("", np.array([], dtype=np.float32), [])

        if time_since_last_process < self._process_interval:
            return AdditionalOutputs(
                self.accumulated_transcript, self.full_audio, self.segments
            )

        self._last_process_time = current_time

        processed_output = self.asr_processor.process_iter()

        if processed_output is None:
            return AdditionalOutputs(
                self.accumulated_transcript, self.full_audio, self.segments
            )

        beg, end, text_delta = processed_output
        if text_delta:
            logger.debug(
                f"[EMIT] Handler {self.handler_id}: NEW SEGMENT '{text_delta[:30]}...' "
                f"({beg:.2f}s-{end:.2f}s)"
            )

            segment_data = {"start": beg, "end": end, "text": text_delta}

            if self._diarization_enabled and self.store.diarization_processor:
                diarization_proc = self.store.diarization_processor
                if diarization_proc.is_enabled and len(self.full_audio) > 0:
                    sample_rate = self.store.sample_rate
                    if self._audio_for_diarization is None:
                        self._audio_for_diarization = self.full_audio.copy()
                    else:
                        self._audio_for_diarization = np.concatenate(
                            [self._audio_for_diarization, self.full_audio]
                        )

                    if len(self._audio_for_diarization) > 0:
                        try:
                            speaker_segments = diarization_proc.process_audio(
                                self._audio_for_diarization, sample_rate
                            )
                            self._speaker_segments = speaker_segments

                            if speaker_segments:
                                segment_data["speaker"] = speaker_segments[0].speaker
                        except Exception as e:
                            logger.warning(f"Diarization failed: {e}")

            self.segments.append(segment_data)

            # Correctly append the new text delta with a separator
            separator = self.store.separator
            if self.accumulated_transcript and text_delta.strip():
                self.accumulated_transcript += separator + text_delta.strip()
            elif text_delta.strip():
                self.accumulated_transcript = text_delta.strip()

        return AdditionalOutputs(
            self.accumulated_transcript, self.full_audio, self.segments
        )

    def copy(self) -> "RealTimeASRHandler":
        """
        Create a new instance of the handler for a new client connection.

        This is a factory method required by the fastRTC `Stream` object.

        Returns:
            A new RealTimeASRHandler instance
        """
        logger.info(
            f"RealTimeASRHandler.copy() called for master handler [{self.handler_id}]."
        )
        return RealTimeASRHandler(self.store)

    def shutdown(self) -> None:
        """Clean up resources when a client connection is closed."""
        logger.info(f"Handler [{self.handler_id}] shutting down.")
        if self.asr_processor:
            self.asr_processor.init(offset=0.0)
        self.asr_processor = None
