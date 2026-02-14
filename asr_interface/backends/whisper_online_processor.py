"""Whisper Online ASR Processor for real-time streaming transcription."""

import logging
import sys
import time
from typing import Any

import numpy as np

from ..core.config import ASRConfig
from ..core.protocols import ASRProcessor

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

SAMPLING_RATE = 16000


class HypothesisBuffer:
    """
    A buffer for managing and stabilizing ASR output in real-time streaming.

    This class handles the complex logic of:
    1. Managing overlapping predictions from consecutive ASR calls
    2. Stabilizing transcripts by finding common prefixes
    3. Avoiding stuttering output by detecting and removing duplicates
    4. Maintaining timing information for each word
    """

    def __init__(self, logfile=sys.stderr):
        self.commited_in_buffer = []  # list which stores finalized words
        self.buffer = []  # stores the previous hypothesis from the ASR
        self.new = []  # holds the current incoming hypothesis

        self.last_commited_time = (
            0  # The end timestamp of the last word that was committed
        )
        self.last_commited_word = None

        self.logfile = logfile

    def insert(self, new, offset):
        """
        Insert new hypothesis with timing offset.

        The offset is the start time of the audio chunk that was processed.
        The ASR will give timestamps relative to this chunk (e.g., from 0.0 seconds).
        This converts those relative timestamps to absolute timestamps in the audio stream.
        """
        # Convert relative timestamps to absolute timestamps
        new = [(a + offset, b + offset, t) for a, b, t in new]
        self.new = [(a, b, t) for a, b, t in new if a > self.last_commited_time - 0.1]

        if len(self.new) >= 1:
            # Handle overlapping predictions
            a, b, t = self.new[0]
            if abs(a - self.last_commited_time) < 1:
                if self.commited_in_buffer:
                    # Search for 1, 2, ..., 5 consecutive words (n-grams) that are identical
                    cn = len(self.commited_in_buffer)
                    nn = len(self.new)
                    for i in range(1, min(min(cn, nn), 5) + 1):  # 5 is the maximum
                        c = " ".join(
                            [self.commited_in_buffer[-j][2] for j in range(1, i + 1)][
                                ::-1
                            ]
                        )
                        tail = " ".join(self.new[j - 1][2] for j in range(1, i + 1))
                        if c == tail:
                            words = []
                            for j in range(i):
                                words.append(repr(self.new.pop(0)))
                            words_msg = " ".join(words)
                            logger.debug(f"removing last {i} words: {words_msg}")
                            break

    def flush(self):
        """
        Returns committed chunk = the longest common prefix of 2 last inserts.
        """
        commit = []
        while self.new:
            na, nb, nt = self.new[0]

            if len(self.buffer) == 0:
                break

            if nt == self.buffer[0][2]:
                commit.append((na, nb, nt))
                self.last_commited_word = nt
                self.last_commited_time = nb
                self.buffer.pop(0)
                self.new.pop(0)
            else:
                break
        self.buffer = self.new
        self.new = []
        self.commited_in_buffer.extend(commit)
        return commit

    def pop_commited(self, time):
        """Remove committed words that are older than the given time."""
        while self.commited_in_buffer and self.commited_in_buffer[0][1] <= time:
            self.commited_in_buffer.pop(0)

    def complete(self):
        """Return the current buffer as complete."""
        return self.buffer

    def force_commit_all(self):
        """Force commit all words in buffer (for final output)."""
        all_words = self.commited_in_buffer + self.buffer + self.new
        self.commited_in_buffer = []
        self.buffer = []
        self.new = []
        return all_words


class OnlineASRProcessor(ASRProcessor):
    """
    Manages the real-time, streaming processing of audio for an ASR model.

    This class acts as the main engine for a streaming ASR system. It is
    responsible for:
    1. Buffering incoming audio chunks.
    2. Calling an ASR backend to transcribe the audio.
    3. Using a HypothesisBuffer to stabilize the ASR's output.
    4. Intelligently managing the audio buffer's size to ensure low latency
       and memory usage in long-running sessions.
    5. Handling errors and ASR failures gracefully.
    """

    SAMPLING_RATE = 16000

    def __init__(
        self,
        asr: Any,
        buffer_trimming: tuple[str, int] = ("segment", 15),
        min_chunk_sec: float = 1.0,
        logfile=sys.stderr,
    ):
        self.asr = asr
        self.logfile = logfile
        self.min_chunk_sec = min_chunk_sec
        self.buffer_trimming_way, self.buffer_trimming_sec = buffer_trimming
        self.max_buffer_sec = 30.0
        self.min_words_for_commit = 2

        self.MAX_CONSECUTIVE_ASR_FAILURES = 3
        self.FALLBACK_TRIM_THRESHOLD_SEC = 5

        self.init()

    def init(self, offset: float = 0.0):
        """
        Reset the processor to a clean initial state.

        This is useful for starting a new audio stream without creating a new
        processor instance.

        Args:
            offset: The initial time offset for the audio stream. Defaults to 0.0.
        """
        logger.debug(f"[INIT] Resetting processor with offset={offset}")
        self.audio_buffer = np.array([], dtype=np.float32)
        self.transcript_buffer = HypothesisBuffer(logfile=self.logfile)
        self.buffer_time_offset = offset
        self.transcript_buffer.last_commited_time = self.buffer_time_offset
        self.commited = []
        self.consecutive_asr_failures = 0
        logger.info(f"[INIT] Processor initialized, buffer empty")

    def insert_audio_chunk(self, audio: np.ndarray):
        """Append a new chunk of audio to the internal buffer."""
        prev_len = len(self.audio_buffer)
        self.audio_buffer = np.append(self.audio_buffer, audio)
        new_len = len(self.audio_buffer)
        logger.debug(
            f"[INSERT] Added {len(audio)} samples, buffer now {new_len} samples "
            f"({new_len / self.SAMPLING_RATE:.2f}s)"
        )

    def process_iter(self) -> tuple[float | None, float | None, str]:
        """
        Perform one complete iteration of the processing loop.

        This method orchestrates the transcription, stabilization, and buffer
        management steps.

        Returns:
            A tuple containing the start time, end time, and text of the newly
            committed transcript segment. Returns (None, None, "") if no new
            segment is committed.
        """
        iter_start = time.perf_counter()

        buffer_duration = len(self.audio_buffer) / self.SAMPLING_RATE
        if buffer_duration < self.min_chunk_sec:
            logger.debug(
                f"[PROCESS_ITER] Buffer too short: {buffer_duration:.2f}s < {self.min_chunk_sec}s min"
            )
            return (None, None, "")

        logger.debug(
            f"[PROCESS_ITER] Processing buffer: {buffer_duration:.2f}s of audio"
        )

        asr_start = time.perf_counter()
        asr_result, asr_success = self._transcribe_audio()
        asr_time = time.perf_counter() - asr_start
        logger.debug(f"[PROCESS_ITER] ASR transcription took {asr_time * 1000:.1f}ms")

        # Stage 3: Stabilize the transcript and get the newly committed part.
        stabilize_start = time.perf_counter()
        committed_words = self._stabilize_transcript(asr_result)
        stabilize_time = time.perf_counter() - stabilize_start
        logger.debug(f"[PROCESS_ITER] Stabilization took {stabilize_time * 1000:.1f}ms")

        # Stage 4: Manage the audio buffer based on the ASR result.
        self._manage_audio_buffer(asr_result, asr_success)

        # Stage 5: Format and return the output.
        output = self._format_output(committed_words)
        total_time = time.perf_counter() - iter_start
        logger.debug(f"[PROCESS_ITER] Total iteration took {total_time * 1000:.1f}ms")

        if output[2]:
            logger.info(
                f"[PROCESS_ITER] Output: '{output[2][:50]}...' ({output[0]:.2f}-{output[1]:.2f})"
            )

        return output

    def _transcribe_audio(self) -> tuple[Any, bool]:
        """
        Transcribe the current audio buffer using the ASR backend.

        Returns:
            A tuple containing the ASR result and a boolean indicating success.
        """
        try:
            prompt = self._get_prompt()

            audio_to_process = self.audio_buffer
            audio_duration = len(audio_to_process) / self.SAMPLING_RATE

            window_size_sec = 3.0
            if audio_duration > window_size_sec:
                window_samples = int(window_size_sec * self.SAMPLING_RATE)
                audio_to_process = audio_to_process[-window_samples:]
                logger.debug(
                    f"[TRANSCRIBE] Processing sliding window: {window_size_sec}s "
                    f"(buffer: {audio_duration:.1f}s)"
                )

            result = self.asr.transcribe(audio_to_process, init_prompt=prompt)

            self.consecutive_asr_failures = 0
            return result, True

        except Exception as e:
            self.consecutive_asr_failures += 1
            logger.warning(
                f"ASR transcription failed (attempt {self.consecutive_asr_failures}): {e}"
            )
            return None, False

    def _stabilize_transcript(self, asr_result: Any) -> list[tuple[float, float, str]]:
        """
        Stabilize the transcript using the hypothesis buffer.

        Args:
            asr_result: The raw ASR result.

        Returns:
            A list of committed word tuples (start_time, end_time, word).
        """
        if asr_result is None:
            return []

        words = self.asr.ts_words(asr_result)
        logger.debug(
            f"[STABILIZE] Got {len(words)} words from ASR: {words[:3] if words else 'none'}, buffer_offset={self.buffer_time_offset}"
        )

        if not words:
            return []

        self.transcript_buffer.insert(words, self.buffer_time_offset)

        committed_words = self.transcript_buffer.flush()

        if not committed_words:
            committed_words = list(words)
            logger.debug(
                f"[STABILIZE] Returning {len(committed_words)} words without stabilization"
            )

        return committed_words

    def _manage_audio_buffer(self, asr_result: Any, asr_success: bool):
        """
        Manage the audio buffer based on the ASR result and success status.
        """
        if asr_success and asr_result is not None:
            # Use the ASR result to trim the buffer intelligently
            self._trim_buffer_by_segment(asr_result)
        else:
            # Apply fallback trimming if ASR failed
            self._apply_fallback_trim(asr_success)

    def _trim_buffer_by_segment(self, asr_result: Any):
        """
        Trim the audio buffer based on ASR segment boundaries.
        """
        buffer_duration = len(self.audio_buffer) / self.SAMPLING_RATE

        if buffer_duration > self.max_buffer_sec:
            trim_to = max(self.buffer_trimming_sec, self.max_buffer_sec - 5)
            self._chunk_at_timestamp(trim_to)
            logger.debug(
                f"[TRIM] Buffer too large ({buffer_duration:.1f}s), trimmed to {trim_to}s"
            )
            return

        try:
            segment_ends = self.asr.segments_end_ts(asr_result)

            if (
                segment_ends
                and len(self.audio_buffer) / self.SAMPLING_RATE
                > self.buffer_trimming_sec
            ):
                for end_ts in reversed(segment_ends):
                    if end_ts <= self.buffer_trimming_sec:
                        self._chunk_at_timestamp(end_ts)
                        break

        except Exception as e:
            logger.warning(f"Failed to trim buffer by segment: {e}")
            print(asr_result)
            self._apply_fallback_trim(True)

    def _apply_fallback_trim(self, asr_success: bool):
        """
        Apply fallback buffer trimming when ASR-based trimming fails.
        """
        buffer_duration = len(self.audio_buffer) / self.SAMPLING_RATE

        if buffer_duration > self.FALLBACK_TRIM_THRESHOLD_SEC:
            # Trim to keep only the last portion of the buffer
            trim_samples = int(self.FALLBACK_TRIM_THRESHOLD_SEC * self.SAMPLING_RATE)
            if len(self.audio_buffer) > trim_samples:
                self.audio_buffer = self.audio_buffer[-trim_samples:]
                self.buffer_time_offset += (
                    buffer_duration - self.FALLBACK_TRIM_THRESHOLD_SEC
                )
                logger.debug(
                    f"Applied fallback buffer trimming, kept last {self.FALLBACK_TRIM_THRESHOLD_SEC}s"
                )

    def _chunk_at_timestamp(self, time: float):
        """
        Trim the audio buffer at a specific timestamp.
        """
        samples_to_keep = int(time * self.SAMPLING_RATE)
        if samples_to_keep < len(self.audio_buffer):
            self.audio_buffer = self.audio_buffer[samples_to_keep:]
            self.buffer_time_offset += time
            logger.debug(f"Trimmed buffer at timestamp {time}s")

    def _get_prompt(self) -> str:
        """
        Get the current prompt for the ASR model.
        """
        if self.commited:
            # Use the last few committed words as context
            last_words = [word[2] for word in self.commited[-10:]]  # Last 10 words
            return " ".join(last_words)
        return ""

    def _format_output(
        self, words: list[tuple[float, float, str]]
    ) -> tuple[float | None, float | None, str]:
        """
        Format the output from a list of word tuples.

        Args:
            words: List of word tuples (start_time, end_time, word).

        Returns:
            A tuple containing (start_time, end_time, text).
        """
        if not words:
            return (None, None, "")

        start_time = words[0][0]
        end_time = words[-1][1]
        text = " ".join(word[2] for word in words)

        # Update the committed words list
        self.commited.extend(words)

        return (start_time, end_time, text)

    def finish(self) -> tuple[float | None, float | None, str]:
        """
        Finalize processing and return any remaining results.

        Returns:
            A tuple containing (start_time, end_time, text) of the final segment.
        """
        if len(self.audio_buffer) > 0:
            asr_result, _ = self._transcribe_audio()
            final_words = self._stabilize_transcript(asr_result)

            remaining_words = self.transcript_buffer.force_commit_all()
            if remaining_words:
                final_words.extend(remaining_words)

            logger.info(f"Final transcript: {final_words}")
            self.init()
            return self._format_output(final_words)

        return (None, None, "")


# def asr_factory(config: ASRConfig) -> tuple[Any, OnlineASRProcessor]:
#     """
#     Create and configure ASR and OnlineASRProcessor instances.

#     This factory creates the appropriate ASR backend based on the configuration
#     and initializes the online processor.

#     Args:
#         config: The ASR configuration containing model parameters.

#     Returns:
#         A tuple containing the initialized ASR backend and the online ASR processor.

#     Raises:
#         ValueError: If an unsupported backend is specified.
#         RuntimeError: If model loading fails.
#     """
#     logger.info(
#         f"Creating ASR factory for model '{config.model}' with backend '{config.backend}'..."
#     )

#     # Import the appropriate backend based on configuration
#     if config.backend == "whisper_timestamped":
#         from .whisper_adapters import WhisperTimestampedAdapter
#         from .whisper_timestamped_loader import WhisperTimestampedProcessor

#         processor = WhisperTimestampedProcessor(
#             model_size=config.model,
#             language=config.lan,
#             min_chunk_sec=config.min_chunk_size,
#         )
#         # Wrap the processor with the legacy-compatible adapter
#         asr = WhisperTimestampedAdapter(processor, original_language=config.lan)

#     elif config.backend == "mlx-whisper":
#         from .mlx_whisper_loader import MLXWhisperProcessor
#         from .whisper_adapters import MLXWhisperAdapter

#         processor = MLXWhisperProcessor(
#             model_size=config.model,
#             language=config.lan,
#             min_chunk_sec=config.min_chunk_size,
#         )
#         # Wrap the processor with the legacy-compatible adapter
#         asr = MLXWhisperAdapter(processor, original_language=config.lan)

#     else:
#         raise ValueError(
#             f"Unsupported ASR backend: '{config.backend}'. "
#             f"Available backends are: whisper_timestamped, mlx-whisper"
#         )

#     # Create the online processor that wraps the ASR backend
#     online = OnlineASRProcessor(
#         asr=asr,
#         buffer_trimming=(config.buffer_trimming, int(config.buffer_trimming_sec)),
#         min_chunk_sec=config.min_chunk_size,
#     )

#     logger.info("ASR factory created successfully")
#     return asr, online
