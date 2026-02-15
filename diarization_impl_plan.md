# Diarization Implementation Plan

Based on `diarization.spec`, here is the detailed implementation plan:

---

## Phase 1: Infrastructure (Tasks 1-5)

### Task 1: Create Diarization Backend Interface
**File**: `asr_interface/backends/diarization/__init__.py`
- Create abstract base class `DiarizationBackend`
- Define interface methods: `process(audio) -> List[SpeakerSegment]`
- Define `SpeakerSegment` dataclass with `start`, `end`, `speaker_id`

### Task 2: Add DiarizationConfig to ASRConfig
**File**: `asr_interface/core/config.py`
- Add `DiarizationConfig` class with `enabled`, `backend`, `min_speakers`, `max_speakers`
- Add `diarization` field to `ASRConfig`

### Task 3: Create Diarization Processor
**File**: `asr_interface/core/diarization_processor.py`
- Create `DiarizationProcessor` class
- Implement alignment algorithm (match ASR word timestamps to diarization segments)
- Handle buffering for real-time mode

### Task 4: Register Diarization Backend
**File**: `asr_interface/backends/diarization/registry.py`
- Create registry for diarization backends
- Register PyanoteAdapter when available

### Task 5: Update Store for Diarization
**File**: `asr_interface/core/store.py`
- Add `diarization_enabled` state
- Add `diarization_config` storage

---

## Phase 2: Pyanote Adapter (Tasks 6-8)

### Task 6: Install Dependencies
- Add `pyanote.audio` to `pyproject.toml`
- Document required models (pyanote/pipeline.yaml)

### Task 7: Implement PyanoteAdapter
**File**: `asr_interface/backends/diarization/pyanote_adapter.py`
- Implement `DiarizationBackend` interface
- Load pyannote pipeline
- Process audio and return speaker segments

### Task 8: Test Offline Diarization
- Create test script with sample multi-speaker audio
- Verify speaker segmentation output format
- Benchmark latency

---

## Phase 3: ASR Integration (Tasks 9-12)

### Task 9: Modify ASR Processor
**File**: `asr_interface/backends/whisper_online_processor.py`
- Add diarization processor reference
- Call diarization after ASR transcription
- Merge speaker labels with word timestamps

### Task 10: Update Output Format
- Modify `process_iter()` to include speaker info when diarization enabled
- Add `speaker` field to segment output

### Task 11: Handle Real-Time Buffering
- Implement sliding window for diarization (10-30s)
- Handle late speaker assignments

### Task 12: Batch Mode Support
- Add `finish()` method to process remaining audio
- Handle end-of-recording diarization

---

## Phase 4: UI/UX (Tasks 13-16)

### Task 13: Add Diarization Toggle
**File**: `index.html`
- Add checkbox for "Enable Speaker Diarization"
- Add to settings/model selection area

### Task 14: Update Frontend JavaScript
**File**: `scripts/audio_recording_script.js`
- Add diarization toggle state
- Pass diarization setting to backend
- Handle speaker-colored response

### Task 15: Style Speaker Labels
**File**: `index.html` or `scripts/`
- Define CSS colors for SPEAKER_1 through SPEAKER_6
- Style transcript with speaker labels and colors

### Task 16: Handle Toggle in API
**File**: `asr_interface/web/server.py`
- Add `enable_diarization` to load model request
- Store diarization preference per session

---

## Phase 5: Testing & Polish (Tasks 17-20)

### Task 17: Unit Tests
- Test alignment algorithm
- Test PyanoteAdapter
- Test integration with ASR

### Task 18: Integration Tests
- Test full pipeline with sample audio
- Test real-time streaming with diarization

### Task 19: Error Handling Tests
- Test fallback when diarization fails
- Test with invalid audio

### Task 20: Documentation
- Update README with diarization usage
- Document latency expectations
- Document available backends

---

## Task Dependencies

```
Task 1 → Task 2 → Task 3 → Task 5
              ↓
Task 4 ← Task 2
      ↓
Task 6 ← Task 4
      ↓
Task 7 ← Task 6
      ↓
Task 8 ← Task 7
      ↓
Task 9 ← Task 3 + Task 8
      ↓
Task 10 ← Task 9
      ↓
Task 11 ← Task 10
      ↓
Task 12 ← Task 10
      ↓
Task 13 ← Task 12
      ↓
Task 14 ← Task 13
      ↓
Task 15 ← Task 14
      ↓
Task 16 ← Task 15
      ↓
Task 17-20 ← All previous
```

---

## Estimated Complexity

| Phase | Tasks | Complexity |
|-------|-------|------------|
| Infrastructure | 5 | Medium |
| Pyanote Adapter | 3 | High |
| ASR Integration | 4 | High |
| UI/UX | 4 | Low |
| Testing | 4 | Medium |

---

## Quick Start (Minimum Viable)

To get diarization working quickly:

1. **Task 1-2**: Create interface + config
2. **Task 6-7**: Implement Pyanote adapter
3. **Task 9-10**: Basic ASR integration
4. **Task 13-15**: Simple toggle + display

Skip real-time buffering (Task 11) for MVP - process at recording end only.
