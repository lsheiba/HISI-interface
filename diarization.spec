# Speaker Diarization Specification

## Project Overview

**Purpose**: Add speaker diarization to separate transcription by speaker in real-time and batch ASR processing.

**Goal**: Enable users to toggle speaker diarization on/off, with clearly separated speaker segments and visual differentiation (colors).

---

## Requirements

### Core Features

1. **Toggle Diarization**
   - Enable/disable diarization via UI checkbox or API parameter
   - Diarization off: standard single-stream transcription
   - Diarization on: multi-speaker transcription with speaker labels

2. **Speaker Separation**
   - Each speaker gets a unique identifier (SPEAKER_1, SPEAKER_2, etc.)
   - Segments clearly delimited by speaker
   - Speaker turns properly ordered in output

3. **Visual Differentiation**
   - Different colors for each speaker in UI
   - Transcript display shows speaker labels with distinct styling
   - Frontend maps speaker_id to color (not backend)

4. **Backend Options** (implement ONE initially)
   - **Pyanote.audio** - Open source, high quality (PRIMARY)
   - WavLM/SpeakerLM - Future consideration
   - Meta Voice Separation - Future consideration

### Non-Functional Requirements

- **Real-time capable**: Process audio in sliding windows, emit with latency
- **Configurable**: Allow selecting backend, number of speakers
- **Minimal latency impact**: Diarization should not significantly delay transcription

### Real-Time Strategy

For real-time streaming, diarization works with inherent latency:

1. **Buffer**: Accumulate 10-30 seconds of audio
2. **Process**: Run diarization on buffered audio
3. **Emit**: Return speaker-segmented transcription
4. **Latency expectation**: 5-15 seconds delay acceptable for real-time

For batch mode (recording end): Process full audio with full accuracy.

---

## Technical Architecture

### Components

1. **Diarization Backend Interface** (`asr_interface/backends/diarization/`)
   - Abstract base class `DiarizationBackend`
   - Implementation: `PyanoteAdapter`

2. **Diarization Processor** (`asr_interface/core/diarization_processor.py`)
   - Handles audio chunking for diarization
   - Merges diarization results with ASR output
   - Alignment strategy: match ASR word timestamps to diarization segments

3. **UI Updates** (`index.html`, `audio_recording_script.js`)
   - Toggle switch for diarization
   - Speaker-colored transcript display (frontend maps speaker_id → color)

4. **API Changes**
   - Add `diarization` field to ASRConfig
   - Add `enable_diarization` parameter to endpoints

---

## API Design

### Configuration

```python
from typing import Literal

class DiarizationConfig:
    enabled: bool = False
    backend: Literal["pyanote"] = "pyanote"  # Start with one backend
    min_speakers: int = 1
    max_speakers: int = 10
```

### Output Format

```json
{
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "text": "Hello everyone",
      "speaker": "SPEAKER_1"
    },
    {
      "start": 2.5,
      "end": 5.0,
      "text": "Thank you",
      "speaker": "SPEAKER_2"
    }
  ],
  "speaker_colors": {
    "SPEAKER_1": "#FF5733",
    "SPEAKER_2": "#33FF57"
  }
}
```

Note: `speaker_colors` only included when diarization enabled. Frontend handles color mapping.

---

## Alignment Strategy

How to merge ASR output with diarization:

1. ASR returns word-level timestamps (start_time, end_time, word)
2. Diarization returns speaker segments (start, end, speaker)
3. For each word: find which speaker segment contains its timestamp
4. Assign speaker to word based on overlapping segment

---

## Error Handling

- **Model load failure**: Log error, disable diarization, continue without it
- **Diarization crash**: Fallback to non-diarized transcription
- **Invalid audio**: Skip diarization, return standard transcript
- **No speakers detected**: Return SPEAKER_1 as default

---

## Implementation Phases

### Phase 1: Infrastructure
- Create diarization module structure
- Define `DiarizationBackend` abstract class
- Add `DiarizationConfig` to ASRConfig

### Phase 2: Pyanote Adapter
- Implement PyanoteAdapter with `pyanote.audio`
- Test offline diarization on sample audio
- Verify speaker segmentation output

### Phase 3: Real-Time Integration
- Integrate with ASR processor
- Implement alignment algorithm
- Handle sliding window processing

### Phase 4: UI/UX
- Add diarization toggle to frontend
- Implement speaker-colored transcript display
- Handle toggle state in API

### Phase 5: Testing & Optimization
- Test with various audio samples (single speaker, multi-speaker, overlapping)
- Optimize window size for latency vs accuracy
- Document latency expectations

---

## Dependencies

### Required (Pyanote)
```toml
pyanote.audio>=3.0.0
torch>=2.0.0
```

### For Future Backends (Not Now)
- `speechbrain` - For WavLM embeddings
- `asteroid-filterbanks` - For Meta models

---

## UI Mockup

```
[✓] Enable Speaker Diarization

Transcript:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[SPEAKER_1] Hello everyone
[SPEAKER_2] Thank you for coming
[SPEAKER_1] Let's get started
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Colors handled by CSS:
.speaker-1 { color: #FF5733; }
.speaker-2 { color: #33FF57; }
```

---

## Backward Compatibility

- Diarization disabled by default
- Existing API responses unchanged when diarization off
- `speaker` field present only when diarization enabled
- No breaking changes to existing endpoints
