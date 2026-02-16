# Refactoring Plan

## Current Architecture Overview

### Backend (Python)
- **asr_interface/core/**: Core protocols, config, and state store
  - `protocols.py`: Defines `ASRProcessor`, `ModelLoader` protocols and `ASRBase` ABC
  - `config.py`: Pydantic models for configuration
  - `store.py`: Thread-safe state management (`ASRComponentsStore`)
  - `diarization_processor.py`: Speaker diarization processing

- **asr_interface/backends/**: ASR model loaders
  - `registry.py`: Model loader registry
  - `whisper_online_processor.py`: Main online ASR processor
  - `whisper_loader.py`, `mlx_whisper_loader.py`, `mlx_audio_loader.py`: Model loaders
  - `diarization/`: Diarization adapters

- **asr_interface/handlers/**: Stream handlers
  - `stream_handler.py`: Real-time ASR handler for WebRTC

- **asr_interface/web/**: FastAPI server
  - `server.py`: Main web server with endpoints
  - `streaming.py`: SSE streaming utilities

### Frontend (JavaScript)
- `scripts/audio_recording_script.js`: Recording mode (WebRTC)
- `scripts/audio_file_script.js`: File upload mode
- `scripts/wavesurfers.js`: Waveform visualization

---

## Identified Issues

### 1. JavaScript Code Duplication (HIGH PRIORITY)

**Problem**: `audio_recording_script.js` and `audio_file_script.js` share ~60% identical code:
- `formatTime()`, `formatDuration()` functions
- Segment table rendering logic
- Timeline update logic
- Speaker legend logic
- Play segment functionality

**Solution**: Create shared utilities module (`scripts/shared_utils.js`)

### 2. Legacy Code Cleanup (MEDIUM PRIORITY)

**Problem**: `legacy/` directory contains old implementations that may no longer be needed:
- `legacy/real_time_asr_server.py`
- `legacy/offline_asr_server.py`
- `legacy/real_time_stream_handler.py`
- `legacy/real_time_asr_protocols.py`
- `legacy/whisper_online.py`
- `legacy/real_time_asr_backend/`

**Solution**: 
- Review and confirm legacy code is unused
- Remove if unnecessary
- Document if still needed

### 3. HTML/JS Typo (LOW PRIORITY)

**Problem**: `start-trasncript-btn` has typo (missing 'i')

**Solution**: Rename to `start-transcript-btn` (requires updating JS references)

### 4. Debug Logging Cleanup (LOW PRIORITY)

**Problem**: Console.log statements left in production code

**Solution**: Remove or wrap in development-only blocks

### 5. CSS Organization (MEDIUM PRIORITY)

**Problem**: `scripts/style.css` is 1247+ lines with mixed concerns

**Solution**: Consider splitting into:
- `style/base.css` - Reset and variables
- `style/components.css` - Component styles
- `style/layout.css` - Layout styles

### 6. Error Handling Consistency (MEDIUM PRIORITY)

**Problem**: Inconsistent error handling patterns between Python and JS

**Solution**: Standardize error response format across API

---

## Refactoring Tasks

### Phase 1: Extract Shared JavaScript Utilities

1. Create `scripts/shared_utils.js`:
   - `formatTime(seconds)`
   - `formatDuration(seconds)`
   - `createSegmentRow(segment)`
   - `updateSpeakerLegend(segments, containerId)`
   - `playSegment(start, end)`

2. Update both scripts to import from shared module

### Phase 2: Fix Typos and Quick Wins

1. Rename `start-trasncript-btn` → `start-transcript-btn`
2. Remove console.log statements from production

### Phase 3: Legacy Cleanup

1. Verify legacy imports are unused
2. Remove or move to deprecated section
3. Update imports if needed

### Phase 4: CSS Refactoring

1. Split CSS into logical sections
2. Add CSS custom properties for theming

---

## Implementation Order

1. **Week 1**: Extract shared JavaScript utilities (Phase 1)
2. **Week 2**: Fix typos and debug cleanup (Phase 2)
3. **Week 3**: Legacy code cleanup (Phase 3)
4. **Week 4**: CSS organization (Phase 4)

---

## Notes

- Maintain backward compatibility during refactoring
- Test thoroughly after each phase
- Keep commits focused and atomic
