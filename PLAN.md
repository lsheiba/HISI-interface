# Implementation Plan

## 1. Save Configuration Between Sessions
**Priority: High**

Save user's preferred settings (backend, model, language, streaming, diarization, ASR parameters) to localStorage and restore on page load.

**Tasks:**
- [ ] 1.1 - Add localStorage save on config change in `audio_recording_script.js`
- [ ] 1.2 - Add localStorage load on page init in `audio_recording_script.js`  
- [ ] 1.3 - Restore UI values from saved config

**Files:**
- `scripts/audio_recording_script.js`

---

## 2. Add Speakers Column to Segments Table
**Priority: High**

Display speaker labels in the transcription segments table.

**Tasks:**
- [ ] 2.1 - Add speaker column header to HTML table in `index.html`
- [ ] 2.2 - Style speaker column in `style.css`
- [ ] 2.3 - Verify diarization speaker data displays correctly

**Files:**
- `index.html`
- `scripts/style.css`

---

## 3. Color-Coded Speaker Transcript Display
**Priority: Medium**

Display different speakers' text in different colors. Show list of detected speakers with their assigned colors.

**Tasks:**
- [ ] 3.1 - Define speaker color palette in `style.css`
- [ ] 3.2 - Apply speaker-specific colors to transcript segments
- [ ] 3.3 - Add speaker legend panel showing detected speakers and colors
- [ ] 3.4 - Update JS to assign consistent colors per speaker

**Files:**
- `scripts/style.css`
- `scripts/audio_recording_script.js`
- `index.html`

---

## 4. Transcript Window Expand/Contract
**Priority: Medium**

Make the transcript panel resizable (expandable/collapsible).

**Tasks:**
- [ ] 4.1 - Add toggle button for transcript panel
- [ ] 4.2 - Add CSS transition for expand/collapse animation
- [ ] 4.3 - Save expand state to localStorage
- [ ] 4.4 - Handle responsive layout for mobile

**Files:** 
- `index.html`
- `scripts/style.css`
- `scripts/audio_recording_script.js`

---

## 5. Stop Transcript Functionality
**Priority: High**

Add stop button and control for active transcription sessions.

**Tasks:**
- [ ] 5.1 - Add stop transcript button to UI in `index.html`
- [ ] 5.2 - Add stop button styling in `style.css`
- [ ] 5.3 - Implement stop endpoint in `server.py` (if needed)
- [ ] 5.4 - Handle stop via WebRTC connection close or SSE abort
- [ ] 5.5 - Reset UI state after stop

**Files:**
- `index.html`
- `scripts/style.css`
- `scripts/audio_recording_script.js`
- `asr_interface/web/server.py`

---

## Implementation Order

1. **Week 1**: Items 1, 5 (Config persistence, Stop functionality)
2. **Week 2**: Item 2 (Speakers column)
3. **Week 3**: Item 3 (Color-coded speakers)
4. **Week 4**: Item 4 (Expandable transcript)

---

## Notes

- All UI state should persist via localStorage
- Speaker colors should be consistent across sessions
- Stop should gracefully end WebRTC connection
- Transcript window should remember user's preferred size
