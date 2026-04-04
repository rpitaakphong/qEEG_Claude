# NeuroEP Studio — Master Project Prompt
# Paste this into VS Code Copilot Chat / AI assistant when starting each module.
# Keep this file open as a reference throughout development.

---

## Project overview

You are helping build **NeuroEP Studio** — a Python desktop application for recording,
stimulating, and analysing auditory and visual evoked potentials (EP) using an OpenBCI
Cyton+Daisy EEG board. The software is designed for two use cases:

1. **Research mode** — testing on healthy subjects (including the developer themselves)
2. **Clinical DoC mode** — Disorders of Consciousness patients who cannot cooperate,
   press buttons, or fixate their gaze

The application must be production-quality, well-commented, and maintainable.
Do not write placeholder stubs — write complete, working implementations.

---

## Hardware

| Component | Detail |
|---|---|
| EEG board | OpenBCI Cyton + Daisy (16 channels) |
| Sample rate | 125 Hz (Cyton+Daisy runs at 125 Hz, NOT 250 Hz) |
| Connection | USB dongle → serial port (Windows: COM3, Linux: /dev/ttyUSB0) |
| Stimulus screen | Secondary monitor (screen index 1), fullscreen, driven by PsychoPy |
| Control screen | Primary monitor (screen index 0), PyQt6 GUI |
| Photodiode | Connected to Cyton analog input pin D11 for timing validation |
| BrainFlow SDK | Used for all board communication — never use pyserial directly |

---

## Tech stack — use exactly these libraries, no substitutions

| Purpose | Library |
|---|---|
| Board communication | `brainflow` |
| GUI framework | `PyQt6` |
| Fast EEG plotting | `pyqtgraph` |
| Stimulus delivery | `psychopy` |
| Signal processing | `scipy`, `numpy` |
| EP analysis | `mne` |
| Trigger sync | `pylsl` (Lab Streaming Layer) |
| Data export | `mne` (EDF+), `numpy` (CSV), `matplotlib` (PNG) |

---

## Project file structure

```
neuroeP/
├── main.py                        ← Entry point. Launches PyQt6 app, shows connect dialog.
├── config.py                      ← All constants: board params, filter defaults, paths.
├── requirements.txt
│
├── acquisition/
│   ├── __init__.py
│   ├── board.py                   ← BrainFlow board manager, ring buffer, marker injection.
│   └── markers.py                 ← LSL outlet, timestamp logging, trigger codes enum.
│
├── stimuli/
│   ├── __init__.py
│   ├── base.py                    ← PsychoPy window on screen 2, shared timing utilities.
│   ├── vep_pattern.py             ← Checkerboard pattern-reversal paradigm.
│   ├── vep_flash.py               ← Full-field flash VEP (DoC mode).
│   ├── aep.py                     ← Auditory click / pure tone stimulus.
│   └── p300_passive.py            ← Passive auditory oddball (no button required).
│
├── processing/
│   ├── __init__.py
│   ├── filters.py                 ← Real-time IIR filter chain (HP, LP, notch) with SOS.
│   ├── epochs.py                  ← Epoch extraction, baseline correction.
│   ├── artifact.py                ← Amplitude rejection, ICA wrapper.
│   ├── averaging.py               ← Running average, SNR computation per epoch.
│   └── components.py              ← Peak detection: P100, N1, P2, P300.
│
├── validation/
│   ├── __init__.py
│   ├── synthetic_test.py          ← Mode A: synthetic board pipeline validation.
│   ├── photodiode_test.py         ← Mode B: photodiode hardware jitter measurement.
│   └── squarewave_test.py         ← Mode C: Cyton internal square wave timing test.
│
├── ui/
│   ├── __init__.py
│   ├── theme.py                   ← PyQt6 dark/light stylesheet, colour palette.
│   ├── main_window.py             ← QMainWindow: layout, menu bar, status bar.
│   ├── connect_dialog.py          ← Board connection dialog (port, board type, test).
│   ├── eeg_panel.py               ← Live 16-channel scrolling EEG viewer (pyqtgraph).
│   ├── control_sidebar.py         ← Paradigm selector, filter sliders, run/stop button.
│   ├── averaging_panel.py         ← Real-time EP averaging plots + save/export buttons.
│   └── validation_window.py       ← Timing validation dialog with histogram.
│
└── output/
    ├── __init__.py
    ├── exporter.py                ← Save PNG (matplotlib), CSV (numpy), EDF+ (mne).
    └── report.py                  ← Per-session clinical summary text/PDF.
```

---

## config.py — complete contents (reference this in every module)

```python
BOARD_SAMPLE_RATE   = 125       # Hz
N_CHANNELS          = 16
SERIAL_PORT         = "COM3"    # adjust per machine

CHANNEL_NAMES = [
    "Fp1","Fp2","F3","F4","C3","C4","P3","P4",
    "O1","O2","F7","F8","T3","T4","Fz","Oz",
]

CHANNEL_COLORS = [
    "#1D9E75","#534AB7","#D85A30","#378ADD",
    "#639922","#D4537E","#BA7517","#5DCAA5",
    "#7F77DD","#F0997B","#85B7EB","#97C459",
    "#ED93B1","#EF9F27","#9FE1CB","#AFA9EC",
]

EP_CHANNELS = {
    "VEP":  [8, 9, 15],    # O1, O2, Oz
    "AEP":  [14, 4, 5],    # Fz, C3, C4
    "P300": [14, 7, 15],   # Fz, P4, Oz
}

DISPLAY_SECONDS     = 5
DISPLAY_FPS         = 30
CHUNK_SIZE          = 10

DEFAULT_HP_HZ       = 1.0
DEFAULT_LP_HZ       = 40.0
DEFAULT_NOTCH_HZ    = 50.0
DEFAULT_SENSITIVITY = 50.0      # µV

EPOCH_PRE_MS        = 100
EPOCH_POST_MS       = 500
BASELINE_MS         = 100
ARTIFACT_UV         = 100.0

PHOTODIODE_CHANNEL  = 15
PHOTODIODE_THRESH   = 50.0
TIMING_N_TRIALS     = 50

OUTPUT_DIR          = "sessions"
```

---

## Requirement 1 — Live 16-channel EEG viewer (ui/eeg_panel.py)

### What it does
Displays all 16 EEG channels as scrolling waveforms in real time.
Each channel gets its own coloured trace stacked vertically.
Sensitivity, high-pass, low-pass, and notch controls update the display live
without dropping or resetting the stream.

### Implementation spec

**Widget class:** `EEGPanel(QWidget)`

**Rendering:**
- Use a single `pyqtgraph.PlotWidget` with `useOpenGL=True` for GPU-accelerated rendering
- Create 16 `PlotCurveItem` objects, one per channel
- Stack channels vertically by adding an offset: `offset = (15 - ch_idx) * channel_spacing`
- `channel_spacing` is controlled by the sensitivity slider (larger sensitivity = smaller spacing)
- X-axis: time in seconds, scrolling window of DISPLAY_SECONDS
- Y-axis: hidden — channel labels are drawn as `TextItem` on the left margin
- Background: `#0e1117` (near-black, easier on eyes for long sessions)
- Each trace uses its corresponding colour from `CHANNEL_COLORS`

**Data flow:**
```
BoardManager.get_new_samples()
    → FilterChain.process(chunk)
    → RingBuffer.push(filtered_chunk)
    → QTimer fires at DISPLAY_FPS → read RingBuffer.snapshot() → update 16 PlotCurveItems
```

**Controls (live — no restart required):**
| Control | Widget | Effect |
|---|---|---|
| Sensitivity | QSlider 5–200 µV | Changes channel_spacing, rescales Y |
| High-pass | QSlider 0.1–10 Hz | Calls FilterChain.set_highpass(), resets state |
| Low-pass | QSlider 10–100 Hz | Calls FilterChain.set_lowpass(), resets state |
| Notch | QComboBox: Off/50Hz/60Hz | Calls FilterChain.set_notch() |

**Performance requirements:**
- Must render at 30 fps with 16 channels at 125 Hz without dropping frames
- Do NOT use matplotlib for the live view — pyqtgraph only
- Use `setClipToView(True)` and `setDownsampling(auto=True)` on each curve item

**Channel labels:**
- Left of the plot, aligned with each trace's zero line
- Text colour matches trace colour
- Font: monospace 10pt

---

## Requirement 2 — Real-time averaging panel (ui/averaging_panel.py)

### What it does
Three side-by-side sub-panels below the EEG viewer, updating after every accepted epoch:

**Sub-panel A — Averaged EP waveform**
- X-axis: time in ms from −EPOCH_PRE_MS to +EPOCH_POST_MS
- Y-axis: µV, baseline-corrected
- Draws every previous average as a faint trace (opacity decreasing with age) so you
  can watch the waveform converge — like watching noise cancel in real time
- Current (latest) average drawn as a bold trace in the paradigm colour
- Vertical dashed line at t=0 (stimulus onset)
- Horizontal dashed line at 0 µV
- Auto-detected component markers: P100, N1, P2, P300 — labelled with latency in ms
- Epoch counter in top-right: "47 / 100 epochs"
- Buttons: "Save PNG", "Export CSV", "Clear"

**Sub-panel B — Timing jitter histogram**
- X-axis: offset in ms (from −10 to +10 ms)
- Y-axis: trial count
- Updates after each timing validation trial
- Shows mean ± SD as text overlay
- Colour: amber if SD > 2ms (warning), green if SD < 2ms (acceptable)

**Sub-panel C — SNR growth curve**
- X-axis: epoch number
- Y-axis: SNR in dB
- Plots SNR after each epoch to show the √N improvement
- Horizontal dashed line at 6 dB (minimum acceptable for EP interpretation)
- When SNR crosses 6 dB, line turns green

### Save / export
- **Save PNG:** `matplotlib.pyplot.savefig(path, dpi=150)` — saves the averaged EP waveform
  with a clean white background, axis labels, component markers, session metadata as title
- **Export CSV:** `numpy.savetxt` — columns: time_ms, amplitude_uV. Header row with
  session ID, date, paradigm, n_epochs, channel name, P100/N1/P300 latency and amplitude

---

## Requirement 3 — UI/UX design (ui/main_window.py, ui/theme.py, ui/control_sidebar.py)

### Overall layout
```
┌─────────────────────────────────────────────────────────────┐
│  Top bar: logo · connection status · impedance · session ID  │
├──────────────┬──────────────────────────────────────────────┤
│              │                                              │
│   Sidebar    │         Live 16-channel EEG viewer          │
│   220px      │         (pyqtgraph, dark background)        │
│              │                                              │
├──────────────┴──────────────────────────────────────────────┤
│  Averaging panel: [EP waveform] [Jitter histogram] [SNR]    │
│  220px tall                                                  │
├─────────────────────────────────────────────────────────────┤
│  Status bar: board · SR · accepted · rejected · P100 · time  │
└─────────────────────────────────────────────────────────────┘
```

### Sidebar sections (top to bottom)
1. **Paradigm selector** — four buttons, radio-style (only one active at a time):
   - Pattern VEP (purple accent) — research/self-test
   - Flash VEP (blue accent) — DoC clinical mode
   - Auditory EP (teal accent) — click/tone
   - P300 Passive (amber accent) — oddball, no button
   
   When DoC mode paradigms (Flash VEP, P300 Passive) are selected, a red "DoC mode"
   badge appears and target epoch count auto-reduces to 50.

2. **Signal controls** — sliders as described in Req 1

3. **Protocol settings**
   - Target epochs: QSpinBox 50–300
   - Stimulus rate: QDoubleSpinBox 0.5–4.0 Hz
   - Eye tested: QComboBox Left / Right / Both / N/A
   - Subject ID: QLineEdit (alphanumeric)

4. **Action buttons**
   - "▶ Start session" — large, primary colour, full sidebar width
   - "■ Stop" — appears and replaces Start when running
   - "⚡ Timing validation" — amber outlined button, opens validation window

### Theme (ui/theme.py)
Implement as a `QApplication.setStyleSheet()` call with a single stylesheet string.

Colour tokens:
```
Background primary:   #0f1117   (EEG panel, deep dark)
Background secondary: #1a1d27   (sidebar, panels)
Background tertiary:  #242736   (cards, hover states)
Border:               #2e3148
Text primary:         #e8e6de
Text secondary:       #9a9891
Accent purple:        #534AB7   (VEP, primary actions)
Accent teal:          #1D9E75   (AEP, success states)
Accent amber:         #BA7517   (P300, warnings)
Accent blue:          #378ADD   (Flash VEP, info)
Danger red:           #D85A30   (artifact rejection, errors)
```

Provide both a dark theme (default) and a light theme toggle in the View menu.

### Connect dialog (ui/connect_dialog.py)
Shown on startup before the main window appears.
Fields:
- Board type: QComboBox — "Cyton+Daisy (16ch)", "Synthetic (testing)"
- Serial port: QLineEdit pre-filled from config.SERIAL_PORT
- "Scan ports" button: auto-detects available COM ports using `serial.tools.list_ports`
- "Connect" button: attempts BoardManager.connect(), shows progress spinner
- "Test connection" button: connects synthetic board, runs 3 seconds, disconnects

---

## Requirement 4 — Timing validation (validation/)

### Why this matters
Evoked potential latency measurements are only meaningful if the time between
stimulus delivery and the EEG marker is accurate and consistent.
A 10ms jitter in trigger timing directly degrades the averaged EP by smearing
the waveform. Target: mean jitter < 2ms, SD < 1ms.

### Three validation modes

#### Mode A — Synthetic pipeline test (validation/synthetic_test.py)
**Purpose:** Validate that the Python processing pipeline preserves timing.
**No hardware required.**

```
1. Connect BoardIds.SYNTHETIC_BOARD
2. Start stream
3. Record precise timestamp T1 = time.perf_counter()
4. Call BoardShim.insert_marker(99) 
5. Wait 200ms
6. Fetch data: board.get_board_data()
7. Find marker 99 in the marker channel
8. Convert marker sample index → timestamp T2 using sample rate
9. Jitter = T2 - T1 (in ms)
10. Repeat 50 times, compute mean ± SD, plot histogram
```

Acceptable result: mean < 1ms, SD < 0.5ms.
If worse: there is a bug in marker injection or timestamp conversion.

#### Mode B — Hardware photodiode test (validation/photodiode_test.py)
**Purpose:** Measure true end-to-end latency including display refresh.
**Requires:** Photodiode wired to Cyton analog input pin D11 (channel index 15 in data).

```
1. Connect real Cyton+Daisy board
2. Start PsychoPy fullscreen window on screen 2
3. For each trial (N=50):
   a. Show black background (2 seconds)
   b. T_software = psychopy core.getTime()
   c. Flash full white screen for one refresh frame (win.flip())
   d. Insert marker 1 into BrainFlow marker channel
   e. Wait 500ms (photodiode response window)
   f. Fetch data buffer
   g. Find photodiode onset in channel 15: first sample crossing PHOTODIODE_THRESH
   h. Convert sample index → time
   i. Jitter = (photodiode onset time) - (marker time)
4. Plot histogram of 50 jitter values
5. Report mean ± SD in ms
```

Acceptable result: mean < 5ms, SD < 2ms.
If mean > 10ms: likely missing a monitor refresh cycle — adjust PsychoPy flip timing.
If SD > 5ms: timing is unreliable — check USB load, close background processes.

#### Mode C — Cyton square wave test (validation/squarewave_test.py)
**Purpose:** Validate BrainFlow marker channel timing relative to EEG data channel.
**Requires:** Real Cyton board (not Daisy required for this test).

```
1. Connect board
2. Send serial command to activate internal test signal on channel 1:
   BoardShim.config_board("x1060110X")  ← square wave at ~8 Hz on ch1
3. Insert software marker simultaneously with a known square wave edge
4. Detect square wave edge in ch1 data (threshold crossing)
5. Measure offset between edge and marker
6. Repeat 50 times
```

### Validation results UI (ui/validation_window.py)
- QDialog, modal
- Three tabs: one per validation mode
- Each tab: run button, progress bar, live-updating histogram (pyqtgraph BarGraphItem)
- Summary table: mean, SD, min, max, n_trials, pass/fail verdict
- "Export results" button → CSV

---

## Processing pipeline — critical implementation details

### Filter chain (processing/filters.py)
- Use `scipy.signal.butter` with `output='sos'` — NOT ba coefficients (numerically unstable)
- Use `scipy.signal.sosfilt` with `zi` (initial conditions) carried between chunks
- Filter state `zi` shape: `(n_sections, 2, n_channels)` — one zi per channel
- When the user adjusts a filter cutoff, rebuild the SOS coefficients and reset zi to
  `sosfilt_zi(sos)` — do NOT try to adapt zi smoothly, just reset it
- High-pass: Butterworth order 4
- Low-pass:  Butterworth order 4
- Notch:     `scipy.signal.iirnotch`, Q=30, converted to SOS via `zpk2sos`

### Epoch extraction (processing/epochs.py)
```
pre_samples  = int(EPOCH_PRE_MS  / 1000 * BOARD_SAMPLE_RATE)   # 12 samples
post_samples = int(EPOCH_POST_MS / 1000 * BOARD_SAMPLE_RATE)   # 62 samples
epoch_len    = pre_samples + post_samples                        # 75 samples total

On marker received at sample index M:
    epoch = ring_buffer[:, M - pre_samples : M + post_samples]  # (16, 75)
    baseline_mean = epoch[:, :pre_samples].mean(axis=1, keepdims=True)
    epoch_bc = epoch - baseline_mean   ← baseline-corrected epoch
```

### Artifact rejection (processing/artifact.py)
```
Simple amplitude threshold (first pass):
    if np.any(np.abs(epoch_bc) > ARTIFACT_UV):
        reject = True   ← discard this epoch

Future: ICA (mne.preprocessing.ICA) for ocular artifact removal — Phase 2
```

### Running average (processing/averaging.py)
```
Keep a running sum and count:
    running_sum   += epoch_bc          # (16, epoch_len)
    epoch_count   += 1
    current_avg    = running_sum / epoch_count

SNR after N epochs:
    signal_power = np.var(current_avg, axis=1)        # across time
    noise_power  = np.var(epoch_bc - current_avg, axis=1)
    snr_db       = 10 * np.log10(signal_power / noise_power)
```

---

## Stimulus timing — critical constraint

**PsychoPy must send the marker AFTER win.flip() returns, not before.**
`win.flip()` blocks until the display refresh actually occurs.
This means the timestamp recorded after flip() is the true stimulus onset time.

```python
# CORRECT:
win.flip()
T_onset = core.getTime()
board.insert_marker(trigger_code)

# WRONG (marker sent before screen updates):
board.insert_marker(trigger_code)
win.flip()
```

For auditory stimuli, use `psychopy.sound.Sound` with `latencyMode=3` (exclusive mode)
and record onset with `sound.play(); T_onset = core.getTime()`.

---

## Trigger codes (acquisition/markers.py)

```python
class TriggerCode(IntEnum):
    VEP_PATTERN_REVERSAL = 1
    VEP_FLASH            = 2
    AEP_CLICK            = 3
    AEP_TONE_STANDARD    = 4
    AEP_TONE_ODDBALL     = 5
    P300_STANDARD        = 6
    P300_ODDBALL         = 7
    SESSION_START        = 10
    SESSION_END          = 11
    EPOCH_REJECTED       = 20
    TIMING_TEST          = 99
```

---

## DoC clinical mode — special constraints

When Flash VEP or P300 Passive is selected:
- Target epoch count auto-sets to 50 (patients fatigue quickly)
- Stimulus rate auto-sets to maximum 1 Hz (gentler)
- "Eye tested" field becomes mandatory before Start is enabled
- A "Clinical note" QTextEdit appears for free-text documentation
- Session automatically stops at target epoch count (no manual stop needed)
- The output folder is named `{date}_{subject_id}_DoC_{paradigm}`
- A warning dialog appears if any channel impedance is > 10 kΩ

---

## Error handling requirements

- Board disconnection mid-session: catch `BrainFlowError`, show QMessageBox,
  offer to save partial data before exiting
- PsychoPy window crash: catch Exception in stimulus thread, emit Qt signal
  to main thread to stop session cleanly
- Artifact rejection > 30%: show yellow warning in status bar
  "High rejection rate — check electrodes"
- Filter cutoffs out of range: clamp silently, update slider display to clamped value

---

## When writing any module, follow these rules

1. Every class and public method gets a docstring.
2. Use type hints on all function signatures.
3. No bare `except:` — always catch specific exceptions.
4. Thread safety: ring buffer, marker queue, and epoch queue must use `threading.Lock`
   or `queue.Queue` — never access them from two threads without synchronisation.
5. PyQt signals must be used to pass data from background threads to the UI thread —
   never call Qt widgets directly from a non-main thread.
6. All file paths use `pathlib.Path`, not string concatenation.
7. Filter state is never silently dropped — if a filter update resets zi, log it.

---

## How to use this prompt in VS Code

To build a specific module, paste this entire document into Copilot Chat, then add:

> "Now implement `acquisition/board.py` in full, following all specs above.
>  Include the RingBuffer class, BoardManager class, and all methods described."

Or for the UI:

> "Now implement `ui/eeg_panel.py` — the live 16-channel EEG viewer widget —
>  following all specs above. Use pyqtgraph. Include the filter controls."

Always build and test in this order:
  Phase 1: acquisition/board.py → ui/eeg_panel.py → main.py  (live EEG viewer)
  Phase 2: processing/filters.py → processing/epochs.py → processing/averaging.py
  Phase 3: stimuli/vep_pattern.py → acquisition/markers.py  (first EP paradigm)
  Phase 4: ui/averaging_panel.py  (real-time EP plots)
  Phase 5: validation/ modules    (timing tests)
  Phase 6: stimuli/vep_flash.py, stimuli/aep.py, stimuli/p300_passive.py
  Phase 7: output/exporter.py, output/report.py
