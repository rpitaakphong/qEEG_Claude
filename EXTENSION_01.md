# NeuroEP Studio — Extension Prompt 01
# "Education Panel + Paradigm-Aware Channel Display"
#
# This is an EXTENSION to the completed MASTER_PROMPT.md.
# Paste MASTER_PROMPT.md first in Copilot Chat, then paste this file.
# Do not repeat or rewrite anything from the master prompt.
#
# Say to Copilot:
# "The master prompt describes the completed project. Now implement the
#  following extensions exactly as specified below."

---

## Extension A — Layout change in ui/main_window.py

### Current layout (what was built)
```
┌─────────────────────────────────────────────────────┐
│  Top bar                                            │
├──────────────┬──────────────────────────────────────┤
│              │                                      │
│   Sidebar    │   Live EEG viewer (full width)       │
│   220px      │                                      │
├──────────────┴──────────────────────────────────────┤
│  Averaging panel                                    │
├─────────────────────────────────────────────────────┤
│  Status bar                                         │
└─────────────────────────────────────────────────────┘
```

### New layout (what to build)
```
┌─────────────────────────────────────────────────────────────────┐
│  Top bar                                                        │
├──────────────┬──────────────────────┬──────────────────────────┤
│              │                      │                          │
│   Sidebar    │  Education panel     │  Live EEG viewer         │
│   220px      │  280px               │  (fills remaining width) │
│              │  (upper left col)    │                          │
│              │                      │                          │
├──────────────┴──────────────────────┴──────────────────────────┤
│  Averaging panel                                                │
├─────────────────────────────────────────────────────────────────┤
│  Status bar                                                     │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation

In `ui/main_window.py`, replace the two-column `QSplitter` with a
three-column `QSplitter(Qt.Orientation.Horizontal)`:

```python
self.h_splitter = QSplitter(Qt.Orientation.Horizontal)
self.h_splitter.addWidget(self.sidebar)           # col 0 — 220px fixed
self.h_splitter.addWidget(self.education_panel)   # col 1 — 280px default
self.h_splitter.addWidget(self.eeg_panel)         # col 2 — stretches

self.h_splitter.setSizes([220, 280, 900])
self.h_splitter.setCollapsible(0, False)   # sidebar cannot be collapsed
self.h_splitter.setCollapsible(1, True)    # education panel CAN be collapsed
self.h_splitter.setCollapsible(2, False)   # EEG panel cannot be collapsed
```

The education panel is collapsible so the user can hide it during recording
to give the EEG panel maximum width. Add a small toggle button (◀ ▶)
on the splitter handle or at the top of the education panel.

---

## Extension B — Create ui/education_panel.py (new file)

### Overview
Three stacked sections in a `QWidget` with `QVBoxLayout`, connected to the
paradigm selector via Qt signal. The panel updates entirely when paradigm changes.

```python
class EducationPanel(QWidget):
    def __init__(self, parent=None): ...
    def on_paradigm_changed(self, key: str) -> None:
        self.electrode_map.set_paradigm(key)
        self.description_card.set_paradigm(key)
        self.waveform_sketch.set_paradigm(key)
```

Connect in `main_window.py`:
```python
self.sidebar.paradigm_changed.connect(self.education_panel.on_paradigm_changed)
```

---

### B1 — ElectrodeMapWidget

`QPainter`-based scalp map. All drawing in `paintEvent()`. No pyqtgraph, no matplotlib.

#### Electrode positions (normalised, origin = head centre, +y = down)
```python
ELECTRODE_POSITIONS = {
    "Fp1": (-0.30, -0.83), "Fp2": ( 0.30, -0.83),
    "F7":  (-0.71, -0.46), "F3":  (-0.38, -0.50),
    "Fz":  ( 0.00, -0.46), "F4":  ( 0.38, -0.50),
    "F8":  ( 0.71, -0.46),
    "T3":  (-0.95,  0.00), "C3":  (-0.50,  0.00),
    "Cz":  ( 0.00,  0.00), "C4":  ( 0.50,  0.00),
    "T4":  ( 0.95,  0.00),
    "T5":  (-0.71,  0.46), "P3":  (-0.38,  0.50),
    "Pz":  ( 0.00,  0.46), "P4":  ( 0.38,  0.50),
    "T6":  ( 0.71,  0.46),
    "O1":  (-0.30,  0.83), "Oz":  ( 0.00,  0.90),
    "O2":  ( 0.30,  0.83),
}
```

Convert to pixel coords dynamically in `paintEvent`:
```python
cx, cy = self.rect().center().x(), self.rect().center().y()
r = min(self.rect().width(), self.rect().height()) * 0.42
px = cx + norm_x * r
py = cy + norm_y * r
```

#### Role definitions per paradigm
```python
PARADIGM_CHANNELS = {
    "VEP_PATTERN": {
        "color":       "#534AB7",
        "color_light": "#EEEDFE",
        "essential":   ["Oz", "O1", "O2"],
        "useful":      ["Pz"],
        "artifact":    ["Fz", "Fp1", "Fp2"],
    },
    "VEP_FLASH": {
        "color":       "#378ADD",
        "color_light": "#E6F1FB",
        "essential":   ["Oz", "O1", "O2"],
        "useful":      ["Pz"],
        "artifact":    ["Fz", "Fp1", "Fp2"],
    },
    "AEP": {
        "color":       "#1D9E75",
        "color_light": "#E1F5EE",
        "essential":   ["Cz", "Fz"],
        "useful":      ["C3", "C4"],
        "artifact":    ["Fp1", "Fp2", "Oz"],
    },
    "P300": {
        "color":       "#BA7517",
        "color_light": "#FAEEDA",
        "essential":   ["Pz", "Cz", "Fz"],
        "useful":      ["P3", "P4"],
        "artifact":    ["Fp1", "Fp2"],
    },
}
```

#### Drawing style
```
role = "essential" → filled circle r=13, paradigm colour fill, white label
role = "useful"    → outlined circle r=11, paradigm colour border,
                     light fill, coloured label
role = "artifact"  → dashed outline r=9, grey border, grey fill, grey label
role = "inactive"  → small solid circle r=6, neutral grey, tiny grey label
```

Always call `painter.setRenderHint(QPainter.RenderHint.Antialiasing)`.

Draw a small legend at the bottom of the widget (12px text):
```
● Essential   ◉ Useful   ○ Artifact   · Inactive
```

Electrode tooltip on hover: override `mouseMoveEvent`, detect which electrode
the cursor is over, call `QToolTip.showText()` with name + role description.

`setMinimumSize(260, 260)`

---

### B2 — ParadigmDescriptionCard

Simple `QFrame` with coloured left border accent, a title label, optional
DoC badge, and a read-only description text area.

```python
PARADIGM_INFO = {
    "VEP_PATTERN": {
        "title": "Pattern reversal VEP",
        "doc_mode": False,
        "text": (
            "Checkerboard reversal at 2 Hz on the stimulus screen.\n"
            "Subject must fixate the central cross — fixation is mandatory.\n"
            "Each reversal is one epoch. Average 100 epochs minimum.\n\n"
            "Key component: P100 at Oz (95–115 ms normal range).\n"
            "Compare O1 vs O2: >50% amplitude asymmetry = significant.\n\n"
            "Not suitable for DoC patients who cannot fixate."
        ),
    },
    "VEP_FLASH": {
        "title": "Flash VEP",
        "doc_mode": True,
        "text": (
            "Full-field white flash at 1 Hz. No fixation required.\n"
            "Eyes must be open — document eye state before starting.\n"
            "Target 100–150 epochs (noisier than pattern VEP).\n\n"
            "Key component: P2 at Oz (~120 ms).\n"
            "Confirms visual cortex receives light — not a test of awareness.\n\n"
            "Primary paradigm for DoC / vegetative state patients."
        ),
    },
    "AEP": {
        "title": "Auditory evoked potential",
        "doc_mode": False,
        "text": (
            "Broadband click at 70–80 dB SPL via insert earphones.\n"
            "Rate: 1–2 clicks/sec. Eyes closed to prevent VEP bleed.\n"
            "Target 100 epochs.\n\n"
            "Key components: N1 (~100 ms) and P2 (~180 ms) at Cz/Fz.\n"
            "Right ear → larger response at left C3 (and vice versa).\n\n"
            "Passive — works in DoC patients. Confirms auditory cortex\n"
            "receives and processes sound."
        ),
    },
    "P300": {
        "title": "P300 passive oddball",
        "doc_mode": True,
        "text": (
            "80% standard tone (1000 Hz) / 20% oddball (2000 Hz).\n"
            "No button press. Fully passive — patient just lies still.\n"
            "Target: 40–60 clean oddball epochs (~250 total stimuli).\n\n"
            "Analyse the DIFFERENCE wave: oddball minus standard.\n"
            "Key component: P300 at Pz (~300 ms) in difference wave.\n\n"
            "P300 without behavioural response = evidence of covert\n"
            "cognition. Basis of Owen/Monti DoC awareness paradigms."
        ),
    },
}
```

Style the card with a coloured left border that changes with paradigm:
```python
card.setStyleSheet(f"""
    QFrame {{
        border-left: 3px solid {paradigm_color};
        background: palette(base);
        padding: 10px 12px;
        border-radius: 0px;
    }}
""")
```

DoC badge: a `QLabel` with stylesheet:
```python
badge.setStyleSheet(
    "background: #FCEBEB; color: #791F1F; border-radius: 8px;"
    "padding: 2px 8px; font-size: 11px; font-weight: 500;"
)
badge.setVisible(info["doc_mode"])
```

---

### B3 — ReferenceWaveformWidget

`QPainter`-based idealised EP waveform. All drawing in `paintEvent()`.

#### Axes
- X: −100 ms to +500 ms. Vertical dashed line at t=0 (stimulus onset).
- Y: amplitude. Horizontal dashed line at 0 µV.
- Convention: NEGATIVE UP, POSITIVE DOWN (standard clinical EP convention).
  Label the Y-axis: "µV (neg ↑)"
- Label X-axis: "−100", "0", "+200", "+400" in ms.

#### Convert time/amplitude to pixel coords
```python
def t_to_x(t_ms):
    # t range: -100 to +500 ms = 600 ms total
    return plot_left + (t_ms + 100) / 600 * plot_width

def amp_to_y(amp_uv, scale_uv=15):
    # centre of plot = 0 µV. Negative = up (smaller y).
    return plot_cy - (amp_uv / scale_uv) * (plot_height / 2)
```

#### Waveform definitions — draw as QPainterPath cubic bezier curves

**VEP_PATTERN and VEP_FLASH** (primary channel: Oz):
```python
# Control points: (time_ms, amplitude_uv)
# Negative amplitude = drawn upward (negative up convention)
VEP_WAVE = [
    (-100, 0), (0, 0),          # flat baseline
    (50,  1.0),                 # slight early positive
    (75,  3.5),                 # N75  — small positive bump
    (100, -12.0),               # P100 — large negative dip (drawn UP) ← highlight
    (145,  5.0),                # N145 — positive return
    (200,  1.0),                # settling
    (400,  0.0),                # back to baseline
]
# Highlight the P100 peak with a vertical amber dashed marker
# Label: "P100  95–115 ms"
```

**AEP** (primary channel: Cz):
```python
AEP_WAVE = [
    (-100, 0), (0, 0),
    (50,   0.5),
    (100, -8.0),                # N1 — drawn UP ← highlight
    (150, -3.0),
    (180,  6.0),                # P2 — drawn DOWN
    (250, -2.0),                # N2
    (400,  0.0),
]
# Highlight N1. Label: "N1  90–110 ms"
```

**P300** — draw TWO waveforms on the same plot:
```python
# Standard tone (grey, dashed line):
P300_STANDARD = [
    (-100, 0), (0, 0),
    (100, -6.0),                # N1
    (180,  4.0),                # P2
    (400,  0.0),                # flat — no P300
]

# Oddball tone (paradigm colour, solid, bold):
P300_ODDBALL = [
    (-100, 0), (0, 0),
    (100, -6.5),                # N1 (same as standard)
    (180,  4.5),                # P2 (same as standard)
    (300, -14.0),               # P300 — large negative = drawn UP ← highlight
    (450,  0.0),
]
# Draw standard first (grey dashed), then oddball on top (colour solid)
# Small legend inside the plot: "— Standard  — Oddball"
# Highlight P300. Label: "P300  280–350 ms"
```

#### Component annotation (for the highlighted peak)
```python
# 1. Vertical amber dashed line from peak point down to x-axis
# 2. Text label above or beside the line: component name + latency range
# 3. Small double-headed vertical arrow showing amplitude
```

#### Styling
- Background: `QColor("#0e1117")` (matches EEG panel dark background)
- Grid lines: `QColor("#1e2130")`, 0.5px
- Zero lines: `QColor("#2e3148")`, 1px dashed
- Waveform: 2px, paradigm colour
- Annotation lines: `QColor("#BA7517")`, 1px dashed (amber)
- Annotation text: `QColor("#EF9F27")`, 10px
- Axis labels: `QColor("#5F5E5A")`, 10px

`setMinimumSize(260, 160)`

---

## Extension C — Paradigm-aware channel filtering in ui/eeg_panel.py

### What changes
The live EEG viewer currently shows all 16 channels at all times.

After this extension, when a paradigm is selected:
- **EP-relevant channels** (essential + useful + artifact for that paradigm)
  are shown at **full brightness and normal height**
- **Non-relevant channels** are shown **dimmed** (30% opacity) and at
  **half height** to save space, but remain visible

The user can override this with a "Show all channels" toggle in the sidebar.

### Channel visibility map

Add this to `config.py`:

```python
# All channels relevant to each paradigm (essential + useful + artifact combined)
# These get full brightness. Everything else gets dimmed.
PARADIGM_VISIBLE_CHANNELS = {
    "VEP_PATTERN": ["Oz", "O1", "O2", "Pz", "Fz", "Fp1", "Fp2"],
    "VEP_FLASH":   ["Oz", "O1", "O2", "Pz", "Fz", "Fp1", "Fp2"],
    "AEP":         ["Cz", "Fz", "C3", "C4", "Fp1", "Fp2", "Oz"],
    "P300":        ["Pz", "Cz", "Fz", "P3", "P4", "Fp1", "Fp2"],
    "ALL":         None,   # None = show all at full brightness
}
```

### Changes to EEGPanel class

Add a new public method:
```python
def set_paradigm(self, paradigm_key: str) -> None:
    """
    Update channel display based on selected paradigm.
    Called by main_window when paradigm_changed signal fires.
    """
    self._active_paradigm = paradigm_key
    visible = PARADIGM_VISIBLE_CHANNELS.get(paradigm_key)

    for i, ch_name in enumerate(CHANNEL_NAMES):
        curve  = self._curves[i]       # PlotCurveItem for this channel
        label  = self._ch_labels[i]    # TextItem for channel name

        if visible is None or ch_name in visible:
            # Full brightness
            color = QColor(CHANNEL_COLORS[i])
            color.setAlphaF(1.0)
            curve.setPen(pg.mkPen(color, width=1))
            label.setColor(color)
            self._channel_height[i] = self._base_height       # full row height
        else:
            # Dimmed — still visible but clearly secondary
            color = QColor(CHANNEL_COLORS[i])
            color.setAlphaF(0.25)
            curve.setPen(pg.mkPen(color, width=0.5))
            label.setColor(color)
            self._channel_height[i] = self._base_height * 0.5 # half height

    self._recompute_offsets()   # recalculate Y offsets with new heights
    self._plot_widget.update()
```

Add `_recompute_offsets()`:
```python
def _recompute_offsets(self) -> None:
    """
    Recalculate the vertical Y offset for each channel trace based on
    current channel heights (which vary when paradigm filtering is active).
    Full-height channels take normal spacing; dimmed channels are compacted.
    """
    running_y = 0
    for i in range(N_CHANNELS - 1, -1, -1):   # bottom to top (ch 0 = bottom)
        self._offsets[i] = running_y
        running_y += self._channel_height[i] + self._ch_gap
```

### Show all channels toggle

In `ui/control_sidebar.py`, add a `QCheckBox` below the paradigm buttons:
```python
self.show_all_cb = QCheckBox("Show all 16 channels")
self.show_all_cb.setChecked(False)
self.show_all_cb.stateChanged.connect(self._on_show_all_toggled)

def _on_show_all_toggled(self, state: int) -> None:
    key = "ALL" if state == Qt.CheckState.Checked.value else self._current_paradigm
    self.paradigm_changed.emit(key)
```

When "Show all" is checked, emit `"ALL"` as the paradigm key — the EEG panel
treats `None` visible list as full brightness for every channel.

---

## Signal chain summary (all three extensions together)

```
User clicks paradigm button in sidebar
    → sidebar emits: paradigm_changed(str)
        → education_panel.on_paradigm_changed(key)
            → electrode_map.set_paradigm(key)      [repaints head diagram]
            → description_card.set_paradigm(key)   [updates text + badge]
            → waveform_sketch.set_paradigm(key)    [repaints EP sketch]
        → eeg_panel.set_paradigm(key)
            [dims non-relevant channels, compacts their height]
```

All four connections are made in `main_window.py`:
```python
self.sidebar.paradigm_changed.connect(self.education_panel.on_paradigm_changed)
self.sidebar.paradigm_changed.connect(self.eeg_panel.set_paradigm)
```

---

## How to use this in VS Code

Open Copilot Chat and paste in this order:

  1. MASTER_PROMPT.md   (the full completed master prompt)
  2. This file          (EXTENSION_01.md)

Then say:

> "The master prompt describes the completed codebase. Now implement
>  these three extensions:
>  A) Layout change in main_window.py — add education panel as third column
>  B) Create ui/education_panel.py with ElectrodeMapWidget,
>     ParadigmDescriptionCard, and ReferenceWaveformWidget
>  C) Update ui/eeg_panel.py to dim and compact non-paradigm channels
>  Implement all three in sequence. Start with A, confirm it compiles,
>  then move to B, then C."

Build order within each extension:
  A: modify h_splitter in main_window.py → test layout renders
  B1: ElectrodeMapWidget → test all 20 electrodes appear correctly
  B2: ParadigmDescriptionCard → test paradigm switching updates text
  B3: ReferenceWaveformWidget → build VEP first, then AEP, then P300
  B4: assemble EducationPanel, connect signal
  C: add set_paradigm() to EEGPanel, add toggle checkbox to sidebar
