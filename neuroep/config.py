"""
config.py — Global constants for NeuroEP Studio.
All modules import from here; never hard-code these values elsewhere.
"""

# ── Board ──────────────────────────────────────────────────────────────────
BOARD_SAMPLE_RATE: int = 125        # Hz — Cyton+Daisy runs at 125 Hz only
N_CHANNELS: int = 16
SERIAL_PORT: str = "COM3"           # Adjust per machine (Linux: /dev/ttyUSB0)

# ── Channel configuration ──────────────────────────────────────────────────
CHANNEL_NAMES: list[str] = [
    "Fp1", "Fp2", "F3",  "F4",
    "C3",  "C4",  "P3",  "P4",
    "O1",  "O2",  "F7",  "F8",
    "T3",  "T4",  "Fz",  "Oz",
]

CHANNEL_COLORS: list[str] = [
    "#1D9E75", "#534AB7", "#D85A30", "#378ADD",
    "#639922", "#D4537E", "#BA7517", "#5DCAA5",
    "#7F77DD", "#F0997B", "#85B7EB", "#97C459",
    "#ED93B1", "#EF9F27", "#9FE1CB", "#AFA9EC",
]

# Channels of interest per paradigm (0-based indices into CHANNEL_NAMES)
EP_CHANNELS: dict[str, list[int]] = {
    "VEP":  [8, 9, 15],    # O1, O2, Oz
    "AEP":  [14, 4, 5],    # Fz, C3, C4
    "P300": [14, 7, 15],   # Fz, P4, Oz
}

# ── Display ────────────────────────────────────────────────────────────────
DISPLAY_SECONDS: int = 5
DISPLAY_FPS: int = 30
CHUNK_SIZE: int = 10                # Samples pulled per timer tick

# ── Filter defaults ────────────────────────────────────────────────────────
DEFAULT_HP_HZ: float = 1.0
DEFAULT_LP_HZ: float = 40.0
DEFAULT_NOTCH_HZ: float = 50.0
DEFAULT_SENSITIVITY: float = 50.0   # µV peak-to-peak half-range for display

# ── Epoch / averaging ──────────────────────────────────────────────────────
EPOCH_PRE_MS: int = 100
EPOCH_POST_MS: int = 500
BASELINE_MS: int = 100
ARTIFACT_UV: float = 100.0          # Amplitude threshold for rejection

# ── Timing validation ──────────────────────────────────────────────────────
PHOTODIODE_CHANNEL: int = 15        # Cyton analog input D11 → index 15
PHOTODIODE_THRESH: float = 50.0     # ADC units for photodiode onset detection
TIMING_N_TRIALS: int = 50

# ── Output ─────────────────────────────────────────────────────────────────
OUTPUT_DIR: str = "sessions"
