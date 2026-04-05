"""
stimuli/aep.py — Auditory Evoked Potential (AEP) paradigm.

Delivers click or pure-tone stimuli via sounddevice (numpy array playback).
A fixation cross is shown on the stimulus monitor via Qt QPainter.

Timing
------
``sd.play()`` is called, then ``time.perf_counter()`` is recorded
immediately after as T_onset.  The marker is inserted after play() returns.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Literal

import numpy as np

try:
    import sounddevice as _sd
    _SOUNDDEVICE_ERROR: str | None = None
except Exception as _e:
    _sd = None  # type: ignore[assignment]
    _SOUNDDEVICE_ERROR = str(_e)

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QPen, QColor

from neuroep.acquisition.markers import TriggerCode
from neuroep.stimuli.base import BaseParadigm

logger = logging.getLogger(__name__)

# Audio parameters
_SAMPLE_RATE  = 44100
_CLICK_DUR    = 0.001   # seconds
_TONE_DUR     = 0.050   # seconds
_TONE_RAMP    = 0.010   # cosine ramp
_DEFAULT_FREQ = 1000.0  # Hz
_VOLUME       = 0.8

StimulusType = Literal["click", "tone"]


def _make_click(sample_rate: int = _SAMPLE_RATE) -> np.ndarray:
    n = max(1, int(_CLICK_DUR * sample_rate))
    return np.full(n, _VOLUME, dtype=np.float32)


def _make_tone(
    freq:        float = _DEFAULT_FREQ,
    duration:    float = _TONE_DUR,
    ramp:        float = _TONE_RAMP,
    sample_rate: int   = _SAMPLE_RATE,
) -> np.ndarray:
    n      = int(duration * sample_rate)
    n_ramp = int(ramp * sample_rate)
    t      = np.linspace(0, duration, n, endpoint=False)
    wave   = _VOLUME * np.sin(2 * np.pi * freq * t).astype(np.float32)
    ramp_fn = 0.5 * (1 - np.cos(np.pi * np.arange(n_ramp) / n_ramp)).astype(np.float32)
    wave[:n_ramp]  *= ramp_fn
    wave[-n_ramp:] *= ramp_fn[::-1]
    return wave


def _fixation_fn(painter: QPainter) -> None:
    """Draw a white fixation cross centred on a black background."""
    r = painter.device().rect()
    painter.fillRect(r, Qt.GlobalColor.black)
    cx, cy = r.width() // 2, r.height() // 2
    arm = min(r.width(), r.height()) // 20
    pen = QPen(QColor(255, 255, 255))
    pen.setWidth(max(2, arm // 4))
    painter.setPen(pen)
    painter.drawLine(cx - arm, cy, cx + arm, cy)
    painter.drawLine(cx, cy - arm, cx, cy + arm)


class AuditoryEP(BaseParadigm):
    """
    Auditory EP paradigm delivering clicks or pure tones.

    Parameters
    ----------
    board_manager : BoardManager
    n_trials : int
        Number of stimuli (default 100).
    stim_rate : float
        Stimuli per second (default 2.0 Hz).
    stimulus_type : "click" or "tone"
    freq_hz : float
        Tone frequency in Hz (ignored for clicks).
    screen : int
        Qt screen index (default 1).
    """

    def __init__(
        self,
        board_manager,
        n_trials:      int          = 100,
        stim_rate:     float        = 2.0,
        stimulus_type: StimulusType = "click",
        freq_hz:       float        = _DEFAULT_FREQ,
        screen:        int          = 1,
        parent                      = None,
    ) -> None:
        super().__init__(
            board_manager = board_manager,
            n_trials      = n_trials,
            stim_rate     = stim_rate,
            screen        = screen,
            parent        = parent,
        )
        self._stim_type = stimulus_type
        self._freq_hz   = freq_hz

    def _run_trial_loop(self) -> None:
        if _sd is None:
            raise RuntimeError(
                f"sounddevice failed to load: {_SOUNDDEVICE_ERROR}\n"
                "Run: pip install sounddevice"
            )
        sd = _sd

        if self._stim_type == "click":
            waveform     = _make_click()
            trigger_code = TriggerCode.AEP_CLICK
        else:
            waveform     = _make_tone(self._freq_hz)
            trigger_code = TriggerCode.AEP_TONE_STANDARD

        # Show fixation cross
        self._flip(_fixation_fn)
        if not self._wait(1.0):
            return

        iti_base = self._inter_trial_interval()

        for trial in range(1, self._n_trials + 1):
            if self._stop_event.is_set():
                break

            # ── CRITICAL TIMING ──────────────────────────────────────────
            sd.play(waveform, samplerate=_SAMPLE_RATE, blocking=False)
            t_onset = time.perf_counter()
            self._send_marker(trigger_code, t_onset)
            # ─────────────────────────────────────────────────────────────

            self.trial_completed.emit(trial)
            logger.debug("AuditoryEP trial %d  T=%.4f s", trial, t_onset)

            jitter = random.uniform(-0.1, 0.1) * iti_base
            if not self._wait(iti_base + jitter):
                break

        self._flip(None)
