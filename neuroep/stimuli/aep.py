"""
stimuli/aep.py — Auditory Evoked Potential (AEP) paradigm.

Delivers click or pure-tone stimuli via PsychoPy Sound with exclusive
(low-latency) audio mode.  The N1-P2 complex in frontal/central channels
(Fz, C3, C4) is the primary target response.

Stimulus types
--------------
- Click : broadband, 1 ms duration — maximal N1 amplitude
- Tone  : pure sine at a specified frequency, 50 ms, 10 ms cosine ramps

Timing
------
``sound.play()`` is called, then ``core.getTime()`` is recorded immediately
after as T_onset.  The marker is inserted after play() returns.
Note: PsychoPy latencyMode=3 (exclusive) minimises audio latency on
      supported drivers.  On macOS, use latencyMode=1 (normal) if exclusive
      mode is not available.
"""

from __future__ import annotations

import logging
import random
from typing import Literal

import numpy as np

from neuroep.acquisition.markers import TriggerCode
from neuroep.stimuli.base import BaseParadigm

logger = logging.getLogger(__name__)

# Audio parameters
_SAMPLE_RATE  = 44100    # Hz
_CLICK_DUR    = 0.001    # seconds (1 ms)
_TONE_DUR     = 0.050    # seconds (50 ms)
_TONE_RAMP    = 0.010    # cosine ramp at onset/offset (ms)
_DEFAULT_FREQ = 1000.0   # Hz (standard audiometric tone)
_VOLUME       = 0.8      # 0.0–1.0

StimulusType = Literal["click", "tone"]


def _make_click(sample_rate: int = _SAMPLE_RATE) -> np.ndarray:
    """Return a 1 ms rectangular click as a float32 array."""
    n = max(1, int(_CLICK_DUR * sample_rate))
    return np.full(n, _VOLUME, dtype=np.float32)


def _make_tone(
    freq:        float = _DEFAULT_FREQ,
    duration:    float = _TONE_DUR,
    ramp:        float = _TONE_RAMP,
    sample_rate: int   = _SAMPLE_RATE,
) -> np.ndarray:
    """Return a pure-tone burst with cosine onset/offset ramps."""
    n       = int(duration * sample_rate)
    n_ramp  = int(ramp * sample_rate)
    t       = np.linspace(0, duration, n, endpoint=False)
    wave    = _VOLUME * np.sin(2 * np.pi * freq * t).astype(np.float32)

    # Apply cosine ramps to prevent clicks at onset/offset
    ramp_fn = 0.5 * (1 - np.cos(np.pi * np.arange(n_ramp) / n_ramp)).astype(np.float32)
    wave[:n_ramp]  *= ramp_fn
    wave[-n_ramp:] *= ramp_fn[::-1]

    return wave


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
        Waveform to deliver.
    freq_hz : float
        Tone frequency in Hz (ignored for clicks).
    screen : int
        PsychoPy screen index (default 1, used for fixation cross only).
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
        from psychopy import core, sound, visual

        win = self._stim_win.win

        # Fixation cross on stimulus monitor (patient looks at screen)
        fixation = visual.TextStim(
            win,
            text      = "+",
            height    = 0.1,
            color     = [1.0, 1.0, 1.0],
            colorSpace = "rgb",
        )

        # Build audio waveform
        if self._stim_type == "click":
            waveform = _make_click()
            trigger_code = TriggerCode.AEP_CLICK
        else:
            waveform = _make_tone(self._freq_hz)
            trigger_code = TriggerCode.AEP_TONE_STANDARD

        # Attempt exclusive (low-latency) mode; fall back to normal
        try:
            snd = sound.Sound(
                waveform,
                sampleRate  = _SAMPLE_RATE,
                latencyMode = 3,
            )
            logger.info("AEP audio: exclusive latency mode.")
        except Exception:  # pylint: disable=broad-except
            snd = sound.Sound(
                waveform,
                sampleRate  = _SAMPLE_RATE,
                latencyMode = 1,
            )
            logger.info("AEP audio: normal latency mode (exclusive unavailable).")

        # Show fixation cross
        fixation.draw()
        win.flip()
        if not self._wait(1.0):
            return

        iti_base = self._inter_trial_interval()

        for trial in range(1, self._n_trials + 1):
            if self._stop_event.is_set():
                break

            # ── CRITICAL TIMING ──────────────────────────────────────────
            snd.play()
            t_onset = core.getTime()        # timestamp after play() starts
            self._send_marker(trigger_code, t_onset)
            # ─────────────────────────────────────────────────────────────

            self.trial_completed.emit(trial)
            logger.debug("AuditoryEP trial %d  T=%.4f s", trial, t_onset)

            # Redraw fixation cross each trial
            fixation.draw()
            win.flip()

            jitter = random.uniform(-0.1, 0.1) * iti_base
            if not self._wait(iti_base + jitter):
                break

        # Blank screen on exit
        win.color = (-1.0, -1.0, -1.0)
        win.flip()
