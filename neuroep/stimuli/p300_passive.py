"""
stimuli/p300_passive.py — Passive auditory oddball P300 paradigm.

Presents a random sequence of standard (frequent) and deviant (rare) tones.
No button press is required — suitable for DoC patients.

Paradigm structure
------------------
- Standard tone : 1000 Hz, 80% probability → TriggerCode.P300_STANDARD
- Deviant tone  : 2000 Hz, 20% probability → TriggerCode.P300_ODDBALL
- The P300 (300–600 ms) is evoked by the rare deviant in frontal/parietal
  channels (Fz, P4, Oz).
- Averaged separately: standard epochs (for comparison) and oddball epochs.

The difference waveform (oddball − standard average) isolates the P300.

Timing
------
Same as AEP: ``sound.play()`` then ``core.getTime()`` then marker.
"""

from __future__ import annotations

import logging
import random

import numpy as np

from neuroep.acquisition.markers import TriggerCode
from neuroep.stimuli.aep import _make_tone, _SAMPLE_RATE
from neuroep.stimuli.base import BaseParadigm

logger = logging.getLogger(__name__)

# Oddball parameters
_STANDARD_FREQ  = 1000.0   # Hz
_DEVIANT_FREQ   = 2000.0   # Hz
_ODDBALL_PROB   = 0.20     # 20% deviant

# Minimum number of standards between two deviants (avoids back-to-back oddballs)
_MIN_STANDARDS_BETWEEN = 2


def _generate_sequence(n_total: int, oddball_prob: float) -> list[bool]:
    """
    Generate a pseudo-random oddball sequence.

    Returns a list of booleans: True = oddball, False = standard.
    Guarantees at least ``_MIN_STANDARDS_BETWEEN`` standards between oddballs
    and that the first stimulus is always a standard.
    """
    sequence: list[bool] = []
    since_last_oddball   = _MIN_STANDARDS_BETWEEN  # start ready for oddball

    for _ in range(n_total):
        if since_last_oddball >= _MIN_STANDARDS_BETWEEN and random.random() < oddball_prob:
            sequence.append(True)
            since_last_oddball = 0
        else:
            sequence.append(False)
            since_last_oddball += 1

    return sequence


class PassiveP300(BaseParadigm):
    """
    Passive auditory oddball P300 paradigm.

    Counts *accepted oddball epochs* toward the target; standard epochs
    are collected in parallel for the difference waveform.

    Parameters
    ----------
    board_manager : BoardManager
    n_trials : int
        Total stimuli to deliver (default 250 → ~50 oddballs at 20%).
    stim_rate : float
        Stimuli per second (default 1.0 Hz for DoC).
    screen : int
        PsychoPy screen index (default 1).
    """

    def __init__(
        self,
        board_manager,
        n_trials:  int   = 250,
        stim_rate: float = 1.0,
        screen:    int   = 1,
        parent           = None,
    ) -> None:
        super().__init__(
            board_manager = board_manager,
            n_trials      = n_trials,
            stim_rate     = stim_rate,
            screen        = screen,
            parent        = parent,
        )

    def _run_trial_loop(self) -> None:
        from psychopy import core, sound, visual

        win = self._stim_win.win

        # Fixation cross
        fixation = visual.TextStim(
            win,
            text       = "+",
            height     = 0.1,
            color      = [1.0, 1.0, 1.0],
            colorSpace = "rgb",
        )

        # Build standard and deviant audio
        std_wave = _make_tone(_STANDARD_FREQ)
        dev_wave = _make_tone(_DEVIANT_FREQ)

        def _make_sound(wave: np.ndarray):
            try:
                return sound.Sound(wave, sampleRate=_SAMPLE_RATE, latencyMode=3)
            except Exception:  # pylint: disable=broad-except
                return sound.Sound(wave, sampleRate=_SAMPLE_RATE, latencyMode=1)

        std_sound = _make_sound(std_wave)
        dev_sound = _make_sound(dev_wave)

        # Generate trial sequence
        sequence = _generate_sequence(self._n_trials, _ODDBALL_PROB)

        # Initial fixation
        fixation.draw()
        win.flip()
        if not self._wait(1.0):
            return

        iti_base = self._inter_trial_interval()
        n_oddballs = 0

        for trial, is_oddball in enumerate(sequence, start=1):
            if self._stop_event.is_set():
                break

            snd          = dev_sound if is_oddball else std_sound
            trigger_code = TriggerCode.P300_ODDBALL if is_oddball else TriggerCode.P300_STANDARD

            # ── CRITICAL TIMING ──────────────────────────────────────────
            snd.play()
            t_onset = core.getTime()
            self._send_marker(trigger_code, t_onset)
            # ─────────────────────────────────────────────────────────────

            if is_oddball:
                n_oddballs += 1

            self.trial_completed.emit(trial)
            logger.debug(
                "P300 trial %d  %s  T=%.4f s  (oddballs so far: %d)",
                trial,
                "ODDBALL" if is_oddball else "standard",
                t_onset,
                n_oddballs,
            )

            fixation.draw()
            win.flip()

            jitter = random.uniform(-0.1, 0.1) * iti_base
            if not self._wait(iti_base + jitter):
                break

        win.color = (-1.0, -1.0, -1.0)
        win.flip()
        logger.info(
            "PassiveP300 complete: %d trials, %d oddballs delivered.",
            trial, n_oddballs,
        )
