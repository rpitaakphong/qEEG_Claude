"""
stimuli/p300_passive.py — Passive auditory oddball P300 paradigm.

Presents standard (frequent) and deviant (rare) tones via sounddevice.
Fixation cross on stimulus monitor via Qt QPainter.

Paradigm structure
------------------
- Standard : 1000 Hz, 80% probability → TriggerCode.P300_STANDARD
- Deviant  : 2000 Hz, 20% probability → TriggerCode.P300_ODDBALL
"""

from __future__ import annotations

import logging
import random
import time

import numpy as np

try:
    import sounddevice as _sd
    _SOUNDDEVICE_ERROR: str | None = None
except Exception as _e:
    _sd = None  # type: ignore[assignment]
    _SOUNDDEVICE_ERROR = str(_e)

from neuroep.acquisition.markers import TriggerCode
from neuroep.stimuli.aep import _make_tone, _fixation_fn, _SAMPLE_RATE
from neuroep.stimuli.base import BaseParadigm

logger = logging.getLogger(__name__)

_STANDARD_FREQ         = 1000.0
_DEVIANT_FREQ          = 2000.0
_ODDBALL_PROB          = 0.20
_MIN_STANDARDS_BETWEEN = 2


def _generate_sequence(n_total: int, oddball_prob: float) -> list[bool]:
    sequence: list[bool] = []
    since_last = _MIN_STANDARDS_BETWEEN

    for _ in range(n_total):
        if since_last >= _MIN_STANDARDS_BETWEEN and random.random() < oddball_prob:
            sequence.append(True)
            since_last = 0
        else:
            sequence.append(False)
            since_last += 1

    return sequence


class PassiveP300(BaseParadigm):
    """
    Passive auditory oddball P300 paradigm.

    Parameters
    ----------
    board_manager : BoardManager
    n_trials : int
        Total stimuli (default 250 → ~50 oddballs at 20%).
    stim_rate : float
        Stimuli per second (default 1.0 Hz).
    screen : int
        Qt screen index (default 1).
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
        if _sd is None:
            raise RuntimeError(
                f"sounddevice failed to load: {_SOUNDDEVICE_ERROR}\n"
                "Run: pip install sounddevice"
            )
        sd = _sd

        std_wave = _make_tone(_STANDARD_FREQ)
        dev_wave = _make_tone(_DEVIANT_FREQ)
        sequence = _generate_sequence(self._n_trials, _ODDBALL_PROB)

        self._flip(_fixation_fn)
        if not self._wait(1.0):
            return

        iti_base   = self._inter_trial_interval()
        n_oddballs = 0

        for trial, is_oddball in enumerate(sequence, start=1):
            if self._stop_event.is_set():
                break

            wave         = dev_wave if is_oddball else std_wave
            trigger_code = TriggerCode.P300_ODDBALL if is_oddball else TriggerCode.P300_STANDARD

            # ── CRITICAL TIMING ──────────────────────────────────────────
            sd.play(wave, samplerate=_SAMPLE_RATE, blocking=False)
            t_onset = time.perf_counter()
            self._send_marker(trigger_code, t_onset)
            # ─────────────────────────────────────────────────────────────

            if is_oddball:
                n_oddballs += 1

            self.trial_completed.emit(trial)
            logger.debug(
                "P300 trial %d  %s  T=%.4f s  (oddballs: %d)",
                trial,
                "ODDBALL" if is_oddball else "standard",
                t_onset,
                n_oddballs,
            )

            jitter = random.uniform(-0.1, 0.1) * iti_base
            if not self._wait(iti_base + jitter):
                break

        self._flip(None)
        logger.info(
            "PassiveP300 complete: %d trials, %d oddballs delivered.",
            trial, n_oddballs,
        )
