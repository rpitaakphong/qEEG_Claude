"""
stimuli/vep_flash.py — Full-field flash VEP paradigm (DoC clinical mode).

A single frame of full-screen white flashed against a black background.
Rendering uses Qt QPainter — no OpenGL or PsychoPy required.

Timing
------
Marker is inserted AFTER _flip() returns (per the rule in base.py).
"""

from __future__ import annotations

import logging
import random

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter

from neuroep.acquisition.markers import TriggerCode
from neuroep.stimuli.base import BaseParadigm

logger = logging.getLogger(__name__)

_FLASH_HOLD_S = 0.033   # ~2 frames at 60 Hz


def _white(painter: QPainter) -> None:
    painter.fillRect(painter.device().rect(), Qt.GlobalColor.white)


class FlashVEP(BaseParadigm):
    """
    Full-field flash VEP paradigm for DoC clinical assessment.

    Parameters
    ----------
    board_manager : BoardManager
    n_trials : int
        Number of flashes (default 50).
    stim_rate : float
        Flashes per second (default 1.0 Hz).
    screen : int
        Qt screen index (default 1).
    """

    def __init__(
        self,
        board_manager,
        n_trials:  int   = 50,
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
        # Initial blank
        self._flip(None)

        if not self._wait(1.0):
            return

        iti_base = self._inter_trial_interval()

        for trial in range(1, self._n_trials + 1):
            if self._stop_event.is_set():
                break

            # ── CRITICAL TIMING ──────────────────────────────────────────
            t_onset = self._flip(_white)
            self._send_marker(TriggerCode.VEP_FLASH, t_onset)
            # ─────────────────────────────────────────────────────────────

            self.trial_completed.emit(trial)
            logger.debug("FlashVEP trial %d  T=%.4f s", trial, t_onset)

            # Hold flash briefly then go dark
            if not self._wait(_FLASH_HOLD_S):
                break
            self._flip(None)

            jitter = random.uniform(-0.1, 0.1) * iti_base
            if not self._wait(iti_base - _FLASH_HOLD_S + jitter):
                break

        self._flip(None)
