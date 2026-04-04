"""
stimuli/vep_flash.py — Full-field flash VEP paradigm (DoC clinical mode).

A single frame of full-screen white flashed against a black background.
Suitable for patients who cannot fixate or cooperate — no pattern recognition
is required, only the brainstem/cortical response to a light transient.

Clinical DoC constraints (enforced by the UI, documented here for completeness)
--------------------------------------------------------------------------------
- Stimulus rate  ≤ 1 Hz (default)
- Target epochs  = 50
- Eye tested field is mandatory
- Session stops automatically at target epoch count

Timing
------
Identical to PatternVEP: marker inserted AFTER win.flip() returns.
"""

from __future__ import annotations

import logging
import random

from neuroep.acquisition.markers import TriggerCode
from neuroep.stimuli.base import BaseParadigm

logger = logging.getLogger(__name__)

# Flash duration in seconds (one frame at 60 Hz ≈ 16.7 ms)
# Using 2 frames to ensure the flash is visible on slower monitors
_FLASH_FRAMES = 2


class FlashVEP(BaseParadigm):
    """
    Full-field flash VEP paradigm for DoC clinical assessment.

    Each trial:
    1. Draws a full-screen white rectangle.
    2. Flips (``win.flip()``) — T_onset captured after.
    3. Inserts ``TriggerCode.VEP_FLASH``.
    4. Holds for *_FLASH_FRAMES* display frames.
    5. Returns to black and waits for ISI.

    Parameters
    ----------
    board_manager : BoardManager
    n_trials : int
        Number of flashes (default 50 for DoC mode).
    stim_rate : float
        Flashes per second (default 1.0 Hz).
    screen : int
        PsychoPy screen index (default 1).
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
        from psychopy import visual

        win = self._stim_win.win

        # Full-screen white flash stimulus
        flash = visual.Rect(
            win,
            width  = 2.0,
            height = 2.0,
            units  = "norm",
            fillColor = [1.0, 1.0, 1.0],
            lineColor = None,
        )

        # Dark background rect (redrawn each ISI to keep buffer clean)
        blank = visual.Rect(
            win,
            width  = 2.0,
            height = 2.0,
            units  = "norm",
            fillColor = [-1.0, -1.0, -1.0],
            lineColor = None,
        )

        # Initial blank
        blank.draw()
        win.flip()

        if not self._wait(1.0):
            return

        iti_base = self._inter_trial_interval()

        for trial in range(1, self._n_trials + 1):
            if self._stop_event.is_set():
                break

            # Draw flash to back buffer
            flash.draw()

            # ── CRITICAL TIMING ──────────────────────────────────────────
            t_onset = self._stim_win.flip()
            self._send_marker(TriggerCode.VEP_FLASH, t_onset)
            # ─────────────────────────────────────────────────────────────

            self.trial_completed.emit(trial)
            logger.debug("FlashVEP trial %d  T=%.4f s", trial, t_onset)

            # Hold flash for _FLASH_FRAMES additional frames then go blank
            for _ in range(_FLASH_FRAMES - 1):
                flash.draw()
                win.flip()

            blank.draw()
            win.flip()

            # ISI with ±10% jitter
            jitter = random.uniform(-0.1, 0.1) * iti_base
            if not self._wait(iti_base + jitter):
                break

        blank.draw()
        win.flip()
