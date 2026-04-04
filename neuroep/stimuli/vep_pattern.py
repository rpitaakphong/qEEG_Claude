"""
stimuli/vep_pattern.py — Checkerboard pattern-reversal VEP paradigm.

Each trial shows a high-contrast black-and-white checkerboard that reverses
phase (black squares become white and vice versa) at the stimulus onset.
The P100 component in occipital channels (O1, Oz, O2) is the target response.

Stimulus parameters
-------------------
- Check size  : ~1° of visual angle (configurable)
- Contrast    : 100% (black / white)
- Field size  : full screen
- Reversal    : instantaneous (single win.flip())
- ISI         : 1 / stim_rate seconds ± 10% jitter to avoid entrainment

Timing
------
The marker is inserted AFTER win.flip() returns, per the critical timing
constraint in stimuli/base.py.
"""

from __future__ import annotations

import logging
import random

import numpy as np

from neuroep.acquisition.markers import TriggerCode
from neuroep.stimuli.base import BaseParadigm

logger = logging.getLogger(__name__)

# Default checkerboard parameters
_N_CHECKS_X = 8     # columns of squares across the screen
_N_CHECKS_Y = 6     # rows of squares


class PatternVEP(BaseParadigm):
    """
    Checkerboard pattern-reversal VEP paradigm.

    Each stimulus event:
    1. Reverses the checkerboard phase (flip black ↔ white).
    2. Flips the display buffer (``win.flip()``).
    3. Records T_onset = ``core.getTime()`` after the flip.
    4. Inserts marker ``TriggerCode.VEP_PATTERN_REVERSAL``.
    5. Waits for the inter-stimulus interval (with ±10% jitter).

    Parameters
    ----------
    board_manager : BoardManager
        Connected board for marker injection.
    n_trials : int
        Number of pattern reversals (default 100).
    stim_rate : float
        Average reversals per second (default 2.0 Hz).
    check_size : int
        Number of checks along the shorter screen axis (default 8).
    screen : int
        PsychoPy screen index for the stimulus monitor (default 1).
    """

    def __init__(
        self,
        board_manager,
        n_trials:   int   = 100,
        stim_rate:  float = 2.0,
        check_size: int   = _N_CHECKS_X,
        screen:     int   = 1,
        parent            = None,
    ) -> None:
        super().__init__(
            board_manager = board_manager,
            n_trials      = n_trials,
            stim_rate     = stim_rate,
            screen        = screen,
            parent        = parent,
        )
        self._check_size = check_size
        self._phase      = 0    # 0 or 1 — alternates each trial

    # ── Trial loop ─────────────────────────────────────────────────────────

    def _run_trial_loop(self) -> None:
        """
        Create the checkerboard stimulus and run *n_trials* pattern reversals.
        """
        from psychopy import core, visual

        win   = self._stim_win.win
        clock = self._stim_win.clock

        # Build two pre-rendered checkerboard textures (phase 0 and phase 1)
        board_a, board_b = self._build_checkerboards(win)

        # Show the initial board before starting the trial clock
        board_a.draw()
        win.flip()
        self._phase = 0

        # Brief fixation period
        if not self._wait(1.0):
            return

        iti_base = self._inter_trial_interval()

        for trial in range(1, self._n_trials + 1):
            if self._stop_event.is_set():
                break

            # Select the OPPOSITE phase for this reversal
            self._phase = 1 - self._phase
            stimulus = board_b if self._phase == 1 else board_a

            # Draw the reversed board to the back buffer
            stimulus.draw()

            # ── CRITICAL TIMING ──────────────────────────────────────────
            t_onset = self._stim_win.flip()          # blocks until vsync
            self._send_marker(                       # marker AFTER flip
                TriggerCode.VEP_PATTERN_REVERSAL, t_onset
            )
            # ─────────────────────────────────────────────────────────────

            self.trial_completed.emit(trial)
            logger.debug("PatternVEP trial %d  T=%.4f s", trial, t_onset)

            # Inter-stimulus interval with ±10% jitter
            jitter = random.uniform(-0.1, 0.1) * iti_base
            if not self._wait(iti_base + jitter):
                break

        # Return to blank screen on exit
        win.color = (-1.0, -1.0, -1.0)
        win.flip()

    # ── Checkerboard builder ───────────────────────────────────────────────

    def _build_checkerboards(
        self, win
    ) -> tuple:
        """
        Create two ``visual.ImageStim`` objects with opposite checkerboard
        phases using a numpy-generated texture.

        Returns
        -------
        tuple[visual.ImageStim, visual.ImageStim]
            (phase_0_stim, phase_1_stim)
        """
        from psychopy import visual

        n_cols = self._check_size
        n_rows = max(1, int(n_cols * win.size[1] / win.size[0]))

        # Tile: True = white (+1), False = black (−1)
        tile = np.indices((n_rows, n_cols)).sum(axis=0) % 2  # 0 or 1
        tex_a = (tile.astype(np.float32) * 2.0 - 1.0)       # −1 and +1
        tex_b = -tex_a                                        # reversed phase

        # Stretch to a power-of-2 size for OpenGL texture compatibility
        size = 512
        from PIL import Image
        img_a = np.array(
            Image.fromarray(
                ((tex_a + 1.0) * 127.5).astype(np.uint8)
            ).resize((size, size), Image.NEAREST)
        )
        img_b = np.array(
            Image.fromarray(
                ((tex_b + 1.0) * 127.5).astype(np.uint8)
            ).resize((size, size), Image.NEAREST)
        )

        # Convert 0–255 to −1 … +1 float for PsychoPy
        tex_a_f = (img_a.astype(np.float32) / 127.5) - 1.0
        tex_b_f = (img_b.astype(np.float32) / 127.5) - 1.0

        stim_a = visual.ImageStim(
            win, image=tex_a_f, size=(2.0, 2.0), units="norm"
        )
        stim_b = visual.ImageStim(
            win, image=tex_b_f, size=(2.0, 2.0), units="norm"
        )
        return stim_a, stim_b
