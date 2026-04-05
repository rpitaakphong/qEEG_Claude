"""
stimuli/vep_pattern.py — Checkerboard pattern-reversal VEP paradigm.

Each trial flips a high-contrast black-and-white checkerboard.  Rendering
uses Qt QPainter on the main thread via BlockingQueuedConnection — no
OpenGL context is created on a background thread.

Timing
------
Marker is inserted AFTER _flip() returns (per the rule in base.py).
"""

from __future__ import annotations

import logging
import random
from typing import Callable

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter

from neuroep.acquisition.markers import TriggerCode
from neuroep.stimuli.base import BaseParadigm

logger = logging.getLogger(__name__)

_N_CHECKS_X = 8


def _checkerboard_fn(widget_w: int, widget_h: int, n_cols: int, phase: int) -> Callable:
    """Return a QPainter callback that draws a checkerboard with the given phase."""
    cell_w = widget_w / n_cols
    n_rows = max(1, round(widget_h / cell_w))
    cell_h = widget_h / n_rows

    def draw(painter: QPainter) -> None:
        for row in range(n_rows):
            for col in range(n_cols):
                is_white = (row + col + phase) % 2 == 0
                color = Qt.GlobalColor.white if is_white else Qt.GlobalColor.black
                painter.fillRect(
                    int(col * cell_w),
                    int(row * cell_h),
                    int(cell_w) + 1,
                    int(cell_h) + 1,
                    color,
                )

    return draw


class PatternVEP(BaseParadigm):
    """
    Checkerboard pattern-reversal VEP paradigm.

    Parameters
    ----------
    board_manager : BoardManager
    n_trials : int
        Number of pattern reversals (default 100).
    stim_rate : float
        Average reversals per second (default 2.0 Hz).
    check_size : int
        Number of check columns across the screen (default 8).
    screen : int
        Qt screen index for the stimulus monitor (default 1).
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
        self._phase      = 0

    def _run_trial_loop(self) -> None:
        w = self._stim_widget.width()
        h = self._stim_widget.height()
        n_cols = self._check_size

        draw_a = _checkerboard_fn(w, h, n_cols, phase=0)
        draw_b = _checkerboard_fn(w, h, n_cols, phase=1)

        # Show initial board
        self._flip(draw_a)
        self._phase = 0

        if not self._wait(1.0):
            return

        iti_base = self._inter_trial_interval()

        for trial in range(1, self._n_trials + 1):
            if self._stop_event.is_set():
                break

            self._phase = 1 - self._phase
            draw_fn = draw_b if self._phase == 1 else draw_a

            # ── CRITICAL TIMING ──────────────────────────────────────────
            t_onset = self._flip(draw_fn)
            self._send_marker(TriggerCode.VEP_PATTERN_REVERSAL, t_onset)
            # ─────────────────────────────────────────────────────────────

            self.trial_completed.emit(trial)
            logger.debug("PatternVEP trial %d  T=%.4f s", trial, t_onset)

            jitter = random.uniform(-0.1, 0.1) * iti_base
            if not self._wait(iti_base + jitter):
                break

        self._flip(None)  # clear to black
