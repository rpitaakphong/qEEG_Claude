"""
stimuli/base.py — PsychoPy window management and shared stimulus utilities.

All paradigms inherit from ``BaseParadigm``, which owns the PsychoPy Window
on screen 2, runs the trial loop in a background QThread, and communicates
with the Qt main thread via signals.

Critical timing rule (enforced in every paradigm)
--------------------------------------------------
    win.flip()                   # blocks until display refresh
    T_onset = core.getTime()     # timestamp AFTER flip
    board.insert_marker(code)    # marker inserted AFTER flip

Never insert the marker before win.flip() — it would precede the actual
stimulus onset by one full monitor refresh period (~16 ms at 60 Hz).
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

from PyQt6.QtCore import QThread, pyqtSignal

logger = logging.getLogger(__name__)

def psychopy_available() -> bool:
    """Return True if PsychoPy is importable (lightweight check, no side effects)."""
    import importlib.util
    return importlib.util.find_spec("psychopy") is not None


def _init_psychopy_in_thread():
    """
    Import pyglet and psychopy.visual INSIDE the worker thread.

    pyglet requires that EventLoop.run() is called on the same thread that
    first imports pyglet.app.  Deferring all imports to the QThread's run()
    call satisfies this constraint.  Returns (core, visual) or raises.
    """
    try:
        import pyglet
        pyglet.options["shadow_window"] = False
        pyglet.options["debug_gl"] = False
    except Exception:  # pylint: disable=broad-except
        pass

    from psychopy import core, visual  # noqa: PLC0415
    return core, visual


class StimulusWindow:
    """
    Thin wrapper around a PsychoPy ``visual.Window`` on screen index 1.

    Opens fullscreen on the secondary monitor.  Call ``close()`` when done.

    Parameters
    ----------
    screen : int
        Screen index (0 = primary, 1 = secondary stimulus monitor).
    color : tuple[float, float, float]
        Background colour in PsychoPy normalised units (−1 to +1).
    """

    def __init__(
        self,
        screen: int = 1,
        color:  tuple[float, float, float] = (-1.0, -1.0, -1.0),
    ) -> None:
        if not psychopy_available():
            raise RuntimeError(
                "PsychoPy is not installed. "
                "Run: pip install psychopy"
            )
        # Import here (inside the QThread worker) so pyglet.app is imported on
        # the same thread that will run its event loop — satisfying pyglet's
        # threading constraint.
        self._core, visual = _init_psychopy_in_thread()
        self.win = visual.Window(
            fullscr      = True,
            screen       = screen,
            color        = color,
            colorSpace   = "rgb",
            units        = "norm",
            allowGUI     = False,
            waitBlanking = True,
        )
        self.clock = self._core.Clock()
        logger.info(
            "PsychoPy window opened on screen %d (%dx%d).",
            screen,
            self.win.size[0],
            self.win.size[1],
        )

    def flip(self) -> float:
        """
        Flip the back buffer and return the time (seconds) AFTER the flip.

        This is the canonical stimulus onset timestamp — always use the
        return value as T_onset.
        """
        self.win.flip()
        return self._core.getTime()

    def close(self) -> None:
        """Close the PsychoPy window."""
        try:
            self.win.close()
        except Exception:  # pylint: disable=broad-except
            pass
        logger.info("PsychoPy window closed.")


# ── Base paradigm QThread ──────────────────────────────────────────────────

class BaseParadigm(QThread):
    """
    Abstract base class for all EP stimulus paradigms.

    Runs the trial loop in a background QThread so the Qt GUI stays
    responsive.  Subclasses must implement ``_run_trial_loop()``.

    Signals
    -------
    marker_sent(int, float)
        Emitted after each stimulus onset: (trigger_code, T_onset_seconds).
    trial_completed(int)
        Emitted after each trial: (trial_number, 1-based).
    paradigm_finished()
        Emitted when all trials are done or the paradigm is stopped.
    error_occurred(str)
        Emitted if an exception is raised inside the trial loop.
    """

    marker_sent      = pyqtSignal(int, float)    # (code, T_onset)
    trial_completed  = pyqtSignal(int)           # (trial_number)
    paradigm_finished = pyqtSignal()
    error_occurred   = pyqtSignal(str)

    def __init__(
        self,
        board_manager,                           # BoardManager (avoid circular import)
        n_trials:    int   = 100,
        stim_rate:   float = 2.0,               # Hz
        screen:      int   = 1,
        parent                  = None,
    ) -> None:
        super().__init__(parent)
        self._board      = board_manager
        self._n_trials   = n_trials
        self._stim_rate  = stim_rate
        self._screen     = screen
        self._stop_event = threading.Event()
        self._stim_win: Optional[StimulusWindow] = None

    # ── QThread entry point ────────────────────────────────────────────────

    def run(self) -> None:
        """Open the stimulus window, run trials, clean up."""
        try:
            if not psychopy_available():
                raise RuntimeError("PsychoPy is not installed.")

            self._stim_win = StimulusWindow(screen=self._screen)
            self._run_trial_loop()

        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Paradigm error: %s", exc, exc_info=True)
            self.error_occurred.emit(str(exc))

        finally:
            if self._stim_win is not None:
                self._stim_win.close()
                self._stim_win = None
            self.paradigm_finished.emit()
            logger.info("%s finished.", self.__class__.__name__)

    def stop(self) -> None:
        """Request the trial loop to stop after the current trial."""
        self._stop_event.set()

    # ── To be overridden ───────────────────────────────────────────────────

    def _run_trial_loop(self) -> None:
        """Override in subclass to implement the trial sequence."""
        raise NotImplementedError

    # ── Helpers available to subclasses ───────────────────────────────────

    def _inter_trial_interval(self) -> float:
        """Return the inter-trial interval in seconds based on stim_rate."""
        return 1.0 / self._stim_rate

    def _send_marker(self, code: int, t_onset: float) -> None:
        """
        Insert marker into BrainFlow and emit the ``marker_sent`` signal.

        Parameters
        ----------
        code : int
            Trigger code (use ``TriggerCode`` enum).
        t_onset : float
            PsychoPy ``core.getTime()`` value recorded immediately after
            ``win.flip()`` returned.
        """
        self._board.insert_marker(code)
        self.marker_sent.emit(code, t_onset)

    def _wait(self, seconds: float) -> bool:
        """
        Sleep for *seconds* while honouring the stop event.

        Returns
        -------
        bool
            ``True`` if the wait completed normally, ``False`` if stop was
            requested during the wait.
        """
        from psychopy import core as _core
        deadline = _core.getTime() + seconds
        while _core.getTime() < deadline:
            if self._stop_event.is_set():
                return False
            _core.wait(0.001)   # 1 ms polling
        return True
