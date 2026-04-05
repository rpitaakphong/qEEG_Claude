"""
stimuli/base.py — Qt-native stimulus window and shared paradigm base.

All paradigms inherit from BaseParadigm, which runs the trial loop in a
background QThread and drives a StimulusWidget on the main thread via
BlockingQueuedConnection signals.  No PsychoPy or OpenGL is required.

Critical timing rule
--------------------
    t_onset = self._flip(draw_fn)     # blocks until main-thread repaint
    self._send_marker(code, t_onset)  # marker AFTER flip
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QPainter
from PyQt6.QtWidgets import QApplication, QWidget

logger = logging.getLogger(__name__)


class StimulusWidget(QWidget):
    """
    Fullscreen stimulus display on the secondary monitor.

    Lives on the main (GUI) thread.  The timing thread drives it exclusively
    through ``flip()``, which is connected with BlockingQueuedConnection so
    the caller blocks until the synchronous repaint completes.

    Parameters
    ----------
    screen_index : int
        Qt screen index (0 = primary, 1 = secondary).  Falls back to
        screen 0 if fewer screens are available.
    """

    def __init__(self, screen_index: int = 1) -> None:
        super().__init__()
        self._draw_fn: Optional[Callable[[QPainter], None]] = None
        self._last_flip_time: float = 0.0

        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setCursor(Qt.CursorShape.BlankCursor)

        screens = QApplication.screens()
        screen = screens[min(screen_index, len(screens) - 1)]
        geo = screen.geometry()
        self.setGeometry(geo)
        self.showFullScreen()
        if self.windowHandle():
            self.windowHandle().setScreen(screen)
            self.setGeometry(geo)

        logger.info(
            "StimulusWidget opened on screen %d (%dx%d).",
            screen_index,
            geo.width(),
            geo.height(),
        )

    # ── Paint ──────────────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        if self._draw_fn is not None:
            self._draw_fn(painter)
        else:
            painter.fillRect(self.rect(), Qt.GlobalColor.black)
        painter.end()

    # ── Slot called on main thread via BlockingQueuedConnection ───────────

    @pyqtSlot(object)
    def flip(self, draw_fn: Optional[Callable]) -> None:
        """
        Set the draw function and repaint synchronously.

        Called from the timing QThread via BlockingQueuedConnection — the
        worker thread blocks here until this method returns.  ``repaint()``
        forces an immediate synchronous paint before returning.
        """
        self._draw_fn = draw_fn
        self.repaint()
        self._last_flip_time = time.perf_counter()


# ── Base paradigm QThread ──────────────────────────────────────────────────


class BaseParadigm(QThread):
    """
    Abstract base class for all EP stimulus paradigms.

    Runs the trial loop in a background QThread.  Visual output is driven
    via ``_flip(draw_fn)`` which routes to ``StimulusWidget.flip`` on the
    main thread through a BlockingQueuedConnection.

    Signals
    -------
    marker_sent(int, float)   : (trigger_code, T_onset_seconds)
    trial_completed(int)      : (trial_number, 1-based)
    paradigm_finished()       : all trials done or paradigm stopped
    error_occurred(str)       : exception message from inside trial loop
    """

    marker_sent       = pyqtSignal(int, float)
    trial_completed   = pyqtSignal(int)
    paradigm_finished = pyqtSignal()
    error_occurred    = pyqtSignal(str)

    _flip_signal = pyqtSignal(object)   # internal: routed to widget.flip

    def __init__(
        self,
        board_manager,
        n_trials:  int   = 100,
        stim_rate: float = 2.0,
        screen:    int   = 1,
        parent           = None,
    ) -> None:
        super().__init__(parent)
        self._board      = board_manager
        self._n_trials   = n_trials
        self._stim_rate  = stim_rate
        self._screen     = screen
        self._stop_event = threading.Event()
        self._stim_widget: Optional[StimulusWidget] = None

    # ── Widget attachment (call on main thread before start()) ─────────────

    def attach_widget(self, widget: StimulusWidget) -> None:
        """
        Connect the flip signal to *widget*.

        Must be called on the main thread before ``start()``.
        BlockingQueuedConnection ensures ``_flip_signal.emit(fn)`` blocks
        the worker thread until the main thread has finished repainting.
        """
        self._stim_widget = widget
        self._flip_signal.connect(
            widget.flip,
            Qt.ConnectionType.BlockingQueuedConnection,
        )

    # ── QThread entry point ────────────────────────────────────────────────

    def run(self) -> None:
        try:
            self._run_trial_loop()
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Paradigm error: %s", exc, exc_info=True)
            self.error_occurred.emit(str(exc))
        finally:
            self.paradigm_finished.emit()
            logger.info("%s finished.", self.__class__.__name__)

    def stop(self) -> None:
        """Request the trial loop to stop after the current trial."""
        self._stop_event.set()

    # ── To be overridden ───────────────────────────────────────────────────

    def _run_trial_loop(self) -> None:
        raise NotImplementedError

    # ── Helpers available to subclasses ───────────────────────────────────

    def _flip(self, draw_fn: Optional[Callable] = None) -> float:
        """
        Render a frame on the stimulus widget and return the flip timestamp.

        Blocks the calling thread until the main-thread repaint completes.
        ``draw_fn(painter: QPainter)`` is executed on the main thread.
        Pass ``None`` to clear to black.

        Returns
        -------
        float
            ``time.perf_counter()`` recorded immediately after ``repaint()``
            on the main thread — the canonical stimulus onset timestamp.
        """
        self._flip_signal.emit(draw_fn)
        return self._stim_widget._last_flip_time

    def _inter_trial_interval(self) -> float:
        return 1.0 / self._stim_rate

    def _send_marker(self, code: int, t_onset: float) -> None:
        self._board.insert_marker(code)
        self.marker_sent.emit(code, t_onset)

    def _wait(self, seconds: float) -> bool:
        """
        Sleep for *seconds* while honouring the stop event.

        Returns ``True`` if the wait completed, ``False`` if stopped early.
        """
        deadline = time.perf_counter() + seconds
        while time.perf_counter() < deadline:
            if self._stop_event.is_set():
                return False
            time.sleep(0.001)
        return True
