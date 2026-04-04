"""
validation/synthetic_test.py — Mode A: synthetic board pipeline timing test.

Validates that the Python processing pipeline preserves marker timing without
hardware.  Uses BrainFlow's synthetic board so the test can run on any machine.

Algorithm (per trial)
---------------------
1. T1 = time.perf_counter()
2. BoardShim.insert_marker(TriggerCode.TIMING_TEST)
3. Wait 200 ms for the board to buffer the sample
4. Fetch all data: board.get_board_data()
5. Find the marker in the marker channel
6. Convert marker sample index → time using sample rate
7. Jitter = T2 − T1  (in ms)

Acceptable result: mean < 1 ms, SD < 0.5 ms.
Worse values indicate a bug in marker injection or timestamp conversion.

Public API
----------
SyntheticTimingTest(QThread)
    Signals: progress(int), result(list[float]), error(str)
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
from brainflow.board_shim import BoardIds, BoardShim, BrainFlowError, BrainFlowInputParams
from PyQt6.QtCore import QThread, pyqtSignal

from neuroep import config
from neuroep.acquisition.markers import TriggerCode

logger = logging.getLogger(__name__)

_WAIT_AFTER_MARKER = 0.25   # seconds — time to wait before fetching data
_MARKER_CODE       = int(TriggerCode.TIMING_TEST)


class SyntheticTimingTest(QThread):
    """
    Mode A timing test — runs on the synthetic BrainFlow board.

    No real hardware required.  Tests the round-trip latency of:
        insert_marker() → board buffer → get_board_data() → find marker

    Signals
    -------
    progress(int)         Percentage complete (0–100).
    result(list[float])   Jitter values in ms, one per trial.
    error(str)            Error message if the test fails.
    """

    progress = pyqtSignal(int)
    result   = pyqtSignal(list)
    error    = pyqtSignal(str)

    def __init__(self, n_trials: int = config.TIMING_N_TRIALS, parent=None) -> None:
        super().__init__(parent)
        self._n_trials  = n_trials
        self._stop_flag = False

    def stop(self) -> None:
        """Request early termination."""
        self._stop_flag = True

    def run(self) -> None:
        board: Optional[BoardShim] = None
        try:
            board = self._connect()
            jitters = self._run_trials(board)
            self.result.emit(jitters)
        except BrainFlowError as exc:
            logger.error("SyntheticTimingTest BrainFlowError: %s", exc)
            self.error.emit(f"BrainFlow error: {exc}")
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("SyntheticTimingTest error: %s", exc, exc_info=True)
            self.error.emit(str(exc))
        finally:
            if board is not None:
                try:
                    board.stop_stream()
                    board.release_session()
                except BrainFlowError:
                    pass

    # ── Internals ──────────────────────────────────────────────────────────

    def _connect(self) -> BoardShim:
        BoardShim.disable_board_logger()
        params = BrainFlowInputParams()
        board  = BoardShim(BoardIds.SYNTHETIC_BOARD.value, params)
        board.prepare_session()
        board.start_stream()
        time.sleep(0.5)   # let the buffer fill
        board.get_board_data()   # flush any stale data
        return board

    def _run_trials(self, board: BoardShim) -> list[float]:
        jitters: list[float] = []
        board_id  = BoardIds.SYNTHETIC_BOARD.value
        marker_ch = BoardShim.get_marker_channel(board_id)
        ts_ch     = BoardShim.get_timestamp_channel(board_id)

        for i in range(self._n_trials):
            if self._stop_flag:
                break

            # Flush any stale data so we start with an empty buffer
            board.get_board_data()

            # Record wall-clock time at marker insertion (Unix seconds, matches BrainFlow)
            t1 = time.time()
            board.insert_marker(float(_MARKER_CODE))

            # Wait for the board to buffer the sample
            time.sleep(_WAIT_AFTER_MARKER)

            # Fetch buffered data
            data = board.get_board_data()

            marker_row = data[marker_ch, :]
            ts_row     = data[ts_ch, :]
            hits = np.where(np.abs(marker_row - _MARKER_CODE) < 0.5)[0]

            if len(hits) == 0:
                logger.warning("Trial %d: marker not found in buffer.", i + 1)
                continue

            # T2 = BrainFlow timestamp of the sample containing the marker
            sample_idx = int(hits[0])
            t2 = float(ts_row[sample_idx]) if len(ts_row) > sample_idx else (
                t1 + sample_idx / config.BOARD_SAMPLE_RATE
            )
            jitter_ms = (t2 - t1) * 1000.0
            jitters.append(jitter_ms)

            logger.debug("Trial %d: jitter = %.3f ms", i + 1, jitter_ms)
            self.progress.emit(int((i + 1) / self._n_trials * 100))

            time.sleep(0.05)

        return jitters
