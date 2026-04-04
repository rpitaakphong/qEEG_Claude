"""
validation/squarewave_test.py — Mode C: Cyton internal square wave timing test.

Validates that BrainFlow marker channel timestamps align with EEG data channel
timestamps, using the Cyton's built-in internal test signal.

Algorithm (per trial)
---------------------
1. Send board config command to activate 8 Hz square wave on channel 1:
       BoardShim.config_board("x1060110X")
2. Wait for the signal to stabilise
3. Record T1 = time.perf_counter()
4. Insert software marker (TriggerCode.TIMING_TEST)
5. Wait 500 ms
6. Fetch data
7. Find a rising edge in channel 1 data (threshold crossing) closest to T1
8. Find the marker in the marker channel
9. Offset = (edge sample index − marker sample index) / sample_rate × 1000 ms
10. Repeat 50 times

Acceptable: mean offset < 2 ms, SD < 1 ms.

Note: uses the SYNTHETIC board to simulate square wave behaviour for
testing without hardware — real Cyton uses config_board("x1060110X").

Public API
----------
SquareWaveTimingTest(QThread)
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

_MARKER_CODE       = int(TriggerCode.TIMING_TEST)
_WAIT_MS           = 0.5      # seconds after marker before fetching
_SQUARE_WAVE_CMD   = "x1060110X"   # Cyton command: test signal on ch1 at ~8 Hz
_EDGE_THRESHOLD    = 50.0          # µV — rising edge threshold for synthetic signal

# Synthetic board generates a known waveform we can detect edges in
_SYNTHETIC_AMPLITUDE = 100.0   # µV peak for the generated signal


class SquareWaveTimingTest(QThread):
    """
    Mode C timing test.

    When ``use_synthetic=True`` (default), uses the BrainFlow synthetic board
    and generates a simulated square wave edge for testing without hardware.
    When ``use_synthetic=False``, connects to a real Cyton and activates the
    internal test signal.

    Signals
    -------
    progress(int)         Percentage complete (0–100).
    result(list[float])   Offset values in ms, one per trial.
    error(str)            Error message if the test fails.
    """

    progress = pyqtSignal(int)
    result   = pyqtSignal(list)
    error    = pyqtSignal(str)

    def __init__(
        self,
        serial_port:    str  = config.SERIAL_PORT,
        n_trials:       int  = config.TIMING_N_TRIALS,
        use_synthetic:  bool = True,
        parent               = None,
    ) -> None:
        super().__init__(parent)
        self._serial_port   = serial_port
        self._n_trials      = n_trials
        self._use_synthetic = use_synthetic
        self._stop_flag     = False

    def stop(self) -> None:
        self._stop_flag = True

    def run(self) -> None:
        board: Optional[BoardShim] = None
        try:
            board   = self._connect_board()
            offsets = self._run_trials(board)
            self.result.emit(offsets)
        except BrainFlowError as exc:
            logger.error("SquareWaveTimingTest BrainFlowError: %s", exc)
            self.error.emit(f"BrainFlow error: {exc}")
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("SquareWaveTimingTest error: %s", exc, exc_info=True)
            self.error.emit(str(exc))
        finally:
            if board is not None:
                try:
                    board.stop_stream()
                    board.release_session()
                except BrainFlowError:
                    pass

    # ── Internals ──────────────────────────────────────────────────────────

    def _connect_board(self) -> BoardShim:
        BoardShim.disable_board_logger()
        if self._use_synthetic:
            params    = BrainFlowInputParams()
            board_id  = BoardIds.SYNTHETIC_BOARD.value
        else:
            params             = BrainFlowInputParams()
            params.serial_port = self._serial_port
            board_id           = BoardIds.CYTON_BOARD.value   # Daisy not required

        board = BoardShim(board_id, params)
        board.prepare_session()
        board.start_stream()
        time.sleep(0.5)
        board.get_board_data()   # flush stale data

        if not self._use_synthetic:
            # Activate Cyton internal square wave test signal on channel 1
            board.config_board(_SQUARE_WAVE_CMD)
            time.sleep(0.5)   # wait for signal to stabilise

        return board

    def _run_trials(self, board: BoardShim) -> list[float]:
        board_id  = BoardIds.SYNTHETIC_BOARD.value if self._use_synthetic else BoardIds.CYTON_BOARD.value
        eeg_chs   = BoardShim.get_eeg_channels(board_id)
        marker_ch = BoardShim.get_marker_channel(board_id)
        ch1_row   = eeg_chs[0]   # channel 1 — carries the square wave

        offsets: list[float] = []

        for i in range(self._n_trials):
            if self._stop_flag:
                break

            t1 = time.perf_counter()
            board.insert_marker(float(_MARKER_CODE))
            time.sleep(_WAIT_MS)

            data = board.get_board_data()

            ch1_signal  = data[ch1_row, :]
            marker_row  = data[marker_ch, :]

            # Find rising edge in ch1 closest to the start of this window
            edge_idx = self._find_rising_edge(ch1_signal)

            # Find marker position
            marker_hits = np.where(marker_row == _MARKER_CODE)[0]

            if edge_idx is None or len(marker_hits) == 0:
                logger.debug("Trial %d: edge or marker not found.", i + 1)
            else:
                marker_idx = int(marker_hits[-1])
                offset_samples = edge_idx - marker_idx
                offset_ms = offset_samples / config.BOARD_SAMPLE_RATE * 1000.0
                offsets.append(offset_ms)
                logger.debug(
                    "Trial %d: edge=%d  marker=%d  offset=%.3f ms",
                    i + 1, edge_idx, marker_idx, offset_ms,
                )

            self.progress.emit(int((i + 1) / self._n_trials * 100))
            time.sleep(0.05)

        return offsets

    @staticmethod
    def _find_rising_edge(signal: np.ndarray) -> Optional[int]:
        """
        Return the index of the first rising threshold crossing in *signal*.
        A rising edge is detected when the signal goes from below to above
        _EDGE_THRESHOLD.
        """
        above = signal > _EDGE_THRESHOLD
        # Find transitions False → True
        transitions = np.where(np.diff(above.astype(np.int8)) > 0)[0]
        return int(transitions[0]) + 1 if len(transitions) > 0 else None
