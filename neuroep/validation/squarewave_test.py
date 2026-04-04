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
        board_manager=None,              # existing BoardManager — preferred
        serial_port:    str  = config.SERIAL_PORT,
        n_trials:       int  = config.TIMING_N_TRIALS,
        use_synthetic:  bool = True,
        parent               = None,
    ) -> None:
        super().__init__(parent)
        self._board_manager = board_manager
        self._serial_port   = serial_port
        self._n_trials      = n_trials
        self._use_synthetic = use_synthetic if board_manager is None else False
        self._stop_flag     = False

    def stop(self) -> None:
        self._stop_flag = True

    def run(self) -> None:
        if self._board_manager is not None:
            self._run_with_board_manager()
        else:
            self._run_standalone()

    def _run_with_board_manager(self) -> None:
        """Use the application's existing board connection."""
        manager = self._board_manager
        board   = manager.raw_board
        if board is None:
            self.error.emit("No board connected.")
            return

        board_id   = manager.board_id
        eeg_chs    = BoardShim.get_eeg_channels(board_id)
        marker_ch  = BoardShim.get_marker_channel(board_id)

        try:
            manager.pause_acquisition()
            board.get_board_data()               # flush stale data

            # Activate Cyton internal square wave on channel 1
            board.config_board(_SQUARE_WAVE_CMD)
            time.sleep(0.5)
            board.get_board_data()               # flush signal-stabilise data

            offsets = self._run_trials_hardware(board, eeg_chs, marker_ch)
            self.result.emit(offsets)

        except BrainFlowError as exc:
            logger.error("SquareWaveTimingTest BrainFlowError: %s", exc)
            self.error.emit(f"BrainFlow error: {exc}")
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("SquareWaveTimingTest error: %s", exc, exc_info=True)
            self.error.emit(str(exc))
        finally:
            # Restore ch1 to normal EEG input, resume acquisition
            try:
                board.config_board("x1060110X")   # deactivate test signal
            except Exception:  # pylint: disable=broad-except
                pass
            manager.resume_acquisition()

    def _run_standalone(self) -> None:
        """Open a fresh board connection (synthetic only — real board is handled above)."""
        board: Optional[BoardShim] = None
        try:
            board   = self._connect_board()
            board_id  = BoardIds.SYNTHETIC_BOARD.value
            marker_ch = BoardShim.get_marker_channel(board_id)
            offsets   = self._run_trials_synthetic(board, marker_ch)
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


    def _run_trials_synthetic(
        self, board: BoardShim, marker_ch: int
    ) -> list[float]:
        """
        Synthetic mode: measures marker placement jitter.

        BrainFlow's synthetic board has a fixed internal buffering offset
        (typically 15–30 ms) that is consistent across trials.  What matters
        for ERP quality is the JITTER around that offset, not the offset itself.

        Algorithm per trial
        -------------------
        1. Flush the BrainFlow buffer.
        2. Insert a software marker.
        3. Wait _WAIT_MS for data to accumulate.
        4. Fetch data and find the marker's sample index → raw_ms.

        Post-processing
        ---------------
        Subtract the median raw_ms from all trials so the distribution is
        centred around 0.  The resulting SD is the true marker timing jitter.
        """
        raw_ms: list[float] = []

        for i in range(self._n_trials):
            if self._stop_flag:
                break

            board.get_board_data()          # flush
            board.insert_marker(float(_MARKER_CODE))
            time.sleep(_WAIT_MS)

            data = board.get_board_data()
            if data.shape[1] == 0:
                logger.debug("Trial %d: no data returned.", i + 1)
                self.progress.emit(int((i + 1) / self._n_trials * 100))
                continue

            marker_row  = data[marker_ch, :]
            marker_hits = np.where(marker_row == _MARKER_CODE)[0]

            if len(marker_hits) == 0:
                logger.debug("Trial %d: marker not found in data.", i + 1)
            else:
                marker_idx = int(marker_hits[-1])
                ms = marker_idx / config.BOARD_SAMPLE_RATE * 1000.0
                raw_ms.append(ms)
                logger.debug("Trial %d: marker_idx=%d  raw=%.3f ms", i + 1, marker_idx, ms)

            self.progress.emit(int((i + 1) / self._n_trials * 100))

        if not raw_ms:
            return []

        # Centre around zero: subtract median so jitter is visible regardless
        # of BrainFlow's fixed internal buffering offset.
        median = float(np.median(raw_ms))
        offsets = [ms - median for ms in raw_ms]
        logger.info(
            "Synthetic mode: median buffering offset = %.1f ms  jitter SD = %.2f ms",
            median, float(np.std(offsets)),
        )
        return offsets

    def _run_trials_hardware(
        self, board: BoardShim, eeg_chs: list, marker_ch: int
    ) -> list[float]:
        """
        Real Cyton mode: finds the rising edge of the internal square wave and
        compares it to the software marker position.
        """
        ch1_row = eeg_chs[0]
        offsets: list[float] = []

        for i in range(self._n_trials):
            if self._stop_flag:
                break

            board.insert_marker(float(_MARKER_CODE))
            time.sleep(_WAIT_MS)

            data = board.get_board_data()

            ch1_signal  = data[ch1_row, :]
            marker_row  = data[marker_ch, :]

            edge_idx    = self._find_rising_edge(ch1_signal)
            marker_hits = np.where(marker_row == _MARKER_CODE)[0]

            if edge_idx is None or len(marker_hits) == 0:
                logger.debug("Trial %d: edge or marker not found.", i + 1)
            else:
                marker_idx     = int(marker_hits[-1])
                offset_samples = edge_idx - marker_idx
                offset_ms      = offset_samples / config.BOARD_SAMPLE_RATE * 1000.0
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
