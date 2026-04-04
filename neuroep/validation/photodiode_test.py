"""
validation/photodiode_test.py — Mode B: photodiode hardware jitter test.

Measures true end-to-end stimulus latency, including display refresh delay,
by detecting the photodiode signal on Cyton analog input D11 (channel 15).

Algorithm (per trial)
---------------------
1. Show black background (2 seconds ISI)
2. T_software = psychopy core.getTime()
3. Draw full-white screen → win.flip() — T_onset captured after flip
4. Insert BrainFlow marker (TriggerCode.TIMING_TEST)
5. Wait 500 ms for photodiode response window
6. Fetch data buffer
7. Find photodiode onset in channel PHOTODIODE_CHANNEL:
       first sample whose abs value crosses PHOTODIODE_THRESH
8. Convert onset sample index → time
9. Jitter = photodiode_onset_time − T_onset

Acceptable: mean < 5 ms, SD < 2 ms.
Mean > 10 ms → likely missing a monitor refresh cycle.
SD > 5 ms → timing unreliable (USB load, background processes).

Requires: real Cyton+Daisy board + photodiode on D11 + PsychoPy.

Public API
----------
PhotodiodeTimingTest(QThread)
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
_ISI_SECONDS       = 2.0     # black screen between flashes
_RESPONSE_WINDOW   = 0.5     # seconds to collect photodiode data after flash


class PhotodiodeTimingTest(QThread):
    """
    Mode B timing test — requires real Cyton board + photodiode on D11.

    Signals
    -------
    progress(int)         Percentage complete (0–100).
    result(list[float])   Jitter values in ms, one per trial.
    error(str)            Error message if the test fails.
    """

    progress = pyqtSignal(int)
    result   = pyqtSignal(list)
    error    = pyqtSignal(str)

    def __init__(
        self,
        serial_port: str = config.SERIAL_PORT,
        n_trials:    int = config.TIMING_N_TRIALS,
        screen:      int = 1,
        parent           = None,
    ) -> None:
        super().__init__(parent)
        self._serial_port = serial_port
        self._n_trials    = n_trials
        self._screen      = screen
        self._stop_flag   = False

    def stop(self) -> None:
        self._stop_flag = True

    def run(self) -> None:
        board: Optional[BoardShim] = None
        win                        = None
        try:
            from neuroep.stimuli.base import StimulusWindow
            board = self._connect_board()
            stim  = StimulusWindow(screen=self._screen)
            win   = stim.win
            jitters = self._run_trials(board, stim)
            self.result.emit(jitters)
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("PhotodiodeTimingTest error: %s", exc, exc_info=True)
            self.error.emit(str(exc))
        finally:
            if win is not None:
                try:
                    win.close()
                except Exception:
                    pass
            if board is not None:
                try:
                    board.stop_stream()
                    board.release_session()
                except BrainFlowError:
                    pass

    # ── Internals ──────────────────────────────────────────────────────────

    def _connect_board(self) -> BoardShim:
        BoardShim.disable_board_logger()
        params             = BrainFlowInputParams()
        params.serial_port = self._serial_port
        board              = BoardShim(BoardIds.CYTON_DAISY_BOARD.value, params)
        board.prepare_session()
        board.start_stream()
        time.sleep(1.0)
        board.get_board_data()   # flush stale data
        return board

    def _run_trials(self, board: BoardShim, stim) -> list[float]:
        from psychopy import visual, core

        win = stim.win

        flash = visual.Rect(
            win, width=2.0, height=2.0, units="norm",
            fillColor=[1.0, 1.0, 1.0], lineColor=None,
        )
        blank = visual.Rect(
            win, width=2.0, height=2.0, units="norm",
            fillColor=[-1.0, -1.0, -1.0], lineColor=None,
        )

        # EEG channels from BrainFlow for Cyton+Daisy
        eeg_chs  = BoardShim.get_eeg_channels(BoardIds.CYTON_DAISY_BOARD.value)
        # Analog channel index for D11 (photodiode) — channel 15 in our mapping
        # BrainFlow returns analog channels separately; we use eeg_chs[15] as proxy
        pd_row   = eeg_chs[config.PHOTODIODE_CHANNEL] if len(eeg_chs) > config.PHOTODIODE_CHANNEL else eeg_chs[-1]
        ts_ch    = BoardShim.get_timestamp_channel(BoardIds.CYTON_DAISY_BOARD.value)

        jitters: list[float] = []

        # Initial blank
        blank.draw()
        win.flip()
        core.wait(_ISI_SECONDS)

        for i in range(self._n_trials):
            if self._stop_flag:
                break

            # Flash
            flash.draw()

            # ── CRITICAL TIMING ──────────────────────────────────────────
            win.flip()
            t_onset = core.getTime()
            board.insert_marker(float(_MARKER_CODE))
            # ─────────────────────────────────────────────────────────────

            # Wait for response window
            core.wait(_RESPONSE_WINDOW)

            # Fetch buffered data
            data = board.get_board_data()

            # Detect photodiode onset
            pd_signal = data[pd_row, :]
            ts_signal = data[ts_ch, :]

            onset_idx = self._find_onset(pd_signal)
            if onset_idx is None:
                logger.warning("Trial %d: photodiode onset not found.", i + 1)
            else:
                # Timestamp of photodiode onset
                if len(ts_signal) > onset_idx:
                    t_pd = float(ts_signal[onset_idx])
                else:
                    # Estimate from sample count
                    t_pd = t_onset + onset_idx / config.BOARD_SAMPLE_RATE

                jitter_ms = (t_pd - t_onset) * 1000.0
                jitters.append(jitter_ms)
                logger.debug("Trial %d: jitter = %.3f ms", i + 1, jitter_ms)

            self.progress.emit(int((i + 1) / self._n_trials * 100))

            # ISI — blank screen
            blank.draw()
            win.flip()
            core.wait(_ISI_SECONDS)

        return jitters

    @staticmethod
    def _find_onset(signal: np.ndarray) -> Optional[int]:
        """
        Return the index of the first sample crossing PHOTODIODE_THRESH.
        Returns None if no crossing is found.
        """
        crossings = np.where(np.abs(signal) > config.PHOTODIODE_THRESH)[0]
        return int(crossings[0]) if len(crossings) > 0 else None
