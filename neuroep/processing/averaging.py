"""
processing/averaging.py — Running grand average and SNR computation.

After each accepted epoch the running sum and count are updated.
The current average is ``running_sum / epoch_count``.

SNR is computed per channel as:
    signal_power = var(current_avg)          # variance of the average across time
    noise_power  = var(epoch - current_avg)  # residual noise in latest epoch
    snr_db       = 10 * log10(signal_power / noise_power)

The SNR grows roughly as √N as epochs accumulate (central limit theorem).

Public API
----------
RunningAverage : accumulates epochs and exposes current_avg, snr_db, history.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from neuroep import config
from neuroep.processing.epochs import EPOCH_LEN, Epoch

logger = logging.getLogger(__name__)

# Minimum epoch count before SNR is meaningful
_MIN_EPOCHS_FOR_SNR = 2


class RunningAverage:
    """
    Incrementally updates the grand-average EP waveform.

    Only *accepted* epochs (``epoch.accepted == True``) are included.
    The history of previous averages is stored so the averaging panel can
    draw faint ghost traces showing convergence.

    Parameters
    ----------
    n_channels : int
        Number of EEG channels.
    epoch_len : int
        Number of samples per epoch.
    max_history : int
        Maximum number of previous averages to keep (default 50).
    """

    def __init__(
        self,
        n_channels:  int = config.N_CHANNELS,
        epoch_len:   int = EPOCH_LEN,
        max_history: int = 50,
    ) -> None:
        self._n_ch        = n_channels
        self._ep_len      = epoch_len
        self._max_history = max_history

        self._running_sum  = np.zeros((n_channels, epoch_len), dtype=np.float64)
        self._epoch_count  = 0

        # History: list of (epoch_count_at_save, average_array)
        self._history: list[tuple[int, np.ndarray]] = []

        # SNR per channel after the most recent update
        self._snr_db: np.ndarray = np.full(n_channels, np.nan, dtype=np.float32)

        # Per-epoch SNR growth curve (one scalar per epoch = mean across EP channels)
        self._snr_curve: list[float] = []

    # ── Public API ─────────────────────────────────────────────────────────

    def add(self, epoch: Epoch) -> None:
        """
        Add one accepted epoch to the running average.

        If ``epoch.accepted`` is ``False`` the epoch is silently ignored.

        Parameters
        ----------
        epoch : Epoch
            A baseline-corrected, artifact-checked epoch.
        """
        if not epoch.accepted:
            return

        if epoch.data.shape != (self._n_ch, self._ep_len):
            raise ValueError(
                f"Expected epoch shape ({self._n_ch}, {self._ep_len}), "
                f"got {epoch.data.shape}."
            )

        # Save current average to history before updating
        if self._epoch_count > 0:
            prev_avg = self._running_sum / self._epoch_count
            self._push_history(self._epoch_count, prev_avg.astype(np.float32))

        self._running_sum += epoch.data.astype(np.float64)
        self._epoch_count += 1

        # Update SNR
        if self._epoch_count >= _MIN_EPOCHS_FOR_SNR:
            self._update_snr(epoch.data.astype(np.float64))

        valid = self._snr_db[np.isfinite(self._snr_db)]
        snr_mean = float(np.mean(valid)) if len(valid) > 0 else float("nan")
        self._snr_curve.append(snr_mean)

        logger.debug(
            "RunningAverage: n=%d  mean_snr=%.1f dB", self._epoch_count, snr_mean
        )

    def reset(self) -> None:
        """Clear all accumulated data (call at session start)."""
        self._running_sum[:] = 0
        self._epoch_count    = 0
        self._history.clear()
        self._snr_db[:] = np.nan
        self._snr_curve.clear()
        logger.info("RunningAverage reset.")

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def epoch_count(self) -> int:
        """Number of accepted epochs accumulated so far."""
        return self._epoch_count

    @property
    def current_avg(self) -> Optional[np.ndarray]:
        """
        Current grand average, shape ``(n_channels, epoch_len)``, or ``None``
        if no epochs have been added yet.
        """
        if self._epoch_count == 0:
            return None
        return (self._running_sum / self._epoch_count).astype(np.float32)

    @property
    def snr_db(self) -> np.ndarray:
        """
        Per-channel SNR in dB, shape ``(n_channels,)``.
        ``NaN`` for channels where SNR is not yet computable.
        """
        return self._snr_db.copy()

    @property
    def snr_curve(self) -> list[float]:
        """Mean SNR (dB) after each epoch — for the SNR growth plot."""
        return list(self._snr_curve)

    @property
    def history(self) -> list[tuple[int, np.ndarray]]:
        """
        Previous averages as ``(epoch_count, avg_array)`` pairs, oldest first.
        Used to draw ghost traces in the averaging panel.
        """
        return list(self._history)

    # ── Internals ──────────────────────────────────────────────────────────

    def _push_history(self, count: int, avg: np.ndarray) -> None:
        """Append to history, capping at max_history entries."""
        self._history.append((count, avg))
        if len(self._history) > self._max_history:
            self._history.pop(0)

    def _update_snr(self, latest_epoch: np.ndarray) -> None:
        """
        Recompute per-channel SNR.

        signal_power = variance of current average across time
        noise_power  = variance of (latest_epoch − current_average) across time
        snr_db       = 10 * log10(signal_power / noise_power)
        """
        current = self._running_sum / self._epoch_count

        signal_power = np.var(current, axis=1)                     # (n_ch,)
        noise_power  = np.var(latest_epoch - current, axis=1)      # (n_ch,)

        # Guard against division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(noise_power > 0, signal_power / noise_power, np.nan)
            self._snr_db = (10.0 * np.log10(ratio)).astype(np.float32)
