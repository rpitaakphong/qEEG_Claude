"""
processing/filters.py — Real-time IIR filter chain for EEG data.

All filters use second-order sections (SOS) coefficients to avoid the
numerical instability of direct-form (ba) designs at low cutoff frequencies.
Filter state (zi) is carried between chunks so there are no discontinuities
at chunk boundaries.

Public API
----------
FilterChain : Applies high-pass, low-pass, and notch filters in sequence.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.signal import (
    butter,
    iirnotch,
    sosfilt,
    sosfilt_zi,
    zpk2sos,
)

from neuroep import config

logger = logging.getLogger(__name__)

# Butterworth filter order for HP and LP stages
_BUTTER_ORDER = 4


def _butter_sos(
    cutoff_hz: float,
    btype: str,
    fs: float = config.BOARD_SAMPLE_RATE,
) -> np.ndarray:
    """
    Design a Butterworth filter and return SOS coefficients.

    Parameters
    ----------
    cutoff_hz : float
        Cutoff frequency in Hz.
    btype : str
        ``"highpass"`` or ``"lowpass"``.
    fs : float
        Sample rate in Hz.

    Returns
    -------
    np.ndarray
        SOS coefficient array, shape ``(n_sections, 6)``.
    """
    nyq = fs / 2.0
    wn  = np.clip(cutoff_hz / nyq, 1e-6, 1.0 - 1e-6)
    return butter(_BUTTER_ORDER, wn, btype=btype, output="sos")


def _notch_sos(
    freq_hz: float,
    q: float = 30.0,
    fs: float = config.BOARD_SAMPLE_RATE,
) -> np.ndarray:
    """
    Design a notch filter and return SOS coefficients.

    Parameters
    ----------
    freq_hz : float
        Notch centre frequency in Hz.
    q : float
        Quality factor (bandwidth = freq_hz / q).
    fs : float
        Sample rate in Hz.

    Returns
    -------
    np.ndarray
        SOS coefficient array.
    """
    b, a = iirnotch(freq_hz, q, fs)
    # Convert via zpk to SOS for numerical stability
    from scipy.signal import tf2zpk
    z, p, k = tf2zpk(b, a)
    return zpk2sos(z, p, k)


def _make_zi(sos: np.ndarray, n_channels: int) -> np.ndarray:
    """
    Initialise filter state for *n_channels* independent channel signals.

    Returns
    -------
    np.ndarray, shape (n_sections, 2, n_channels)
    """
    zi_one = sosfilt_zi(sos)                        # shape (n_sections, 2)
    return np.stack([zi_one] * n_channels, axis=2)  # (n_sections, 2, n_channels)


class FilterChain:
    """
    Applies a high-pass → low-pass → notch IIR filter chain sample-by-sample
    (chunk at a time) while maintaining filter state across chunks.

    Parameters
    ----------
    n_channels : int
        Number of EEG channels processed in parallel.
    hp_hz : float
        Initial high-pass cutoff frequency (Hz).
    lp_hz : float
        Initial low-pass cutoff frequency (Hz).
    notch_hz : float or None
        Notch frequency in Hz, or ``None`` / ``0`` to disable.
    fs : float
        Sample rate (Hz).
    """

    def __init__(
        self,
        n_channels: int              = config.N_CHANNELS,
        hp_hz: float                 = config.DEFAULT_HP_HZ,
        lp_hz: float                 = config.DEFAULT_LP_HZ,
        notch_hz: Optional[float]    = config.DEFAULT_NOTCH_HZ,
        fs: float                    = config.BOARD_SAMPLE_RATE,
    ) -> None:
        self._n_channels = n_channels
        self._fs         = fs

        # SOS coefficients — rebuilt when cutoffs change
        self._hp_sos:    Optional[np.ndarray] = None
        self._lp_sos:    Optional[np.ndarray] = None
        self._notch_sos: Optional[np.ndarray] = None

        # Filter states — shape (n_sections, 2, n_channels)
        self._hp_zi:    Optional[np.ndarray] = None
        self._lp_zi:    Optional[np.ndarray] = None
        self._notch_zi: Optional[np.ndarray] = None

        # Set initial cutoffs (this builds SOS and zi)
        self.set_highpass(hp_hz)
        self.set_lowpass(lp_hz)
        self.set_notch(notch_hz)

    # ── Cutoff setters ─────────────────────────────────────────────────────

    def set_highpass(self, hz: float) -> None:
        """
        Update the high-pass cutoff and reset the filter state.

        Parameters
        ----------
        hz : float
            New cutoff in Hz.  Clamped to [0.1, fs/2 − 1].
        """
        hz = float(np.clip(hz, 0.1, self._fs / 2 - 1))
        self._hp_sos = _butter_sos(hz, "highpass", self._fs)
        self._hp_zi  = _make_zi(self._hp_sos, self._n_channels)
        logger.debug("HP filter reset to %.2f Hz (state reset).", hz)

    def set_lowpass(self, hz: float) -> None:
        """
        Update the low-pass cutoff and reset the filter state.

        Parameters
        ----------
        hz : float
            New cutoff in Hz.  Clamped to [1, fs/2 − 1].
        """
        hz = float(np.clip(hz, 1.0, self._fs / 2 - 1))
        self._lp_sos = _butter_sos(hz, "lowpass", self._fs)
        self._lp_zi  = _make_zi(self._lp_sos, self._n_channels)
        logger.debug("LP filter reset to %.2f Hz (state reset).", hz)

    def set_notch(self, hz: Optional[float]) -> None:
        """
        Enable or disable the notch filter.

        Parameters
        ----------
        hz : float or None
            Notch centre frequency in Hz (50 or 60 typically), or ``None`` / ``0``
            to disable the notch stage entirely.
        """
        if hz and hz > 0:
            self._notch_sos = _notch_sos(hz, fs=self._fs)
            self._notch_zi  = _make_zi(self._notch_sos, self._n_channels)
            logger.debug("Notch filter reset to %.1f Hz (state reset).", hz)
        else:
            self._notch_sos = None
            self._notch_zi  = None
            logger.debug("Notch filter disabled.")

    # ── Processing ─────────────────────────────────────────────────────────

    def process(self, chunk: np.ndarray) -> np.ndarray:
        """
        Filter *chunk* through the active filter stages.

        The filter state is updated in-place after each stage so that the
        next call continues seamlessly without transients.

        Parameters
        ----------
        chunk : np.ndarray, shape (n_channels, n_samples)
            Raw EEG data in µV.

        Returns
        -------
        np.ndarray, shape (n_channels, n_samples)
            Filtered EEG data.
        """
        if chunk.ndim != 2 or chunk.shape[0] != self._n_channels:
            raise ValueError(
                f"Expected shape ({self._n_channels}, N), got {chunk.shape}"
            )

        out = chunk.astype(np.float64)   # promote to float64 for filter precision

        # Apply each stage channel-by-channel (scipy sosfilt is 1-D per call)
        out, self._hp_zi    = self._apply_stage(out, self._hp_sos,    self._hp_zi)
        out, self._lp_zi    = self._apply_stage(out, self._lp_sos,    self._lp_zi)
        out, self._notch_zi = self._apply_stage(out, self._notch_sos, self._notch_zi)

        return out.astype(np.float32)

    # ── Internals ──────────────────────────────────────────────────────────

    def _apply_stage(
        self,
        data: np.ndarray,
        sos:  Optional[np.ndarray],
        zi:   Optional[np.ndarray],
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Run one SOS filter stage across all channels.

        If *sos* is ``None`` the stage is bypassed (data returned unchanged).

        Returns
        -------
        tuple[np.ndarray, Optional[np.ndarray]]
            Filtered data and updated filter state.
        """
        if sos is None or zi is None:
            return data, zi

        out    = np.empty_like(data)
        new_zi = np.empty_like(zi)

        for ch in range(self._n_channels):
            out[ch, :], new_zi[:, :, ch] = sosfilt(
                sos, data[ch, :], zi=zi[:, :, ch]
            )

        return out, new_zi
