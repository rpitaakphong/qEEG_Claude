"""
processing/components.py — Automatic EP component detection.

Detects the canonical EP peaks in a grand-average waveform by searching
within physiologically defined latency windows.  Each component is located
as an extremum (max or min) within its window.

Components and search windows
------------------------------
| Component | Polarity | Window (ms)  | Paradigm        |
|-----------|----------|--------------|-----------------|
| P100      | +        |  80 – 140 ms | VEP             |
| N1        | −        |  80 – 150 ms | AEP             |
| P2        | +        | 150 – 250 ms | AEP             |
| N2        | −        | 180 – 280 ms | P300            |
| P300      | +        | 250 – 600 ms | P300 / AEP      |

Public API
----------
ComponentResult  : dataclass holding latency_ms, amplitude_uv, channel index.
ComponentDetector: detects all or paradigm-specific components in a waveform.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal import savgol_filter

from neuroep.processing.epochs import EPOCH_TIME_MS

logger = logging.getLogger(__name__)


@dataclass
class ComponentResult:
    """
    Result of detecting one EP component.

    Attributes
    ----------
    name : str
        Component label (e.g. ``"P100"``).
    latency_ms : float
        Peak latency in ms relative to stimulus onset.
    amplitude_uv : float
        Peak amplitude in µV (baseline-corrected).
    channel : int
        Channel index where the component was detected.
    """

    name:         str
    latency_ms:   float
    amplitude_uv: float
    channel:      int


# ── Window definitions ─────────────────────────────────────────────────────
# Each entry: (name, polarity, onset_ms, offset_ms)
# polarity: +1 → search for positive peak, −1 → negative peak
_WINDOWS: list[tuple[str, int, float, float]] = [
    ("P100", +1,  80.0, 140.0),
    ("N1",   -1,  80.0, 150.0),
    ("P2",   +1, 150.0, 250.0),
    ("N2",   -1, 180.0, 280.0),
    ("P300", +1, 250.0, 600.0),
]

# Paradigm → which component names to look for
_PARADIGM_COMPONENTS: dict[str, list[str]] = {
    "vep_pattern": ["P100"],
    "vep_flash":   ["P100"],
    "aep":         ["N1", "P2", "P300"],
    "p300_passive":["N2", "P300"],
}


class ComponentDetector:
    """
    Detects EP component latencies and amplitudes in a grand-average waveform.

    Parameters
    ----------
    smooth : bool
        Apply a Savitzky-Golay smoothing filter before peak search
        (recommended — reduces noise-driven spurious peaks).
    smooth_window : int
        Savitzky-Golay window length in samples (must be odd, default 7).
    smooth_poly : int
        Savitzky-Golay polynomial order (default 3).
    """

    def __init__(
        self,
        smooth:        bool = True,
        smooth_window: int  = 7,
        smooth_poly:   int  = 3,
    ) -> None:
        self._smooth        = smooth
        self._smooth_window = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
        self._smooth_poly   = smooth_poly

    def detect(
        self,
        avg: np.ndarray,
        channel: int,
        paradigm: Optional[str] = None,
    ) -> list[ComponentResult]:
        """
        Detect EP components in a single-channel average waveform.

        Parameters
        ----------
        avg : np.ndarray, shape (n_channels, epoch_len) or (epoch_len,)
            Grand-average waveform.  If 2-D, ``channel`` selects the row.
        channel : int
            Channel index to use when ``avg`` is 2-D.
        paradigm : str or None
            Paradigm key (e.g. ``"vep_pattern"``).  If provided, only
            components relevant to that paradigm are searched.  Pass
            ``None`` to search all components.

        Returns
        -------
        list[ComponentResult]
            Detected components sorted by latency.
        """
        if avg.ndim == 2:
            waveform = avg[channel, :].astype(np.float64)
        else:
            waveform = avg.astype(np.float64)

        if self._smooth and len(waveform) >= self._smooth_window:
            try:
                waveform = savgol_filter(
                    waveform, self._smooth_window, self._smooth_poly
                )
            except ValueError:
                pass   # silently skip smoothing if window > signal length

        # Which components to look for
        if paradigm and paradigm in _PARADIGM_COMPONENTS:
            target_names = set(_PARADIGM_COMPONENTS[paradigm])
        else:
            target_names = {w[0] for w in _WINDOWS}

        results: list[ComponentResult] = []

        for name, polarity, onset_ms, offset_ms in _WINDOWS:
            if name not in target_names:
                continue

            result = self._find_peak(
                waveform, name, polarity, onset_ms, offset_ms, channel
            )
            if result is not None:
                results.append(result)

        results.sort(key=lambda r: r.latency_ms)
        return results

    def detect_all_channels(
        self,
        avg: np.ndarray,
        ep_channels: list[int],
        paradigm: Optional[str] = None,
    ) -> dict[int, list[ComponentResult]]:
        """
        Detect components on each channel in *ep_channels*.

        Parameters
        ----------
        avg : np.ndarray, shape (n_channels, epoch_len)
        ep_channels : list[int]
            Indices of channels relevant to the current paradigm
            (e.g. ``config.EP_CHANNELS["VEP"]``).
        paradigm : str or None

        Returns
        -------
        dict[int, list[ComponentResult]]
            Mapping from channel index to detected components.
        """
        return {
            ch: self.detect(avg, ch, paradigm)
            for ch in ep_channels
        }

    # ── Internals ──────────────────────────────────────────────────────────

    def _find_peak(
        self,
        waveform:  np.ndarray,
        name:      str,
        polarity:  int,
        onset_ms:  float,
        offset_ms: float,
        channel:   int,
    ) -> Optional[ComponentResult]:
        """
        Find the extremum of *waveform* within [onset_ms, offset_ms].

        Returns ``None`` if the window falls outside the epoch or the
        waveform is flat.
        """
        # Convert ms to sample indices
        mask = (EPOCH_TIME_MS >= onset_ms) & (EPOCH_TIME_MS <= offset_ms)
        indices = np.where(mask)[0]

        if len(indices) == 0:
            return None

        segment = waveform[indices]

        if polarity == +1:
            peak_rel = int(np.argmax(segment))
        else:
            peak_rel = int(np.argmin(segment))

        peak_idx  = indices[peak_rel]
        latency   = float(EPOCH_TIME_MS[peak_idx])
        amplitude = float(waveform[peak_idx])

        logger.debug(
            "Component %s detected: lat=%.1f ms  amp=%.2f µV  ch=%d",
            name, latency, amplitude, channel,
        )

        return ComponentResult(
            name         = name,
            latency_ms   = latency,
            amplitude_uv = amplitude,
            channel      = channel,
        )
