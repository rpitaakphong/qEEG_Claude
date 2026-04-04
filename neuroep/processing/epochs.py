"""
processing/epochs.py — Epoch extraction and baseline correction.

An epoch is a fixed-length segment of EEG cut around a stimulus marker.
The segment runs from EPOCH_PRE_MS before the marker to EPOCH_POST_MS after.
Baseline correction subtracts the mean of the pre-stimulus window from every
sample in the epoch.

Public API
----------
EpochExtractor : watches a BoardManager's marker queue and cuts epochs from
                 the filtered ring buffer on demand.
Epoch          : dataclass holding one baseline-corrected epoch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from neuroep import config

logger = logging.getLogger(__name__)

# ── Derived constants ──────────────────────────────────────────────────────
PRE_SAMPLES  = int(config.EPOCH_PRE_MS  / 1000 * config.BOARD_SAMPLE_RATE)  # 12
POST_SAMPLES = int(config.EPOCH_POST_MS / 1000 * config.BOARD_SAMPLE_RATE)  # 62
EPOCH_LEN    = PRE_SAMPLES + POST_SAMPLES                                     # 74

# Time axis in ms for plotting (−pre … +post)
EPOCH_TIME_MS: np.ndarray = np.linspace(
    -config.EPOCH_PRE_MS,
    config.EPOCH_POST_MS,
    EPOCH_LEN,
    dtype=np.float32,
)


@dataclass
class Epoch:
    """
    One baseline-corrected EEG epoch.

    Attributes
    ----------
    data : np.ndarray, shape (N_CHANNELS, EPOCH_LEN)
        Baseline-corrected amplitudes in µV.
    trigger_code : int
        The marker code that triggered this epoch.
    sample_index : int
        Raw ring-buffer sample index of the marker (for debugging).
    accepted : bool
        False if the epoch was rejected by the artifact checker.
    """

    data:         np.ndarray
    trigger_code: int
    sample_index: int
    accepted:     bool = True
    _baseline:    np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def time_ms(self) -> np.ndarray:
        """Time axis in ms, shape (EPOCH_LEN,)."""
        return EPOCH_TIME_MS


class EpochExtractor:
    """
    Cuts baseline-corrected epochs from a filtered data buffer.

    The caller passes a snapshot of the *filtered* ring buffer together
    with a list of ``(trigger_code, sample_index)`` markers.  For each
    marker the extractor checks whether sufficient pre- and post-stimulus
    data is available, then cuts and baseline-corrects the epoch.

    Parameters
    ----------
    n_channels : int
        Number of EEG channels (default ``config.N_CHANNELS``).
    pre_samples : int
        Samples before marker (default ``PRE_SAMPLES``).
    post_samples : int
        Samples after marker (default ``POST_SAMPLES``).
    """

    def __init__(
        self,
        n_channels:   int = config.N_CHANNELS,
        pre_samples:  int = PRE_SAMPLES,
        post_samples: int = POST_SAMPLES,
    ) -> None:
        self._n_ch   = n_channels
        self._pre    = pre_samples
        self._post   = post_samples
        self._ep_len = pre_samples + post_samples

    def extract(
        self,
        filtered_buffer: np.ndarray,
        markers: list[tuple[int, int]],
        buffer_start_index: int = 0,
    ) -> list[Epoch]:
        """
        Extract epochs for every marker in *markers*.

        Parameters
        ----------
        filtered_buffer : np.ndarray, shape (n_channels, total_samples)
            Filtered EEG in chronological order.  Column 0 corresponds
            to sample *buffer_start_index* in absolute time.
        markers : list of (trigger_code, absolute_sample_index)
            Markers from ``BoardManager.pop_markers()``.
        buffer_start_index : int
            Absolute sample index of column 0 in *filtered_buffer*.

        Returns
        -------
        list[Epoch]
            One ``Epoch`` per marker for which sufficient data existed.
            Markers too close to the buffer edges are silently skipped.
        """
        total = filtered_buffer.shape[1]
        epochs: list[Epoch] = []

        for code, abs_idx in markers:
            # Convert absolute index to buffer-relative column
            rel = abs_idx - buffer_start_index

            start = rel - self._pre
            end   = rel + self._post

            if start < 0 or end > total:
                logger.debug(
                    "Marker %d at abs %d skipped — not enough buffer "
                    "(rel=%d, need [%d,%d], have [0,%d]).",
                    code, abs_idx, rel, start, end, total,
                )
                continue

            raw_epoch = filtered_buffer[:, start:end].copy()   # (n_ch, ep_len)
            epoch_bc  = self._baseline_correct(raw_epoch)

            epochs.append(Epoch(
                data         = epoch_bc,
                trigger_code = code,
                sample_index = abs_idx,
            ))
            logger.debug("Epoch extracted: code=%d  abs_idx=%d", code, abs_idx)

        return epochs

    # ── Internals ──────────────────────────────────────────────────────────

    def _baseline_correct(self, epoch: np.ndarray) -> np.ndarray:
        """
        Subtract the mean of the pre-stimulus window from every sample.

        Parameters
        ----------
        epoch : np.ndarray, shape (n_channels, epoch_len)

        Returns
        -------
        np.ndarray, shape (n_channels, epoch_len)
            Baseline-corrected epoch.
        """
        baseline_mean = epoch[:, : self._pre].mean(axis=1, keepdims=True)
        return (epoch - baseline_mean).astype(np.float32)
