"""
processing/artifact.py — Epoch artifact detection and rejection.

Phase 1: amplitude threshold — reject any epoch where any channel exceeds
         ARTIFACT_UV peak-to-peak.  Fast, requires no history.

Phase 2 (future): ICA-based ocular artifact removal via mne.preprocessing.ICA.

Public API
----------
ArtifactChecker : stateless amplitude-threshold checker.
RejectionTracker : tracks accept/reject counts and warns on high rejection rate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from neuroep import config
from neuroep.processing.epochs import Epoch

logger = logging.getLogger(__name__)

# Rejection rate above this fraction triggers a warning in the status bar
_WARN_REJECTION_RATE = 0.30


class ArtifactChecker:
    """
    Amplitude-threshold artifact checker.

    An epoch is rejected if *any* sample on *any* channel exceeds
    ``threshold_uv`` in absolute value.

    Parameters
    ----------
    threshold_uv : float
        Rejection threshold in µV (default ``config.ARTIFACT_UV``).
    channels : list[int] or None
        Channel indices to check.  ``None`` means all channels.
    """

    def __init__(
        self,
        threshold_uv: float          = config.ARTIFACT_UV,
        channels:     list[int] | None = None,
    ) -> None:
        self._threshold = float(threshold_uv)
        self._channels  = channels   # None → check all

    @property
    def threshold_uv(self) -> float:
        """Current rejection threshold in µV."""
        return self._threshold

    @threshold_uv.setter
    def threshold_uv(self, value: float) -> None:
        self._threshold = float(value)

    def check(self, epoch: Epoch) -> Epoch:
        """
        Mark *epoch* as accepted or rejected based on amplitude threshold.

        Modifies ``epoch.accepted`` in-place and returns the same object
        for chaining convenience.

        Parameters
        ----------
        epoch : Epoch
            A baseline-corrected epoch from ``EpochExtractor``.

        Returns
        -------
        Epoch
            The same epoch with ``accepted`` set.
        """
        data = epoch.data
        if self._channels is not None:
            data = data[self._channels, :]

        if np.any(np.abs(data) > self._threshold):
            epoch.accepted = False
            logger.debug(
                "Epoch rejected (code=%d, max=%.1f µV > threshold=%.1f µV).",
                epoch.trigger_code,
                float(np.max(np.abs(data))),
                self._threshold,
            )
        else:
            epoch.accepted = True

        return epoch

    def check_batch(self, epochs: list[Epoch]) -> list[Epoch]:
        """
        Apply :meth:`check` to every epoch in *epochs*.

        Parameters
        ----------
        epochs : list[Epoch]

        Returns
        -------
        list[Epoch]
            Same list with ``accepted`` fields updated.
        """
        for ep in epochs:
            self.check(ep)
        return epochs


@dataclass
class RejectionStats:
    """Running accept/reject counters for one session."""

    accepted: int = 0
    rejected: int = 0

    @property
    def total(self) -> int:
        return self.accepted + self.rejected

    @property
    def rejection_rate(self) -> float:
        """Fraction of epochs rejected (0.0–1.0).  Returns 0 if no epochs yet."""
        return self.rejected / self.total if self.total > 0 else 0.0

    @property
    def high_rejection(self) -> bool:
        """True if rejection rate exceeds the warning threshold."""
        return self.total >= 10 and self.rejection_rate > _WARN_REJECTION_RATE

    def update(self, epoch: Epoch) -> None:
        """Record one epoch's outcome."""
        if epoch.accepted:
            self.accepted += 1
        else:
            self.rejected += 1
        if self.high_rejection:
            logger.warning(
                "High rejection rate: %.0f%% (%d/%d) — check electrodes.",
                self.rejection_rate * 100,
                self.rejected,
                self.total,
            )

    def reset(self) -> None:
        """Clear counters (call at session start)."""
        self.accepted = 0
        self.rejected = 0
