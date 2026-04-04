"""
acquisition/markers.py — Trigger codes, LSL outlet, and timestamp logging.

Provides:
- TriggerCode  : IntEnum of all stimulus and session event codes.
- MarkerStream : LSL outlet wrapper for broadcasting markers to
                 any recording or analysis tool on the local network.
"""

from __future__ import annotations

import logging
import time
from enum import IntEnum
from typing import Optional

logger = logging.getLogger(__name__)

# ── Attempt LSL import; degrade gracefully if not installed ────────────────
try:
    from pylsl import StreamInfo, StreamOutlet
    _LSL_AVAILABLE = True
except ImportError:
    _LSL_AVAILABLE = False
    logger.warning("pylsl not found — LSL marker stream disabled.")


class TriggerCode(IntEnum):
    """Numeric codes inserted into the BrainFlow marker channel."""

    VEP_PATTERN_REVERSAL = 1
    VEP_FLASH            = 2
    AEP_CLICK            = 3
    AEP_TONE_STANDARD    = 4
    AEP_TONE_ODDBALL     = 5
    P300_STANDARD        = 6
    P300_ODDBALL         = 7
    SESSION_START        = 10
    SESSION_END          = 11
    EPOCH_REJECTED       = 20
    TIMING_TEST          = 99


class MarkerStream:
    """
    Wraps a pylsl StreamOutlet and a flat log list so that every marker
    sent is time-stamped and retrievable for post-hoc analysis.

    Parameters
    ----------
    stream_name : str
        Human-readable LSL stream name (default ``"NeuroEP_Markers"``).
    stream_id : str
        Unique LSL source ID (default ``"neuroep_markers_1"``).
    """

    def __init__(
        self,
        stream_name: str = "NeuroEP_Markers",
        stream_id: str   = "neuroep_markers_1",
    ) -> None:
        self._outlet: Optional["StreamOutlet"] = None
        self._log: list[tuple[float, int]] = []    # (perf_counter, code)

        if _LSL_AVAILABLE:
            info = StreamInfo(
                name       = stream_name,
                type       = "Markers",
                channel_count = 1,
                nominal_srate = 0,            # irregular stream
                channel_format = "int32",
                source_id  = stream_id,
            )
            self._outlet = StreamOutlet(info)
            logger.info("LSL marker outlet '%s' created.", stream_name)
        else:
            logger.info("LSL unavailable — markers logged locally only.")

    def send(self, code: TriggerCode) -> float:
        """
        Broadcast *code* on the LSL outlet and record its timestamp.

        Parameters
        ----------
        code : TriggerCode
            The trigger code to send.

        Returns
        -------
        float
            ``time.perf_counter()`` value at the moment of dispatch.
        """
        t = time.perf_counter()
        if self._outlet is not None:
            self._outlet.push_sample([int(code)])
        self._log.append((t, int(code)))
        logger.debug("Marker sent: %s (%d) at %.6f s", code.name, int(code), t)
        return t

    def get_log(self) -> list[tuple[float, int]]:
        """Return a copy of the full marker log as ``(timestamp, code)`` pairs."""
        return list(self._log)

    def clear_log(self) -> None:
        """Discard all logged markers (e.g. between sessions)."""
        self._log.clear()

    def close(self) -> None:
        """Release the LSL outlet (call on application exit)."""
        self._outlet = None
        logger.info("MarkerStream closed.")
