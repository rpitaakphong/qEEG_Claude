"""
acquisition/board.py — BrainFlow board manager and thread-safe ring buffer.

Public API
----------
RingBuffer   : Fixed-size circular buffer with Lock-protected access.
BoardManager : Owns the BrainFlow session; drives data into a RingBuffer
               and exposes marker injection.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import numpy as np
from brainflow.board_shim import BoardIds, BoardShim, BrainFlowError, BrainFlowInputParams

from neuroep import config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# RingBuffer
# ─────────────────────────────────────────────────────────────────────────────

class RingBuffer:
    """
    Thread-safe circular buffer for EEG data.

    Stores the most recent *capacity* samples across *n_channels* channels.
    All read/write operations are protected by an internal ``threading.Lock``.

    Parameters
    ----------
    n_channels : int
        Number of EEG channels (rows).
    capacity : int
        Maximum number of samples (columns) to keep in memory.
    """

    def __init__(self, n_channels: int, capacity: int) -> None:
        self._n_channels = n_channels
        self._capacity   = capacity
        self._buf        = np.zeros((n_channels, capacity), dtype=np.float32)
        self._pos        = 0          # write-head (next column to overwrite)
        self._filled     = 0          # samples written so far (up to capacity)
        self._lock       = threading.Lock()

    # ── Write ──────────────────────────────────────────────────────────────

    def push(self, chunk: np.ndarray) -> None:
        """
        Append *chunk* to the buffer, overwriting the oldest samples when full.

        Parameters
        ----------
        chunk : np.ndarray, shape (n_channels, n_samples)
            New samples to add.  ``n_channels`` must match the buffer width.
        """
        if chunk.ndim != 2 or chunk.shape[0] != self._n_channels:
            raise ValueError(
                f"Expected chunk shape ({self._n_channels}, N), got {chunk.shape}"
            )
        n = chunk.shape[1]
        with self._lock:
            if n >= self._capacity:
                # Entire buffer will be overwritten; keep only the last *capacity* samples
                self._buf[:] = chunk[:, -self._capacity :]
                self._pos    = 0
                self._filled = self._capacity
                return

            end = self._pos + n
            if end <= self._capacity:
                self._buf[:, self._pos : end] = chunk
            else:
                # Wrap around
                first = self._capacity - self._pos
                self._buf[:, self._pos :] = chunk[:, :first]
                self._buf[:, : n - first] = chunk[:, first:]

            self._pos    = end % self._capacity
            self._filled = min(self._filled + n, self._capacity)

    # ── Read ───────────────────────────────────────────────────────────────

    def snapshot(self) -> np.ndarray:
        """
        Return a copy of the entire valid buffer contents in chronological order.

        Returns
        -------
        np.ndarray, shape (n_channels, filled_samples)
        """
        with self._lock:
            if self._filled < self._capacity:
                return self._buf[:, : self._filled].copy()
            # Re-order so oldest sample is first
            return np.concatenate(
                [self._buf[:, self._pos :], self._buf[:, : self._pos]], axis=1
            ).copy()

    @property
    def n_samples(self) -> int:
        """Number of valid samples currently held."""
        with self._lock:
            return self._filled

    def clear(self) -> None:
        """Reset the buffer to empty."""
        with self._lock:
            self._buf[:] = 0
            self._pos    = 0
            self._filled = 0


# ─────────────────────────────────────────────────────────────────────────────
# BoardManager
# ─────────────────────────────────────────────────────────────────────────────

class BoardManager:
    """
    Manages the BrainFlow board session lifecycle and background data thread.

    Responsibilities
    ----------------
    - Connect / disconnect the board.
    - Run a background thread that pulls chunks of samples from BrainFlow
      and pushes them into a ``RingBuffer``.
    - Inject markers into the BrainFlow marker channel.
    - Expose ``get_new_samples()`` for the processing pipeline.

    Parameters
    ----------
    board_id : int
        A ``BoardIds`` value (e.g. ``BoardIds.CYTON_DAISY_BOARD`` or
        ``BoardIds.SYNTHETIC_BOARD``).
    serial_port : str
        COM port string (e.g. ``"COM3"`` or ``"/dev/ttyUSB0"``).
        Ignored for synthetic boards.
    """

    def __init__(
        self,
        board_id: int   = BoardIds.SYNTHETIC_BOARD.value,
        serial_port: str = config.SERIAL_PORT,
    ) -> None:
        self._board_id    = board_id
        self._serial_port = serial_port
        self._board: Optional[BoardShim] = None
        self._is_streaming = False

        # Capacity = 30 s of data
        capacity = config.BOARD_SAMPLE_RATE * 30
        self.ring_buffer = RingBuffer(config.N_CHANNELS, capacity)

        # Queue for epoch extraction: stores (marker_code, sample_index) tuples
        self._marker_queue: list[tuple[int, int]] = []
        self._marker_lock  = threading.Lock()
        self._sample_count = 0          # total samples received this session

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Suppress BrainFlow's own verbose logging unless debug is needed
        BoardShim.disable_board_logger()

    # ── Connection ─────────────────────────────────────────────────────────

    def connect(self) -> None:
        """
        Initialise the BrainFlow session and begin streaming.

        Raises
        ------
        BrainFlowError
            If the board cannot be found or the session fails to open.
        """
        params = BrainFlowInputParams()
        if self._board_id != BoardIds.SYNTHETIC_BOARD.value:
            params.serial_port = self._serial_port

        logger.info(
            "Connecting board_id=%d serial='%s'", self._board_id, self._serial_port
        )
        self._board = BoardShim(self._board_id, params)
        self._board.prepare_session()
        self._board.start_stream()
        self._is_streaming = True
        self._sample_count = 0
        self.ring_buffer.clear()
        self._stop_event.clear()

        self._thread = threading.Thread(
            target=self._acquisition_loop,
            name="BrainFlow-Acquisition",
            daemon=True,
        )
        self._thread.start()
        logger.info("Board connected and acquisition thread started.")

    @property
    def raw_board(self) -> Optional[BoardShim]:
        """Direct access to the underlying BoardShim (use with care)."""
        return self._board

    @property
    def board_id(self) -> int:
        """The BrainFlow board ID used for this session."""
        return self._board_id

    def pause_acquisition(self) -> None:
        """Stop the background acquisition thread without disconnecting the board."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        logger.info("Acquisition thread paused.")

    def resume_acquisition(self) -> None:
        """Restart the background acquisition thread after a pause."""
        if self._board is None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._acquisition_loop,
            name="BrainFlow-Acquisition",
            daemon=True,
        )
        self._thread.start()
        logger.info("Acquisition thread resumed.")

    def disconnect(self) -> None:
        """Stop streaming and release the BrainFlow session."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

        if self._board is not None:
            try:
                if self._is_streaming:
                    self._board.stop_stream()
                self._board.release_session()
            except BrainFlowError as exc:
                logger.warning("Error during disconnect: %s", exc)
            finally:
                self._board = None
                self._is_streaming = False

        logger.info("Board disconnected.")

    @property
    def is_connected(self) -> bool:
        """``True`` if the board session is active."""
        return self._board is not None and self._is_streaming

    @property
    def sample_count(self) -> int:
        """Total samples received since streaming started (unbounded)."""
        with self._marker_lock:
            return self._sample_count

    # ── Marker injection ───────────────────────────────────────────────────

    def insert_marker(self, code: int) -> int:
        """
        Write *code* into the BrainFlow marker channel and record the
        approximate sample index for epoch extraction.

        Parameters
        ----------
        code : int
            Trigger value (use ``TriggerCode`` enum values).

        Returns
        -------
        int
            Estimated sample index at the time of insertion.

        Raises
        ------
        RuntimeError
            If the board is not connected.
        """
        if self._board is None:
            raise RuntimeError("Board is not connected — cannot insert marker.")

        self._board.insert_marker(float(code))
        with self._marker_lock:
            idx = self._sample_count
            self._marker_queue.append((code, idx))
        logger.debug("Marker %d inserted at sample ~%d", code, idx)
        return idx

    def pop_markers(self) -> list[tuple[int, int]]:
        """
        Return and clear all pending markers as ``(code, sample_index)`` pairs.
        Thread-safe.
        """
        with self._marker_lock:
            markers = list(self._marker_queue)
            self._marker_queue.clear()
        return markers

    # ── Data access ────────────────────────────────────────────────────────

    def get_new_samples(self) -> Optional[np.ndarray]:
        """
        Pull all available samples from BrainFlow and push them into the
        ring buffer.  Called by the acquisition thread; also callable from
        the main thread for one-shot fetches (e.g. validation modes).

        Returns
        -------
        np.ndarray or None
            Shape ``(N_CHANNELS, n_new_samples)`` if data arrived, else ``None``.
        """
        if self._board is None:
            return None

        try:
            data = self._board.get_board_data()   # shape: (all_channels, n_samples)
        except BrainFlowError as exc:
            logger.error("BrainFlowError fetching data: %s", exc)
            return None

        if data.shape[1] == 0:
            return None

        # Extract only EEG channels (first N_CHANNELS rows by convention for
        # Cyton+Daisy; BrainFlow returns them in columns 1..16)
        eeg_rows = BoardShim.get_eeg_channels(self._board_id)
        if len(eeg_rows) >= config.N_CHANNELS:
            eeg = data[eeg_rows[: config.N_CHANNELS], :].astype(np.float32)
        else:
            # Fallback: use whatever rows are available, zero-pad if needed
            available = data[eeg_rows, :].astype(np.float32)
            pad = np.zeros(
                (config.N_CHANNELS - len(eeg_rows), data.shape[1]), dtype=np.float32
            )
            eeg = np.vstack([available, pad])

        # Replace synthetic board signal with a clean low-amplitude sine wave
        if self._board_id == BoardIds.SYNTHETIC_BOARD.value:
            n_samples = eeg.shape[1]
            t = (self._sample_count + np.arange(n_samples)) / config.BOARD_SAMPLE_RATE
            sine = (20.0 * np.sin(2 * np.pi * 10.0 * t)).astype(np.float32)
            eeg = np.tile(sine, (config.N_CHANNELS, 1))

        self.ring_buffer.push(eeg)
        with self._marker_lock:
            self._sample_count += eeg.shape[1]

        return eeg

    # ── Background thread ──────────────────────────────────────────────────

    def _acquisition_loop(self) -> None:
        """
        Background thread: pulls data from BrainFlow at CHUNK_SIZE intervals.
        Runs until ``_stop_event`` is set or an unrecoverable error occurs.
        """
        interval = config.CHUNK_SIZE / config.BOARD_SAMPLE_RATE  # seconds between polls

        while not self._stop_event.is_set():
            try:
                self.get_new_samples()
            except Exception as exc:  # pylint: disable=broad-except
                logger.error("Acquisition thread error: %s", exc, exc_info=True)
                break
            time.sleep(interval)

        logger.info("Acquisition thread stopped.")
