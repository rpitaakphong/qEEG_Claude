"""
ui/eeg_panel.py — Live 16-channel scrolling EEG viewer widget.

Renders all 16 EEG channels as colour-coded traces on a single pyqtgraph
PlotWidget at DISPLAY_FPS frames per second.  Signal controls (sensitivity,
HP, LP, notch) update the display live without restarting the stream.

Data flow (correct incremental approach)
-----------------------------------------
Each timer tick:
  1. Snapshot the raw ring buffer to find how many new samples arrived.
  2. Extract ONLY the new samples (delta since last tick).
  3. Pass those through FilterChain — zi state advances by exactly those samples.
  4. Write filtered samples into a local display buffer (circular).
  5. Plot the display buffer.

Filtering the full 5-second snapshot every frame would re-feed old data
through the filter each tick, corrupting the zi state and producing
smearing/jumping artefacts.

Public API
----------
EEGPanel(QWidget) : drop-in widget; call ``set_board(manager)`` to attach
                     a live BoardManager.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from neuroep import config
from neuroep.acquisition.board import BoardManager
from neuroep.processing.filters import FilterChain

logger = logging.getLogger(__name__)

# Interval between display refreshes (ms)
_REFRESH_MS = int(1000 / config.DISPLAY_FPS)

# Number of samples held in the display window
_WINDOW_SAMPLES = config.BOARD_SAMPLE_RATE * config.DISPLAY_SECONDS


class EEGPanel(QWidget):
    """
    Live 16-channel EEG viewer.

    Channels are rendered as stacked, colour-coded waveforms on a single
    pyqtgraph canvas.  The Y axis is hidden; channel labels are drawn as
    ``TextItem`` objects on the left margin.

    Parameters
    ----------
    parent : QWidget, optional
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self._board:   Optional[BoardManager] = None
        self._filters: Optional[FilterChain]  = None

        # Sensitivity in µV — controls vertical spacing between traces
        self._sensitivity: float = config.DEFAULT_SENSITIVITY

        # Filtered display buffer — pre-allocated, written circularly
        # Shape: (N_CHANNELS, _WINDOW_SAMPLES)
        self._disp_buf = np.zeros(
            (config.N_CHANNELS, _WINDOW_SAMPLES), dtype=np.float32
        )
        self._disp_pos = 0          # next write column in _disp_buf

        # How many raw samples we have already processed through the filter
        self._processed_raw = 0

        self._build_plot()
        self._timer = QTimer(self)
        self._timer.setInterval(_REFRESH_MS)
        self._timer.timeout.connect(self._refresh)

    # ── Public API ─────────────────────────────────────────────────────────

    def set_board(self, manager: BoardManager) -> None:
        """
        Attach a connected ``BoardManager`` and start the refresh timer.

        Parameters
        ----------
        manager : BoardManager
            A board that is already streaming.
        """
        self._board   = manager
        self._filters = FilterChain()
        self._disp_buf[:] = 0
        self._disp_pos     = 0
        self._processed_raw = 0
        self._timer.start()
        logger.info("EEGPanel: board attached, refresh timer started.")

    def detach_board(self) -> None:
        """Stop refreshing and detach the board."""
        self._timer.stop()
        self._board   = None
        self._filters = None
        logger.info("EEGPanel: board detached.")

    # ── Filter control slots (called by control sidebar) ───────────────────

    def set_sensitivity(self, uv: float) -> None:
        """Update the channel-spacing sensitivity (µV half-range)."""
        self._sensitivity = max(1.0, float(uv))
        self._update_channel_spacing()

    def set_highpass(self, hz: float) -> None:
        """Update the high-pass cutoff on the live filter chain."""
        if self._filters:
            self._filters.set_highpass(hz)

    def set_lowpass(self, hz: float) -> None:
        """Update the low-pass cutoff on the live filter chain."""
        if self._filters:
            self._filters.set_lowpass(hz)

    def set_notch(self, hz: Optional[float]) -> None:
        """Enable or disable the notch filter."""
        if self._filters:
            self._filters.set_notch(hz)

    # ── Build UI ───────────────────────────────────────────────────────────

    def _build_plot(self) -> None:
        """Construct the pyqtgraph PlotWidget and 16 PlotDataItems."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._plot_widget = pg.PlotWidget(background="#0e1117")
        self._plot_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        plot = self._plot_widget.getPlotItem()
        plot.hideAxis("left")
        plot.hideAxis("bottom")
        plot.setMenuEnabled(False)
        plot.setMouseEnabled(x=False, y=False)

        # Fixed time axis — always -DISPLAY_SECONDS … 0
        self._t_axis = np.linspace(
            -config.DISPLAY_SECONDS, 0, _WINDOW_SAMPLES, dtype=np.float32
        )

        self._curves: list[pg.PlotDataItem] = []
        self._labels: list[pg.TextItem]     = []

        label_font = QFont("Courier New", 9)

        for ch in range(config.N_CHANNELS):
            color = config.CHANNEL_COLORS[ch]

            curve = pg.PlotDataItem(
                pen=pg.mkPen(color=color, width=1),
                antialias=False,
                clipToView=True,
                autoDownsample=True,
                downsampleMethod="peak",
            )
            plot.addItem(curve)
            self._curves.append(curve)

            label = pg.TextItem(
                text=config.CHANNEL_NAMES[ch],
                color=color,
                anchor=(1.0, 0.5),
            )
            label.setFont(label_font)
            plot.addItem(label)
            self._labels.append(label)

        plot.setXRange(-config.DISPLAY_SECONDS, 0, padding=0)
        self._update_channel_spacing()

        layout.addWidget(self._plot_widget)

    def _update_channel_spacing(self) -> None:
        """Recalculate Y range and label positions after sensitivity change."""
        spacing = self._sensitivity * 2.5
        total   = spacing * config.N_CHANNELS

        plot = self._plot_widget.getPlotItem()
        plot.setYRange(-total / 2, total / 2, padding=0)

        for ch in range(config.N_CHANNELS):
            self._labels[ch].setPos(
                -config.DISPLAY_SECONDS - 0.05, self._channel_offset(ch)
            )

    def _channel_offset(self, ch: int) -> float:
        """Vertical offset in µV for channel *ch* (ch=0 at top, ch=15 at bottom)."""
        spacing = self._sensitivity * 2.5
        return ((config.N_CHANNELS - 1) / 2.0 - ch) * spacing

    # ── Refresh slot ───────────────────────────────────────────────────────

    def _refresh(self) -> None:
        """
        Incremental refresh: filter only samples that arrived since the last
        tick, write them into the circular display buffer, then plot.
        """
        if self._board is None or self._filters is None:
            return

        # Full raw snapshot (chronological order)
        raw = self._board.ring_buffer.snapshot()   # (16, total_available)
        total_available = raw.shape[1]

        if total_available == 0:
            return

        # Determine how many samples are genuinely new
        new_count = total_available - self._processed_raw
        if new_count <= 0:
            # No new data — just redraw existing buffer
            self._draw()
            return

        # Cap: if we fell behind by more than _WINDOW_SAMPLES, skip ahead
        if new_count > _WINDOW_SAMPLES:
            new_count = _WINDOW_SAMPLES

        # Extract only the new samples from the end of the snapshot
        new_raw = raw[:, -new_count:]

        # Filter the new chunk (zi state advances by new_count samples only)
        try:
            new_filtered = self._filters.process(new_raw)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Filter error: %s", exc)
            new_filtered = new_raw.astype(np.float32)

        # Write new filtered samples into the circular display buffer
        self._write_to_disp_buf(new_filtered)
        self._processed_raw = total_available

        self._draw()

    def _write_to_disp_buf(self, chunk: np.ndarray) -> None:
        """Write *chunk* (16, n) into the circular display buffer."""
        n = chunk.shape[1]
        end = self._disp_pos + n
        if end <= _WINDOW_SAMPLES:
            self._disp_buf[:, self._disp_pos : end] = chunk
        else:
            first = _WINDOW_SAMPLES - self._disp_pos
            self._disp_buf[:, self._disp_pos :] = chunk[:, :first]
            self._disp_buf[:, : n - first]       = chunk[:, first:]
        self._disp_pos = end % _WINDOW_SAMPLES

    def _draw(self) -> None:
        """Re-order the circular buffer into chronological order and update curves."""
        # Unroll circular buffer: oldest → newest
        p = self._disp_pos
        display = np.concatenate(
            [self._disp_buf[:, p:], self._disp_buf[:, :p]], axis=1
        )

        for ch in range(config.N_CHANNELS):
            y = display[ch].astype(np.float32) + self._channel_offset(ch)
            self._curves[ch].setData(x=self._t_axis, y=y)
