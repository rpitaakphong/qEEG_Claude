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
from PyQt6.QtGui import QColor, QFont
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

        # Per-channel height system (in µV display units)
        self._base_height: float = self._sensitivity * 2.0
        self._ch_gap: float = 0.0
        self._channel_height: list[float] = [self._base_height] * config.N_CHANNELS
        self._offsets: list[float] = [0.0] * config.N_CHANNELS

        # Active paradigm (canonical key e.g. "VEP_PATTERN" or "ALL")
        self._active_paradigm: str = "ALL"

        # Filtered display buffer — pre-allocated, written circularly
        # Shape: (N_CHANNELS, _WINDOW_SAMPLES)
        self._disp_buf = np.zeros(
            (config.N_CHANNELS, _WINDOW_SAMPLES), dtype=np.float32
        )
        self._disp_pos = 0          # next write column in _disp_buf

        # How many raw samples we have already processed through the filter
        self._processed_raw = 0

        self._build_plot()
        self._recompute_offsets()

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

    def set_paradigm(self, paradigm_key: str) -> None:
        """
        Show only the ERP-relevant channels; hide everything else.
        Called by main_window when paradigm_changed signal fires.

        Parameters
        ----------
        paradigm_key : str
            Sidebar key (e.g. ``"vep_pattern"``) or canonical key
            (e.g. ``"VEP_PATTERN"``).  ``"ALL"`` shows every channel.
        """
        # Normalise sidebar keys → canonical keys
        canonical = config.PARADIGM_KEY_MAP.get(paradigm_key, paradigm_key)
        self._active_paradigm = canonical

        visible_set = config.PARADIGM_VISIBLE_CHANNELS.get(canonical)  # None = all

        for i, ch_name in enumerate(config.CHANNEL_NAMES):
            show = visible_set is None or ch_name in visible_set
            self._curves[i].setVisible(show)
            self._ch_labels[i].setVisible(show)
            if show:
                color = QColor(config.CHANNEL_COLORS[i])
                self._curves[i].setPen(pg.mkPen(color, width=1))
                self._ch_labels[i].setColor(color)
                self._channel_height[i] = self._base_height
            else:
                self._channel_height[i] = 0.0   # takes no Y space

        self._recompute_offsets()
        self._plot_widget.update()

    # ── Filter control slots (called by control sidebar) ───────────────────

    def set_sensitivity(self, uv: float) -> None:
        """Update the channel-spacing sensitivity (µV half-range)."""
        self._sensitivity = max(1.0, float(uv))
        self._base_height = self._sensitivity * 2.5
        # Re-apply current paradigm heights at new scale
        self.set_paradigm(self._active_paradigm)

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
        plot.showAxis("bottom")
        plot.setLabel("bottom", "Time (s)", **{"color": "#9a9891", "font-size": "11pt"})
        plot.getAxis("bottom").setTextPen(pg.mkPen("#9a9891"))
        plot.getAxis("bottom").setPen(pg.mkPen("#2e3148"))
        plot.getAxis("bottom").setTickFont(QFont("Segoe UI", 9))
        plot.showGrid(x=True, y=False, alpha=0.18)
        plot.setMenuEnabled(False)
        plot.setMouseEnabled(x=False, y=False)

        # Fixed time axis — always -DISPLAY_SECONDS … 0
        self._t_axis = np.linspace(
            -config.DISPLAY_SECONDS, 0, _WINDOW_SAMPLES, dtype=np.float32
        )

        self._curves:    list[pg.PlotDataItem] = []
        self._ch_labels: list[pg.TextItem]     = []

        label_font = QFont("Segoe UI", 8)

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

            # Label: "ch01 Fp1" so technician can cross-reference the amplifier
            label = pg.TextItem(
                text=f"ch{ch + 1:02d} {config.CHANNEL_NAMES[ch]}",
                color=color,
                anchor=(1.0, 0.5),
            )
            label.setFont(label_font)
            plot.addItem(label)
            self._ch_labels.append(label)

        # Extend X range left to create a label margin column.
        # Custom tick positions are set so the bottom axis only labels the
        # actual EEG data zone (−5 … 0), not the margin.
        self._x_left = -(config.DISPLAY_SECONDS + 0.6)
        plot.setXRange(self._x_left, 0, padding=0)

        # Only show time ticks inside the real data window
        bottom_axis = plot.getAxis("bottom")
        tick_major = [(float(t), str(t)) for t in range(-config.DISPLAY_SECONDS, 1)]
        bottom_axis.setTicks([tick_major, []])

        # ── Axis decorations ──────────────────────────────────────────────

        # Rotated "µV" label on the left margin (inside the label zone)
        y_label = pg.TextItem(text="µV", color="#9a9891", angle=90, anchor=(0.5, 0.5))
        y_label.setFont(QFont("Segoe UI", 10))
        self._y_axis_label = y_label
        plot.addItem(y_label)

        # Vertical divider line separating the label margin from the EEG traces
        self._margin_line = pg.InfiniteLine(
            pos=-config.DISPLAY_SECONDS,
            angle=90,
            pen=pg.mkPen("#2e3148", width=1),
        )
        plot.addItem(self._margin_line)

        # Scale bar: vertical reference line near the right edge + text label.
        # Shows the current sensitivity value so the reader knows 1 channel slot height.
        self._scale_bar = pg.PlotDataItem(
            pen=pg.mkPen("#9a9891", width=1.5),
        )
        self._scale_bar_top = pg.PlotDataItem(   # horizontal cap – top
            pen=pg.mkPen("#9a9891", width=1.5),
        )
        self._scale_bar_bot = pg.PlotDataItem(   # horizontal cap – bottom
            pen=pg.mkPen("#9a9891", width=1.5),
        )
        self._scale_text = pg.TextItem(
            text="", color="#9a9891", anchor=(0.0, 0.5),
        )
        self._scale_text.setFont(QFont("Courier New", 7))
        for item in (self._scale_bar, self._scale_bar_top,
                     self._scale_bar_bot, self._scale_text):
            plot.addItem(item)

        layout.addWidget(self._plot_widget)

    def _recompute_offsets(self) -> None:
        """
        Recalculate Y offsets for visible channels only.
        Hidden channels (height == 0) are skipped so no blank rows appear.
        Offsets are centred around zero.  Also repositions axis decorations.
        """
        running_y = 0.0
        for i in range(config.N_CHANNELS - 1, -1, -1):   # bottom → top
            self._offsets[i] = running_y
            if self._channel_height[i] > 0:
                running_y += self._channel_height[i] + self._ch_gap

        # Centre around zero
        center = max(running_y / 2.0, 1.0)   # guard against empty selection
        for i in range(config.N_CHANNELS):
            self._offsets[i] -= center

        # Set Y range tightly around visible channels (± half a slot for padding)
        # Using (-center, center) wastes a full slot at the top; use actual extents.
        visible_offsets = [
            self._offsets[i] for i in range(config.N_CHANNELS)
            if self._channel_height[i] > 0
        ]
        half_slot = self._base_height * 0.5
        if visible_offsets:
            y_min = min(visible_offsets) - half_slot
            y_max = max(visible_offsets) + half_slot
        else:
            y_min, y_max = -center, center

        plot = self._plot_widget.getPlotItem()
        plot.setYRange(y_min, y_max, padding=0)

        # Channel labels sit inside the margin zone, right-aligned to the divider
        label_x = -config.DISPLAY_SECONDS - 0.08
        for ch in range(config.N_CHANNELS):
            self._ch_labels[ch].setPos(label_x, self._offsets[ch])

        # ── Axis decorations ──────────────────────────────────────────────

        # Rotated "µV" label — centred in the margin zone, away from channel labels
        if hasattr(self, "_y_axis_label"):
            self._y_axis_label.setPos(-config.DISPLAY_SECONDS - 0.42, 0)

        # Scale bar — vertical reference in the top-right corner showing
        # the current sensitivity (µV) so amplitude can be judged at a glance.
        if hasattr(self, "_scale_bar"):
            bar_x   = -0.15                          # 0.15 s before "now"
            cap_hw  = 0.06                           # half-width of horizontal caps
            bar_top = y_max - half_slot * 0.4        # near top of visible area
            bar_bot = bar_top - self._sensitivity    # one sensitivity step down

            self._scale_bar.setData(
                x=[bar_x, bar_x], y=[bar_bot, bar_top]
            )
            self._scale_bar_top.setData(
                x=[bar_x - cap_hw, bar_x + cap_hw], y=[bar_top, bar_top]
            )
            self._scale_bar_bot.setData(
                x=[bar_x - cap_hw, bar_x + cap_hw], y=[bar_bot, bar_bot]
            )
            self._scale_text.setText(f"{int(self._sensitivity)} µV")
            self._scale_text.setPos(bar_x + cap_hw + 0.02,
                                    (bar_top + bar_bot) / 2.0)

    # ── Refresh slot ───────────────────────────────────────────────────────

    def _refresh(self) -> None:
        """
        Incremental refresh: filter only samples that arrived since the last
        tick, write them into the circular display buffer, then plot.
        """
        if self._board is None or self._filters is None:
            return

        # Full raw snapshot (chronological order)
        raw = self._board.ring_buffer.snapshot()   # (16, samples_in_buffer)
        total_in_buffer = raw.shape[1]

        if total_in_buffer == 0:
            return

        # Use the board's unbounded sample_count so the comparison never stalls
        # after the ring buffer fills (ring buffer caps at 30 s; sample_count grows forever).
        current_count = self._board.sample_count
        new_count = current_count - self._processed_raw
        if new_count <= 0:
            # No new data — just redraw existing buffer
            self._draw()
            return

        # Cap to what's actually in the buffer and the display window
        new_count = min(new_count, total_in_buffer, _WINDOW_SAMPLES)

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
        self._processed_raw = current_count

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
            y = display[ch].astype(np.float32) + self._offsets[ch]
            self._curves[ch].setData(x=self._t_axis, y=y)
