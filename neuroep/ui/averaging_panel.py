"""
ui/averaging_panel.py — Real-time EP averaging panel.

Three side-by-side sub-panels that update after every accepted epoch:

Sub-panel A — Averaged EP waveform
  • Ghost traces (previous averages, fading with age) show convergence.
  • Bold current average in paradigm accent colour.
  • Auto-detected component markers (P100, N1, P2, P300) with latency labels.
  • Vertical t=0 and horizontal 0 µV dashed reference lines.
  • Epoch counter "47 / 100 epochs" in top-right corner.
  • Save PNG / Export CSV / Clear buttons.

Sub-panel B — Timing jitter histogram
  • Bars show distribution of stimulus-to-marker offsets in ms.
  • Mean ± SD text overlay.
  • Amber if SD > 2 ms, green if SD ≤ 2 ms.

Sub-panel C — SNR growth curve
  • One point per accepted epoch.
  • Horizontal dashed line at 6 dB (minimum acceptable).
  • Line turns green when SNR crosses 6 dB.

Public API
----------
AveragingPanel(QWidget)
    update_average(avg, epoch_count, target, components, paradigm)
    add_jitter_sample(offset_ms)
    update_snr(snr_curve)
    reset()
    set_save_callback(fn)   — fn(path) called on Save PNG
    set_export_callback(fn) — fn(path) called on Export CSV
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from neuroep import config
from neuroep.processing.components import ComponentResult
from neuroep.processing.epochs import EPOCH_TIME_MS

logger = logging.getLogger(__name__)

# Minimum acceptable SNR threshold
_SNR_MIN_DB = 6.0

# Jitter histogram bin edges (ms)
_JITTER_BINS = np.linspace(-10.0, 10.0, 41, dtype=np.float32)   # 0.5 ms bins

# Accent colours per paradigm key
_PARADIGM_COLORS: dict[str, str] = {
    "vep_pattern":  "#534AB7",
    "vep_flash":    "#378ADD",
    "aep":          "#1D9E75",
    "p300_passive": "#BA7517",
}
_DEFAULT_COLOR = "#534AB7"


class AveragingPanel(QWidget):
    """
    Three-pane real-time averaging display.

    Parameters
    ----------
    parent : QWidget, optional
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(400)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        self._paradigm_color = _DEFAULT_COLOR
        self._save_cb:   Optional[Callable[[Path], None]] = None
        self._export_cb: Optional[Callable[[Path], None]] = None

        # History of previous averages for ghost traces: list of np.ndarray
        self._avg_history: list[np.ndarray] = []

        # Jitter samples (ms)
        self._jitter_samples: list[float] = []

        self._build_ui()

    # ── Public API ─────────────────────────────────────────────────────────

    def update_average(
        self,
        avg:         np.ndarray,
        epoch_count: int,
        target:      int,
        components:  list[ComponentResult],
        paradigm:    str = "vep_pattern",
        channel:     int = 0,
    ) -> None:
        """
        Refresh sub-panel A with a new grand average.

        Parameters
        ----------
        avg : np.ndarray, shape (n_channels, epoch_len)
        epoch_count : int
        target : int
            Target epoch count (shown in counter label).
        components : list[ComponentResult]
        paradigm : str
        channel : int
            Which channel row to display.
        """
        self._paradigm_color = _PARADIGM_COLORS.get(paradigm, _DEFAULT_COLOR)
        waveform = avg[channel, :].astype(np.float32)

        # Save current to history before drawing
        if len(self._avg_history) == 0 or not np.array_equal(
            self._avg_history[-1], waveform
        ):
            self._avg_history.append(waveform.copy())
            if len(self._avg_history) > 30:
                self._avg_history.pop(0)

        self._draw_ep(waveform, epoch_count, target, components)

    def add_jitter_sample(self, offset_ms: float) -> None:
        """Add one timing jitter measurement and refresh the histogram."""
        self._jitter_samples.append(float(offset_ms))
        self._draw_jitter()

    def update_snr(self, snr_curve: list[float]) -> None:
        """Refresh sub-panel C with the latest SNR growth curve."""
        self._draw_snr(snr_curve)

    def reset(self) -> None:
        """Clear all panels (call at session start)."""
        self._avg_history.clear()
        self._jitter_samples.clear()
        self._ep_plot.clear()
        self._jitter_plot.clear()
        self._snr_plot.clear()
        self._epoch_label.setText("0 / — epochs")
        self._snr_threshold_line.setVisible(True)
        logger.info("AveragingPanel reset.")

    def set_save_callback(self, fn: Callable[[Path], None]) -> None:
        """Register a callback invoked with the chosen file path on Save PNG."""
        self._save_cb = fn

    def set_export_callback(self, fn: Callable[[Path], None]) -> None:
        """Register a callback invoked with the chosen file path on Export CSV."""
        self._export_cb = fn

    # ── Build UI ───────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        outer = QHBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

        outer.addWidget(self._build_ep_panel(),     stretch=3)
        outer.addWidget(self._build_jitter_panel(), stretch=2)
        outer.addWidget(self._build_snr_panel(),    stretch=2)

    # ── Sub-panel A: EP waveform ───────────────────────────────────────────

    def _build_ep_panel(self) -> QWidget:
        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(2, 2, 2, 2)
        vbox.setSpacing(4)

        # Title row
        title_row = QHBoxLayout()
        lbl = QLabel("Averaged EP")
        lbl.setStyleSheet("font-weight: bold; font-size: 9pt;")
        title_row.addWidget(lbl)
        title_row.addStretch()
        self._epoch_label = QLabel("0 / — epochs")
        self._epoch_label.setStyleSheet("color: #9a9891; font-size: 9pt;")
        title_row.addWidget(self._epoch_label)
        vbox.addLayout(title_row)

        # Plot
        self._ep_widget = pg.PlotWidget(background="#0f1117")
        self._ep_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        ep = self._ep_widget.getPlotItem()
        ep.setLabel("bottom", "Time (ms)")
        ep.setLabel("left", "µV")
        ep.showGrid(x=True, y=True, alpha=0.15)
        ep.setXRange(float(EPOCH_TIME_MS[0]), float(EPOCH_TIME_MS[-1]), padding=0.02)

        # Reference lines
        ep.addItem(pg.InfiniteLine(
            pos=0, angle=90,
            pen=pg.mkPen(color="#ffffff", style=Qt.PenStyle.DashLine, width=1),
        ))
        ep.addItem(pg.InfiniteLine(
            pos=0, angle=0,
            pen=pg.mkPen(color="#ffffff", style=Qt.PenStyle.DashLine, width=1),
        ))

        self._ep_plot = ep
        vbox.addWidget(self._ep_widget)

        # Buttons
        btn_row = QHBoxLayout()
        for label, slot in [
            ("Save PNG",    self._on_save_png),
            ("Export CSV",  self._on_export_csv),
            ("Clear",       self._on_clear),
        ]:
            btn = QPushButton(label)
            btn.setFixedHeight(24)
            btn.setStyleSheet("font-size: 9pt; padding: 2px 8px;")
            btn.clicked.connect(slot)
            btn_row.addWidget(btn)
        vbox.addLayout(btn_row)

        return container

    # ── Sub-panel B: Jitter histogram ──────────────────────────────────────

    def _build_jitter_panel(self) -> QWidget:
        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(2, 2, 2, 2)
        vbox.setSpacing(4)

        lbl = QLabel("Timing jitter")
        lbl.setStyleSheet("font-weight: bold; font-size: 9pt;")
        vbox.addWidget(lbl)

        self._jitter_widget = pg.PlotWidget(background="#0f1117")
        self._jitter_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        jt = self._jitter_widget.getPlotItem()
        jt.setLabel("bottom", "Offset (ms)")
        jt.setLabel("left", "Count")
        jt.setXRange(-10, 10, padding=0.02)
        jt.showGrid(x=True, y=True, alpha=0.15)

        self._jitter_plot = jt
        self._jitter_bar  = pg.BarGraphItem(
            x0=[], x1=[], height=[], pen=None, brush="#1D9E75"
        )
        jt.addItem(self._jitter_bar)

        self._jitter_stats = pg.TextItem(
            text="", color="#e8e6de", anchor=(1.0, 0.0)
        )
        self._jitter_stats.setFont(QFont("Courier New", 8))
        jt.addItem(self._jitter_stats)

        vbox.addWidget(self._jitter_widget)
        return container

    # ── Sub-panel C: SNR growth curve ──────────────────────────────────────

    def _build_snr_panel(self) -> QWidget:
        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(2, 2, 2, 2)
        vbox.setSpacing(4)

        lbl = QLabel("SNR growth")
        lbl.setStyleSheet("font-weight: bold; font-size: 9pt;")
        vbox.addWidget(lbl)

        self._snr_widget = pg.PlotWidget(background="#0f1117")
        self._snr_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        snr = self._snr_widget.getPlotItem()
        snr.setLabel("bottom", "Epoch #")
        snr.setLabel("left", "SNR (dB)")
        snr.showGrid(x=True, y=True, alpha=0.15)

        # 6 dB threshold line
        self._snr_threshold_line = pg.InfiniteLine(
            pos=_SNR_MIN_DB, angle=0,
            pen=pg.mkPen(color="#BA7517", style=Qt.PenStyle.DashLine, width=1),
            label=f"{_SNR_MIN_DB} dB",
            labelOpts={"color": "#BA7517", "position": 0.05},
        )
        snr.addItem(self._snr_threshold_line)

        self._snr_curve_item = pg.PlotDataItem(
            pen=pg.mkPen(color="#534AB7", width=2),
            symbol="o",
            symbolSize=4,
            symbolBrush="#534AB7",
        )
        snr.addItem(self._snr_curve_item)

        self._snr_plot = snr
        vbox.addWidget(self._snr_widget)
        return container

    # ── Draw methods ───────────────────────────────────────────────────────

    def _draw_ep(
        self,
        waveform:    np.ndarray,
        epoch_count: int,
        target:      int,
        components:  list[ComponentResult],
    ) -> None:
        """Redraw the EP sub-panel."""
        self._ep_plot.clear()

        # Re-add reference lines after clear()
        self._ep_plot.addItem(pg.InfiniteLine(
            pos=0, angle=90,
            pen=pg.mkPen(color="#ffffff", style=Qt.PenStyle.DashLine, width=1),
        ))
        self._ep_plot.addItem(pg.InfiniteLine(
            pos=0, angle=0,
            pen=pg.mkPen(color="#ffffff", style=Qt.PenStyle.DashLine, width=1),
        ))

        n_history = len(self._avg_history)

        # Ghost traces — older averages fade out
        for i, old_avg in enumerate(self._avg_history[:-1]):
            age    = n_history - 1 - i           # 0 = most recent ghost
            alpha  = max(20, 120 - age * 8)      # fade: 120 → 20
            color  = pg.mkColor(self._paradigm_color)
            color.setAlpha(alpha)
            ghost = pg.PlotDataItem(
                x=EPOCH_TIME_MS, y=old_avg,
                pen=pg.mkPen(color=color, width=1),
            )
            self._ep_plot.addItem(ghost)

        # Current average — bold, full opacity
        self._ep_plot.addItem(pg.PlotDataItem(
            x=EPOCH_TIME_MS, y=waveform,
            pen=pg.mkPen(color=self._paradigm_color, width=2),
        ))

        # Component markers
        for comp in components:
            idx = int(np.argmin(np.abs(EPOCH_TIME_MS - comp.latency_ms)))
            amp = float(waveform[idx])

            # Vertical tick at peak
            self._ep_plot.addItem(pg.PlotDataItem(
                x=[comp.latency_ms, comp.latency_ms],
                y=[amp - 1.5, amp + 1.5],
                pen=pg.mkPen(color="#ffffff", width=2),
            ))
            # Label
            lbl = pg.TextItem(
                text=f"{comp.name}\n{comp.latency_ms:.0f} ms",
                color="#e8e6de",
                anchor=(0.5, 1.0),
            )
            lbl.setFont(QFont("Courier New", 8))
            lbl.setPos(comp.latency_ms, amp + 2.0)
            self._ep_plot.addItem(lbl)

        # Epoch counter
        self._epoch_label.setText(f"{epoch_count} / {target} epochs")

        # Auto-scale Y to waveform range
        ymin, ymax = float(waveform.min()) - 3, float(waveform.max()) + 3
        self._ep_plot.setYRange(ymin, ymax, padding=0.05)

    def _draw_jitter(self) -> None:
        """Redraw the jitter histogram."""
        if not self._jitter_samples:
            return

        arr   = np.array(self._jitter_samples, dtype=np.float32)
        counts, edges = np.histogram(arr, bins=_JITTER_BINS)

        self._jitter_bar.setOpts(
            x0=edges[:-1],
            x1=edges[1:],
            height=counts,
        )

        mean_j = float(arr.mean())
        sd_j   = float(arr.std())

        # Colour: amber = bad timing, green = good
        bar_color = "#1D9E75" if sd_j <= 2.0 else "#BA7517"
        self._jitter_bar.setOpts(brush=bar_color)

        self._jitter_stats.setText(
            f"mean {mean_j:+.2f} ms\nSD {sd_j:.2f} ms  n={len(arr)}"
        )
        # Anchor label to top-right of current view
        vb = self._jitter_plot.getViewBox()
        x_max = float(_JITTER_BINS[-1])
        y_max = float(counts.max()) if counts.max() > 0 else 1.0
        self._jitter_stats.setPos(x_max, y_max)
        self._jitter_plot.setYRange(0, y_max * 1.15, padding=0)

    def _draw_snr(self, snr_curve: list[float]) -> None:
        """Redraw the SNR growth curve."""
        if not snr_curve:
            return

        x = np.arange(1, len(snr_curve) + 1, dtype=np.float32)
        y = np.array(snr_curve, dtype=np.float32)

        # Turn the curve green once SNR crosses the threshold
        latest_snr = y[-1] if len(y) > 0 else float("nan")
        curve_color = "#1D9E75" if np.isfinite(latest_snr) and latest_snr >= _SNR_MIN_DB else "#534AB7"

        self._snr_curve_item.setData(
            x=x, y=y,
            pen=pg.mkPen(color=curve_color, width=2),
            symbolBrush=curve_color,
        )
        self._snr_plot.setXRange(1, max(len(snr_curve), 10), padding=0.05)
        finite = y[np.isfinite(y)]
        if len(finite) > 0:
            ymin = min(float(finite.min()) - 2, _SNR_MIN_DB - 3)
            ymax = max(float(finite.max()) + 2, _SNR_MIN_DB + 3)
            self._snr_plot.setYRange(ymin, ymax, padding=0)

    # ── Button slots ───────────────────────────────────────────────────────

    def _on_save_png(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save EP waveform", "", "PNG image (*.png)"
        )
        if path and self._save_cb:
            self._save_cb(Path(path))

    def _on_export_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export EP data", "", "CSV file (*.csv)"
        )
        if path and self._export_cb:
            self._export_cb(Path(path))

    def _on_clear(self) -> None:
        self.reset()
