"""
ui/validation_window.py — Timing validation dialog with three tabs.

Each tab corresponds to one validation mode:
  Tab A — Synthetic pipeline test  (no hardware)
  Tab B — Photodiode hardware test (requires Cyton + photodiode)
  Tab C — Square wave test         (requires Cyton or uses synthetic)

Each tab contains:
  • Description label
  • Run button + progress bar
  • Live-updating histogram (pyqtgraph BarGraphItem)
  • Summary table: mean, SD, min, max, n_trials, pass/fail verdict
  • Export results button → CSV

Pass/fail criteria
------------------
Mode A: mean < 1 ms,  SD < 0.5 ms
Mode B: mean < 5 ms,  SD < 2.0 ms
Mode C: mean < 2 ms,  SD < 1.0 ms
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from neuroep import config

logger = logging.getLogger(__name__)

# ── Pass/fail thresholds per mode ──────────────────────────────────────────
@dataclass
class _Thresholds:
    mean_ms: float
    sd_ms:   float
    label:   str

_THRESHOLDS = {
    "A": _Thresholds(mean_ms=1.0,  sd_ms=0.5, label="Synthetic pipeline"),
    "B": _Thresholds(mean_ms=5.0,  sd_ms=2.0, label="Photodiode hardware"),
    "C": _Thresholds(mean_ms=2.0,  sd_ms=1.0, label="Cyton square wave"),
}

# Histogram bin edges in ms
_HIST_BINS_A = np.linspace(-5.0,  5.0,  41)   # ±5 ms, 0.25 ms bins
_HIST_BINS_B = np.linspace(-20.0, 20.0, 41)   # ±20 ms, 1 ms bins
_HIST_BINS_C = np.linspace(-10.0, 10.0, 41)   # ±10 ms, 0.5 ms bins

_HIST_BINS = {"A": _HIST_BINS_A, "B": _HIST_BINS_B, "C": _HIST_BINS_C}


class ValidationWindow(QDialog):
    """
    Modal timing validation dialog.

    Parameters
    ----------
    serial_port : str
        Serial port for real-hardware tests.
    parent : QWidget, optional
    """

    def __init__(self, serial_port: str = config.SERIAL_PORT, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Timing Validation")
        self.setMinimumSize(700, 520)
        self.setModal(False)   # non-modal so EEG continues
        self._serial_port = serial_port
        self._workers: dict[str, object] = {}   # mode → QThread
        self._samples: dict[str, list[float]] = {"A": [], "B": [], "C": []}
        self._build_ui()

    # ── Build UI ───────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)

        tabs = QTabWidget()
        tabs.addTab(self._build_tab("A"), "A — Synthetic pipeline")
        tabs.addTab(self._build_tab("B"), "B — Photodiode hardware")
        tabs.addTab(self._build_tab("C"), "C — Square wave")
        root.addWidget(tabs)

        close_btn = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close_btn.rejected.connect(self.reject)
        root.addWidget(close_btn)

    def _build_tab(self, mode: str) -> QWidget:
        th = _THRESHOLDS[mode]
        tab = QWidget()
        vbox = QVBoxLayout(tab)
        vbox.setSpacing(8)

        # Description
        desc_text = {
            "A": (
                "Tests the Python processing pipeline without hardware.\n"
                "Inserts markers into the synthetic BrainFlow board and measures\n"
                "the round-trip latency.\n"
                f"Pass criterion: mean < {th.mean_ms} ms, SD < {th.sd_ms} ms."
            ),
            "B": (
                "Measures true end-to-end latency including display refresh.\n"
                "Requires a real Cyton+Daisy board and a photodiode wired to D11.\n"
                "A full-screen flash is presented and the photodiode onset is detected.\n"
                f"Pass criterion: mean < {th.mean_ms} ms, SD < {th.sd_ms} ms."
            ),
            "C": (
                "Validates BrainFlow marker channel timing vs EEG data channel.\n"
                "Uses the Cyton internal ~8 Hz square wave on channel 1.\n"
                "Synthetic mode available for testing without hardware.\n"
                f"Pass criterion: mean < {th.mean_ms} ms, SD < {th.sd_ms} ms."
            ),
        }[mode]
        desc = QLabel(desc_text)
        desc.setStyleSheet("color: #9a9891; font-size: 9pt;")
        desc.setWordWrap(True)
        vbox.addWidget(desc)

        # Controls row
        ctrl_row = QHBoxLayout()
        run_btn  = QPushButton(f"▶  Run Mode {mode}")
        run_btn.setFixedWidth(160)
        run_btn.clicked.connect(lambda _, m=mode: self._run_test(m))
        ctrl_row.addWidget(run_btn)

        stop_btn = QPushButton("■  Stop")
        stop_btn.setFixedWidth(80)
        stop_btn.setEnabled(False)
        stop_btn.clicked.connect(lambda _, m=mode: self._stop_test(m))
        ctrl_row.addWidget(stop_btn)

        ctrl_row.addStretch()
        n_label = QLabel(f"n = {config.TIMING_N_TRIALS} trials")
        n_label.setStyleSheet("color: #9a9891; font-size: 9pt;")
        ctrl_row.addWidget(n_label)
        vbox.addLayout(ctrl_row)

        # Progress bar
        progress = QProgressBar()
        progress.setRange(0, 100)
        progress.setValue(0)
        progress.setFixedHeight(16)
        vbox.addWidget(progress)

        # Histogram
        hist_widget = pg.PlotWidget(background="#0f1117")
        hist_widget.setFixedHeight(160)
        hist_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        hist_plot = hist_widget.getPlotItem()
        hist_plot.setLabel("bottom", "Offset (ms)")
        hist_plot.setLabel("left", "Count")
        hist_plot.showGrid(x=True, y=True, alpha=0.15)
        bins = _HIST_BINS[mode]
        hist_plot.setXRange(float(bins[0]), float(bins[-1]), padding=0.02)

        bar_item = pg.BarGraphItem(x0=[], x1=[], height=[], pen=None, brush="#534AB7")
        hist_plot.addItem(bar_item)

        stats_text = pg.TextItem(text="", color="#e8e6de", anchor=(1.0, 0.0))
        stats_text.setFont(QFont("Courier New", 8))
        hist_plot.addItem(stats_text)

        vbox.addWidget(hist_widget)

        # Summary table
        table = QTableWidget(6, 2)
        table.setHorizontalHeaderLabels(["Metric", "Value"])
        table.verticalHeader().setVisible(False)
        table.setFixedHeight(160)
        table.horizontalHeader().setStretchLastSection(True)
        for row, label in enumerate(
            ["Mean (ms)", "SD (ms)", "Min (ms)", "Max (ms)", "n trials", "Verdict"]
        ):
            table.setItem(row, 0, QTableWidgetItem(label))
            table.setItem(row, 1, QTableWidgetItem("—"))
        vbox.addWidget(table)

        # Export button
        export_btn = QPushButton("Export results CSV")
        export_btn.setEnabled(False)
        export_btn.clicked.connect(lambda _, m=mode: self._export_csv(m))
        vbox.addWidget(export_btn)

        # Store refs keyed by mode (accessed as self._run_btn_A, etc.)
        setattr(self, f"_run_btn_{mode}",    run_btn)
        setattr(self, f"_stop_btn_{mode}",   stop_btn)
        setattr(self, f"_progress_{mode}",   progress)
        setattr(self, f"_bar_{mode}",        bar_item)
        setattr(self, f"_stats_text_{mode}", stats_text)
        setattr(self, f"_hist_plot_{mode}",  hist_plot)
        setattr(self, f"_table_{mode}",      table)
        setattr(self, f"_export_btn_{mode}", export_btn)

        return tab

    # ── Run / stop ─────────────────────────────────────────────────────────

    def _run_test(self, mode: str) -> None:
        """Launch the appropriate worker thread for *mode*."""
        if mode in self._workers and self._workers[mode].isRunning():
            return

        self._samples[mode] = []
        self._set_running(mode, True)

        if mode == "A":
            from neuroep.validation.synthetic_test import SyntheticTimingTest
            worker = SyntheticTimingTest()
        elif mode == "B":
            from neuroep.validation.photodiode_test import PhotodiodeTimingTest
            worker = PhotodiodeTimingTest(serial_port=self._serial_port)
        else:
            from neuroep.validation.squarewave_test import SquareWaveTimingTest
            worker = SquareWaveTimingTest(use_synthetic=True)

        worker.progress.connect(lambda v, m=mode: self._on_progress(m, v))
        worker.result.connect(lambda vals, m=mode: self._on_result(m, vals))
        worker.error.connect(lambda msg, m=mode: self._on_error(m, msg))
        self._workers[mode] = worker
        worker.start()

    def _stop_test(self, mode: str) -> None:
        worker = self._workers.get(mode)
        if worker and worker.isRunning():
            worker.stop()
            worker.wait(3000)
        self._set_running(mode, False)

    def _set_running(self, mode: str, running: bool) -> None:
        getattr(self, f"_run_btn_{mode}").setEnabled(not running)
        getattr(self, f"_stop_btn_{mode}").setEnabled(running)
        if not running:
            getattr(self, f"_progress_{mode}").setValue(0)

    # ── Worker signal handlers ─────────────────────────────────────────────

    def _on_progress(self, mode: str, value: int) -> None:
        getattr(self, f"_progress_{mode}").setValue(value)

    def _on_result(self, mode: str, values: list[float]) -> None:
        self._samples[mode] = values
        self._set_running(mode, False)
        getattr(self, f"_export_btn_{mode}").setEnabled(bool(values))
        self._update_histogram(mode, values)
        self._update_table(mode, values)

    def _on_error(self, mode: str, message: str) -> None:
        self._set_running(mode, False)
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.critical(self, f"Mode {mode} error", message)

    # ── Histogram + table update ───────────────────────────────────────────

    def _update_histogram(self, mode: str, values: list[float]) -> None:
        if not values:
            return
        arr    = np.array(values, dtype=np.float32)
        bins   = _HIST_BINS[mode]
        counts, edges = np.histogram(arr, bins=bins)

        bar   = getattr(self, f"_bar_{mode}")
        bar.setOpts(x0=edges[:-1], x1=edges[1:], height=counts)

        th  = _THRESHOLDS[mode]
        sd  = float(arr.std())
        clr = "#1D9E75" if sd <= th.sd_ms else "#BA7517"
        bar.setOpts(brush=clr)

        stats = getattr(self, f"_stats_text_{mode}")
        stats.setText(
            f"mean {arr.mean():+.2f} ms\n"
            f"SD   {sd:.2f} ms\n"
            f"n    {len(arr)}"
        )
        plot = getattr(self, f"_hist_plot_{mode}")
        y_max = float(counts.max()) if counts.max() > 0 else 1
        plot.setYRange(0, y_max * 1.2, padding=0)
        stats.setPos(float(bins[-1]), y_max * 1.15)

    def _update_table(self, mode: str, values: list[float]) -> None:
        if not values:
            return
        arr = np.array(values, dtype=np.float64)
        th  = _THRESHOLDS[mode]
        mean_v = float(arr.mean())
        sd_v   = float(arr.std())
        passed = mean_v < th.mean_ms and sd_v < th.sd_ms
        verdict = "✓ PASS" if passed else "✗ FAIL"

        table = getattr(self, f"_table_{mode}")
        data  = [
            f"{mean_v:.3f}",
            f"{sd_v:.3f}",
            f"{float(arr.min()):.3f}",
            f"{float(arr.max()):.3f}",
            str(len(arr)),
            verdict,
        ]
        for row, val in enumerate(data):
            item = QTableWidgetItem(val)
            if row == 5:
                item.setForeground(
                    pg.mkColor("#1D9E75") if passed else pg.mkColor("#D85A30")
                )
            table.setItem(row, 1, item)

    # ── Export ─────────────────────────────────────────────────────────────

    def _export_csv(self, mode: str) -> None:
        values = self._samples.get(mode, [])
        if not values:
            return

        path, _ = QFileDialog.getSaveFileName(
            self, f"Export Mode {mode} results", "", "CSV file (*.csv)"
        )
        if not path:
            return

        th  = _THRESHOLDS[mode]
        arr = np.array(values, dtype=np.float64)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["# NeuroEP Studio — Timing Validation"])
            writer.writerow([f"# Mode {mode}: {th.label}"])
            writer.writerow([f"# mean_ms={arr.mean():.3f}", f"sd_ms={arr.std():.3f}"])
            writer.writerow(["trial", "offset_ms"])
            for i, v in enumerate(values, start=1):
                writer.writerow([i, f"{v:.4f}"])

        logger.info("Validation results exported to %s", path)
