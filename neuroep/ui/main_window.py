"""
ui/main_window.py — QMainWindow: top-level layout, menu bar, status bar,
and wiring between EEGPanel, ControlSidebar, and BoardManager.

Layout
------
┌─────────────────────────────────────────────────────────────┐
│  Top bar: logo · connection status · impedance · session ID  │
├──────────────┬──────────────────────────────────────────────┤
│              │                                              │
│   Sidebar    │         Live 16-channel EEG viewer          │
│   220px      │         (pyqtgraph, dark background)        │
│              │                                              │
├──────────────┴──────────────────────────────────────────────┤
│  Averaging panel placeholder (Phase 4)                       │
├─────────────────────────────────────────────────────────────┤
│  Status bar: board · SR · accepted · rejected · P100 · time  │
└─────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QCloseEvent
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from neuroep import config
from neuroep.acquisition.board import BoardManager
from neuroep.acquisition.markers import TriggerCode
from neuroep.processing.artifact import ArtifactChecker, RejectionStats
from neuroep.processing.averaging import RunningAverage
from neuroep.processing.components import ComponentDetector
from neuroep.processing.epochs import EpochExtractor
from neuroep.stimuli.base import BaseParadigm, psychopy_available
from neuroep.output.exporter import Exporter
from neuroep.ui.averaging_panel import AveragingPanel
from neuroep.ui.control_sidebar import ControlSidebar
from neuroep.ui.eeg_panel import EEGPanel
from neuroep.ui.theme import apply_dark_theme, apply_light_theme

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """
    Primary application window.

    Parameters
    ----------
    manager : BoardManager
        A connected board manager (handed over from ConnectDialog).
    """

    def __init__(self, manager: BoardManager) -> None:
        super().__init__()
        self._manager        = manager
        self._session_active = False
        self._session_start_time: Optional[datetime.datetime] = None
        self._accepted_epochs = 0
        self._rejected_epochs = 0

        # Processing pipeline (created fresh each session)
        self._extractor:  Optional[EpochExtractor]  = None
        self._checker:    Optional[ArtifactChecker] = None
        self._stats:      Optional[RejectionStats]  = None
        self._averager:   Optional[RunningAverage]  = None
        self._detector:   Optional[ComponentDetector] = None

        # Active paradigm thread
        self._paradigm: Optional[BaseParadigm] = None

        # Exporter for current session
        self._exporter: Optional[Exporter] = None

        # Buffer sample index at session start (for epoch extraction)
        self._session_buffer_start = 0

        self.setWindowTitle("NeuroEP Studio")
        self.setMinimumSize(1100, 700)

        self._build_ui()
        self._build_menu()
        self._connect_signals()

        # Attach board to EEG panel immediately
        self._eeg_panel.set_board(manager)

        # Clock timer for status bar
        self._clock_timer = QTimer(self)
        self._clock_timer.setInterval(1000)
        self._clock_timer.timeout.connect(self._update_clock)
        self._clock_timer.start()

        self._refresh_status_bar()
        logger.info("MainWindow initialised.")

    # ── UI construction ────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_top_bar())

        # Main content: sidebar + EEG panel side by side
        content_splitter = QSplitter(Qt.Orientation.Horizontal)
        content_splitter.setHandleWidth(2)

        self._sidebar   = ControlSidebar()
        self._eeg_panel = EEGPanel()

        content_splitter.addWidget(self._sidebar)
        content_splitter.addWidget(self._eeg_panel)
        content_splitter.setStretchFactor(0, 0)   # sidebar: fixed
        content_splitter.setStretchFactor(1, 1)   # eeg: expands
        content_splitter.setSizes([220, 900])

        root.addWidget(content_splitter, stretch=1)

        # Averaging panel (Phase 4)
        self._avg_panel = AveragingPanel()
        self._avg_panel.setStyleSheet("border-top: 1px solid #2e3148;")
        root.addWidget(self._avg_panel)

        # Status bar
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_labels: dict[str, QLabel] = {}
        for key in ["board", "sr", "accepted", "rejected", "p100", "elapsed"]:
            lbl = QLabel()
            lbl.setContentsMargins(8, 0, 8, 0)
            self._status_bar.addPermanentWidget(lbl)
            self._status_labels[key] = lbl

    def _build_top_bar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(48)
        bar.setStyleSheet("background: #1a1d27; border-bottom: 1px solid #2e3148;")

        hbox = QHBoxLayout(bar)
        hbox.setContentsMargins(16, 0, 16, 0)

        logo = QLabel("🧠 NeuroEP Studio")
        logo.setStyleSheet(
            "font-size: 14pt; font-weight: bold; color: #e8e6de; background: transparent;"
        )
        hbox.addWidget(logo)
        hbox.addStretch()

        self._conn_label = QLabel("● Connected")
        self._conn_label.setStyleSheet("color: #1D9E75; background: transparent;")
        hbox.addWidget(self._conn_label)

        hbox.addSpacing(24)

        self._session_id_label = QLabel("No session")
        self._session_id_label.setStyleSheet(
            "color: #9a9891; background: transparent; font-size: 9pt;"
        )
        hbox.addWidget(self._session_id_label)

        return bar

    def _build_menu(self) -> None:
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("File")

        act_new = QAction("New session", self)
        act_new.setShortcut("Ctrl+N")
        act_new.triggered.connect(self._new_session)
        file_menu.addAction(act_new)

        file_menu.addSeparator()

        act_report = QAction("Save clinical report…", self)
        act_report.setShortcut("Ctrl+R")
        act_report.triggered.connect(self._save_report)
        file_menu.addAction(act_report)

        act_save_edf = QAction("Save raw EEG (EDF+)…", self)
        act_save_edf.triggered.connect(self._save_edf)
        file_menu.addAction(act_save_edf)

        file_menu.addSeparator()

        act_quit = QAction("Quit", self)
        act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        # View menu
        view_menu = menu_bar.addMenu("View")

        act_dark = QAction("Dark theme", self)
        act_dark.triggered.connect(
            lambda: apply_dark_theme(QApplication.instance())  # type: ignore[arg-type]
        )
        view_menu.addAction(act_dark)

        act_light = QAction("Light theme", self)
        act_light.triggered.connect(
            lambda: apply_light_theme(QApplication.instance())  # type: ignore[arg-type]
        )
        view_menu.addAction(act_light)

        # Tools menu
        tools_menu = menu_bar.addMenu("Tools")

        act_timing = QAction("Timing validation…", self)
        act_timing.triggered.connect(self._open_timing_validation)
        tools_menu.addAction(act_timing)

    def _connect_signals(self) -> None:
        sb = self._sidebar

        # Filter controls → EEG panel
        sb.sensitivity_changed.connect(self._eeg_panel.set_sensitivity)
        sb.highpass_changed.connect(self._eeg_panel.set_highpass)
        sb.lowpass_changed.connect(self._eeg_panel.set_lowpass)
        sb.notch_changed.connect(self._eeg_panel.set_notch)

        # Session controls
        sb.session_start.connect(self._start_session)
        sb.session_stop.connect(self._stop_session)
        sb.timing_validation_requested.connect(self._open_timing_validation)

    # ── Session management ─────────────────────────────────────────────────

    def _start_session(self) -> None:
        if self._session_active:
            return

        subject  = self._sidebar.get_subject_id()
        paradigm = self._sidebar.get_paradigm_key()
        n_trials = self._sidebar.get_target_epochs()
        stim_rate = self._sidebar.get_stim_rate()

        self._session_active          = True
        self._session_start_time      = datetime.datetime.now()
        self._accepted_epochs         = 0
        self._rejected_epochs         = 0
        self._session_buffer_start    = self._manager.ring_buffer.n_samples

        # Fresh processing pipeline
        self._extractor = EpochExtractor()
        self._checker   = ArtifactChecker()
        self._stats     = RejectionStats()
        self._averager  = RunningAverage()
        self._detector  = ComponentDetector()

        session_id = (
            f"{self._session_start_time.strftime('%Y%m%d_%H%M%S')}"
            f"_{subject or 'ANON'}_{paradigm}"
        )
        self._session_id_label.setText(f"Session: {session_id}")
        self._sidebar.set_running(True)
        self._refresh_status_bar()
        logger.info("Session started: %s", session_id)

        # Reset averaging panel and wire save/export callbacks
        self._avg_panel.reset()
        self._exporter = Exporter(
            session_id   = session_id,
            paradigm     = paradigm,
            subject_id   = subject or "ANON",
            n_epochs     = 0,
            channel_name = config.CHANNEL_NAMES[
                config.EP_CHANNELS.get(paradigm.upper().split("_")[0], [0])[0]
            ],
        )
        self._avg_panel.set_save_callback(self._on_save_png)
        self._avg_panel.set_export_callback(self._on_export_csv)

        # Launch paradigm thread (requires PsychoPy)
        if not psychopy_available():
            QMessageBox.warning(
                self,
                "PsychoPy not installed",
                "PsychoPy is required for stimulus delivery.\n"
                "Run: pip install psychopy\n\n"
                "EEG recording continues without stimuli.",
            )
            return

        self._paradigm = self._create_paradigm(paradigm, n_trials, stim_rate)
        if self._paradigm:
            self._paradigm.marker_sent.connect(self._on_marker_sent)
            self._paradigm.trial_completed.connect(self._on_trial_completed)
            self._paradigm.paradigm_finished.connect(self._on_paradigm_finished)
            self._paradigm.error_occurred.connect(self._on_paradigm_error)
            self._paradigm.start()

    def _stop_session(self) -> None:
        if not self._session_active:
            return

        if self._paradigm and self._paradigm.isRunning():
            self._paradigm.stop()
            self._paradigm.wait(3000)

        self._session_active = False
        self._paradigm = None
        self._sidebar.set_running(False)
        self._refresh_status_bar()
        logger.info(
            "Session stopped. Accepted: %d  Rejected: %d",
            self._accepted_epochs,
            self._rejected_epochs,
        )

    def _create_paradigm(
        self, key: str, n_trials: int, stim_rate: float
    ) -> Optional[BaseParadigm]:
        """Instantiate the correct paradigm class for *key*."""
        from neuroep.stimuli.vep_pattern import PatternVEP
        from neuroep.stimuli.vep_flash   import FlashVEP
        from neuroep.stimuli.aep         import AuditoryEP
        from neuroep.stimuli.p300_passive import PassiveP300

        kwargs = dict(
            board_manager = self._manager,
            n_trials      = n_trials,
            stim_rate     = stim_rate,
        )
        paradigm_map = {
            "vep_pattern":  lambda: PatternVEP(**kwargs),
            "vep_flash":    lambda: FlashVEP(**kwargs),
            "aep":          lambda: AuditoryEP(**kwargs),
            "p300_passive": lambda: PassiveP300(
                board_manager = self._manager,
                n_trials      = n_trials * 5,   # ~20% oddballs → 5× total trials
                stim_rate     = stim_rate,
            ),
        }
        factory = paradigm_map.get(key)
        if factory is None:
            logger.error("Unknown paradigm key: %s", key)
            return None
        return factory()

    # ── Paradigm signal handlers ───────────────────────────────────────────

    def _on_marker_sent(self, code: int, t_onset: float) -> None:
        """Process a new marker: extract epoch, check artifact, update average + panel."""
        if (self._extractor is None or self._checker is None
                or self._stats is None or self._averager is None):
            return

        raw            = self._manager.ring_buffer.snapshot()
        abs_marker_idx = self._manager._sample_count

        markers = [(code, abs_marker_idx)]
        epochs  = self._extractor.extract(raw, markers, buffer_start_index=0)

        for ep in epochs:
            self._checker.check(ep)
            self._stats.update(ep)
            self._averager.add(ep)

            if ep.accepted:
                self._accepted_epochs += 1
            else:
                self._rejected_epochs += 1

        self._update_averaging_panel()
        self._refresh_status_bar()

        if self._stats and self._stats.high_rejection:
            self.statusBar().showMessage(
                "⚠ High rejection rate — check electrodes", 5000
            )

    def _on_trial_completed(self, trial_num: int) -> None:
        """Update status bar trial counter."""
        target = self._sidebar.get_target_epochs()
        self._status_labels["accepted"].setText(
            f"Accepted: {self._accepted_epochs} / {target}"
        )
        # Auto-stop for DoC paradigms when target reached
        paradigm = self._sidebar.get_paradigm_key()
        if paradigm in ("vep_flash", "p300_passive"):
            if self._accepted_epochs >= target:
                self._stop_session()

    def _on_paradigm_finished(self) -> None:
        """Called when the paradigm thread exits naturally — offer to save report."""
        if not self._session_active:
            return
        self._stop_session()
        if self._averager and self._averager.epoch_count > 0:
            reply = QMessageBox.question(
                self,
                "Session complete",
                f"Collected {self._accepted_epochs} epochs.\n\n"
                "Save a clinical report now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._save_report()

    def _on_paradigm_error(self, message: str) -> None:
        """Show error dialog if paradigm thread crashes."""
        QMessageBox.critical(self, "Stimulus error", message)
        self._stop_session()

    def _update_averaging_panel(self) -> None:
        """Detect components, push to averaging panel, update status bar."""
        if self._averager is None or self._averager.epoch_count == 0:
            return
        avg = self._averager.current_avg
        if avg is None:
            return

        paradigm = self._sidebar.get_paradigm_key()
        ep_key   = paradigm.upper().split("_")[0]
        ep_chs   = config.EP_CHANNELS.get(ep_key, [0])
        channel  = ep_chs[0] if ep_chs else 0
        target   = self._sidebar.get_target_epochs()

        components = (
            self._detector.detect(avg, channel, paradigm)
            if self._averager.epoch_count >= 5 else []
        )

        # Update averaging panel sub-panels A and C
        self._avg_panel.update_average(
            avg         = avg,
            epoch_count = self._accepted_epochs,
            target      = target,
            components  = components,
            paradigm    = paradigm,
            channel     = channel,
        )
        self._avg_panel.update_snr(self._averager.snr_curve)

        # Status bar component label
        if components:
            c = components[0]
            self._status_labels["p100"].setText(f"{c.name}: {c.latency_ms:.0f} ms")

    def _on_save_png(self, path) -> None:
        """Save EP waveform PNG via Exporter."""
        if self._exporter is None or self._averager is None:
            return
        avg = self._averager.current_avg
        if avg is None:
            return
        paradigm = self._sidebar.get_paradigm_key()
        ep_ch    = config.EP_CHANNELS.get(paradigm.upper().split("_")[0], [0])[0]
        components = self._detector.detect(avg, ep_ch, paradigm) if self._detector else []
        self._exporter._n_epochs = self._accepted_epochs
        try:
            self._exporter.save_png(path, avg, ep_ch, components)
            self.statusBar().showMessage(f"Saved PNG: {path.name}", 4000)
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))

    def _on_export_csv(self, path) -> None:
        """Export EP CSV via Exporter."""
        if self._exporter is None or self._averager is None:
            return
        avg = self._averager.current_avg
        if avg is None:
            return
        paradigm = self._sidebar.get_paradigm_key()
        ep_ch    = config.EP_CHANNELS.get(paradigm.upper().split("_")[0], [0])[0]
        components = self._detector.detect(avg, ep_ch, paradigm) if self._detector else []
        self._exporter._n_epochs = self._accepted_epochs
        try:
            self._exporter.save_csv(path, avg, ep_ch, components)
            self.statusBar().showMessage(f"Exported CSV: {path.name}", 4000)
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))

    def _build_report_data(self):
        """Collect all session data into a ReportData instance."""
        from neuroep.output.report import ReportData
        avg        = self._averager.current_avg if self._averager else None
        paradigm   = self._sidebar.get_paradigm_key()
        ep_key     = paradigm.upper().split("_")[0]
        ep_ch      = config.EP_CHANNELS.get(ep_key, [0])[0]
        components = []
        if avg is not None and self._detector:
            components = self._detector.detect(avg, ep_ch, paradigm)

        snr = float("nan")
        if self._averager and len(self._averager.snr_curve) > 0:
            valid = [v for v in self._averager.snr_curve if np.isfinite(v)]
            if valid:
                snr = valid[-1]

        return ReportData(
            session_id      = self._session_id_label.text().replace("Session: ", ""),
            subject_id      = self._sidebar.get_subject_id(),
            paradigm        = paradigm,
            avg             = avg,
            display_channel = ep_ch,
            channel_name    = config.CHANNEL_NAMES[ep_ch],
            components      = components,
            n_accepted      = self._accepted_epochs,
            n_rejected      = self._rejected_epochs,
            snr_db          = snr,
            eye_tested      = self._sidebar.get_eye_tested(),
            clinical_note   = self._sidebar.get_clinical_note(),
        )

    def _save_report(self) -> None:
        """Prompt for file path and save a PDF clinical report."""
        if self._averager is None or self._averager.epoch_count == 0:
            QMessageBox.information(
                self, "No data",
                "Run a session and collect epochs before saving a report."
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save clinical report", "", "PDF report (*.pdf)"
        )
        if not path:
            return

        from neuroep.output.report import ReportWriter
        try:
            writer = ReportWriter(self._build_report_data())
            writer.save_pdf(Path(path))
            self.statusBar().showMessage(f"Report saved: {Path(path).name}", 5000)
        except Exception as exc:
            QMessageBox.critical(self, "Report failed", str(exc))

    def _save_edf(self) -> None:
        """Prompt for file path and save the raw EEG as EDF+."""
        if not self._manager.is_connected:
            QMessageBox.information(self, "No data", "No board connected.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save raw EEG", "", "EDF+ file (*.edf)"
        )
        if not path:
            return

        raw = self._manager.ring_buffer.snapshot()
        if raw.shape[1] == 0:
            QMessageBox.information(self, "No data", "Ring buffer is empty.")
            return

        try:
            self._exporter = self._exporter or Exporter()
            self._exporter.save_edf(Path(path), raw)
            self.statusBar().showMessage(f"EDF+ saved: {Path(path).name}", 5000)
        except Exception as exc:
            QMessageBox.critical(self, "EDF+ save failed", str(exc))

    def _new_session(self) -> None:
        if self._session_active:
            reply = QMessageBox.question(
                self,
                "New session",
                "A session is currently running. Stop it and start a new session?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            self._stop_session()

        self._accepted_epochs = 0
        self._rejected_epochs = 0
        self._session_id_label.setText("No session")
        self._refresh_status_bar()

    # ── Status bar ─────────────────────────────────────────────────────────

    def _refresh_status_bar(self) -> None:
        board_name = "Synthetic" if not self._manager.is_connected else "Cyton+Daisy"
        self._status_labels["board"].setText(f"Board: {board_name}")
        self._status_labels["sr"].setText(f"SR: {config.BOARD_SAMPLE_RATE} Hz")
        self._status_labels["accepted"].setText(
            f"Accepted: {self._accepted_epochs}"
        )
        self._status_labels["rejected"].setText(
            f"Rejected: {self._rejected_epochs}"
        )
        self._status_labels["p100"].setText("P100: —")

    def _update_clock(self) -> None:
        if self._session_active and self._session_start_time:
            elapsed = datetime.datetime.now() - self._session_start_time
            s = int(elapsed.total_seconds())
            self._status_labels["elapsed"].setText(
                f"Elapsed: {s // 60:02d}:{s % 60:02d}"
            )
        else:
            self._status_labels["elapsed"].setText("")

    # ── Timing validation ──────────────────────────────────────────────────

    def _open_timing_validation(self) -> None:
        from neuroep.ui.validation_window import ValidationWindow
        dlg = ValidationWindow(serial_port=config.SERIAL_PORT, parent=self)
        dlg.show()

    # ── Close event ────────────────────────────────────────────────────────

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._session_active:
            reply = QMessageBox.question(
                self,
                "Quit",
                "A session is running. Stop it and quit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()
                return

        self._eeg_panel.detach_board()
        self._manager.disconnect()
        event.accept()
        logger.info("Application closed.")


