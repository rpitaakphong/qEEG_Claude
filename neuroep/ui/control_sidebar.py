"""
ui/control_sidebar.py — Left sidebar with paradigm selector, filter controls,
protocol settings, and session action buttons.

Emits Qt signals consumed by MainWindow to drive EEGPanel and session logic.
"""

from __future__ import annotations

import logging
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from neuroep import config

logger = logging.getLogger(__name__)

# Paradigm definitions ──────────────────────────────────────────────────────
_PARADIGMS: list[dict] = [
    {
        "key":    "vep_pattern",
        "label":  "Pattern VEP",
        "accent": "#534AB7",
        "doc":    False,
        "epochs": 100,
        "rate":   2.0,
    },
    {
        "key":    "vep_flash",
        "label":  "Flash VEP",
        "accent": "#378ADD",
        "doc":    True,
        "epochs": 50,
        "rate":   1.0,
    },
    {
        "key":    "aep",
        "label":  "Auditory EP",
        "accent": "#1D9E75",
        "doc":    False,
        "epochs": 100,
        "rate":   2.0,
    },
    {
        "key":    "p300_passive",
        "label":  "P300 Passive",
        "accent": "#BA7517",
        "doc":    True,
        "epochs": 50,
        "rate":   1.0,
    },
]


class ControlSidebar(QWidget):
    """
    Left sidebar widget.

    Signals
    -------
    sensitivity_changed(float)   : µV half-range (5–200)
    highpass_changed(float)      : HP cutoff Hz
    lowpass_changed(float)       : LP cutoff Hz
    notch_changed(object)        : float Hz or None
    paradigm_changed(str)        : paradigm key string
    session_start()              : user clicked Start
    session_stop()               : user clicked Stop
    """

    sensitivity_changed = pyqtSignal(float)
    highpass_changed    = pyqtSignal(float)
    lowpass_changed     = pyqtSignal(float)
    notch_changed       = pyqtSignal(object)
    paradigm_changed    = pyqtSignal(str)
    session_start       = pyqtSignal()
    session_stop        = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(300)
        self._current_paradigm: dict = _PARADIGMS[0]
        self._running: bool = False
        self._build_ui()

    # ── Public API ─────────────────────────────────────────────────────────

    def set_running(self, running: bool) -> None:
        """Toggle Start/Stop button appearance."""
        self._running = running
        self._btn_start.setVisible(not running)
        self._btn_stop.setVisible(running)
        self._paradigm_group_box.setEnabled(not running)

    def get_subject_id(self) -> str:
        return self._subject_edit.text().strip()

    def get_eye_tested(self) -> str:
        return self._eye_combo.currentText()

    def get_target_epochs(self) -> int:
        return self._epochs_spin.value()

    def get_stim_rate(self) -> float:
        return self._rate_spin.value()

    def get_clinical_note(self) -> str:
        if hasattr(self, "_note_edit"):
            return self._note_edit.toPlainText().strip()
        return ""

    def get_paradigm_key(self) -> str:
        return self._current_paradigm["key"]

    # ── Build UI ───────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Scrollable content area ────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")

        content = QWidget()
        content.setStyleSheet("background: transparent;")
        vbox = QVBoxLayout(content)
        vbox.setContentsMargins(10, 12, 10, 12)
        vbox.setSpacing(14)

        vbox.addWidget(self._build_paradigm_section())
        vbox.addWidget(self._build_signal_section())
        vbox.addWidget(self._build_protocol_section())
        vbox.addStretch()
        vbox.addWidget(self._build_action_section())

        scroll.setWidget(content)
        outer.addWidget(scroll)

    # ── Section builders ───────────────────────────────────────────────────

    def _build_paradigm_section(self) -> QGroupBox:
        self._paradigm_group_box = QGroupBox("Paradigm")
        vbox = QVBoxLayout(self._paradigm_group_box)
        vbox.setContentsMargins(12, 16, 12, 12)
        vbox.setSpacing(10)

        self._paradigm_btn_group = QButtonGroup(self)
        self._paradigm_btn_group.setExclusive(True)
        self._paradigm_radio: list[QRadioButton] = []

        for i, p in enumerate(_PARADIGMS):
            rb = QRadioButton(p["label"])
            rb.setStyleSheet(
                f"QRadioButton {{ padding: 2px 0; }}"
                f"QRadioButton::indicator:checked {{ background-color: {p['accent']}; "
                f"border: 2px solid {p['accent']}; border-radius: 6px; }}"
            )
            self._paradigm_btn_group.addButton(rb, i)
            vbox.addWidget(rb)
            self._paradigm_radio.append(rb)
            if i == 0:
                rb.setChecked(True)

        self._paradigm_btn_group.idClicked.connect(self._on_paradigm_changed)

        # Divider
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet("border: none; border-top: 1px solid #2e3148; margin: 2px 0;")
        vbox.addWidget(divider)

        # Show-all toggle
        self.show_all_cb = QCheckBox("Show all 16 channels")
        self.show_all_cb.setChecked(False)
        self.show_all_cb.setStyleSheet("QCheckBox { padding: 2px 0; }")
        self.show_all_cb.stateChanged.connect(self._on_show_all_toggled)
        vbox.addWidget(self.show_all_cb)

        # DoC badge
        self._doc_badge = QLabel("DoC mode")
        self._doc_badge.setObjectName("doc_badge")
        self._doc_badge.setVisible(False)
        vbox.addWidget(self._doc_badge)

        return self._paradigm_group_box

    def _build_signal_section(self) -> QGroupBox:
        grp = QGroupBox("Signal controls")
        vbox = QVBoxLayout(grp)
        vbox.setContentsMargins(12, 16, 12, 12)
        vbox.setSpacing(16)

        # Sensitivity
        self._sens_label = QLabel(f"{int(config.DEFAULT_SENSITIVITY)} µV")
        self._sens_slider = _make_slider(5, 200, int(config.DEFAULT_SENSITIVITY))
        self._sens_slider.valueChanged.connect(self._on_sensitivity)
        vbox.addLayout(_slider_row("Sensitivity", self._sens_label, self._sens_slider))

        # High-pass
        self._hp_label = QLabel(f"{config.DEFAULT_HP_HZ:.1f} Hz")
        self._hp_slider = _make_slider(1, 100, int(config.DEFAULT_HP_HZ * 10))
        self._hp_slider.valueChanged.connect(self._on_hp)
        vbox.addLayout(_slider_row("High-pass", self._hp_label, self._hp_slider))

        # Low-pass
        self._lp_label = QLabel(f"{int(config.DEFAULT_LP_HZ)} Hz")
        self._lp_slider = _make_slider(10, 100, int(config.DEFAULT_LP_HZ))
        self._lp_slider.valueChanged.connect(self._on_lp)
        vbox.addLayout(_slider_row("Low-pass", self._lp_label, self._lp_slider))

        # Notch — inline label + combo
        notch_row = QHBoxLayout()
        notch_row.setSpacing(8)
        notch_row.addWidget(QLabel("Notch filter"))
        notch_row.addStretch()
        self._notch_combo = QComboBox()
        self._notch_combo.addItems(["50 Hz", "60 Hz", "Off"])
        self._notch_combo.setFixedWidth(90)
        self._notch_combo.currentTextChanged.connect(self._on_notch)
        notch_row.addWidget(self._notch_combo)
        vbox.addLayout(notch_row)

        return grp

    def _build_protocol_section(self) -> QGroupBox:
        grp  = QGroupBox("Protocol")
        form = QFormLayout(grp)
        form.setContentsMargins(12, 16, 12, 12)
        form.setVerticalSpacing(12)
        form.setHorizontalSpacing(12)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)

        self._epochs_spin = QSpinBox()
        self._epochs_spin.setRange(50, 300)
        self._epochs_spin.setValue(100)
        form.addRow("Target epochs:", self._epochs_spin)

        self._rate_spin = QDoubleSpinBox()
        self._rate_spin.setRange(0.5, 4.0)
        self._rate_spin.setSingleStep(0.5)
        self._rate_spin.setDecimals(1)
        self._rate_spin.setValue(2.0)
        form.addRow("Stim rate (Hz):", self._rate_spin)

        self._eye_combo = QComboBox()
        self._eye_combo.addItems(["N/A", "Left", "Right", "Both"])
        form.addRow("Eye tested:", self._eye_combo)

        self._subject_edit = QLineEdit()
        self._subject_edit.setPlaceholderText("Subject ID")
        form.addRow("Subject ID:", self._subject_edit)

        # Clinical note — only visible in DoC mode
        self._note_label = QLabel("Clinical note:")
        self._note_edit  = QTextEdit()
        self._note_edit.setFixedHeight(72)
        self._note_edit.setPlaceholderText("Optional free-text note…")
        self._note_label.setVisible(False)
        self._note_edit.setVisible(False)
        form.addRow(self._note_label, self._note_edit)

        self._protocol_group_box = grp
        return grp

    def _build_action_section(self) -> QWidget:
        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(0, 4, 0, 0)
        vbox.setSpacing(8)

        self._btn_start = QPushButton("▶  Start session")
        self._btn_start.setObjectName("btn_start")
        self._btn_start.clicked.connect(self._on_start)
        vbox.addWidget(self._btn_start)

        self._btn_stop = QPushButton("■  Stop")
        self._btn_stop.setObjectName("btn_stop")
        self._btn_stop.setVisible(False)
        self._btn_stop.clicked.connect(self._on_stop)
        vbox.addWidget(self._btn_stop)

        return container

    # ── Slots ──────────────────────────────────────────────────────────────

    def _on_paradigm_changed(self, idx: int) -> None:
        p = _PARADIGMS[idx]
        self._current_paradigm = p

        is_doc = p["doc"]
        self._doc_badge.setVisible(is_doc)
        self._note_label.setVisible(is_doc)
        self._note_edit.setVisible(is_doc)

        if is_doc:
            self._epochs_spin.setValue(50)
            self._rate_spin.setValue(1.0)
            self._eye_combo.setCurrentText("N/A")

        self.paradigm_changed.emit(p["key"])

    def _on_sensitivity(self, value: int) -> None:
        self._sens_label.setText(f"{value} µV")
        self.sensitivity_changed.emit(float(value))

    def _on_hp(self, value: int) -> None:
        hz = value / 10.0
        self._hp_label.setText(f"{hz:.1f} Hz")
        self.highpass_changed.emit(hz)

    def _on_lp(self, value: int) -> None:
        self._lp_label.setText(f"{value} Hz")
        self.lowpass_changed.emit(float(value))

    def _on_notch(self, text: str) -> None:
        if text == "Off":
            self.notch_changed.emit(None)
        elif text == "50 Hz":
            self.notch_changed.emit(50.0)
        elif text == "60 Hz":
            self.notch_changed.emit(60.0)

    def _on_show_all_toggled(self, state: int) -> None:
        key = "ALL" if state == Qt.CheckState.Checked.value else self._current_paradigm["key"]
        self.paradigm_changed.emit(key)

    def _on_start(self) -> None:
        self.session_start.emit()

    def _on_stop(self) -> None:
        self.session_stop.emit()


# ── Helpers ─────────────────────────────────────────────────────────────────

def _make_slider(minimum: int, maximum: int, value: int) -> QSlider:
    """Return a configured horizontal QSlider."""
    s = QSlider(Qt.Orientation.Horizontal)
    s.setMinimum(minimum)
    s.setMaximum(maximum)
    s.setValue(value)
    return s


def _slider_row(label_text: str, value_label: QLabel, slider: QSlider) -> QVBoxLayout:
    """
    Build a two-line control: [Label Name]  [current value]
                               [========= slider =========]
    """
    container = QVBoxLayout()
    container.setSpacing(5)

    header = QHBoxLayout()
    header.setSpacing(8)
    name_lbl = QLabel(label_text)
    value_label.setAlignment(
        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
    )
    value_label.setStyleSheet("color: #9a9891;")
    header.addWidget(name_lbl)
    header.addStretch()
    header.addWidget(value_label)

    container.addLayout(header)
    container.addWidget(slider)
    return container
