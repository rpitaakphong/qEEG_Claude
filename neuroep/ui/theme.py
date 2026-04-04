"""
ui/theme.py — PyQt6 application stylesheet and colour palette.

Usage
-----
    from neuroep.ui.theme import apply_dark_theme, apply_light_theme, PALETTE

    apply_dark_theme(app)   # call once after QApplication is created
"""

from __future__ import annotations

from PyQt6.QtWidgets import QApplication

# ── Colour tokens ──────────────────────────────────────────────────────────
PALETTE: dict[str, str] = {
    # Backgrounds
    "bg_primary":   "#0f1117",
    "bg_secondary": "#1a1d27",
    "bg_tertiary":  "#242736",
    # Borders / dividers
    "border":       "#2e3148",
    # Text
    "text_primary":   "#e8e6de",
    "text_secondary": "#9a9891",
    "text_disabled":  "#5a584f",
    # Accent colours
    "accent_purple": "#534AB7",   # VEP / primary actions
    "accent_teal":   "#1D9E75",   # AEP / success
    "accent_amber":  "#BA7517",   # P300 / warnings
    "accent_blue":   "#378ADD",   # Flash VEP / info
    "danger_red":    "#D85A30",   # artifact rejection / errors
    # Hover / pressed variants (slightly lighter)
    "hover_purple":  "#6559d4",
    "hover_teal":    "#23b887",
    "hover_amber":   "#d48920",
    "hover_blue":    "#4d9de8",
}

# ── Dark theme stylesheet ──────────────────────────────────────────────────
_DARK_QSS = """
/* ── Global ────────────────────────────────────────────────────────── */
QWidget {{
    background-color: {bg_primary};
    color: {text_primary};
    font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    font-size: 11pt;
}}

QMainWindow, QDialog {{
    background-color: {bg_primary};
}}

/* ── Panels / frames ────────────────────────────────────────────────── */
QFrame {{
    background-color: {bg_secondary};
    border: 1px solid {border};
    border-radius: 4px;
}}

QGroupBox {{
    background-color: {bg_secondary};
    border: 1px solid {border};
    border-radius: 4px;
    margin-top: 8px;
    padding-top: 6px;
    font-weight: bold;
    color: {text_secondary};
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 4px;
    color: {text_secondary};
}}

/* ── Buttons ────────────────────────────────────────────────────────── */
QPushButton {{
    background-color: {bg_tertiary};
    color: {text_primary};
    border: 1px solid {border};
    border-radius: 4px;
    padding: 6px 14px;
    font-size: 10pt;
}}

QPushButton:hover {{
    background-color: {accent_purple};
    border-color: {accent_purple};
}}

QPushButton:pressed {{
    background-color: {hover_purple};
}}

QPushButton:disabled {{
    color: {text_disabled};
    border-color: {border};
    background-color: {bg_secondary};
}}

QPushButton#btn_start {{
    background-color: {accent_purple};
    color: {text_primary};
    border: none;
    font-size: 12pt;
    font-weight: bold;
    padding: 10px;
}}

QPushButton#btn_start:hover {{
    background-color: {hover_purple};
}}

QPushButton#btn_stop {{
    background-color: {danger_red};
    color: {text_primary};
    border: none;
    font-size: 12pt;
    font-weight: bold;
    padding: 10px;
}}

QPushButton#btn_timing {{
    background-color: transparent;
    color: {accent_amber};
    border: 1px solid {accent_amber};
    font-size: 10pt;
    padding: 6px 14px;
}}

QPushButton#btn_timing:hover {{
    background-color: {accent_amber};
    color: {text_primary};
}}

/* ── Sliders ────────────────────────────────────────────────────────── */
QSlider::groove:horizontal {{
    height: 4px;
    background: {border};
    border-radius: 2px;
}}

QSlider::handle:horizontal {{
    background: {accent_purple};
    border: none;
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}}

QSlider::sub-page:horizontal {{
    background: {accent_purple};
    border-radius: 2px;
}}

/* ── ComboBox ────────────────────────────────────────────────────────── */
QComboBox {{
    background-color: {bg_tertiary};
    color: {text_primary};
    border: 1px solid {border};
    border-radius: 4px;
    padding: 4px 8px;
    selection-background-color: {accent_purple};
}}

QComboBox::drop-down {{
    border: none;
    width: 20px;
}}

QComboBox QAbstractItemView {{
    background-color: {bg_tertiary};
    color: {text_primary};
    border: 1px solid {border};
    selection-background-color: {accent_purple};
}}

/* ── SpinBox ─────────────────────────────────────────────────────────── */
QSpinBox, QDoubleSpinBox {{
    background-color: {bg_tertiary};
    color: {text_primary};
    border: 1px solid {border};
    border-radius: 4px;
    padding: 4px 8px;
    selection-background-color: {accent_purple};
}}

/* ── LineEdit / TextEdit ─────────────────────────────────────────────── */
QLineEdit, QTextEdit, QPlainTextEdit {{
    background-color: {bg_tertiary};
    color: {text_primary};
    border: 1px solid {border};
    border-radius: 4px;
    padding: 4px 8px;
    selection-background-color: {accent_purple};
}}

QLineEdit:focus, QTextEdit:focus {{
    border-color: {accent_purple};
}}

/* ── Labels ─────────────────────────────────────────────────────────── */
QLabel {{
    background: transparent;
    color: {text_primary};
    border: none;
}}

QLabel#label_secondary {{
    color: {text_secondary};
    font-size: 9pt;
}}

QLabel#doc_badge {{
    background-color: {danger_red};
    color: {text_primary};
    border-radius: 4px;
    padding: 2px 8px;
    font-weight: bold;
    font-size: 9pt;
}}

/* ── Tab widget ─────────────────────────────────────────────────────── */
QTabWidget::pane {{
    border: 1px solid {border};
    background-color: {bg_secondary};
}}

QTabBar::tab {{
    background-color: {bg_primary};
    color: {text_secondary};
    padding: 6px 16px;
    border: 1px solid {border};
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}}

QTabBar::tab:selected {{
    background-color: {bg_secondary};
    color: {text_primary};
    border-bottom-color: {bg_secondary};
}}

/* ── Scroll bars ─────────────────────────────────────────────────────── */
QScrollBar:vertical {{
    background: {bg_primary};
    width: 8px;
}}

QScrollBar::handle:vertical {{
    background: {border};
    border-radius: 4px;
    min-height: 20px;
}}

QScrollBar::handle:vertical:hover {{
    background: {accent_purple};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

/* ── Status bar ─────────────────────────────────────────────────────── */
QStatusBar {{
    background-color: {bg_secondary};
    color: {text_secondary};
    border-top: 1px solid {border};
    font-size: 9pt;
}}

/* ── Menu bar ────────────────────────────────────────────────────────── */
QMenuBar {{
    background-color: {bg_secondary};
    color: {text_primary};
    border-bottom: 1px solid {border};
}}

QMenuBar::item:selected {{
    background-color: {accent_purple};
}}

QMenu {{
    background-color: {bg_tertiary};
    color: {text_primary};
    border: 1px solid {border};
}}

QMenu::item:selected {{
    background-color: {accent_purple};
}}

/* ── Progress bar ────────────────────────────────────────────────────── */
QProgressBar {{
    background-color: {bg_tertiary};
    border: 1px solid {border};
    border-radius: 4px;
    text-align: center;
    color: {text_primary};
}}

QProgressBar::chunk {{
    background-color: {accent_purple};
    border-radius: 4px;
}}

/* ── Table widget ────────────────────────────────────────────────────── */
QTableWidget {{
    background-color: {bg_secondary};
    color: {text_primary};
    gridline-color: {border};
    border: none;
}}

QHeaderView::section {{
    background-color: {bg_tertiary};
    color: {text_secondary};
    border: 1px solid {border};
    padding: 4px 8px;
    font-weight: bold;
}}

QTableWidget::item:selected {{
    background-color: {accent_purple};
}}

/* ── Splitter ────────────────────────────────────────────────────────── */
QSplitter::handle {{
    background-color: {border};
}}
"""

# ── Light theme stylesheet (minimal — just inverts key colours) ────────────
_LIGHT_QSS = """
QWidget {{
    background-color: #f5f5f0;
    color: #1a1a1a;
    font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    font-size: 11pt;
}}

QMainWindow, QDialog {{
    background-color: #f5f5f0;
}}

QFrame {{
    background-color: #ffffff;
    border: 1px solid #d0d0cc;
    border-radius: 4px;
}}

QGroupBox {{
    background-color: #ffffff;
    border: 1px solid #d0d0cc;
    border-radius: 4px;
    margin-top: 8px;
    padding-top: 6px;
    font-weight: bold;
    color: #555550;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 4px;
    color: #555550;
}}

QPushButton {{
    background-color: #e8e8e4;
    color: #1a1a1a;
    border: 1px solid #c0c0bc;
    border-radius: 4px;
    padding: 6px 14px;
}}

QPushButton:hover {{
    background-color: #534AB7;
    color: #ffffff;
    border-color: #534AB7;
}}

QPushButton#btn_start {{
    background-color: #534AB7;
    color: #ffffff;
    border: none;
    font-size: 12pt;
    font-weight: bold;
    padding: 10px;
}}

QPushButton#btn_stop {{
    background-color: #D85A30;
    color: #ffffff;
    border: none;
    font-size: 12pt;
    font-weight: bold;
    padding: 10px;
}}

QSlider::groove:horizontal {{
    height: 4px;
    background: #c0c0bc;
    border-radius: 2px;
}}

QSlider::handle:horizontal {{
    background: #534AB7;
    border: none;
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}}

QSlider::sub-page:horizontal {{
    background: #534AB7;
    border-radius: 2px;
}}

QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit, QTextEdit, QPlainTextEdit {{
    background-color: #ffffff;
    color: #1a1a1a;
    border: 1px solid #c0c0bc;
    border-radius: 4px;
    padding: 4px 8px;
}}

QStatusBar {{
    background-color: #e8e8e4;
    color: #555550;
    border-top: 1px solid #c0c0bc;
}}

QMenuBar {{
    background-color: #e8e8e4;
    color: #1a1a1a;
}}

QTabBar::tab {{
    background-color: #e8e8e4;
    color: #555550;
    padding: 6px 16px;
    border: 1px solid #c0c0bc;
    border-bottom: none;
}}

QTabBar::tab:selected {{
    background-color: #ffffff;
    color: #1a1a1a;
}}

QScrollBar:vertical {{
    background: #f0f0ec;
    width: 8px;
}}

QScrollBar::handle:vertical {{
    background: #c0c0bc;
    border-radius: 4px;
    min-height: 20px;
}}
"""


def _format_qss(template: str, palette: dict[str, str]) -> str:
    """Substitute palette tokens into a QSS template."""
    return template.format(**palette)


def apply_dark_theme(app: QApplication) -> None:
    """
    Apply the dark stylesheet to *app*.

    Parameters
    ----------
    app : QApplication
        The running Qt application instance.
    """
    app.setStyleSheet(_format_qss(_DARK_QSS, PALETTE))


def apply_light_theme(app: QApplication) -> None:
    """
    Apply the light stylesheet to *app*.

    Parameters
    ----------
    app : QApplication
        The running Qt application instance.
    """
    app.setStyleSheet(_format_qss(_LIGHT_QSS, PALETTE))
