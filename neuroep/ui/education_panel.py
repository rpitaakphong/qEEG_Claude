"""
ui/education_panel.py — Education panel: electrode map, paradigm description,
and reference waveform widget.

Three stacked sub-widgets update together whenever the paradigm changes:
  • ElectrodeMapWidget    — QPainter scalp map showing electrode roles
  • ParadigmDescriptionCard — coloured card with protocol description + DoC badge
  • ReferenceWaveformWidget  — idealised EP waveform sketch (QPainter)

Connect MainWindow:
    sidebar.paradigm_changed.connect(education_panel.on_paradigm_changed)
"""

from __future__ import annotations

import math
from typing import Optional

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import (
    QColor,
    QFont,
    QPainter,
    QPainterPath,
    QPen,
    QBrush,
)
from PyQt6.QtWidgets import (
    QFrame,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtWidgets import QToolTip

from neuroep import config

# Channel number lookup: electrode name → 1-based channel number.
# Only covers the 16 actual recording channels; electrodes absent from
# CHANNEL_NAMES (Cz, Pz, T5, T6) will not have a channel number label.
_CH_NUMBER: dict[str, int] = {
    name: i + 1 for i, name in enumerate(config.CHANNEL_NAMES)
}

# ── Key normalisation ──────────────────────────────────────────────────────

_KEY_MAP: dict[str, str] = {
    "vep_pattern":  "VEP_PATTERN",
    "vep_flash":    "VEP_FLASH",
    "aep":          "AEP",
    "p300_passive": "P300",
    # Pass-through for already-canonical keys
    "VEP_PATTERN":  "VEP_PATTERN",
    "VEP_FLASH":    "VEP_FLASH",
    "AEP":          "AEP",
    "P300":         "P300",
    "ALL":          "VEP_PATTERN",   # default to first paradigm when "show all"
}


def _norm(key: str) -> str:
    return _KEY_MAP.get(key, "VEP_PATTERN")


# ── B1 data ────────────────────────────────────────────────────────────────

ELECTRODE_POSITIONS: dict[str, tuple[float, float]] = {
    "Fp1": (-0.30, -0.83), "Fp2": ( 0.30, -0.83),
    "F7":  (-0.71, -0.46), "F3":  (-0.38, -0.50),
    "Fz":  ( 0.00, -0.46), "F4":  ( 0.38, -0.50),
    "F8":  ( 0.71, -0.46),
    "T3":  (-0.95,  0.00), "C3":  (-0.50,  0.00),
    "Cz":  ( 0.00,  0.00), "C4":  ( 0.50,  0.00),
    "T4":  ( 0.95,  0.00),
    "T5":  (-0.71,  0.46), "P3":  (-0.38,  0.50),
    "Pz":  ( 0.00,  0.46), "P4":  ( 0.38,  0.50),
    "T6":  ( 0.71,  0.46),
    "O1":  (-0.30,  0.83), "Oz":  ( 0.00,  0.90),
    "O2":  ( 0.30,  0.83),
}

PARADIGM_CHANNELS: dict[str, dict] = {
    "VEP_PATTERN": {
        "color":       "#534AB7",
        "color_light": "#EEEDFE",
        "essential":   ["Oz", "O1", "O2"],
        "useful":      ["Pz"],
        "artifact":    ["Fz", "Fp1", "Fp2"],
    },
    "VEP_FLASH": {
        "color":       "#378ADD",
        "color_light": "#E6F1FB",
        "essential":   ["Oz", "O1", "O2"],
        "useful":      ["Pz"],
        "artifact":    ["Fz", "Fp1", "Fp2"],
    },
    "AEP": {
        "color":       "#1D9E75",
        "color_light": "#E1F5EE",
        "essential":   ["Cz", "Fz"],
        "useful":      ["C3", "C4"],
        "artifact":    ["Fp1", "Fp2", "Oz"],
    },
    "P300": {
        "color":       "#BA7517",
        "color_light": "#FAEEDA",
        "essential":   ["Pz", "Cz", "Fz"],
        "useful":      ["P3", "P4"],
        "artifact":    ["Fp1", "Fp2"],
    },
}


# ── B2 data ────────────────────────────────────────────────────────────────

PARADIGM_INFO: dict[str, dict] = {
    "VEP_PATTERN": {
        "title": "Pattern reversal VEP",
        "doc_mode": False,
        "text": (
            "Checkerboard reversal at 2 Hz on the stimulus screen.\n"
            "Subject must fixate the central cross — fixation is mandatory.\n"
            "Each reversal is one epoch. Average 100 epochs minimum.\n\n"
            "Key component: P100 at Oz (95–115 ms normal range).\n"
            "Compare O1 vs O2: >50% amplitude asymmetry = significant.\n\n"
            "Not suitable for DoC patients who cannot fixate."
        ),
    },
    "VEP_FLASH": {
        "title": "Flash VEP",
        "doc_mode": True,
        "text": (
            "Full-field white flash at 1 Hz. No fixation required.\n"
            "Eyes must be open — document eye state before starting.\n"
            "Target 100–150 epochs (noisier than pattern VEP).\n\n"
            "Key component: P2 at Oz (~120 ms).\n"
            "Confirms visual cortex receives light — not a test of awareness.\n\n"
            "Primary paradigm for DoC / vegetative state patients."
        ),
    },
    "AEP": {
        "title": "Auditory evoked potential",
        "doc_mode": False,
        "text": (
            "Broadband click at 70–80 dB SPL via insert earphones.\n"
            "Rate: 1–2 clicks/sec. Eyes closed to prevent VEP bleed.\n"
            "Target 100 epochs.\n\n"
            "Key components: N1 (~100 ms) and P2 (~180 ms) at Cz/Fz.\n"
            "Right ear → larger response at left C3 (and vice versa).\n\n"
            "Passive — works in DoC patients. Confirms auditory cortex\n"
            "receives and processes sound."
        ),
    },
    "P300": {
        "title": "P300 passive oddball",
        "doc_mode": True,
        "text": (
            "80% standard tone (1000 Hz) / 20% oddball (2000 Hz).\n"
            "No button press. Fully passive — patient just lies still.\n"
            "Target: 40–60 clean oddball epochs (~250 total stimuli).\n\n"
            "Analyse the DIFFERENCE wave: oddball minus standard.\n"
            "Key component: P300 at Pz (~300 ms) in difference wave.\n\n"
            "P300 without behavioural response = evidence of covert\n"
            "cognition. Basis of Owen/Monti DoC awareness paradigms."
        ),
    },
}


# ── B3 data ────────────────────────────────────────────────────────────────

# Wave definitions: list of (time_ms, amplitude_uv) key-frame tuples.
# Negative amplitude = upward in "negative up" clinical convention.

VEP_WAVE = [
    (-100,  0.0), (0,   0.0),
    (50,    1.0),
    (75,    3.5),
    (100, -12.0),   # P100 — highlight
    (145,   5.0),
    (200,   1.0),
    (400,   0.0),
]

AEP_WAVE = [
    (-100,  0.0), (0,  0.0),
    (50,    0.5),
    (100,  -8.0),   # N1 — highlight
    (150,  -3.0),
    (180,   6.0),
    (250,  -2.0),
    (400,   0.0),
]

P300_STANDARD = [
    (-100,  0.0), (0,  0.0),
    (100,  -6.0),
    (180,   4.0),
    (400,   0.0),
]

P300_ODDBALL = [
    (-100,  0.0), (0,  0.0),
    (100,  -6.5),
    (180,   4.5),
    (300, -14.0),   # P300 — highlight
    (450,   0.0),
]

# Annotation for each paradigm: (time_ms, amplitude_uv, label_text)
WAVE_ANNOTATIONS: dict[str, tuple[float, float, str]] = {
    "VEP_PATTERN": (100, -12.0, "P100  95–115 ms"),
    "VEP_FLASH":   (100, -12.0, "P100  95–115 ms"),
    "AEP":         (100,  -8.0, "N1  90–110 ms"),
    "P300":        (300, -14.0, "P300  280–350 ms"),
}


# ══════════════════════════════════════════════════════════════════════════
# B1 — ElectrodeMapWidget
# ══════════════════════════════════════════════════════════════════════════

class ElectrodeMapWidget(QWidget):
    """QPainter-based scalp electrode map, colour-coded by paradigm role."""

    _LEGEND_H = 28   # pixels reserved at the bottom for the legend
    _NOSE_H   = 14   # pixels reserved at the top for the nose protrusion

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._paradigm_key = "VEP_PATTERN"
        self.setMinimumSize(260, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

    def set_paradigm(self, key: str) -> None:
        self._paradigm_key = _norm(key)
        self.update()

    # ── Helpers ───────────────────────────────────────────────────────────

    def _pixel_pos(self, norm_x: float, norm_y: float) -> QPointF:
        cx = self.rect().center().x()
        usable_h = self.rect().height() - self._LEGEND_H - self._NOSE_H
        cy = self._NOSE_H + usable_h / 2
        r  = min(self.rect().width(), usable_h) * 0.42
        return QPointF(cx + norm_x * r, cy + norm_y * r)

    def _electrode_role(self, name: str) -> str:
        info = PARADIGM_CHANNELS.get(self._paradigm_key, {})
        for role in ("essential", "useful", "artifact"):
            if name in info.get(role, []):
                return role
        return "inactive"

    def _electrode_at(self, pos: QPointF) -> Optional[str]:
        """Return the electrode name under *pos*, or None."""
        for name, (nx, ny) in ELECTRODE_POSITIONS.items():
            px = self._pixel_pos(nx, ny)
            dx = pos.x() - px.x()
            dy = pos.y() - px.y()
            if math.hypot(dx, dy) <= 14:
                return name
        return None

    # ── Events ────────────────────────────────────────────────────────────

    def mouseMoveEvent(self, event) -> None:
        name = self._electrode_at(QPointF(event.position()))
        if name:
            role = self._electrode_role(name)
            role_desc = {
                "essential": "Essential — primary signal channel",
                "useful":    "Useful — secondary / reference channel",
                "artifact":  "Artifact monitor — watch for noise",
                "inactive":  "Inactive for this paradigm",
            }.get(role, role)
            ch_num = _CH_NUMBER.get(name)
            ch_str = f"  ·  ch{ch_num}" if ch_num is not None else "  ·  (not recorded)"
            QToolTip.showText(
                event.globalPosition().toPoint(),
                f"{name}{ch_str}  —  {role_desc}",
                self,
            )
        else:
            QToolTip.hideText()
        super().mouseMoveEvent(event)

    # ── Paint ─────────────────────────────────────────────────────────────

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        info = PARADIGM_CHANNELS.get(self._paradigm_key, PARADIGM_CHANNELS["VEP_PATTERN"])
        paradigm_color  = QColor(info["color"])
        paradigm_light  = QColor(info["color_light"])
        grey_border     = QColor("#5F5E5A")
        grey_fill       = QColor("#2e3148")
        inactive_color  = QColor("#3a3a4a")
        inactive_label  = QColor("#5F5E5A")

        w = self.width()
        h = self.height()

        # Background
        painter.fillRect(self.rect(), QColor("#0e1117"))

        # Head outline — centred within the usable area (excluding legend + nose margins)
        cx = w / 2
        usable_h = h - self._LEGEND_H - self._NOSE_H
        cy = self._NOSE_H + usable_h / 2
        r  = min(w, usable_h) * 0.42
        head_pen = QPen(QColor("#3a3c4e"), 1.5)
        painter.setPen(head_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QPointF(cx, cy), r, r)

        # Nose
        nose_w = r * 0.10
        painter.drawLine(
            QPointF(cx - nose_w, cy - r + 2),
            QPointF(cx, cy - r - nose_w * 1.5),
        )
        painter.drawLine(
            QPointF(cx, cy - r - nose_w * 1.5),
            QPointF(cx + nose_w, cy - r + 2),
        )

        # Electrodes
        label_font = QFont()
        label_font.setPointSize(8)
        label_font.setBold(False)
        ch_font = QFont()
        ch_font.setPointSize(7)
        painter.setFont(label_font)

        for name, (nx, ny) in ELECTRODE_POSITIONS.items():
            px = self._pixel_pos(nx, ny)
            role = self._electrode_role(name)
            ch_num = _CH_NUMBER.get(name)   # None for Cz, Pz, T5, T6

            if role == "essential":
                radius = 13
                painter.setPen(QPen(paradigm_color, 1.5))
                painter.setBrush(QBrush(paradigm_color))
                painter.drawEllipse(px, radius, radius)
                painter.setPen(QPen(QColor("white")))
                painter.setFont(label_font)
                painter.drawText(
                    QRectF(px.x() - radius, px.y() - radius, radius * 2, radius * 2),
                    Qt.AlignmentFlag.AlignCenter, name,
                )
                # Channel number below circle
                if ch_num is not None:
                    painter.setFont(ch_font)
                    painter.setPen(QPen(QColor("#aaa9a0")))
                    painter.drawText(
                        QRectF(px.x() - 16, px.y() + radius + 1, 32, 11),
                        Qt.AlignmentFlag.AlignCenter, f"ch{ch_num}",
                    )
                    painter.setFont(label_font)

            elif role == "useful":
                radius = 11
                painter.setPen(QPen(paradigm_color, 1.5))
                painter.setBrush(QBrush(paradigm_light))
                painter.drawEllipse(px, radius, radius)
                painter.setPen(QPen(paradigm_color))
                painter.setFont(label_font)
                painter.drawText(
                    QRectF(px.x() - radius, px.y() - radius, radius * 2, radius * 2),
                    Qt.AlignmentFlag.AlignCenter, name,
                )
                if ch_num is not None:
                    painter.setFont(ch_font)
                    painter.setPen(QPen(QColor("#aaa9a0")))
                    painter.drawText(
                        QRectF(px.x() - 16, px.y() + radius + 1, 32, 11),
                        Qt.AlignmentFlag.AlignCenter, f"ch{ch_num}",
                    )
                    painter.setFont(label_font)

            elif role == "artifact":
                radius = 9
                dash_pen = QPen(grey_border, 1.2, Qt.PenStyle.DashLine)
                painter.setPen(dash_pen)
                painter.setBrush(QBrush(grey_fill))
                painter.drawEllipse(px, radius, radius)
                painter.setPen(QPen(grey_border))
                painter.setFont(label_font)
                painter.drawText(
                    QRectF(px.x() - radius, px.y() - radius, radius * 2, radius * 2),
                    Qt.AlignmentFlag.AlignCenter, name,
                )
                if ch_num is not None:
                    painter.setFont(ch_font)
                    painter.setPen(QPen(QColor("#5F5E5A")))
                    painter.drawText(
                        QRectF(px.x() - 16, px.y() + radius + 1, 32, 11),
                        Qt.AlignmentFlag.AlignCenter, f"ch{ch_num}",
                    )
                    painter.setFont(label_font)

            else:  # inactive
                radius = 6
                painter.setPen(QPen(inactive_color, 1))
                painter.setBrush(QBrush(inactive_color))
                painter.drawEllipse(px, radius, radius)
                painter.setFont(ch_font)
                painter.setPen(QPen(inactive_label))
                # Name line
                painter.drawText(
                    QRectF(px.x() - 14, px.y() + radius + 1, 28, 11),
                    Qt.AlignmentFlag.AlignCenter, name,
                )
                # Channel number line (if this is a recording channel)
                if ch_num is not None:
                    painter.setPen(QPen(QColor("#3a3a50")))
                    painter.drawText(
                        QRectF(px.x() - 14, px.y() + radius + 12, 28, 11),
                        Qt.AlignmentFlag.AlignCenter, f"ch{ch_num}",
                    )
                painter.setFont(label_font)

        # Legend sits in the reserved bottom margin
        legend_y = h - 10
        legend_font = QFont()
        legend_font.setPointSize(9)
        painter.setFont(legend_font)

        items = [
            (paradigm_color, "● Essential"),
            (paradigm_color, "◉ Useful"),
            (grey_border,    "○ Artifact"),
            (inactive_label, "· Inactive"),
        ]
        x = 8
        for color, text in items:
            painter.setPen(QPen(color))
            painter.drawText(int(x), int(legend_y), text)
            fm = painter.fontMetrics()
            x += fm.horizontalAdvance(text) + 10

        painter.end()


# ══════════════════════════════════════════════════════════════════════════
# B2 — ParadigmDescriptionCard
# ══════════════════════════════════════════════════════════════════════════

class ParadigmDescriptionCard(QFrame):
    """Coloured-border card showing paradigm title, optional DoC badge, and description."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._build_ui()
        self.set_paradigm("VEP_PATTERN")

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        self._title_label = QLabel()
        title_font = QFont()
        title_font.setPointSize(11)
        title_font.setBold(True)
        self._title_label.setFont(title_font)
        layout.addWidget(self._title_label)

        self._doc_badge = QLabel("DoC mode")
        self._doc_badge.setStyleSheet(
            "background: #3a1f1f; color: #E07070; border-radius: 4px;"
            "padding: 2px 10px; font-size: 10px; font-weight: 600;"
        )
        self._doc_badge.setFixedHeight(22)
        self._doc_badge.setVisible(False)
        layout.addWidget(self._doc_badge)

        # Description text (read-only)
        self._desc = QTextEdit()
        self._desc.setReadOnly(True)
        self._desc.setFrameShape(QFrame.Shape.NoFrame)
        self._desc.setFont(QFont("Segoe UI", 9))
        self._desc.setStyleSheet("background: transparent; color: #c8c6be;")
        self._desc.setMinimumHeight(130)
        layout.addWidget(self._desc)

    def set_paradigm(self, key: str) -> None:
        key = _norm(key)
        info  = PARADIGM_INFO.get(key, PARADIGM_INFO["VEP_PATTERN"])
        pinfo = PARADIGM_CHANNELS.get(key, PARADIGM_CHANNELS["VEP_PATTERN"])
        color = pinfo["color"]

        self.setStyleSheet(f"""
            QFrame {{
                border-left: 3px solid {color};
                background: #1a1d27;
                border-radius: 4px;
            }}
        """)

        self._title_label.setText(info["title"])
        self._title_label.setStyleSheet(f"color: {color};")
        self._doc_badge.setVisible(info["doc_mode"])
        self._desc.setPlainText(info["text"])


# ══════════════════════════════════════════════════════════════════════════
# B3 — ReferenceWaveformWidget
# ══════════════════════════════════════════════════════════════════════════

_MARGIN_L  = 42
_MARGIN_R  = 12
_MARGIN_T  = 18
_MARGIN_B  = 28
_SCALE_UV  = 15.0   # µV half-range for Y axis


class ReferenceWaveformWidget(QWidget):
    """QPainter-based idealised EP waveform sketch with component annotations."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._paradigm_key = "VEP_PATTERN"
        self.setMinimumSize(260, 160)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_paradigm(self, key: str) -> None:
        self._paradigm_key = _norm(key)
        self.update()

    # ── Coordinate helpers ────────────────────────────────────────────────

    def _coords(self):
        w = self.width()
        h = self.height()
        pl = _MARGIN_L
        pr = w - _MARGIN_R
        pt = _MARGIN_T
        pb = h - _MARGIN_B
        pw = pr - pl
        ph = pb - pt
        pcy = (pt + pb) / 2
        return pl, pr, pt, pb, pw, ph, pcy

    def _t_to_x(self, t_ms: float, pl: float, pw: float) -> float:
        return pl + (t_ms + 100) / 600.0 * pw

    def _amp_to_y(self, amp_uv: float, pcy: float, ph: float) -> float:
        return pcy - (amp_uv / _SCALE_UV) * (ph / 2.0)

    # ── Drawing helpers ───────────────────────────────────────────────────

    def _draw_wave(
        self,
        painter: QPainter,
        points: list,
        color: QColor,
        width: float,
        dash: bool,
        pl: float, pw: float, pcy: float, ph: float,
    ) -> None:
        if len(points) < 2:
            return

        px_pts = [
            QPointF(self._t_to_x(t, pl, pw), self._amp_to_y(a, pcy, ph))
            for t, a in points
        ]

        path = QPainterPath()
        path.moveTo(px_pts[0])
        n = len(px_pts)
        for i in range(n - 1):
            p0 = px_pts[max(0, i - 1)]
            p1 = px_pts[i]
            p2 = px_pts[i + 1]
            p3 = px_pts[min(n - 1, i + 2)]
            c1 = QPointF(
                p1.x() + (p2.x() - p0.x()) / 6.0,
                p1.y() + (p2.y() - p0.y()) / 6.0,
            )
            c2 = QPointF(
                p2.x() - (p3.x() - p1.x()) / 6.0,
                p2.y() - (p3.y() - p1.y()) / 6.0,
            )
            path.cubicTo(c1, c2, p2)

        pen = QPen(color, width)
        if dash:
            pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)

    # ── Paint ─────────────────────────────────────────────────────────────

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        pl, pr, pt, pb, pw, ph, pcy = self._coords()
        key   = self._paradigm_key
        pinfo = PARADIGM_CHANNELS.get(key, PARADIGM_CHANNELS["VEP_PATTERN"])
        wave_color = QColor(pinfo["color"])

        # Background
        painter.fillRect(self.rect(), QColor("#0e1117"))

        # Plot area background
        painter.fillRect(
            int(pl), int(pt), int(pw), int(ph),
            QColor("#0e1117"),
        )

        # Grid lines
        grid_pen = QPen(QColor("#1e2130"), 1)
        grid_pen.setWidthF(0.5)
        painter.setPen(grid_pen)
        for t in (-50, 50, 100, 150, 200, 250, 300, 350, 400, 450):
            x = self._t_to_x(t, pl, pw)
            painter.drawLine(QPointF(x, pt), QPointF(x, pb))
        for frac in (-0.5, 0.5):
            y = pcy + frac * (ph / 2.0)
            painter.drawLine(QPointF(pl, y), QPointF(pr, y))

        # Zero lines (t=0 vertical, amp=0 horizontal)
        zero_pen = QPen(QColor("#2e3148"), 1, Qt.PenStyle.DashLine)
        painter.setPen(zero_pen)
        x0 = self._t_to_x(0, pl, pw)
        painter.drawLine(QPointF(x0, pt), QPointF(x0, pb))
        painter.drawLine(QPointF(pl, pcy), QPointF(pr, pcy))

        # ── Waveforms ─────────────────────────────────────────────────────
        if key in ("VEP_PATTERN", "VEP_FLASH"):
            self._draw_wave(painter, VEP_WAVE, wave_color, 2.0, False,
                            pl, pw, pcy, ph)

        elif key == "AEP":
            self._draw_wave(painter, AEP_WAVE, wave_color, 2.0, False,
                            pl, pw, pcy, ph)

        elif key == "P300":
            # Standard (grey dashed) first, then oddball on top
            self._draw_wave(painter, P300_STANDARD, QColor("#5F5E5A"), 1.5, True,
                            pl, pw, pcy, ph)
            self._draw_wave(painter, P300_ODDBALL, wave_color, 2.0, False,
                            pl, pw, pcy, ph)
            # Inner legend
            leg_font = QFont()
            leg_font.setPointSize(8)
            painter.setFont(leg_font)
            painter.setPen(QPen(QColor("#5F5E5A")))
            painter.drawText(int(pl + 4), int(pt + 12), "— Standard")
            painter.setPen(QPen(wave_color))
            painter.drawText(int(pl + 4), int(pt + 24), "— Oddball")

        # ── Component annotation ──────────────────────────────────────────
        ann = WAVE_ANNOTATIONS.get(key)
        if ann:
            t_ann, a_ann, label_text = ann
            ann_x = self._t_to_x(t_ann, pl, pw)
            ann_y = self._amp_to_y(a_ann, pcy, ph)
            ann_color  = QColor("#BA7517")
            ann_text_c = QColor("#EF9F27")

            # Vertical amber dashed line from peak to x-axis
            ann_pen = QPen(ann_color, 1.0, Qt.PenStyle.DashLine)
            painter.setPen(ann_pen)
            painter.drawLine(QPointF(ann_x, ann_y), QPointF(ann_x, pcy))

            # Small double-headed amplitude arrow
            arrow_x = ann_x + 6
            painter.drawLine(QPointF(arrow_x, ann_y), QPointF(arrow_x, pcy))
            ah = 5
            for tip_y in (ann_y, pcy):
                direction = 1 if tip_y == ann_y else -1
                painter.drawLine(
                    QPointF(arrow_x - 3, tip_y + direction * ah),
                    QPointF(arrow_x,     tip_y),
                )
                painter.drawLine(
                    QPointF(arrow_x + 3, tip_y + direction * ah),
                    QPointF(arrow_x,     tip_y),
                )

            # Text label
            ann_font = QFont()
            ann_font.setPointSize(9)
            painter.setFont(ann_font)
            painter.setPen(QPen(ann_text_c))
            label_x = ann_x - painter.fontMetrics().horizontalAdvance(label_text) - 6
            if label_x < pl:
                label_x = ann_x + 10
            painter.drawText(QPointF(label_x, ann_y - 4), label_text)

        # ── Axis labels ───────────────────────────────────────────────────
        ax_font = QFont()
        ax_font.setPointSize(8)
        painter.setFont(ax_font)
        painter.setPen(QPen(QColor("#5F5E5A")))
        fm = painter.fontMetrics()

        for t_lbl, txt in ((-100, "−100"), (0, "0"), (200, "+200"), (400, "+400")):
            lx = self._t_to_x(t_lbl, pl, pw)
            painter.drawText(
                QPointF(lx - fm.horizontalAdvance(txt) / 2, pb + 14), txt
            )

        # Y-axis label
        painter.save()
        painter.translate(pl - 28, pcy)
        painter.rotate(-90)
        painter.drawText(
            QRectF(-40, -10, 80, 20),
            Qt.AlignmentFlag.AlignCenter,
            "µV (neg ↑)",
        )
        painter.restore()

        painter.end()


# ══════════════════════════════════════════════════════════════════════════
# EducationPanel — assembles the three sub-widgets
# ══════════════════════════════════════════════════════════════════════════

class EducationPanel(QWidget):
    """
    280px-wide education panel housing:
      • ElectrodeMapWidget
      • ParadigmDescriptionCard
      • ReferenceWaveformWidget

    Connect the sidebar's ``paradigm_changed`` signal to
    ``on_paradigm_changed(key)``.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(260)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Scrollable inner area so the panel works at any height
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")

        inner = QWidget()
        inner.setStyleSheet("background: transparent;")
        vbox  = QVBoxLayout(inner)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(12)

        self.electrode_map    = ElectrodeMapWidget()
        self.description_card = ParadigmDescriptionCard()
        self.waveform_sketch  = ReferenceWaveformWidget()

        vbox.addWidget(self.electrode_map)
        vbox.addWidget(self.description_card)
        vbox.addWidget(self.waveform_sketch)
        vbox.addStretch()

        scroll.setWidget(inner)
        root.addWidget(scroll)

    # ── Public slot ───────────────────────────────────────────────────────

    def on_paradigm_changed(self, key: str) -> None:
        """Update all three sub-widgets when the paradigm selector changes."""
        self.electrode_map.set_paradigm(key)
        self.description_card.set_paradigm(key)
        self.waveform_sketch.set_paradigm(key)
