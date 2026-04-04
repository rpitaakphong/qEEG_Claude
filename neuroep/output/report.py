"""
output/report.py — Per-session clinical summary report (PDF + plain text).

Generates a structured clinical document containing:
  • Session metadata (subject, date, paradigm, operator)
  • EP waveform figure (embedded PNG rendered by matplotlib)
  • Component latency / amplitude table (P100, N1, P2, N2, P300)
  • Epoch statistics (accepted, rejected, rejection rate)
  • SNR summary
  • Timing validation results (if available)
  • Clinical notes (free-text, DoC mode)
  • Interpretation guidelines (normative latency ranges)

Output formats
--------------
  • PDF — via ReportLab (professional, printable)
  • TXT — plain-text fallback if ReportLab is not installed

Public API
----------
ReportData   : dataclass holding all inputs needed to generate a report.
ReportWriter : generates PDF and/or TXT from a ReportData instance.
"""

from __future__ import annotations

import datetime
import io
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from neuroep import config
from neuroep.processing.components import ComponentResult
from neuroep.processing.epochs import EPOCH_TIME_MS

logger = logging.getLogger(__name__)

# ── Normative latency ranges (ms) — healthy adults ─────────────────────────
_NORMATIVE: dict[str, tuple[float, float]] = {
    "P100": (90.0,  115.0),
    "N1":   (80.0,  130.0),
    "P2":   (150.0, 230.0),
    "N2":   (180.0, 260.0),
    "P300": (280.0, 450.0),
}


@dataclass
class ReportData:
    """
    All data needed to generate one clinical session report.

    Attributes
    ----------
    session_id : str
    subject_id : str
    paradigm   : str
    operator   : str
    date       : datetime.datetime
    avg        : np.ndarray, shape (N_CHANNELS, epoch_len)
        Grand-average EP waveform.
    display_channel : int
        Channel index used for the main waveform plot.
    channel_name    : str
        Human-readable channel label (e.g. ``"Oz"``).
    components      : list[ComponentResult]
    n_accepted      : int
    n_rejected      : int
    snr_db          : float
        Mean SNR across EP channels at end of session.
    eye_tested      : str
        "Left", "Right", "Both", or "N/A".
    clinical_note   : str
        Free-text clinical observation (DoC mode).
    timing_mean_ms  : float or None
        Mode A timing test result.
    timing_sd_ms    : float or None
    """

    session_id:      str
    subject_id:      str
    paradigm:        str
    operator:        str                     = "NeuroEP Studio"
    date:            datetime.datetime       = field(
        default_factory=datetime.datetime.now
    )
    avg:             Optional[np.ndarray]    = None
    display_channel: int                     = 0
    channel_name:    str                     = "Oz"
    components:      list[ComponentResult]   = field(default_factory=list)
    n_accepted:      int                     = 0
    n_rejected:      int                     = 0
    snr_db:          float                   = float("nan")
    eye_tested:      str                     = "N/A"
    clinical_note:   str                     = ""
    timing_mean_ms:  Optional[float]         = None
    timing_sd_ms:    Optional[float]         = None

    @property
    def rejection_rate(self) -> float:
        total = self.n_accepted + self.n_rejected
        return self.n_rejected / total if total > 0 else 0.0

    @property
    def is_doc_mode(self) -> bool:
        return self.paradigm in ("vep_flash", "p300_passive")


class ReportWriter:
    """
    Generates a clinical PDF (and plain-text fallback) from a ``ReportData``.

    Parameters
    ----------
    data : ReportData
    """

    def __init__(self, data: ReportData) -> None:
        self._d = data

    # ── Public interface ───────────────────────────────────────────────────

    def save_pdf(self, path: Path) -> None:
        """
        Write a PDF clinical report to *path*.

        Falls back to plain-text if ReportLab is not installed.
        """
        try:
            from reportlab.lib import colors
            self._write_pdf(path)
        except ImportError:
            logger.warning("ReportLab not installed — writing plain text instead.")
            self._write_txt(path.with_suffix(".txt"))

    def save_txt(self, path: Path) -> None:
        """Write a plain-text clinical summary to *path*."""
        self._write_txt(path)

    # ── PDF generation ─────────────────────────────────────────────────────

    def _write_pdf(self, path: Path) -> None:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            HRFlowable,
            Image,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )

        path.parent.mkdir(parents=True, exist_ok=True)
        doc = SimpleDocTemplate(
            str(path),
            pagesize     = A4,
            leftMargin   = 2 * cm,
            rightMargin  = 2 * cm,
            topMargin    = 2 * cm,
            bottomMargin = 2 * cm,
        )

        styles = getSampleStyleSheet()
        style_h1 = ParagraphStyle(
            "H1", parent=styles["Heading1"],
            fontSize=16, textColor=colors.HexColor("#1a1a8c"),
            spaceAfter=4,
        )
        style_h2 = ParagraphStyle(
            "H2", parent=styles["Heading2"],
            fontSize=11, textColor=colors.HexColor("#333333"),
            spaceBefore=10, spaceAfter=4,
        )
        style_body = ParagraphStyle(
            "Body", parent=styles["Normal"],
            fontSize=9, leading=13,
        )
        style_note = ParagraphStyle(
            "Note", parent=styles["Normal"],
            fontSize=9, leading=13,
            backColor=colors.HexColor("#f5f5f0"),
            borderPad=6,
        )

        d = self._d
        story = []

        # ── Header ─────────────────────────────────────────────────────────
        story.append(Paragraph("NeuroEP Studio — Clinical Report", style_h1))
        story.append(HRFlowable(width="100%", thickness=1,
                                color=colors.HexColor("#1a1a8c")))
        story.append(Spacer(1, 0.3 * cm))

        # Metadata table
        meta_rows = [
            ["Session ID",  d.session_id],
            ["Subject ID",  d.subject_id or "Anonymous"],
            ["Date / Time", d.date.strftime("%Y-%m-%d  %H:%M:%S")],
            ["Paradigm",    d.paradigm.replace("_", " ").upper()],
            ["Eye tested",  d.eye_tested],
            ["Operator",    d.operator],
        ]
        meta_table = Table(meta_rows, colWidths=[4 * cm, 12 * cm])
        meta_table.setStyle(TableStyle([
            ("FONTSIZE",       (0, 0), (-1, -1), 9),
            ("FONTNAME",       (0, 0), (0, -1), "Helvetica-Bold"),
            ("TEXTCOLOR",      (0, 0), (0, -1), colors.HexColor("#333333")),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1),
             [colors.white, colors.HexColor("#f8f8f8")]),
            ("GRID",           (0, 0), (-1, -1), 0.25, colors.HexColor("#dddddd")),
            ("TOPPADDING",     (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING",  (0, 0), (-1, -1), 3),
        ]))
        story.append(meta_table)

        if d.is_doc_mode:
            story.append(Spacer(1, 0.2 * cm))
            story.append(Paragraph(
                "⚠  Disorders of Consciousness (DoC) mode — passive paradigm",
                ParagraphStyle("doc_badge", parent=style_body,
                               textColor=colors.HexColor("#D85A30"),
                               fontName="Helvetica-Bold"),
            ))

        story.append(Spacer(1, 0.4 * cm))

        # ── EP Waveform ────────────────────────────────────────────────────
        story.append(Paragraph("Averaged EP Waveform", style_h2))

        if d.avg is not None:
            png_bytes = self._render_waveform_png(d.avg, d.display_channel,
                                                   d.components)
            img = Image(io.BytesIO(png_bytes), width=14 * cm, height=6 * cm)
            story.append(img)
        else:
            story.append(Paragraph(
                "No averaged waveform available (session may not have collected epochs).",
                style_body,
            ))
        story.append(Spacer(1, 0.3 * cm))

        # ── Component table ────────────────────────────────────────────────
        story.append(Paragraph("Detected Components", style_h2))

        comp_rows = [["Component", "Latency (ms)", "Amplitude (µV)",
                      "Normative range (ms)", "Status"]]
        if d.components:
            for comp in d.components:
                norm = _NORMATIVE.get(comp.name)
                if norm:
                    norm_str = f"{norm[0]:.0f} – {norm[1]:.0f}"
                    in_range = norm[0] <= comp.latency_ms <= norm[1]
                    status   = "✓ Normal" if in_range else "⚠ Outside range"
                else:
                    norm_str = "—"
                    status   = "—"
                comp_rows.append([
                    comp.name,
                    f"{comp.latency_ms:.1f}",
                    f"{comp.amplitude_uv:.2f}",
                    norm_str,
                    status,
                ])
        else:
            comp_rows.append(["No components detected", "—", "—", "—", "—"])

        comp_table = Table(comp_rows, colWidths=[3*cm, 3*cm, 3.5*cm, 4*cm, 3.5*cm])
        comp_style = TableStyle([
            ("BACKGROUND",     (0, 0), (-1, 0),  colors.HexColor("#1a1a8c")),
            ("TEXTCOLOR",      (0, 0), (-1, 0),  colors.white),
            ("FONTNAME",       (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",       (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#f0f0f8")]),
            ("GRID",           (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
            ("TOPPADDING",     (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING",  (0, 0), (-1, -1), 3),
            ("ALIGN",          (1, 1), (-1, -1), "CENTER"),
        ])
        # Colour "Outside range" rows
        for row_idx, comp in enumerate(d.components, start=1):
            norm = _NORMATIVE.get(comp.name)
            if norm and not (norm[0] <= comp.latency_ms <= norm[1]):
                comp_style.add("TEXTCOLOR", (4, row_idx), (4, row_idx),
                               colors.HexColor("#D85A30"))
        comp_table.setStyle(comp_style)
        story.append(comp_table)
        story.append(Spacer(1, 0.3 * cm))

        # ── Epoch statistics ───────────────────────────────────────────────
        story.append(Paragraph("Epoch Statistics", style_h2))
        total   = d.n_accepted + d.n_rejected
        rej_pct = d.rejection_rate * 100
        snr_str = f"{d.snr_db:.1f} dB" if np.isfinite(d.snr_db) else "N/A"

        stat_rows = [
            ["Accepted epochs",  str(d.n_accepted)],
            ["Rejected epochs",  f"{d.n_rejected}  ({rej_pct:.1f}%)"],
            ["Total epochs",     str(total)],
            ["Mean SNR",         snr_str],
        ]
        if d.timing_mean_ms is not None:
            stat_rows.append([
                "Pipeline timing",
                f"mean {d.timing_mean_ms:.2f} ms  SD {d.timing_sd_ms:.2f} ms",
            ])

        stat_table = Table(stat_rows, colWidths=[5 * cm, 11 * cm])
        stat_table.setStyle(TableStyle([
            ("FONTSIZE",       (0, 0), (-1, -1), 9),
            ("FONTNAME",       (0, 0), (0, -1), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1),
             [colors.white, colors.HexColor("#f8f8f8")]),
            ("GRID",           (0, 0), (-1, -1), 0.25, colors.HexColor("#dddddd")),
            ("TOPPADDING",     (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING",  (0, 0), (-1, -1), 3),
        ]))
        # Highlight high rejection rate
        if rej_pct > 30:
            stat_table.setStyle(TableStyle([
                ("TEXTCOLOR", (1, 1), (1, 1), colors.HexColor("#D85A30")),
            ]))
        story.append(stat_table)
        story.append(Spacer(1, 0.3 * cm))

        # ── Clinical note ──────────────────────────────────────────────────
        if d.clinical_note.strip():
            story.append(Paragraph("Clinical Note", style_h2))
            story.append(Paragraph(d.clinical_note.replace("\n", "<br/>"),
                                   style_note))
            story.append(Spacer(1, 0.3 * cm))

        # ── Interpretation guidelines ──────────────────────────────────────
        story.append(Paragraph("Normative Reference", style_h2))
        guidelines = [
            ["Component", "Normal latency (ms)", "Clinical relevance"],
            ["P100 (VEP)", "90 – 115",
             "Delayed: optic neuritis, MS, compressive lesion"],
            ["N1 (AEP)",   "80 – 130",
             "Absent/delayed: peripheral or central auditory pathway dysfunction"],
            ["P2 (AEP)",   "150 – 230",
             "Cortical auditory processing"],
            ["N2 (P300)",  "180 – 260",
             "Pre-attentive discrimination; preserved in DoC"],
            ["P300",       "280 – 450",
             "Cognitive processing; often absent in DoC; latency increases with severity"],
        ]
        g_table = Table(guidelines, colWidths=[3*cm, 4*cm, 10*cm])
        g_table.setStyle(TableStyle([
            ("BACKGROUND",     (0, 0), (-1, 0), colors.HexColor("#555588")),
            ("TEXTCOLOR",      (0, 0), (-1, 0), colors.white),
            ("FONTNAME",       (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",       (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#f0f0f8")]),
            ("GRID",           (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
            ("TOPPADDING",     (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING",  (0, 0), (-1, -1), 3),
            ("VALIGN",         (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(g_table)
        story.append(Spacer(1, 0.4 * cm))

        # ── Footer ─────────────────────────────────────────────────────────
        story.append(HRFlowable(width="100%", thickness=0.5,
                                color=colors.HexColor("#aaaaaa")))
        story.append(Paragraph(
            f"Generated by NeuroEP Studio  |  "
            f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  "
            "For clinical use by qualified neurophysiology personnel only.",
            ParagraphStyle("footer", parent=style_body,
                           fontSize=7, textColor=colors.HexColor("#888888")),
        ))

        doc.build(story)
        logger.info("PDF report saved to %s", path)

    # ── PNG waveform helper ────────────────────────────────────────────────

    def _render_waveform_png(
        self,
        avg:        np.ndarray,
        channel:    int,
        components: list[ComponentResult],
    ) -> bytes:
        """Render the averaged EP waveform to a PNG byte string."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        waveform = avg[channel, :].astype(np.float64)
        t_ms     = EPOCH_TIME_MS.astype(np.float64)

        fig, ax = plt.subplots(figsize=(7, 3), dpi=120)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("#fafafa")

        ax.plot(t_ms, waveform, color="#1a1a8c", linewidth=1.5)
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.invert_yaxis()

        for comp in components:
            idx = int(np.argmin(np.abs(t_ms - comp.latency_ms)))
            amp = float(waveform[idx])
            ax.annotate(
                f"{comp.name} {comp.latency_ms:.0f}ms",
                xy=(comp.latency_ms, amp),
                xytext=(comp.latency_ms + 15, amp + 1.5),
                fontsize=7,
                arrowprops=dict(arrowstyle="-", color="#888888", lw=0.7),
                color="#333333",
            )
            ax.plot(comp.latency_ms, amp, "v", color="#c00000", markersize=4)

        ax.set_xlabel("Time (ms)", fontsize=8)
        ax.set_ylabel("Amplitude (µV)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_xlim(float(t_ms[0]), float(t_ms[-1]))
        ax.grid(True, alpha=0.25)
        ax.set_title(
            f"Grand average  |  Ch: {self._d.channel_name}  "
            f"|  n={self._d.n_accepted} epochs",
            fontsize=8,
        )
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    # ── Plain-text fallback ────────────────────────────────────────────────

    def _write_txt(self, path: Path) -> None:
        d = self._d
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "=" * 72,
            "  NeuroEP Studio — Clinical Report",
            "=" * 72,
            f"  Session ID : {d.session_id}",
            f"  Subject ID : {d.subject_id or 'Anonymous'}",
            f"  Date/Time  : {d.date.strftime('%Y-%m-%d %H:%M:%S')}",
            f"  Paradigm   : {d.paradigm.upper()}",
            f"  Eye tested : {d.eye_tested}",
            f"  Operator   : {d.operator}",
            "",
            "-" * 72,
            "  DETECTED COMPONENTS",
            "-" * 72,
        ]
        if d.components:
            lines.append(
                f"  {'Component':<10} {'Latency(ms)':>12} {'Amplitude(µV)':>14}"
                f"  {'Normative(ms)':>14}  Status"
            )
            for comp in d.components:
                norm = _NORMATIVE.get(comp.name)
                norm_str   = f"{norm[0]:.0f}–{norm[1]:.0f}" if norm else "—"
                in_range   = (norm[0] <= comp.latency_ms <= norm[1]) if norm else None
                status_str = ("Normal" if in_range else "Outside range") if in_range is not None else "—"
                lines.append(
                    f"  {comp.name:<10} {comp.latency_ms:>12.1f} {comp.amplitude_uv:>14.2f}"
                    f"  {norm_str:>14}  {status_str}"
                )
        else:
            lines.append("  No components detected.")

        lines += [
            "",
            "-" * 72,
            "  EPOCH STATISTICS",
            "-" * 72,
            f"  Accepted : {d.n_accepted}",
            f"  Rejected : {d.n_rejected}  ({d.rejection_rate * 100:.1f}%)",
            f"  SNR      : {d.snr_db:.1f} dB" if np.isfinite(d.snr_db) else "  SNR      : N/A",
        ]

        if d.clinical_note.strip():
            lines += [
                "",
                "-" * 72,
                "  CLINICAL NOTE",
                "-" * 72,
            ]
            lines += [f"  {ln}" for ln in d.clinical_note.splitlines()]

        lines += [
            "",
            "-" * 72,
            f"  Generated by NeuroEP Studio  {datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
            "=" * 72,
        ]

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        logger.info("Text report saved to %s", path)
