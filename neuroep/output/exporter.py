"""
output/exporter.py — Save session data in PNG, CSV, and EDF+ formats.

Public API
----------
Exporter : saves the averaged EP waveform to PNG, CSV, or EDF+.
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from neuroep import config
from neuroep.processing.components import ComponentResult
from neuroep.processing.epochs import EPOCH_TIME_MS

logger = logging.getLogger(__name__)


class Exporter:
    """
    Saves EP session data to disk.

    Parameters
    ----------
    session_id : str
        Identifier used in file titles and CSV headers.
    paradigm : str
        Paradigm key (e.g. ``"vep_pattern"``).
    subject_id : str
        Subject identifier for metadata.
    n_epochs : int
        Number of accepted epochs in the average.
    channel_name : str
        Name of the displayed channel (e.g. ``"Oz"``).
    """

    def __init__(
        self,
        session_id:   str = "session",
        paradigm:     str = "unknown",
        subject_id:   str = "ANON",
        n_epochs:     int = 0,
        channel_name: str = "Oz",
    ) -> None:
        self._session_id   = session_id
        self._paradigm     = paradigm
        self._subject_id   = subject_id
        self._n_epochs     = n_epochs
        self._channel_name = channel_name

    # ── PNG ────────────────────────────────────────────────────────────────

    def save_png(
        self,
        path:       Path,
        avg:        np.ndarray,
        channel:    int,
        components: list[ComponentResult],
    ) -> None:
        """
        Save the averaged EP waveform as a publication-quality PNG.

        Uses matplotlib with a white background, labelled axes, and
        vertical markers for each detected component.

        Parameters
        ----------
        path : Path
            Output file path (should end in .png).
        avg : np.ndarray, shape (n_channels, epoch_len)
        channel : int
            Row of *avg* to plot.
        components : list[ComponentResult]
        """
        import matplotlib
        matplotlib.use("Agg")   # off-screen rendering
        import matplotlib.pyplot as plt

        waveform = avg[channel, :].astype(np.float64)
        t_ms     = EPOCH_TIME_MS.astype(np.float64)

        fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
        fig.patch.set_facecolor("white")

        # Waveform
        ax.plot(t_ms, waveform, color="#1a1a8c", linewidth=1.5, label="Grand average")

        # Reference lines
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.7)

        # Component markers
        for comp in components:
            idx = int(np.argmin(np.abs(t_ms - comp.latency_ms)))
            amp = float(waveform[idx])
            ax.annotate(
                f"{comp.name}\n{comp.latency_ms:.0f} ms",
                xy=(comp.latency_ms, amp),
                xytext=(comp.latency_ms + 10, amp + 1.5),
                fontsize=7,
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.8),
                color="#333333",
            )
            ax.plot(comp.latency_ms, amp, "v", color="#c00000", markersize=5)

        # Axes
        ax.set_xlabel("Time (ms)", fontsize=9)
        ax.set_ylabel("Amplitude (µV)", fontsize=9)
        ax.invert_yaxis()   # EEG convention: negative up
        ax.grid(True, alpha=0.3)
        ax.set_xlim(float(t_ms[0]), float(t_ms[-1]))

        # Title with metadata
        ts  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        title = (
            f"{self._paradigm.upper().replace('_', ' ')}  |  "
            f"Ch: {self._channel_name}  |  n={self._n_epochs}  |  "
            f"Subject: {self._subject_id}  |  {ts}"
        )
        ax.set_title(title, fontsize=8, color="#333333")

        plt.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info("EP waveform saved to %s", path)

    # ── CSV ────────────────────────────────────────────────────────────────

    def save_csv(
        self,
        path:       Path,
        avg:        np.ndarray,
        channel:    int,
        components: list[ComponentResult],
    ) -> None:
        """
        Export the averaged EP waveform as a CSV file.

        Columns: ``time_ms``, ``amplitude_uv``.
        A header block records session metadata and component latencies.

        Parameters
        ----------
        path : Path
            Output file path (should end in .csv).
        avg : np.ndarray, shape (n_channels, epoch_len)
        channel : int
        components : list[ComponentResult]
        """
        waveform = avg[channel, :].astype(np.float64)
        t_ms     = EPOCH_TIME_MS.astype(np.float64)

        path.parent.mkdir(parents=True, exist_ok=True)

        # Build header comment lines
        ts = datetime.datetime.now().isoformat(timespec="seconds")
        header_lines = [
            f"# NeuroEP Studio export",
            f"# session_id: {self._session_id}",
            f"# subject_id: {self._subject_id}",
            f"# paradigm:   {self._paradigm}",
            f"# channel:    {self._channel_name}",
            f"# n_epochs:   {self._n_epochs}",
            f"# date:       {ts}",
            f"# sample_rate: {config.BOARD_SAMPLE_RATE} Hz",
        ]
        for comp in components:
            header_lines.append(
                f"# {comp.name}: latency={comp.latency_ms:.1f} ms  "
                f"amplitude={comp.amplitude_uv:.2f} µV"
            )
        header_lines.append("# ---")
        header_lines.append("time_ms,amplitude_uv")

        with open(path, "w", encoding="utf-8") as f:
            for line in header_lines:
                f.write(line + "\n")
            for t, a in zip(t_ms, waveform):
                f.write(f"{t:.4f},{a:.6f}\n")

        logger.info("EP data exported to %s (%d rows)", path, len(waveform))

    # ── EDF+ ───────────────────────────────────────────────────────────────

    def save_edf(
        self,
        path:    Path,
        raw_data: np.ndarray,
    ) -> None:
        """
        Save the full 16-channel raw recording as an EDF+ file using MNE.

        Parameters
        ----------
        path : Path
            Output file path (should end in .edf).
        raw_data : np.ndarray, shape (N_CHANNELS, n_samples)
            Raw (unfiltered) EEG in µV.
        """
        import mne

        ch_types = ["eeg"] * config.N_CHANNELS
        info = mne.create_info(
            ch_names = config.CHANNEL_NAMES,
            sfreq    = config.BOARD_SAMPLE_RATE,
            ch_types = ch_types,
        )
        # MNE expects Volts — convert from µV
        raw_v = raw_data.astype(np.float64) * 1e-6
        raw   = mne.io.RawArray(raw_v, info, verbose=False)

        path.parent.mkdir(parents=True, exist_ok=True)
        mne.export.export_raw(str(path), raw, fmt="edf", overwrite=True, verbose=False)
        logger.info("EDF+ file saved to %s", path)
