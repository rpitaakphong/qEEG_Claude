"""
main.py — NeuroEP Studio entry point.

Launches the PyQt6 application, shows the board connection dialog, then
opens the main window if connection succeeds.

Run with:
    python -m neuroep.main
or:
    python neuroep/main.py
"""

from __future__ import annotations

import logging
import os
import sys

# Must be set before pyglet/PsychoPy is imported anywhere.
# Disables the shadow window that requires ARB_pixel_format OpenGL support.
os.environ.setdefault("PYGLET_SHADOW_WINDOW", "0")

from PyQt6.QtWidgets import QApplication, QMessageBox

from neuroep.ui.connect_dialog import ConnectDialog
from neuroep.ui.main_window import MainWindow
from neuroep.ui.theme import apply_dark_theme

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Application entry point."""
    app = QApplication(sys.argv)
    app.setApplicationName("NeuroEP Studio")
    app.setApplicationVersion("0.1.0")

    apply_dark_theme(app)

    # Show connection dialog before main window
    dialog = ConnectDialog()
    result = dialog.exec()

    if result != ConnectDialog.DialogCode.Accepted:
        logger.info("User cancelled connection dialog — exiting.")
        sys.exit(0)

    manager = dialog.take_manager()
    if manager is None:
        QMessageBox.critical(None, "Error", "No board manager returned — exiting.")
        sys.exit(1)

    window = MainWindow(manager)
    window.showMaximized()

    logger.info("NeuroEP Studio started.")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
