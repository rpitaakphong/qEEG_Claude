"""
ui/connect_dialog.py — Board connection dialog shown at application startup.

Fields
------
- Board type  : Cyton+Daisy (16ch) or Synthetic (testing)
- Serial port : pre-filled from config.SERIAL_PORT, editable
- Scan ports  : auto-detect available COM/USB ports via serial.tools.list_ports
- Connect     : open board session, show progress spinner
- Test        : run 3-second synthetic board self-test then disconnect
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from brainflow.board_shim import BoardIds, BrainFlowError
from PyQt6.QtCore import QThread, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from neuroep import config
from neuroep.acquisition.board import BoardManager

logger = logging.getLogger(__name__)

# ── Port scanning ──────────────────────────────────────────────────────────
try:
    import serial.tools.list_ports as _list_ports
    _SERIAL_AVAILABLE = True
except ImportError:
    _SERIAL_AVAILABLE = False
    logger.warning("pyserial not found — port scanning disabled.")


def _scan_ports() -> list[str]:
    """Return available serial port names, sorted."""
    if not _SERIAL_AVAILABLE:
        return []
    ports = sorted(p.device for p in _list_ports.comports())
    return ports


# ── Background connection worker ───────────────────────────────────────────

class _ConnectWorker(QThread):
    """
    Runs ``BoardManager.connect()`` off the main thread so the UI stays
    responsive during the BrainFlow handshake.

    Signals
    -------
    succeeded : emitted with the connected BoardManager on success.
    failed    : emitted with an error message string on failure.
    """

    succeeded: pyqtSignal = pyqtSignal(object)
    failed:    pyqtSignal = pyqtSignal(str)

    def __init__(self, board_id: int, serial_port: str) -> None:
        super().__init__()
        self._board_id    = board_id
        self._serial_port = serial_port

    def run(self) -> None:
        try:
            manager = BoardManager(
                board_id    = self._board_id,
                serial_port = self._serial_port,
            )
            manager.connect()
            self.succeeded.emit(manager)
        except BrainFlowError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # pylint: disable=broad-except
            self.failed.emit(f"Unexpected error: {exc}")


class _SelfTestWorker(QThread):
    """
    Connects a synthetic board, streams for 3 seconds, then disconnects.
    Reports success or failure via signals.
    """

    succeeded: pyqtSignal = pyqtSignal(str)
    failed:    pyqtSignal = pyqtSignal(str)

    def run(self) -> None:
        try:
            manager = BoardManager(board_id=BoardIds.SYNTHETIC_BOARD.value)
            manager.connect()
            time.sleep(3)
            n = manager.ring_buffer.n_samples
            manager.disconnect()
            self.succeeded.emit(
                f"Self-test passed — received {n} samples in 3 s "
                f"(expected ~{config.BOARD_SAMPLE_RATE * 3})."
            )
        except Exception as exc:  # pylint: disable=broad-except
            self.failed.emit(str(exc))


# ── Dialog ─────────────────────────────────────────────────────────────────

class ConnectDialog(QDialog):
    """
    Modal dialog for board selection and connection.

    After ``exec()`` returns ``QDialog.Accepted``, call ``take_manager()``
    to retrieve the connected ``BoardManager``.

    Parameters
    ----------
    parent : QWidget, optional
    """

    # Board types offered to the user
    _BOARD_OPTIONS: list[tuple[str, int]] = [
        ("Cyton + Daisy (16 ch)", BoardIds.CYTON_DAISY_BOARD.value),
        ("Synthetic (testing)",   BoardIds.SYNTHETIC_BOARD.value),
    ]

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Connect to EEG Board")
        self.setMinimumWidth(420)
        self.setModal(True)

        self._manager: Optional[BoardManager] = None
        self._worker:  Optional[QThread]      = None

        self._build_ui()

    # ── UI construction ────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(12)
        root.setContentsMargins(20, 20, 20, 20)

        # Title
        title = QLabel("NeuroEP Studio — Board Connection")
        title.setStyleSheet("font-size: 13pt; font-weight: bold;")
        root.addWidget(title)

        # Form fields
        form = QFormLayout()
        form.setSpacing(8)

        self._board_combo = QComboBox()
        for label, _ in self._BOARD_OPTIONS:
            self._board_combo.addItem(label)
        self._board_combo.currentIndexChanged.connect(self._on_board_changed)
        form.addRow("Board type:", self._board_combo)

        port_row = QHBoxLayout()
        self._port_edit = QLineEdit(config.SERIAL_PORT)
        self._port_edit.setPlaceholderText("e.g. COM3 or /dev/ttyUSB0")
        port_row.addWidget(self._port_edit)

        self._scan_btn = QPushButton("Scan ports")
        self._scan_btn.setFixedWidth(100)
        self._scan_btn.clicked.connect(self._scan_ports)
        port_row.addWidget(self._scan_btn)
        form.addRow("Serial port:", port_row)

        root.addLayout(form)

        # Progress bar (hidden until connecting)
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)   # indeterminate
        self._progress.setVisible(False)
        root.addWidget(self._progress)

        # Status label
        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        self._status_label.setObjectName("label_secondary")
        root.addWidget(self._status_label)

        # Buttons
        btn_row = QHBoxLayout()

        self._test_btn = QPushButton("⚡ Test (synthetic)")
        self._test_btn.clicked.connect(self._run_self_test)
        btn_row.addWidget(self._test_btn)

        btn_row.addStretch()

        self._connect_btn = QPushButton("Connect")
        self._connect_btn.setDefault(True)
        self._connect_btn.clicked.connect(self._connect)
        btn_row.addWidget(self._connect_btn)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(self._cancel_btn)

        root.addLayout(btn_row)

        # Reflect initial board selection
        self._on_board_changed(0)

    # ── Slot helpers ───────────────────────────────────────────────────────

    def _on_board_changed(self, index: int) -> None:
        """Enable / disable serial port field based on board type."""
        _, board_id = self._BOARD_OPTIONS[index]
        is_synthetic = board_id == BoardIds.SYNTHETIC_BOARD.value
        self._port_edit.setEnabled(not is_synthetic)
        self._scan_btn.setEnabled(not is_synthetic)

    def _scan_ports(self) -> None:
        """Populate a combo-box popup with detected ports."""
        ports = _scan_ports()
        if not ports:
            self._status_label.setText(
                "No serial ports found. "
                + ("" if _SERIAL_AVAILABLE else "(pyserial not installed)")
            )
            return

        # Show a small popup to pick from detected ports
        picker = QComboBox(self)
        from PyQt6.QtWidgets import QInputDialog
        choice, ok = QInputDialog.getItem(
            self,
            "Select port",
            "Available serial ports:",
            ports,
            0,
            False,
        )
        if ok and choice:
            self._port_edit.setText(choice)

    def _set_busy(self, busy: bool) -> None:
        """Toggle the progress spinner and disable inputs while connecting."""
        self._progress.setVisible(busy)
        self._connect_btn.setEnabled(not busy)
        self._test_btn.setEnabled(not busy)
        self._board_combo.setEnabled(not busy)
        self._port_edit.setEnabled(not busy and not self._is_synthetic())
        self._scan_btn.setEnabled(not busy and not self._is_synthetic())

    def _is_synthetic(self) -> bool:
        _, board_id = self._BOARD_OPTIONS[self._board_combo.currentIndex()]
        return board_id == BoardIds.SYNTHETIC_BOARD.value

    # ── Connect ────────────────────────────────────────────────────────────

    def _connect(self) -> None:
        """Start the background connection worker."""
        index = self._board_combo.currentIndex()
        _, board_id = self._BOARD_OPTIONS[index]
        serial_port  = self._port_edit.text().strip()

        self._status_label.setText("Connecting…")
        self._set_busy(True)

        self._worker = _ConnectWorker(board_id, serial_port)
        self._worker.succeeded.connect(self._on_connect_success)
        self._worker.failed.connect(self._on_connect_failed)
        self._worker.start()

    def _on_connect_success(self, manager: BoardManager) -> None:
        self._manager = manager
        self._set_busy(False)
        self._status_label.setText("Connected successfully.")
        logger.info("Board connected via connect dialog.")
        self.accept()

    def _on_connect_failed(self, message: str) -> None:
        self._set_busy(False)
        self._status_label.setText(f"Connection failed: {message}")
        logger.error("Board connection failed: %s", message)
        QMessageBox.critical(self, "Connection failed", message)

    # ── Self-test ──────────────────────────────────────────────────────────

    def _run_self_test(self) -> None:
        """Connect synthetic board, run 3 s, disconnect, report."""
        self._status_label.setText("Running 3-second synthetic board test…")
        self._set_busy(True)

        self._worker = _SelfTestWorker()
        self._worker.succeeded.connect(self._on_test_success)
        self._worker.failed.connect(self._on_test_failed)
        self._worker.start()

    def _on_test_success(self, message: str) -> None:
        self._set_busy(False)
        self._status_label.setText(message)
        QMessageBox.information(self, "Self-test passed", message)

    def _on_test_failed(self, message: str) -> None:
        self._set_busy(False)
        self._status_label.setText(f"Self-test failed: {message}")
        QMessageBox.critical(self, "Self-test failed", message)

    # ── Public API ─────────────────────────────────────────────────────────

    def take_manager(self) -> Optional[BoardManager]:
        """
        Return the connected ``BoardManager`` and clear the internal reference.

        Must only be called after the dialog has been accepted.
        """
        manager = self._manager
        self._manager = None
        return manager
