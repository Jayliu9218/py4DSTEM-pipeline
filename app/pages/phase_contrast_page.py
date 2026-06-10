from __future__ import annotations

import re
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.services.phase_contrast_service import (
    DPCParams,
    ParallaxParams,
    PhaseContrastResult,
    PhaseContrastService,
    PtychographyParams,
)
from app.services.result_registry import ResultRegistry
from app.services.workflow_state import STALE_RESULTS_MESSAGE, WorkflowState, WorkflowStep
from app.widgets.image_viewer import ImageViewer
from app.widgets.log_panel import LogPanel, ProcessSnapshot
from app.widgets.numeric_line_edit import NumericLineEdit
from app.widgets.progress_stream import ProgressStream


class PhaseContrastWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self, operation) -> None:
        super().__init__()
        self.operation = operation

    def run(self) -> None:
        try:
            stream = ProgressStream(self.progress.emit)
            with redirect_stdout(stream), redirect_stderr(stream):
                self.finished.emit(self.operation())
            stream.flush()
        except Exception as exc:
            self.failed.emit(str(exc))


class PhaseContrastPage(QWidget):
    phase_contrast_ready = Signal(object)

    def __init__(
        self,
        source_provider: Callable[[], object | None],
        log_panel: LogPanel,
        workflow_state: WorkflowState,
        result_registry: ResultRegistry | None = None,
    ) -> None:
        super().__init__()
        self.source_provider = source_provider
        self.log_panel = log_panel
        self.workflow_state = workflow_state
        self.result_registry = result_registry
        self.service = PhaseContrastService()
        self.worker_thread: QThread | None = None
        self.worker: PhaseContrastWorker | None = None
        self.current_process_step = WorkflowStep.PHASE_CONTRAST
        self.result: PhaseContrastResult | None = None

        self.method_combo = QComboBox()
        self.method_combo.addItems([
            PhaseContrastService.PTYCHOGRAPHY,
            PhaseContrastService.PARALLAX,
            PhaseContrastService.DPC,
        ])
        self.method_combo.currentTextChanged.connect(self._sync_method_params)

        self.status_label = QLabel("Idle")
        self.status_label.setWordWrap(True)

        self.ptych_energy = self._float_input(10, 1000000, 80000, decimals=0, unit="eV")
        self.ptych_defocus = self._float_input(0, 100000, 500, decimals=1, unit="A")
        self.ptych_vacuum_probe_label = QLabel("No vacuum probe loaded")
        self.ptych_vacuum_probe_path: str | None = None
        self.ptych_vacuum_probe_btn = QPushButton("Load Vacuum Probe")
        self.ptych_vacuum_probe_btn.clicked.connect(self._load_vacuum_probe)
        self.ptych_num_iter = self._int_input(1, 512, 64, unit="iterations")
        self.ptych_batch_size = self._int_input(1, 4096, 512, unit="positions")
        self.ptych_object_type = QComboBox()
        self.ptych_object_type.addItems(["potential", "complex"])
        self.ptych_positivity_label = QLabel("object_positivity: enabled")

        self.parallax_energy = self._float_input(10, 1000000, 300000, decimals=0, unit="eV")
        self.parallax_padding = self._int_input(0, 128, 16, unit="px")
        self.parallax_edge_blend = self._int_input(0, 64, 8, unit="px")
        self.parallax_alignment_bins = QLabel("32,32,32,16,16,8,8")

        self.dpc_energy = self._float_input(10, 1000000, 200000, decimals=0, unit="eV")

        self.run_button = QPushButton("Reconstruct")
        self.run_button.clicked.connect(self._run)

        self.viewer = ImageViewer()

        self._watch_parameters()
        self.workflow_state.changed.connect(self._refresh_stale_status)
        self._build_layout()
        self._sync_method_params(self.method_combo.currentText())

    def _build_layout(self) -> None:
        ptych_group = QWidget()
        ptych_form = QFormLayout(ptych_group)
        ptych_form.addRow("Energy", self.ptych_energy)
        ptych_form.addRow("Defocus", self.ptych_defocus)
        ptych_form.addRow(self.ptych_vacuum_probe_btn)
        ptych_form.addRow(self.ptych_vacuum_probe_label)
        ptych_form.addRow("Iterations", self.ptych_num_iter)
        ptych_form.addRow("Max Batch", self.ptych_batch_size)
        ptych_form.addRow("Object Type", self.ptych_object_type)
        ptych_form.addRow(self.ptych_positivity_label)

        parallax_group = QWidget()
        parallax_form = QFormLayout(parallax_group)
        parallax_form.addRow("Energy", self.parallax_energy)
        parallax_form.addRow("Padding", self.parallax_padding)
        parallax_form.addRow("Edge Blend", self.parallax_edge_blend)
        parallax_form.addRow("Alignment Bins", self.parallax_alignment_bins)

        dpc_group = QWidget()
        dpc_form = QFormLayout(dpc_group)
        dpc_form.addRow("Energy", self.dpc_energy)

        self._param_widgets = {
            PhaseContrastService.PTYCHOGRAPHY: ptych_group,
            PhaseContrastService.PARALLAX: parallax_group,
            PhaseContrastService.DPC: dpc_group,
        }

        controls = QWidget()
        layout = QVBoxLayout(controls)
        layout.setContentsMargins(0, 0, 0, 0)
        form = QFormLayout()
        form.addRow("Method", self.method_combo)
        layout.addLayout(form)
        for widget in self._param_widgets.values():
            layout.addWidget(widget)
        layout.addWidget(self.run_button)
        layout.addWidget(self.status_label)
        layout.addStretch(1)

        self.controls_panel = controls

        main_layout = QHBoxLayout(self)
        main_layout.addWidget(self.viewer, 1)

    def _sync_method_params(self, method: str) -> None:
        for key, widget in self._param_widgets.items():
            widget.setVisible(key == method)

    def _load_vacuum_probe(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Vacuum Probe", "", "HDF5 Files (*.h5 *.hdf5);;All Files (*)"
        )
        if path:
            self.ptych_vacuum_probe_path = path
            self.ptych_vacuum_probe_label.setText(f"Probe: {Path(path).name}")

    def _current_params(self):
        method = self.method_combo.currentText()
        if method == PhaseContrastService.PTYCHOGRAPHY:
            return PtychographyParams(
                energy=self.ptych_energy.value(),
                defocus=self.ptych_defocus.value(),
                vacuum_probe_path=self.ptych_vacuum_probe_path,
                num_iter=int(self.ptych_num_iter.value()),
                max_batch_size=int(self.ptych_batch_size.value()),
                object_type=self.ptych_object_type.currentText(),
            )
        elif method == PhaseContrastService.PARALLAX:
            return ParallaxParams(
                energy=self.parallax_energy.value(),
                object_padding_px=(int(self.parallax_padding.value()), int(self.parallax_padding.value())),
                edge_blend=int(self.parallax_edge_blend.value()),
            )
        else:
            return DPCParams(
                energy=self.dpc_energy.value(),
            )

    def _run(self) -> None:
        source = self.source_provider()
        if source is None:
            QMessageBox.information(self, "Phase Contrast", "Load a 4D DataCube first.")
            return

        method = self.method_combo.currentText()
        params = self._current_params()
        self.status_label.setText(f"Running {method}...")
        self.run_button.setEnabled(False)

        self.log_panel.log(f"Phase contrast reconstruction started: {method}")
        self.log_panel.process_started("Phase contrast", method)
        self.log_panel.process_snapshot(
            ProcessSnapshot(
                step=method,
                parameters={"method": method, "energy": params.energy},
            )
        )

        if method == PhaseContrastService.PTYCHOGRAPHY:
            operation = lambda: self.service.run_ptychography(source, params)
        elif method == PhaseContrastService.PARALLAX:
            operation = lambda: self.service.run_parallax(source, params)
        else:
            operation = lambda: self.service.run_dpc(source, params)

        self.worker_thread = QThread()
        self.worker = PhaseContrastWorker(operation)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._handle_finished)
        self.worker.failed.connect(self._handle_failed)
        self.worker.progress.connect(self._handle_progress)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.start()

    def _handle_finished(self, result: PhaseContrastResult) -> None:
        self.result = result
        self.run_button.setEnabled(True)
        self.status_label.setText(f"Done in {result.elapsed_seconds:.1f}s")
        self.log_panel.process_finished("Phase contrast done", f"{result.elapsed_seconds:.1f}s")

        if result.rotation_degrees is not None:
            self.log_panel.log(f"Estimated rotation: {result.rotation_degrees:.1f} degrees")

        for name, image in result.images.items():
            self.viewer.set_image(image, name=name)
            break

        self.phase_contrast_ready.emit(result)
        self.workflow_state.mark_completed(self.current_process_step)

    def _handle_failed(self, error: str) -> None:
        self.run_button.setEnabled(True)
        self.status_label.setText("Failed")
        self.log_panel.log(f"Phase contrast failed: {error}")
        self.log_panel.process_finished("Phase contrast failed", error)

    def _handle_progress(self, message: str) -> None:
        self.log_panel.log(message)
        m = re.search(r"(\d+)%", message)
        if m:
            pct = int(m.group(1))
            self.log_panel.process_progress(f"Phase contrast {pct}%")

    def _refresh_stale_status(self) -> None:
        steps = [WorkflowStep.PHASE_CONTRAST]
        if self.workflow_state.any_stale(steps):
            self.status_label.setText(STALE_RESULTS_MESSAGE)
            self.status_label.setStyleSheet("color: orange;")

    def _watch_parameters(self) -> None:
        for spin in [
            self.ptych_energy, self.ptych_defocus,
            self.ptych_num_iter, self.ptych_batch_size,
            self.parallax_energy, self.parallax_padding, self.parallax_edge_blend,
            self.dpc_energy,
        ]:
            self.workflow_state.watch(spin, WorkflowStep.PHASE_CONTRAST, "valueChanged")

    def _float_input(self, minimum, maximum, value, decimals=2, unit="") -> NumericLineEdit:
        return NumericLineEdit(minimum, maximum, value, decimals=decimals, unit=unit)

    def _int_input(self, minimum, maximum, value, unit="") -> NumericLineEdit:
        return NumericLineEdit(minimum, maximum, value, decimals=0, unit=unit)

    def params_snapshot(self) -> dict[str, object]:
        return {
            "method": self.method_combo.currentText(),
            "ptych_energy": self.ptych_energy.value(),
            "ptych_defocus": self.ptych_defocus.value(),
            "ptych_num_iter": int(self.ptych_num_iter.value()),
            "ptych_batch_size": int(self.ptych_batch_size.value()),
            "ptych_object_type": self.ptych_object_type.currentText(),
            "parallax_energy": self.parallax_energy.value(),
            "parallax_padding": int(self.parallax_padding.value()),
            "parallax_edge_blend": int(self.parallax_edge_blend.value()),
            "dpc_energy": self.dpc_energy.value(),
        }