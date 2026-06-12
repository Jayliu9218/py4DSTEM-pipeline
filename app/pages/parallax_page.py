from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox, QLabel,
    QMessageBox, QPushButton, QVBoxLayout, QWidget,
)

from app.services.parallax_service import (
    BFMaskParams, ParallaxAdvancedParams, ParallaxAlignmentParams,
    ParallaxService, ParallaxStageResult,
)
from app.services.phase_contrast_service import PhaseContrastResult
from app.services.result_registry import ResultRegistry
from app.services.workflow_state import STALE_RESULTS_MESSAGE, WorkflowState, WorkflowStep
from app.widgets.adaptive_image_workspace import AdaptiveImageWorkspace, FigureResult
from app.widgets.log_panel import LogPanel, ProcessSnapshot
from app.widgets.numeric_line_edit import NumericLineEdit


class ParallaxWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(str, float)

    def __init__(self, operation) -> None:
        super().__init__()
        self.operation = operation

    def run(self) -> None:
        try:
            self.finished.emit(self.operation(self.progress.emit))
        except Exception as exc:
            self.failed.emit(str(exc))


class ParallaxPage(QWidget):
    parallax_result_ready = Signal(object)

    def __init__(
        self,
        source_provider: Callable[[], object | None],
        log_panel: LogPanel,
        workflow_state: WorkflowState,
        result_registry: ResultRegistry | None = None,
        service: ParallaxService | None = None,
        stage_mode: str = "bf",
    ) -> None:
        super().__init__()
        self.source_provider = source_provider
        self.log_panel = log_panel
        self.workflow_state = workflow_state
        self.result_registry = result_registry
        self.service = service or ParallaxService()
        self.stage_mode = stage_mode
        self.worker_thread: QThread | None = None
        self.worker: ParallaxWorker | None = None
        self.pending_operation = ""
        self.status_label = QLabel("Idle")
        self.status_label.setWordWrap(True)
        self.workspace = AdaptiveImageWorkspace()

        self.threshold = self._float(0, 1, 0.5, 3)
        self.mask_mode = QComboBox()
        self.mask_mode.addItems(["Threshold", "Interactive circle"])
        self.center_x = self._float(-1, 100000, -1, 2)
        self.center_y = self._float(-1, 100000, -1, 2)
        self.radius = self._float(0.1, 100000, 10, 2)
        self.virtual_count = self._int(1, 4, 4)
        self.finite_dose = QCheckBox("Enable finite-dose tutorial simulation")
        self.prepare_button = QPushButton("Prepare BF Disk & Virtual BF")
        self.accept_bf_button = QPushButton("Accept BF Disk")
        self.prepare_button.clicked.connect(self.prepare_bf)
        self.accept_bf_button.clicked.connect(self.accept_bf)

        self.energy = self._float(10, 1_000_000, 300_000, 0)
        self.padding = self._int(0, 256, 16)
        self.edge_blend = self._int(0, 128, 8)
        self.normalize = QCheckBox("Normalize images")
        self.alignment_threshold = self._float(0, 1, 0.6, 3)
        self.regularize = QCheckBox("Regularize shifts")
        self.upsample = self._int(1, 64, 8)
        self.align_button = QPushButton("Run Parallax Alignment")
        self.align_button.clicked.connect(self.run_alignment)

        self.accept_alignment_button = QPushButton("Accept Alignment Review")
        self.accept_alignment_button.clicked.connect(self.accept_alignment)

        self.subpixel = QCheckBox("Run subpixel alignment")
        self.subpixel.setChecked(True)
        self.kde_upsample = self._int(1, 32, 4)
        self.kde_sigma = self._float(0.001, 10, 0.125, 4)
        self.aberration_fit = QCheckBox("Fit low-order aberrations")
        self.aberration_correction = QCheckBox("Apply aberration correction")
        self.high_order = QCheckBox("Expert: high-order aberration fit")
        self.ctf_fit = QCheckBox("Expert: CTF fit")
        self.max_radial = self._int(1, 20, 3)
        self.max_angular = self._int(0, 20, 4)
        self.advanced_button = QPushButton("Run Advanced Reconstruction")
        self.advanced_button.clicked.connect(self.run_advanced)

        self.save_figures = QCheckBox("Save rendered figures")
        self.save_button = QPushButton("Save Parallax Package")
        self.save_button.clicked.connect(self.save_package)

        self._build_layout()
        self._configure_stage()
        self._watch_parameters()
        self.workflow_state.changed.connect(self._refresh_stale_status)

    def _build_layout(self) -> None:
        self.bf_group = QGroupBox("BF Disk & Virtual BF")
        form = QFormLayout(self.bf_group)
        form.addRow("Mask mode", self.mask_mode)
        form.addRow("Threshold", self.threshold)
        form.addRow("Circle center axis 0 (-1 auto)", self.center_x)
        form.addRow("Circle center axis 1 (-1 auto)", self.center_y)
        form.addRow("Circle radius", self.radius)
        form.addRow("Virtual BF count", self.virtual_count)
        form.addRow(self.finite_dose)
        form.addRow(self.prepare_button)
        form.addRow(self.accept_bf_button)

        self.align_group = QGroupBox("Parallax Alignment")
        form = QFormLayout(self.align_group)
        form.addRow("Energy", self.energy)
        form.addRow("Object padding", self.padding)
        form.addRow("Edge blend", self.edge_blend)
        form.addRow(self.normalize)
        form.addRow("Threshold intensity", self.alignment_threshold)
        form.addRow(self.regularize)
        form.addRow("Cross-correlation upsample", self.upsample)
        form.addRow(self.align_button)

        self.review_group = QGroupBox("Alignment Review")
        form = QFormLayout(self.review_group)
        form.addRow(self.accept_alignment_button)

        self.advanced_group = QGroupBox("Advanced Reconstruction")
        form = QFormLayout(self.advanced_group)
        form.addRow(self.subpixel)
        form.addRow("KDE upsample factor", self.kde_upsample)
        form.addRow("KDE sigma", self.kde_sigma)
        form.addRow(self.aberration_fit)
        form.addRow(self.aberration_correction)
        form.addRow(self.high_order)
        form.addRow(self.ctf_fit)
        form.addRow("Maximum radial order", self.max_radial)
        form.addRow("Maximum angular order", self.max_angular)
        form.addRow(self.advanced_button)

        self.export_group = QGroupBox("Parallax Package")
        form = QFormLayout(self.export_group)
        form.addRow(self.save_figures)
        form.addRow(self.save_button)

        controls = QWidget()
        layout = QVBoxLayout(controls)
        layout.setContentsMargins(0, 0, 0, 0)
        for group in (self.bf_group, self.align_group, self.review_group, self.advanced_group, self.export_group):
            layout.addWidget(group)
        layout.addWidget(self.status_label)
        layout.addStretch(1)
        self.controls_panel = controls
        main = QVBoxLayout(self)
        main.setContentsMargins(0, 0, 0, 0)
        main.addWidget(self.workspace)

    def _configure_stage(self) -> None:
        visible = {
            "bf": {self.bf_group},
            "alignment": {self.align_group},
            "review": {self.review_group},
            "advanced": {self.advanced_group},
            "export": {self.export_group},
        }[self.stage_mode]
        for group in (self.bf_group, self.align_group, self.review_group, self.advanced_group, self.export_group):
            group.setVisible(group in visible)
        self.refresh_stage()

    def refresh_stage(self) -> None:
        ctx = self.service.context
        self.accept_bf_button.setEnabled(ctx.bf_result is not None)
        self.align_button.setEnabled(ctx.accepted_bf_mask is not None)
        self.accept_alignment_button.setEnabled(ctx.alignment_result is not None)
        self.advanced_button.setEnabled(ctx.alignment_accepted)
        self.save_button.setEnabled(ctx.parallax is not None)
        result = {
            "bf": ctx.bf_result,
            "alignment": ctx.alignment_result,
            "review": ctx.alignment_result,
            "advanced": ctx.advanced_result,
        }.get(self.stage_mode)
        if result is not None:
            self._show_result(result)

    def prepare_bf(self) -> None:
        source = self._source()
        if source is None:
            return
        params = BFMaskParams(
            threshold=self.threshold.value(),
            use_circle=self.mask_mode.currentText() == "Interactive circle",
            center_x=None if self.center_x.value() < 0 else self.center_x.value(),
            center_y=None if self.center_y.value() < 0 else self.center_y.value(),
            radius=self.radius.value(),
            virtual_bf_count=int(self.virtual_count.value()),
            finite_dose_enabled=self.finite_dose.isChecked(),
        )
        try:
            result = self.service.prepare_bf(source, params)
        except Exception as exc:
            self._failed(str(exc))
            return
        self._show_result(result)
        if self.mask_mode.currentText() == "Interactive circle" and self.workspace.panels:
            viewer = self.workspace.panels[0].viewer
            set_circle = getattr(viewer, "set_interactive_circle", None)
            if callable(set_circle):
                shape = np.asarray(result.images["Mean diffraction pattern"]).shape
                cx = self.center_x.value() if self.center_x.value() >= 0 else shape[0] / 2
                cy = self.center_y.value() if self.center_y.value() >= 0 else shape[1] / 2
                set_circle(cx, cy, self.radius.value())
                try:
                    viewer.circle_changed.disconnect(self._circle_changed)
                except (RuntimeError, TypeError):
                    pass
                viewer.circle_changed.connect(self._circle_changed)
        self.accept_bf_button.setEnabled(True)
        self.status_label.setText("BF disk prepared. Review and accept it before alignment.")
        self._register(result.images, result.metadata)
        self.workflow_state.mark_completed(WorkflowStep.PARALLAX_BF)

    def _circle_changed(self, center_x: float, center_y: float, radius: float) -> None:
        for control, value in (
            (self.center_x, center_x), (self.center_y, center_y), (self.radius, radius)
        ):
            control.blockSignals(True)
            control.setValue(value)
            control.blockSignals(False)
        self.service.context.accepted_bf_mask = None
        self.service.context.alignment_accepted = False
        self.status_label.setText("Circle changed; prepare and accept the BF disk again.")
        self.workflow_state.parameters_updated(WorkflowStep.PARALLAX_BF)

    def accept_bf(self) -> None:
        try:
            self.service.accept_bf_mask()
        except Exception as exc:
            self._failed(str(exc))
            return
        self.status_label.setText("BF disk accepted. Its immutable snapshot will be passed as dp_mask.")
        self.log_panel.log("Parallax BF disk accepted; alignment preprocess will use dp_mask.")
        self.workflow_state.mark_completed(WorkflowStep.PARALLAX_BF_ACCEPT)

    def run_alignment(self) -> None:
        source = self._source()
        if source is None:
            return
        params = ParallaxAlignmentParams(
            energy=self.energy.value(),
            object_padding_px=(int(self.padding.value()), int(self.padding.value())),
            edge_blend=int(self.edge_blend.value()),
            normalize_images=self.normalize.isChecked(),
            threshold_intensity=self.alignment_threshold.value(),
            regularize_shifts=self.regularize.isChecked(),
            cross_correlation_upsample_factor=int(self.upsample.value()),
        )
        self._start("Parallax Alignment", lambda progress: self.service.align(source, params, progress))

    def accept_alignment(self) -> None:
        try:
            result = self.service.accept_alignment()
        except Exception as exc:
            self._failed(str(exc))
            return
        self.parallax_result_ready.emit(result)
        self.status_label.setText("Alignment review accepted.")
        self.workflow_state.mark_completed(WorkflowStep.PARALLAX_REVIEW)
        self.workflow_state.mark_completed(WorkflowStep.PARALLAX)

    def run_advanced(self) -> None:
        params = ParallaxAdvancedParams(
            run_subpixel=self.subpixel.isChecked(),
            kde_upsample_factor=int(self.kde_upsample.value()),
            kde_sigma_px=self.kde_sigma.value(),
            run_aberration_fit=self.aberration_fit.isChecked(),
            run_aberration_correction=self.aberration_correction.isChecked(),
            run_high_order_fit=self.high_order.isChecked(),
            run_ctf_fit=self.ctf_fit.isChecked(),
            max_radial_order=int(self.max_radial.value()),
            max_angular_order=int(self.max_angular.value()),
        )
        self._start("Advanced Reconstruction", lambda progress: self.service.run_advanced(params, progress))

    def save_package(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Save Parallax Package")
        if not directory:
            return
        try:
            saved = self.service.save_package(Path(directory), self.save_figures.isChecked())
        except Exception as exc:
            self._failed(str(exc))
            return
        self.status_label.setText(f"Saved {len(saved)} package file(s).")
        self.log_panel.process_progress("Export 100%")

    def _start(self, name: str, operation) -> None:
        self.pending_operation = name
        self.status_label.setText(f"Running {name}...")
        self.log_panel.process_started(name, name)
        self.log_panel.process_snapshot(ProcessSnapshot(step=name, parameters=self.params_snapshot()))
        self.worker_thread = QThread()
        self.worker = ParallaxWorker(operation)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._progress)
        self.worker.finished.connect(self._finished)
        self.worker.failed.connect(self._failed)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.start()

    def _finished(self, result: PhaseContrastResult) -> None:
        self._show_result(result)
        self._register(result.images, self.params_snapshot())
        self.status_label.setText(f"{self.pending_operation} complete.")
        self.log_panel.process_finished(self.pending_operation, f"elapsed={result.elapsed_seconds:.2f}s")
        if self.pending_operation == "Parallax Alignment":
            self.workflow_state.mark_completed(WorkflowStep.PARALLAX_ALIGNMENT)
        else:
            self.parallax_result_ready.emit(result)
            self.workflow_state.mark_completed(WorkflowStep.PARALLAX_ADVANCED)
            self.workflow_state.mark_completed(WorkflowStep.PARALLAX)

    def _progress(self, message: str, fraction: float) -> None:
        self.log_panel.log(f"Parallax: {message}")
        self.log_panel.process_progress(f"{message} {int(fraction * 100)}%")

    def _show_result(self, result: ParallaxStageResult | PhaseContrastResult) -> None:
        images = result.images
        figures = []
        for name, image in images.items():
            cmap = "RdBu_r" if name.startswith("Shift ") and name != "Shift Magnitude" else "gray"
            scaling = "linear" if cmap == "RdBu_r" or "Mask" in name or "Convergence" in name else "log"
            figures.append(FigureResult(name, np.asarray(image), colormap=cmap, scaling=scaling))
        self.workspace.set_results(figures[:6])

    def _register(self, images: dict[str, np.ndarray], metadata: dict[str, object]) -> None:
        if self.result_registry is None:
            return
        safe_metadata = {
            key: (f"array shape={value.shape}" if isinstance(value, np.ndarray) else value)
            for key, value in metadata.items()
        }
        for name, image in images.items():
            self.result_registry.register(
                f"parallax_{name.lower().replace(' ', '_')}", "Phase Retrieval",
                image, ("npy", "png", "tiff"), safe_metadata,
            )

    def _source(self):
        source = self.source_provider()
        if source is None:
            QMessageBox.information(self, "Parallax", "Load and assign a Target DataCube first.")
        return source

    def _failed(self, error: str) -> None:
        self.status_label.setText("Failed")
        self.log_panel.log(f"Parallax failed: {error}")
        self.log_panel.process_finished("Parallax failed", error)

    def _watch_parameters(self) -> None:
        step = {
            "bf": WorkflowStep.PARALLAX_BF,
            "alignment": WorkflowStep.PARALLAX_ALIGNMENT,
            "review": WorkflowStep.PARALLAX_REVIEW,
            "advanced": WorkflowStep.PARALLAX_ADVANCED,
            "export": WorkflowStep.PARALLAX,
        }[self.stage_mode]
        for control in self.findChildren(NumericLineEdit):
            control.valueChanged.connect(lambda *_args, s=step: self._parameters_changed(s))
        for control in self.findChildren(QComboBox):
            control.currentTextChanged.connect(lambda *_args, s=step: self._parameters_changed(s))
        for control in self.findChildren(QCheckBox):
            control.toggled.connect(lambda *_args, s=step: self._parameters_changed(s))

    def _parameters_changed(self, step: str) -> None:
        if step == WorkflowStep.PARALLAX_BF and self.service.context.accepted_bf_mask is not None:
            self.service.context.accepted_bf_mask = None
            self.service.context.alignment_accepted = False
            self.status_label.setText("BF parameters changed; prepare and accept the BF disk again.")
        if step == WorkflowStep.PARALLAX_ALIGNMENT:
            self.service.context.alignment_accepted = False
        self.workflow_state.parameters_updated(step)

    def _refresh_stale_status(self) -> None:
        step = {
            "bf": WorkflowStep.PARALLAX_BF,
            "alignment": WorkflowStep.PARALLAX_ALIGNMENT,
            "review": WorkflowStep.PARALLAX_REVIEW,
            "advanced": WorkflowStep.PARALLAX_ADVANCED,
            "export": WorkflowStep.PARALLAX,
        }[self.stage_mode]
        if self.workflow_state.is_stale(step):
            self.status_label.setText(STALE_RESULTS_MESSAGE)
            self.status_label.setStyleSheet("color: orange;")

    def params_snapshot(self) -> dict[str, object]:
        return {
            "stage": self.stage_mode,
            "bf_threshold": self.threshold.value(),
            "bf_mode": self.mask_mode.currentText(),
            "energy": self.energy.value(),
            "padding": int(self.padding.value()),
            "edge_blend": int(self.edge_blend.value()),
            "subpixel": self.subpixel.isChecked(),
            "aberration_fit": self.aberration_fit.isChecked(),
            "aberration_correction": self.aberration_correction.isChecked(),
            "high_order": self.high_order.isChecked(),
            "ctf_fit": self.ctf_fit.isChecked(),
            "bf_accepted": self.service.context.accepted_bf_mask is not None,
            "alignment_accepted": self.service.context.alignment_accepted,
        }

    def apply_params_snapshot(self, params: dict[str, object]) -> None:
        if "parallax_threshold" in params:
            self.threshold.setValue(float(params["parallax_threshold"]))
        if "parallax_energy" in params:
            self.energy.setValue(float(params["parallax_energy"]))
        if "parallax_padding" in params:
            self.padding.setValue(float(params["parallax_padding"]))
        if "parallax_edge_blend" in params:
            self.edge_blend.setValue(float(params["parallax_edge_blend"]))

    @staticmethod
    def _float(minimum, maximum, value, decimals) -> NumericLineEdit:
        return NumericLineEdit(minimum, maximum, value, decimals=decimals)

    @staticmethod
    def _int(minimum, maximum, value) -> NumericLineEdit:
        return NumericLineEdit(minimum, maximum, value, decimals=0, integer=True)
