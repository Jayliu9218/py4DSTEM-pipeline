from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox, QLabel,
    QLineEdit, QMessageBox, QPushButton, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from app.services.parallax_service import (
    BFMaskParams, FiniteDoseParams, ParallaxAdvancedParams,
    FAST_ALIGNMENT_BINS, NOTEBOOK_ALIGNMENT_BINS, ParallaxAlignmentParams,
    ParallaxService, ParallaxStageResult,
)
from app.services.phase_contrast_service import PhaseContrastResult
from app.services.result_registry import ResultRegistry
from app.services.workflow_state import STALE_RESULTS_MESSAGE, WorkflowState, WorkflowStep
from app.theme import Theme
from app.widgets.adaptive_image_workspace import AdaptiveImageWorkspace, FigureResult
from app.widgets.log_panel import LogPanel, ProcessSnapshot
from app.widgets.numeric_line_edit import NumericLineEdit
from app.widgets.worker_runner import WorkerRunner


class ParallaxPage(QWidget, WorkerRunner):
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
        self._init_worker_runner()
        self.cuda_enabled = False
        self._display_signature: tuple[object, ...] | None = None
        self.status_label = QLabel("Idle")
        self.status_label.setWordWrap(True)
        self.workspace = AdaptiveImageWorkspace()

        self.threshold = self._float(0, 1, 0.5, 3)
        self.mask_mode = QComboBox()
        self.mask_mode.addItems(["Threshold", "Interactive circle"])
        self.center_x = self._float(-1, 100000, -1, 2)
        self.center_y = self._float(-1, 100000, -1, 2)
        self.radius = self._float(0.1, 100000, 10, 2)
        self.virtual_count = self._int(1, 5, 5)
        self.virtual_crop = self._int(1, 100000, 48)
        self.bf_view = QComboBox()
        self.bf_view.addItems(["Disk Definition", "Virtual BF Stack"])
        self.bf_view.currentTextChanged.connect(self.refresh_stage)
        self.prepare_button = QPushButton("Prepare BF Disk & Virtual BF")
        self.accept_bf_button = QPushButton("Accept BF Disk")
        self.prepare_button.clicked.connect(self.prepare_bf)
        self.accept_bf_button.clicked.connect(self.accept_bf)

        self.energy = self._float(10, 1_000_000, 300_000, 0)
        self.padding = self._int(0, 256, 16)
        self.edge_blend = self._int(0, 128, 8)
        self.normalize = QCheckBox("Normalize images")
        self.regularize = QCheckBox("Regularize shifts")
        self.alignment_preset = QComboBox()
        self.alignment_preset.addItems(["Fast", "Notebook Quality", "Custom"])
        self.upsample = self._int(1, 64, 4)
        self.alignment_bins = QLineEdit(",".join(str(value) for value in FAST_ALIGNMENT_BINS))
        self.alignment_preset.currentTextChanged.connect(self._apply_alignment_preset)
        self.upsample.valueChanged.connect(self._mark_alignment_custom)
        self.alignment_bins.textChanged.connect(self._mark_alignment_custom)
        self.align_button = QPushButton("Run Parallax Alignment")
        self.align_button.clicked.connect(self.run_alignment)

        self.accept_alignment_button = QPushButton("Accept Alignment Review")
        self.accept_alignment_button.clicked.connect(self.accept_alignment)
        self.review_view = QComboBox()
        self.review_view.addItems([
            "Notebook review",
            "Scalar shift maps",
            "Convergence",
            "Finite-dose aligned BF",
            "Finite-dose diffraction montage",
        ])
        self.review_show_button = QPushButton("Show Selected Review View")
        self.review_show_button.clicked.connect(lambda: self._refresh_display(force=True))
        self.finite_dose_enabled = QCheckBox("Enable finite-dose tutorial comparison")
        self.finite_doses = QLineEdit("100,50,10")
        self.finite_seed = self._int(0, 2_147_483_647, 1234)
        self.finite_button = QPushButton("Run Finite-Dose Comparison")
        self.finite_button.clicked.connect(self.run_finite_dose)
        self.finite_dose_enabled.toggled.connect(self.refresh_stage)

        self.kde_upsample = self._int(1, 32, 3)
        self.kde_sigma = self._float(0.001, 10, 0.125, 4)
        self.high_order = QCheckBox("Expert: high-order aberration fit")
        self.ctf_fit = QCheckBox("Expert: CTF / Thon-ring fit")
        self.max_radial = self._int(1, 20, 3)
        self.max_angular = self._int(0, 20, 4)
        self.max_thon_rings = self._int(1, 50, 5)
        self.capability_label = QLabel("")
        self.capability_label.setWordWrap(True)
        self.advanced_view = QComboBox()
        self.advanced_view.addItems([
            "Latest result",
            "Original vs subpixel BF",
            "Original vs subpixel FFT",
            "Cone-weighted diagnostics",
            "Measured vs fitted shifts",
            "CTF comparison",
            "Aberration corrected BF",
        ])
        self.advanced_show_button = QPushButton("Show Selected Diagnostic")
        self.advanced_show_button.clicked.connect(lambda: self._refresh_display(force=True))
        self.aberration_table = QTableWidget(0, 2)
        self.aberration_table.setHorizontalHeaderLabels(["Coefficient", "Value"])
        self.aberration_table.horizontalHeader().setStretchLastSection(True)
        self.subpixel_button = QPushButton("1. Run Subpixel Reconstruction")
        self.fit_button = QPushButton("2. Fit Aberrations")
        self.correction_button = QPushButton("3. Apply Aberration Correction")
        self.subpixel_button.clicked.connect(self.run_subpixel)
        self.fit_button.clicked.connect(self.fit_aberrations)
        self.correction_button.clicked.connect(self.apply_correction)

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
        for label, control in (
            ("Mask mode", self.mask_mode), ("Threshold", self.threshold),
            ("Circle center axis 0 (-1 auto)", self.center_x),
            ("Circle center axis 1 (-1 auto)", self.center_y),
            ("Circle radius", self.radius), ("Virtual BF count", self.virtual_count),
            ("Virtual BF crop", self.virtual_crop), ("Process view", self.bf_view),
        ):
            form.addRow(label, control)
        form.addRow(self.prepare_button)
        form.addRow(self.accept_bf_button)

        self.align_group = QGroupBox("Parallax Alignment")
        form = QFormLayout(self.align_group)
        for label, control in (
            ("Preset", self.alignment_preset), ("Energy", self.energy), ("Object padding", self.padding),
            ("Edge blend", self.edge_blend), ("Cross-correlation upsample", self.upsample),
            ("Alignment bins", self.alignment_bins),
        ):
            form.addRow(label, control)
        form.addRow(self.normalize)
        form.addRow(self.regularize)
        form.addRow(self.align_button)

        self.review_group = QGroupBox("Alignment Review")
        form = QFormLayout(self.review_group)
        form.addRow("Review view", self.review_view)
        form.addRow(self.review_show_button)
        form.addRow(self.accept_alignment_button)
        form.addRow(self.finite_dose_enabled)
        form.addRow("Comparison doses", self.finite_doses)
        form.addRow("Random seed", self.finite_seed)
        form.addRow(self.finite_button)

        self.advanced_group = QGroupBox("Advanced Reconstruction")
        form = QFormLayout(self.advanced_group)
        form.addRow("KDE upsample factor", self.kde_upsample)
        form.addRow("KDE sigma", self.kde_sigma)
        form.addRow(self.high_order)
        form.addRow(self.ctf_fit)
        form.addRow("Maximum radial order", self.max_radial)
        form.addRow("Maximum angular order", self.max_angular)
        form.addRow("Maximum Thon rings", self.max_thon_rings)
        form.addRow(self.capability_label)
        form.addRow("Diagnostic view", self.advanced_view)
        form.addRow(self.advanced_show_button)
        form.addRow(self.aberration_table)
        form.addRow(self.subpixel_button)
        form.addRow(self.fit_button)
        form.addRow(self.correction_button)

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
            "bf": {self.bf_group}, "alignment": {self.align_group},
            "review": {self.review_group}, "advanced": {self.advanced_group},
            "export": {self.export_group},
        }[self.stage_mode]
        for group in (self.bf_group, self.align_group, self.review_group, self.advanced_group, self.export_group):
            group.setVisible(group in visible)
        self.refresh_stage()

    def refresh_stage(self, *_args) -> None:
        ctx = self.service.context
        capabilities = self.service.adapter.capabilities()
        mask_acceptable = (
            ctx.bf_result is not None
            and bool(ctx.bf_result.metadata.get("mask_acceptable", True))
        )
        self.accept_bf_button.setEnabled(mask_acceptable)
        self.align_button.setEnabled(ctx.accepted_bf_mask is not None)
        self.accept_alignment_button.setEnabled(ctx.alignment_result is not None)
        self.finite_button.setEnabled(self.finite_dose_enabled.isChecked() and ctx.alignment_result is not None)
        self.subpixel_button.setEnabled(ctx.alignment_accepted and capabilities.subpixel_alignment)
        self.fit_button.setEnabled(ctx.alignment_accepted and capabilities.aberration_fit)
        self.correction_button.setEnabled(ctx.aberration_result is not None and capabilities.aberration_correction)
        self.ctf_fit.setEnabled(capabilities.ctf_thon_ring_fit)
        self.capability_label.setText(
            "Native CTF / Thon-ring fitting is unavailable; derived CTF diagnostics remain available."
            if not capabilities.ctf_thon_ring_fit else "Native CTF / Thon-ring fitting is available."
        )
        self.save_button.setEnabled(ctx.parallax is not None)
        self._refresh_aberration_table()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt API name
        super().showEvent(event)
        self._refresh_display()

    def prepare_bf(self) -> None:
        source = self._source()
        if source is None:
            return
        try:
            result = self.service.prepare_bf(
                source,
                BFMaskParams(
                    threshold=self.threshold.value(),
                    use_circle=self.mask_mode.currentText() == "Interactive circle",
                    center_x=None if self.center_x.value() < 0 else self.center_x.value(),
                    center_y=None if self.center_y.value() < 0 else self.center_y.value(),
                    radius=self.radius.value(),
                    virtual_bf_count=int(self.virtual_count.value()),
                    virtual_bf_crop=int(self.virtual_crop.value()),
                ),
            )
        except Exception as exc:
            self._failed(str(exc))
            return
        self._show_result(result, force=True)
        if self.mask_mode.currentText() == "Interactive circle" and self.workspace.panels:
            viewer = self.workspace.panels[0].viewer
            if hasattr(viewer, "set_interactive_circle"):
                shape = result.images["Mean diffraction pattern"].shape
                viewer.set_interactive_circle(
                    self.center_x.value() if self.center_x.value() >= 0 else shape[0] / 2,
                    self.center_y.value() if self.center_y.value() >= 0 else shape[1] / 2,
                    self.radius.value(),
                )
                try:
                    viewer.circle_changed.disconnect(self._circle_changed)
                except (RuntimeError, TypeError):
                    pass
                viewer.circle_changed.connect(self._circle_changed)
        if bool(result.metadata.get("mask_acceptable", True)):
            self.status_label.setText("BF disk prepared. Review and accept it before alignment.")
        else:
            self.status_label.setText(
                "BF disk covers more than 75% of the detector. Refine it before acceptance."
            )
        self._register(result.images, result.metadata)
        self.workflow_state.mark_completed(WorkflowStep.PARALLAX_BF)

    def _circle_changed(self, center_x: float, center_y: float, radius: float) -> None:
        for control, value in ((self.center_x, center_x), (self.center_y, center_y), (self.radius, radius)):
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
        self.workflow_state.mark_completed(WorkflowStep.PARALLAX_BF_ACCEPT)

    def run_alignment(self) -> None:
        source = self._source()
        if source is not None:
            self._start("Parallax Alignment", lambda progress: self.service.align(source, self._alignment_params(), progress))

    def accept_alignment(self) -> None:
        try:
            result = self.service.accept_alignment()
        except Exception as exc:
            self._failed(str(exc))
            return
        self.status_label.setText("Alignment review accepted.")
        self.workflow_state.mark_completed_many(
            {WorkflowStep.PARALLAX_REVIEW, WorkflowStep.PARALLAX}
        )

    def run_finite_dose(self) -> None:
        source = self._source()
        if source is not None:
            self._start(
                "Finite-Dose Comparison",
                lambda progress: self.service.run_finite_dose_comparison(
                    source, self._alignment_params(),
                    FiniteDoseParams(self._float_list(self.finite_doses.text()), int(self.finite_seed.value())),
                    progress,
                ),
            )

    def run_subpixel(self) -> None:
        self._start("Subpixel Reconstruction", lambda progress: self.service.run_subpixel(self._advanced_params(), progress))

    def fit_aberrations(self) -> None:
        self._start("Aberration Fitting", lambda progress: self.service.fit_aberrations(self._advanced_params(), progress))

    def apply_correction(self) -> None:
        self._start("Aberration Correction", lambda progress: self.service.apply_aberration_correction(progress))

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
        # operation already has signature (progress_callback) -> result.
        self._start_background(name, operation, capture_stdout=False,
                               parameters=self.params_snapshot())

    def _handle_result(self, result: PhaseContrastResult | ParallaxStageResult) -> None:
        if self.pending_operation == "Finite-Dose Comparison":
            self.review_view.setCurrentText("Finite-dose aligned BF")
        elif self.pending_operation == "Subpixel Reconstruction":
            self.advanced_view.setCurrentText("Original vs subpixel BF")
        elif self.pending_operation == "Aberration Fitting":
            self.advanced_view.setCurrentText("Measured vs fitted shifts")
        elif self.pending_operation == "Aberration Correction":
            self.advanced_view.setCurrentText("Aberration corrected BF")
        self._show_result(result, force=True)
        self._register(result.images, self.params_snapshot())
        self._refresh_aberration_table()
        self.status_label.setText(f"{self.pending_operation} complete.")
        self.log_panel.process_finished(self.pending_operation, f"elapsed={getattr(result, 'elapsed_seconds', 0):.2f}s")
        if self.pending_operation == "Parallax Alignment":
            self.workflow_state.mark_completed(WorkflowStep.PARALLAX_ALIGNMENT)
        elif self.pending_operation in {"Subpixel Reconstruction", "Aberration Fitting", "Aberration Correction"}:
            self.workflow_state.mark_completed(WorkflowStep.PARALLAX_ADVANCED)
            self.workflow_state.mark_completed(WorkflowStep.PARALLAX)

    def _handle_progress(self, message: str, fraction: float) -> None:
        self.log_panel.log(f"Parallax: {message}")
        super()._handle_progress(message, fraction)

    def _refresh_display(self, force: bool = False) -> None:
        ctx = self.service.context
        result = {
            "bf": ctx.bf_result,
            "alignment": ctx.alignment_result,
            "review": (
                ctx.finite_dose_result
                if self.review_view.currentText().startswith("Finite-dose")
                else ctx.alignment_result
            ),
            "advanced": ctx.correction_result or ctx.aberration_result or ctx.subpixel_result,
        }.get(self.stage_mode)
        if result is not None:
            self._show_result(result, force=force)

    def _show_result(
        self,
        result: ParallaxStageResult | PhaseContrastResult,
        force: bool = False,
    ) -> None:
        view = (
            self.review_view.currentText() if self.stage_mode == "review"
            else self.advanced_view.currentText() if self.stage_mode == "advanced"
            else self.bf_view.currentText() if self.stage_mode == "bf"
            else self.stage_mode
        )
        signature = (id(result), self.service.context.revision, self.stage_mode, view)
        if not force and signature == self._display_signature:
            return
        self._display_signature = signature
        images = result.images
        if result is self.service.context.bf_result:
            names = (
                ["Mean diffraction pattern", "Accepted-mask preview", "Incoherent BF"]
                if self.bf_view.currentText() == "Disk Definition"
                else [name for name in images if name.startswith("Tilted virtual BF")]
            )
            images = {name: images[name] for name in names if name in images}
        names = self._view_names(images, view)
        figures = []
        for name in names:
            image = images.get(name)
            if name not in images:
                if name in {"Measured BF shifts", "Fitted BF shifts"}:
                    image = images.get("Fitted Shift Magnitude", images.get("Shift Magnitude"))
                    if image is None:
                        continue
                else:
                    continue
            cmap = "RdBu_r" if name.startswith("Shift ") and name != "Shift Magnitude" else "gray"
            vectors = None
            vector_stride = 1
            mask = None
            if self.stage_mode == "review" and view == "Notebook review" and name == "Shift Magnitude":
                vectors = self.service.context.shift_vectors
                vector_stride = 4
                mask = self.service.context.accepted_bf_mask
            if self.stage_mode == "advanced" and view == "Measured vs fitted shifts":
                key = "measured_shift_vectors" if name == "Measured BF shifts" else "fitted_shift_vectors"
                vectors = np.asarray(result.metadata.get(key, []))
                vector_stride = 4
            figures.append(FigureResult(
                name, np.asarray(image), colormap=cmap,
                scaling="linear" if cmap == "RdBu_r" or "Mask" in name or "Convergence" in name else "log",
                points=np.asarray(result.metadata.get("selected_points", [])) if name == "Accepted-mask preview" and isinstance(result, ParallaxStageResult) else None,
                vectors=vectors, vector_stride=vector_stride, mask=mask,
            ))
        if self.stage_mode == "advanced" and view == "Cone-weighted diagnostics":
            x = result.metadata.get("radial_cone_frequency")
            y = result.metadata.get("radial_cone_values")
            if x is not None and y is not None:
                figures.append(FigureResult(
                    "Median-filtered radial cone-weighted FFT",
                    self._curve_rgb(np.asarray(x), np.asarray(y), "Spatial frequency", "Cone-weighted FFT"),
                    image_kind="rgb",
                ))
        self.workspace.set_results(figures[:6])

    def _view_names(self, images: dict[str, np.ndarray], view: str) -> list[str]:
        if self.stage_mode == "review":
            return {
                "Notebook review": ["Aligned BF", "Shift Magnitude"],
                "Scalar shift maps": ["Shift X", "Shift Y", "Shift Magnitude"],
                "Convergence": ["Convergence"],
                "Finite-dose aligned BF": [name for name in images if name.startswith("Aligned BF ")],
                "Finite-dose diffraction montage": [name for name in images if name.startswith("Diffraction montage ")],
            }.get(view, [])
        if self.stage_mode == "advanced":
            return {
                "Latest result": list(images),
                "Original vs subpixel BF": ["Original Aligned BF", "Subpixel Aligned BF"],
                "Original vs subpixel FFT": ["Original Aligned BF FFT", "Subpixel Aligned BF FFT"],
                "Cone-weighted diagnostics": ["Cone-weighted FFT"],
                "Measured vs fitted shifts": ["Measured BF shifts", "Fitted BF shifts"],
                "CTF comparison": ["Aligned BF FFT", "Fitted CTF"],
                "Aberration corrected BF": ["Aberration Corrected BF"],
            }.get(view, [])
        return list(images)

    def _refresh_aberration_table(self) -> None:
        values = self.service.context.aberrations_dict_polar
        self.aberration_table.setRowCount(len(values))
        for row, (name, value) in enumerate(sorted(values.items())):
            self.aberration_table.setItem(row, 0, QTableWidgetItem(str(name)))
            self.aberration_table.setItem(row, 1, QTableWidgetItem(f"{float(value):.6g}"))

    @staticmethod
    def _curve_rgb(x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str) -> np.ndarray:
        figure = Figure(figsize=(6, 4), dpi=100, tight_layout=True)
        canvas = FigureCanvasAgg(figure)
        axis = figure.subplots()
        axis.plot(x, y)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        canvas.draw()
        return np.asarray(canvas.buffer_rgba())[..., :3].copy()

    def _register(self, images: dict[str, np.ndarray], metadata: dict[str, object]) -> None:
        if self.result_registry is None:
            return
        safe_metadata = {key: (f"array shape={value.shape}" if isinstance(value, np.ndarray) else value) for key, value in metadata.items()}
        for name, image in images.items():
            self.result_registry.register(
                f"parallax_{name.lower().replace(' ', '_')}", "Phase Retrieval",
                image, ("npy", "png", "tiff"), safe_metadata,
            )
        if self.service.context.aberrations_dict_polar:
            self.result_registry.register(
                "parallax_aberrations_dict_polar", "Phase Retrieval",
                self.service.context.aberrations_dict_polar, ("npz",), safe_metadata,
            )

    def _alignment_params(self) -> ParallaxAlignmentParams:
        return ParallaxAlignmentParams(
            energy=self.energy.value(),
            device="gpu" if self.cuda_enabled else "cpu",
            object_padding_px=(int(self.padding.value()), int(self.padding.value())),
            edge_blend=int(self.edge_blend.value()),
            normalize_images=self.normalize.isChecked(),
            regularize_shifts=self.regularize.isChecked(),
            cross_correlation_upsample_factor=int(self.upsample.value()),
            alignment_bin_values=self._int_list(self.alignment_bins.text()),
        )

    def set_cuda_enabled(self, enabled: bool) -> None:
        self.cuda_enabled = bool(enabled)

    def _apply_alignment_preset(self, preset: str) -> None:
        settings = {
            "Fast": (FAST_ALIGNMENT_BINS, 4),
            "Notebook Quality": (NOTEBOOK_ALIGNMENT_BINS, 8),
        }.get(preset)
        if settings is None:
            return
        bins, upsample = settings
        self.alignment_bins.blockSignals(True)
        self.upsample.blockSignals(True)
        self.alignment_bins.setText(",".join(str(value) for value in bins))
        self.upsample.setValue(upsample)
        self.upsample.blockSignals(False)
        self.alignment_bins.blockSignals(False)

    def _mark_alignment_custom(self, *_args) -> None:
        if self.alignment_preset.currentText() == "Custom":
            return
        self.alignment_preset.blockSignals(True)
        self.alignment_preset.setCurrentText("Custom")
        self.alignment_preset.blockSignals(False)

    def _advanced_params(self) -> ParallaxAdvancedParams:
        return ParallaxAdvancedParams(
            kde_upsample_factor=int(self.kde_upsample.value()),
            kde_sigma_px=self.kde_sigma.value(),
            high_order_fit=self.high_order.isChecked(),
            max_radial_order=int(self.max_radial.value()),
            max_angular_order=int(self.max_angular.value()),
            ctf_thon_ring_fit=self.ctf_fit.isChecked(),
            max_thon_rings=int(self.max_thon_rings.value()),
        )

    def _source(self):
        source = self.source_provider()
        if source is None:
            QMessageBox.information(self, "Parallax", "Load and assign a Target DataCube first.")
        return source

    def _failed(self, error: str) -> None:
        self.status_label.setText("Failed")
        self.log_panel.log(f"Parallax failed: {error}")
        self.log_panel.process_failed(self.pending_operation or "Parallax", error)

    def _handle_error(self, message: str) -> None:
        self._failed(message)

    def _watch_parameters(self) -> None:
        step = {
            "bf": WorkflowStep.PARALLAX_BF, "alignment": WorkflowStep.PARALLAX_ALIGNMENT,
            "review": WorkflowStep.PARALLAX_REVIEW, "advanced": WorkflowStep.PARALLAX_ADVANCED,
            "export": WorkflowStep.PARALLAX,
        }[self.stage_mode]
        for control in self.findChildren(NumericLineEdit):
            if control in {self.finite_seed}:
                continue
            control.valueChanged.connect(lambda *_args, s=step: self._parameters_changed(s))
        for control in self.findChildren(QComboBox):
            if control is self.bf_view:
                continue
            control.currentTextChanged.connect(lambda *_args, s=step: self._parameters_changed(s))
        for control in self.findChildren(QCheckBox):
            if control is self.finite_dose_enabled:
                continue
            control.toggled.connect(lambda *_args, s=step: self._parameters_changed(s))
        for control in self.findChildren(QLineEdit):
            if control is self.finite_doses:
                continue
            control.textChanged.connect(lambda *_args, s=step: self._parameters_changed(s))

    def _parameters_changed(self, step: str) -> None:
        if step == WorkflowStep.PARALLAX_BF and self.service.context.accepted_bf_mask is not None:
            self.service.context.accepted_bf_mask = None
            self.service.context.alignment_accepted = False
        if step == WorkflowStep.PARALLAX_ALIGNMENT:
            self.service.context.alignment_accepted = False
        self.workflow_state.parameters_updated(step)
        self.refresh_stage()

    def _refresh_stale_status(self) -> None:
        step = {
            "bf": WorkflowStep.PARALLAX_BF, "alignment": WorkflowStep.PARALLAX_ALIGNMENT,
            "review": WorkflowStep.PARALLAX_REVIEW, "advanced": WorkflowStep.PARALLAX_ADVANCED,
            "export": WorkflowStep.PARALLAX,
        }[self.stage_mode]
        if self.workflow_state.is_stale(step):
            self.status_label.setText(STALE_RESULTS_MESSAGE)
            self.status_label.setStyleSheet(f"color: {Theme.STALE};")

    def params_snapshot(self) -> dict[str, object]:
        return {
            "stage": self.stage_mode, "bf_threshold": self.threshold.value(),
            "bf_mode": self.mask_mode.currentText(), "virtual_bf_crop": int(self.virtual_crop.value()),
            "energy": self.energy.value(), "padding": int(self.padding.value()),
            "edge_blend": int(self.edge_blend.value()), "alignment_preset": self.alignment_preset.currentText(),
            "alignment_bins": self.alignment_bins.text(), "alignment_upsample": int(self.upsample.value()),
            "device": "gpu" if self.cuda_enabled else "cpu",
            "high_order": self.high_order.isChecked(), "ctf_fit": self.ctf_fit.isChecked(),
            "max_thon_rings": int(self.max_thon_rings.value()),
            "bf_accepted": self.service.context.accepted_bf_mask is not None,
            "alignment_accepted": self.service.context.alignment_accepted,
        }

    def apply_params_snapshot(self, params: dict[str, object]) -> None:
        for key, control in (
            ("parallax_threshold", self.threshold), ("bf_threshold", self.threshold),
            ("parallax_energy", self.energy), ("energy", self.energy),
            ("parallax_padding", self.padding), ("padding", self.padding),
            ("parallax_edge_blend", self.edge_blend), ("edge_blend", self.edge_blend),
            ("virtual_bf_crop", self.virtual_crop),
        ):
            if key in params:
                control.setValue(float(params[key]))
        if "alignment_bins" in params:
            self.alignment_bins.setText(str(params["alignment_bins"]))
        if "alignment_upsample" in params:
            self.upsample.setValue(float(params["alignment_upsample"]))
        if "alignment_preset" in params:
            self.alignment_preset.setCurrentText(str(params["alignment_preset"]))

    @staticmethod
    def _int_list(text: str) -> tuple[int, ...]:
        values = tuple(int(value.strip()) for value in text.split(",") if value.strip())
        if not values:
            raise ValueError("Enter at least one alignment bin.")
        return values

    @staticmethod
    def _float_list(text: str) -> tuple[float, ...]:
        values = tuple(float(value.strip()) for value in text.split(",") if value.strip())
        if not values:
            raise ValueError("Enter at least one finite-dose comparison value.")
        return values

    @staticmethod
    def _float(minimum, maximum, value, decimals) -> NumericLineEdit:
        return NumericLineEdit(minimum, maximum, value, decimals=decimals)

    @staticmethod
    def _int(minimum, maximum, value) -> NumericLineEdit:
        return NumericLineEdit(minimum, maximum, value, decimals=0, integer=True)
