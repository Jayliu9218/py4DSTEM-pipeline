from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.pages.virtual_detector_page import VirtualDetectorPage
from app.pages.bragg_peaks_page import BraggPeaksPage
from app.pages.calibration_page import CalibrationPage
from app.pages.orientation_page import OrientationPage
from app.pages.strain_map_page import StrainMapPage
from app.services.bragg_strain_service import BraggStrainService, BraggStrainServiceError
from app.services.hdf5_service import Hdf5Service
from app.services.project_state_service import ProjectState, ProjectStateService
from app.services.py4dstem_service import Py4DSTEMService, Py4DSTEMServiceError
from app.services.report_service import ReportService
from app.services.result_registry import ResultRegistry, ResultRegistryError
from app.services.workflow_state import WorkflowState, WorkflowStep
from app.widgets.hdf5_tree_widget import Hdf5TreeWidget
from app.widgets.image_viewer import ImageViewer
from app.widgets.log_panel import LogPanel
from app.widgets.numeric_line_edit import NumericLineEdit
from app.widgets.pipeline_shell import (
    ModuleControlPanel,
    MultiViewWorkspace,
    ProjectToolbar,
    RouteModule,
    TechnicalRouteBar,
)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("py4DSTEM Pipeline")
        self.resize(1600, 900)

        self.hdf5_service = Hdf5Service()
        self.py4dstem_service = Py4DSTEMService()
        self.bragg_strain_service = BraggStrainService()
        self.workflow_state = WorkflowState()
        self.project_state_service = ProjectStateService()
        self.result_registry = ResultRegistry()
        self.report_service = ReportService()
        self.current_file: h5py.File | None = None
        self.current_file_path: Path | None = None
        self.current_dataset_path: str | None = None
        self.current_dataset_shape: tuple[int, ...] | None = None
        self.current_4d_source: str | None = None
        self.selected_hdf5_path: str | None = None
        self.selected_node_kind: str | None = None
        self.current_attrs: dict[str, object] = {}
        self.image_scaling = ImageViewer.DEFAULT_SCALING
        self.image_cmap = ImageViewer.DEFAULT_CMAP
        self.cuda_enabled = False
        self.recent_export_dir: Path | None = None
        self.braggvectors_by_datacube: dict[str, object] = {}
        self.reference_braggvectors_cache: dict[str, object] = {}
        self.current_route_key = "data_setup"
        self.route_modules: list[RouteModule] = []

        self.tree = Hdf5TreeWidget()
        self.scan_viewer = ImageViewer()
        self.diffraction_viewer = ImageViewer()
        self.log_panel = LogPanel()
        self.virtual_detector_page = VirtualDetectorPage(
            source_provider=self._get_virtual_detector_source,
            shape_provider=self._get_current_4d_shape,
            probe_geometry_provider=self._get_probe_geometry,
            log_panel=self.log_panel,
            workflow_state=self.workflow_state,
            result_registry=self.result_registry,
        )
        self.bragg_peaks_page = BraggPeaksPage(
            datacube_provider=self._get_py4dstem_datacube,
            shape_provider=self._get_current_4d_shape,
            virtual_image_provider=self._get_virtual_detector_image,
            service=self.bragg_strain_service,
            log_panel=self.log_panel,
            workflow_state=self.workflow_state,
            result_registry=self.result_registry,
        )
        self.calibration_page = CalibrationPage(
            datacube_provider=self._get_py4dstem_datacube,
            braggvectors_provider=self._get_braggvectors,
            ellipse_braggvectors_provider=self._get_ellipse_reference_braggvectors,
            transfer_targets_provider=self._get_calibration_transfer_targets,
            rotation_reference_provider=self._get_rotation_reference_image,
            service=self.bragg_strain_service,
            log_panel=self.log_panel,
            workflow_state=self.workflow_state,
            result_registry=self.result_registry,
        )
        self.strain_map_page = StrainMapPage(
            braggvectors_provider=self._get_braggvectors,
            service=self.bragg_strain_service,
            log_panel=self.log_panel,
            workflow_state=self.workflow_state,
            result_registry=self.result_registry,
        )
        self.orientation_page = OrientationPage(
            braggvectors_provider=self._get_braggvectors,
            log_panel=self.log_panel,
            workflow_state=self.workflow_state,
            result_registry=self.result_registry,
        )

        self.datacube_name_label = QLabel("-")
        self.scan_shape_label = QLabel("-")
        self.diffraction_shape_label = QLabel("-")
        self.role_labels = {
            "target_datacube": QLabel("-"),
            "polycrystal_calibration": QLabel("-"),
            "vacuum_probe": QLabel("-"),
            "defocused_cbed": QLabel("-"),
        }
        self.path_label = QLabel("-")
        self.type_label = QLabel("-")
        self.shape_label = QLabel("-")
        self.dtype_label = QLabel("-")
        self.attrs_table = QTableWidget(0, 2)
        self.attrs_table.setHorizontalHeaderLabels(["Attribute", "Value"])
        self.attrs_table.horizontalHeader().setStretchLastSection(True)

        self.rx_spin = NumericLineEdit(0, 100000, 0, decimals=0, unit="px", integer=True)
        self.ry_spin = NumericLineEdit(0, 100000, 0, decimals=0, unit="px", integer=True)
        self.rx_spin.valueChanged.connect(self._refresh_current_4d_image)
        self.ry_spin.valueChanged.connect(self._refresh_current_4d_image)

        self._build_menu()
        self._build_layout()
        self._apply_cuda_setting(self.cuda_enabled)
        self.setWindowState(self.windowState() | Qt.WindowMaximized)

        self.tree.node_selected.connect(self._handle_node_selected)
        self.scan_viewer.image_clicked.connect(self._handle_scan_image_clicked)
        self.bragg_peaks_page.braggvectors_ready.connect(self._store_current_braggvectors)
        self.bragg_peaks_page.braggvectors_ready.connect(self.calibration_page.refresh_status)
        self.bragg_peaks_page.braggvectors_ready.connect(self.strain_map_page.notify_braggvectors_ready)
        self.bragg_peaks_page.braggvectors_ready.connect(self.calibration_page.show_braggvectors_histogram)
        self.virtual_detector_page.virtual_image_ready.connect(self.bragg_peaks_page.set_virtual_image)
        self.virtual_detector_page.virtual_image_ready.connect(self._show_virtual_image_in_scan_viewer)
        self.workflow_state.changed.connect(self._refresh_pipeline_state)
        self.log_panel.log("Application started.")
        self._apply_image_scaling(self.image_scaling)
        self._apply_image_colormap(self.image_cmap)
        self._update_structure_route()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API name
        self._close_current_file()
        event.accept()

    def _build_menu(self) -> None:
        self.file_menu = self.menuBar().addMenu("&Files")

        open_action = self.file_menu.addAction("&Open")
        open_action.triggered.connect(self.open_file)

        self.file_menu.addSeparator()

        save_project_action = self.file_menu.addAction("&Save Project")
        save_project_action.triggered.connect(self.save_project)

        load_project_action = self.file_menu.addAction("&Load Project")
        load_project_action.triggered.connect(self.load_project)

        self.file_menu.addSeparator()

        export_results_action = self.file_menu.addAction("Export &Results")
        export_results_action.triggered.connect(self.export_registered_result)

        report_action = self.file_menu.addAction("Generate &Report")
        report_action.triggered.connect(self.generate_report)

        self.file_menu.addSeparator()

        exit_action = self.file_menu.addAction("E&xit")
        exit_action.triggered.connect(self.close)

        self.mode_menu = self.menuBar().addMenu("&Mode")
        self.crystalline_mode_action = self.mode_menu.addAction("Crystalline")
        self.amorphous_mode_action = self.mode_menu.addAction("Amorphous")
        self.mixed_mode_action = self.mode_menu.addAction("Mixed / Nanocrystalline")

        self.setting_action = self.menuBar().addAction("&Setting")
        self.setting_action.triggered.connect(self.open_settings)

        self.help_menu = self.menuBar().addMenu("&Help")
        self.help_menu.addAction("About")
        self.help_menu.addAction("License")
        self.help_menu.addAction("Tutorials")

    def open_settings(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Settings")
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        scaling_combo = QComboBox()
        scaling_combo.addItems(["log", "linear"])
        scaling_combo.setCurrentText(self.image_scaling)
        cmap_combo = QComboBox()
        cmap_combo.addItems(["gray", "viridis", "magma", "plasma", "inferno", "cividis"])
        cmap_combo.setCurrentText(self.image_cmap)
        cuda_check = QCheckBox("Enable CUDA when supported")
        cuda_check.setChecked(self.cuda_enabled)
        form.addRow("Image scaling", scaling_combo)
        form.addRow("Colormap", cmap_combo)
        form.addRow("CUDA", cuda_check)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QDialog.Accepted:
            return
        self.image_scaling = scaling_combo.currentText()
        self.image_cmap = cmap_combo.currentText()
        self.cuda_enabled = cuda_check.isChecked()
        self._apply_image_scaling(self.image_scaling)
        self._apply_image_colormap(self.image_cmap)
        self._apply_cuda_setting(self.cuda_enabled)
        self.log_panel.log(
            f"Image scaling set to {self.image_scaling}; colormap={self.image_cmap}; "
            f"CUDA {'on' if self.cuda_enabled else 'off'}."
        )

    def _apply_image_scaling(self, scaling: str) -> None:
        ImageViewer.DEFAULT_SCALING = scaling
        for viewer in self.findChildren(ImageViewer):
            viewer.set_scaling(scaling)

    def _apply_image_colormap(self, colormap: str) -> None:
        ImageViewer.DEFAULT_CMAP = colormap
        for viewer in self.findChildren(ImageViewer):
            viewer.set_colormap(colormap)

    def _apply_cuda_setting(self, enabled: bool) -> None:
        self.cuda_enabled = enabled
        self.bragg_peaks_page.set_cuda_enabled(enabled)
        self.orientation_page.set_cuda_enabled(enabled)

    def _build_layout(self) -> None:
        data_browser = QWidget()
        data_browser_layout = QVBoxLayout(data_browser)
        data_browser_layout.setContentsMargins(8, 8, 8, 8)
        data_title = QLabel("Data Tree")
        data_title.setObjectName("sectionTitle")
        data_browser_layout.addWidget(data_title)
        data_browser_layout.addWidget(self.tree, 1)
        data_browser.setFixedWidth(250)
        data_browser.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

        self.main_view = MultiViewWorkspace(self.scan_viewer, self.diffraction_viewer)
        self.viewer_stack = QStackedWidget()
        self.viewer_pages = {
            "overview": self.main_view,
            "virtual": self.virtual_detector_page,
            "bragg": self.bragg_peaks_page,
            "calibration": self.calibration_page,
            "orientation": self.orientation_page,
            "strain": self.strain_map_page,
        }
        for page in self.viewer_pages.values():
            self.viewer_stack.addWidget(page)

        self.module_panel = ModuleControlPanel()
        self.module_panel.setFixedWidth(400)
        self.module_panel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(data_browser)
        main_splitter.addWidget(self.viewer_stack)
        main_splitter.addWidget(self.module_panel)
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setStretchFactor(2, 0)
        main_splitter.setSizes([250, 900, 400])

        log_panel_widget = self.log_panel
        log_panel_widget.setFixedHeight(180)
        log_panel_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        workspace = QSplitter(Qt.Vertical)
        workspace.addWidget(main_splitter)
        workspace.addWidget(log_panel_widget)
        workspace.setStretchFactor(0, 1)
        workspace.setStretchFactor(1, 0)
        workspace.setSizes([720, 180])

        self.project_toolbar = ProjectToolbar()
        self.route_bar = TechnicalRouteBar()
        self.data_setup_controls = self._build_role_panel()
        self.project_toolbar.load_clicked.connect(self.open_file)
        self.project_toolbar.project_clicked.connect(self.load_project)
        self.project_toolbar.save_clicked.connect(self.save_project)
        self.project_toolbar.export_clicked.connect(self.export_registered_result)
        self.project_toolbar.structure_changed.connect(self._update_structure_route)
        self.project_toolbar.goal_changed.connect(self._update_structure_route)
        self.route_bar.module_selected.connect(self._select_route_module)
        self.crystalline_mode_action.triggered.connect(
            lambda: self.project_toolbar.structure.setCurrentText("Crystalline")
        )
        self.amorphous_mode_action.triggered.connect(
            lambda: self.project_toolbar.structure.setCurrentText("Amorphous")
        )
        self.mixed_mode_action.triggered.connect(
            lambda: self.project_toolbar.structure.setCurrentText("Mixed / Nanocrystalline")
        )

        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        central_layout.addWidget(self.project_toolbar)
        central_layout.addWidget(self.route_bar)
        central_layout.addWidget(workspace, 1)
        self.setCentralWidget(central)

        self._set_index_controls_visible(False)
        self._compact_input_controls()
        self._bold_section_titles()
        self._set_preview_empty()

    def _bold_section_titles(self) -> None:
        for group in self.findChildren(QGroupBox):
            font = group.font()
            font.setBold(True)
            group.setFont(font)
            for child in group.findChildren(QWidget):
                if isinstance(child, QGroupBox):
                    continue
                child_font = child.font()
                child_font.setBold(False)
                child.setFont(child_font)

    def _update_structure_route(self, *_args) -> None:
        structure = self.project_toolbar.structure.currentText()
        goals = {
            "Crystalline": ["Strain", "Orientation", "Phase"],
            "Amorphous": ["RDF", "Amorphous Strain", "FEM"],
            "Mixed / Nanocrystalline": ["Branch Analysis", "Component Mapping", "Region Review"],
        }[structure]
        if not self.project_toolbar.goal.count() or self.project_toolbar.goal.currentText() not in goals:
            self.project_toolbar.set_goals(goals)
        goal = self.project_toolbar.goal.currentText() or goals[0]
        common_data = RouteModule(
            "data_setup",
            "Data Setup",
            "overview",
            "Open an HDF5 / EMD file, assign the Target DataCube, and configure virtual imaging.",
            "Validated DataCube, dataset roles, virtual image preview, and display-ready outputs.",
        )
        if structure == "Crystalline":
            analysis_page = "strain" if goal == "Strain" else "orientation" if goal == "Orientation" else "overview"
            analysis_step = (
                WorkflowStep.STRAIN_MAP
                if goal == "Strain"
                else WorkflowStep.ORIENTATION_MATCH
                if goal == "Orientation"
                else None
            )
            modules = [
                common_data,
                RouteModule(
                    "bragg_detection", "Bragg Detection", "bragg",
                    "Target DataCube; probe kernel is optional but recommended.",
                    "BraggVectors, BVM, peak tables, and detection diagnostics.",
                    WorkflowStep.BRAGG_FULL, "data_setup",
                ),
                RouteModule(
                    "calibration", "Calibration", "calibration",
                    "BraggVectors; ellipse and rotation references when required.",
                    "Applied origin, ellipse, pixel-size, and rotation calibration.",
                    WorkflowStep.CALIBRATION_APPLY, "bragg_detection",
                ),
                RouteModule(
                    "crystal_analysis", goal, analysis_page,
                    "Calibrated BraggVectors and analysis-specific reference inputs.",
                    f"{goal} result maps and quality diagnostics.",
                    analysis_step, "calibration", goal != "Phase",
                ),
                RouteModule(
                    "export", "Export", "overview",
                    "At least one registered result.",
                    "Result arrays, images, project state, or scientific report.",
                    prerequisite="crystal_analysis",
                ),
            ]
        elif structure == "Amorphous":
            modules = [
                common_data,
                RouteModule("ring_centering", "Ring Centering", "overview", "Target DataCube or averaged diffraction pattern.", "Centered ring pattern and center-fit diagnostics.", prerequisite="data_setup", implemented=False),
                RouteModule("radial_profile", "Radial Profile", "overview", "Centered diffraction rings.", "Radial intensity profile and peak diagnostics.", prerequisite="ring_centering", implemented=False),
                RouteModule("amorphous_analysis", goal, "overview", "Validated radial profile and analysis-specific parameters.", f"{goal} maps and diagnostics.", prerequisite="radial_profile", implemented=False),
                RouteModule("export", "Export", "overview", "At least one registered result.", "Result arrays, images, project state, or scientific report.", prerequisite="amorphous_analysis"),
            ]
        else:
            modules = [
                common_data,
                RouteModule("feature_extraction", "Feature Extraction", "overview", "Target DataCube and selected feature families.", "Feature stack and feature-quality metrics.", prerequisite="data_setup", implemented=False),
                RouteModule("region_classification", "Region Classification", "overview", "Extracted features and classification settings.", "Region labels and confidence maps.", prerequisite="feature_extraction", implemented=False),
                RouteModule("branch_analysis", f"Branch Analysis: {goal}", "overview", "Classified regions and branch-specific inputs.", "Per-region crystalline/amorphous analysis outputs.", prerequisite="region_classification", implemented=False),
                RouteModule("component_map", "Component Map", "overview", "Completed branch analysis.", "Component map, uncertainty, and export-ready layers.", prerequisite="branch_analysis", implemented=False),
            ]
        self.route_modules = modules
        self.route_bar.set_modules(modules)
        if self.current_route_key not in {module.key for module in modules}:
            self.current_route_key = "data_setup"
        self._refresh_pipeline_state()

    def _route_states(self) -> dict[str, str]:
        states: dict[str, str] = {}
        data_ready = self.current_dataset_shape is not None and len(self.current_dataset_shape) == 4
        for module in self.route_modules:
            if module.key == "data_setup":
                states[module.key] = "Ready" if data_ready else "Ready"
                continue
            if module.state_step and self.workflow_state.is_stale(module.state_step):
                states[module.key] = "Warning"
            elif module.state_step and self.workflow_state.is_completed(module.state_step):
                states[module.key] = "Completed"
            else:
                states[module.key] = "Ready"
        return states

    def _select_route_module(self, key: str) -> None:
        self.current_route_key = key
        self._refresh_pipeline_state()

    def _refresh_pipeline_state(self) -> None:
        if not hasattr(self, "route_bar"):
            return
        states = self._route_states()
        self.route_bar.update_states(states, self.current_route_key)
        module = next(item for item in self.route_modules if item.key == self.current_route_key)
        self.viewer_stack.setCurrentWidget(self.viewer_pages[module.page_key])
        controls = self._controls_for_route(module.key)
        self.module_panel.set_module(module, controls)
        self._bold_section_titles()

    def _controls_for_route(self, key: str) -> QWidget | None:
        return {
            "data_setup": self.data_setup_controls,
            "bragg_detection": self.bragg_peaks_page.controls_panel,
            "calibration": self.calibration_page.controls_panel,
            "crystal_analysis": (
                self.strain_map_page.controls_panel
                if self.project_toolbar.goal.currentText() == "Strain"
                else self.orientation_page.controls_panel
                if self.project_toolbar.goal.currentText() == "Orientation"
                else None
            ),
        }.get(key)

    def _build_role_panel(self) -> QWidget:
        virtual_controls = self.virtual_detector_page.controls_panel
        
        roles_group = QGroupBox("Dataset Roles / Sources")
        roles_layout = QVBoxLayout(roles_group)
        for label, role in [
            ("Set as Target", "target_datacube"),
            ("Set as Vacuum Probe", "vacuum_probe"),
            ("Set as Ellipse Ref", "polycrystal_calibration"),
            ("Set as Rotation Ref", "defocused_cbed"),
        ]:
            button = QPushButton(label)
            button.clicked.connect(lambda _checked=False, role=role: self._assign_current_role(role))
            roles_layout.addWidget(button)
        
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(virtual_controls)
        layout.addWidget(roles_group)
        layout.addStretch(1)
        return panel

    def _populate_sidebar_controls(self) -> None:
        for page in [
            self.virtual_detector_page,
            self.bragg_peaks_page,
            self.calibration_page,
            self.orientation_page,
            self.strain_map_page,
        ]:
            self.sidebar_controls.addWidget(self._wrap_sidebar_panel(page.controls_panel))

    def _wrap_sidebar_panel(self, widget: QWidget) -> QWidget:
        if isinstance(widget, QScrollArea):
            return widget
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(widget)
        return scroll

    def _compact_input_controls(self) -> None:
        for widget_type in (NumericLineEdit, QComboBox):
            for widget in self.findChildren(widget_type):
                widget.setMinimumWidth(125)
                widget.setMaximumWidth(320)

    def open_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open HDF5 or EMD file",
            "",
            "HDF5/EMD files (*.h5 *.hdf5 *.emd);;All files (*.*)",
        )
        if not file_path:
            return

        self._open_file_path(file_path)

    def _open_file_path(self, file_path: str) -> None:
        try:
            self._close_current_file()
            self.current_file = self.hdf5_service.open_file(file_path)
            self.current_file_path = Path(file_path)
            self.result_registry.clear()
            self.braggvectors_by_datacube.clear()
            self.reference_braggvectors_cache.clear()
            self.tree.populate(self.current_file)
            self._set_preview_empty()
            self._clear_dataset_info()
            self.virtual_detector_page.viewer.clear()
            self.bragg_strain_service.braggvectors = None
            self.bragg_strain_service.strainmap = None
            self.bragg_strain_service.strain_result = None
            self.bragg_strain_service.probe_kernel = None
            self.workflow_state.data_source_updated()
            self.py4dstem_service.defer_open_file(file_path)
            self.log_panel.log(f"Opened file: {file_path}")
            self.log_panel.log(
                "Opened in safe HDF5 mode. py4DSTEM import is deferred so startup and browsing "
                "do not trigger native library errors."
            )
        except Exception as exc:
            self.current_file = None
            self.current_file_path = None
            self.log_panel.log(f"Failed to open file: {exc}")
            QMessageBox.critical(self, "Open failed", str(exc))

    def save_project(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save project",
            str(self._default_output_dir() / "project.json"),
            "Project JSON (*.json)",
        )
        if not path:
            return
        output_path = Path(path)
        if output_path.suffix.lower() != ".json":
            output_path = output_path.with_suffix(".json")
        try:
            result_data = {}
            result_entries = []
            for entry in self.result_registry.list_entries():
                result_data[entry.key] = entry.data
                result_entries.append({
                    "key": entry.key,
                    "name": entry.name,
                    "category": entry.category,
                    "export_formats": list(entry.export_formats),
                    "metadata": {str(k): str(v) for k, v in entry.metadata.items()},
                })
            state = ProjectState(
                file_path=str(self.current_file_path) if self.current_file_path else None,
                selected_hdf5_path=self.selected_hdf5_path,
                image_scaling=self.image_scaling,
                image_cmap=self.image_cmap,
                cuda_enabled=self.cuda_enabled,
                recent_export_dir=str(self.recent_export_dir) if self.recent_export_dir else None,
                dataset_roles={
                    "target_datacube": roles.target_datacube,
                    "polycrystal_calibration": roles.polycrystal_calibration,
                    "vacuum_probe": roles.vacuum_probe,
                    "defocused_cbed": roles.defocused_cbed,
                },
                page_params={
                    "virtual_detector": self.virtual_detector_page.params_snapshot(),
                    "bragg_peaks": self.bragg_peaks_page.params_snapshot(),
                    "calibration": self.calibration_page.params_snapshot(),
                    "orientation": self.orientation_page.params_snapshot(),
                    "strain_map": self.strain_map_page.params_snapshot(),
                },
                result_entries=result_entries,
            )
            if result_data:
                self.project_state_service.save_with_results(output_path, state, result_data)
            else:
                self.project_state_service.save(output_path, state)
            self.recent_export_dir = output_path.parent
            self.log_panel.log(f"Project saved: {output_path}")
        except Exception as exc:
            self.log_panel.log(f"Project save failed: {exc}")
            QMessageBox.warning(self, "Save Project", str(exc))

    def load_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load project",
            str(self._default_output_dir()),
            "Project JSON (*.json);;All files (*.*)",
        )
        if not path:
            return
        try:
            state = self.project_state_service.load(path)
            self._apply_project_state(state)
            if state.result_entries:
                results = self.project_state_service.load_results(path, state.result_entries)
                for entry_info in state.result_entries:
                    key = entry_info.get("key", "")
                    if key and key in results:
                        self.result_registry.register(
                            name=entry_info.get("name", key),
                            category=entry_info.get("category", ""),
                            data=results[key],
                            export_formats=tuple(entry_info.get("export_formats", ("npy",))),
                            metadata=entry_info.get("metadata", {}),
                        )
            self.recent_export_dir = Path(path).parent
            self.log_panel.log(f"Project loaded: {path}")
        except Exception as exc:
            self.log_panel.log(f"Project load failed: {exc}")
            QMessageBox.warning(self, "Load Project", str(exc))

    def export_registered_result(self) -> None:
        entries = self.result_registry.list_entries()
        if not entries:
            QMessageBox.information(
                self,
                "Export Results",
                "No results are available yet. Run a workflow step first.",
            )
            return
        labels = [entry.key for entry in entries]
        key, ok = QInputDialog.getItem(
            self,
            "Export Results",
            "Result",
            labels,
            0,
            False,
        )
        if not ok or not key:
            return
        entry = self.result_registry.get(key)
        filters = self._filters_for_entry(entry.export_formats)
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export result",
            str(self._default_output_dir() / self._safe_result_filename(entry.name, entry.export_formats[0])),
            ";;".join(filters),
        )
        if not path:
            return
        output_path = self._path_with_supported_suffix(Path(path), entry.export_formats[0])
        try:
            exported = self.result_registry.export(key, output_path)
            self.recent_export_dir = exported.parent
            self.log_panel.log(f"Result exported: {exported}")
        except ResultRegistryError as exc:
            self.log_panel.log(f"Result export failed: {exc}")
            QMessageBox.warning(self, "Export Results", str(exc))

    def generate_report(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Generate report",
            str(self._default_output_dir() / "py4dstem_report.md"),
            "Markdown (*.md)",
        )
        if not path:
            return
        output_path = Path(path)
        if output_path.suffix.lower() != ".md":
            output_path = output_path.with_suffix(".md")
        try:
            self.report_service.generate_markdown(
                output_path,
                self._project_state(),
                self.result_registry,
                self.log_panel.event_log.toPlainText(),
                self.log_panel.process_log.toPlainText(),
            )
            self.recent_export_dir = output_path.parent
            self.log_panel.log(f"Report generated: {output_path}")
        except Exception as exc:
            self.log_panel.log(f"Report generation failed: {exc}")
            QMessageBox.warning(self, "Generate Report", str(exc))

    def _handle_node_selected(self, hdf5_path: str, node_kind: str) -> None:
        if self.current_file is None:
            return

        self.log_panel.log(f"Selected {node_kind}: {hdf5_path}")
        self.selected_hdf5_path = hdf5_path
        self.selected_node_kind = node_kind
        self.current_dataset_path = None
        self.current_dataset_shape = None
        self.current_4d_source = None
        self._set_index_controls_visible(False)

        try:
            node = self.current_file[hdf5_path]
            info = self.hdf5_service.describe_node(node, hdf5_path)
            self._show_node_info(info)

            if node_kind == "group":
                if self._try_load_py4dstem_datacube(hdf5_path, show_warning=False):
                    return
                displayed = self._try_display_first_dataset(node, hdf5_path)
                if not displayed:
                    self._set_preview_empty("Select a 4D dataset or py4DSTEM DataCube group.")
                    self._clear_datacube_info()
                return

            if node_kind != "dataset":
                self._set_preview_empty("Select a displayable dataset from the HDF5 tree.")
                return

            self.current_dataset_path = hdf5_path
            shape = tuple(int(dim) for dim in node.shape)
            self.current_dataset_shape = shape

            if len(shape) == 2:
                image = self.hdf5_service.read_2d_dataset(node)
                self.scan_viewer.clear()
                self.diffraction_viewer.set_image(image)
                self._clear_datacube_info()
                self.log_panel.log(f"Displayed 2D image: {hdf5_path} shape={shape}")
            elif len(shape) == 4:
                if not self._try_load_py4dstem_datacube(hdf5_path, show_warning=False):
                    self._load_raw_4d_dataset(hdf5_path, shape)
                self._configure_4d_controls(shape)
                self._display_4d_slice(rx=0, ry=0)
            else:
                self._set_preview_empty("This dataset is not displayable as a 2D image or 4D DataCube.")
                self._clear_datacube_info()
                self.log_panel.log(f"Dataset is not displayable as an image: shape={shape}")
        except Exception as exc:
            self._set_preview_empty("Could not display this node.")
            self.log_panel.log(f"Failed to inspect node: {exc}")

    def _try_display_first_dataset(self, group: h5py.Group, group_path: str) -> bool:
        for name in sorted(group.keys()):
            child_path = f"{group_path}/{name}" if group_path != "/" else f"/{name}"
            child = group[name]
            if not isinstance(child, h5py.Dataset):
                continue
            shape = tuple(int(dim) for dim in child.shape)
            if len(shape) == 2:
                try:
                    image = self.hdf5_service.read_2d_dataset(child)
                except Exception:
                    continue
                self.scan_viewer.clear()
                self.diffraction_viewer.set_image(image)
                self._clear_datacube_info()
                self.current_dataset_path = child_path
                self.current_dataset_shape = shape
                self.selected_hdf5_path = child_path
                self.selected_node_kind = "dataset"
                self.log_panel.log(f"Auto-displayed 2D dataset: {child_path} shape={shape}")
                return True
            if len(shape) == 4:
                if self._try_load_py4dstem_datacube(child_path, show_warning=True):
                    return True
                try:
                    self._load_raw_4d_dataset(child_path, shape)
                    self._configure_4d_controls(shape)
                    self._display_4d_slice(rx=0, ry=0)
                    return True
                except Exception:
                    continue
        return False

    def _show_node_info(self, info: dict[str, object]) -> None:
        self.path_label.setText(str(info.get("path", "-")))
        self.type_label.setText(str(info.get("type", "-")))
        self.shape_label.setText(str(info.get("shape", "-")))
        self.dtype_label.setText(str(info.get("dtype", "-")))

        attrs = info.get("attrs", {})
        if not isinstance(attrs, dict):
            attrs = {}
        self.current_attrs = dict(attrs)

        self.attrs_table.setRowCount(len(attrs))
        for row, (key, value) in enumerate(attrs.items()):
            self.attrs_table.setItem(row, 0, QTableWidgetItem(str(key)))
            self.attrs_table.setItem(row, 1, QTableWidgetItem(str(value)))
        self.attrs_table.resizeColumnsToContents()
        self._refresh_tree_data_info(attrs=attrs)

    def _clear_dataset_info(self) -> None:
        self._clear_datacube_info()
        self.path_label.setText("-")
        self.type_label.setText("-")
        self.shape_label.setText("-")
        self.dtype_label.setText("-")
        self.attrs_table.setRowCount(0)
        self.current_dataset_path = None
        self.current_dataset_shape = None
        self.current_4d_source = None
        self._set_index_controls_visible(False)
        self.selected_hdf5_path = None
        self.selected_node_kind = None
        self.current_attrs = {}
        self._refresh_tree_data_info()

    def _clear_datacube_info(self) -> None:
        self.datacube_name_label.setText("-")
        self.scan_shape_label.setText("-")
        self.diffraction_shape_label.setText("-")
        self._refresh_tree_data_info()

    def _configure_4d_controls(self, shape: tuple[int, ...]) -> None:
        self.rx_spin.blockSignals(True)
        self.ry_spin.blockSignals(True)
        self.rx_spin.setMaximum(max(shape[0] - 1, 0))
        self.ry_spin.setMaximum(max(shape[1] - 1, 0))
        self.rx_spin.setValue(0)
        self.ry_spin.setValue(0)
        self.rx_spin.blockSignals(False)
        self.ry_spin.blockSignals(False)
        self._set_index_controls_visible(True)

    def _set_index_controls_visible(self, visible: bool) -> None:
        self.rx_spin.setEnabled(visible)
        self.ry_spin.setEnabled(visible)

    def _refresh_current_4d_image(self) -> None:
        if self.current_dataset_path is None or self.current_dataset_shape is None:
            return
        if len(self.current_dataset_shape) != 4:
            return
        self._display_4d_slice(self.rx_spin.value(), self.ry_spin.value())

    def _display_4d_slice(self, rx: int, ry: int) -> None:
        try:
            if self.current_4d_source == "py4dstem":
                image = self.py4dstem_service.get_diffraction_pattern(rx, ry)
                self.diffraction_viewer.set_image(image)
                info = self.py4dstem_service.describe_current_datacube()
                datapath = info.get("datapath", "DataCube")
                self.log_panel.log(f"Displayed py4DSTEM diffraction pattern: {datapath}[{rx}, {ry}]")
            elif self.current_4d_source == "hdf5":
                if self.current_file is None or self.current_dataset_path is None:
                    return
                dataset = self.current_file[self.current_dataset_path]
                image = self.hdf5_service.read_4d_diffraction_pattern(dataset, rx=rx, ry=ry)
                self.diffraction_viewer.set_image(image)
                self.log_panel.log(
                    f"Displayed HDF5 diffraction pattern: {self.current_dataset_path}[{rx}, {ry}, :, :]"
                )
            else:
                return
            self._refresh_tree_data_info()
        except Exception as exc:
            self.log_panel.log(f"Failed to display diffraction pattern: {exc}")
            QMessageBox.warning(self, "Diffraction pattern error", str(exc))

    def _try_load_py4dstem_datacube(self, hdf5_path: str, show_warning: bool = True) -> bool:
        try:
            info = self.py4dstem_service.load_datacube(hdf5_path)
            self.bragg_strain_service.braggvectors = None
            self.bragg_strain_service.strainmap = None
            self.bragg_strain_service.strain_result = None
            self.bragg_strain_service.probe_kernel = None
            self.workflow_state.data_source_updated()
            scan_image = self.py4dstem_service.get_scan_image()
            self.scan_viewer.set_image(scan_image)
            self.current_4d_source = "py4dstem"
            self.current_dataset_path = hdf5_path
            self.current_dataset_shape = info.shape
            self._restore_current_braggvectors()
            self._show_datacube_info(info.name, info.scan_shape, info.diffraction_shape)
            try:
                geometry = self.py4dstem_service.measure_probe_geometry()
                self.log_panel.log(
                    "Measured probe geometry: "
                    f"radius={geometry.radius:.3g}, center=({geometry.center_x:.3g}, {geometry.center_y:.3g})."
                )
            except Py4DSTEMServiceError as exc:
                self.log_panel.log(str(exc))
            self._configure_4d_controls(info.shape)
            self._display_4d_slice(0, 0)
            self.virtual_detector_page.refresh_defaults_from_datacube()
            self.bragg_peaks_page.refresh_from_datacube()
            self.calibration_page.refresh_status()
            self.log_panel.log(f"Loaded py4DSTEM DataCube: {info.name} at {hdf5_path}")
            return True
        except Py4DSTEMServiceError as exc:
            self.current_4d_source = None
            self.current_dataset_path = None
            self.current_dataset_shape = None
            self.log_panel.log(str(exc))
            if show_warning:
                QMessageBox.information(self, "py4DSTEM", str(exc))
            return False

    def _load_raw_4d_dataset(self, hdf5_path: str, shape: tuple[int, ...]) -> None:
        if self.current_file is None:
            return

        dataset = self.current_file[hdf5_path]
        info = self.py4dstem_service.load_raw_4d_array(dataset, hdf5_path)
        self.workflow_state.data_source_updated()
        scan_image = self.hdf5_service.read_4d_scan_image(dataset)
        self.scan_viewer.set_image(scan_image)
        self.current_4d_source = "hdf5"
        self.current_dataset_path = hdf5_path
        self.current_dataset_shape = shape
        self._restore_current_braggvectors()
        self._show_datacube_info(info.name, info.scan_shape, info.diffraction_shape)
        self.virtual_detector_page.refresh_defaults_from_datacube()
        self.log_panel.log(f"Loaded raw 4D HDF5 dataset: {hdf5_path}")

    def _show_datacube_info(
        self,
        name: str,
        scan_shape: tuple[int, int],
        diffraction_shape: tuple[int, int],
    ) -> None:
        self.datacube_name_label.setText(name)
        self.scan_shape_label.setText(str(scan_shape))
        self.diffraction_shape_label.setText(str(diffraction_shape))
        self._refresh_tree_data_info()

    def _handle_scan_image_clicked(self, x: int, y: int) -> None:
        if self.current_dataset_shape is None or len(self.current_dataset_shape) != 4:
            return

        rx = min(max(x, 0), self.current_dataset_shape[0] - 1)
        ry = min(max(y, 0), self.current_dataset_shape[1] - 1)

        self.rx_spin.blockSignals(True)
        self.ry_spin.blockSignals(True)
        self.rx_spin.setValue(rx)
        self.ry_spin.setValue(ry)
        self.rx_spin.blockSignals(False)
        self.ry_spin.blockSignals(False)

        self.log_panel.log(f"Scan image clicked: rx={rx}, ry={ry}")
        self._display_4d_slice(rx, ry)

    def _show_virtual_image_in_scan_viewer(self, image) -> None:
        try:
            import numpy as np
            self.scan_viewer.set_image(np.asarray(image))
        except Exception:
            pass

    def _close_current_file(self) -> None:
        if self.current_file is not None:
            try:
                self.current_file.close()
                if self.current_file_path is not None:
                    self.log_panel.log(f"Closed file: {self.current_file_path}")
            except Exception as exc:
                self.log_panel.log(f"Failed to close file cleanly: {exc}")
        self.current_file = None
        self.current_file_path = None
        self.py4dstem_service.close()
        self.bragg_strain_service.braggvectors = None
        self.bragg_strain_service.strainmap = None
        self.bragg_strain_service.strain_result = None
        self.bragg_strain_service.probe_kernel = None
        self.braggvectors_by_datacube.clear()
        self.reference_braggvectors_cache.clear()
        self.workflow_state.set_dataset_role("target_datacube", None)
        self.workflow_state.set_dataset_role("polycrystal_calibration", None)
        self.workflow_state.set_dataset_role("vacuum_probe", None)
        self.workflow_state.set_dataset_role("defocused_cbed", None)
        self._refresh_role_labels()

    def _get_virtual_detector_source(self):
        if self.current_4d_source == "py4dstem":
            return self.py4dstem_service.datacube
        if self.current_4d_source == "hdf5" and self.current_file is not None and self.current_dataset_path:
            return self.current_file[self.current_dataset_path]
        return None

    def _get_virtual_detector_image(self):
        return self.virtual_detector_page.result

    def _get_py4dstem_datacube(self):
        if self.current_4d_source == "py4dstem":
            return self.py4dstem_service.datacube
        return None

    def _get_braggvectors(self):
        if self.current_dataset_path and self.current_dataset_path in self.braggvectors_by_datacube:
            return self.braggvectors_by_datacube[self.current_dataset_path]
        return self.bragg_strain_service.braggvectors

    def _get_ellipse_reference_braggvectors(self):
        role_path = self.workflow_state.dataset_roles.polycrystal_calibration
        if not role_path or role_path == self.current_dataset_path:
            return self.bragg_strain_service.braggvectors
        if role_path in self.reference_braggvectors_cache:
            return self.reference_braggvectors_cache[role_path]

        datacube = self.py4dstem_service.read_datapath(role_path)
        params = self.bragg_peaks_page.bragg_detection_params()
        result = BraggStrainService().compute_braggvectors(datacube, params)
        self.reference_braggvectors_cache[role_path] = result.braggvectors
        self.result_registry.register(
            "ellipse reference bragg vector map",
            "Calibration",
            result.bragg_vector_map,
            ("npy", "png", "tiff"),
            {"source": role_path, "peak_count": result.peak_count},
        )
        return result.braggvectors

    def _get_calibration_transfer_targets(self) -> list[tuple[str, object]]:
        targets: list[tuple[str, object]] = []
        for path, braggvectors in self.braggvectors_by_datacube.items():
            if path == self.current_dataset_path:
                continue
            targets.append((path, braggvectors))
        for path, braggvectors in self.reference_braggvectors_cache.items():
            if path == self.current_dataset_path:
                continue
            if any(path == existing_path for existing_path, _ in targets):
                continue
            targets.append((path, braggvectors))
        return targets

    def _get_rotation_reference_image(self):
        images: dict[str, np.ndarray] = {}
        target_image = self._get_target_bright_field_image()
        if target_image is not None:
            images["target DataCube bright-field"] = target_image

        role_path = self.workflow_state.dataset_roles.defocused_cbed
        if not role_path:
            return images or None
        if self.current_file is not None:
            try:
                node = self.current_file[role_path]
                if isinstance(node, h5py.Dataset) and len(tuple(node.shape)) == 2:
                    images["rotation reference bright-field"] = np.asarray(node[...])
                    return images
            except Exception:
                pass
        obj = self.py4dstem_service.read_datapath(role_path)
        image = np.asarray(getattr(obj, "data", obj))
        if image.ndim == 2:
            images["rotation reference bright-field"] = image
            return images
        raise BraggStrainServiceError(
            f"Rotation Reference CBED must be a 2D diffraction image, got shape {image.shape}."
        )

    def _get_target_bright_field_image(self) -> np.ndarray | None:
        try:
            if self.current_4d_source == "py4dstem":
                return self.py4dstem_service.get_scan_image()
            if (
                self.current_4d_source == "hdf5"
                and self.current_file is not None
                and self.current_dataset_path
            ):
                node = self.current_file[self.current_dataset_path]
                if isinstance(node, h5py.Dataset) and len(tuple(node.shape)) == 4:
                    return self.hdf5_service.read_4d_scan_image(node)
        except Exception:
            return None
        return None

    def _get_probe_geometry(self):
        return self.py4dstem_service.probe_geometry

    def _store_current_braggvectors(self) -> None:
        if self.current_dataset_path is None or self.bragg_strain_service.braggvectors is None:
            return
        self.braggvectors_by_datacube[self.current_dataset_path] = self.bragg_strain_service.braggvectors
        self.log_panel.log(f"BraggVectors stored for DataCube: {self.current_dataset_path}")

    def _restore_current_braggvectors(self) -> None:
        if self.current_dataset_path is None:
            self.bragg_strain_service.braggvectors = None
            return
        self.bragg_strain_service.braggvectors = self.braggvectors_by_datacube.get(
            self.current_dataset_path
        )

    def _get_current_4d_shape(self) -> tuple[int, int, int, int] | None:
        if self.current_dataset_shape is None or len(self.current_dataset_shape) != 4:
            return None
        return self.current_dataset_shape

    def _assign_current_role(self, role: str) -> None:
        if self.selected_hdf5_path is None:
            QMessageBox.information(self, "Dataset Roles", "Select a node in the HDF5 tree first.")
            return
        self.workflow_state.set_dataset_role(role, self.selected_hdf5_path)
        self._refresh_role_labels()
        self.log_panel.log(f"Assigned {role}: {self.selected_hdf5_path}")

    def _refresh_role_labels(self) -> None:
        roles = self.workflow_state.dataset_roles
        for role, label in self.role_labels.items():
            label.setText(getattr(roles, role) or "-")
        self._refresh_tree_data_info()

    def _refresh_tree_data_info(self, attrs: dict[str, object] | None = None) -> None:
        if attrs is not None:
            self.current_attrs = dict(attrs)
        roles = self.workflow_state.dataset_roles
        self.tree.set_data_info(
            datacube={
                "DataCube": self.datacube_name_label.text(),
                "Scan shape": self.scan_shape_label.text(),
                "Diffraction shape": self.diffraction_shape_label.text(),
            },
            selection={
                "Path": self.path_label.text(),
                "Type": self.type_label.text(),
                "Shape": self.shape_label.text(),
                "Dtype": self.dtype_label.text(),
                "rx": self.rx_spin.value() if self.rx_spin.isEnabled() else "-",
                "ry": self.ry_spin.value() if self.ry_spin.isEnabled() else "-",
            },
            roles={
                "Target DataCube": roles.target_datacube or "-",
                "Ellipse Reference": roles.polycrystal_calibration or "-",
                "Vacuum Probe": roles.vacuum_probe or "-",
                "Rotation Reference CBED": roles.defocused_cbed or "-",
            },
            attrs=self.current_attrs,
        )

    def _set_preview_empty(self, message: str | None = None) -> None:
        base = message or "No DataCube loaded. Open an HDF5 file and select a 4D-STEM dataset."
        self.scan_viewer.clear("Mean real-space image / virtual bright field preview")
        self.diffraction_viewer.clear(base)

    def _project_state(self) -> ProjectState:
        roles = self.workflow_state.dataset_roles
        return ProjectState(
            file_path=str(self.current_file_path) if self.current_file_path else None,
            selected_hdf5_path=self.selected_hdf5_path,
            image_scaling=self.image_scaling,
            image_cmap=self.image_cmap,
            cuda_enabled=self.cuda_enabled,
            recent_export_dir=str(self.recent_export_dir) if self.recent_export_dir else None,
            dataset_roles={
                "target_datacube": roles.target_datacube,
                "polycrystal_calibration": roles.polycrystal_calibration,
                "vacuum_probe": roles.vacuum_probe,
                "defocused_cbed": roles.defocused_cbed,
            },
            page_params={
                "virtual_detector": self.virtual_detector_page.params_snapshot(),
                "bragg_peaks": self.bragg_peaks_page.params_snapshot(),
                "calibration": self.calibration_page.params_snapshot(),
                "orientation": self.orientation_page.params_snapshot(),
                "strain_map": self.strain_map_page.params_snapshot(),
            },
        )

    def _apply_project_state(self, state: ProjectState) -> None:
        if state.file_path and Path(state.file_path).exists():
            self._open_file_path(state.file_path)
        self.image_scaling = state.image_scaling
        self.image_cmap = state.image_cmap
        self.cuda_enabled = state.cuda_enabled
        self._apply_image_scaling(self.image_scaling)
        self._apply_image_colormap(self.image_cmap)
        self._apply_cuda_setting(self.cuda_enabled)
        if state.recent_export_dir:
            self.recent_export_dir = Path(state.recent_export_dir)
        for role, value in state.dataset_roles.items():
            self.workflow_state.set_dataset_role(role, value)
        self._refresh_role_labels()
        self._apply_page_params(state.page_params)
        if self.current_file is not None and state.selected_hdf5_path:
            try:
                node = self.current_file[state.selected_hdf5_path]
                node_kind = "dataset" if isinstance(node, h5py.Dataset) else "group"
                self._handle_node_selected(state.selected_hdf5_path, node_kind)
            except Exception as exc:
                self.log_panel.log(f"Saved HDF5 selection is unavailable: {exc}")

    def _apply_page_params(self, page_params: dict[str, dict[str, object]]) -> None:
        for key, page in [
            ("virtual_detector", self.virtual_detector_page),
            ("bragg_peaks", self.bragg_peaks_page),
            ("calibration", self.calibration_page),
            ("orientation", self.orientation_page),
            ("strain_map", self.strain_map_page),
        ]:
            params = page_params.get(key)
            if params:
                page.apply_params_snapshot(params)

    def _default_output_dir(self) -> Path:
        if self.recent_export_dir is not None:
            return self.recent_export_dir
        if self.current_file_path is not None:
            return self.current_file_path.parent
        return Path.cwd()

    def _filters_for_entry(self, formats: tuple[str, ...]) -> list[str]:
        labels = {
            "npy": "NumPy array (*.npy)",
            "npz": "NumPy archive (*.npz)",
            "png": "PNG image (*.png)",
            "tiff": "TIFF image (*.tif *.tiff)",
        }
        return [labels[item] for item in formats if item in labels]

    def _safe_result_filename(self, name: str, extension: str) -> str:
        safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in name)
        suffix = "tif" if extension == "tiff" else extension
        return f"{safe}.{suffix}"

    def _path_with_supported_suffix(self, path: Path, extension: str) -> Path:
        suffix = "tif" if extension == "tiff" else extension
        if path.suffix:
            return path
        return path.with_suffix(f".{suffix}")
