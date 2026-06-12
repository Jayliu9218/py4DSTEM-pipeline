from __future__ import annotations

from pathlib import Path
from dataclasses import replace

import h5py
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
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
from app.pages.preprocessing_page import PreprocessingPage
from app.pages.bragg_peaks_page import BraggPeaksPage
from app.pages.calibration_page import CalibrationPage
from app.pages.orientation_page import OrientationPage
from app.pages.strain_map_page import StrainMapPage
from app.pages.phase_contrast_page import PhaseContrastPage
from app.pages.bf_df_preview_page import BFDFPreviewPage
from app.pages.dpc_page import DPCPage
from app.pages.parallax_page import ParallaxPage
from app.pages.ptychography_page import PtychographyPage
from app.pages.method_comparison_page import MethodComparisonPage
from app.pages.structural_phase_page import StructuralPhasePage
from app.pages.radial_profile_page import RadialProfilePage
from app.pages.rdf_page import RDFPage
from app.pages.fem_page import FEMPage
from app.pages.amorphous_strain_page import AmorphousStrainPage
from app.services.phase_contrast_service import PhaseContrastResult, PhaseContrastService
from app.services.parallax_service import ParallaxService
from app.services.bragg_strain_service import BraggStrainService, BraggStrainServiceError
from app.services.hdf5_service import Hdf5Service
from app.services.project_state_service import ProjectState, ProjectStateService
from app.services.py4dstem_service import Py4DSTEMService, Py4DSTEMServiceError
from app.services.report_service import ReportService
from app.services.result_registry import ResultRegistry, ResultRegistryError
from app.services.workflow_state import WorkflowState, WorkflowStep
from app.widgets.hdf5_tree_widget import Hdf5TreeWidget
from app.widgets.image_viewer import ImageViewer
from app.widgets.adaptive_image_workspace import AdaptiveImageWorkspace
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
        self.preprocessing_page = PreprocessingPage(
            source_provider=self._get_py4dstem_datacube,
            selected_source_provider=self._get_selected_display_source,
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
        self.phase_contrast_page = PhaseContrastPage(
            source_provider=self._get_py4dstem_datacube,
            log_panel=self.log_panel,
            workflow_state=self.workflow_state,
            result_registry=self.result_registry,
        )
        self.bf_df_preview_page = BFDFPreviewPage(
            source_provider=self._get_py4dstem_datacube,
            shape_provider=self._get_current_4d_shape,
            probe_geometry_provider=self._get_probe_geometry,
            log_panel=self.log_panel,
            workflow_state=self.workflow_state,
            result_registry=self.result_registry,
        )
        self.dpc_service = PhaseContrastService()
        self.dpc_segmented_page = DPCPage(
            source_provider=self._get_py4dstem_datacube,
            log_panel=self.log_panel,
            workflow_state=self.workflow_state,
            result_registry=self.result_registry,
            service=self.dpc_service,
            stage_mode="segmented",
        )
        self.dpc_preprocess_page = DPCPage(
            source_provider=self._get_py4dstem_datacube,
            log_panel=self.log_panel,
            workflow_state=self.workflow_state,
            result_registry=self.result_registry,
            service=self.dpc_service,
            stage_mode="preprocess",
        )
        self.dpc_review_page = DPCPage(
            source_provider=self._get_py4dstem_datacube,
            log_panel=self.log_panel,
            workflow_state=self.workflow_state,
            result_registry=self.result_registry,
            service=self.dpc_service,
            stage_mode="review",
        )
        self.dpc_reconstruction_page = DPCPage(
            source_provider=self._get_py4dstem_datacube,
            log_panel=self.log_panel,
            workflow_state=self.workflow_state,
            result_registry=self.result_registry,
            service=self.dpc_service,
            stage_mode="reconstruct",
        )
        self.dpc_legacy_page = DPCPage(
            source_provider=self._get_py4dstem_datacube,
            log_panel=self.log_panel,
            workflow_state=self.workflow_state,
            result_registry=self.result_registry,
            service=self.dpc_service,
            stage_mode="all",
        )
        self.dpc_page = self.dpc_reconstruction_page
        self.parallax_service = ParallaxService()
        self.parallax_bf_page = ParallaxPage(
            source_provider=self._get_py4dstem_datacube,
            log_panel=self.log_panel,
            workflow_state=self.workflow_state,
            result_registry=self.result_registry,
            service=self.parallax_service,
            stage_mode="bf",
        )
        self.parallax_alignment_page = ParallaxPage(
            source_provider=self._get_py4dstem_datacube,
            log_panel=self.log_panel,
            workflow_state=self.workflow_state,
            result_registry=self.result_registry,
            service=self.parallax_service,
            stage_mode="alignment",
        )
        self.parallax_review_page = ParallaxPage(
            source_provider=self._get_py4dstem_datacube,
            log_panel=self.log_panel,
            workflow_state=self.workflow_state,
            result_registry=self.result_registry,
            service=self.parallax_service,
            stage_mode="review",
        )
        self.parallax_advanced_page = ParallaxPage(
            source_provider=self._get_py4dstem_datacube,
            log_panel=self.log_panel,
            workflow_state=self.workflow_state,
            result_registry=self.result_registry,
            service=self.parallax_service,
            stage_mode="advanced",
        )
        self.parallax_export_page = ParallaxPage(
            source_provider=self._get_py4dstem_datacube,
            log_panel=self.log_panel,
            workflow_state=self.workflow_state,
            result_registry=self.result_registry,
            service=self.parallax_service,
            stage_mode="export",
        )
        self.parallax_page = self.parallax_alignment_page
        self.ptychography_page = PtychographyPage(
            source_provider=self._get_py4dstem_datacube,
            log_panel=self.log_panel,
            workflow_state=self.workflow_state,
            result_registry=self.result_registry,
            dpc_result_provider=lambda: self.phase_retrieval_results.get("DPC"),
            parallax_result_provider=lambda: self.phase_retrieval_results.get("Parallax"),
        )
        self.method_comparison_page = MethodComparisonPage(
            dpc_result_provider=lambda: self.phase_retrieval_results.get("DPC"),
            parallax_result_provider=lambda: self.phase_retrieval_results.get("Parallax"),
            ptychography_result_provider=lambda: self.phase_retrieval_results.get("Ptychography"),
            log_panel=self.log_panel,
            workflow_state=self.workflow_state,
        )
        self.structural_phase_page = StructuralPhasePage()
        self.radial_profile_page = RadialProfilePage()
        self.rdf_page = RDFPage()
        self.fem_page = FEMPage()
        self.amorphous_strain_page = AmorphousStrainPage()
        self.phase_retrieval_results: dict[str, PhaseContrastResult] = {}

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
        self.dpc_reconstruction_page.dpc_result_ready.connect(self._store_dpc_result)
        self.dpc_legacy_page.dpc_result_ready.connect(self._store_dpc_result)
        self.parallax_review_page.parallax_result_ready.connect(self._store_parallax_result)
        self.parallax_advanced_page.parallax_result_ready.connect(self._store_parallax_result)
        self.ptychography_page.ptychography_result_ready.connect(self._store_ptychography_result)
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

        report_action = self.file_menu.addAction("Generate &Report")
        report_action.triggered.connect(self.generate_report)

        self.file_menu.addSeparator()

        exit_action = self.file_menu.addAction("E&xit")
        exit_action.triggered.connect(self.close)

        self.mode_menu = self.menuBar().addMenu("&Mode")
        self.crystalline_mode_action = self.mode_menu.addAction("Crystalline / Bragg-based")
        self.amorphous_mode_action = self.mode_menu.addAction("Amorphous / Diffuse-scattering")
        self.phase_retrieval_mode_action = self.mode_menu.addAction("Phase Retrieval / Ptychography")

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
        self.tree.setFrameShape(QFrame.NoFrame)
        self.tree.setStyleSheet("QTreeWidget { border: 1px solid black; }")
        data_browser.setMinimumWidth(220)
        data_browser.setMaximumWidth(500)
        data_browser.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        self.main_view = MultiViewWorkspace(self.scan_viewer, self.diffraction_viewer)
        self.viewer_stack = QStackedWidget()
        self.viewer_pages = {
            "overview": self.main_view,
            "preprocess": self.preprocessing_page,
            "virtual": self.virtual_detector_page,
            "bragg": self.bragg_peaks_page,
            "calibration": self.calibration_page,
            "orientation": self.orientation_page,
            "strain": self.strain_map_page,
            "structural_phase": self.structural_phase_page,
            "phase_contrast": self.phase_contrast_page,
            "bf_df": self.bf_df_preview_page,
            "dpc_segmented": self.dpc_segmented_page,
            "dpc_preprocess": self.dpc_preprocess_page,
            "dpc_review": self.dpc_review_page,
            "dpc": self.dpc_reconstruction_page,
            "dpc_legacy": self.dpc_legacy_page,
            "parallax_bf": self.parallax_bf_page,
            "parallax_alignment": self.parallax_alignment_page,
            "parallax_review": self.parallax_review_page,
            "parallax_advanced": self.parallax_advanced_page,
            "parallax_export": self.parallax_export_page,
            "parallax": self.parallax_alignment_page,
            "ptychography": self.ptychography_page,
            "method_comparison": self.method_comparison_page,
            "radial_profile": self.radial_profile_page,
            "rdf": self.rdf_page,
            "fem": self.fem_page,
            "amorphous_strain": self.amorphous_strain_page,
        }
        for page in self.viewer_pages.values():
            self.viewer_stack.addWidget(page)

        self.module_panel = ModuleControlPanel()
        self.module_panel.setFixedWidth(400)
        self.module_panel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.addWidget(data_browser)
        self.main_splitter.addWidget(self.viewer_stack)
        self.main_splitter.addWidget(self.module_panel)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setStretchFactor(2, 0)
        self.main_splitter.setCollapsible(0, False)
        self.main_splitter.setCollapsible(2, False)
        self.main_splitter.setHandleWidth(1)
        self.main_splitter.setStyleSheet("QSplitter::handle { background: black; }")
        self.main_splitter.setSizes([300, 900, 350])

        log_panel_widget = self.log_panel
        log_panel_widget.setFixedHeight(140)
        log_panel_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.project_toolbar = ProjectToolbar()
        self.route_bar = TechnicalRouteBar()
        self.data_setup_controls = self._build_role_panel()
        self.export_controls = self._build_export_panel()
        self.project_toolbar.structure_changed.connect(self._update_structure_route)
        self.project_toolbar.goal_changed.connect(self._update_structure_route)
        self.route_bar.module_selected.connect(self._select_route_module)
        self.crystalline_mode_action.triggered.connect(
            lambda: self.project_toolbar.structure.setCurrentText("Crystalline / Bragg-based")
        )
        self.amorphous_mode_action.triggered.connect(
            lambda: self.project_toolbar.structure.setCurrentText("Amorphous / Diffuse-scattering")
        )
        self.phase_retrieval_mode_action.triggered.connect(
            lambda: self.project_toolbar.structure.setCurrentText("Phase Retrieval / Ptychography")
        )

        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        central_layout.addWidget(self.project_toolbar)
        central_layout.addWidget(self.route_bar)

        self.workflow_divider = self._horizontal_divider("workflowDivider")
        central_layout.addWidget(self.workflow_divider)
        central_layout.addWidget(self.main_splitter, 1)

        self.log_divider = self._horizontal_divider("logDivider")
        central_layout.addWidget(self.log_divider)

        central_layout.addWidget(log_panel_widget, 0)
        self.setCentralWidget(central)

        self._set_index_controls_visible(False)
        self._compact_input_controls()
        self._bold_section_titles()
        self._set_preview_empty()

    def _horizontal_divider(self, object_name: str) -> QFrame:
        divider = QFrame()
        divider.setObjectName(object_name)
        divider.setFrameShape(QFrame.NoFrame)
        divider.setFixedHeight(1)
        divider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        divider.setStyleSheet("""
            QFrame {
                background: black;
                border: 0px;
                min-height: 1px;
                max-height: 1px;
            }
        """)
        return divider

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
            "Crystalline / Bragg-based": ["Orientation Mapping", "Strain Mapping", "Structural Phase Mapping"],
            "Amorphous / Diffuse-scattering": ["RDF", "FEM", "Amorphous Strain"],
            "Phase Retrieval / Ptychography": ["DPC / CoM", "Parallax", "Ptychography", "Method Comparison"],
        }[structure]
        if not self.project_toolbar.goal.count() or self.project_toolbar.goal.currentText() not in goals:
            self.project_toolbar.set_goals(goals)
        goal = self.project_toolbar.goal.currentText() or goals[0]
        notebook_strain_route = structure == "Crystalline / Bragg-based" and goal == "Strain Mapping"
        common_data = RouteModule(
            "data_setup",
            "Data & Preprocess" if notebook_strain_route else "Data Setup",
            "preprocess" if notebook_strain_route else "overview",
            "Open an HDF5 / EMD file, assign dataset roles, inspect the DataCube, and preview/apply preprocessing.",
            "Validated roles, DataCube diagnostics, and an explicitly preprocessed working DataCube.",
        )
        if structure == "Crystalline / Bragg-based":
            analysis_page = (
                "strain" if goal == "Strain Mapping"
                else "orientation" if goal == "Orientation Mapping"
                else "structural_phase" if goal == "Structural Phase Mapping"
                else "overview"
            )
            analysis_step = (
                WorkflowStep.STRAIN_MAP
                if goal == "Strain Mapping"
                else WorkflowStep.ORIENTATION_MATCH
                if goal == "Orientation Mapping"
                else WorkflowStep.STRUCTURAL_PHASE
                if goal == "Structural Phase Mapping"
                else None
            )
            analysis_implemented = goal != "Structural Phase Mapping"
            if goal == "Strain Mapping":
                modules = [
                    common_data,
                    RouteModule(
                        "virtual_imaging", "Virtual Imaging", "virtual",
                        "Target DataCube; measured probe geometry is optional.",
                        "BF, annular/off-axis DF, and ROI virtual diffraction.",
                        WorkflowStep.VIRTUAL_DETECTOR, "data_setup",
                    ),
                    RouteModule(
                        "bragg_detection", "Probe & Bragg", "bragg",
                        "Target DataCube; vacuum probe or vacuum ROI recommended.",
                        "Probe kernel, target/reference BraggVectors, histograms, and diagnostics.",
                        WorkflowStep.BRAGG_FULL, "virtual_imaging",
                    ),
                    RouteModule(
                        "calibration", "Calibration", "calibration",
                        "Target and ellipse-reference BraggVectors plus rotation reference.",
                        "Applied origin, ellipse, pixel-size, and rotation calibration.",
                        WorkflowStep.CALIBRATION_APPLY, "bragg_detection",
                    ),
                    RouteModule(
                        "crystal_analysis", "Strain Analysis", "strain",
                        "Target BraggVectors; calibration is recommended for accuracy.",
                        "Basis diagnostics, strain components, quality maps, and references.",
                        WorkflowStep.STRAIN_MAP, "bragg_detection",
                    ),
                    RouteModule(
                        "export", "Export & Report", "overview",
                        "At least one registered result.",
                        "Result arrays, project state, and scientific report.",
                        prerequisite="crystal_analysis",
                    ),
                ]
            else:
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
                        analysis_step, "calibration", analysis_implemented,
                    ),
                    RouteModule(
                        "export", "Export", "overview",
                        "At least one registered result.",
                        "Result arrays, images, project state, or scientific report.",
                        prerequisite="crystal_analysis",
                    ),
                ]
        elif structure == "Amorphous / Diffuse-scattering":
            goal_page = {
                "RDF": "rdf",
                "FEM": "fem",
                "Amorphous Strain": "amorphous_strain",
            }.get(goal, "overview")
            modules = [
                common_data,
                RouteModule("radial_profile", "Radial Profile", "radial_profile",
                    "Target DataCube for radial analysis.",
                    "Radial intensity profile and peak diagnostics.",
                    WorkflowStep.RADIAL_PROFILE, "data_setup", implemented=False),
                RouteModule("amorphous_analysis", goal, goal_page,
                    "Validated radial profile and analysis-specific parameters.",
                    f"{goal} maps and diagnostics.",
                    prerequisite="radial_profile", implemented=False),
                RouteModule("export", "Export", "overview",
                    "At least one registered result.",
                    "Result arrays, images, project state, or scientific report.",
                    prerequisite="amorphous_analysis"),
            ]
        elif structure == "Phase Retrieval / Ptychography":
            goal_page = {
                "DPC / CoM": "dpc",
                "Parallax": "parallax",
                "Ptychography": "ptychography",
                "Method Comparison": "method_comparison",
            }.get(goal, "dpc")
            goal_step = {
                "DPC / CoM": WorkflowStep.DPC,
                "Parallax": WorkflowStep.PARALLAX,
                "Ptychography": WorkflowStep.PTYCHOGRAPHY,
                "Method Comparison": WorkflowStep.METHOD_COMPARISON,
            }.get(goal)
            if goal == "DPC / CoM":
                modules = [
                    common_data,
                    RouteModule(
                        "bf_df_preview", "BF / DF Preview", "bf_df",
                        "Target DataCube for bright-field / dark-field virtual imaging.",
                        "BF and DF virtual images for phase retrieval workflow entry.",
                        WorkflowStep.BF_DF_PREVIEW, "data_setup",
                    ),
                    RouteModule(
                        "dpc_segmented", "Segmented DPC", "dpc_segmented",
                        "Target DataCube; BF/DF preview recommended for mask calibration.",
                        "Four masks, segment intensities, opposing DPC, and weighted CoM.",
                        WorkflowStep.DPC_SEGMENTED, "data_setup",
                    ),
                    RouteModule(
                        "dpc_preprocess", "CoM Preprocessing", "dpc_preprocess",
                        "Target DataCube; segmented DPC is an optional demonstration.",
                        "Measured, fitted, normalized, and corrected pixelated CoM.",
                        WorkflowStep.DPC_PREPROCESS, "data_setup",
                    ),
                    RouteModule(
                        "dpc_review", "CoM Review & Accept", "dpc_review",
                        "Completed pixelated CoM preprocessing.",
                        "Reviewed rotation, transpose, corrected CoM, and explicit acceptance.",
                        WorkflowStep.DPC_REVIEW, "dpc_preprocess",
                    ),
                    RouteModule(
                        "dpc", "Integrated Reconstruction", "dpc",
                        "Accepted CoM preprocessing review.",
                        "Integrated DPC potential, convergence, and stored iterations.",
                        WorkflowStep.DPC, "dpc_review",
                    ),
                    RouteModule(
                        "export", "Export", "overview",
                        "At least one registered result.",
                        "Intermediate and final DPC results, project state, and report.",
                        prerequisite="dpc",
                    ),
                ]
            elif goal == "Parallax":
                modules = [
                    common_data,
                    RouteModule(
                        "parallax_bf", "BF Disk & Virtual BF", "parallax_bf",
                        "Target DataCube.",
                        "Reviewed and explicitly accepted bright-field disk mask.",
                        WorkflowStep.PARALLAX_BF_ACCEPT, "data_setup",
                    ),
                    RouteModule(
                        "parallax_alignment", "Parallax Alignment", "parallax_alignment",
                        "Accepted BF disk mask.",
                        "Core cross-correlation Parallax alignment.",
                        WorkflowStep.PARALLAX_ALIGNMENT, "parallax_bf",
                    ),
                    RouteModule(
                        "parallax_review", "Alignment Review", "parallax_review",
                        "Completed Parallax alignment.",
                        "Reviewed shifts and explicitly accepted aligned BF result.",
                        WorkflowStep.PARALLAX_REVIEW, "parallax_alignment",
                    ),
                    RouteModule(
                        "parallax_advanced", "Advanced Reconstruction", "parallax_advanced",
                        "Accepted alignment review.",
                        "Optional subpixel reconstruction and expert aberration processing.",
                        WorkflowStep.PARALLAX_ADVANCED, "parallax_review",
                    ),
                    RouteModule(
                        "export", "Export", "parallax_export",
                        "Accepted core alignment or advanced reconstruction.",
                        "Registered results and an explicitly saved Parallax package.",
                        prerequisite="parallax_review",
                    ),
                ]
            else:
                modules = [
                    common_data,
                    RouteModule(
                        "bf_df_preview", "BF / DF Preview", "bf_df",
                        "Target DataCube for bright-field / dark-field virtual imaging.",
                        "BF and DF virtual images for phase retrieval workflow entry.",
                        WorkflowStep.BF_DF_PREVIEW, "data_setup",
                    ),
                    RouteModule(
                        "dpc", "DPC / CoM", "dpc_legacy",
                        "Target DataCube; DPC recommended for downstream initialization.",
                        "Integrated DPC result when available.",
                        WorkflowStep.DPC, "data_setup",
                    ),
                    RouteModule(
                        "parallax", "Parallax", "parallax",
                        "Target DataCube; DPC recommended for rotation and defocus estimates.",
                        "Aligned bright-field image, shift maps, and aberration estimates.",
                        WorkflowStep.PARALLAX, "dpc",
                    ),
                    RouteModule(
                        "ptychography", "Ptychography", "ptychography",
                        "Target DataCube and optional vacuum probe; Parallax recommended for initialization.",
                        "Phase and amplitude reconstruction from iterative ptychography.",
                        WorkflowStep.PTYCHOGRAPHY, "parallax",
                    ),
                    RouteModule(
                        "method_comparison", "Method Comparison", "method_comparison",
                        "At least one phase retrieval result (DPC, Parallax, or Ptychography).",
                        "Side-by-side comparison of phase retrieval method outputs.",
                        WorkflowStep.METHOD_COMPARISON, "ptychography",
                    ),
                    RouteModule(
                        "export", "Export", "overview",
                        "At least one registered result.",
                        "Result arrays, images, project state, or scientific report.",
                        prerequisite="method_comparison",
                    ),
                ]
        else:
            modules = [common_data]
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
        page = self.viewer_pages[module.page_key]
        refresh_stage = getattr(page, "refresh_stage", None)
        if callable(refresh_stage):
            refresh_stage()
        self.viewer_stack.setCurrentWidget(page)
        controls = self._controls_for_route(module.key)
        self.module_panel.set_module(module, controls)
        workspace = self._workspace_for_page(self.viewer_pages[module.page_key])
        if workspace is not None:
            workspace.refresh_layout()
        self._bold_section_titles()

    def _controls_for_route(self, key: str) -> QWidget | None:
        goal = self.project_toolbar.goal.currentText()
        return {
            "data_setup": self.data_setup_controls,
            "virtual_imaging": self.virtual_detector_page.controls_panel,
            "bragg_detection": self.bragg_peaks_page.controls_panel,
            "calibration": self.calibration_page.controls_panel,
            "crystal_analysis": (
                self.strain_map_page.controls_panel
                if goal == "Strain Mapping"
                else self.orientation_page.controls_panel
                if goal == "Orientation Mapping"
                else self.structural_phase_page.controls_panel
                if goal == "Structural Phase Mapping"
                else None
            ),
            "bf_df_preview": self.bf_df_preview_page.controls_panel,
            "dpc_segmented": self.dpc_segmented_page.controls_panel,
            "dpc_preprocess": self.dpc_preprocess_page.controls_panel,
            "dpc_review": self.dpc_review_page.controls_panel,
            "dpc": (
                self.dpc_reconstruction_page.controls_panel
                if goal == "DPC / CoM"
                else self.dpc_legacy_page.controls_panel
            ),
            "parallax_bf": self.parallax_bf_page.controls_panel,
            "parallax_alignment": self.parallax_alignment_page.controls_panel,
            "parallax_review": self.parallax_review_page.controls_panel,
            "parallax_advanced": self.parallax_advanced_page.controls_panel,
            "parallax": self.parallax_alignment_page.controls_panel,
            "ptychography": self.ptychography_page.controls_panel,
            "method_comparison": self.method_comparison_page.controls_panel,
            "radial_profile": self.radial_profile_page.controls_panel,
            "amorphous_analysis": self.amorphous_strain_page.controls_panel if goal == "Amorphous Strain" else (
                self.rdf_page.controls_panel if goal == "RDF" else self.fem_page.controls_panel
            ),
            "export": (
                self.parallax_export_page.controls_panel
                if goal == "Parallax"
                else self.export_controls
            ),
        }.get(key)

    def _build_role_panel(self) -> QWidget:
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
        layout.addWidget(self.preprocessing_page.controls_panel)
        layout.addStretch(1)
        layout.addWidget(roles_group)
        layout.addStretch(1)
        return panel

    def _build_export_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        export_button = QPushButton("Export Registered Result")
        export_button.clicked.connect(self.export_registered_result)
        save_button = QPushButton("Save Project")
        save_button.clicked.connect(self.save_project)
        report_button = QPushButton("Generate Report")
        report_button.clicked.connect(self.generate_report)
        layout.addWidget(export_button)
        layout.addWidget(save_button)
        layout.addWidget(report_button)
        layout.addStretch(1)
        return panel

    def export_registered_result(self) -> None:
        entries = self.result_registry.list_entries()
        if not entries:
            QMessageBox.information(
                self, "Export Results", "No results are available yet. Run a workflow step first."
            )
            return
        labels = [entry.key for entry in entries]
        key, ok = QInputDialog.getItem(self, "Export Results", "Result", labels, 0, False)
        if not ok or not key:
            return
        entry = self.result_registry.get(key)
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export result",
            str(self._default_output_dir() / self._safe_result_filename(entry.name, entry.export_formats[0])),
            ";;".join(self._filters_for_entry(entry.export_formats)),
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

    def _populate_sidebar_controls(self) -> None:
        for page in [
            self.virtual_detector_page,
            self.bragg_peaks_page,
            self.calibration_page,
            self.orientation_page,
            self.strain_map_page,
            self.phase_contrast_page,
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
                widget.setMinimumWidth(0)
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
            self._clear_all_image_workspaces()
            self.bragg_strain_service.braggvectors = None
            self.bragg_strain_service.strainmap = None
            self.bragg_strain_service.strain_result = None
            self.bragg_strain_service.probe_kernel = None
            self.dpc_service.reset_dpc_workflow()
            self.parallax_service.reset()
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
            state = replace(self._project_state(), result_entries=result_entries)
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
        self.phase_retrieval_results.clear()
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

    def _get_selected_display_source(self):
        path = self._current_tree_selection_path()
        if path and self.current_file is not None:
            try:
                node = self.current_file[path]
                if isinstance(node, h5py.Dataset):
                    return node
            except Exception:
                pass
            try:
                return self.py4dstem_service.read_datapath(path)
            except Exception:
                pass
        target_path = self.workflow_state.dataset_roles.target_datacube
        if target_path and target_path != path:
            try:
                return self.py4dstem_service.read_datapath(target_path)
            except Exception:
                pass
        return self._get_virtual_detector_source()

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

    def _store_dpc_result(self, result: PhaseContrastResult) -> None:
        self.phase_retrieval_results["DPC"] = result
        self.log_panel.log("DPC result stored for method comparison")

    def _store_parallax_result(self, result: PhaseContrastResult) -> None:
        self.phase_retrieval_results["Parallax"] = result
        self.log_panel.log("Parallax result stored for method comparison")

    def _store_ptychography_result(self, result: PhaseContrastResult) -> None:
        self.phase_retrieval_results["Ptychography"] = result
        self.log_panel.log("Ptychography result stored for method comparison")

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
        selected_path = self._current_tree_selection_path()
        if selected_path is None:
            QMessageBox.information(self, "Dataset Roles", "Select a node in the HDF5 tree first.")
            return
        self.selected_hdf5_path = selected_path
        previous_target = self.workflow_state.dataset_roles.target_datacube
        self.workflow_state.set_dataset_role(role, selected_path)
        if role == "target_datacube" and previous_target != selected_path:
            self.dpc_service.reset_dpc_workflow()
            self.parallax_service.reset()
            if previous_target is None:
                self._clear_all_image_workspaces(exclude_keys={"preprocess"})
            else:
                self._clear_all_image_workspaces()
                self.result_registry.clear()
        self._refresh_role_labels()
        self.log_panel.log(f"Assigned {role}: {selected_path}")

    def _current_tree_selection_path(self) -> str | None:
        item = self.tree.currentItem()
        if item is not None:
            path = item.data(0, 256)
            kind = item.data(0, 257)
            if path and kind:
                return str(path)
        return self.selected_hdf5_path or self.current_dataset_path

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
                "preprocessing": self.preprocessing_page.params_snapshot(),
                "bragg_peaks": self.bragg_peaks_page.params_snapshot(),
                "calibration": self.calibration_page.params_snapshot(),
                "orientation": self.orientation_page.params_snapshot(),
                "strain_map": self.strain_map_page.params_snapshot(),
                "phase_contrast": self.phase_contrast_page.params_snapshot(),
                "bf_df_preview": self.bf_df_preview_page.params_snapshot(),
                "dpc": self._dpc_params_snapshot(),
                "dpc_segmented": self.dpc_segmented_page.params_snapshot(),
                "dpc_preprocess": self.dpc_preprocess_page.params_snapshot(),
                "dpc_review": self.dpc_review_page.params_snapshot(),
                "dpc_reconstruction": self.dpc_reconstruction_page.params_snapshot(),
                "dpc_legacy": self.dpc_legacy_page.params_snapshot(),
                "parallax": self.parallax_page.params_snapshot(),
                "parallax_bf": self.parallax_bf_page.params_snapshot(),
                "parallax_alignment": self.parallax_alignment_page.params_snapshot(),
                "parallax_review": self.parallax_review_page.params_snapshot(),
                "parallax_advanced": self.parallax_advanced_page.params_snapshot(),
                "parallax_export": self.parallax_export_page.params_snapshot(),
                "ptychography": self.ptychography_page.params_snapshot(),
                "method_comparison": self.method_comparison_page.params_snapshot(),
            },
            grid_states=self._grid_states(),
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
        for key, grid_state in state.grid_states.items():
            workspace = self._named_workspaces().get(key)
            if workspace is not None:
                workspace.restore_grid_state(grid_state)
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
            ("preprocessing", self.preprocessing_page),
            ("bragg_peaks", self.bragg_peaks_page),
            ("calibration", self.calibration_page),
            ("orientation", self.orientation_page),
            ("strain_map", self.strain_map_page),
            ("bf_df_preview", self.bf_df_preview_page),
            ("ptychography", self.ptychography_page),
        ]:
            params = page_params.get(key)
            if params:
                page.apply_params_snapshot(params)
        legacy_parallax = page_params.get("parallax", {})
        for key, page in [
            ("parallax_bf", self.parallax_bf_page),
            ("parallax_alignment", self.parallax_alignment_page),
            ("parallax_review", self.parallax_review_page),
            ("parallax_advanced", self.parallax_advanced_page),
            ("parallax_export", self.parallax_export_page),
        ]:
            params = page_params.get(key) or legacy_parallax
            if params:
                page.apply_params_snapshot(params)
        legacy_dpc = page_params.get("dpc", {})
        for key, page in [
            ("dpc_segmented", self.dpc_segmented_page),
            ("dpc_preprocess", self.dpc_preprocess_page),
            ("dpc_review", self.dpc_review_page),
            ("dpc_reconstruction", self.dpc_reconstruction_page),
            ("dpc_legacy", self.dpc_legacy_page),
        ]:
            params = page_params.get(key) or legacy_dpc
            if params:
                page.apply_params_snapshot(params)

    def _dpc_params_snapshot(self) -> dict[str, object]:
        snapshot: dict[str, object] = {}
        for page in (
            self.dpc_segmented_page,
            self.dpc_preprocess_page,
            self.dpc_review_page,
            self.dpc_reconstruction_page,
            self.dpc_legacy_page,
        ):
            snapshot.update(page.params_snapshot())
        return snapshot

    def _default_output_dir(self) -> Path:
        if self.recent_export_dir is not None:
            return self.recent_export_dir
        if self.current_file_path is not None:
            return self.current_file_path.parent
        return Path.cwd()

    def _named_workspaces(self) -> dict[str, AdaptiveImageWorkspace]:
        workspaces: dict[str, AdaptiveImageWorkspace] = {}
        for key, page in self.viewer_pages.items():
            workspace = self._workspace_for_page(page)
            if workspace is not None:
                workspaces[key] = workspace
        return workspaces

    def _workspace_for_page(self, page: QWidget) -> AdaptiveImageWorkspace | None:
        if isinstance(page, AdaptiveImageWorkspace):
            return page
        for attribute in ("workspace", "viewers", "selected_grid"):
            workspace = getattr(page, attribute, None)
            if isinstance(workspace, AdaptiveImageWorkspace):
                return workspace
        return page.findChild(AdaptiveImageWorkspace)

    def _grid_states(self) -> dict[str, dict[str, object]]:
        return {key: workspace.grid_state() for key, workspace in self._named_workspaces().items()}

    def _clear_all_image_workspaces(self, exclude_keys: set[str] | None = None) -> None:
        excluded = set(exclude_keys or ())
        cleared: set[int] = set()
        for key, page in self.viewer_pages.items():
            if key in excluded:
                continue
            if id(page) in cleared:
                continue
            cleared.add(id(page))
            clear_results = getattr(page, "clear_results", None)
            if callable(clear_results):
                clear_results()
                continue
            workspace = self._workspace_for_page(page)
            if workspace is not None:
                workspace.clear_results()

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
