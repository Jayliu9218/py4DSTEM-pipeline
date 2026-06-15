from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
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
    QTextBrowser,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from app.controllers.route_coordinator import RouteCoordinator
from app.controllers.application_pages import ApplicationPages
from app.controllers.project_coordinator import ProjectCoordinator
from app.controllers.data_session_controller import DataSessionController
from app.theme import Theme
from app.services.phase_contrast_service import PhaseContrastResult
from app.services.bragg_strain_service import BraggStrainService, BraggStrainServiceError
from app.services.hdf5_service import Hdf5Service
from app.services.project_state_service import ProjectState, ProjectStateService
from app.services.py4dstem_service import Py4DSTEMService, Py4DSTEMServiceError
from app.services.report_service import ReportService
from app.services.result_registry import ResultRegistry
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
    ABOUT_HTML = """
    <h2>py4DSTEM Pipeline</h2>
    <p>A desktop workflow application for browsing, processing, reconstructing,
    reviewing, and exporting 4D-STEM data with py4DSTEM.</p>
    <h3>Current situation</h3>
    <p>The application provides guided crystalline and phase-retrieval workflows,
    shared calculation progress, project-state persistence, scientific diagnostics,
    and result export. The amorphous-analysis routes are visible but remain under
    development.</p>
    <h3>Current improvements</h3>
    <ul>
      <li>Staged workflows with explicit review and acceptance gates.</li>
      <li>CPU/GPU execution choices and clearer CUDA or memory failure guidance.</li>
      <li>Thread-safe background calculations with live progress reporting.</li>
      <li>Reusable Ptychography profiles, Quick Reconstruction, QC, and Advanced Reconstruction.</li>
    </ul>
    <p>Results should always be reviewed using appropriate experimental knowledge;
    automated diagnostics support scientific judgment but do not replace it.</p>
    """

    LICENSE_HTML = """
    <h2>License</h2>
    <p>This project is intended for distribution under the
    <b>GNU General Public License version 3 (GPLv3)</b>.</p>
    <p>py4DSTEM is open source software distributed under a GPLv3 license. It is
    free to use, alter, or build on, provided that any work derived from py4DSTEM
    is also kept free and open under a GPLv3 license.</p>
    <p>Reference:
    <a href="https://www.gnu.org/licenses/gpl-3.0.html">GNU GPLv3 license</a><br>
    py4DSTEM:
    <a href="https://github.com/py4dstem/py4DSTEM">github.com/py4dstem/py4DSTEM</a></p>
    """

    TUTORIAL_HTML = """
    <h2>Workflow Tutorial</h2>
    <p>Start by opening an HDF5/EMD file, assigning the Target DataCube and any
    optional reference roles, then choose a structure type and analysis goal.</p>
    <h3>Shared Data Setup</h3>
    <p>Inspect the DataCube, assign dataset roles, preview preprocessing, and apply
    corrections explicitly before downstream analysis.</p>
    <h3>Crystalline / Bragg-based</h3>
    <p><b>Orientation Mapping:</b> detect Bragg peaks, calibrate reciprocal space,
    load a crystal structure, create an orientation plan, and match orientations.</p>
    <p><b>Strain Mapping:</b> generate virtual images, prepare a probe kernel,
    calculate BraggVectors, apply calibration, select a reference, and calculate
    strain and quality maps.</p>
    <p><b>Structural Phase Mapping:</b> uses calibrated Bragg information for
    phase-specific analysis; this route is still being expanded.</p>
    <h3>Phase Retrieval / Ptychography</h3>
    <p><b>DPC / CoM:</b> preview BF/DF contrast, inspect segmented DPC, preprocess
    and accept CoM fields, then run integrated reconstruction.</p>
    <p><b>Parallax:</b> accept a bright-field disk, align virtual BF images, review
    shifts, and optionally run subpixel or aberration processing.</p>
    <p><b>Ptychography:</b> inspect data and probe, accept geometry and preprocessing,
    run a Quick Reconstruction, review QC, optionally optimize parameters, then run
    Advanced Reconstruction and export the retained results.</p>
    <p><b>Method Comparison:</b> compare retained DPC and Ptychography results when
    both are available.</p>
    <h3>Amorphous / Diffuse-scattering</h3>
    <p>Radial Profile, RDF, FEM, and Amorphous Strain routes are planned workflows
    and are not yet production-ready.</p>
    <h3>Reading Workflow Status</h3>
    <p>Completed stages are retained. Changing upstream parameters marks affected
    downstream results as stale. Re-run and re-accept stale stages before relying
    on later results.</p>
    """

    current_file = property(
        lambda self: self.session.current_file,
        lambda self, value: setattr(self.session, "current_file", value),
    )
    current_file_path = property(
        lambda self: self.session.current_file_path,
        lambda self, value: setattr(self.session, "current_file_path", value),
    )
    current_dataset_path = property(
        lambda self: self.session.current_dataset_path,
        lambda self, value: setattr(self.session, "current_dataset_path", value),
    )
    current_dataset_shape = property(
        lambda self: self.session.current_dataset_shape,
        lambda self, value: setattr(self.session, "current_dataset_shape", value),
    )
    current_4d_source = property(
        lambda self: self.session.current_4d_source,
        lambda self, value: setattr(self.session, "current_4d_source", value),
    )
    selected_hdf5_path = property(
        lambda self: self.session.selected_hdf5_path,
        lambda self, value: setattr(self.session, "selected_hdf5_path", value),
    )
    selected_node_kind = property(
        lambda self: self.session.selected_node_kind,
        lambda self, value: setattr(self.session, "selected_node_kind", value),
    )
    current_attrs = property(
        lambda self: self.session.current_attrs,
        lambda self, value: setattr(self.session, "current_attrs", value),
    )
    raw_scan_image_cache_path = property(
        lambda self: self.session.raw_scan_image_cache_path,
        lambda self, value: setattr(self.session, "raw_scan_image_cache_path", value),
    )
    raw_scan_image_cache = property(
        lambda self: self.session.raw_scan_image_cache,
        lambda self, value: setattr(self.session, "raw_scan_image_cache", value),
    )
    braggvectors_by_datacube = property(lambda self: self.session.braggvectors_by_datacube)
    reference_braggvectors_cache = property(lambda self: self.session.reference_braggvectors_cache)

    def __init__(self, progress_callback=None) -> None:
        # Optional coarse progress reporter (used by the startup splash).
        self._progress = progress_callback or (lambda _message: None)
        super().__init__()
        self.setWindowTitle("py4DSTEM Pipeline")
        self.resize(1600, 900)

        self._progress("Initializing services…")
        self.hdf5_service = Hdf5Service()
        self.py4dstem_service = Py4DSTEMService()
        self.session = DataSessionController(self.hdf5_service, self.py4dstem_service)
        self.bragg_strain_service = BraggStrainService()
        self.workflow_state = WorkflowState()
        self.project_state_service = ProjectStateService()
        self.result_registry = ResultRegistry()
        self.report_service = ReportService()
        self.image_scaling = ImageViewer.DEFAULT_SCALING
        self.image_cmap = ImageViewer.DEFAULT_CMAP
        self.cuda_enabled = False
        self.recent_export_dir: Path | None = None
        self.current_route_key = "data_setup"
        self.route_modules: list[RouteModule] = []
        self._initial_layout_applied = False
        self.selected_preview_kind = "Not displayable"
        self.selected_preview_shape: tuple[int, ...] | None = None
        self.preview_status = "Not rendered"

        self.tree = Hdf5TreeWidget()
        self.scan_viewer = ImageViewer()
        self.diffraction_viewer = ImageViewer()
        self.log_panel = LogPanel()
        self.phase_retrieval_results: dict[str, PhaseContrastResult] = {}
        self._progress("Building analysis pages…")
        page_objects, self.dpc_service, self.parallax_service = ApplicationPages.build_page_objects(
            providers={
                "virtual_source": self._get_virtual_detector_source,
                "shape": self._get_current_4d_shape,
                "probe_geometry": self._get_probe_geometry,
                "show_data_source": self._get_show_data_source,
                "datacube": self._get_py4dstem_datacube,
                "vacuum_probe_path": self._get_vacuum_probe_source,
                "virtual_image": self._get_virtual_detector_image,
                "braggvectors": self._get_braggvectors,
                "ellipse_braggvectors": self._get_ellipse_reference_braggvectors,
                "transfer_targets": self._get_calibration_transfer_targets,
                "rotation_reference": self._get_rotation_reference_image,
            },
            bragg_strain_service=self.bragg_strain_service,
            log_panel=self.log_panel,
            workflow_state=self.workflow_state,
            result_registry=self.result_registry,
            phase_retrieval_results=self.phase_retrieval_results,
        )
        for name, page in page_objects.items():
            setattr(self, name, page)

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
        self.rx_spin.valueChanged.connect(lambda _value: self._refresh_tree_data_info())
        self.ry_spin.valueChanged.connect(lambda _value: self._refresh_tree_data_info())
        self.preview_button = QPushButton("Preview Selected")
        self.preview_button.setEnabled(False)
        self.preview_button.clicked.connect(self._preview_selected_node)

        self._progress("Laying out workspace…")
        self._build_menu()
        self._build_layout()
        self._build_status_bar()
        self.pages = ApplicationPages(
            viewer_pages=self.viewer_pages,
            route_controls={
                "data_setup": self.data_setup_controls,
                "virtual_imaging": self.virtual_detector_page.controls_panel,
                "bragg_detection": self.bragg_peaks_page.controls_panel,
                "calibration": self.calibration_page.controls_panel,
                "orientation_setup": self.orientation_setup_page.controls_panel,
                "crystalline_results": self.crystalline_results_page.controls_panel,
                "bf_df_preview": self.bf_df_preview_page.controls_panel,
                "dpc_segmented": self.dpc_segmented_page.controls_panel,
                "dpc_preprocess": self.dpc_preprocess_page.controls_panel,
                "parallax_bf": self.parallax_bf_page.controls_panel,
                "parallax_alignment": self.parallax_alignment_page.controls_panel,
                "parallax_review": self.parallax_review_page.controls_panel,
                "parallax_advanced": self.parallax_advanced_page.controls_panel,
                "parallax": self.parallax_alignment_page.controls_panel,
                "ptychography": self.ptychography_page.controls_panel,
                "ptychography_data": self.ptychography_data_page.controls_panel,
                "ptychography_geometry": self.ptychography_geometry_page.controls_panel,
                "ptychography_preprocess": self.ptychography_preprocess_page.controls_panel,
                "ptychography_quick": self.ptychography_quick_page.controls_panel,
                "ptychography_review": self.ptychography_review_page.controls_panel,
                "ptychography_optimization": self.ptychography_optimization_page.controls_panel,
                "ptychography_advanced": self.ptychography_advanced_page.controls_panel,
                "method_comparison": self.method_comparison_page.controls_panel,
                "radial_profile": self.radial_profile_page.controls_panel,
            },
            crystal_controls={
                "Strain Mapping": self.strain_map_page.controls_panel,
                "Orientation Mapping": self.orientation_page.controls_panel,
                "Structural Phase Mapping": self.structural_phase_page.controls_panel,
            },
            amorphous_controls={
                "Amorphous Strain": self.amorphous_strain_page.controls_panel,
                "RDF": self.rdf_page.controls_panel,
                "FEM": self.fem_page.controls_panel,
            },
            dpc_controls={
                "DPC / CoM": self.dpc_reconstruction_page.controls_panel,
                "default": self.dpc_legacy_page.controls_panel,
            },
            export_controls={
                "Parallax": self.parallax_export_page.controls_panel,
                "Ptychography": self.ptychography_export_page.controls_panel,
                "default": self.export_controls,
            },
        )
        self.project_coordinator = ProjectCoordinator(
            workflow_state=self.workflow_state,
            pages=self.pages,
            page_objects={
                "virtual_detector": self.virtual_detector_page,
                "preprocessing": self.preprocessing_page,
                "bragg_peaks": self.bragg_peaks_page,
                "calibration": self.calibration_page,
                "orientation": self.orientation_setup_page,
                "orientation_map": self.orientation_map_page,
                "strain_map": self.strain_map_page,
                "phase_contrast": self.phase_contrast_page,
                "bf_df_preview": self.bf_df_preview_page,
                "ptychography": self.ptychography_page,
                "ptychography_data": self.ptychography_data_page,
                "ptychography_geometry": self.ptychography_geometry_page,
                "ptychography_preprocess": self.ptychography_preprocess_page,
                "ptychography_quick": self.ptychography_quick_page,
                "ptychography_review": self.ptychography_review_page,
                "ptychography_optimization": self.ptychography_optimization_page,
                "ptychography_advanced": self.ptychography_advanced_page,
                "ptychography_export": self.ptychography_export_page,
                "method_comparison": self.method_comparison_page,
            },
            dpc_pages=(
                self.dpc_segmented_page,
                self.dpc_preprocess_page,
                self.dpc_review_page,
                self.dpc_reconstruction_page,
                self.dpc_legacy_page,
            ),
            parallax_pages={
                "parallax": self.parallax_page,
                "parallax_bf": self.parallax_bf_page,
                "parallax_alignment": self.parallax_alignment_page,
                "parallax_review": self.parallax_review_page,
                "parallax_advanced": self.parallax_advanced_page,
                "parallax_export": self.parallax_export_page,
            },
            state_service=self.project_state_service,
            result_registry=self.result_registry,
            report_service=self.report_service,
            log_panel=self.log_panel,
        )
        self._progress("Wiring routes and signals…")
        self.route_coordinator = RouteCoordinator(
            toolbar=self.project_toolbar,
            route_bar=self.route_bar,
            module_panel=self.module_panel,
            viewer_stack=self.viewer_stack,
            viewer_pages=self.viewer_pages,
            workflow_state=self.workflow_state,
            controls_provider=lambda key, _goal: self._route_controls(key),
            workspace_provider=self.pages.workspace_for_page,
            style_refresher=self._bold_section_titles,
            data_ready_provider=lambda: self._get_current_4d_shape() is not None,
        )
        self._apply_cuda_setting(self.cuda_enabled)

        self.tree.node_selected.connect(self._handle_node_selected)
        self.scan_viewer.image_clicked.connect(self._handle_scan_image_clicked)
        self.bragg_peaks_page.braggvectors_ready.connect(self._store_current_braggvectors)
        self.bragg_peaks_page.braggvectors_ready.connect(self.calibration_page.refresh_status)
        self.bragg_peaks_page.braggvectors_ready.connect(self.strain_map_page.notify_braggvectors_ready)
        self.bragg_peaks_page.braggvectors_ready.connect(self.calibration_page.show_braggvectors_histogram)
        self.virtual_detector_page.virtual_image_ready.connect(self.bragg_peaks_page.set_virtual_image)
        self.virtual_detector_page.virtual_image_ready.connect(self._show_virtual_image_in_scan_viewer)
        self.preprocessing_page.scan_overview_ready.connect(self._cache_scan_overview)
        self.workflow_state.changed.connect(self._refresh_pipeline_state)
        self.workflow_state.changed.connect(self._update_status_for_workflow)
        self.dpc_reconstruction_page.dpc_result_ready.connect(self._store_dpc_result)
        self.dpc_legacy_page.dpc_result_ready.connect(self._store_dpc_result)
        self.ptychography_advanced_page.ptychography_result_ready.connect(self._store_ptychography_result)
        self.log_panel.log("Application started.")
        self._apply_image_scaling(self.image_scaling)
        self._apply_image_colormap(self.image_cmap)
        self._update_structure_route()
        self._progress("Ready")

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

        self.layout_menu = self.menuBar().addMenu("&Layout")
        self.reset_layout_action = self.layout_menu.addAction("Reset &Layout")
        self.reset_layout_action.setStatusTip("Restore docks to default positions and sizes")
        self.reset_layout_action.triggered.connect(lambda: self._reset_layout())
        self.layout_menu.addSeparator()
        self.toggle_data_dock = self.layout_menu.addAction("Data &Browser")
        self.toggle_data_dock.setCheckable(True)
        self.toggle_data_dock.setChecked(True)
        self.toggle_controls_dock = self.layout_menu.addAction("&Controls")
        self.toggle_controls_dock.setCheckable(True)
        self.toggle_controls_dock.setChecked(True)
        self.toggle_output_dock = self.layout_menu.addAction("&Output")
        self.toggle_output_dock.setCheckable(True)
        self.toggle_output_dock.setChecked(True)

        self.view_menu = self.menuBar().addMenu("&View")
        self.theme_action_group = None
        self.dark_theme_action = self.view_menu.addAction("Dark Theme")
        self.dark_theme_action.setCheckable(True)
        self.dark_theme_action.setChecked(False)
        self.light_theme_action = self.view_menu.addAction("Light Theme")
        self.light_theme_action.setCheckable(True)
        self.light_theme_action.setChecked(True)
        self.theme_action_group = QActionGroup(self)
        self.theme_action_group.addAction(self.dark_theme_action)
        self.theme_action_group.addAction(self.light_theme_action)
        self.dark_theme_action.triggered.connect(lambda: self._apply_theme("dark"))
        self.light_theme_action.triggered.connect(lambda: self._apply_theme("light"))

        self.setting_action = self.menuBar().addAction("&Setting")
        self.setting_action.triggered.connect(self.open_settings)

        self.help_menu = self.menuBar().addMenu("&Help")
        self.about_action = self.help_menu.addAction("&About")
        self.about_action.setStatusTip("Overview, current capabilities, and improvements")
        self.about_action.triggered.connect(self.show_about)
        self.license_action = self.help_menu.addAction("&License")
        self.license_action.setStatusTip("GNU GPLv3 and py4DSTEM licensing information")
        self.license_action.triggered.connect(self.show_license)
        self.tutorials_action = self.help_menu.addAction("&Workflow Tutorials")
        self.tutorials_action.setStatusTip("Brief introduction to each analysis workflow")
        self.tutorials_action.triggered.connect(self.show_tutorials)

    def _build_status_bar(self) -> None:
        status = self.statusBar()
        self.status_coord = QLabel("x: -, y: -, value: -")
        self.status_coord.setMinimumWidth(220)
        self.status_datacube = QLabel("No DataCube loaded")
        self.status_led = QLabel(f"<span style='color:{Theme.NEUTRAL};'>{Theme.LED_CHAR}</span> Ready")
        status.addWidget(self.status_coord)
        status.addWidget(self.status_datacube)
        status.addPermanentWidget(self.status_led)
        self.scan_viewer.coordinate_changed.connect(self._update_status_coord)

    def _apply_theme(self, theme: str) -> None:
        """Switch the global application stylesheet between dark and light."""
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            return
        qss_name = "theme.qss" if theme == "dark" else "theme_light.qss"
        # __file__ lives in app/, so the QSS files are siblings, not under an "app/" subdir.
        qss_path = Path(__file__).parent / qss_name
        try:
            app.setStyleSheet(qss_path.read_text(encoding="utf-8"))
        except OSError as exc:
            self.log_panel.log(f"Could not load theme '{theme}': {exc}")
            return
        self.log_panel.log(f"Theme switched to {theme}.")

    def _update_status_coord(self, text: str) -> None:
        self.status_coord.setText(text)

    def _update_status_datacube(self) -> None:
        name = self.datacube_name_label.text()
        if name == "-":
            self.status_datacube.setText("No DataCube loaded")
        else:
            self.status_datacube.setText(
                f"{name}  |  scan={self.scan_shape_label.text()}  "
                f"diff={self.diffraction_shape_label.text()}"
            )

    def _update_status_led(self, state: str, message: str = "") -> None:
        color = {
            "ready": Theme.READY,
            "running": Theme.RUNNING,
            "stale": Theme.STALE,
            "failed": Theme.FAILED,
            "neutral": Theme.NEUTRAL,
        }.get(state, Theme.NEUTRAL)
        label = message or state.capitalize()
        self.status_led.setText(
            f"<span style='color:{color};'>{Theme.LED_CHAR}</span> {label}"
        )

    def _update_status_for_workflow(self) -> None:
        if self.datacube_name_label.text() == "-":
            self._update_status_led("neutral", "Ready")
        elif self.workflow_state._stale:
            self._update_status_led("stale", "Stale results")
        else:
            self._update_status_led("ready", "DataCube loaded")

    def show_about(self) -> None:
        self._show_help_dialog("About py4DSTEM Pipeline", self.ABOUT_HTML)

    def show_license(self) -> None:
        self._show_help_dialog("License", self.LICENSE_HTML)

    def show_tutorials(self) -> None:
        self._show_help_dialog("Workflow Tutorials", self.TUTORIAL_HTML)

    def _show_help_dialog(self, title: str, html: str) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.resize(720, 620)
        layout = QVBoxLayout(dialog)
        browser = QTextBrowser(dialog)
        browser.setOpenExternalLinks(True)
        browser.setHtml(html)
        layout.addWidget(browser, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Close, parent=dialog)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec()

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
        for page in (
            self.orientation_setup_page,
            self.orientation_map_page,
        ):
            page.set_cuda_enabled(enabled)
        for page in (
            self.parallax_bf_page,
            self.parallax_alignment_page,
            self.parallax_review_page,
            self.parallax_advanced_page,
            self.parallax_export_page,
            self.ptychography_data_page,
            self.ptychography_geometry_page,
            self.ptychography_preprocess_page,
            self.ptychography_quick_page,
            self.ptychography_review_page,
            self.ptychography_optimization_page,
            self.ptychography_advanced_page,
            self.ptychography_export_page,
        ):
            page.set_cuda_enabled(enabled)

    def _build_layout(self) -> None:
        data_browser = QWidget()
        data_browser_layout = QVBoxLayout(data_browser)
        data_browser_layout.setContentsMargins(0, 0, 0, 0)
        data_browser_layout.setSpacing(0)
        data_browser_layout.addWidget(self.tree, 1)
        preview_bar = QWidget()
        preview_layout = QHBoxLayout(preview_bar)
        preview_layout.setContentsMargins(3, 3, 3, 3)
        preview_layout.setSpacing(3)
        preview_layout.addWidget(QLabel("rx"))
        preview_layout.addWidget(self.rx_spin)
        preview_layout.addWidget(QLabel("ry"))
        preview_layout.addWidget(self.ry_spin)
        preview_layout.addWidget(self.preview_button)
        data_browser_layout.addWidget(preview_bar)
        self.tree.setFrameShape(QFrame.NoFrame)

        self.main_view = MultiViewWorkspace(self.scan_viewer, self.diffraction_viewer)
        self.viewer_stack = QStackedWidget()
        self.viewer_pages = {
            "overview": self.main_view,
            "preprocess": self.preprocessing_page,
            "virtual": self.virtual_detector_page,
            "bragg": self.bragg_peaks_page,
            "calibration": self.calibration_page,
            "orientation": self.orientation_setup_page,
            "crystalline_results": self.crystalline_results_page,
            "strain": self.strain_map_page,
            "structural_phase": self.structural_phase_page,
            "phase_contrast": self.phase_contrast_page,
            "bf_df": self.bf_df_preview_page,
            "dpc_segmented": self.dpc_segmented_page,
            "dpc_preprocess": self.dpc_preprocess_page,
            "dpc": self.dpc_reconstruction_page,
            "dpc_legacy": self.dpc_legacy_page,
            "parallax_bf": self.parallax_bf_page,
            "parallax_alignment": self.parallax_alignment_page,
            "parallax_review": self.parallax_review_page,
            "parallax_advanced": self.parallax_advanced_page,
            "parallax_export": self.parallax_export_page,
            "parallax": self.parallax_alignment_page,
            "ptychography": self.ptychography_page,
            "ptychography_data": self.ptychography_data_page,
            "ptychography_geometry": self.ptychography_geometry_page,
            "ptychography_preprocess": self.ptychography_preprocess_page,
            "ptychography_quick": self.ptychography_quick_page,
            "ptychography_review": self.ptychography_review_page,
            "ptychography_optimization": self.ptychography_optimization_page,
            "ptychography_advanced": self.ptychography_advanced_page,
            "ptychography_export": self.ptychography_export_page,
            "method_comparison": self.method_comparison_page,
            "radial_profile": self.radial_profile_page,
            "rdf": self.rdf_page,
            "fem": self.fem_page,
            "amorphous_strain": self.amorphous_strain_page,
        }
        for page in self.viewer_pages.values():
            self.viewer_stack.addWidget(page)

        self.module_panel = ModuleControlPanel()
        self.module_panel.setMinimumWidth(300)
        self.module_panel.setMaximumWidth(500)
        self.module_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        # --- Docks (dockable, floatable, movable per SEM/FIB convention) ---
        self.data_dock = QDockWidget("Data Browser", self)
        self.data_dock.setObjectName("dataDock")
        self.data_dock.setWidget(data_browser)
        self.data_dock.setAllowedAreas(
            Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea | Qt.BottomDockWidgetArea | Qt.TopDockWidgetArea
        )
        self.data_dock.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetClosable
        )
        self.addDockWidget(Qt.LeftDockWidgetArea, self.data_dock)

        self.controls_dock = QDockWidget("Controls", self)
        self.controls_dock.setObjectName("controlsDock")
        self.controls_dock.setWidget(self.module_panel)
        self.controls_dock.setAllowedAreas(
            Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea | Qt.BottomDockWidgetArea | Qt.TopDockWidgetArea
        )
        self.controls_dock.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetClosable
        )
        self.addDockWidget(Qt.RightDockWidgetArea, self.controls_dock)

        self.output_dock = QDockWidget("Output", self)
        self.output_dock.setObjectName("outputDock")
        log_panel_widget = self.log_panel
        log_panel_widget.setMinimumHeight(80)
        log_panel_widget.setMaximumHeight(400)
        log_panel_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.output_dock.setWidget(log_panel_widget)
        self.output_dock.setAllowedAreas(
            Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea | Qt.BottomDockWidgetArea | Qt.TopDockWidgetArea
        )
        self.output_dock.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetClosable
        )
        self.addDockWidget(Qt.BottomDockWidgetArea, self.output_dock)

        self.project_toolbar = ProjectToolbar()
        self.route_bar = TechnicalRouteBar()
        self.route_bar.setObjectName("routeBar")
        self.data_setup_controls = self._build_role_panel()
        self.export_controls = self._build_export_panel()
        self.project_toolbar.structure_changed.connect(self._update_structure_route)
        self.project_toolbar.goal_changed.connect(self._update_structure_route)
        self.project_toolbar.export_clicked.connect(self._show_export_workspace)
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

        # --- Central widget: top toolbar/route bar + viewer stack ---
        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        central_layout.addWidget(self.project_toolbar)
        central_layout.addWidget(self.route_bar)
        self.workflow_divider = self._horizontal_divider("workflowDivider")
        central_layout.addWidget(self.workflow_divider)
        central_layout.addWidget(self.viewer_stack, 1)
        self.setCentralWidget(central)

        self.log_divider = self._horizontal_divider("logDivider")

        # Placeholder kept for backward compatibility (tests reference main_splitter.sizes).
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setVisible(False)

        # Wire View-menu dock toggles now that docks exist.
        # The reverse connection (visibilityChanged -> setChecked) uses
        # blockSignals so programmatic setChecked (fired on minimize/restore)
        # cannot re-enter setVisible and hide the docks on window restore.
        # shiboken.isValid guards the late visibilityChanged emissions Qt may
        # deliver after the menu's QAction has already been torn down at close.
        from shiboken6 import isValid as _shiboken_is_valid

        def _wire_dock_toggle(action, dock):
            def toggle_visibility(checked):
                dock.setVisible(checked)
            def sync_action(visible):
                if not _shiboken_is_valid(action):
                    return
                was = action.blockSignals(True)
                action.setChecked(visible)
                action.blockSignals(was)
            action.toggled.connect(toggle_visibility)
            dock.visibilityChanged.connect(sync_action)
        _wire_dock_toggle(self.toggle_data_dock, self.data_dock)
        _wire_dock_toggle(self.toggle_controls_dock, self.controls_dock)
        _wire_dock_toggle(self.toggle_output_dock, self.output_dock)

        self._set_index_controls_visible(False)
        self._compact_input_controls()
        self._bold_section_titles()
        self._set_preview_empty()
        # Dock sizes are applied after the first show/maximize event. Applying
        # them here is too early: Qt redistributes dock geometry when the real
        # screen size becomes available, making startup differ from Reset Layout.

    def showEvent(self, event) -> None:  # noqa: N802 - Qt API name
        super().showEvent(event)
        if self._initial_layout_applied:
            return
        self._initial_layout_applied = True
        self._reset_layout(silent=True)

    def _reset_layout(self, silent: bool = False) -> None:
        """Restore all docks to their default dock areas and sizes."""
        for dock in (self.data_dock, self.controls_dock, self.output_dock):
            dock.setFloating(False)
        # Re-add to canonical areas to undo any tabbing/dragging.
        self.removeDockWidget(self.data_dock)
        self.removeDockWidget(self.controls_dock)
        self.removeDockWidget(self.output_dock)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.data_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.controls_dock)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.output_dock)
        # Explicitly show — removeDockWidget hides them and addDockWidget does not un-hide.
        for dock in (self.data_dock, self.controls_dock, self.output_dock):
            dock.setVisible(True)
            dock.setFeatures(
                QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetClosable
            )
        self.resizeDocks(
            [self.data_dock, self.controls_dock],
            [300, 350],
            Qt.Horizontal,
        )
        self.resizeDocks([self.output_dock], [100], Qt.Vertical)
        self.toggle_data_dock.setChecked(True)
        self.toggle_controls_dock.setChecked(True)
        self.toggle_output_dock.setChecked(True)
        if not silent:
            self.log_panel.log("Layout reset to default.")

    def _horizontal_divider(self, object_name: str) -> QFrame:
        divider = QFrame()
        divider.setObjectName(object_name)
        divider.setFrameShape(QFrame.NoFrame)
        divider.setFixedHeight(1)
        divider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        return divider

    def _bold_section_titles(self) -> None:
        # Group-box bold styling is now handled globally by theme.qss.
        # This hook is retained as the route_coordinator style_refresher callback.
        pass

    def _update_structure_route(self, *_args) -> None:
        self.route_coordinator.current_key = self.current_route_key
        self.route_coordinator.update_structure()
        self.route_modules = self.route_coordinator.modules
        self.current_route_key = self.route_coordinator.current_key

    def _route_states(self) -> dict[str, str]:
        return self.route_coordinator.states()

    def _select_route_module(self, key: str) -> None:
        self.route_coordinator.select(key)
        self.current_route_key = self.route_coordinator.current_key

    def _show_export_workspace(self) -> None:
        """Switch the central viewer and control panel to the unified Export view.

        Bypasses the route system so Export is reachable from any route via the
        toolbar button, independent of the breadcrumb selection.
        """
        export_module = RouteModule(
            key="export", title="Export", page_key="overview",
            requirements="Registered results to export.",
            output_target="Exported result files and project state.",
        )
        self.viewer_stack.setCurrentWidget(self.main_view)
        self.module_panel.set_module(export_module, self.export_controls)

    def _refresh_pipeline_state(self) -> None:
        if not hasattr(self, "route_coordinator"):
            return
        self.route_coordinator.current_key = self.current_route_key
        self.route_coordinator.modules = self.route_modules
        self.route_coordinator.refresh()
        self.current_route_key = self.route_coordinator.current_key

    def _controls_for_route(self, key: str) -> QWidget | None:
        return self._route_controls(key)

    def _route_controls(self, key: str) -> QWidget | None:
        return self.pages.controls_for_route(key, self.project_toolbar.goal.currentText())

    def _build_role_panel(self) -> QWidget:
        roles_group = QGroupBox("Dataset Roles / Sources")
        roles_layout = QVBoxLayout(roles_group)
        for label, role in [
            ("Set as Target", "target_datacube"),
            ("Vacuum Probe", "vacuum_probe"),
            ("Ellipse Reference", "polycrystal_calibration"),
            ("Rotation Reference", "defocused_cbed"),
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
        csv_button = QPushButton("Export Data to CSV")
        csv_button.clicked.connect(self.export_results_to_csv)
        save_button = QPushButton("Save Project")
        save_button.clicked.connect(self.save_project)
        report_button = QPushButton("Generate Report")
        report_button.clicked.connect(self.generate_report)
        layout.addWidget(export_button)
        layout.addWidget(csv_button)
        layout.addWidget(save_button)
        layout.addWidget(report_button)
        layout.addStretch(1)
        return panel

    def export_registered_result(self) -> None:
        output_dir = self.project_coordinator.export_registered_result(
            self, self._default_output_dir()
        )
        if output_dir is not None:
            self.recent_export_dir = output_dir

    def export_results_to_csv(self) -> None:
        """Batch-export every registered numeric result to CSV files in a folder.

        Each result is written as ``<category>_<name>.csv``. Arrays become
        long-format tables (row/col/value); scalar dicts become key/value rows.
        """
        entries = self.result_registry.list_entries()
        if not entries:
            QMessageBox.information(
                self, "Export to CSV", "No results have been registered yet. Run an analysis step first."
            )
            return
        start_dir = str(self.recent_export_dir or self._default_output_dir())
        directory = QFileDialog.getExistingDirectory(
            self, "Export Registered Results to CSV", start_dir
        )
        if not directory:
            return
        output_dir = Path(directory)
        succeeded, failed = 0, 0
        for entry in entries:
            safe_name = f"{entry.category}_{entry.name}".replace(" ", "_").replace("/", "_")
            csv_path = output_dir / f"{safe_name}.csv"
            try:
                self.result_registry.export(entry.key, csv_path)
                succeeded += 1
            except Exception as exc:  # noqa: BLE001 - keep exporting remaining entries
                failed += 1
                self.log_panel.log(f"CSV export failed for {entry.key}: {exc}")
        self.log_panel.log(
            f"CSV export complete: {succeeded} succeeded, {failed} failed, in {output_dir}."
        )
        self.recent_export_dir = output_dir

    def _compact_input_controls(self) -> None:
        for widget_type in (NumericLineEdit, QComboBox):
            for widget in self.findChildren(widget_type):
                widget.setMinimumWidth(0)
                widget.setMaximumWidth(280)

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
            self.current_file = self.session.open_file(file_path)
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
        output_dir = self.project_coordinator.save_project(
            self, self._default_output_dir(), self._project_state()
        )
        if output_dir is not None:
            self.recent_export_dir = output_dir

    def load_project(self) -> None:
        state = self.project_coordinator.choose_and_load_project(
            self, self._default_output_dir()
        )
        if state is not None:
            self._apply_project_state(state)
            self.project_coordinator.restore_loaded_results(state)
            if self.project_coordinator.loaded_project_path is not None:
                self.recent_export_dir = self.project_coordinator.loaded_project_path.parent

    def generate_report(self) -> None:
        output_dir = self.project_coordinator.generate_report(
            self,
            self._default_output_dir(),
            self._project_state(),
            self.log_panel.event_log.toPlainText(),
            self.log_panel.process_log.toPlainText(),
        )
        if output_dir is not None:
            self.recent_export_dir = output_dir

    def _handle_node_selected(self, hdf5_path: str, node_kind: str) -> None:
        if self.current_file is None:
            return

        self.log_panel.log(f"Selected {node_kind}: {hdf5_path}")
        self.selected_hdf5_path = hdf5_path
        self.selected_node_kind = node_kind
        self.selected_preview_kind = "Not displayable"
        self.selected_preview_shape = None
        self.preview_status = "Not rendered"

        try:
            node = self.current_file[hdf5_path]
            info = self.hdf5_service.describe_node(node, hdf5_path)
            self._show_node_info(info)
            preview = self.hdf5_service.describe_preview(node)
            self.selected_preview_kind = str(preview["kind"])
            shape = preview.get("shape")
            self.selected_preview_shape = tuple(shape) if isinstance(shape, tuple) else None
            self._configure_4d_controls(self.selected_preview_shape)
            self.preview_button.setEnabled(self.selected_preview_kind != "Not displayable")
            self._refresh_tree_data_info()
            self.log_panel.log(
                f"Selection ready for lazy preview: {hdf5_path} ({self.selected_preview_kind})."
            )
        except Exception as exc:
            self.preview_button.setEnabled(False)
            self.log_panel.log(f"Failed to inspect node: {exc}")

    def _preview_selected_node(self) -> None:
        if self.current_file is None or self.selected_hdf5_path is None:
            return
        try:
            node = self.current_file[self.selected_hdf5_path]
            if self.selected_preview_kind == "Diffraction slice":
                if not isinstance(node, h5py.Dataset):
                    raise ValueError("The selected diffraction slice is not a dataset.")
                image = self.hdf5_service.read_2d_dataset(node)
                self.diffraction_viewer.set_image(image)
                self.preview_status = "Rendered diffraction slice"
            elif self.selected_preview_kind == "DataCube":
                if not self._activate_selected_datacube():
                    raise ValueError("The selected node is not a displayable DataCube.")
                self._display_4d_slice(self.rx_spin.value(), self.ry_spin.value())
                self.preview_status = (
                    f"Rendered DataCube diffraction slice [{self.rx_spin.value()}, {self.ry_spin.value()}]"
                )
            else:
                return
            self._refresh_tree_data_info()
            self.log_panel.log(f"{self.preview_status}: {self.selected_hdf5_path}")
        except Exception as exc:
            self.preview_status = f"Preview failed: {exc}"
            self._refresh_tree_data_info()
            self.log_panel.log(self.preview_status)
            QMessageBox.warning(self, "Preview error", str(exc))

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
        self.selected_preview_kind = "Not displayable"
        self.selected_preview_shape = None
        self.preview_status = "Not rendered"
        self.preview_button.setEnabled(False)
        self._refresh_tree_data_info()

    def _clear_datacube_info(self) -> None:
        self.datacube_name_label.setText("-")
        self.scan_shape_label.setText("-")
        self.diffraction_shape_label.setText("-")
        self._refresh_tree_data_info()
        self._update_status_datacube()
        self._update_status_led("neutral", "Ready")

    def _configure_4d_controls(self, shape: tuple[int, ...] | None) -> None:
        is_4d = shape is not None and len(shape) == 4
        self.rx_spin.blockSignals(True)
        self.ry_spin.blockSignals(True)
        self.rx_spin.setMaximum(max(shape[0] - 1, 0) if is_4d else 0)
        self.ry_spin.setMaximum(max(shape[1] - 1, 0) if is_4d else 0)
        self.rx_spin.setValue(0)
        self.ry_spin.setValue(0)
        self.rx_spin.blockSignals(False)
        self.ry_spin.blockSignals(False)
        self._set_index_controls_visible(is_4d)

    def _set_index_controls_visible(self, visible: bool) -> None:
        self.rx_spin.setEnabled(visible)
        self.ry_spin.setEnabled(visible)

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
                image = self.session.diffraction_pattern(self.current_dataset_path, dataset, rx, ry)
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

    def _try_load_py4dstem_datacube(
        self,
        hdf5_path: str,
        show_warning: bool = True,
        render_overview: bool = False,
    ) -> bool:
        try:
            info = self.py4dstem_service.load_datacube(hdf5_path)
            self.bragg_strain_service.braggvectors = None
            self.bragg_strain_service.strainmap = None
            self.bragg_strain_service.strain_result = None
            self.bragg_strain_service.probe_kernel = None
            self.workflow_state.data_source_updated()
            if render_overview:
                scan_image = self.py4dstem_service.get_scan_image()
                self.scan_viewer.set_image(scan_image)
            self.current_4d_source = "py4dstem"
            self.current_dataset_path = info.datapath
            self.current_dataset_shape = info.shape
            self._restore_current_braggvectors()
            self._show_datacube_info(info.name, info.scan_shape, info.diffraction_shape)
            if render_overview:
                try:
                    geometry = self.py4dstem_service.measure_probe_geometry()
                    self.log_panel.log(
                        "Measured probe geometry: "
                        f"radius={geometry.radius:.3g}, center=({geometry.center_x:.3g}, {geometry.center_y:.3g})."
                    )
                except Py4DSTEMServiceError as exc:
                    self.log_panel.log(str(exc))
            self._configure_4d_controls(info.shape)
            if render_overview:
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

    def _load_raw_4d_dataset(
        self,
        hdf5_path: str,
        shape: tuple[int, ...],
        *,
        render_scan_image: bool = False,
    ) -> None:
        if self.current_file is None:
            return

        dataset = self.current_file[hdf5_path]
        info = self.py4dstem_service.load_raw_4d_array(dataset, hdf5_path)
        self.workflow_state.data_source_updated()
        if render_scan_image:
            scan_image = self._raw_scan_image(hdf5_path, dataset)
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
        self._update_status_datacube()
        self._update_status_led("ready", "DataCube loaded")

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
        closed_path, error = self.session.close_file()
        if closed_path is not None and error is None:
            self.log_panel.log(f"Closed file: {closed_path}")
        if error is not None:
            self.log_panel.log(f"Failed to close file cleanly: {error}")
        self.bragg_strain_service.braggvectors = None
        self.bragg_strain_service.strainmap = None
        self.bragg_strain_service.strain_result = None
        self.bragg_strain_service.probe_kernel = None
        self.phase_retrieval_results.clear()
        self.workflow_state.set_dataset_role("target_datacube", None)
        self.workflow_state.set_dataset_role("polycrystal_calibration", None)
        self.workflow_state.set_dataset_role("vacuum_probe", None)
        self.workflow_state.set_dataset_role("defocused_cbed", None)
        self._refresh_role_labels()

    def _get_virtual_detector_source(self):
        return self.session.virtual_detector_source()

    def _get_virtual_detector_image(self):
        return self.virtual_detector_page.result

    def _get_py4dstem_datacube(self):
        return self.session.py4dstem_datacube()

    def _get_vacuum_probe_source(self):
        role_path = self.workflow_state.dataset_roles.vacuum_probe
        if not role_path:
            return None
        if self.current_file is not None:
            try:
                node = self.current_file[role_path]
                if isinstance(node, h5py.Dataset):
                    return np.asarray(node[...])
            except Exception:
                pass
        return self.py4dstem_service.read_datapath(role_path)

    def _get_selected_display_source(self):
        path = self._current_tree_selection_path()
        target_path = self.workflow_state.dataset_roles.target_datacube
        return self.session.selected_display_source(path, target_path)

    def _get_show_data_source(self):
        selected_path = self._current_tree_selection_path()
        if (
            selected_path
            and self.current_file is not None
            and self.selected_preview_kind == "DataCube"
            and self._activate_selected_datacube()
        ):
            active_path = self.current_dataset_path or selected_path
            if self.workflow_state.dataset_roles.target_datacube != active_path:
                self._assign_role_path("target_datacube", active_path)
            return self._get_virtual_detector_source()
        return self._get_selected_display_source()

    def _activate_selected_datacube(self) -> bool:
        selected_path = self._current_tree_selection_path()
        if self.current_file is None or selected_path is None:
            return False
        node = self.current_file[selected_path]
        if self.hdf5_service.describe_preview(node)["kind"] != "DataCube":
            return False
        self.scan_viewer.clear("Scan overview deferred until a workflow requests it.")
        if self._try_load_py4dstem_datacube(
            selected_path,
            show_warning=False,
            render_overview=False,
        ):
            return True
        dataset = self.hdf5_service.resolve_4d_dataset(node)
        self._load_raw_4d_dataset(dataset.name, tuple(dataset.shape), render_scan_image=False)
        return True

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
        result = self.bragg_strain_service.compute_braggvectors(datacube, params)
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
        return self.session.target_bright_field_image()

    def _cache_scan_overview(self, source, image) -> None:
        if self.session.cache_scan_overview(source, image):
            self.scan_viewer.set_image(np.asarray(image))
            self.log_panel.log("Cached scan overview for downstream workflows.")

    def _raw_scan_image(self, hdf5_path: str, dataset: h5py.Dataset) -> np.ndarray:
        return self.session.raw_scan_image(hdf5_path, dataset)

    def _clear_raw_scan_image_cache(self) -> None:
        self.session.clear_raw_scan_image_cache()

    def _get_probe_geometry(self):
        if self.py4dstem_service.probe_geometry is None and self.current_4d_source == "py4dstem":
            try:
                geometry = self.py4dstem_service.measure_probe_geometry()
                self.log_panel.log(
                    "Measured probe geometry on demand: "
                    f"radius={geometry.radius:.3g}, center=({geometry.center_x:.3g}, {geometry.center_y:.3g})."
                )
            except Py4DSTEMServiceError as exc:
                self.log_panel.log(str(exc))
        return self.py4dstem_service.probe_geometry

    def _store_current_braggvectors(self) -> None:
        if self.current_dataset_path is None or self.bragg_strain_service.braggvectors is None:
            return
        self.braggvectors_by_datacube[self.current_dataset_path] = self.bragg_strain_service.braggvectors
        self.log_panel.log(f"BraggVectors stored for DataCube: {self.current_dataset_path}")

    def _store_dpc_result(self, result: PhaseContrastResult) -> None:
        self.phase_retrieval_results["DPC"] = result
        self.log_panel.log("DPC result stored for method comparison")

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
        return self.session.current_4d_shape()

    def _assign_current_role(self, role: str) -> None:
        selected_path = self._current_tree_selection_path()
        if selected_path is None:
            QMessageBox.information(self, "Dataset Roles", "Select a node in the HDF5 tree first.")
            return
        self.selected_hdf5_path = selected_path
        self._assign_role_path(role, selected_path)

    def _assign_role_path(self, role: str, selected_path: str) -> None:
        self.session.assign_role(
            role,
            selected_path,
            workflow_state=self.workflow_state,
            dpc_service=self.dpc_service,
            parallax_service=self.parallax_service,
            clear_workspaces=self._clear_all_image_workspaces,
            result_registry=self.result_registry,
        )
        self.ptychography_page.service.reset()
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
                "Preview type": self.selected_preview_kind,
                "Preview status": self.preview_status,
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
        window_state = bytes(self.saveState().toBase64()).decode("ascii") if self.data_dock else None
        return self.project_coordinator.snapshot(
            file_path=self.current_file_path,
            selected_hdf5_path=self.selected_hdf5_path,
            image_scaling=self.image_scaling,
            image_cmap=self.image_cmap,
            cuda_enabled=self.cuda_enabled,
            recent_export_dir=self.recent_export_dir,
            window_state=window_state,
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
        self.project_coordinator.apply_page_params(state.page_params)
        self.project_coordinator.restore_grid_states(state.grid_states)
        if self.current_file is not None and state.selected_hdf5_path:
            try:
                node = self.current_file[state.selected_hdf5_path]
                node_kind = "dataset" if isinstance(node, h5py.Dataset) else "group"
                self._handle_node_selected(state.selected_hdf5_path, node_kind)
            except Exception as exc:
                self.log_panel.log(f"Saved HDF5 selection is unavailable: {exc}")
        if state.window_state:
            from PySide6.QtCore import QByteArray
            self.restoreState(QByteArray.fromBase64(state.window_state.encode("ascii")))

    def _apply_page_params(self, page_params: dict[str, dict[str, object]]) -> None:
        self.project_coordinator.apply_page_params(page_params)

    def _dpc_params_snapshot(self) -> dict[str, object]:
        return self.project_coordinator.dpc_params_snapshot()

    def _default_output_dir(self) -> Path:
        return self.project_coordinator.default_output_dir(
            self.recent_export_dir, self.current_file_path
        )

    def _named_workspaces(self) -> dict[str, AdaptiveImageWorkspace]:
        return self.pages.named_workspaces()

    def _workspace_for_page(self, page: QWidget) -> AdaptiveImageWorkspace | None:
        return self.pages.workspace_for_page(page)

    def _grid_states(self) -> dict[str, dict[str, object]]:
        return self.pages.grid_states()

    def _clear_all_image_workspaces(self, exclude_keys: set[str] | None = None) -> None:
        self.pages.clear_workspaces(exclude_keys)

    def _filters_for_entry(self, formats: tuple[str, ...]) -> list[str]:
        return self.project_coordinator.filters_for_entry(formats)

    def _safe_result_filename(self, name: str, extension: str) -> str:
        return self.project_coordinator.safe_result_filename(name, extension)

    def _path_with_supported_suffix(self, path: Path, extension: str) -> Path:
        return self.project_coordinator.path_with_supported_suffix(path, extension)
