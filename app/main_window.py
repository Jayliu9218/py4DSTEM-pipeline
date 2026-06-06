from __future__ import annotations

from pathlib import Path

import h5py
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.pages.virtual_detector_page import VirtualDetectorPage
from app.pages.bragg_peaks_page import BraggPeaksPage
from app.pages.calibration_page import CalibrationPage
from app.pages.strain_map_page import StrainMapPage
from app.services.bragg_strain_service import BraggStrainService
from app.services.hdf5_service import Hdf5Service
from app.services.py4dstem_service import Py4DSTEMService, Py4DSTEMServiceError
from app.widgets.hdf5_tree_widget import Hdf5TreeWidget
from app.widgets.image_viewer import ImageViewer
from app.widgets.log_panel import LogPanel


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("py4DSTEM Pipeline - HDF5 Viewer")
        self.resize(1280, 820)

        self.hdf5_service = Hdf5Service()
        self.py4dstem_service = Py4DSTEMService()
        self.bragg_strain_service = BraggStrainService()
        self.current_file: h5py.File | None = None
        self.current_file_path: Path | None = None
        self.current_dataset_path: str | None = None
        self.current_dataset_shape: tuple[int, ...] | None = None
        self.current_4d_source: str | None = None

        self.tree = Hdf5TreeWidget()
        self.scan_viewer = ImageViewer()
        self.diffraction_viewer = ImageViewer()
        self.log_panel = LogPanel()
        self.virtual_detector_page = VirtualDetectorPage(
            source_provider=self._get_virtual_detector_source,
            shape_provider=self._get_current_4d_shape,
            log_panel=self.log_panel,
        )
        self.bragg_peaks_page = BraggPeaksPage(
            datacube_provider=self._get_py4dstem_datacube,
            shape_provider=self._get_current_4d_shape,
            service=self.bragg_strain_service,
            log_panel=self.log_panel,
        )
        self.calibration_page = CalibrationPage(
            datacube_provider=self._get_py4dstem_datacube,
            braggvectors_provider=self._get_braggvectors,
            service=self.bragg_strain_service,
            log_panel=self.log_panel,
        )
        self.strain_map_page = StrainMapPage(
            braggvectors_provider=self._get_braggvectors,
            service=self.bragg_strain_service,
            log_panel=self.log_panel,
        )

        self.datacube_name_label = QLabel("-")
        self.scan_shape_label = QLabel("-")
        self.diffraction_shape_label = QLabel("-")
        self.path_label = QLabel("-")
        self.type_label = QLabel("-")
        self.shape_label = QLabel("-")
        self.dtype_label = QLabel("-")
        self.attrs_table = QTableWidget(0, 2)
        self.attrs_table.setHorizontalHeaderLabels(["Attribute", "Value"])
        self.attrs_table.horizontalHeader().setStretchLastSection(True)

        self.rx_spin = QSpinBox()
        self.ry_spin = QSpinBox()
        self.rx_spin.setMinimum(0)
        self.ry_spin.setMinimum(0)
        self.rx_spin.valueChanged.connect(self._refresh_current_4d_image)
        self.ry_spin.valueChanged.connect(self._refresh_current_4d_image)

        self._build_menu()
        self._build_layout()

        self.tree.node_selected.connect(self._handle_node_selected)
        self.scan_viewer.image_clicked.connect(self._handle_scan_image_clicked)
        self.bragg_peaks_page.braggvectors_ready.connect(self.calibration_page.refresh_status)
        self.bragg_peaks_page.braggvectors_ready.connect(self.strain_map_page.notify_braggvectors_ready)
        self.log_panel.log("Application started.")

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API name
        self._close_current_file()
        event.accept()

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")

        open_action = file_menu.addAction("&Open")
        open_action.triggered.connect(self.open_file)

        file_menu.addSeparator()

        exit_action = file_menu.addAction("E&xit")
        exit_action.triggered.connect(self.close)

    def _build_layout(self) -> None:
        info_panel = QWidget()
        info_layout = QVBoxLayout(info_panel)

        form = QFormLayout()
        form.addRow("DataCube", self.datacube_name_label)
        form.addRow("Scan shape", self.scan_shape_label)
        form.addRow("Diffraction shape", self.diffraction_shape_label)
        form.addRow("rx", self.rx_spin)
        form.addRow("ry", self.ry_spin)
        form.addRow("", QLabel(""))
        form.addRow("Path", self.path_label)
        form.addRow("Type", self.type_label)
        form.addRow("Shape", self.shape_label)
        form.addRow("Dtype", self.dtype_label)

        info_layout.addLayout(form)
        info_layout.addWidget(QLabel("Attributes"))
        info_layout.addWidget(self.attrs_table)

        image_splitter = QSplitter(Qt.Vertical)
        image_splitter.addWidget(self.scan_viewer)
        image_splitter.addWidget(self.diffraction_viewer)
        image_splitter.setSizes([300, 350])

        top_splitter = QSplitter(Qt.Horizontal)
        top_splitter.addWidget(self.tree)
        top_splitter.addWidget(image_splitter)
        top_splitter.addWidget(info_panel)
        top_splitter.setSizes([300, 650, 330])

        main_splitter = QSplitter(Qt.Vertical)
        main_splitter.addWidget(top_splitter)
        main_splitter.addWidget(self.log_panel)
        main_splitter.setSizes([650, 170])

        browser_page = QWidget()
        browser_layout = QHBoxLayout(browser_page)
        browser_layout.setContentsMargins(8, 8, 8, 8)
        browser_layout.addWidget(main_splitter)

        tabs = QTabWidget()
        tabs.addTab(browser_page, "Data Browser")
        tabs.addTab(self.virtual_detector_page, "Virtual Detector")
        tabs.addTab(self.bragg_peaks_page, "Bragg Peaks")
        tabs.addTab(self.calibration_page, "Calibration")
        tabs.addTab(self.strain_map_page, "Strain Map")
        self.setCentralWidget(tabs)

        self._set_index_controls_visible(False)

    def open_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open HDF5 or EMD file",
            "",
            "HDF5/EMD files (*.h5 *.hdf5 *.emd);;All files (*.*)",
        )
        if not file_path:
            return

        try:
            self._close_current_file()
            self.current_file = self.hdf5_service.open_file(file_path)
            self.current_file_path = Path(file_path)
            self.tree.populate(self.current_file)
            self.scan_viewer.clear()
            self.diffraction_viewer.clear()
            self._clear_dataset_info()
            self.virtual_detector_page.viewer.clear()
            self.bragg_strain_service.braggvectors = None
            self.bragg_strain_service.strainmap = None
            self.bragg_strain_service.strain_result = None
            self.log_panel.log(f"Opened file: {file_path}")

            try:
                self.py4dstem_service.open_file(file_path)
                self.log_panel.log("py4DSTEM tree loaded successfully.")
            except Py4DSTEMServiceError as exc:
                self.log_panel.log(str(exc))
        except Exception as exc:
            self.current_file = None
            self.current_file_path = None
            self.log_panel.log(f"Failed to open file: {exc}")
            QMessageBox.critical(self, "Open failed", str(exc))

    def _handle_node_selected(self, hdf5_path: str, node_kind: str) -> None:
        if self.current_file is None:
            return

        self.log_panel.log(f"Selected {node_kind}: {hdf5_path}")
        self.current_dataset_path = None
        self.current_dataset_shape = None
        self.current_4d_source = None
        self._set_index_controls_visible(False)

        try:
            node = self.current_file[hdf5_path]
            info = self.hdf5_service.describe_node(node, hdf5_path)
            self._show_node_info(info)

            if node_kind == "group":
                if not self._try_load_py4dstem_datacube(hdf5_path):
                    self.scan_viewer.clear()
                    self.diffraction_viewer.clear()
                    self._clear_datacube_info()
                return

            if node_kind != "dataset":
                self.scan_viewer.clear()
                self.diffraction_viewer.clear()
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
                self.scan_viewer.clear()
                self.diffraction_viewer.clear()
                self._clear_datacube_info()
                self.log_panel.log(f"Dataset is not displayable as an image: shape={shape}")
        except Exception as exc:
            self.scan_viewer.clear()
            self.diffraction_viewer.clear()
            self.log_panel.log(f"Failed to inspect node: {exc}")
            QMessageBox.warning(self, "Dataset error", str(exc))

    def _show_node_info(self, info: dict[str, object]) -> None:
        self.path_label.setText(str(info.get("path", "-")))
        self.type_label.setText(str(info.get("type", "-")))
        self.shape_label.setText(str(info.get("shape", "-")))
        self.dtype_label.setText(str(info.get("dtype", "-")))

        attrs = info.get("attrs", {})
        if not isinstance(attrs, dict):
            attrs = {}

        self.attrs_table.setRowCount(len(attrs))
        for row, (key, value) in enumerate(attrs.items()):
            self.attrs_table.setItem(row, 0, QTableWidgetItem(str(key)))
            self.attrs_table.setItem(row, 1, QTableWidgetItem(str(value)))
        self.attrs_table.resizeColumnsToContents()

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

    def _clear_datacube_info(self) -> None:
        self.datacube_name_label.setText("-")
        self.scan_shape_label.setText("-")
        self.diffraction_shape_label.setText("-")

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
                return

            if self.current_file is None or self.current_dataset_path is None:
                return

            dataset = self.current_file[self.current_dataset_path]
            image = self.hdf5_service.read_4d_diffraction_pattern(dataset, rx=rx, ry=ry)
            self.diffraction_viewer.set_image(image)
            self.log_panel.log(
                f"Displayed HDF5 diffraction pattern: {self.current_dataset_path}[{rx}, {ry}, :, :]"
            )
        except Exception as exc:
            self.log_panel.log(f"Failed to display diffraction pattern: {exc}")
            QMessageBox.warning(self, "Diffraction pattern error", str(exc))

    def _try_load_py4dstem_datacube(self, hdf5_path: str, show_warning: bool = True) -> bool:
        try:
            info = self.py4dstem_service.load_datacube(hdf5_path)
            scan_image = self.py4dstem_service.get_scan_image()
            self.scan_viewer.set_image(scan_image)
            self.current_4d_source = "py4dstem"
            self.current_dataset_path = hdf5_path
            self.current_dataset_shape = info.shape
            self._show_datacube_info(info.name, info.scan_shape, info.diffraction_shape)
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
        scan_image = self.hdf5_service.read_4d_scan_image(dataset)
        self.scan_viewer.set_image(scan_image)
        self.current_4d_source = "hdf5"
        self.current_dataset_path = hdf5_path
        self.current_dataset_shape = shape
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

    def _get_virtual_detector_source(self):
        if self.current_4d_source == "py4dstem":
            return self.py4dstem_service.datacube
        if self.current_4d_source == "hdf5" and self.current_file is not None and self.current_dataset_path:
            return self.current_file[self.current_dataset_path]
        return None

    def _get_py4dstem_datacube(self):
        if self.current_4d_source == "py4dstem":
            return self.py4dstem_service.datacube
        return None

    def _get_braggvectors(self):
        return self.bragg_strain_service.braggvectors

    def _get_current_4d_shape(self) -> tuple[int, int, int, int] | None:
        if self.current_dataset_shape is None or len(self.current_dataset_shape) != 4:
            return None
        return self.current_dataset_shape
