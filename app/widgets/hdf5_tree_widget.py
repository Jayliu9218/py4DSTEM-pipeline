from __future__ import annotations

import h5py
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem


class Hdf5TreeWidget(QTreeWidget):
    node_selected = Signal(str, str)

    def __init__(self) -> None:
        super().__init__()
        self.setHeaderLabels(["HDF5 tree"])
        self.setTextElideMode(Qt.ElideMiddle)
        self.currentItemChanged.connect(self._emit_selected_node)
        self.itemExpanded.connect(self._load_group_children)
        self.info_root_item: QTreeWidgetItem | None = None
        self._hdf5_file: h5py.File | None = None

    def populate(self, hdf5_file: h5py.File) -> None:
        self.clear()
        self._hdf5_file = hdf5_file
        self.setHeaderLabels(["HDF5 tree"])
        root_item = QTreeWidgetItem(["/"])
        root_item.setToolTip(0, "/")
        root_item.setData(0, 256, "/")
        root_item.setData(0, 257, "group")
        self.addTopLevelItem(root_item)

        self._add_children(root_item, hdf5_file, "/")
        root_item.setData(0, 258, True)
        root_item.setExpanded(True)
        spacer = QTreeWidgetItem([""])
        spacer.setFlags(Qt.NoItemFlags)
        spacer.setSizeHint(0, QSize(0, 12))
        self.addTopLevelItem(spacer)
        self.info_root_item = QTreeWidgetItem(["Data info"])
        self.info_root_item.setToolTip(0, "Data info")
        self.addTopLevelItem(self.info_root_item)
        self.set_data_info()

    def populate_direct_source(self, label: str, source_path: str) -> None:
        self.clear()
        self._hdf5_file = None
        self.setHeaderLabels(["Data source"])
        source_item = QTreeWidgetItem([label])
        source_item.setToolTip(0, source_path)
        source_item.setData(0, 256, source_path)
        source_item.setData(0, 257, "file")
        self.addTopLevelItem(source_item)
        spacer = QTreeWidgetItem([""])
        spacer.setFlags(Qt.NoItemFlags)
        spacer.setSizeHint(0, QSize(0, 12))
        self.addTopLevelItem(spacer)
        self.info_root_item = QTreeWidgetItem(["Data info"])
        self.info_root_item.setToolTip(0, "Data info")
        self.addTopLevelItem(self.info_root_item)
        self.set_data_info()

    def _add_children(self, parent_item: QTreeWidgetItem, group: h5py.Group, group_path: str) -> None:
        for name in sorted(group.keys()):
            node = group[name]
            node_path = f"/{name}" if group_path == "/" else f"{group_path}/{name}"

            if isinstance(node, h5py.Dataset):
                label = f"{name}  {tuple(node.shape)}"
                node_kind = "dataset"
            else:
                label = name
                node_kind = "group"

            item = QTreeWidgetItem([label])
            item.setToolTip(0, label)
            item.setData(0, 256, node_path)
            item.setData(0, 257, node_kind)
            parent_item.addChild(item)

            if isinstance(node, h5py.Group):
                item.setData(0, 258, False)
                if len(node):
                    placeholder = QTreeWidgetItem(["Loading..."])
                    placeholder.setFlags(Qt.NoItemFlags)
                    item.addChild(placeholder)

    def _load_group_children(self, item: QTreeWidgetItem) -> None:
        if item.data(0, 257) != "group" or bool(item.data(0, 258)):
            return
        if self._hdf5_file is None:
            return
        path = item.data(0, 256)
        try:
            group = self._hdf5_file[str(path)]
            if not isinstance(group, h5py.Group):
                return
            item.takeChildren()
            self._add_children(item, group, str(path))
            item.setData(0, 258, True)
        except (KeyError, OSError, RuntimeError):
            item.takeChildren()
            error = QTreeWidgetItem(["Unable to load group"])
            error.setFlags(Qt.NoItemFlags)
            item.addChild(error)

    def _emit_selected_node(self, item: QTreeWidgetItem | None, *_args) -> None:
        if item is None:
            return
        hdf5_path = item.data(0, 256)
        node_kind = item.data(0, 257)
        if hdf5_path and node_kind:
            self.node_selected.emit(str(hdf5_path), str(node_kind))

    def set_data_info(
        self,
        datacube: dict[str, object] | None = None,
        selection: dict[str, object] | None = None,
        roles: dict[str, object] | None = None,
        attrs: dict[str, object] | None = None,
    ) -> None:
        if self.info_root_item is None:
            return
        self.info_root_item.takeChildren()
        for title, values in [
            ("Loaded DataCube", datacube or {}),
            ("Selection", selection or {}),
            ("Dataset roles", roles or {}),
            ("Attributes", attrs or {}),
        ]:
            group = QTreeWidgetItem([title])
            group.setToolTip(0, title)
            self.info_root_item.addChild(group)
            if values:
                for key, value in values.items():
                    text = f"{key}: {value}"
                    child = QTreeWidgetItem([text])
                    child.setToolTip(0, text)
                    group.addChild(child)
            else:
                child = QTreeWidgetItem(["-"])
                child.setToolTip(0, "-")
                group.addChild(child)
            group.setExpanded(True)
        self.info_root_item.setExpanded(True)
