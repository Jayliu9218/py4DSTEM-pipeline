from __future__ import annotations

import h5py
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem


class Hdf5TreeWidget(QTreeWidget):
    node_selected = Signal(str, str)

    def __init__(self) -> None:
        super().__init__()
        self.setHeaderLabels(["HDF5 tree"])
        self.currentItemChanged.connect(self._emit_selected_node)
        self.info_root_item: QTreeWidgetItem | None = None

    def populate(self, hdf5_file: h5py.File) -> None:
        self.clear()
        root_item = QTreeWidgetItem(["/"])
        root_item.setData(0, 256, "/")
        root_item.setData(0, 257, "group")
        self.addTopLevelItem(root_item)

        self._add_children(root_item, hdf5_file, "/")
        root_item.setExpanded(True)
        self.info_root_item = QTreeWidgetItem(["Data info"])
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
            item.setData(0, 256, node_path)
            item.setData(0, 257, node_kind)
            parent_item.addChild(item)

            if isinstance(node, h5py.Group):
                self._add_children(item, node, node_path)

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
            self.info_root_item.addChild(group)
            if values:
                for key, value in values.items():
                    group.addChild(QTreeWidgetItem([f"{key}: {value}"]))
            else:
                group.addChild(QTreeWidgetItem(["-"]))
            group.setExpanded(True)
        self.info_root_item.setExpanded(True)
