from __future__ import annotations

from PySide6.QtCore import QLocale, Qt, Signal
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QSizePolicy, QWidget


class NumericLineEdit(QWidget):
    valueChanged = Signal(float)

    def __init__(
        self,
        minimum: float = -1e12,
        maximum: float = 1e12,
        value: float = 0.0,
        decimals: int = 2,
        unit: str = "",
        integer: bool = False,
    ) -> None:
        super().__init__()
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.decimals = 0 if integer else int(decimals)
        self.integer = integer
        self._value = 0.0

        self.line_edit = QLineEdit()
        self.line_edit.setMinimumWidth(90)
        self.line_edit.setMaximumWidth(16777215)
        self.line_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.line_edit.setAlignment(Qt.AlignLeft)
        self.line_edit.setValidator(self._validator())
        self.line_edit.editingFinished.connect(self._commit_text)
        self.line_edit.textChanged.connect(self._handle_text_changed)

        self.unit_label = QLabel(unit)
        self.unit_label.setMinimumWidth(0)
        self.unit_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self.line_edit, 1)
        layout.addWidget(self.unit_label)
        self.setMinimumWidth(0)
        self.setMaximumWidth(16777215)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setValue(value)

    def value(self):
        if self.integer:
            return int(round(self._value))
        return float(self._value)

    def setValue(self, value) -> None:
        self._set_value(float(value), emit=False)

    def setRange(self, minimum: float, maximum: float) -> None:
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.line_edit.setValidator(self._validator())
        self.setValue(self._value)

    def setMinimum(self, minimum: float) -> None:
        self.setRange(minimum, self.maximum)

    def setMaximum(self, maximum: float) -> None:
        self.setRange(self.minimum, maximum)

    def setDecimals(self, decimals: int) -> None:
        self.decimals = 0 if self.integer else int(decimals)
        self.line_edit.setValidator(self._validator())
        self.setValue(self._value)

    def setSingleStep(self, _step: float) -> None:
        return

    def text(self) -> str:
        return self.line_edit.text()

    def setText(self, text: str) -> None:
        self.line_edit.setText(text)
        self._commit_text()

    def _validator(self) -> QDoubleValidator:
        validator = QDoubleValidator(self.minimum, self.maximum, self.decimals, self)
        validator.setNotation(QDoubleValidator.StandardNotation)
        locale = QLocale(QLocale.C)
        locale.setNumberOptions(QLocale.RejectGroupSeparator)
        validator.setLocale(locale)
        return validator

    def _handle_text_changed(self, text: str) -> None:
        state, _, _ = self.line_edit.validator().validate(text, 0)
        if state != QDoubleValidator.Acceptable:
            return
        value = float(text)
        if self.integer:
            value = round(value)
        if value == self._value:
            return
        self._value = value
        self.valueChanged.emit(self.value())

    def _commit_text(self) -> None:
        text = self.line_edit.text().strip()
        try:
            value = float(text)
        except ValueError:
            value = self._value
        self._set_value(value, emit=True)

    def _set_value(self, value: float, emit: bool) -> None:
        value = min(max(value, self.minimum), self.maximum)
        if self.integer:
            value = round(value)
        changed = value != self._value
        self._value = value
        self.line_edit.blockSignals(True)
        self.line_edit.setText(self._format_value(value))
        self.line_edit.blockSignals(False)
        if emit and changed:
            self.valueChanged.emit(self.value())

    def _format_value(self, value: float) -> str:
        if self.integer:
            return str(int(round(value)))
        text = f"{value:.{self.decimals}f}"
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return text or "0"
