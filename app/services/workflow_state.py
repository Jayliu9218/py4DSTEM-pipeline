from __future__ import annotations

from collections.abc import Iterable

from PySide6.QtCore import QObject, Signal


STALE_RESULTS_MESSAGE = (
    "Parameters have been updated, but the calculation update has not yet been completed."
)


class WorkflowStep:
    VIRTUAL_DETECTOR = "virtual_detector"
    PROBE_KERNEL = "probe_kernel"
    BRAGG_SINGLE = "bragg_single"
    BRAGG_SELECTED = "bragg_selected"
    BRAGG_FULL = "bragg_full"
    CALIBRATION_ORIGIN = "calibration_origin"
    CALIBRATION_ELLIPSE = "calibration_ellipse"
    CALIBRATION_PIXEL = "calibration_pixel"
    CALIBRATION_ROTATION = "calibration_rotation"
    CALIBRATION_APPLY = "calibration_apply"
    ORIENTATION_PLAN = "orientation_plan"
    ORIENTATION_MATCH = "orientation_match"
    STRAIN_MAP = "strain_map"


class WorkflowState(QObject):
    changed = Signal()

    _DEPENDENCIES = {
        WorkflowStep.PROBE_KERNEL: {
            WorkflowStep.BRAGG_SINGLE,
            WorkflowStep.BRAGG_SELECTED,
            WorkflowStep.BRAGG_FULL,
        },
        WorkflowStep.BRAGG_FULL: {
            WorkflowStep.CALIBRATION_ORIGIN,
            WorkflowStep.CALIBRATION_ELLIPSE,
            WorkflowStep.CALIBRATION_PIXEL,
            WorkflowStep.CALIBRATION_ROTATION,
        },
        WorkflowStep.CALIBRATION_ORIGIN: {WorkflowStep.CALIBRATION_APPLY},
        WorkflowStep.CALIBRATION_ELLIPSE: {WorkflowStep.CALIBRATION_APPLY},
        WorkflowStep.CALIBRATION_PIXEL: {WorkflowStep.CALIBRATION_APPLY},
        WorkflowStep.CALIBRATION_ROTATION: {WorkflowStep.CALIBRATION_APPLY},
        WorkflowStep.CALIBRATION_APPLY: {
            WorkflowStep.ORIENTATION_MATCH,
            WorkflowStep.STRAIN_MAP,
        },
        WorkflowStep.ORIENTATION_PLAN: {WorkflowStep.ORIENTATION_MATCH},
    }

    def __init__(self) -> None:
        super().__init__()
        self._completed: set[str] = set()
        self._stale: set[str] = set()

    def mark_completed(self, step: str) -> None:
        downstream = self._with_downstream({step}) - {step}
        self._stale.update(downstream & self._completed)
        self._completed.add(step)
        self._stale.discard(step)
        self.changed.emit()

    def parameters_updated(self, steps: str | Iterable[str]) -> None:
        roots = {steps} if isinstance(steps, str) else set(steps)
        newly_stale = self._with_downstream(roots) & self._completed
        if not newly_stale.issubset(self._stale):
            self._stale.update(newly_stale)
            self.changed.emit()

    def data_source_updated(self) -> None:
        if self._completed:
            self._stale.update(self._completed)
            self.changed.emit()

    def is_stale(self, step: str) -> bool:
        return step in self._stale

    def any_stale(self, steps: Iterable[str]) -> bool:
        return bool(self._stale.intersection(steps))

    def watch(self, widget: QObject, steps: str | Iterable[str], signal_name: str) -> None:
        getattr(widget, signal_name).connect(lambda *_: self.parameters_updated(steps))

    def _with_downstream(self, roots: set[str]) -> set[str]:
        affected = set(roots)
        pending = list(roots)
        while pending:
            for downstream in self._DEPENDENCIES.get(pending.pop(), set()):
                if downstream not in affected:
                    affected.add(downstream)
                    pending.append(downstream)
        return affected
