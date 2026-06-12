from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from PySide6.QtCore import QObject, Signal


STALE_RESULTS_MESSAGE = (
    "Parameters have been updated, but the calculation update has not yet been completed."
)


class WorkflowStep:
    DATA_ROLES = "data_roles"
    PREPROCESS_APPLY = "preprocess_apply"
    VIRTUAL_DETECTOR = "virtual_detector"
    VIRTUAL_DIFFRACTION = "virtual_diffraction"
    PROBE_KERNEL = "probe_kernel"
    BRAGG_SINGLE = "bragg_single"
    BRAGG_SELECTED = "bragg_selected"
    BRAGG_FULL = "bragg_full"
    BRAGG_VECTOR_MAP = "bragg_vector_map"
    CALIBRATION_ORIGIN = "calibration_origin"
    CALIBRATION_ELLIPSE = "calibration_ellipse"
    CALIBRATION_PIXEL = "calibration_pixel"
    CALIBRATION_ROTATION = "calibration_rotation"
    CALIBRATION_APPLY = "calibration_apply"
    ORIENTATION_PLAN = "orientation_plan"
    ORIENTATION_MATCH = "orientation_match"
    STRAIN_MAP = "strain_map"
    STRUCTURAL_PHASE = "structural_phase"
    PHASE_CONTRAST = "phase_contrast"
    BF_DF_PREVIEW = "bf_df_preview"
    DPC = "dpc"
    PARALLAX = "parallax"
    PTYCHOGRAPHY = "ptychography"
    METHOD_COMPARISON = "method_comparison"
    RADIAL_PROFILE = "radial_profile"
    RDF = "rdf"
    FEM = "fem"
    AMORPHOUS_STRAIN = "amorphous_strain"


@dataclass(frozen=True)
class DatasetRoles:
    target_datacube: str | None = None
    polycrystal_calibration: str | None = None
    vacuum_probe: str | None = None
    defocused_cbed: str | None = None

    def with_role(self, role: str, path: str | None) -> "DatasetRoles":
        if not hasattr(self, role):
            raise ValueError(f"Unsupported dataset role: {role}")
        return DatasetRoles(
            target_datacube=path if role == "target_datacube" else self.target_datacube,
            polycrystal_calibration=(
                path if role == "polycrystal_calibration" else self.polycrystal_calibration
            ),
            vacuum_probe=path if role == "vacuum_probe" else self.vacuum_probe,
            defocused_cbed=path if role == "defocused_cbed" else self.defocused_cbed,
        )


class WorkflowState(QObject):
    changed = Signal()

    _DEPENDENCIES = {
        WorkflowStep.DATA_ROLES: {
            WorkflowStep.PREPROCESS_APPLY,
            WorkflowStep.VIRTUAL_DETECTOR,
            WorkflowStep.VIRTUAL_DIFFRACTION,
            WorkflowStep.PROBE_KERNEL,
            WorkflowStep.BRAGG_SINGLE,
            WorkflowStep.BRAGG_SELECTED,
            WorkflowStep.BRAGG_FULL,
            WorkflowStep.CALIBRATION_ORIGIN,
            WorkflowStep.CALIBRATION_ELLIPSE,
            WorkflowStep.CALIBRATION_PIXEL,
            WorkflowStep.CALIBRATION_ROTATION,
            WorkflowStep.CALIBRATION_APPLY,
            WorkflowStep.ORIENTATION_PLAN,
            WorkflowStep.ORIENTATION_MATCH,
            WorkflowStep.STRAIN_MAP,
            WorkflowStep.BF_DF_PREVIEW,
        },
        WorkflowStep.PREPROCESS_APPLY: {
            WorkflowStep.VIRTUAL_DETECTOR,
            WorkflowStep.VIRTUAL_DIFFRACTION,
            WorkflowStep.PROBE_KERNEL,
            WorkflowStep.BRAGG_SINGLE,
            WorkflowStep.BRAGG_SELECTED,
            WorkflowStep.BRAGG_FULL,
            WorkflowStep.CALIBRATION_ORIGIN,
            WorkflowStep.CALIBRATION_ELLIPSE,
            WorkflowStep.CALIBRATION_PIXEL,
            WorkflowStep.CALIBRATION_ROTATION,
            WorkflowStep.CALIBRATION_APPLY,
            WorkflowStep.STRAIN_MAP,
        },
        WorkflowStep.PROBE_KERNEL: {
            WorkflowStep.BRAGG_SINGLE,
            WorkflowStep.BRAGG_SELECTED,
            WorkflowStep.BRAGG_FULL,
        },
        WorkflowStep.BRAGG_FULL: {
            WorkflowStep.BRAGG_VECTOR_MAP,
            WorkflowStep.CALIBRATION_ORIGIN,
            WorkflowStep.CALIBRATION_ELLIPSE,
            WorkflowStep.CALIBRATION_PIXEL,
            WorkflowStep.CALIBRATION_ROTATION,
            WorkflowStep.STRAIN_MAP,
        },
        WorkflowStep.BRAGG_VECTOR_MAP: {
            WorkflowStep.CALIBRATION_ORIGIN,
            WorkflowStep.CALIBRATION_APPLY,
            WorkflowStep.ORIENTATION_MATCH,
            WorkflowStep.STRAIN_MAP,
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
        WorkflowStep.BF_DF_PREVIEW: {WorkflowStep.DPC},
        WorkflowStep.DPC: {WorkflowStep.PARALLAX},
        WorkflowStep.PARALLAX: {WorkflowStep.PTYCHOGRAPHY},
    }

    def __init__(self) -> None:
        super().__init__()
        self._completed: set[str] = set()
        self._stale: set[str] = set()
        self.dataset_roles = DatasetRoles()

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

    def set_dataset_role(self, role: str, path: str | None) -> None:
        next_roles = self.dataset_roles.with_role(role, path)
        if next_roles == self.dataset_roles:
            return
        self.dataset_roles = next_roles
        self.parameters_updated(WorkflowStep.DATA_ROLES)
        if not self._completed:
            self.changed.emit()

    def is_stale(self, step: str) -> bool:
        return step in self._stale

    def is_completed(self, step: str) -> bool:
        return step in self._completed

    def prerequisite_message(self, required_steps: Iterable[str]) -> str:
        missing = [step.replace("_", " ") for step in required_steps if step not in self._completed]
        stale = [step.replace("_", " ") for step in required_steps if step in self._stale]
        if missing:
            return f"Complete required step(s) first: {', '.join(missing)}."
        if stale:
            return f"Recalculate stale required step(s): {', '.join(stale)}."
        return ""

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
