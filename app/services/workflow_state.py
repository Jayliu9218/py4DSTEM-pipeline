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
    ORIENTATION_REVIEW = "orientation_review"
    ORIENTATION_REVIEW_ACCEPT = "orientation_review_accept"
    ORIENTATION_MATCH = "orientation_match"
    STRAIN_MAP = "strain_map"
    STRUCTURAL_PHASE_PLAN = "structural_phase_plan"
    STRUCTURAL_PHASE_MATCH = "structural_phase_match"
    STRUCTURAL_PHASE = "structural_phase"
    CRYSTAL_STRUCTURE_FACTORS = "crystal_structure_factors"
    CRYSTAL_SIMULATED_DIFFRACTION = "crystal_simulated_diffraction"
    CRYSTAL_PHASE = "crystal_phase"
    CRYSTAL_ORIENTATION = "crystal_orientation"
    CRYSTAL_GRAIN = "crystal_grain"
    CRYSTAL_STRAIN = "crystal_strain"
    PHASE_CONTRAST = "phase_contrast"
    BF_DF_PREVIEW = "bf_df_preview"
    DPC_SEGMENTED = "dpc_segmented"
    DPC_PREPROCESS = "dpc_preprocess"
    DPC_REVIEW = "dpc_review"
    DPC = "dpc"
    PARALLAX_BF = "parallax_bf"
    PARALLAX_BF_ACCEPT = "parallax_bf_accept"
    PARALLAX_ALIGNMENT = "parallax_alignment"
    PARALLAX_REVIEW = "parallax_review"
    PARALLAX_ADVANCED = "parallax_advanced"
    PARALLAX = "parallax"
    PARALLAX_EXPORT = "parallax_export"
    PTYCHOGRAPHY = "ptychography"
    PTYCHOGRAPHY_DATA = "ptychography_data"
    PTYCHOGRAPHY_GEOMETRY = "ptychography_geometry"
    PTYCHOGRAPHY_PREPROCESS = "ptychography_preprocess"
    PTYCHOGRAPHY_PREPROCESS_ACCEPT = "ptychography_preprocess_accept"
    PTYCHOGRAPHY_QUICK = "ptychography_quick"
    PTYCHOGRAPHY_QC = "ptychography_qc"
    PTYCHOGRAPHY_QC_ACCEPT = "ptychography_qc_accept"
    PTYCHOGRAPHY_OPTIMIZATION = "ptychography_optimization"
    PTYCHOGRAPHY_ADVANCED = "ptychography_advanced"
    PTYCHOGRAPHY_EXPORT = "ptychography_export"
    PTYCHOGRAPHY_SETUP = PTYCHOGRAPHY_DATA
    PTYCHOGRAPHY_RECONSTRUCTION = PTYCHOGRAPHY_ADVANCED
    PTYCHOGRAPHY_REVIEW = PTYCHOGRAPHY_QC
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
            WorkflowStep.ORIENTATION_REVIEW,
            WorkflowStep.ORIENTATION_REVIEW_ACCEPT,
            WorkflowStep.ORIENTATION_MATCH,
            WorkflowStep.STRAIN_MAP,
            WorkflowStep.STRUCTURAL_PHASE_PLAN,
            WorkflowStep.STRUCTURAL_PHASE_MATCH,
            WorkflowStep.STRUCTURAL_PHASE,
            WorkflowStep.CRYSTAL_STRUCTURE_FACTORS,
            WorkflowStep.CRYSTAL_SIMULATED_DIFFRACTION,
            WorkflowStep.CRYSTAL_PHASE,
            WorkflowStep.CRYSTAL_ORIENTATION,
            WorkflowStep.CRYSTAL_GRAIN,
            WorkflowStep.CRYSTAL_STRAIN,
            WorkflowStep.BF_DF_PREVIEW,
            WorkflowStep.DPC_SEGMENTED,
            WorkflowStep.DPC_PREPROCESS,
            WorkflowStep.DPC_REVIEW,
            WorkflowStep.DPC,
            WorkflowStep.PARALLAX,
            WorkflowStep.PARALLAX_BF,
            WorkflowStep.PARALLAX_BF_ACCEPT,
            WorkflowStep.PARALLAX_ALIGNMENT,
            WorkflowStep.PARALLAX_REVIEW,
            WorkflowStep.PARALLAX_ADVANCED,
            WorkflowStep.PARALLAX_EXPORT,
            WorkflowStep.PTYCHOGRAPHY,
            WorkflowStep.PTYCHOGRAPHY_DATA,
            WorkflowStep.PTYCHOGRAPHY_GEOMETRY,
            WorkflowStep.PTYCHOGRAPHY_PREPROCESS,
            WorkflowStep.PTYCHOGRAPHY_PREPROCESS_ACCEPT,
            WorkflowStep.PTYCHOGRAPHY_QUICK,
            WorkflowStep.PTYCHOGRAPHY_QC,
            WorkflowStep.PTYCHOGRAPHY_QC_ACCEPT,
            WorkflowStep.PTYCHOGRAPHY_OPTIMIZATION,
            WorkflowStep.PTYCHOGRAPHY_ADVANCED,
            WorkflowStep.PTYCHOGRAPHY_EXPORT,
            WorkflowStep.METHOD_COMPARISON,
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
            WorkflowStep.ORIENTATION_REVIEW,
            WorkflowStep.ORIENTATION_REVIEW_ACCEPT,
            WorkflowStep.ORIENTATION_MATCH,
            WorkflowStep.STRAIN_MAP,
            WorkflowStep.STRUCTURAL_PHASE_PLAN,
            WorkflowStep.STRUCTURAL_PHASE_MATCH,
            WorkflowStep.STRUCTURAL_PHASE,
            WorkflowStep.CRYSTAL_STRUCTURE_FACTORS,
            WorkflowStep.CRYSTAL_SIMULATED_DIFFRACTION,
            WorkflowStep.CRYSTAL_PHASE,
            WorkflowStep.CRYSTAL_ORIENTATION,
            WorkflowStep.CRYSTAL_GRAIN,
            WorkflowStep.CRYSTAL_STRAIN,
            WorkflowStep.DPC_SEGMENTED,
            WorkflowStep.DPC_PREPROCESS,
            WorkflowStep.DPC_REVIEW,
            WorkflowStep.DPC,
            WorkflowStep.PARALLAX,
            WorkflowStep.PARALLAX_BF,
            WorkflowStep.PARALLAX_BF_ACCEPT,
            WorkflowStep.PARALLAX_ALIGNMENT,
            WorkflowStep.PARALLAX_REVIEW,
            WorkflowStep.PARALLAX_ADVANCED,
            WorkflowStep.PARALLAX_EXPORT,
            WorkflowStep.PTYCHOGRAPHY,
            WorkflowStep.PTYCHOGRAPHY_DATA,
            WorkflowStep.PTYCHOGRAPHY_GEOMETRY,
            WorkflowStep.PTYCHOGRAPHY_PREPROCESS,
            WorkflowStep.PTYCHOGRAPHY_PREPROCESS_ACCEPT,
            WorkflowStep.PTYCHOGRAPHY_QUICK,
            WorkflowStep.PTYCHOGRAPHY_QC,
            WorkflowStep.PTYCHOGRAPHY_QC_ACCEPT,
            WorkflowStep.PTYCHOGRAPHY_OPTIMIZATION,
            WorkflowStep.PTYCHOGRAPHY_ADVANCED,
            WorkflowStep.PTYCHOGRAPHY_EXPORT,
            WorkflowStep.METHOD_COMPARISON,
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
            WorkflowStep.ORIENTATION_REVIEW,
            WorkflowStep.ORIENTATION_REVIEW_ACCEPT,
            WorkflowStep.ORIENTATION_MATCH,
            WorkflowStep.STRAIN_MAP,
            WorkflowStep.STRUCTURAL_PHASE_PLAN,
            WorkflowStep.CRYSTAL_PHASE,
            WorkflowStep.CRYSTAL_ORIENTATION,
            WorkflowStep.CRYSTAL_GRAIN,
            WorkflowStep.CRYSTAL_STRAIN,
        },
        WorkflowStep.BRAGG_VECTOR_MAP: {
            WorkflowStep.CALIBRATION_ORIGIN,
            WorkflowStep.CALIBRATION_APPLY,
            WorkflowStep.ORIENTATION_REVIEW,
            WorkflowStep.ORIENTATION_REVIEW_ACCEPT,
            WorkflowStep.ORIENTATION_MATCH,
            WorkflowStep.STRAIN_MAP,
            WorkflowStep.STRUCTURAL_PHASE_PLAN,
            WorkflowStep.CRYSTAL_PHASE,
            WorkflowStep.CRYSTAL_ORIENTATION,
            WorkflowStep.CRYSTAL_GRAIN,
            WorkflowStep.CRYSTAL_STRAIN,
        },
        WorkflowStep.CALIBRATION_ORIGIN: {WorkflowStep.CALIBRATION_APPLY},
        WorkflowStep.CALIBRATION_ELLIPSE: {WorkflowStep.CALIBRATION_APPLY},
        WorkflowStep.CALIBRATION_PIXEL: {WorkflowStep.CALIBRATION_APPLY},
        WorkflowStep.CALIBRATION_ROTATION: {WorkflowStep.CALIBRATION_APPLY},
        WorkflowStep.CALIBRATION_APPLY: {
            WorkflowStep.ORIENTATION_REVIEW,
            WorkflowStep.ORIENTATION_REVIEW_ACCEPT,
            WorkflowStep.ORIENTATION_MATCH,
            WorkflowStep.STRAIN_MAP,
            WorkflowStep.STRUCTURAL_PHASE_PLAN,
            WorkflowStep.CRYSTAL_PHASE,
            WorkflowStep.CRYSTAL_ORIENTATION,
            WorkflowStep.CRYSTAL_GRAIN,
            WorkflowStep.CRYSTAL_STRAIN,
        },
        WorkflowStep.ORIENTATION_PLAN: {
            WorkflowStep.ORIENTATION_REVIEW,
            WorkflowStep.ORIENTATION_REVIEW_ACCEPT,
            WorkflowStep.ORIENTATION_MATCH,
        },
        WorkflowStep.ORIENTATION_REVIEW: {
            WorkflowStep.ORIENTATION_REVIEW_ACCEPT,
            WorkflowStep.ORIENTATION_MATCH,
        },
        WorkflowStep.ORIENTATION_REVIEW_ACCEPT: {WorkflowStep.ORIENTATION_MATCH},
        WorkflowStep.STRUCTURAL_PHASE_PLAN: {
            WorkflowStep.STRUCTURAL_PHASE_MATCH,
            WorkflowStep.STRUCTURAL_PHASE,
        },
        WorkflowStep.STRUCTURAL_PHASE_MATCH: {WorkflowStep.STRUCTURAL_PHASE},
        WorkflowStep.CRYSTAL_STRUCTURE_FACTORS: {
            WorkflowStep.CRYSTAL_SIMULATED_DIFFRACTION,
            WorkflowStep.CRYSTAL_PHASE,
            WorkflowStep.CRYSTAL_ORIENTATION,
            WorkflowStep.CRYSTAL_GRAIN,
            WorkflowStep.CRYSTAL_STRAIN,
        },
        WorkflowStep.CRYSTAL_SIMULATED_DIFFRACTION: {
            WorkflowStep.CRYSTAL_PHASE,
            WorkflowStep.CRYSTAL_ORIENTATION,
            WorkflowStep.CRYSTAL_GRAIN,
            WorkflowStep.CRYSTAL_STRAIN,
        },
        WorkflowStep.CRYSTAL_PHASE: {
            WorkflowStep.CRYSTAL_ORIENTATION,
            WorkflowStep.CRYSTAL_GRAIN,
            WorkflowStep.CRYSTAL_STRAIN,
        },
        WorkflowStep.CRYSTAL_ORIENTATION: {
            WorkflowStep.CRYSTAL_GRAIN,
            WorkflowStep.CRYSTAL_STRAIN,
        },
        WorkflowStep.CRYSTAL_GRAIN: {WorkflowStep.CRYSTAL_STRAIN},
        WorkflowStep.DPC_PREPROCESS: {WorkflowStep.DPC_REVIEW},
        WorkflowStep.DPC_REVIEW: {WorkflowStep.DPC},
        WorkflowStep.PARALLAX_BF: {WorkflowStep.PARALLAX_BF_ACCEPT},
        WorkflowStep.PARALLAX_BF_ACCEPT: {WorkflowStep.PARALLAX_ALIGNMENT},
        WorkflowStep.PARALLAX_ALIGNMENT: {WorkflowStep.PARALLAX_REVIEW},
        WorkflowStep.PARALLAX_REVIEW: {WorkflowStep.PARALLAX_ADVANCED, WorkflowStep.PARALLAX},
        WorkflowStep.PARALLAX: {WorkflowStep.PARALLAX_EXPORT, WorkflowStep.PTYCHOGRAPHY},
        WorkflowStep.PTYCHOGRAPHY_DATA: {WorkflowStep.PTYCHOGRAPHY_GEOMETRY},
        WorkflowStep.PTYCHOGRAPHY_GEOMETRY: {WorkflowStep.PTYCHOGRAPHY_PREPROCESS},
        WorkflowStep.PTYCHOGRAPHY_PREPROCESS: {WorkflowStep.PTYCHOGRAPHY_PREPROCESS_ACCEPT},
        WorkflowStep.PTYCHOGRAPHY_PREPROCESS_ACCEPT: {
            WorkflowStep.PTYCHOGRAPHY_OPTIMIZATION,
            WorkflowStep.PTYCHOGRAPHY_QUICK,
        },
        WorkflowStep.PTYCHOGRAPHY_QUICK: {WorkflowStep.PTYCHOGRAPHY_QC},
        WorkflowStep.PTYCHOGRAPHY_QC: {WorkflowStep.PTYCHOGRAPHY_QC_ACCEPT},
        WorkflowStep.PTYCHOGRAPHY_QC_ACCEPT: {WorkflowStep.PTYCHOGRAPHY_ADVANCED},
        WorkflowStep.PTYCHOGRAPHY_ADVANCED: {
            WorkflowStep.PTYCHOGRAPHY,
            WorkflowStep.PTYCHOGRAPHY_EXPORT,
        },
    }

    def __init__(self) -> None:
        super().__init__()
        self._completed: set[str] = set()
        self._stale: set[str] = set()
        self.dataset_roles = DatasetRoles()

    def mark_completed(self, step: str) -> None:
        self.mark_completed_many({step})

    def mark_completed_many(self, steps: Iterable[str]) -> None:
        completed = set(steps)
        downstream = self._with_downstream(completed) - completed
        self._stale.update(downstream & self._completed)
        self._completed.update(completed)
        self._stale.difference_update(completed)
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
