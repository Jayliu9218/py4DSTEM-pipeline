from __future__ import annotations

from collections.abc import Callable

from PySide6.QtWidgets import QWidget

from app.services.workflow_state import WorkflowState, WorkflowStep
from app.widgets.pipeline_shell import (
    ModuleControlPanel,
    ProjectToolbar,
    RouteModule,
    TechnicalRouteBar,
)


GOALS_BY_STRUCTURE = {
    "Crystalline / Bragg-based": [
        "Orientation Mapping",
        "Strain Mapping",
        "Structural Phase Mapping",
    ],
    "Amorphous / Diffuse-scattering": ["RDF", "FEM", "Amorphous Strain"],
    "Phase Retrieval / Ptychography": [
        "DPC / CoM",
        "Parallax",
        "Ptychography",
        "Method Comparison",
    ],
}


def build_route_modules(structure: str, goal: str) -> list[RouteModule]:
    crystalline_route = structure == "Crystalline / Bragg-based"
    common = RouteModule(
        "data_setup",
        "Data & Preprocess" if crystalline_route else "Data Setup",
        "preprocess" if crystalline_route else "overview",
        "Open an HDF5 / EMD file, assign dataset roles, inspect the DataCube, and preview/apply preprocessing.",
        "Validated roles, DataCube diagnostics, and an explicitly preprocessed working DataCube.",
    )
    if crystalline_route:
        return _crystalline_modules(common, goal)
    if structure == "Amorphous / Diffuse-scattering":
        return _amorphous_modules(common, goal)
    if structure == "Phase Retrieval / Ptychography":
        return _phase_retrieval_modules(common, goal)
    return [common]


def _crystalline_modules(common: RouteModule, goal: str) -> list[RouteModule]:
    shared = [
        common,
        RouteModule("virtual_imaging", "Virtual Imaging", "virtual",
            "Target DataCube; measured probe geometry is optional.",
            "BF, annular/off-axis DF, and ROI virtual diffraction.",
            WorkflowStep.VIRTUAL_DETECTOR, "data_setup"),
        RouteModule("bragg_detection", "Probe & Bragg", "bragg",
            "Virtual image and target DataCube; vacuum probe or vacuum ROI recommended.",
            "Probe kernel, target/reference BraggVectors, histograms, and diagnostics.",
            WorkflowStep.BRAGG_FULL, "virtual_imaging"),
        RouteModule("calibration", "Calibration", "calibration",
            "Target and ellipse-reference BraggVectors plus rotation reference.",
            "Applied origin, ellipse, pixel-size, and rotation calibration.",
            WorkflowStep.CALIBRATION_APPLY, "bragg_detection"),
    ]
    if goal == "Strain Mapping":
        return shared + [
            RouteModule("crystal_analysis", "Strain Analysis", "strain",
                "Target BraggVectors; calibration is recommended for accuracy.",
                "Basis diagnostics, strain components, quality maps, and references.",
                WorkflowStep.STRAIN_MAP, "calibration"),
            RouteModule("crystalline_results", "Results & Quality", "crystalline_results",
                "Completed strain analysis.",
                "Filtered final results, quality maps, and diagnostics.",
                WorkflowStep.STRAIN_MAP, "crystal_analysis"),
            RouteModule("export", "Export", "crystalline_results",
                "Reviewed crystalline results.",
                "Result arrays, project state, and scientific report.",
                prerequisite="crystalline_results"),
        ]
    if goal == "Orientation Mapping":
        return shared + [
            RouteModule("orientation_setup", "Orientation Setup & Validation", "orientation",
                "Calibrated BraggVectors and a CIF or manual crystal; incomplete calibration is allowed with warnings.",
                "Reviewed candidates and an explicitly accepted single-pattern match.",
                WorkflowStep.ORIENTATION_REVIEW_ACCEPT, "calibration"),
            RouteModule("crystalline_results", "Results & Quality", "crystalline_results",
                "Explicitly accepted single-pattern match review.",
                "Orientation mapping and quality review.",
                WorkflowStep.ORIENTATION_MATCH, "orientation_setup"),
            RouteModule("export", "Export", "crystalline_results",
                "Reviewed crystalline results.",
                "Result arrays, project state, and scientific report.",
                prerequisite="crystalline_results"),
        ]
    page = {
        "Orientation Mapping": "orientation",
        "Structural Phase Mapping": "structural_phase",
    }.get(goal, "overview")
    step = {
        "Orientation Mapping": WorkflowStep.ORIENTATION_MATCH,
        "Structural Phase Mapping": WorkflowStep.STRUCTURAL_PHASE,
    }.get(goal)
    return [
        common,
        RouteModule("bragg_detection", "Bragg Detection", "bragg",
            "Target DataCube; probe kernel is optional but recommended.",
            "BraggVectors, BVM, peak tables, and detection diagnostics.",
            WorkflowStep.BRAGG_FULL, "data_setup"),
        RouteModule("calibration", "Calibration", "calibration",
            "BraggVectors; ellipse and rotation references when required.",
            "Applied origin, ellipse, pixel-size, and rotation calibration.",
            WorkflowStep.CALIBRATION_APPLY, "bragg_detection"),
        RouteModule("crystal_analysis", goal, page,
            "Calibrated BraggVectors and analysis-specific reference inputs.",
            f"{goal} result maps and quality diagnostics.",
            step, "calibration", goal != "Structural Phase Mapping"),
        RouteModule("export", "Export", "overview",
            "At least one registered result.",
            "Result arrays, images, project state, or scientific report.",
            prerequisite="crystal_analysis"),
    ]


def _amorphous_modules(common: RouteModule, goal: str) -> list[RouteModule]:
    page = {"RDF": "rdf", "FEM": "fem", "Amorphous Strain": "amorphous_strain"}.get(
        goal, "overview"
    )
    return [
        common,
        RouteModule("radial_profile", "Radial Profile", "radial_profile",
            "Target DataCube for radial analysis.",
            "Radial intensity profile and peak diagnostics.",
            WorkflowStep.RADIAL_PROFILE, "data_setup", implemented=False),
        RouteModule("amorphous_analysis", goal, page,
            "Validated radial profile and analysis-specific parameters.",
            f"{goal} maps and diagnostics.",
            prerequisite="radial_profile", implemented=False),
        RouteModule("export", "Export", "overview",
            "At least one registered result.",
            "Result arrays, images, project state, or scientific report.",
            prerequisite="amorphous_analysis"),
    ]


def _phase_retrieval_modules(common: RouteModule, goal: str) -> list[RouteModule]:
    if goal == "DPC / CoM":
        return [
            common,
            RouteModule("bf_df_preview", "BF / DF Preview", "bf_df",
                "Target DataCube for bright-field / dark-field virtual imaging.",
                "BF and DF virtual images for phase retrieval workflow entry.",
                WorkflowStep.BF_DF_PREVIEW, "data_setup"),
            RouteModule("dpc_segmented", "Segmented DPC", "dpc_segmented",
                "Target DataCube; BF/DF preview recommended for mask calibration.",
                "Four masks, segment intensities, opposing DPC, and weighted CoM.",
                WorkflowStep.DPC_SEGMENTED, "data_setup"),
            RouteModule("dpc_preprocess", "CoM Preprocessing & Review", "dpc_preprocess",
                "Target DataCube; segmented DPC is an optional demonstration.",
                "Measured, fitted, normalized, corrected CoM, review, and explicit acceptance.",
                WorkflowStep.DPC_REVIEW, "data_setup"),
            RouteModule("dpc", "Integrated Reconstruction", "dpc",
                "Accepted CoM preprocessing review.",
                "Integrated DPC potential, convergence, and stored iterations.",
                WorkflowStep.DPC, "dpc_preprocess"),
            RouteModule("export", "Export", "overview",
                "At least one registered result.",
                "Intermediate and final DPC results, project state, and report.",
                prerequisite="dpc"),
        ]
    if goal == "Parallax":
        return [
            common,
            RouteModule("parallax_bf", "BF Disk & Virtual BF", "parallax_bf",
                "Target DataCube.", "Reviewed and explicitly accepted bright-field disk mask.",
                WorkflowStep.PARALLAX_BF_ACCEPT, "data_setup"),
            RouteModule("parallax_alignment", "Parallax Alignment", "parallax_alignment",
                "Accepted BF disk mask.", "Core cross-correlation Parallax alignment.",
                WorkflowStep.PARALLAX_ALIGNMENT, "parallax_bf"),
            RouteModule("parallax_review", "Alignment Review", "parallax_review",
                "Completed Parallax alignment.",
                "Reviewed shifts and explicitly accepted aligned BF result.",
                WorkflowStep.PARALLAX_REVIEW, "parallax_alignment"),
            RouteModule("parallax_advanced", "Advanced Reconstruction", "parallax_advanced",
                "Accepted alignment review.",
                "Optional subpixel reconstruction and expert aberration processing.",
                WorkflowStep.PARALLAX_ADVANCED, "parallax_review"),
            RouteModule("export", "Export", "parallax_export",
                "Accepted core alignment or advanced reconstruction.",
                "Registered results and an explicitly saved Parallax package.",
                prerequisite="parallax_review"),
        ]
    if goal == "Ptychography":
        return [
            common,
            RouteModule("ptychography_data", "Data & Probe", "ptychography_data",
                "Target DataCube and optional vacuum probe.",
                "Data suitability diagnostics, probe source, and selected reusable profile.",
                WorkflowStep.PTYCHOGRAPHY_DATA, "data_setup"),
            RouteModule("ptychography_geometry", "Calibration / Geometry", "ptychography_geometry",
                "Reviewed DataCube and probe.", "Recorded automatic, existing, or manual geometry.",
                WorkflowStep.PTYCHOGRAPHY_GEOMETRY, "ptychography_data"),
            RouteModule("ptychography_preprocess", "Preprocess", "ptychography_preprocess",
                "Recorded geometry.", "Accepted coordinate alignment, probe, sampling, and field of view.",
                WorkflowStep.PTYCHOGRAPHY_PREPROCESS_ACCEPT, "ptychography_geometry"),
            RouteModule("ptychography_quick", "Quick Reconstruction", "ptychography_quick",
                "Accepted preprocessing.", "Low-cost diagnostic reconstruction retained for QC.",
                WorkflowStep.PTYCHOGRAPHY_QUICK, "ptychography_preprocess"),
            RouteModule("ptychography_review", "Review & QC", "ptychography_review",
                "Completed Quick Reconstruction.", "QC metrics, risk guidance, and explicit human acceptance.",
                WorkflowStep.PTYCHOGRAPHY_QC_ACCEPT, "ptychography_quick"),
            RouteModule("ptychography_optimization", "Parameter Optimization", "ptychography_optimization",
                "Accepted preprocessing; optional expert branch.",
                "Best self-consistency candidate and selected value.",
                WorkflowStep.PTYCHOGRAPHY_OPTIMIZATION, "ptychography_review"),
            RouteModule("ptychography_advanced", "Advanced Reconstruction", "ptychography_advanced",
                "Accepted QC; optimization is optional.",
                "Retained formal single-slice, constrained, or mixed-state reconstruction.",
                WorkflowStep.PTYCHOGRAPHY_ADVANCED, "ptychography_review"),
            RouteModule("export", "Export", "ptychography_export",
                "Completed Advanced Reconstruction.", "NPZ, JSON, registered images, and optional native HDF5.",
                WorkflowStep.PTYCHOGRAPHY_EXPORT, "ptychography_advanced"),
        ]
    return [
        common,
        RouteModule("bf_df_preview", "BF / DF Preview", "bf_df",
            "Target DataCube for bright-field / dark-field virtual imaging.",
            "BF and DF virtual images for phase retrieval workflow entry.",
            WorkflowStep.BF_DF_PREVIEW, "data_setup"),
        RouteModule("dpc", "DPC / CoM", "dpc_legacy",
            "Target DataCube; DPC recommended for downstream initialization.",
            "Integrated DPC result when available.", WorkflowStep.DPC, "data_setup"),
        RouteModule("parallax", "Parallax", "parallax",
            "Target DataCube; DPC recommended for rotation and defocus estimates.",
            "Aligned bright-field image, shift maps, and aberration estimates.",
            WorkflowStep.PARALLAX, "dpc"),
        RouteModule("ptychography", "Ptychography", "ptychography",
            "Target DataCube and optional vacuum probe.",
            "Phase and amplitude reconstruction from iterative ptychography.",
            WorkflowStep.PTYCHOGRAPHY, "data_setup"),
        RouteModule("method_comparison", "Method Comparison", "method_comparison",
            "At least one retained DPC or Ptychography result.",
            "Side-by-side comparison of retained phase retrieval outputs.",
            WorkflowStep.METHOD_COMPARISON, "ptychography"),
        RouteModule("export", "Export", "overview",
            "At least one registered result.",
            "Result arrays, images, project state, or scientific report.",
            prerequisite="method_comparison"),
    ]


class RouteCoordinator:
    def __init__(
        self,
        toolbar: ProjectToolbar,
        route_bar: TechnicalRouteBar,
        module_panel: ModuleControlPanel,
        viewer_stack,
        viewer_pages: dict[str, QWidget],
        workflow_state: WorkflowState,
        controls_provider: Callable[[str, str], QWidget | None],
        workspace_provider: Callable[[QWidget], object | None],
        style_refresher: Callable[[], None],
        data_ready_provider: Callable[[], bool],
    ) -> None:
        self.toolbar = toolbar
        self.route_bar = route_bar
        self.module_panel = module_panel
        self.viewer_stack = viewer_stack
        self.viewer_pages = viewer_pages
        self.workflow_state = workflow_state
        self.controls_provider = controls_provider
        self.workspace_provider = workspace_provider
        self.style_refresher = style_refresher
        self.data_ready_provider = data_ready_provider
        self.current_key = "data_setup"
        self.modules: list[RouteModule] = []

    def update_structure(self) -> None:
        structure = self.toolbar.structure.currentText()
        goals = GOALS_BY_STRUCTURE[structure]
        if not self.toolbar.goal.count() or self.toolbar.goal.currentText() not in goals:
            self.toolbar.set_goals(goals)
        goal = self.toolbar.goal.currentText() or goals[0]
        self.modules = build_route_modules(structure, goal)
        self.route_bar.set_modules(self.modules)
        if self.current_key not in {module.key for module in self.modules}:
            self.current_key = "data_setup"
        self.refresh()

    def states(self) -> dict[str, str]:
        states: dict[str, str] = {}
        for module in self.modules:
            if module.state_step and self.workflow_state.is_stale(module.state_step):
                states[module.key] = "Warning"
            elif module.state_step and self.workflow_state.is_completed(module.state_step):
                states[module.key] = "Completed"
            else:
                states[module.key] = "Ready"
        return states

    def select(self, key: str) -> None:
        self.current_key = key
        self.refresh()

    def refresh(self) -> None:
        if not self.modules:
            return
        self.route_bar.update_states(self.states(), self.current_key)
        module = next(item for item in self.modules if item.key == self.current_key)
        page = self.viewer_pages[module.page_key]
        refresh_stage = getattr(page, "refresh_stage", None)
        if callable(refresh_stage):
            refresh_stage()
        self.viewer_stack.setCurrentWidget(page)
        self.module_panel.set_module(
            module, self.controls_provider(module.key, self.toolbar.goal.currentText())
        )
        workspace = self.workspace_provider(page)
        if workspace is not None:
            workspace.refresh_layout()
        self.style_refresher()
