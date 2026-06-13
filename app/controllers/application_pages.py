from __future__ import annotations

from PySide6.QtWidgets import QWidget

from app.pages.amorphous_strain_page import AmorphousStrainPage
from app.pages.bf_df_preview_page import BFDFPreviewPage
from app.pages.bragg_peaks_page import BraggPeaksPage
from app.pages.calibration_page import CalibrationPage
from app.pages.dpc_page import DPCPage
from app.pages.fem_page import FEMPage
from app.pages.method_comparison_page import MethodComparisonPage
from app.pages.orientation_page import OrientationPage
from app.pages.parallax_page import ParallaxPage
from app.pages.phase_contrast_page import PhaseContrastPage
from app.pages.preprocessing_page import PreprocessingPage
from app.pages.ptychography_page import PtychographyPage
from app.pages.radial_profile_page import RadialProfilePage
from app.pages.rdf_page import RDFPage
from app.pages.strain_map_page import StrainMapPage
from app.pages.structural_phase_page import StructuralPhasePage
from app.pages.virtual_detector_page import VirtualDetectorPage
from app.services.parallax_service import ParallaxService
from app.services.phase_contrast_service import PhaseContrastService
from app.services.ptychography_service import PtychographyService
from app.widgets.adaptive_image_workspace import AdaptiveImageWorkspace


class ApplicationPages:
    """Registry for page lookup, route controls, and workspace lifecycle."""

    def __init__(
        self,
        viewer_pages: dict[str, QWidget],
        route_controls: dict[str, QWidget],
        crystal_controls: dict[str, QWidget],
        amorphous_controls: dict[str, QWidget],
        dpc_controls: dict[str, QWidget],
        export_controls: dict[str, QWidget],
    ) -> None:
        self.viewer_pages = viewer_pages
        self.route_controls = route_controls
        self.crystal_controls = crystal_controls
        self.amorphous_controls = amorphous_controls
        self.dpc_controls = dpc_controls
        self.export_controls = export_controls

    @staticmethod
    def build_page_objects(
        *,
        providers: dict[str, object],
        bragg_strain_service,
        log_panel,
        workflow_state,
        result_registry,
        phase_retrieval_results: dict[str, object],
    ) -> tuple[dict[str, QWidget], PhaseContrastService, ParallaxService]:
        common = {
            "log_panel": log_panel,
            "workflow_state": workflow_state,
            "result_registry": result_registry,
        }
        pages: dict[str, QWidget] = {}
        pages["virtual_detector_page"] = VirtualDetectorPage(
            source_provider=providers["virtual_source"],
            shape_provider=providers["shape"],
            probe_geometry_provider=providers["probe_geometry"],
            **common,
        )
        pages["preprocessing_page"] = PreprocessingPage(
            source_provider=providers["datacube"],
            selected_source_provider=providers["selected_source"],
            **common,
        )
        pages["bragg_peaks_page"] = BraggPeaksPage(
            datacube_provider=providers["datacube"],
            shape_provider=providers["shape"],
            virtual_image_provider=providers["virtual_image"],
            service=bragg_strain_service,
            **common,
        )
        pages["calibration_page"] = CalibrationPage(
            datacube_provider=providers["datacube"],
            braggvectors_provider=providers["braggvectors"],
            ellipse_braggvectors_provider=providers["ellipse_braggvectors"],
            transfer_targets_provider=providers["transfer_targets"],
            rotation_reference_provider=providers["rotation_reference"],
            service=bragg_strain_service,
            **common,
        )
        pages["strain_map_page"] = StrainMapPage(
            braggvectors_provider=providers["braggvectors"],
            service=bragg_strain_service,
            **common,
        )
        pages["orientation_page"] = OrientationPage(
            braggvectors_provider=providers["braggvectors"], **common
        )
        pages["phase_contrast_page"] = PhaseContrastPage(
            source_provider=providers["datacube"], **common
        )
        pages["bf_df_preview_page"] = BFDFPreviewPage(
            source_provider=providers["datacube"],
            shape_provider=providers["shape"],
            probe_geometry_provider=providers["probe_geometry"],
            **common,
        )

        dpc_service = PhaseContrastService()
        for name, stage in (
            ("dpc_segmented_page", "segmented"),
            ("dpc_preprocess_page", "preprocess"),
            ("dpc_reconstruction_page", "reconstruct"),
            ("dpc_legacy_page", "all"),
        ):
            pages[name] = DPCPage(
                source_provider=providers["datacube"],
                service=dpc_service,
                stage_mode=stage,
                **common,
            )
        pages["dpc_review_page"] = pages["dpc_preprocess_page"]
        pages["dpc_page"] = pages["dpc_reconstruction_page"]

        parallax_service = ParallaxService()
        for name, stage in (
            ("parallax_bf_page", "bf"),
            ("parallax_alignment_page", "alignment"),
            ("parallax_review_page", "review"),
            ("parallax_advanced_page", "advanced"),
            ("parallax_export_page", "export"),
        ):
            pages[name] = ParallaxPage(
                source_provider=providers["datacube"],
                service=parallax_service,
                stage_mode=stage,
                **common,
            )
        pages["parallax_page"] = pages["parallax_alignment_page"]
        ptychography_service = PtychographyService()
        for name, stage in (
            ("ptychography_data_page", "data"),
            ("ptychography_geometry_page", "geometry"),
            ("ptychography_preprocess_page", "preprocess"),
            ("ptychography_quick_page", "quick"),
            ("ptychography_review_page", "review"),
            ("ptychography_optimization_page", "optimization"),
            ("ptychography_advanced_page", "advanced"),
            ("ptychography_export_page", "export"),
        ):
            pages[name] = PtychographyPage(
                source_provider=providers["datacube"],
                vacuum_probe_provider=providers.get("vacuum_probe_path"),
                service=ptychography_service,
                stage_mode=stage,
                **common,
            )
        pages["ptychography_page"] = pages["ptychography_advanced_page"]
        pages["ptychography_setup_page"] = pages["ptychography_data_page"]
        pages["ptychography_reconstruction_page"] = pages["ptychography_advanced_page"]
        pages["method_comparison_page"] = MethodComparisonPage(
            dpc_result_provider=lambda: phase_retrieval_results.get("DPC"),
            ptychography_result_provider=lambda: phase_retrieval_results.get("Ptychography"),
            log_panel=log_panel,
            workflow_state=workflow_state,
        )
        pages["structural_phase_page"] = StructuralPhasePage()
        pages["radial_profile_page"] = RadialProfilePage()
        pages["rdf_page"] = RDFPage()
        pages["fem_page"] = FEMPage()
        pages["amorphous_strain_page"] = AmorphousStrainPage()
        return pages, dpc_service, parallax_service

    def controls_for_route(self, key: str, goal: str) -> QWidget | None:
        if key == "crystal_analysis":
            return self.crystal_controls.get(goal)
        if key == "amorphous_analysis":
            return self.amorphous_controls.get(goal)
        if key == "dpc":
            return self.dpc_controls.get(goal)
        if key == "export":
            return self.export_controls.get(goal, self.export_controls.get("default"))
        return self.route_controls.get(key)

    def named_workspaces(self) -> dict[str, AdaptiveImageWorkspace]:
        workspaces: dict[str, AdaptiveImageWorkspace] = {}
        for key, page in self.viewer_pages.items():
            workspace = self.workspace_for_page(page)
            if workspace is not None:
                workspaces[key] = workspace
        return workspaces

    @staticmethod
    def workspace_for_page(page: QWidget) -> AdaptiveImageWorkspace | None:
        if isinstance(page, AdaptiveImageWorkspace):
            return page
        for attribute in ("workspace", "viewers", "selected_grid"):
            workspace = getattr(page, attribute, None)
            if isinstance(workspace, AdaptiveImageWorkspace):
                return workspace
        return page.findChild(AdaptiveImageWorkspace)

    def grid_states(self) -> dict[str, dict[str, object]]:
        return {
            key: workspace.grid_state()
            for key, workspace in self.named_workspaces().items()
        }

    def clear_workspaces(self, exclude_keys: set[str] | None = None) -> None:
        excluded = set(exclude_keys or ())
        cleared: set[int] = set()
        for key, page in self.viewer_pages.items():
            if key in excluded or id(page) in cleared:
                continue
            cleared.add(id(page))
            clear_results = getattr(page, "clear_results", None)
            if callable(clear_results):
                clear_results()
                continue
            workspace = self.workspace_for_page(page)
            if workspace is not None:
                workspace.clear_results()
