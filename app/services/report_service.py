from __future__ import annotations

from datetime import datetime
from pathlib import Path

from app.services.project_state_service import ProjectState
from app.services.result_registry import ResultRegistry


class ReportService:
    def generate_markdown(
        self,
        path: str | Path,
        project_state: ProjectState,
        result_registry: ResultRegistry,
        activity_log: str = "",
        process_log: str = "",
    ) -> Path:
        output_path = Path(path)
        output_path.write_text(
            self.render_markdown(project_state, result_registry, activity_log, process_log),
            encoding="utf-8",
        )
        return output_path

    def render_markdown(
        self,
        project_state: ProjectState,
        result_registry: ResultRegistry,
        activity_log: str = "",
        process_log: str = "",
    ) -> str:
        lines = [
            "# py4DSTEM Pipeline Report",
            "",
            f"Generated: {datetime.now().isoformat(timespec='seconds')}",
            "",
            "## Project",
            "",
            f"- Data file: {project_state.file_path or '-'}",
            f"- Selected HDF5 path: {project_state.selected_hdf5_path or '-'}",
            f"- Image scaling: {project_state.image_scaling}",
            f"- Image colormap: {project_state.image_cmap}",
            f"- CUDA: {'enabled' if project_state.cuda_enabled else 'disabled'}",
            "",
            "## Dataset Roles",
            "",
        ]
        for role, value in sorted(project_state.dataset_roles.items()):
            lines.append(f"- {role}: {value or '-'}")

        lines.extend(["", "## Parameters", ""])
        if project_state.page_params:
            for page, params in sorted(project_state.page_params.items()):
                lines.append(f"### {page}")
                for key, value in sorted(params.items()):
                    lines.append(f"- {key}: {value}")
                lines.append("")
        else:
            lines.append("- No page parameters recorded.")
            lines.append("")

        lines.extend(["## Registered Results", ""])
        entries = result_registry.list_entries()
        if entries:
            for entry in entries:
                formats = ", ".join(entry.export_formats)
                lines.append(f"- {entry.key}: {formats}")
        else:
            lines.append("- No results have been registered yet.")

        lines.extend(["", "## Activity Log", "", "```text", activity_log.strip() or "-", "```"])
        lines.extend(["", "## Calculation Process", "", "```text", process_log.strip() or "-", "```"])
        lines.append("")
        return "\n".join(lines)
