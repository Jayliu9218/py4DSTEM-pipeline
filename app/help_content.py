from __future__ import annotations

from app.version import RELEASE_CHANNEL, __version__


ABOUT_HTML = f"""
<h2>py4DSTEM Pipeline</h2>
<p><b>Version:</b> {__version__} ({RELEASE_CHANNEL})</p>
<p>A desktop workflow application for browsing, processing, reconstructing,
reviewing, and exporting 4D-STEM data with py4DSTEM.</p>
<h3>Current situation</h3>
<p>The application provides guided crystalline and phase-retrieval workflows,
shared calculation progress, project-state persistence, scientific diagnostics,
and result export. The amorphous-analysis routes are visible but remain under
development.</p>
<h3>Current improvements</h3>
<ul>
  <li>Staged workflows with explicit review and acceptance gates.</li>
  <li>CPU/GPU execution choices and clearer CUDA or memory failure guidance.</li>
  <li>Thread-safe background calculations with live progress reporting.</li>
  <li>Reusable Ptychography profiles, Quick Reconstruction, QC, and Advanced Reconstruction.</li>
</ul>
<p>Results should always be reviewed using appropriate experimental knowledge;
automated diagnostics support scientific judgment but do not replace it.</p>
"""

LICENSE_HTML = """
<h2>License</h2>
<p>This project is intended for distribution under the
<b>GNU General Public License version 3 (GPLv3)</b>.</p>
<p>py4DSTEM is open source software distributed under a GPLv3 license. It is
free to use, alter, or build on, provided that any work derived from py4DSTEM
is also kept free and open under a GPLv3 license.</p>
<p>Reference:
<a href="https://www.gnu.org/licenses/gpl-3.0.html">GNU GPLv3 license</a><br>
py4DSTEM:
<a href="https://github.com/py4dstem/py4DSTEM">github.com/py4dstem/py4DSTEM</a></p>
"""

TUTORIAL_HTML = """
<h2>Workflow Tutorial</h2>
<p>Start by opening an HDF5/EMD file, assigning the Target DataCube and any
optional reference roles, then choose a structure type and analysis goal.</p>
<h3>Shared Data Setup</h3>
<p>Select a DataCube and click <b>Show Data</b> to activate and assign it as
the Target DataCube. Tree selection remains lazy; <b>Preview Selected</b>
displays only the selected slice or scan position. Review preprocessing and
apply corrections explicitly before downstream analysis.</p>
<h3>Crystalline / Bragg-based</h3>
<p><b>Orientation Mapping:</b> detect Bragg peaks, calibrate reciprocal space,
load a crystal structure, create an orientation plan, and match orientations.</p>
<p><b>Strain Mapping:</b> generate virtual images, prepare a probe kernel,
calculate BraggVectors, apply calibration, select a reference, and calculate
strain and quality maps.</p>
<p><b>Structural Phase Mapping:</b> uses calibrated Bragg information for
phase-specific analysis; this route is still being expanded.</p>
<h3>Phase Retrieval / Ptychography</h3>
<p><b>DPC / CoM:</b> preview BF/DF contrast, inspect segmented DPC, preprocess
and accept CoM fields, then run integrated reconstruction.</p>
<p><b>Parallax:</b> accept a bright-field disk, align virtual BF images, review
shifts, and optionally run subpixel or aberration processing.</p>
<p><b>Ptychography:</b> inspect data and probe, accept geometry and preprocessing,
run a Quick Reconstruction, calculate QC metrics, and click <b>Confirm QC
Risks</b>. Parameter Optimization is optional. After optimization, click
<b>Apply Best Self-Consistency Value</b>; accepted QC remains valid and Advanced
Reconstruction becomes ready. Upstream optimized values rebuild preprocessing
inside the Advanced background task, while batch/probe options transfer directly.
The best self-consistency value is diagnostic and is not proof of physical accuracy.</p>
<p><b>Method Comparison:</b> compare retained DPC and Ptychography results when
both are available.</p>
<h3>Amorphous / Diffuse-scattering</h3>
<p>Radial Profile, RDF, FEM, and Amorphous Strain routes are planned workflows
and are not yet production-ready.</p>
<h3>Reading Workflow Status</h3>
<p>Completed stages are retained. Changing upstream parameters marks affected
downstream results as stale. Re-run and re-accept stale stages before relying
on later results.</p>
<h3>Performance and Warnings</h3>
<p>Large 4D reductions run in cancellable background tasks. Use the reduction
memory budget to control chunking, follow Progress and Warnings in Output, and
review scientific diagnostics before accepting a stage.</p>
"""
