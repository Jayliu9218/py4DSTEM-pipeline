# Workflow Guide

The application uses a left-to-right route bar. Green stages are complete,
blue is the current stage, and stale stages must be recalculated before their
results are trusted.

## Data Setup

Open an `.h5`, `.hdf5`, or `.emd` file and select a DataCube in the HDF5 tree.
Click **Show Data** to activate it and assign it as the Target DataCube. Tree
selection alone remains lazy and does not calculate a full 4D reduction.

![Ptychography data setup](ptychography%20data%20setup.png)

Use **Preview Selected** when only a selected diffraction slice or scan
position is needed. Use dataset-role buttons for optional vacuum-probe,
ellipse-reference, and rotation-reference inputs.

## Ptychography

Recommended route:

1. **Data & Probe**: inspect suitability, choose a profile and probe source.
2. **Calibration / Geometry**: accept automatic, existing, or manual geometry.
3. **Preprocess**: run preprocessing, inspect outputs, then explicitly accept.
4. **Quick Reconstruction**: run a low-cost reconstruction for diagnostic review.
5. **Review & QC**: calculate QC metrics and click **Confirm QC Risks**.
6. **Parameter Optimization**: optional. Run optimization and click **Apply Best Self-Consistency Value**.
7. **Advanced Reconstruction**: run the retained formal reconstruction.
8. **Export**: save arrays, metadata, and native output when available.

![Ptychography quick reconstruction](ptychography%20quick%20reconstruction.png)

Applying the best self-consistency value does not revoke an already accepted
QC decision. Values that affect setup or geometry are used to rebuild
preprocessing inside the Advanced background task. Batch size, Fix probe, and
Probe modes are transferred directly to Advanced Reconstruction. The optimized
value is a self-consistency candidate, not proof that it is the physically
correct parameter.

![Ptychography advanced reconstruction](ptychography%20advanced%20reconstruction.png)

## DPC / CoM

Review the measured and fitted CoM fields, accept preprocessing, then run the
integrated reconstruction. Changing preprocessing parameters makes the
reconstruction stale.

![CoM preprocessing review](Com%20Prerocessing%20review.png)

![Integrated reconstruction](integrated%20reconstruction.png)

## Strain Mapping

Complete virtual imaging and Bragg detection first. Apply the required
calibration, choose and accept basis vectors and a reference region, then run
the final strain map.

![Strain calibration](strain%20calibration.png)

![Strain analysis](strain%20analysis.png)

## Export

Use the final Export route to export registered results, write numerical arrays
to CSV, save the project state, or generate a report.

![Export controls](data%20export.png)

![Exported CSV results](data%20csv.png)

## Performance And Safety

- Full 4D reductions run as cancellable background tasks with progress reporting.
- The reduction memory budget controls chunk size; it is not a promise that all
  third-party py4DSTEM operations use the same limit.
- Review warnings and acceptance gates before running expensive downstream work.
- Changing accepted upstream parameters marks dependent results stale.
