# General py4DSTEM Crystal Analysis Workflow

This notebook/script workflow is aligned with the app route named
`Crystalline / Bragg-based -> Crystal Analysis`. It is designed for large MIB
datasets such as `512 x 512 x 256 x 256` where the raw cube is about 32 GB, so
the default path is MIB `MEMMAP` loading plus ROI-first parameter tuning.

## Scientific Workflow

```text
Import
  -> Calibration
  -> Virtual Imaging
  -> Bragg Disk Detection
  -> Bragg Vector Map
  -> Crystal Analysis
       -> CIF Manager
       -> Structure Factors
       -> Simulated Diffraction / Orientation Library
       -> Phase Matching
       -> Orientation Matching
       -> Grain Analysis
       -> Strain Analysis (optional)
  -> Results / Export
```

Phase identification and orientation mapping use the same calibrated
`BraggVectors`. For multi-phase materials, each enabled CIF builds its own
`Crystal` library, matches against the same experimental Bragg peaks, and the
best score at each scan position defines the phase map. Orientation, grain, and
strain interpretation then use the winning phase mask.

## MIB Defaults

Use safe defaults for raw MIB input:

```python
dc = py4DSTEM.import_file(mib_file, mem="MEMMAP", scan=(512, 512))
mean_dp = dc.get_dp_mean().data
```

Start with a centered `128 x 128` ROI to tune Bragg detection, phase matching,
and orientation-library sampling. Switch to the full dataset only after the BVM,
phase confidence map, and orientation composite look reasonable.

## Windows PowerShell Example

```powershell
python .\general_py4DSTEM_crystal_workflow.py `
  --input "D:\data\sample.mib" `
  --input-type auto `
  --scan 512 512 `
  --mem MEMMAP `
  --roi 192 320 192 320 `
  --output-dir ".\py4dstem_general_output" `
  --phase-cifs "D:\data\Ti-fcc.cif" "D:\data\Ti-hcp.cif" `
  --phase-names "Ti-fcc" "Ti-hcp" `
  --voltage 300000 `
  --angle-step 5 `
  --in-plane-step 5
```

After confirming the parameters, rerun without `--roi` and with finer angular
sampling:

```powershell
--angle-step 2 --in-plane-step 2 --run-strain
```

## HDF5 / EMD Example

```powershell
python .\general_py4DSTEM_crystal_workflow.py `
  --input "D:\data\sample.h5" `
  --input-type h5 `
  --datacube-path "datacube" `
  --output-dir ".\py4dstem_general_output" `
  --phase-cifs "D:\data\Ti-fcc.cif" "D:\data\Ti-hcp.cif" `
  --phase-names "Ti-fcc" "Ti-hcp" `
  --voltage 300000 `
  --angle-step 5 `
  --in-plane-step 5
```

## Main Outputs

- `01_mean_dp.*`: mean diffraction pattern.
- `02_vbf.*`, `02_adf.*`: virtual bright-field and annular dark-field images.
- `03_comx.npy`, `03_comy.npy`: center-of-mass maps.
- `04_braggvectors.pkl`: detected BraggVectors.
- `04_bvm_raw.*`, `04_bvm_cal.*`: raw and calibrated Bragg vector maps.
- `10_phase_id_map.*`: winning phase ID at each scan point.
- `10_phase_confidence_gap.*`: score gap between the best and second-best phase.
- `10_phase_orientation_composite_rgb.*`: phase plus orientation composite RGB.
- `11_*`: grain outputs when the installed py4DSTEM API supports them.
- `12_*`: phase-masked strain outputs when enabled and supported.
- `workflow_metadata.json`: configuration, warnings, timings, and output paths.
