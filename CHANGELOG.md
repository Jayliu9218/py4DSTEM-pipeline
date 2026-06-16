# Changelog

All notable changes to py4DSTEM Pipeline are tracked here.

This project follows semantic versioning where practical. While the application
is still below 1.0, minor versions may include workflow or project-file changes
that need careful validation on representative 4D-STEM datasets.

## [0.1.0] - 2026-06-16

### Status

- Preview source-code release for internal validation and early group review.
- Marked as a GitHub prerelease because amorphous-analysis routes and some
  advanced packaging paths are still under development.

### Added

- Guided Windows desktop workflow for py4DSTEM/HDF5 data browsing, DataCube role
  assignment, stage-based processing, review gates, and result export.
- Crystalline workflows covering Bragg peak detection, calibration, orientation
  mapping, strain mapping, and a unified crystalline-results review page.
- Phase-retrieval workflows for DPC/CoM, Parallax, Ptychography, and method
  comparison.
- Project-state persistence, dock-layout restoration, scientific report
  generation, CPU/GPU execution selection, and cancellable background workers.
- PyInstaller packaging scripts, portable-zip packaging, Inno Setup installer
  configuration, and packaging-environment validation remain available for local
  testing, but are not attached to the `v0.1.0` GitHub release.

### Known Limitations

- The GitHub release intentionally provides only source-code assets; users should
  run from the development environment instead of installing a packaged exe.
- Radial Profile, RDF, FEM, Amorphous Strain, and Structural Phase routes remain
  visible development areas rather than production-ready analysis workflows.
- py4DSTEM is installed from the upstream `dev` branch for the packaging
  environment, so dependency behavior can shift between rebuilds.
