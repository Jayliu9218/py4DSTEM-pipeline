# Changelog

All notable changes to py4DSTEM Pipeline are tracked here.

This project follows semantic versioning where practical. While the application
is still below 1.0, minor versions may include workflow or project-file changes
that need careful validation on representative 4D-STEM datasets.

## [0.1.0] - 2026-06-16

### Summary

- Source-code preview release for early validation.
- Published as a source-only GitHub prerelease.

### Added

- HDF5/EMD browsing, Target DataCube assignment, guided stages, review gates,
  and result export.
- Crystalline workflows for Bragg detection, calibration, orientation mapping,
  strain mapping, and result review.
- Phase-retrieval workflows for DPC/CoM, Parallax, Ptychography, and method
  comparison.
- Project-state persistence, dock-layout restoration, background progress, and
  CPU/GPU execution selection.

### Notes

- Run from the development environment using `scripts/setup_dev_env.ps1`.
- Amorphous-analysis and structural-phase routes remain under development.
- Validate scientific outputs on representative `.h5` or `.emd` datasets.
