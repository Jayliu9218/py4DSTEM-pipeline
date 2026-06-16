# Release Checklist

Use this checklist before publishing `v0.1.0` or any later GitHub release.

## Version

- Confirm `app/version.py` contains the intended version.
- Confirm `CHANGELOG.md` has a section for the tag being released.
- Create an annotated tag, for example `v0.1.0`.

## Validation

```powershell
.\scripts\setup_dev_env.ps1 -Stage Test
```

Manual validation:

- Launch the application from source:

```powershell
& .\.conda\py4dstem-pipeline-dev\python.exe .\main.py
```

- Open representative `.h5` and `.emd` files.
- Assign a Target DataCube and run at least one crystalline route and one
  phase-retrieval route through their review gates.
- Export at least one NPZ/JSON/PNG/TIFF result and one CSV batch where
  applicable.

## GitHub Release

- Use the generated GitHub release draft for `v0.1.0`.
- Keep `v0.1.0` marked as a prerelease.
- Do not attach generated executable, installer, or portable-zip artifacts.
- Publish with only the GitHub-generated source code assets:
  - `Source code (zip)`
  - `Source code (tar.gz)`
- Include short validation notes and the py4DSTEM source/ref expectation from
  `requirements.txt`.
