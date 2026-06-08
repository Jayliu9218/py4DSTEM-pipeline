# Qt Designer workflow

This project is currently a PySide6 application whose windows and pages are
built directly in Python. Qt Designer can be introduced gradually: keep
calculation logic, services, worker threads, and signal wiring in Python, and
move only the visual widget layout into `.ui` files.

## Open Qt Designer

Activate the development environment first:

```powershell
conda activate 4dstem
```

Open Designer:

```powershell
.\scripts\open_qt_designer.ps1
```

Open a specific UI file:

```powershell
.\scripts\open_qt_designer.ps1 .\ui\calibration_page.ui
```

Check which Designer executable will be used without opening the GUI:

```powershell
.\scripts\open_qt_designer.ps1 -CheckOnly
```

If the command cannot find Designer, make sure PySide6 is installed in the
active environment:

```powershell
python -m pip install PySide6
```

## Recommended migration pattern

Use one `.ui` file per major page:

```text
ui/
  main_window.ui
  data_manager_page.ui
  virtual_detector_page.ui
  bragg_peaks_page.ui
  calibration_page.ui
  orientation_page.ui
  strain_map_page.ui
```

Start with a page that has many controls but limited custom drawing, such as
`calibration_page.ui`. Keep custom widgets such as `ImageViewer`, `LogPanel`,
and `Hdf5TreeWidget` as Python classes and promote placeholder widgets to those
classes inside Designer only after the layout is stable.

## Object names matter

Every widget that Python code needs to access should have a stable
`objectName` in Designer. For example:

```text
analysisTargetCombo
ellipseInnerSpin
ellipseOuterSpin
refreshButton
originButton
viewersTabs
statusLabel
```

The Python page can then bind those widgets by name after loading the `.ui`
file and connect signals in code.

## Keep in Python

- py4DSTEM service calls
- HDF5 file handling
- worker threads
- progress logging
- validation and fallback behavior
- signal connections
- custom image display behavior

## Put in Designer

- page layouts
- labels
- buttons
- spin boxes and combo boxes
- group boxes
- splitters and tabs
- default minimum sizes and stretch behavior

## Runtime options

There are two practical ways to use `.ui` files.

### Option A: load `.ui` files at runtime

This is best while the UI is changing frequently.

```python
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile

loader = QUiLoader()
file = QFile("ui/calibration_page.ui")
file.open(QFile.ReadOnly)
widget = loader.load(file)
file.close()
```

### Option B: compile `.ui` files to Python

This is better when packaging and release stability matter.

```powershell
pyside6-uic .\ui\calibration_page.ui -o .\app\generated\ui_calibration_page.py
```

For this project, use Option A during design iteration, then switch important
pages to Option B before packaging if the generated Python files make release
testing easier.
