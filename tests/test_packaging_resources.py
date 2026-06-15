from __future__ import annotations

import unittest
from pathlib import Path


class PackagingResourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]

    def test_runtime_visual_resources_exist(self) -> None:
        for relative_path in (
            "app/theme.qss",
            "app/theme_light.qss",
            "images/py4DSTEM_logo.png",
        ):
            with self.subTest(relative_path=relative_path):
                self.assertTrue((self.root / relative_path).is_file())

    def test_pyinstaller_collects_runtime_visual_resources(self) -> None:
        spec = (self.root / "packaging" / "py4dstem_pipeline.spec").read_text(encoding="utf-8")
        for resource in (
            '("..\\\\app\\\\theme.qss", "app")',
            '("..\\\\app\\\\theme_light.qss", "app")',
            '("..\\\\images\\\\py4DSTEM_logo.png", "images")',
        ):
            with self.subTest(resource=resource):
                self.assertIn(resource, spec)

    def test_nuitka_collects_runtime_visual_resources(self) -> None:
        script = (self.root / "packaging" / "build_nuitka.ps1").read_text(encoding="utf-8")
        for resource in (
            "--include-data-file=app/theme.qss=app/theme.qss",
            "--include-data-file=app/theme_light.qss=app/theme_light.qss",
            "--include-data-file=images/py4DSTEM_logo.png=images/py4DSTEM_logo.png",
        ):
            with self.subTest(resource=resource):
                self.assertIn(resource, script)


if __name__ == "__main__":
    unittest.main()
