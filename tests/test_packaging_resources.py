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

    def test_release_metadata_is_aligned(self) -> None:
        from app.version import IS_PRERELEASE, RELEASE_TAG, __version__

        inno = (self.root / "packaging" / "inno_setup.iss").read_text(encoding="utf-8")
        changelog = (self.root / "CHANGELOG.md").read_text(encoding="utf-8")
        checklist = (self.root / "RELEASE_CHECKLIST.md").read_text(encoding="utf-8")
        release_workflow = (
            self.root / ".github" / "workflows" / "release.yml"
        ).read_text(encoding="utf-8")

        self.assertEqual(__version__, "0.1.0")
        self.assertEqual(RELEASE_TAG, "v0.1.0")
        self.assertTrue(IS_PRERELEASE)
        self.assertIn('#define MyAppVersion "0.1.0"', inno)
        self.assertIn("## [0.1.0]", changelog)
        self.assertIn("Keep `v0.1.0` marked as a prerelease.", checklist)
        self.assertIn("Do not attach generated executable", checklist)
        self.assertIn("source-only prerelease", release_workflow)
        self.assertIn("prerelease: true", release_workflow)
        self.assertNotIn("build_pyinstaller.ps1", release_workflow)
        self.assertNotIn("files:", release_workflow)


if __name__ == "__main__":
    unittest.main()
