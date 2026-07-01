#!/usr/bin/env python
"""Normalize Jupyter notebook kernelspec metadata.

By default this script only previews changes. Pass --apply to write updates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_NOTEBOOK_DIR = "notebooks"
DEFAULT_KERNEL_NAME = "4dstem"
DEFAULT_DISPLAY_NAME = "4dstem"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize kernelspec metadata for notebooks."
    )
    parser.add_argument(
        "--notebook-dir",
        default=DEFAULT_NOTEBOOK_DIR,
        help=f"Directory containing notebooks (default: {DEFAULT_NOTEBOOK_DIR}).",
    )
    parser.add_argument(
        "--kernel",
        default=DEFAULT_KERNEL_NAME,
        help=f"Kernel name to write into metadata.kernelspec.name (default: {DEFAULT_KERNEL_NAME}).",
    )
    parser.add_argument(
        "--display-name",
        default=DEFAULT_DISPLAY_NAME,
        help=f"Kernel display name to write (default: {DEFAULT_DISPLAY_NAME}).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan notebooks recursively under --notebook-dir.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write changes. Without this flag the script only previews.",
    )
    return parser.parse_args()


def load_notebook(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def dump_notebook(path: Path, notebook: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )


def current_kernelspec(notebook: dict[str, Any]) -> dict[str, Any]:
    metadata = notebook.get("metadata")
    if not isinstance(metadata, dict):
        return {}
    kernelspec = metadata.get("kernelspec")
    if not isinstance(kernelspec, dict):
        return {}
    return kernelspec


def normalize_kernel(
    notebook: dict[str, Any],
    kernel_name: str,
    display_name: str,
) -> dict[str, str]:
    metadata = notebook.setdefault("metadata", {})
    metadata["kernelspec"] = {
        "display_name": display_name,
        "language": "python",
        "name": kernel_name,
    }
    language_info = metadata.setdefault("language_info", {})
    if isinstance(language_info, dict):
        language_info["name"] = "python"
    return metadata["kernelspec"]


def format_kernel(kernelspec: dict[str, Any]) -> str:
    name = kernelspec.get("name", "<missing>")
    display = kernelspec.get("display_name", "<missing>")
    language = kernelspec.get("language", "<missing>")
    return f"name={name}, display_name={display}, language={language}"


def main() -> int:
    args = parse_args()
    notebook_dir = Path(args.notebook_dir)
    if not notebook_dir.exists():
        print(f"Notebook directory not found: {notebook_dir}")
        return 1

    pattern = "**/*.ipynb" if args.recursive else "*.ipynb"
    notebook_paths = sorted(notebook_dir.glob(pattern))
    if not notebook_paths:
        print(f"No notebooks found in: {notebook_dir}")
        return 1

    target = {
        "display_name": args.display_name,
        "language": "python",
        "name": args.kernel,
    }

    changed = 0
    unchanged = 0
    failures = 0

    print("Mode:", "apply" if args.apply else "dry-run")
    print("Target:", format_kernel(target))
    print()

    for path in notebook_paths:
        try:
            notebook = load_notebook(path)
        except (OSError, json.JSONDecodeError) as exc:
            failures += 1
            print(f"[error] {path}: {exc}")
            continue

        before = current_kernelspec(notebook)
        needs_change = before != target

        if needs_change:
            changed += 1
            print(f"[change] {path}")
            print(f"         before: {format_kernel(before)}")
            print(f"         after:  {format_kernel(target)}")
            if args.apply:
                normalize_kernel(notebook, args.kernel, args.display_name)
                dump_notebook(path, notebook)
        else:
            unchanged += 1
            print(f"[ok]     {path}")

    print()
    print("Summary:")
    print(f"  notebooks checked: {len(notebook_paths)}")
    print(f"  would change:      {changed}")
    print(f"  already target:    {unchanged}")
    print(f"  errors:            {failures}")
    if not args.apply and changed:
        print()
        print("Run again with --apply to write these changes.")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
