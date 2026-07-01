"""CLI wrapper for the phase/orientation screening pipeline."""

import os
import sys

os.environ.setdefault("PYTHONUNBUFFERED", "1")
if not any(arg in ("-h", "--help") for arg in sys.argv[1:]):
    print("Starting phase/orientation screening...", flush=True)

from modules.pipeline import main


if __name__ == "__main__":
    raise SystemExit(main())
