import argparse
import importlib.metadata as metadata
import json
import site
import sys
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--expected-ref", default="dev")
args = parser.parse_args()

distribution = metadata.distribution("py4DSTEM")
direct_url_path = Path(distribution._path) / "direct_url.json"

if not direct_url_path.exists():
    raise SystemExit("py4DSTEM is not installed from a source that records direct_url.json")

direct_url = json.loads(direct_url_path.read_text(encoding="utf-8"))
vcs_info = direct_url.get("vcs_info", {})
requested_revision = vcs_info.get("requested_revision")
commit_id = vcs_info.get("commit_id")
source_url = direct_url.get("url")

if source_url != "https://github.com/py4dstem/py4DSTEM.git":
    raise SystemExit(f"Unexpected py4DSTEM source: {source_url}")
if requested_revision != args.expected_ref:
    raise SystemExit(f"Expected py4DSTEM ref {args.expected_ref!r}, got {requested_revision!r}")

print(f"Python: {sys.version.split()[0]}")
print(f"User-site enabled: {site.ENABLE_USER_SITE}")
print(f"py4DSTEM version: {distribution.version}")
print(f"py4DSTEM source: {source_url}")
print(f"py4DSTEM requested ref: {requested_revision}")
print(f"py4DSTEM commit: {commit_id}")
