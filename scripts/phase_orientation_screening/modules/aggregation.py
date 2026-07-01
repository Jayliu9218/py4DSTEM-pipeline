"""Branch aggregation helpers."""

from __future__ import annotations

import json
from pathlib import Path


def load_branch_metadata(branch_dir):
    metas = []
    branch_dir = Path(branch_dir)
    for path in sorted(branch_dir.glob("metadata_branch_*.json")):
        with open(path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        score_path = Path(meta.get("score_path", ""))
        if not score_path.is_absolute():
            score_path = branch_dir / score_path
        if not score_path.exists():
            fallback = path.with_name(path.name.replace("metadata_branch_", "score_branch_").replace(".json", ".npy"))
            score_path = fallback
        meta["score_path"] = str(score_path)
        metas.append(meta)
    return metas
