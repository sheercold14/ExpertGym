"""Mode manifest helpers."""

from __future__ import annotations

import json
from pathlib import Path


def manifest_param_names(mode_manifest_path: str | Path, *, weight_only: bool = True) -> list[str]:
    """Return sorted mergeable parameter names from an OP-VEC mode manifest."""

    manifest = json.loads(Path(mode_manifest_path).expanduser().read_text(encoding="utf-8"))
    names = {str(entry["param_name"]) for entry in manifest.get("basis_entries", []) if "param_name" in entry}
    if weight_only:
        names = {name for name in names if name.endswith(".weight")}
    return sorted(names)
