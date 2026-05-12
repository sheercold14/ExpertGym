"""Path helpers for connecting this verl checkout to the OP-VEC project."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def resolve_opvec_repo_root() -> Path:
    """Return the OP-VEC repository root.

    The preferred contract is ``OPVEC_REPO_ROOT=/path/to/OnPolicyMerge_gated_grpo``.
    When it is not set, this helper assumes the local layout used in this
    project: ``<opvec_repo>/third_party/verl``.
    """

    env_root = os.environ.get("OPVEC_REPO_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[5]


def ensure_opvec_on_path() -> Path:
    root = resolve_opvec_repo_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root
