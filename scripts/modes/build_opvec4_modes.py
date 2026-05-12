#!/usr/bin/env python3
"""Build OP-VEC-4 mode artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from opvec.config import load_config
from opvec.modes.build_modes import build_opvec4_modes


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.include_regex:
        config["modes"]["include_regex"] = args.include_regex
    if args.exclude_regex:
        config["modes"]["exclude_regex"] = args.exclude_regex
    if args.delta_dtype:
        config["modes"]["delta_dtype"] = args.delta_dtype
    manifest = build_opvec4_modes(
        config,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        max_params=args.max_params,
    )
    print(json.dumps({"mode_manifest": manifest.get("format"), "num_params": manifest["selection"]["num_params"], "dry_run": manifest["dry_run"]}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/gated_grpo.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-params", type=int, default=None)
    parser.add_argument("--delta-dtype", default=None, help="Override config modes.delta_dtype, e.g. bfloat16.")
    parser.add_argument(
        "--include-regex",
        action="append",
        default=None,
        help="Override config modes.include_regex. Repeat to allow multiple patterns.",
    )
    parser.add_argument(
        "--exclude-regex",
        action="append",
        default=None,
        help="Override config modes.exclude_regex. Repeat to allow multiple patterns.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
