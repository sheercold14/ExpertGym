#!/usr/bin/env python3
"""Collect RCRF mechanism artifacts into long-form JSONL tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PLATFORM_DIR = REPO_ROOT / "scripts" / "visualization" / "attn" / "analysis_platform"
sys.path.insert(0, str(PLATFORM_DIR))

from rcrf_schema import DEFAULT_INPUT_PATHS, load_mechanism_data, make_example_data, write_longform  # noqa: E402


def main() -> None:
    args = parse_args()
    input_paths: list[str] = []
    input_paths.extend(args.input_dir or [])
    input_paths.extend(args.input_file or [])
    input_paths.extend(args.signed_utility_summary or [])
    input_paths.extend(args.gate_json or [])
    input_paths.extend(args.eval_path or [])

    if args.example:
        data = make_example_data(num_prompts=args.example_prompts)
    else:
        data = load_mechanism_data(
            input_paths,
            include_defaults=not args.no_default_paths,
            example_on_empty=not args.no_example_fallback,
        )

    paths = write_longform(args.output_dir, data)
    payload = {
        "output_dir": str(Path(args.output_dir).expanduser().resolve()),
        "files": paths,
        "used_example": data.used_example,
        "counts": {
            "residual_records": int(len(data.residual)),
            "interference_records": int(len(data.interference)),
            "gate_records": int(len(data.gate)),
            "eval_records": int(len(data.eval)),
        },
        "default_input_paths": list(DEFAULT_INPUT_PATHS) if not args.no_default_paths else [],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(PLATFORM_DIR / "data"),
        help="Directory to write residual/interference/gate/eval JSONL files.",
    )
    parser.add_argument("--input-dir", action="append", help="Directory containing native RCRF artifacts or long-form JSONL.")
    parser.add_argument("--input-file", action="append", help="Specific artifact file to parse.")
    parser.add_argument("--signed-utility-summary", action="append", help="Path to signed_utility_summary.json.")
    parser.add_argument("--gate-json", action="append", help="Path to RCRF gates.json or gate_values.json.")
    parser.add_argument("--eval-path", action="append", help="Eval summary/log directory or file.")
    parser.add_argument("--no-default-paths", action="store_true", help="Do not scan the current default RCRF paths.")
    parser.add_argument("--no-example-fallback", action="store_true", help="Write empty tables instead of example data when no real data is found.")
    parser.add_argument("--example", action="store_true", help="Force deterministic example data.")
    parser.add_argument("--example-prompts", type=int, default=96, help="Number of example prompts when --example is used.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
