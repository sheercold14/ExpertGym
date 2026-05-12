#!/usr/bin/env python3
"""Merge seed manifests with optional per-input task filters."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from opvec.data.io import read_jsonl, write_jsonl
from opvec.data.schema import validate_seed_record


def main() -> None:
    args = parse_args()
    rows, summary = merge_manifests(args)
    if not args.dry_run:
        written = write_jsonl(args.output, rows)
        summary["written"] = written
        Path(args.output).with_suffix(".summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def merge_manifests(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    skipped = Counter()
    source_stats: dict[str, Any] = {}

    for spec in args.input:
        path, task_filter = _parse_input_spec(spec)
        rows = read_jsonl(path)
        selected = []
        for row in rows:
            task = str(row.get("task") or "")
            if task_filter is not None and task not in task_filter:
                skipped[f"{path}:task_filtered"] += 1
                continue
            dedupe_key = str(row.get(args.dedupe_key) or row.get("prompt_id") or "")
            if not dedupe_key:
                skipped[f"{path}:missing_dedupe_key"] += 1
                continue
            if dedupe_key in seen:
                skipped[f"{path}:duplicate"] += 1
                continue
            validate_seed_record(row)
            seen.add(dedupe_key)
            copied = dict(row)
            tags = set(copied.get("tags") or [])
            tags.add("merged_seed_manifest")
            copied["tags"] = sorted(tags)
            copied["merged_seed_source"] = str(path)
            selected.append(copied)
        output_rows.extend(selected)
        source_stats[str(path)] = {
            "task_filter": sorted(task_filter) if task_filter is not None else None,
            "input_rows": len(rows),
            "selected_rows": len(selected),
            "selected_task_counts": dict(sorted(Counter(row["task"] for row in selected).items())),
        }

    summary = {
        "format": "opvec_merged_seed_manifest_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output": str(args.output),
        "inputs": list(args.input),
        "rows": len(output_rows),
        "task_counts": dict(sorted(Counter(row["task"] for row in output_rows).items())),
        "source_stats": source_stats,
        "skipped": dict(sorted(skipped.items())),
        "dedupe_key": args.dedupe_key,
    }
    return output_rows, summary


def _parse_input_spec(spec: str) -> tuple[str, set[str] | None]:
    if "::" not in spec:
        return spec, None
    path, raw_tasks = spec.split("::", 1)
    tasks = {item.strip() for item in raw_tasks.split(",") if item.strip()}
    return path, tasks or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Input manifest, optionally suffixed as path::tool,memory,code. Repeatable.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--dedupe-key", default="prompt_hash")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
