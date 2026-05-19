#!/usr/bin/env python3
"""Merge rollout JSONL shards by prompt id, optionally preserving manifest order."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from opvec.data.io import read_jsonl, write_jsonl


def main() -> None:
    args = parse_args()
    key = str(args.key)
    manifest_order = _manifest_order(Path(args.order_manifest).expanduser().resolve(), key=key) if args.order_manifest else []
    rows_by_key: dict[str, dict[str, Any]] = {}
    sources_by_key: dict[str, str] = {}
    duplicates = Counter()
    input_summary = {}
    for raw_path in args.input:
        path = Path(raw_path).expanduser().resolve()
        rows = read_jsonl(path)
        input_summary[str(path)] = {"rows": len(rows)}
        for row in rows:
            row_key = str(row.get(key) or "")
            if not row_key:
                raise SystemExit(f"Row missing key={key}: {path}")
            if row_key in rows_by_key:
                duplicates[row_key] += 1
                if args.duplicate_policy == "error":
                    raise SystemExit(f"Duplicate {key}={row_key} in {path}; first source={sources_by_key[row_key]}")
                if args.duplicate_policy == "last":
                    rows_by_key[row_key] = row
                    sources_by_key[row_key] = str(path)
                continue
            rows_by_key[row_key] = row
            sources_by_key[row_key] = str(path)

    ordered_keys = [item for item in manifest_order if item in rows_by_key]
    remaining_keys = sorted(key for key in rows_by_key if key not in set(ordered_keys))
    output_rows = [rows_by_key[item] for item in ordered_keys + remaining_keys]
    if args.expected_count is not None and len(output_rows) != int(args.expected_count):
        raise SystemExit(f"Merged row count mismatch: expected={args.expected_count}, got={len(output_rows)}")

    output = Path(args.output).expanduser().resolve()
    write_jsonl(output, output_rows)
    missing_from_manifest = [item for item in manifest_order if item not in rows_by_key]
    summary = {
        "format": "rollout_shard_merge_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "key": key,
        "output": str(output),
        "rows": len(output_rows),
        "inputs": input_summary,
        "order_manifest": str(Path(args.order_manifest).expanduser().resolve()) if args.order_manifest else None,
        "duplicate_policy": args.duplicate_policy,
        "duplicates": {"keys": len(duplicates), "extra_rows": sum(duplicates.values())},
        "missing_from_manifest": len(missing_from_manifest),
    }
    output.with_suffix(".merge_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


def _manifest_order(path: Path, *, key: str) -> list[str]:
    order = []
    for row in read_jsonl(path):
        row_key = str(row.get(key) or "")
        if row_key:
            order.append(row_key)
    return order


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, help="Rollout shard JSONL. Repeatable.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--order-manifest", default=None, help="Seed manifest used to preserve prompt order.")
    parser.add_argument("--key", default="prompt_id")
    parser.add_argument("--expected-count", type=int, default=None)
    parser.add_argument("--duplicate-policy", choices=["first", "last", "error"], default="first")
    return parser.parse_args()


if __name__ == "__main__":
    main()
