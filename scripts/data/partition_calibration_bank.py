#!/usr/bin/env python3
"""Partition seed manifests into disjoint calibration splits.

This script is intentionally narrow: it reads one or more OP-VEC seed
manifests, deduplicates rows, then creates deterministic train/monitor/guard
splits with per-task counts.  When a row-level role field is available, each
task is split approximately proportionally by role so source anchors and
targeted probes do not collapse into one split.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from opvec.data.io import read_jsonl, write_jsonl
from opvec.data.schema import stable_hash, validate_seed_record


TASKS = ("tool", "memory", "code")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    split_specs = [_parse_split_spec(item) for item in args.split_spec]
    rows, source_summary = _load_rows(args.input, args.dedupe_key)
    split_rows, split_summary = _partition_rows(
        rows,
        split_specs=split_specs,
        seed=args.seed,
        stratify_field=args.stratify_field,
        tag=args.tag,
    )

    outputs = {}
    for split_name, selected in split_rows.items():
        out_path = output_dir / f"{split_name}.prompts.jsonl"
        write_jsonl(out_path, selected)
        outputs[split_name] = str(out_path)

    summary = {
        "format": "opvec_calibration_partition_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": int(args.seed),
        "tag": args.tag,
        "dedupe_key": args.dedupe_key,
        "stratify_field": args.stratify_field,
        "inputs": [str(Path(path).expanduser().resolve()) for path in args.input],
        "outputs": outputs,
        "source_summary": source_summary,
        "split_summary": split_summary,
        "split_specs": [
            {"name": name, "task_counts": counts}
            for name, counts in split_specs
        ],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    readme_path = output_dir / "README.md"
    readme_path.write_text(_render_readme(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def _load_rows(paths: list[str], dedupe_key: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    skipped = Counter()
    per_input = {}
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        current = read_jsonl(path)
        selected = []
        for row in current:
            task = str(row.get("task") or "")
            if task not in TASKS:
                skipped[f"{path}:unsupported_task"] += 1
                continue
            key = str(row.get(dedupe_key) or row.get("prompt_id") or "")
            if not key:
                skipped[f"{path}:missing_dedupe_key"] += 1
                continue
            if key in seen:
                skipped[f"{path}:duplicate"] += 1
                continue
            validate_seed_record(row)
            seen.add(key)
            rows.append(row)
            selected.append(row)
        per_input[str(path)] = {
            "input_rows": len(current),
            "selected_rows": len(selected),
            "task_counts": dict(sorted(Counter(str(row.get("task")) for row in selected).items())),
        }
    summary = {
        "rows": len(rows),
        "task_counts": dict(sorted(Counter(str(row.get("task")) for row in rows).items())),
        "per_input": per_input,
        "skipped": dict(sorted(skipped.items())),
    }
    return rows, summary


def _partition_rows(
    rows: list[dict[str, Any]],
    *,
    split_specs: list[tuple[str, dict[str, int]]],
    seed: int,
    stratify_field: str,
    tag: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    by_task_role: dict[str, dict[str, list[dict[str, Any]]]] = {task: defaultdict(list) for task in TASKS}
    for row in rows:
        task = str(row.get("task") or "")
        role = _role_for(row, stratify_field)
        by_task_role[task][role].append(row)

    for task, role_groups in by_task_role.items():
        for role, group in role_groups.items():
            group.sort(key=lambda row: stable_hash({"seed": seed, "task": task, "role": role, "prompt_id": row.get("prompt_id")}))

    output: dict[str, list[dict[str, Any]]] = {name: [] for name, _ in split_specs}
    summary: dict[str, Any] = {}
    for split_name, task_counts in split_specs:
        split_selected: dict[str, list[dict[str, Any]]] = {task: [] for task in TASKS}
        role_counts: dict[str, dict[str, int]] = {}
        for task, target_count in task_counts.items():
            if target_count < 0:
                raise SystemExit(f"Negative split count for {split_name}:{task}={target_count}")
            role_groups = by_task_role.get(task, {})
            selected_by_role = _take_stratified(role_groups, target_count)
            task_rows = []
            role_counts[task] = {}
            for role, selected in selected_by_role.items():
                role_counts[task][role] = len(selected)
                task_rows.extend(selected)
            split_selected[task] = task_rows
        interleaved = _interleave_tasks(split_selected)
        output[split_name] = [
            _annotate_row(row, split_name=split_name, tag=tag)
            for row in interleaved
        ]
        summary[split_name] = {
            "rows": len(output[split_name]),
            "task_counts": dict(sorted(Counter(row["task"] for row in output[split_name]).items())),
            "role_counts": role_counts,
        }

    remaining = {
        task: {role: len(group) for role, group in sorted(role_groups.items())}
        for task, role_groups in by_task_role.items()
    }
    summary["remaining_pool"] = remaining
    return output, summary


def _take_stratified(role_groups: dict[str, list[dict[str, Any]]], target_count: int) -> dict[str, list[dict[str, Any]]]:
    if target_count == 0:
        return {}
    total_remaining = sum(len(group) for group in role_groups.values())
    if total_remaining < target_count:
        raise SystemExit(f"Not enough rows for split: need={target_count}, have={total_remaining}, roles={dict((k, len(v)) for k, v in role_groups.items())}")
    roles = sorted(role_groups)
    exact = {role: target_count * len(role_groups[role]) / total_remaining for role in roles}
    allocation = {role: min(len(role_groups[role]), int(exact[role])) for role in roles}
    missing = target_count - sum(allocation.values())
    while missing > 0:
        candidates = [
            (
                exact[role] - allocation[role],
                len(role_groups[role]) - allocation[role],
                role,
            )
            for role in roles
            if len(role_groups[role]) > allocation[role]
        ]
        if not candidates:
            raise SystemExit(f"Internal allocation failure for target={target_count}")
        candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
        role = candidates[0][2]
        allocation[role] += 1
        missing -= 1

    selected: dict[str, list[dict[str, Any]]] = {}
    for role in roles:
        take = allocation[role]
        if take <= 0:
            continue
        selected[role] = role_groups[role][:take]
        del role_groups[role][:take]
    return selected


def _interleave_tasks(groups: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    max_len = max((len(rows) for rows in groups.values()), default=0)
    for idx in range(max_len):
        for task in TASKS:
            task_rows = groups.get(task, [])
            if idx < len(task_rows):
                output.append(task_rows[idx])
    return output


def _annotate_row(row: dict[str, Any], *, split_name: str, tag: str) -> dict[str, Any]:
    copied = copy.deepcopy(row)
    old_split = copied.get("split")
    tags = set(copied.get("tags") or [])
    tags.add(tag)
    tags.add(f"{tag}:{split_name}")
    copied["tags"] = sorted(tags)
    copied["split"] = f"{tag}_{split_name}"
    copied["calibration_partition"] = {
        "tag": tag,
        "split_name": split_name,
        "original_split": old_split,
        "original_prompt_id": copied.get("prompt_id"),
    }
    validate_seed_record(copied)
    return copied


def _role_for(row: dict[str, Any], field: str) -> str:
    value: Any = row
    for part in field.split("."):
        if not isinstance(value, dict) or part not in value:
            return "default"
        value = value[part]
    text = str(value or "").strip()
    return text or "default"


def _parse_split_spec(spec: str) -> tuple[str, dict[str, int]]:
    if ":" not in spec:
        raise SystemExit(f"Invalid --split-spec {spec!r}; expected name:tool=...,memory=...,code=...")
    name, raw_counts = spec.split(":", 1)
    name = name.strip()
    if not name:
        raise SystemExit(f"Invalid --split-spec {spec!r}: empty split name")
    counts = {task: 0 for task in TASKS}
    for item in raw_counts.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(f"Invalid count item {item!r} in --split-spec {spec!r}")
        task, raw_value = item.split("=", 1)
        task = task.strip()
        if task not in TASKS:
            raise SystemExit(f"Unsupported task {task!r} in --split-spec {spec!r}")
        counts[task] = int(raw_value)
    return name, counts


def _render_readme(summary: dict[str, Any]) -> str:
    lines = [
        f"# {summary['tag']} Calibration Partition",
        "",
        f"created_at: `{summary['created_at']}`",
        f"seed: `{summary['seed']}`",
        "",
        "## Outputs",
        "",
    ]
    for split_name, path in summary["outputs"].items():
        rows = summary["split_summary"][split_name]["rows"]
        counts = summary["split_summary"][split_name]["task_counts"]
        lines.append(f"- `{split_name}`: `{path}` rows={rows} task_counts={counts}")
    lines.extend(
        [
            "",
            "## Policy",
            "",
            "- Splits are disjoint by dedupe key.",
            "- Rows are stratified approximately by `eval_targeted_calibration.role` when present.",
            "- Each output row records the original split in `calibration_partition.original_split`.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, help="Input seed manifest JSONL. Repeatable.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-spec", action="append", required=True, help="Example: train128:tool=32,memory=48,code=48")
    parser.add_argument("--seed", type=int, default=20260518)
    parser.add_argument("--dedupe-key", default="prompt_hash")
    parser.add_argument("--stratify-field", default="eval_targeted_calibration.role")
    parser.add_argument("--tag", default="sota_calib_v2_20260518")
    return parser.parse_args()


if __name__ == "__main__":
    main()
