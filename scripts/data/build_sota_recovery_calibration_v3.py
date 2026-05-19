#!/usr/bin/env python3
"""Build a SOTA-oriented recovery calibration bank.

The purpose of this builder is to separate *training signal* from *audit
pressure*:

* train128 keeps Tool/Memory anchors plus verified-recoverable Code rows, so
  dynamic OPD has dense positives and GRPO/retention still see all tasks.
* monitor64/guard64 keep harder probes, especially Code P0 rows, so a model
  cannot look good only by overfitting recoverable training prompts.

The script only combines existing leakage-controlled manifests.  It does not
run models and does not overwrite earlier banks.
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


TASK_ORDER = ("tool", "memory", "code")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    created_at = datetime.now(timezone.utc).isoformat()
    source_rows = {
        "v2_recoverable": read_jsonl(args.v2_recoverable_train),
        "v2_monitor": read_jsonl(args.v2_monitor),
        "v2_guard": read_jsonl(args.v2_guard),
        "code_p0_recoverable": read_jsonl(args.code_p0_recoverable_train),
        "code_p0_monitor": read_jsonl(args.code_p0_monitor),
        "code_p0_guard": read_jsonl(args.code_p0_guard),
    }

    train = _build_train(source_rows, args=args, created_at=created_at)
    monitor = _build_monitor_or_guard(source_rows, split="monitor", args=args, created_at=created_at)
    guard = _build_monitor_or_guard(source_rows, split="guard", args=args, created_at=created_at)

    outputs = {
        "train128": output_dir / "train128.prompts.jsonl",
        "monitor64": output_dir / "monitor64.prompts.jsonl",
        "guard64": output_dir / "guard64.prompts.jsonl",
    }
    write_jsonl(outputs["train128"], train)
    write_jsonl(outputs["monitor64"], monitor)
    write_jsonl(outputs["guard64"], guard)

    summary = _summary(
        args=args,
        created_at=created_at,
        source_rows=source_rows,
        outputs=outputs,
        splits={"train128": train, "monitor64": monitor, "guard64": guard},
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    readme_path = output_dir / "README.md"
    readme_path.write_text(_render_readme(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def _build_train(source_rows: dict[str, list[dict[str, Any]]], *, args: argparse.Namespace, created_at: str) -> list[dict[str, Any]]:
    v2 = source_rows["v2_recoverable"]
    tool = _take_task(v2, "tool", int(args.train_tool), seed=args.seed, label="train_tool")
    memory = _take_task(v2, "memory", int(args.train_memory), seed=args.seed, label="train_memory")

    code_p0 = _task_rows(source_rows["code_p0_recoverable"], "code")
    code_v2 = _task_rows(v2, "code")
    code = _take_code_train(
        code_p0=code_p0,
        code_v2=code_v2,
        total=int(args.train_code),
        p0_min=int(args.train_code_p0_min),
        seed=int(args.seed),
    )
    groups = {
        "tool": [_annotate(row, split="train128", role="train_recoverable_tool", source_bucket="sota_v2_recoverable", created_at=created_at, tag=args.tag) for row in tool],
        "memory": [_annotate(row, split="train128", role="train_recoverable_memory", source_bucket="sota_v2_recoverable", created_at=created_at, tag=args.tag) for row in memory],
        "code": [_annotate(row, split="train128", role="train_recoverable_code", source_bucket=_source_bucket(row), created_at=created_at, tag=args.tag) for row in code],
    }
    return _interleave(groups)


def _build_monitor_or_guard(
    source_rows: dict[str, list[dict[str, Any]]],
    *,
    split: str,
    args: argparse.Namespace,
    created_at: str,
) -> list[dict[str, Any]]:
    v2_key = "v2_monitor" if split == "monitor" else "v2_guard"
    code_key = "code_p0_monitor" if split == "monitor" else "code_p0_guard"
    split_name = "monitor64" if split == "monitor" else "guard64"

    tool = _take_task(source_rows[v2_key], "tool", int(args.audit_tool), seed=args.seed, label=f"{split}_tool")
    memory = _take_task(source_rows[v2_key], "memory", int(args.audit_memory), seed=args.seed, label=f"{split}_memory")
    code_p0 = _take_task(source_rows[code_key], "code", int(args.audit_code_p0), seed=args.seed, label=f"{split}_code_p0")
    code_v2_need = int(args.audit_code) - len(code_p0)
    if code_v2_need < 0:
        raise SystemExit("--audit-code-p0 cannot exceed --audit-code")
    code_v2 = _take_task(source_rows[v2_key], "code", code_v2_need, seed=args.seed, label=f"{split}_code_v2")
    code = _dedupe_rows(code_p0 + code_v2)

    if len(code) != int(args.audit_code):
        raise SystemExit(f"Could not build {split_name} code split: need={args.audit_code}, got={len(code)}")

    groups = {
        "tool": [_annotate(row, split=split_name, role=f"{split}_tool_audit", source_bucket="sota_v2", created_at=created_at, tag=args.tag) for row in tool],
        "memory": [_annotate(row, split=split_name, role=f"{split}_memory_audit", source_bucket="sota_v2", created_at=created_at, tag=args.tag) for row in memory],
        "code": [_annotate(row, split=split_name, role=f"{split}_code_hard_audit", source_bucket=_source_bucket(row), created_at=created_at, tag=args.tag) for row in code],
    }
    return _interleave(groups)


def _take_code_train(
    *,
    code_p0: list[dict[str, Any]],
    code_v2: list[dict[str, Any]],
    total: int,
    p0_min: int,
    seed: int,
) -> list[dict[str, Any]]:
    p0_sorted = _sort_code_for_training(code_p0, seed=seed, source="code_p0")
    v2_sorted = _sort_code_for_training(code_v2, seed=seed, source="sota_v2")
    selected = _dedupe_rows(p0_sorted[: min(len(p0_sorted), p0_min)])
    if len(selected) < p0_min:
        raise SystemExit(f"Not enough Code P0 recoverable rows: need={p0_min}, got={len(selected)}")
    selected = _dedupe_rows(selected + v2_sorted)
    if len(selected) < total:
        selected = _dedupe_rows(selected + p0_sorted[p0_min:])
    if len(selected) < total:
        raise SystemExit(f"Not enough recoverable code rows: need={total}, got={len(selected)}")
    return selected[:total]


def _sort_code_for_training(rows: list[dict[str, Any]], *, seed: int, source: str) -> list[dict[str, Any]]:
    def sort_key(row: dict[str, Any]) -> tuple[int, int, str]:
        role = _row_role(row)
        role_rank = {
            "frontier": 0,
            "generation": 1,
            "source_code_anchor_from_paper96": 1,
            "on_policy_eval_style_code_probe": 2,
            "partial_edge": 2,
            "stable": 3,
        }.get(role, 4)
        positive_count = 0
        payload = row.get("recoverable_code_calibration")
        if isinstance(payload, dict):
            positive_count = int(payload.get("positive_expert_count") or 0)
        return (
            role_rank,
            -positive_count,
            stable_hash({"seed": seed, "source": source, "prompt_id": row.get("prompt_id")}),
        )

    return sorted(rows, key=sort_key)


def _take_task(rows: list[dict[str, Any]], task: str, count: int, *, seed: int, label: str) -> list[dict[str, Any]]:
    task_rows = _task_rows(rows, task)
    task_rows.sort(key=lambda row: stable_hash({"seed": seed, "label": label, "prompt_id": row.get("prompt_id")}))
    if len(task_rows) < count:
        raise SystemExit(f"Not enough rows for {label}: task={task} need={count}, got={len(task_rows)}")
    return task_rows[:count]


def _task_rows(rows: list[dict[str, Any]], task: str) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        if str(row.get("task")) != task:
            continue
        validate_seed_record(row)
        selected.append(row)
    return selected


def _annotate(
    row: dict[str, Any],
    *,
    split: str,
    role: str,
    source_bucket: str,
    created_at: str,
    tag: str,
) -> dict[str, Any]:
    copied = copy.deepcopy(row)
    old_split = copied.get("split")
    tags = set(copied.get("tags") or [])
    tags.update({tag, f"{tag}:{split}", f"sota_v3_role:{role}"})
    copied["tags"] = sorted(tags)
    copied["split"] = f"{tag}_{split}"
    copied["sota_recovery_calibration"] = {
        "format": "sota_recovery_calibration_v3",
        "tag": tag,
        "split": split,
        "role": role,
        "source_bucket": source_bucket,
        "original_split": old_split,
        "created_at": created_at,
        "policy": (
            "Train uses verified-recoverable rows to keep OPD dense; "
            "monitor/guard use harder disjoint probes to audit generalization."
        ),
    }
    validate_seed_record(copied)
    return copied


def _interleave(groups: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    max_len = max((len(group) for group in groups.values()), default=0)
    for idx in range(max_len):
        for task in TASK_ORDER:
            group = groups.get(task, [])
            if idx < len(group):
                output.append(group[idx])
    return output


def _dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    seen: set[str] = set()
    for row in rows:
        key = str(row.get("prompt_hash") or row.get("prompt_id") or stable_hash(row))
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def _row_role(row: dict[str, Any]) -> str:
    for key in ("sota_recovery_calibration", "code_p0_calibration", "eval_targeted_calibration"):
        payload = row.get(key)
        if isinstance(payload, dict) and payload.get("role"):
            return str(payload["role"])
    metadata = row.get("reference", {}).get("metadata", {}) if isinstance(row.get("reference"), dict) else {}
    if isinstance(metadata, dict) and metadata.get("code_bank_role"):
        return str(metadata["code_bank_role"])
    return "default"


def _source_bucket(row: dict[str, Any]) -> str:
    if isinstance(row.get("recoverable_code_calibration"), dict):
        if isinstance(row.get("code_p0_calibration"), dict):
            return "code_p0_v3_recoverable"
        return "sota_v2_recoverable_code"
    if isinstance(row.get("code_p0_calibration"), dict):
        return "code_p0_v3_audit"
    if isinstance(row.get("eval_targeted_calibration"), dict):
        return "sota_v2_eval_targeted"
    return "source_manifest"


def _summary(
    *,
    args: argparse.Namespace,
    created_at: str,
    source_rows: dict[str, list[dict[str, Any]]],
    outputs: dict[str, Path],
    splits: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    return {
        "format": "sota_recovery_calibration_v3",
        "created_at": created_at,
        "seed": int(args.seed),
        "tag": args.tag,
        "inputs": {
            "v2_recoverable_train": str(Path(args.v2_recoverable_train).expanduser().resolve()),
            "v2_monitor": str(Path(args.v2_monitor).expanduser().resolve()),
            "v2_guard": str(Path(args.v2_guard).expanduser().resolve()),
            "code_p0_recoverable_train": str(Path(args.code_p0_recoverable_train).expanduser().resolve()),
            "code_p0_monitor": str(Path(args.code_p0_monitor).expanduser().resolve()),
            "code_p0_guard": str(Path(args.code_p0_guard).expanduser().resolve()),
        },
        "outputs": {name: str(path) for name, path in outputs.items()},
        "source_counts": {
            name: {
                "rows": len(rows),
                "tasks": dict(sorted(Counter(str(row.get("task")) for row in rows).items())),
                "roles": dict(sorted(Counter(_row_role(row) for row in rows).items())),
            }
            for name, rows in sorted(source_rows.items())
        },
        "split_counts": {
            name: {
                "rows": len(rows),
                "tasks": dict(sorted(Counter(str(row.get("task")) for row in rows).items())),
                "roles": dict(sorted(Counter(str((row.get("sota_recovery_calibration") or {}).get("role")) for row in rows).items())),
                "source_buckets": dict(sorted(Counter(str((row.get("sota_recovery_calibration") or {}).get("source_bucket")) for row in rows).items())),
            }
            for name, rows in splits.items()
        },
        "design": {
            "train": "32 Tool + 48 Memory from sota_v2 recoverable train; 48 Code from Code P0 recoverable plus sota_v2 recoverable anchors.",
            "monitor_guard": "Tool/Memory from sota_v2 monitor/guard; Code primarily from Code P0 hard audit rows plus a small sota_v2 anchor slice.",
            "training_signal": "Dense same-prompt expert positives for OPD, limited frontier quotas for GRPO, all-success rows retained for NLL preservation.",
        },
    }


def _render_readme(summary: dict[str, Any]) -> str:
    lines = [
        "# SOTA Recovery Calibration v3",
        "",
        f"created_at: `{summary['created_at']}`",
        f"seed: `{summary['seed']}`",
        "",
        "## Purpose",
        "",
        "SOTA-oriented train/monitor/guard bank. The training split is optimized for dense verified recovery signal; monitor and guard keep harder probes for generalization checks.",
        "",
        "## Outputs",
        "",
    ]
    for name, path in summary["outputs"].items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(["", "## Split Counts", "", "| split | rows | tool | memory | code |", "|---|---:|---:|---:|---:|"])
    for name, item in summary["split_counts"].items():
        tasks = item["tasks"]
        lines.append(f"| {name} | {item['rows']} | {tasks.get('tool', 0)} | {tasks.get('memory', 0)} | {tasks.get('code', 0)} |")
    lines.extend(
        [
            "",
            "## Training Interpretation",
            "",
            "- `train128` is not a random benchmark subset; it is a high-gradient calibration bank.",
            "- Code train rows are verified-recoverable by ReasonFlux/DeepSeek expert rollouts.",
            "- Hard Code rows remain in `monitor64`/`guard64`, so training reward cannot be the only selection criterion.",
            "- The bank is built from existing leakage-controlled manifests and records all input paths in `summary.json`.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = "/tmp/shared-storage/OnPolicy/data/calibration"
    parser.add_argument("--output-dir", default=f"{root}/sota_recovery_calib_v3_20260518")
    parser.add_argument("--tag", default="sota_recovery_calib_v3_20260518")
    parser.add_argument("--seed", type=int, default=20260518)
    parser.add_argument("--v2-recoverable-train", default=f"{root}/sota_calib_v2_recoverable_code_20260518/train_recoverable101.prompts.jsonl")
    parser.add_argument("--v2-monitor", default=f"{root}/sota_calib_v2_20260518/monitor64.prompts.jsonl")
    parser.add_argument("--v2-guard", default=f"{root}/sota_calib_v2_20260518/guard64.prompts.jsonl")
    parser.add_argument("--code-p0-recoverable-train", default=f"{root}/code_p0_v3_20260518/train_recoverable_code.prompts.jsonl")
    parser.add_argument("--code-p0-monitor", default=f"{root}/code_p0_v3_20260518/monitor_code32.prompts.jsonl")
    parser.add_argument("--code-p0-guard", default=f"{root}/code_p0_v3_20260518/guard_code32.prompts.jsonl")
    parser.add_argument("--train-tool", type=int, default=32)
    parser.add_argument("--train-memory", type=int, default=48)
    parser.add_argument("--train-code", type=int, default=48)
    parser.add_argument("--train-code-p0-min", type=int, default=36)
    parser.add_argument("--audit-tool", type=int, default=16)
    parser.add_argument("--audit-memory", type=int, default=24)
    parser.add_argument("--audit-code", type=int, default=24)
    parser.add_argument("--audit-code-p0", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    main()
