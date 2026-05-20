#!/usr/bin/env python3
"""Build Round16 non-leak TRC calibration with Code pass/fail contrast.

The Code rows are selected only from CodeP0-v3 train rollouts. A contrast row
keeps a successful expert trajectory as `response` and a same-prompt failed
trajectory as `negative_response`. The TRC trainer ignores the negative field
unless contrastive loss is explicitly enabled.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from opvec.data.io import read_jsonl, write_jsonl


DEFAULT_STABLE_BANK = Path(
    "/tmp/shared-storage/OnPolicy/data/calibration/20260520_trc_round5_sota_v2/"
    "mtr_late3_toolv2_codev2/trc96_expert_trajectories.jsonl"
)
DEFAULT_CODE_ROLLOUTS = [
    Path(
        "/tmp/shared-storage/OnPolicy/data/calibration/code_p0_v3_20260518/expert_rollouts/"
        "code_expert_reasonflux_coder7b_code_p0_v3_train64_s8_seed20260518.jsonl"
    ),
    Path(
        "/tmp/shared-storage/OnPolicy/data/calibration/code_p0_v3_20260518/expert_rollouts/"
        "code_expert_deepseek_r1_distill_qwen7b_code_p0_v3_train64_s8_seed20260518.merged.jsonl"
    ),
]
DEFAULT_OUTPUT_DIR = Path(
    "/tmp/shared-storage/OnPolicy/data/calibration/20260520_trc_round16_nonleak_code_contrast_v1"
)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stable_rows = read_jsonl(Path(args.stable_bank).expanduser())
    code_rollout_args = args.code_rollout or [str(path) for path in DEFAULT_CODE_ROLLOUTS]
    code_rollouts = [Path(item).expanduser() for item in code_rollout_args]

    tool_rows = select_task_rows(stable_rows, "tool", int(args.tool_count))
    memory_rows = select_task_rows(stable_rows, "memory", int(args.memory_count))
    contrast_candidates = collect_code_contrast_candidates(
        code_rollouts,
        positive_threshold=float(args.positive_threshold),
        negative_threshold=float(args.negative_threshold),
    )
    contrast_rows = select_contrast_rows(
        contrast_candidates,
        count=int(args.code_contrast_count),
    )
    positive_candidates = collect_code_positive_candidates(
        code_rollouts,
        positive_threshold=float(args.positive_threshold),
    )
    positive_rows = select_positive_fill_rows(
        positive_candidates,
        exclude_prompt_ids={str(row["prompt_id"]) for row in contrast_rows},
        count=int(args.code_positive_fill_count),
    )

    rows: list[dict[str, Any]] = []
    rows.extend(with_provenance(row, component="stable_tool", source="stable_bank") for row in tool_rows)
    rows.extend(with_provenance(row, component="stable_memory", source="stable_bank") for row in memory_rows)
    rows.extend(contrast_rows)
    rows.extend(positive_rows)
    renumber(rows)
    validate(rows, expected_total=int(args.tool_count) + int(args.memory_count) + int(args.code_contrast_count) + int(args.code_positive_fill_count))

    out_jsonl = output_dir / "trc96_expert_trajectories.jsonl"
    write_jsonl(out_jsonl, rows)
    summary = {
        "format": "trc_round16_nonleak_code_contrast_summary_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output": str(out_jsonl),
        "input_banks": {
            "stable_bank": str(Path(args.stable_bank).expanduser().resolve()),
            "code_rollouts": [str(path.resolve()) for path in code_rollouts],
        },
        "leakage_policy": (
            "Code contrast rows come from CodeP0-v3 CodeContests_train rollouts only. "
            "No LiveBench, LiveCodeBench, or formal CURE eval prompt/output/test is used."
        ),
        "num_rows": len(rows),
        "task_counts": dict(sorted(Counter(row.get("task") for row in rows).items())),
        "expert_counts": dict(sorted(Counter(row.get("expert") for row in rows).items())),
        "code_policy": {
            "contrast_count": int(args.code_contrast_count),
            "positive_fill_count": int(args.code_positive_fill_count),
            "positive_threshold": float(args.positive_threshold),
            "negative_threshold": float(args.negative_threshold),
        "selection": "ReasonFlux same-prompt pass/fail first, DeepSeek fallback second, then role/tag-balanced deterministic fill.",
            "expert_assignment": "All Code rows train the existing code task vector; DeepSeek fallback rows are marked in metadata.",
        },
        "code_distribution": summarize_code(rows),
        "contrast_candidate_summary": summarize_code(contrast_candidates),
    }
    write_json(output_dir / "summary.json", summary)
    (output_dir / "README.md").write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def select_task_rows(rows: list[dict[str, Any]], task: str, count: int) -> list[dict[str, Any]]:
    selected = [deepcopy(row) for row in rows if row.get("task") == task]
    if len(selected) < count:
        raise ValueError(f"Need {count} {task} rows, found {len(selected)}")
    return selected[:count]


def collect_code_contrast_candidates(paths: list[Path], *, positive_threshold: float, negative_threshold: float) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for source_priority, path in enumerate(paths):
        source_name = source_name_from_path(path)
        for row_index, row in enumerate(read_jsonl(path)):
            if str(row.get("task") or "code") != "code":
                continue
            samples = list(row.get("samples") or [])
            positive = choose_positive_sample(samples, threshold=positive_threshold)
            negative = choose_negative_sample(samples, threshold=negative_threshold)
            if positive is None or negative is None:
                continue
            materialized = materialize_code_contrast_row(
                row=row,
                row_index=row_index,
                positive=positive,
                negative=negative,
                source_name=source_name,
                source_path=path,
                source_priority=source_priority,
            )
            candidates.append(materialized)
    return candidates


def collect_code_positive_candidates(paths: list[Path], *, positive_threshold: float) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for source_priority, path in enumerate(paths):
        source_name = source_name_from_path(path)
        for row_index, row in enumerate(read_jsonl(path)):
            if str(row.get("task") or "code") != "code":
                continue
            positive = choose_positive_sample(list(row.get("samples") or []), threshold=positive_threshold)
            if positive is None:
                continue
            candidates.append(
                materialize_code_positive_row(
                    row=row,
                    row_index=row_index,
                    positive=positive,
                    source_name=source_name,
                    source_path=path,
                    source_priority=source_priority,
                )
            )
    return candidates


def choose_positive_sample(samples: list[dict[str, Any]], *, threshold: float) -> tuple[int, dict[str, Any], float] | None:
    positives = []
    for index, sample in enumerate(samples):
        reward = sample_reward_train(sample)
        text = str(sample.get("text") or sample.get("response") or "")
        if text.strip() and (bool(sample.get("success")) or reward >= threshold):
            positives.append((index, sample, reward))
    if not positives:
        return None
    positives.sort(key=lambda item: (-item[2], int(item[1].get("length") or len(str(item[1].get("text") or "").split())), item[0]))
    return positives[0]


def choose_negative_sample(samples: list[dict[str, Any]], *, threshold: float) -> tuple[int, dict[str, Any], float] | None:
    negatives = []
    for index, sample in enumerate(samples):
        reward = sample_reward_train(sample)
        text = str(sample.get("text") or sample.get("response") or "")
        if text.strip() and reward < threshold:
            negatives.append((index, sample, reward))
    if not negatives:
        return None
    negatives.sort(key=lambda item: (item[2], int(item[1].get("length") or len(str(item[1].get("text") or "").split())), item[0]))
    return negatives[0]


def materialize_code_contrast_row(
    *,
    row: dict[str, Any],
    row_index: int,
    positive: tuple[int, dict[str, Any], float],
    negative: tuple[int, dict[str, Any], float],
    source_name: str,
    source_path: Path,
    source_priority: int,
) -> dict[str, Any]:
    positive_index, positive_sample, positive_reward = positive
    negative_index, negative_sample, negative_reward = negative
    prompt_id = str(row.get("prompt_id") or row.get("group_id") or f"code_{row_index:05d}")
    sample_id = str(positive_sample.get("sample_id") or f"{prompt_id}__k{positive_index}")
    negative_sample_id = str(negative_sample.get("sample_id") or f"{prompt_id}__k{negative_index}")
    metadata = (row.get("reference") or {}).get("metadata") or {}
    row_out = {
        "format": "trc_expert_trajectory_v1",
        "trajectory_id": f"trc16contrast__{source_name}__{sample_id}",
        "task": "code",
        "expert": "code",
        "source_name": source_name,
        "source_path": str(source_path.resolve()),
        "prompt_id": prompt_id,
        "group_id": str(row.get("group_id") or prompt_id),
        "sample_id": sample_id,
        "prompt": str(row.get("prompt") or ""),
        "rendered_prompt": str(row.get("rendered_prompt") or row.get("prompt") or ""),
        "response": str(positive_sample.get("text") or positive_sample.get("response") or ""),
        "negative_response": str(negative_sample.get("text") or negative_sample.get("response") or ""),
        "negative_reward_train": negative_reward,
        "negative_sample_id": negative_sample_id,
        "reference": dict(row.get("reference") or {}),
        "reward": float(positive_sample.get("reward", positive_reward) or 0.0),
        "reward_train": positive_reward,
        "task_reward": as_optional_float(positive_sample.get("task_reward")),
        "length": int(positive_sample.get("length") or len(str(positive_sample.get("text") or "").split())),
        "success": True,
        "row_metadata": {
            "round16_nonleak_code_contrast": {
                "component": "code_pass_fail_contrast",
                "source_priority": source_priority,
                "source_row": row_index,
                "source_name": source_name,
                "source_path": str(source_path.resolve()),
                "positive_sample_index": positive_index,
                "positive_sample_id": sample_id,
                "positive_reward_train": positive_reward,
                "negative_sample_index": negative_index,
                "negative_sample_id": negative_sample_id,
                "negative_reward_train": negative_reward,
                "code_bank_role": metadata.get("code_bank_role"),
                "primary_code_tag": metadata.get("primary_code_tag"),
                "code_tags": metadata.get("code_tags"),
            },
            "frontier": row.get("frontier"),
            "run_id": row.get("run_id"),
            "policy_id": row.get("policy_id"),
            "seed": row.get("seed"),
        },
        "sample_metadata": {
            "details": positive_sample.get("details"),
            "critical_spans": positive_sample.get("critical_spans"),
            "behavior_span_reward": positive_sample.get("behavior_span_reward"),
        },
    }
    return row_out


def materialize_code_positive_row(
    *,
    row: dict[str, Any],
    row_index: int,
    positive: tuple[int, dict[str, Any], float],
    source_name: str,
    source_path: Path,
    source_priority: int,
) -> dict[str, Any]:
    positive_index, positive_sample, positive_reward = positive
    prompt_id = str(row.get("prompt_id") or row.get("group_id") or f"code_{row_index:05d}")
    sample_id = str(positive_sample.get("sample_id") or f"{prompt_id}__k{positive_index}")
    metadata = (row.get("reference") or {}).get("metadata") or {}
    return {
        "format": "trc_expert_trajectory_v1",
        "trajectory_id": f"trc16positive__{source_name}__{sample_id}",
        "task": "code",
        "expert": "code",
        "source_name": source_name,
        "source_path": str(source_path.resolve()),
        "prompt_id": prompt_id,
        "group_id": str(row.get("group_id") or prompt_id),
        "sample_id": sample_id,
        "prompt": str(row.get("prompt") or ""),
        "rendered_prompt": str(row.get("rendered_prompt") or row.get("prompt") or ""),
        "response": str(positive_sample.get("text") or positive_sample.get("response") or ""),
        "reference": dict(row.get("reference") or {}),
        "reward": float(positive_sample.get("reward", positive_reward) or 0.0),
        "reward_train": positive_reward,
        "task_reward": as_optional_float(positive_sample.get("task_reward")),
        "length": int(positive_sample.get("length") or len(str(positive_sample.get("text") or "").split())),
        "success": True,
        "row_metadata": {
            "round16_nonleak_code_contrast": {
                "component": "code_positive_fill",
                "source_priority": source_priority,
                "source_row": row_index,
                "source_name": source_name,
                "source_path": str(source_path.resolve()),
                "positive_sample_index": positive_index,
                "positive_sample_id": sample_id,
                "positive_reward_train": positive_reward,
                "code_bank_role": metadata.get("code_bank_role"),
                "primary_code_tag": metadata.get("primary_code_tag"),
                "code_tags": metadata.get("code_tags"),
            },
            "frontier": row.get("frontier"),
            "run_id": row.get("run_id"),
            "policy_id": row.get("policy_id"),
            "seed": row.get("seed"),
        },
        "sample_metadata": {
            "details": positive_sample.get("details"),
            "critical_spans": positive_sample.get("critical_spans"),
            "behavior_span_reward": positive_sample.get("behavior_span_reward"),
        },
    }


def select_contrast_rows(rows: list[dict[str, Any]], *, count: int) -> list[dict[str, Any]]:
    if len(rows) < count:
        raise ValueError(f"Need {count} contrast rows, found {len(rows)}")
    selected: list[dict[str, Any]] = []
    used_prompts: set[str] = set()
    role_counts: Counter[str] = Counter()
    tag_counts: Counter[str] = Counter()
    ordered = sorted(rows, key=contrast_sort_key)
    while ordered and len(selected) < count:
        best_index = min(
            range(len(ordered)),
            key=lambda idx: diversity_sort_key(ordered[idx], role_counts=role_counts, tag_counts=tag_counts, used_prompts=used_prompts),
        )
        row = ordered.pop(best_index)
        prompt_id = str(row.get("prompt_id") or "")
        if prompt_id in used_prompts:
            continue
        selected.append(row)
        used_prompts.add(prompt_id)
        role_counts[code_role(row)] += 1
        tag_counts[primary_code_tag(row)] += 1
    if len(selected) < count:
        raise ValueError(f"Unable to select {count} unique contrast prompts; selected {len(selected)}")
    return selected


def contrast_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    md = (row.get("row_metadata") or {}).get("round16_nonleak_code_contrast") or {}
    return (
        int(md.get("source_priority") or 0),
        str(md.get("code_bank_role") or ""),
        str(md.get("primary_code_tag") or ""),
        float(md.get("negative_reward_train") or 0.0),
        str(row.get("prompt_id") or ""),
        str(row.get("sample_id") or ""),
    )


def diversity_sort_key(row: dict[str, Any], *, role_counts: Counter[str], tag_counts: Counter[str], used_prompts: set[str]) -> tuple[Any, ...]:
    prompt_id = str(row.get("prompt_id") or "")
    return (
        1 if prompt_id in used_prompts else 0,
        int((row.get("row_metadata") or {}).get("round16_nonleak_code_contrast", {}).get("source_priority") or 0),
        role_counts[code_role(row)],
        tag_counts[primary_code_tag(row)],
        contrast_sort_key(row),
    )


def select_positive_fill_rows(rows: list[dict[str, Any]], *, exclude_prompt_ids: set[str], count: int) -> list[dict[str, Any]]:
    primary_candidates = [
        deepcopy(row)
        for row in rows
        if row.get("task") == "code" and str(row.get("prompt_id") or "") not in exclude_prompt_ids
    ]
    fallback_candidates = [deepcopy(row) for row in rows if row.get("task") == "code"]
    if len(fallback_candidates) < count:
        raise ValueError(f"Need {count} positive fill Code rows, found {len(fallback_candidates)}")
    selected: list[dict[str, Any]] = []
    used_prompts: set[str] = set()
    role_counts: Counter[str] = Counter()
    tag_counts: Counter[str] = Counter()
    ordered = sorted(primary_candidates, key=contrast_sort_key)
    while ordered and len(selected) < count:
        best_index = min(
            range(len(ordered)),
            key=lambda idx: diversity_sort_key(ordered[idx], role_counts=role_counts, tag_counts=tag_counts, used_prompts=used_prompts),
        )
        row = ordered.pop(best_index)
        prompt_id = str(row.get("prompt_id") or "")
        if prompt_id in used_prompts:
            continue
        selected.append(row)
        used_prompts.add(prompt_id)
        role_counts[code_role(row)] += 1
        tag_counts[primary_code_tag(row)] += 1
    if len(selected) < count:
        selected_keys = {(str(row.get("source_name") or ""), str(row.get("sample_id") or "")) for row in selected}
        extras = [
            row
            for row in sorted(fallback_candidates, key=contrast_sort_key)
            if (str(row.get("source_name") or ""), str(row.get("sample_id") or "")) not in selected_keys
        ]
        selected.extend(extras[: count - len(selected)])
    if len(selected) < count:
        raise ValueError(f"Unable to select {count} positive fill rows; selected {len(selected)}")
    return selected


def with_provenance(row: dict[str, Any], *, component: str, source: str) -> dict[str, Any]:
    copied = deepcopy(row)
    metadata = dict(copied.get("row_metadata") or {})
    metadata["round16_nonleak_code_contrast"] = {
        "component": component,
        "source": source,
    }
    copied["row_metadata"] = metadata
    return copied


def renumber(rows: list[dict[str, Any]]) -> None:
    for index, row in enumerate(rows):
        old = str(row.get("trajectory_id") or row.get("sample_id") or index)
        row["trajectory_id"] = f"trc16__{index:03d}__{row.get('task')}__{row.get('expert')}__{stable_suffix(old)}"


def validate(rows: list[dict[str, Any]], *, expected_total: int) -> None:
    if len(rows) != expected_total:
        raise ValueError(f"Expected {expected_total} rows, found {len(rows)}")
    counts = Counter(row.get("task") for row in rows)
    if counts.get("tool") != 32 or counts.get("memory") != 32 or counts.get("code") != 32:
        raise ValueError(f"Expected 32 rows per task, got {dict(counts)}")
    for index, row in enumerate(rows):
        if not str(row.get("rendered_prompt") or row.get("prompt") or "").strip():
            raise ValueError(f"row {index} has blank prompt")
        if not str(row.get("response") or "").strip():
            raise ValueError(f"row {index} has blank response")
        if row.get("task") == "code" and str(row.get("negative_response") or "").strip():
            if sample_reward_train(row) < 1.0:
                raise ValueError(f"contrast code row {index} has non-positive response reward")


def summarize_code(rows: list[dict[str, Any]]) -> dict[str, Any]:
    code_rows = [row for row in rows if row.get("task") == "code"]
    roles = Counter()
    tags = Counter()
    components = Counter()
    sources = Counter()
    negative_rewards = []
    for row in code_rows:
        roles[code_role(row)] += 1
        tags[primary_code_tag(row)] += 1
        metadata = (row.get("row_metadata") or {}).get("round16_nonleak_code_contrast") or {}
        components[str(metadata.get("component") or "unknown")] += 1
        sources[str(row.get("source_name") or "unknown")] += 1
        if str(row.get("negative_response") or "").strip():
            negative_rewards.append(float(row.get("negative_reward_train") or 0.0))
    return {
        "rows": len(code_rows),
        "unique_prompts": len({row.get("prompt_id") for row in code_rows}),
        "contrast_rows": sum(1 for row in code_rows if str(row.get("negative_response") or "").strip()),
        "component_counts": dict(sorted(components.items())),
        "source_counts": dict(sorted(sources.items())),
        "role_counts": dict(sorted(roles.items())),
        "primary_tag_counts": dict(sorted(tags.items())),
        "negative_reward_train": {
            "min": min(negative_rewards) if negative_rewards else None,
            "mean": (sum(negative_rewards) / len(negative_rewards)) if negative_rewards else None,
            "max": max(negative_rewards) if negative_rewards else None,
        },
    }


def code_role(row: dict[str, Any]) -> str:
    metadata = (row.get("reference") or {}).get("metadata") or {}
    contrast = (row.get("row_metadata") or {}).get("round16_nonleak_code_contrast") or {}
    return str(metadata.get("code_bank_role") or contrast.get("code_bank_role") or "unknown")


def primary_code_tag(row: dict[str, Any]) -> str:
    metadata = (row.get("reference") or {}).get("metadata") or {}
    contrast = (row.get("row_metadata") or {}).get("round16_nonleak_code_contrast") or {}
    return str(metadata.get("primary_code_tag") or contrast.get("primary_code_tag") or "unknown")


def source_name_from_path(path: Path) -> str:
    name = path.name.lower()
    if "reasonflux" in name:
        return "code_source_reasonflux"
    if "deepseek" in name:
        return "code_source_deepseek"
    return "code_source_" + stable_suffix(path.stem.lower())[:40]


def sample_reward_train(sample: dict[str, Any]) -> float:
    value = sample.get("reward_train", sample.get("reward", 0.0))
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def stable_suffix(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[-96:] or "row"


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Round16 Non-Leak Code Contrast Calibration",
            "",
            f"- Output: `{summary['output']}`",
            f"- Rows: `{summary['num_rows']}`",
            f"- Task counts: `{summary['task_counts']}`",
            f"- Leakage policy: {summary['leakage_policy']}",
            f"- Code distribution: `{summary['code_distribution']}`",
            "",
            "This bank is intended for the paper-main Code pass/fail contrast line.",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stable-bank", default=str(DEFAULT_STABLE_BANK))
    parser.add_argument(
        "--code-rollout",
        action="append",
        default=None,
        help="Code rollout JSONL. Repeat to override the default ReasonFlux + DeepSeek merged sources.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--tool-count", type=int, default=32)
    parser.add_argument("--memory-count", type=int, default=32)
    parser.add_argument(
        "--code-contrast-count",
        type=int,
        default=22,
        help="Strict unique same-prompt pass/fail rows. Default 22 is the available non-leak RF+DS-merged unique count.",
    )
    parser.add_argument("--code-positive-fill-count", type=int, default=10)
    parser.add_argument("--positive-threshold", type=float, default=1.0)
    parser.add_argument("--negative-threshold", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
