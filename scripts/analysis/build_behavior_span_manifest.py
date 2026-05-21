#!/usr/bin/env python3
"""Build behavior-span probe manifests from rollouts or inference outputs.

The output rows are compatible with
`scripts/attention_pauh/probe_signed_utility.py`.  The script is designed for
RCRF-style attribution: select high-confidence successful and failed behavior
trajectories for Tool/Memory/Code, then let signed-utility probing measure which
expert residuals support or harm those spans.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


TASK_ALIASES = {
    "tool": "tool",
    "toolrl": "tool",
    "bfcl": "tool",
    "memory": "memory",
    "mem": "memory",
    "hotpotqa": "memory",
    "memagent": "memory",
    "code": "code",
    "cure": "code",
    "livebench": "code",
    "livecodebench": "code",
}


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_correct = load_summary_correct_indices(args.summary_json)

    records = []
    for source_name, raw_path in args.source:
        path = Path(raw_path).expanduser()
        records.extend(
            load_source_records(
                source_name=source_name,
                path=path,
                summary_correct=summary_correct,
                tool_positive_mode=args.tool_positive_mode,
                min_positive_reward=args.min_positive_reward,
                max_negative_reward=args.max_negative_reward,
                include_unscored_as_positive=args.include_unscored_as_positive,
            )
        )
    positive, negative = select_records(
        records,
        max_positive_per_task=args.max_positive_per_task,
        max_negative_per_task=args.max_negative_per_task,
    )
    all_rows = positive + negative
    pairs = build_pair_rows(positive, negative)

    paths = {
        "positive_jsonl": output_dir / "behavior_positive.jsonl",
        "negative_jsonl": output_dir / "behavior_negative.jsonl",
        "all_jsonl": output_dir / "behavior_all.jsonl",
        "pairs_jsonl": output_dir / "behavior_pairs.jsonl",
        "summary_json": output_dir / "behavior_manifest_summary.json",
        "summary_md": output_dir / "behavior_manifest_summary.md",
    }
    write_jsonl(paths["positive_jsonl"], positive)
    write_jsonl(paths["negative_jsonl"], negative)
    write_jsonl(paths["all_jsonl"], all_rows)
    write_jsonl(paths["pairs_jsonl"], pairs)

    summary = build_summary(args, records, positive, negative, pairs, paths)
    paths["summary_json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["summary_md"].write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "summary": str(paths["summary_json"])}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", nargs=2, action="append", required=True, metavar=("NAME", "JSON_OR_JSONL"))
    parser.add_argument("--summary-json", action="append", default=[], help="Optional evaluation summary with correct_indices.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-positive-per-task", type=int, default=32)
    parser.add_argument("--max-negative-per-task", type=int, default=32)
    parser.add_argument(
        "--tool-positive-mode",
        choices=("exact", "behavior", "reward"),
        default="exact",
        help="Tool positive definition for rollout rows.",
    )
    parser.add_argument("--min-positive-reward", type=float, default=0.95)
    parser.add_argument("--max-negative-reward", type=float, default=0.5)
    parser.add_argument(
        "--include-unscored-as-positive",
        action="store_true",
        help="Use inference rows without summary correctness as positive behavior rows. Default off.",
    )
    return parser.parse_args()


def load_summary_correct_indices(paths: Iterable[str]) -> dict[str, set[int]]:
    correct: dict[str, set[int]] = defaultdict(set)
    for raw in paths:
        path = Path(raw).expanduser()
        if not path.exists():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        for task_name, task_summary in iter_task_summaries(payload):
            indices = task_summary.get("correct_indices") or []
            correct[normalize_task(task_name)].update(int(item) for item in indices)
    return dict(correct)


def iter_task_summaries(payload: Any) -> Iterable[tuple[str, Mapping[str, Any]]]:
    if isinstance(payload, dict):
        if payload.get("task") and isinstance(payload.get("correct_indices"), list):
            yield str(payload["task"]), payload
        for key, value in payload.items():
            if isinstance(value, dict) and isinstance(value.get("correct_indices"), list):
                yield str(key), value


def load_source_records(
    *,
    source_name: str,
    path: Path,
    summary_correct: Mapping[str, set[int]],
    tool_positive_mode: str,
    min_positive_reward: float,
    max_negative_reward: float,
    include_unscored_as_positive: bool,
) -> list[dict[str, Any]]:
    rows = read_json_rows(path)
    records: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        if isinstance(row.get("samples"), list):
            records.extend(
                rollout_records(
                    source_name=source_name,
                    source_path=path,
                    row=row,
                    row_index=row_index,
                    tool_positive_mode=tool_positive_mode,
                    min_positive_reward=min_positive_reward,
                    max_negative_reward=max_negative_reward,
                )
            )
        else:
            item = inference_record(
                source_name=source_name,
                source_path=path,
                row=row,
                row_index=row_index,
                summary_correct=summary_correct,
                include_unscored_as_positive=include_unscored_as_positive,
            )
            if item is not None:
                records.append(item)
    return records


def read_json_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [dict(item) for item in payload]
        if isinstance(payload, dict) and isinstance(payload.get("data"), list):
            return [dict(item) for item in payload["data"]]
        raise ValueError(f"Unsupported JSON shape: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def rollout_records(
    *,
    source_name: str,
    source_path: Path,
    row: Mapping[str, Any],
    row_index: int,
    tool_positive_mode: str,
    min_positive_reward: float,
    max_negative_reward: float,
) -> list[dict[str, Any]]:
    task = normalize_task(row.get("task") or row.get("ability") or row.get("data_source"))
    prompt_id = str(row.get("prompt_id") or row.get("group_id") or f"{source_name}__{row_index:05d}")
    records = []
    for sample_index, sample in enumerate(row.get("samples") or []):
        response = str(sample.get("text") or sample.get("response") or "")
        if not response:
            continue
        score = score_from_sample(sample)
        label, label_reason = classify_rollout_sample(
            task=task,
            sample=sample,
            tool_positive_mode=tool_positive_mode,
            min_positive_reward=min_positive_reward,
            max_negative_reward=max_negative_reward,
        )
        if label is None:
            continue
        records.append(
            make_record(
                source_name=source_name,
                source_path=source_path,
                task=task,
                prompt_id=prompt_id,
                role=label,
                prompt=str(row.get("prompt") or ""),
                rendered_prompt=str(row.get("rendered_prompt") or row.get("prompt") or ""),
                messages=row.get("messages"),
                response=response,
                score=score,
                label_reason=label_reason,
                original_index=row_index,
                original_sample_id=str(sample.get("sample_id") or f"{prompt_id}__k{sample_index}"),
                details=sample.get("details") or {},
                reference=row.get("reference"),
            )
        )
    return records


def classify_rollout_sample(
    *,
    task: str,
    sample: Mapping[str, Any],
    tool_positive_mode: str,
    min_positive_reward: float,
    max_negative_reward: float,
) -> tuple[str | None, str]:
    details = sample.get("details") or {}
    score = score_from_sample(sample)
    if task == "tool":
        exact = bool(details.get("exact_tool_match"))
        behavior = safe_float(sample.get("behavior_span_reward") or details.get("behavior_span_reward"))
        if tool_positive_mode == "exact" and exact:
            return "positive", "tool exact_tool_match"
        if tool_positive_mode == "behavior" and behavior >= 1.0:
            return "positive", "tool behavior_span_reward >= 1"
        if tool_positive_mode == "reward" and score >= min_positive_reward:
            return "positive", f"tool reward >= {min_positive_reward}"
        if not exact or score <= max_negative_reward:
            return "negative", "tool non-exact or low reward"
        return None, "tool ambiguous"
    if bool(sample.get("success")) or score >= min_positive_reward:
        return "positive", f"{task} success or reward >= {min_positive_reward}"
    if score <= max_negative_reward:
        return "negative", f"{task} reward <= {max_negative_reward}"
    return None, f"{task} ambiguous"


def inference_record(
    *,
    source_name: str,
    source_path: Path,
    row: Mapping[str, Any],
    row_index: int,
    summary_correct: Mapping[str, set[int]],
    include_unscored_as_positive: bool,
) -> dict[str, Any] | None:
    task = normalize_task(row.get("task") or row.get("ability") or row.get("data_source"))
    index = int(row.get("index") if row.get("index") is not None else row_index)
    correct_set = summary_correct.get(task, set())
    has_correctness = bool(correct_set)
    if has_correctness:
        label = "positive" if index in correct_set else "negative"
        label_reason = "summary correct_indices" if label == "positive" else "not in summary correct_indices"
    elif include_unscored_as_positive:
        label = "positive"
        label_reason = "unscored included as positive behavior"
    else:
        return None
    response = str(row.get("response") or row.get("completion") or row.get("expert_response") or row.get("chosen_response") or "")
    if not response:
        return None
    prompt = str(row.get("prompt") or "")
    rendered_prompt = str(row.get("rendered_prompt") or prompt)
    return make_record(
        source_name=source_name,
        source_path=source_path,
        task=task,
        prompt_id=str(row.get("prompt_id") or row.get("id") or f"{source_name}__{index:05d}"),
        role=label,
        prompt=prompt,
        rendered_prompt=rendered_prompt,
        messages=row.get("messages"),
        response=response,
        score=1.0 if label == "positive" else 0.0,
        label_reason=label_reason,
        original_index=index,
        original_sample_id=str(row.get("sample_id") or f"{source_name}__{index:05d}"),
        details=row.get("details") or {},
        reference=row.get("ground_truth") or row.get("reference"),
    )


def make_record(
    *,
    source_name: str,
    source_path: Path,
    task: str,
    prompt_id: str,
    role: str,
    prompt: str,
    rendered_prompt: str,
    messages: Any,
    response: str,
    score: float,
    label_reason: str,
    original_index: int,
    original_sample_id: str,
    details: Mapping[str, Any],
    reference: Any,
) -> dict[str, Any]:
    pair_id = f"{task}__{prompt_id}"
    sample_id = f"{pair_id}__{role}__response"
    record = {
        "format": "behavior_span_probe_row_v1",
        "source_name": source_name,
        "source_path": str(source_path),
        "task": task,
        "ability": task,
        "data_source": source_name,
        "split": "mechanism_probe",
        "prompt_id": prompt_id,
        "pair_id": pair_id,
        "sample_id": sample_id,
        "role": role,
        "prompt": prompt,
        "rendered_prompt": rendered_prompt,
        "response": response,
        "score": score,
        "label_reason": label_reason,
        "original_index": original_index,
        "original_sample_id": original_sample_id,
        "details": dict(details),
        "reference": reference,
    }
    if messages:
        record["messages"] = messages
    return record


def select_records(
    records: list[dict[str, Any]], *, max_positive_per_task: int, max_negative_per_task: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        grouped[(record["task"], record["prompt_id"])][record["role"]].append(record)

    best_positive: list[dict[str, Any]] = []
    worst_negative: list[dict[str, Any]] = []
    for (_task, _prompt_id), by_role in grouped.items():
        positives = by_role.get("positive") or []
        negatives = by_role.get("negative") or []
        if positives:
            best_positive.append(max(positives, key=lambda row: safe_float(row.get("score"))))
        if negatives:
            worst_negative.append(min(negatives, key=lambda row: safe_float(row.get("score"))))

    positive = take_per_task(best_positive, max_positive_per_task, reverse=True)
    negative = take_per_task(worst_negative, max_negative_per_task, reverse=False)
    return positive, negative


def take_per_task(rows: list[dict[str, Any]], limit: int, *, reverse: bool) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(row)
    for task in sorted(by_task):
        ranked = sorted(
            by_task[task],
            key=lambda row: (safe_float(row.get("score")), str(row.get("prompt_id"))),
            reverse=reverse,
        )
        output.extend(ranked[:limit])
    return output


def build_pair_rows(positive: list[dict[str, Any]], negative: list[dict[str, Any]]) -> list[dict[str, Any]]:
    positives = {(row["task"], row["prompt_id"]): row for row in positive}
    negatives = {(row["task"], row["prompt_id"]): row for row in negative}
    rows = []
    for key in sorted(set(positives) & set(negatives)):
        pos = positives[key]
        neg = negatives[key]
        row = dict(pos)
        row.update(
            {
                "format": "behavior_span_contrast_row_v1",
                "negative_response": neg["response"],
                "negative_source_name": neg["source_name"],
                "negative_original_sample_id": neg["original_sample_id"],
                "negative_score": neg["score"],
                "negative_label_reason": neg["label_reason"],
            }
        )
        rows.append(row)
    return rows


def build_summary(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    positive: list[dict[str, Any]],
    negative: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    paths: Mapping[str, Path],
) -> dict[str, Any]:
    return {
        "format": "behavior_span_manifest_summary_v1",
        "sources": {name: str(Path(path).expanduser()) for name, path in args.source},
        "config": {
            "max_positive_per_task": args.max_positive_per_task,
            "max_negative_per_task": args.max_negative_per_task,
            "tool_positive_mode": args.tool_positive_mode,
            "min_positive_reward": args.min_positive_reward,
            "max_negative_reward": args.max_negative_reward,
            "include_unscored_as_positive": args.include_unscored_as_positive,
        },
        "record_counts": dict(Counter(row["role"] for row in records)),
        "selected_counts": {
            "positive": dict(Counter(row["task"] for row in positive)),
            "negative": dict(Counter(row["task"] for row in negative)),
            "pairs": dict(Counter(row["task"] for row in pairs)),
        },
        "outputs": {name: str(path) for name, path in paths.items()},
        "probe_commands": {
            "signature_positive": (
                "python scripts/attention_pauh/probe_signed_utility.py "
                "--mode-manifest /tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json "
                f"--trajectory-jsonl {paths['positive_jsonl']} --span signature --scope all-linear --write-row-details "
                "--output-dir <probe_output_dir>"
            ),
            "negative_response": (
                "python scripts/attention_pauh/probe_signed_utility.py "
                "--mode-manifest /tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json "
                f"--trajectory-jsonl {paths['negative_jsonl']} --span signature --scope all-linear --write-row-details "
                "--output-dir <probe_output_dir>"
            ),
        },
    }


def render_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Behavior Span Manifest",
        "",
        "## Sources",
        "",
    ]
    for name, path in sorted(summary["sources"].items()):
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(["", "## Selected Counts", "", "### Positive", "", "| task | count |", "|---|---:|"])
    for task, count in sorted(summary["selected_counts"]["positive"].items()):
        lines.append(f"| {task} | {count} |")
    lines.extend(["", "### Negative", "", "| task | count |", "|---|---:|"])
    for task, count in sorted(summary["selected_counts"]["negative"].items()):
        lines.append(f"| {task} | {count} |")
    lines.extend(["", "### Same-prompt pairs", "", "| task | count |", "|---|---:|"])
    for task, count in sorted(summary["selected_counts"]["pairs"].items()):
        lines.append(f"| {task} | {count} |")
    lines.extend(["", "## Outputs", ""])
    for name, path in sorted(summary["outputs"].items()):
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(["", "## Probe commands", ""])
    for name, command in sorted(summary["probe_commands"].items()):
        lines.extend([f"### {name}", "", "```bash", command, "```", ""])
    return "\n".join(lines)


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def score_from_sample(sample: Mapping[str, Any]) -> float:
    for key in ("reward_train", "task_reward", "reward", "score"):
        if sample.get(key) is not None:
            return safe_float(sample.get(key))
    return 0.0


def normalize_task(value: Any) -> str:
    text = str(value or "").strip().lower()
    return TASK_ALIASES.get(text, text or "unknown")


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    main()
