#!/usr/bin/env python3
"""Probe task-level attention-matrix patterns on calibration trajectories.

The goal is diagnostic insight, not training. For each sampled trajectory, the
script runs the base model with ``output_attentions=True`` and measures where
response tokens attend: prompt, prompt tail, local response context, long-range
response context, attention sinks, and task-marker spans.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.attention_pauh.core import normalize_task_name  # noqa: E402


TASK_MARKERS = {
    "tool": ("tool", "function", "api", "schema", "argument", "call", "tool_call"),
    "memory": ("memory", "evidence", "passage", "observation", "update", "answer", "retrieve"),
    "code": ("```", "def ", "class ", "return", "input", "output", "example", "constraint", "assert"),
}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = tuple(normalize_task_name(task) for task in split_csv(args.tasks))
    rows = load_rows(
        [Path(path).expanduser() for path in args.trajectory_jsonl],
        tasks=tasks,
        samples_per_task=args.samples_per_task,
    )
    config = {
        "format": "attention_matrix_pattern_probe_config_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_model": args.base_model,
        "trajectory_jsonl": [str(Path(path).expanduser()) for path in args.trajectory_jsonl],
        "tasks": list(tasks),
        "samples_per_task": args.samples_per_task,
        "max_seq_length": args.max_seq_length,
        "response_tail_tokens": args.response_tail_tokens,
        "prompt_tail_tokens": args.prompt_tail_tokens,
        "local_window": args.local_window,
        "sink_tokens": args.sink_tokens,
        "torch_dtype": args.torch_dtype,
        "device": args.device,
    }
    write_json(output_dir / "attention_pattern_config.json", config)
    if args.plan_only:
        payload = {"config": config, "row_counts": count_rows_by_task(rows)}
        write_json(output_dir / "attention_pattern_plan.json", payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    summary = run_attention_probe(
        base_model=args.base_model,
        rows=rows,
        device=args.device,
        dtype=args.torch_dtype,
        max_seq_length=args.max_seq_length,
        response_tail_tokens=args.response_tail_tokens,
        prompt_tail_tokens=args.prompt_tail_tokens,
        local_window=args.local_window,
        sink_tokens=args.sink_tokens,
        output_dir=output_dir,
        write_row_details=args.write_row_details,
    )
    payload = {"config": config, **summary}
    write_json(output_dir / "attention_pattern_summary.json", payload)
    write_markdown_summary(output_dir / "attention_pattern_summary.md", payload)
    print(
        json.dumps(
            {
                "summary": str(output_dir / "attention_pattern_summary.json"),
                "row_counts": payload["row_counts"],
                "strongest_task_contrasts": payload["strongest_task_contrasts"][:8],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--trajectory-jsonl", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tasks", default="tool,memory,code")
    parser.add_argument("--samples-per-task", type=int, default=2)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--response-tail-tokens", type=int, default=384)
    parser.add_argument("--prompt-tail-tokens", type=int, default=256)
    parser.add_argument("--local-window", type=int, default=128)
    parser.add_argument("--sink-tokens", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--write-row-details", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def split_csv(raw: str) -> list[str]:
    return [item.strip() for item in str(raw or "").split(",") if item.strip()]


def load_rows(paths: list[Path], *, tasks: tuple[str, ...], samples_per_task: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = defaultdict(int)
    task_set = set(tasks)
    for path in paths:
        for row in read_json_rows(path):
            task = normalize_task_name(str(row.get("task") or row.get("ability") or row.get("data_source") or ""))
            if task not in task_set:
                continue
            if counts[task] >= samples_per_task:
                continue
            if not row_has_prompt_and_response(row):
                continue
            item = dict(row)
            item["task"] = task
            selected.append(item)
            counts[task] += 1
    missing = {task: samples_per_task - counts.get(task, 0) for task in tasks if counts.get(task, 0) < samples_per_task}
    if missing:
        raise ValueError(f"Not enough rows for tasks: {missing}")
    return selected


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


def row_has_prompt_and_response(row: Mapping[str, Any]) -> bool:
    return bool(row.get("rendered_prompt") or row.get("prompt") or row.get("messages")) and bool(
        row.get("response") or row.get("completion") or row.get("expert_response") or row.get("chosen_response")
    )


def count_rows_by_task(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row["task"])] += 1
    return dict(sorted(counts.items()))


def run_attention_probe(
    *,
    base_model: str,
    rows: list[dict[str, Any]],
    device: str,
    dtype: str,
    max_seq_length: int,
    response_tail_tokens: int,
    prompt_tail_tokens: int,
    local_window: int,
    sink_tokens: int,
    output_dir: Path,
    write_row_details: bool,
) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype_map[dtype],
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.to(device)
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    aggregate: dict[str, dict[int, dict[str, float]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    group_aggregate: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    row_writer = None
    if write_row_details:
        row_writer = (output_dir / "attention_pattern_rows.jsonl").open("w", encoding="utf-8")
    try:
        with torch.no_grad():
            for index, row in enumerate(rows):
                encoded = encode_text(
                    tokenizer,
                    row,
                    max_seq_length=max_seq_length,
                    response_tail_tokens=response_tail_tokens,
                    prompt_tail_tokens=prompt_tail_tokens,
                    sink_tokens=sink_tokens,
                )
                masks = encoded.pop("masks")
                task = str(row["task"])
                model_inputs = {key: value.to(device) for key, value in encoded.items()}
                outputs = model(**model_inputs, output_attentions=True, use_cache=False)
                row_metrics = summarize_attention_row(
                    attentions=outputs.attentions,
                    masks={key: value.to(device) for key, value in masks.items()},
                    local_window=local_window,
                    sink_tokens=sink_tokens,
                )
                for layer, metrics in row_metrics.items():
                    update_metric_sums(aggregate[task][layer], metrics)
                    update_metric_sums(group_aggregate[task][layer_group(layer, len(row_metrics))], metrics)
                if row_writer is not None:
                    row_writer.write(
                        json.dumps(
                            {
                                "row_id": str(row.get("sample_id") or row.get("prompt_id") or index),
                                "task": task,
                                "seq_len": int(model_inputs["input_ids"].shape[1]),
                                "metrics": {str(layer): metrics for layer, metrics in row_metrics.items()},
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                del outputs, model_inputs, encoded, masks, row_metrics
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
    finally:
        if row_writer is not None:
            row_writer.close()
        model.cpu()
        del model

    layer_summary = finalize_nested_metrics(aggregate)
    group_summary = finalize_nested_metrics(group_aggregate)
    return {
        "row_counts": count_rows_by_task(rows),
        "layer_summary": layer_summary,
        "layer_group_summary": group_summary,
        "strongest_task_contrasts": strongest_task_contrasts(group_summary),
    }


def encode_text(
    tokenizer: Any,
    row: Mapping[str, Any],
    *,
    max_seq_length: int,
    response_tail_tokens: int,
    prompt_tail_tokens: int,
    sink_tokens: int,
) -> dict[str, Any]:
    prompt = render_prompt(tokenizer, row)
    response = str(row.get("response") or row.get("completion") or row.get("expert_response") or row.get("chosen_response"))
    full_text = prompt + response
    encoded = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = list(encoded.input_ids)
    offsets = list(encoded.offset_mapping)
    prompt_char_len = len(prompt)
    if len(input_ids) > max_seq_length:
        overflow = len(input_ids) - int(max_seq_length)
        input_ids = input_ids[overflow:]
        offsets = offsets[overflow:]
    masks = build_span_masks(
        input_ids=input_ids,
        offsets=offsets,
        prompt_char_len=prompt_char_len,
        tokenizer=tokenizer,
        task=str(row["task"]),
        response_tail_tokens=response_tail_tokens,
        prompt_tail_tokens=prompt_tail_tokens,
        sink_tokens=sink_tokens,
    )
    tensor = torch.tensor([input_ids], dtype=torch.long)
    return {
        "input_ids": tensor,
        "attention_mask": torch.ones_like(tensor),
        "masks": masks,
    }


def render_prompt(tokenizer: Any, row: Mapping[str, Any]) -> str:
    if row.get("rendered_prompt"):
        return str(row["rendered_prompt"])
    if row.get("prompt"):
        return str(row["prompt"])
    if row.get("messages"):
        return tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True)
    raise ValueError("Row has no prompt.")


def build_span_masks(
    *,
    input_ids: list[int],
    offsets: list[tuple[int, int]],
    prompt_char_len: int,
    tokenizer: Any,
    task: str,
    response_tail_tokens: int,
    prompt_tail_tokens: int,
    sink_tokens: int,
) -> dict[str, torch.Tensor]:
    seq_len = len(input_ids)
    prompt = torch.tensor([start < prompt_char_len for start, _end in offsets], dtype=torch.bool)
    response = ~prompt
    if not bool(response.any()):
        response[-1] = True
    prompt_tail = prompt.clone()
    prompt_indices = torch.nonzero(prompt_tail, as_tuple=False).view(-1)
    if prompt_indices.numel() > prompt_tail_tokens:
        keep = prompt_indices[-int(prompt_tail_tokens) :]
        prompt_tail[:] = False
        prompt_tail[keep] = True
    response_tail = response.clone()
    response_indices = torch.nonzero(response_tail, as_tuple=False).view(-1)
    if response_indices.numel() > response_tail_tokens:
        keep = response_indices[-int(response_tail_tokens) :]
        response_tail[:] = False
        response_tail[keep] = True
    token_texts = tokenizer.convert_ids_to_tokens(input_ids)
    marker = build_marker_mask(token_texts, task=task)
    sink = torch.zeros(seq_len, dtype=torch.bool)
    sink[: min(int(sink_tokens), seq_len)] = True
    return {
        "prompt": prompt,
        "prompt_tail": prompt_tail,
        "response": response,
        "response_tail": response_tail,
        "marker": marker,
        "sink": sink,
    }


def build_marker_mask(token_texts: list[str], *, task: str) -> torch.Tensor:
    markers = TASK_MARKERS.get(normalize_task_name(task), ())
    mask = torch.zeros(len(token_texts), dtype=torch.bool)
    if not markers:
        return mask
    lowered = [text.lower().replace("Ġ", " ").replace("▁", " ") for text in token_texts]
    for index, text in enumerate(lowered):
        if any(marker in text for marker in markers):
            mask[index] = True
    return mask


def summarize_attention_row(
    *,
    attentions: tuple[torch.Tensor, ...],
    masks: Mapping[str, torch.Tensor],
    local_window: int,
    sink_tokens: int,
) -> dict[int, dict[str, float]]:
    response_query = masks["response_tail"].bool()
    if not bool(response_query.any()):
        response_query = masks["response"].bool()
    layer_metrics: dict[int, dict[str, float]] = {}
    for layer, attn in enumerate(attentions):
        # [batch, heads, query, key] -> [heads, selected_query, key]
        matrix = attn[0].float()
        selected = matrix[:, response_query, :]
        if selected.numel() == 0:
            continue
        metrics = {
            "prompt_mass": source_mass(selected, masks["prompt"]),
            "prompt_tail_mass": source_mass(selected, masks["prompt_tail"]),
            "response_mass": source_mass(selected, masks["response"]),
            "response_tail_mass": source_mass(selected, masks["response_tail"]),
            "marker_mass": source_mass(selected, masks["marker"]),
            "sink_mass": source_mass(selected, masks["sink"]),
            "entropy_norm": normalized_entropy(selected),
            "head_prompt_std": head_source_std(selected, masks["prompt"]),
        }
        metrics["local_response_mass"] = local_attention_mass(
            matrix=matrix,
            query_mask=response_query,
            source_mask=masks["response"],
            local_window=local_window,
        )
        metrics["long_response_mass"] = max(0.0, metrics["response_mass"] - metrics["local_response_mass"])
        metrics["non_sink_prompt_mass"] = max(0.0, metrics["prompt_mass"] - metrics["sink_mass"])
        layer_metrics[layer] = metrics
    return layer_metrics


def source_mass(selected: torch.Tensor, source_mask: torch.Tensor) -> float:
    mask = source_mask.to(device=selected.device, dtype=torch.bool)
    if not bool(mask.any()):
        return 0.0
    return float(selected[:, :, mask].sum(dim=-1).mean().item())


def normalized_entropy(selected: torch.Tensor) -> float:
    probs = selected.clamp_min(1.0e-12)
    entropy = -(probs * probs.log()).sum(dim=-1)
    denom = math.log(max(selected.shape[-1], 2))
    return float((entropy / denom).mean().item())


def head_source_std(selected: torch.Tensor, source_mask: torch.Tensor) -> float:
    mask = source_mask.to(device=selected.device, dtype=torch.bool)
    if not bool(mask.any()):
        return 0.0
    per_head = selected[:, :, mask].sum(dim=-1).mean(dim=-1)
    return float(per_head.std(unbiased=False).item())


def local_attention_mass(
    *,
    matrix: torch.Tensor,
    query_mask: torch.Tensor,
    source_mask: torch.Tensor,
    local_window: int,
) -> float:
    q_idx = torch.nonzero(query_mask.to(matrix.device), as_tuple=False).view(-1)
    if q_idx.numel() == 0:
        return 0.0
    seq_len = matrix.shape[-1]
    key_positions = torch.arange(seq_len, device=matrix.device)
    source = source_mask.to(matrix.device, dtype=torch.bool)
    local_masks = []
    for query in q_idx:
        local = (key_positions <= query) & (key_positions >= query - int(local_window)) & source
        local_masks.append(local)
    local_matrix = torch.stack(local_masks, dim=0)
    selected = matrix[:, q_idx, :]
    return float((selected * local_matrix.unsqueeze(0).to(selected.dtype)).sum(dim=-1).mean().item())


def layer_group(layer: int, num_layers: int) -> str:
    if layer < num_layers / 3.0:
        return "early"
    if layer < 2.0 * num_layers / 3.0:
        return "middle"
    return "late"


def update_metric_sums(stats: dict[str, float], metrics: Mapping[str, float]) -> None:
    stats["count"] += 1.0
    for key, value in metrics.items():
        stats[f"{key}_sum"] += float(value)


def finalize_nested_metrics(raw: Mapping[str, Mapping[Any, Mapping[str, float]]]) -> dict[str, dict[str, dict[str, float]]]:
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for task, bucket_map in raw.items():
        summary[task] = {}
        for bucket, stats in sorted(bucket_map.items(), key=lambda item: str(item[0])):
            count = max(float(stats.get("count", 0.0)), 1.0)
            summary[task][str(bucket)] = {
                key[:-4]: float(value) / count
                for key, value in sorted(stats.items())
                if key.endswith("_sum")
            }
            summary[task][str(bucket)]["count"] = float(stats.get("count", 0.0))
    return summary


def strongest_task_contrasts(group_summary: Mapping[str, Mapping[str, Mapping[str, float]]]) -> list[dict[str, Any]]:
    metrics = [
        "prompt_mass",
        "prompt_tail_mass",
        "local_response_mass",
        "long_response_mass",
        "marker_mass",
        "sink_mass",
        "entropy_norm",
        "head_prompt_std",
    ]
    contrasts: list[dict[str, Any]] = []
    tasks = sorted(group_summary)
    for group in ("early", "middle", "late"):
        for metric in metrics:
            values = {
                task: group_summary.get(task, {}).get(group, {}).get(metric, 0.0)
                for task in tasks
            }
            if not values:
                continue
            hi_task = max(values, key=values.get)
            lo_task = min(values, key=values.get)
            contrasts.append(
                {
                    "group": group,
                    "metric": metric,
                    "max_task": hi_task,
                    "max_value": values[hi_task],
                    "min_task": lo_task,
                    "min_value": values[lo_task],
                    "gap": values[hi_task] - values[lo_task],
                }
            )
    contrasts.sort(key=lambda item: abs(item["gap"]), reverse=True)
    return contrasts


def write_markdown_summary(path: Path, payload: Mapping[str, Any]) -> None:
    lines = [
        "# Attention Matrix Pattern Probe",
        "",
        "## Config",
        "",
        f"- base_model: `{payload['config']['base_model']}`",
        f"- data: `{payload['config']['trajectory_jsonl']}`",
        f"- samples_per_task: `{payload['config']['samples_per_task']}`",
        f"- max_seq_length: `{payload['config']['max_seq_length']}`",
        f"- response_tail_tokens: `{payload['config']['response_tail_tokens']}`",
        "",
        "## Layer Group Metrics",
        "",
    ]
    metric_names = [
        "prompt_mass",
        "prompt_tail_mass",
        "local_response_mass",
        "long_response_mass",
        "marker_mass",
        "sink_mass",
        "entropy_norm",
    ]
    for task, group_map in sorted(payload["layer_group_summary"].items()):
        lines.append(f"### {task}")
        lines.append("")
        lines.append("| group | " + " | ".join(metric_names) + " |")
        lines.append("| --- | " + " | ".join(["---:"] * len(metric_names)) + " |")
        for group in ("early", "middle", "late"):
            stats = group_map.get(group, {})
            values = [stats.get(metric, 0.0) for metric in metric_names]
            lines.append("| " + group + " | " + " | ".join(f"{value:.4f}" for value in values) + " |")
        lines.append("")
    lines.extend(["## Strongest Task Contrasts", ""])
    lines.append("| group | metric | high task | high | low task | low | gap |")
    lines.append("| --- | --- | --- | ---: | --- | ---: | ---: |")
    for item in payload["strongest_task_contrasts"][:16]:
        lines.append(
            f"| {item['group']} | {item['metric']} | {item['max_task']} | {item['max_value']:.4f} | "
            f"{item['min_task']} | {item['min_value']:.4f} | {item['gap']:.4f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
