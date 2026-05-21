#!/usr/bin/env python3
"""Probe signed task-vector utility with teacher-forced trajectory gradients.

This is a diagnostic script for attention-aware merging. Unlike the original
PAUH exposure score, it asks a signed first-order question:

    If we add expert delta D at this linear module, does the supervised
    trajectory loss locally go down or up?

Positive signed_effect means the delta helps the current task trajectory under
the base model's local gradient. Negative signed_effect means harm. The script
does not train or modify a model; it only writes probe summaries.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.attention_pauh.core import (  # noqa: E402
    default_owner_task,
    is_attention_param,
    linear_delta_probe,
    manifest_expert_names,
    normalize_task_name,
    parse_layer_index,
)


@dataclass(frozen=True)
class ProbeEntry:
    expert: str
    param_name: str
    layer: int
    storage_path: Path


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.mode_manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks = tuple(normalize_task_name(task) for task in split_csv(args.tasks))
    experts = tuple(split_csv(args.experts)) if args.experts else manifest_expert_names(manifest)
    layers = parse_layers(args.layers) if args.layers else None
    entries_by_param = index_probe_entries(
        manifest=manifest,
        manifest_dir=manifest_path.parent,
        experts=set(experts),
        layers=layers,
        scope=args.scope,
    )
    rows = load_trajectory_rows(
        [Path(path).expanduser() for path in args.trajectory_jsonl],
        tasks=tasks,
        samples_per_task=args.samples_per_task,
    )

    config = {
        "format": "signed_utility_probe_config_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode_manifest": str(manifest_path),
        "base_model": args.base_model or manifest["base_model"],
        "trajectory_jsonl": [str(Path(path).expanduser()) for path in args.trajectory_jsonl],
        "tasks": list(tasks),
        "experts": list(experts),
        "scope": args.scope,
        "layers": sorted(layers) if layers is not None else "all",
        "samples_per_task": args.samples_per_task,
        "span": args.span,
        "response_tail_tokens": args.response_tail_tokens,
        "max_seq_length": args.max_seq_length,
        "torch_dtype": args.torch_dtype,
        "device": args.device,
    }
    write_json(output_dir / "probe_config.json", config)

    if args.plan_only:
        payload = {
            "config": config,
            "row_counts": count_rows_by_task(rows),
            "num_target_params": len(entries_by_param),
            "num_target_entries": sum(len(items) for items in entries_by_param.values()),
        }
        write_json(output_dir / "probe_plan.json", payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    summary = run_signed_utility_probe(
        base_model=args.base_model or manifest["base_model"],
        entries_by_param=entries_by_param,
        rows=rows,
        device=args.device,
        dtype=args.torch_dtype,
        max_seq_length=args.max_seq_length,
        span=args.span,
        response_tail_tokens=args.response_tail_tokens,
        write_row_details=args.write_row_details,
        output_dir=output_dir,
    )
    payload = {"config": config, **summary}
    write_json(output_dir / "signed_utility_summary.json", payload)
    write_markdown_summary(output_dir / "signed_utility_summary.md", payload)
    print(
        json.dumps(
            {
                "summary": str(output_dir / "signed_utility_summary.json"),
                "row_counts": payload["row_counts"],
                "top_owner_utility": payload["top_owner_utility"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode-manifest", required=True)
    parser.add_argument("--trajectory-jsonl", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--tasks", default="tool,memory,code")
    parser.add_argument("--experts", default="tool,memory,code")
    parser.add_argument("--scope", choices=["attention", "all-linear"], default="attention")
    parser.add_argument("--layers", default="", help="Optional layer list/range, e.g. 0,1,18-27")
    parser.add_argument("--samples-per-task", type=int, default=4)
    parser.add_argument(
        "--span",
        choices=["response", "prompt", "all", "tool-call", "code-block", "reasoning", "signature"],
        default="response",
        help=(
            "Token span used for the first-order probe. `signature` maps tool->tool-call, "
            "code->code-block, and memory->response."
        ),
    )
    parser.add_argument("--response-tail-tokens", type=int, default=512)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--write-row-details", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def split_csv(raw: str) -> list[str]:
    return [item.strip() for item in str(raw or "").split(",") if item.strip()]


def parse_layers(raw: str) -> set[int]:
    layers: set[int] = set()
    for part in split_csv(raw):
        if "-" in part:
            lo, hi = part.split("-", 1)
            layers.update(range(int(lo), int(hi) + 1))
        else:
            layers.add(int(part))
    return layers


def index_probe_entries(
    *,
    manifest: Mapping[str, Any],
    manifest_dir: Path,
    experts: set[str],
    layers: set[int] | None,
    scope: str,
) -> dict[str, list[ProbeEntry]]:
    entries: dict[str, list[ProbeEntry]] = defaultdict(list)
    for raw in manifest.get("basis_entries", []):
        param_name = str(raw["param_name"])
        expert = str(raw["expert"])
        if experts and expert not in experts:
            continue
        if scope == "attention" and not is_attention_param(param_name):
            continue
        layer = parse_layer_index(param_name)
        if layers is not None and layer not in layers:
            continue
        entries[param_name].append(
            ProbeEntry(
                expert=expert,
                param_name=param_name,
                layer=layer,
                storage_path=manifest_dir / str(raw["storage_path"]),
            )
        )
    if not entries:
        raise ValueError("No OP-VEC entries matched the requested experts/scope/layers.")
    return dict(sorted(entries.items()))


def load_trajectory_rows(paths: list[Path], *, tasks: tuple[str, ...], samples_per_task: int) -> list[dict[str, Any]]:
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
        raise ValueError(f"Not enough trajectory rows for tasks: {missing}")
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


def run_signed_utility_probe(
    *,
    base_model: str,
    entries_by_param: Mapping[str, list[ProbeEntry]],
    rows: list[dict[str, Any]],
    device: str,
    dtype: str,
    max_seq_length: int,
    span: str,
    response_tail_tokens: int,
    write_row_details: bool,
    output_dir: Path,
) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype_map[dtype],
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(True)

    captured: dict[str, dict[str, torch.Tensor]] = {}
    hooks = register_linear_hooks(model, set(entries_by_param), captured)
    if not hooks:
        raise ValueError("No target modules from the mode manifest were found in the base model.")

    layer_task_stats: dict[str, dict[int, dict[str, dict[str, float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    )
    module_task_stats: dict[str, dict[str, dict[str, dict[str, float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    )
    conflict_stats: dict[str, dict[int, dict[str, dict[str, float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    )
    row_writer = None
    if write_row_details:
        row_writer = (output_dir / "signed_utility_rows.jsonl").open("w", encoding="utf-8")

    try:
        for index, row in enumerate(rows):
            model.zero_grad(set_to_none=True)
            captured.clear()
            encoded = encode_teacher_forced(
                tokenizer,
                row,
                max_seq_length=max_seq_length,
                span=span,
                response_tail_tokens=response_tail_tokens,
            )
            token_mask = encoded.pop("probe_token_mask")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            token_mask = token_mask.to(device)
            outputs = model(**encoded, use_cache=False)
            loss = outputs.loss
            loss.backward()
            row_payload = process_captured_row(
                entries_by_param=entries_by_param,
                captured=captured,
                token_mask=token_mask,
                task=str(row["task"]),
                row_id=str(row.get("sample_id") or row.get("prompt_id") or index),
                layer_task_stats=layer_task_stats,
                module_task_stats=module_task_stats,
                conflict_stats=conflict_stats,
                write_row_details=write_row_details,
            )
            if row_writer is not None:
                row_writer.write(json.dumps(row_payload, ensure_ascii=False, sort_keys=True) + "\n")
            model.zero_grad(set_to_none=True)
            del outputs, loss, encoded, token_mask
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    finally:
        for handle in hooks:
            handle.remove()
        if row_writer is not None:
            row_writer.close()
        model.cpu()
        del model

    layer_summary = finalize_layer_summary(layer_task_stats)
    module_summary = finalize_module_summary(module_task_stats)
    conflict_summary = finalize_conflict_summary(conflict_stats)
    return {
        "row_counts": count_rows_by_task(rows),
        "layer_summary": layer_summary,
        "module_summary": module_summary,
        "conflict_summary": conflict_summary,
        "top_owner_utility": top_owner_utility(layer_summary),
        "interpretation": {
            "signed_effect": "Positive values mean adding that expert delta locally reduces teacher-forced trajectory loss.",
            "protected_harm": "Mean max(0, -signed_effect) on non-owner tasks; high values indicate likely cross-task damage.",
            "conflict_cosine": "Cosine between induced residual updates D x for two experts on the same task/module.",
        },
    }


def register_linear_hooks(
    model: torch.nn.Module,
    target_param_names: set[str],
    captured: dict[str, dict[str, torch.Tensor]],
) -> list[Any]:
    hooks = []
    for module_name, module in model.named_modules():
        param_name = f"{module_name}.weight"
        if param_name not in target_param_names:
            continue

        def hook(_module: Any, inputs: tuple[Any, ...], output: Any, *, name: str = param_name) -> None:
            if not inputs:
                return
            hidden = inputs[0]
            projected = output[0] if isinstance(output, tuple) else output
            if not isinstance(hidden, torch.Tensor) or not isinstance(projected, torch.Tensor):
                return
            if hidden.ndim != 3 or projected.ndim != 3:
                return
            projected.retain_grad()
            captured[name] = {"input": hidden, "output": projected}

        hooks.append(module.register_forward_hook(hook))
    return hooks


def encode_teacher_forced(
    tokenizer: Any,
    row: Mapping[str, Any],
    *,
    max_seq_length: int,
    span: str,
    response_tail_tokens: int,
) -> dict[str, torch.Tensor]:
    prompt = render_prompt(tokenizer, row)
    response = str(row.get("response") or row.get("completion") or row.get("expert_response") or row.get("chosen_response"))
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    response_ids = tokenizer(response, add_special_tokens=False).input_ids
    if not prompt_ids or not response_ids:
        raise ValueError("Teacher-forced probe requires non-empty prompt and response.")

    ids = prompt_ids + response_ids
    prompt_len = len(prompt_ids)
    mask = build_probe_token_mask(
        tokenizer=tokenizer,
        response=response,
        response_ids=response_ids,
        seq_len=len(ids),
        prompt_len=prompt_len,
        span=span,
        response_tail_tokens=response_tail_tokens,
        task=normalize_task_name(str(row.get("task") or "")),
    )
    if len(ids) > max_seq_length:
        overflow = len(ids) - int(max_seq_length)
        ids = ids[overflow:]
        mask = mask[overflow:]
        prompt_len = max(0, prompt_len - overflow)
    labels = [-100] * prompt_len + ids[prompt_len:]
    labels = labels[: len(ids)]
    if not any(label != -100 for label in labels):
        labels[-1] = ids[-1]
    if mask.numel() != len(ids):
        raise ValueError(f"Probe mask length {mask.numel()} does not match sequence length {len(ids)}")
    if not bool(mask.any()):
        mask[-1] = True
    input_ids = torch.tensor([ids], dtype=torch.long)
    label_tensor = torch.tensor([labels], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": label_tensor,
        "probe_token_mask": mask,
    }


def render_prompt(tokenizer: Any, row: Mapping[str, Any]) -> str:
    if row.get("rendered_prompt"):
        return str(row["rendered_prompt"])
    if row.get("prompt"):
        return str(row["prompt"])
    if row.get("messages"):
        return tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True)
    raise ValueError("Trajectory row has no rendered_prompt, prompt, or messages.")


def build_probe_token_mask(
    *,
    tokenizer: Any | None = None,
    response: str = "",
    response_ids: list[int] | None = None,
    seq_len: int,
    prompt_len: int,
    span: str,
    response_tail_tokens: int,
    task: str = "",
) -> torch.Tensor:
    mask = torch.zeros(seq_len, dtype=torch.bool)
    resolved_span = resolve_probe_span(span, task)
    if resolved_span == "prompt":
        mask[: max(prompt_len, 1)] = True
    elif resolved_span == "all":
        mask[:] = True
    elif resolved_span == "response":
        start = max(prompt_len - 1, 0)
        mask[start:] = True
        mask = apply_tail_limit(mask, response_tail_tokens)
    else:
        if tokenizer is None or response_ids is None:
            raise ValueError(f"Span `{resolved_span}` requires tokenizer and response_ids.")
        response_mask = response_span_token_mask(
            tokenizer=tokenizer,
            response=response,
            response_ids=response_ids,
            span=resolved_span,
        )
        # Causal LM loss at output position i predicts token i+1, so response
        # token k is affected by the linear activations at prompt_len+k-1.
        for local_index in torch.nonzero(response_mask, as_tuple=False).view(-1).tolist():
            full_index = int(prompt_len) + int(local_index) - 1
            if 0 <= full_index < seq_len:
                mask[full_index] = True
        mask = apply_tail_limit(mask, response_tail_tokens)
    if not bool(mask.any()):
        mask[-1] = True
    return mask


def resolve_probe_span(span: str, task: str) -> str:
    span = str(span)
    if span != "signature":
        return span
    task = normalize_task_name(task)
    if task == "tool":
        return "tool-call"
    if task == "code":
        return "code-block"
    return "response"


def apply_tail_limit(mask: torch.Tensor, response_tail_tokens: int) -> torch.Tensor:
    if response_tail_tokens > 0 and int(mask.sum().item()) > response_tail_tokens:
        true_indices = torch.nonzero(mask, as_tuple=False).view(-1)
        keep = true_indices[-int(response_tail_tokens) :]
        mask = torch.zeros_like(mask)
        mask[keep] = True
    return mask


def response_span_token_mask(
    *,
    tokenizer: Any,
    response: str,
    response_ids: list[int],
    span: str,
) -> torch.Tensor:
    mask = torch.zeros(len(response_ids), dtype=torch.bool)
    intervals = response_char_intervals(response, span=span)
    if not intervals:
        intervals = [(0, len(response))]
    for start_char, end_char in intervals:
        start_tok = len(tokenizer(response[:start_char], add_special_tokens=False).input_ids)
        end_tok = len(tokenizer(response[:end_char], add_special_tokens=False).input_ids)
        start_tok = max(0, min(start_tok, len(response_ids)))
        end_tok = max(start_tok + 1, min(end_tok, len(response_ids)))
        if start_tok < len(response_ids):
            mask[start_tok:end_tok] = True
    if not bool(mask.any()) and len(response_ids) > 0:
        mask[:] = True
    return mask


def response_char_intervals(response: str, *, span: str) -> list[tuple[int, int]]:
    text = str(response or "")
    if not text:
        return []
    if span == "tool-call":
        return regex_intervals(text, r"<tool_call\b[^>]*>.*?</tool_call>")
    if span == "code-block":
        return regex_intervals(text, r"```[A-Za-z0-9_+.-]*\s*\n.*?```")
    if span == "reasoning":
        intervals = regex_intervals(text, r"<think\b[^>]*>.*?</think>")
        if intervals:
            return intervals
        code_intervals = response_char_intervals(text, span="code-block")
        tool_intervals = response_char_intervals(text, span="tool-call")
        cut_points = [start for start, _end in code_intervals + tool_intervals]
        if cut_points:
            end = min(cut_points)
            return [(0, end)] if end > 0 else []
        return [(0, len(text))]
    raise ValueError(f"Unsupported response-specific span: {span}")


def regex_intervals(text: str, pattern: str) -> list[tuple[int, int]]:
    return [(match.start(), match.end()) for match in re.finditer(pattern, text, flags=re.IGNORECASE | re.DOTALL)]


def process_captured_row(
    *,
    entries_by_param: Mapping[str, list[ProbeEntry]],
    captured: Mapping[str, Mapping[str, torch.Tensor]],
    token_mask: torch.Tensor,
    task: str,
    row_id: str,
    layer_task_stats: dict[str, dict[int, dict[str, dict[str, float]]]],
    module_task_stats: dict[str, dict[str, dict[str, dict[str, float]]]],
    conflict_stats: dict[str, dict[int, dict[str, dict[str, float]]]],
    write_row_details: bool,
) -> dict[str, Any]:
    row_details: list[dict[str, Any]] = []
    for param_name, entries in entries_by_param.items():
        tensors = captured.get(param_name)
        if tensors is None:
            continue
        output = tensors["output"]
        grad = output.grad
        if grad is None:
            continue
        inputs = tensors["input"].detach()
        output_grads = grad.detach()
        mean_updates: dict[str, torch.Tensor] = {}
        for entry in entries:
            delta = torch.load(entry.storage_path, map_location="cpu")
            expression, signed_effect, mean_update = linear_delta_probe(
                delta=delta,
                inputs=inputs,
                output_grads=output_grads,
                token_mask=token_mask,
            )
            update_stats(layer_task_stats[entry.expert][entry.layer][task], expression, signed_effect)
            update_stats(module_task_stats[entry.expert][param_name][task], expression, signed_effect)
            mean_updates[entry.expert] = mean_update
            if write_row_details:
                row_details.append(
                    {
                        "row_id": row_id,
                        "task": task,
                        "expert": entry.expert,
                        "layer": entry.layer,
                        "param_name": param_name,
                        "expression": expression,
                        "signed_effect": signed_effect,
                    }
                )
            del delta
        update_conflicts(
            conflict_stats=conflict_stats,
            param_name=param_name,
            task=task,
            layer=entries[0].layer,
            mean_updates=mean_updates,
        )
    return {"row_id": row_id, "task": task, "details": row_details}


def update_stats(stats: dict[str, float], expression: float, signed_effect: float) -> None:
    stats["count"] += 1.0
    stats["expression_sum"] += float(expression)
    stats["signed_sum"] += float(signed_effect)
    stats["positive_count"] += 1.0 if signed_effect > 0.0 else 0.0
    stats["harm_sum"] += max(0.0, -float(signed_effect))


def update_conflicts(
    *,
    conflict_stats: dict[str, dict[int, dict[str, dict[str, float]]]],
    param_name: str,
    task: str,
    layer: int,
    mean_updates: Mapping[str, torch.Tensor],
) -> None:
    experts = sorted(mean_updates)
    for i, left in enumerate(experts):
        for right in experts[i + 1 :]:
            a = mean_updates[left].flatten().float()
            b = mean_updates[right].flatten().float()
            denom = float(a.norm().item() * b.norm().item())
            cosine = 0.0 if denom <= 1.0e-12 else float(torch.dot(a, b).item() / denom)
            key = f"{left}|{right}"
            stats = conflict_stats[task][layer][key]
            stats["count"] += 1.0
            stats["cosine_sum"] += cosine
            stats["negative_count"] += 1.0 if cosine < 0.0 else 0.0
            stats["param_count"] += 1.0
            stats["last_param_hash"] = float(abs(hash(param_name)) % 1_000_000)


def finalize_layer_summary(
    layer_task_stats: Mapping[str, Mapping[int, Mapping[str, Mapping[str, float]]]]
) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for expert, layer_map in layer_task_stats.items():
        owner = default_owner_task(expert)
        expert_payload: dict[str, Any] = {}
        for layer, task_map in sorted(layer_map.items()):
            task_payload = {task: finalize_stats(stats) for task, stats in sorted(task_map.items())}
            owner_utility = task_payload.get(owner, {}).get("signed_effect_mean", 0.0)
            non_owner_harms = [
                stats["harm_mean"]
                for task, stats in task_payload.items()
                if normalize_task_name(task) != owner
            ]
            protected_harm = sum(non_owner_harms) / float(len(non_owner_harms)) if non_owner_harms else 0.0
            expert_payload[str(layer)] = {
                "owner_task": owner,
                "owner_utility": owner_utility,
                "protected_harm": protected_harm,
                "utility_minus_harm": owner_utility - protected_harm,
                "tasks": task_payload,
            }
        summary[expert] = expert_payload
    return summary


def finalize_module_summary(
    module_task_stats: Mapping[str, Mapping[str, Mapping[str, Mapping[str, float]]]]
) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for expert, module_map in module_task_stats.items():
        summary[expert] = {}
        for param_name, task_map in sorted(module_map.items()):
            summary[expert][param_name] = {task: finalize_stats(stats) for task, stats in sorted(task_map.items())}
    return summary


def finalize_stats(stats: Mapping[str, float]) -> dict[str, float]:
    count = max(float(stats.get("count", 0.0)), 1.0)
    return {
        "count": float(stats.get("count", 0.0)),
        "expression_mean": float(stats.get("expression_sum", 0.0)) / count,
        "signed_effect_mean": float(stats.get("signed_sum", 0.0)) / count,
        "harm_mean": float(stats.get("harm_sum", 0.0)) / count,
        "positive_fraction": float(stats.get("positive_count", 0.0)) / count,
    }


def finalize_conflict_summary(
    conflict_stats: Mapping[str, Mapping[int, Mapping[str, Mapping[str, float]]]]
) -> dict[str, dict[str, dict[str, float]]]:
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for task, layer_map in conflict_stats.items():
        summary[task] = {}
        for layer, pair_map in sorted(layer_map.items()):
            for pair, stats in sorted(pair_map.items()):
                count = max(float(stats.get("count", 0.0)), 1.0)
                summary[task][f"layer_{layer}:{pair}"] = {
                    "cosine_mean": float(stats.get("cosine_sum", 0.0)) / count,
                    "negative_fraction": float(stats.get("negative_count", 0.0)) / count,
                    "count": float(stats.get("count", 0.0)),
                }
    return summary


def top_owner_utility(layer_summary: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> dict[str, list[dict[str, float]]]:
    result: dict[str, list[dict[str, float]]] = {}
    for expert, layer_map in layer_summary.items():
        ranked = []
        for layer, item in layer_map.items():
            ranked.append(
                {
                    "layer": float(layer),
                    "owner_utility": float(item["owner_utility"]),
                    "protected_harm": float(item["protected_harm"]),
                    "utility_minus_harm": float(item["utility_minus_harm"]),
                }
            )
        ranked.sort(key=lambda item: item["utility_minus_harm"], reverse=True)
        result[expert] = ranked[:8]
    return result


def write_markdown_summary(path: Path, payload: Mapping[str, Any]) -> None:
    lines = [
        "# Signed Utility Probe Summary",
        "",
        "## Config",
        "",
        f"- base_model: `{payload['config']['base_model']}`",
        f"- mode_manifest: `{payload['config']['mode_manifest']}`",
        f"- trajectory_jsonl: `{payload['config']['trajectory_jsonl']}`",
        f"- tasks: `{payload['config']['tasks']}`",
        f"- experts: `{payload['config']['experts']}`",
        f"- scope/layers: `{payload['config']['scope']}` / `{payload['config']['layers']}`",
        f"- samples_per_task: `{payload['config']['samples_per_task']}`",
        f"- span: `{payload['config']['span']}`",
        "",
        "## Interpretation",
        "",
        "- `owner_utility > 0`: this expert delta locally lowers teacher-forced loss on its owner task.",
        "- `protected_harm > 0`: this expert delta locally raises loss on protected non-owner tasks.",
        "- `utility_minus_harm` is a diagnostic score, not a training objective.",
        "",
        "## Top Owner Utility Minus Harm Layers",
        "",
    ]
    for expert, rows in sorted(payload["top_owner_utility"].items()):
        lines.append(f"### {expert}")
        lines.append("")
        lines.append("| layer | owner utility | protected harm | utility - harm |")
        lines.append("| ---: | ---: | ---: | ---: |")
        for item in rows:
            lines.append(
                f"| {int(item['layer'])} | {item['owner_utility']:.6g} | {item['protected_harm']:.6g} | {item['utility_minus_harm']:.6g} |"
            )
        lines.append("")
    lines.append("## Strongest Conflict Cosines")
    lines.append("")
    lines.append("| task | layer/pair | cosine | negative frac |")
    lines.append("| --- | --- | ---: | ---: |")
    conflict_rows = []
    for task, layer_map in payload["conflict_summary"].items():
        for layer_pair, stats in layer_map.items():
            conflict_rows.append((task, layer_pair, stats))
    conflict_rows.sort(key=lambda item: item[2]["cosine_mean"])
    for task, layer_pair, stats in conflict_rows[:24]:
        lines.append(
            f"| {task} | {layer_pair} | {stats['cosine_mean']:.4f} | {stats['negative_fraction']:.3f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
