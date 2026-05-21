#!/usr/bin/env python3
"""Probe OP-VEC residual exposure across attention and MLP modules.

Attention matrices tell us where tokens route information. This script asks a
complementary question: on the same task spans, where would each expert
task-vector delta produce large residual changes?

It runs the base model once per sampled trajectory, captures linear-module input
activation diagonals for prompt/response spans, and scores every OP-VEC delta by
``E ||Delta W x||^2``. No training or model mutation is performed.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.attention_pauh.core import (  # noqa: E402
    default_owner_task,
    entry_activation_energy,
    manifest_expert_names,
    normalize_task_name,
    parse_layer_index,
)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.mode_manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks = tuple(normalize_task_name(task) for task in split_csv(args.tasks))
    experts = tuple(split_csv(args.experts)) if args.experts else manifest_expert_names(manifest)
    spans = tuple(split_csv(args.spans))
    rows = load_rows(
        [Path(path).expanduser() for path in args.trajectory_jsonl],
        tasks=tasks,
        samples_per_task=args.samples_per_task,
    )
    entries = index_entries(manifest, experts=set(experts), scope=args.scope)
    target_params = sorted({entry["param_name"] for entry in entries})

    config = {
        "format": "linear_module_exposure_probe_config_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode_manifest": str(manifest_path),
        "mode_dir": str(manifest_path.parent),
        "base_model": args.base_model or manifest["base_model"],
        "trajectory_jsonl": [str(Path(path).expanduser()) for path in args.trajectory_jsonl],
        "tasks": list(tasks),
        "experts": list(experts),
        "scope": args.scope,
        "spans": list(spans),
        "samples_per_task": args.samples_per_task,
        "max_seq_length": args.max_seq_length,
        "prompt_tail_tokens": args.prompt_tail_tokens,
        "response_tail_tokens": args.response_tail_tokens,
        "energy_normalization": args.energy_normalization,
        "torch_dtype": args.torch_dtype,
        "device": args.device,
    }
    write_json(output_dir / "linear_exposure_config.json", config)
    if args.plan_only:
        payload = {
            "config": config,
            "row_counts": count_rows_by_task(rows),
            "num_target_params": len(target_params),
            "num_entries": len(entries),
            "family_counts": count_families(target_params),
        }
        write_json(output_dir / "linear_exposure_plan.json", payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    activation_diags, activation_stats = collect_activation_diags(
        base_model=args.base_model or manifest["base_model"],
        rows=rows,
        target_params=target_params,
        spans=spans,
        device=args.device,
        dtype=args.torch_dtype,
        max_seq_length=args.max_seq_length,
        prompt_tail_tokens=args.prompt_tail_tokens,
        response_tail_tokens=args.response_tail_tokens,
    )
    energy_payload = score_delta_exposure(
        entries=entries,
        mode_dir=manifest_path.parent,
        activation_diags=activation_diags,
        normalization=args.energy_normalization,
    )
    payload = {
        "config": config,
        "row_counts": count_rows_by_task(rows),
        "activation_summary": finalize_activation_stats(activation_stats),
        **energy_payload,
    }
    write_json(output_dir / "linear_exposure_summary.json", payload)
    write_markdown_summary(output_dir / "linear_exposure_summary.md", payload)
    print(
        json.dumps(
            {
                "summary": str(output_dir / "linear_exposure_summary.json"),
                "row_counts": payload["row_counts"],
                "top_owner_exposure": payload["top_owner_exposure"],
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
    parser.add_argument("--scope", choices=["all-linear", "attention", "mlp"], default="all-linear")
    parser.add_argument("--spans", default="prompt,response")
    parser.add_argument("--samples-per-task", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--prompt-tail-tokens", type=int, default=512)
    parser.add_argument("--response-tail-tokens", type=int, default=512)
    parser.add_argument("--energy-normalization", choices=["delta-norm", "none"], default="delta-norm")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def split_csv(raw: str) -> list[str]:
    return [item.strip() for item in str(raw or "").split(",") if item.strip()]


def index_entries(manifest: Mapping[str, Any], *, experts: set[str], scope: str) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for raw in manifest.get("basis_entries", []):
        param_name = str(raw["param_name"])
        expert = str(raw["expert"])
        if experts and expert not in experts:
            continue
        family = module_family(param_name)
        if scope == "attention" and not family.startswith("attn_"):
            continue
        if scope == "mlp" and not family.startswith("mlp_"):
            continue
        entries.append(
            {
                "expert": expert,
                "param_name": param_name,
                "storage_path": str(raw["storage_path"]),
                "layer": str(parse_layer_index(param_name)),
                "family": family,
            }
        )
    if not entries:
        raise ValueError("No OP-VEC entries matched the requested scope/experts.")
    return entries


def module_family(param_name: str) -> str:
    if ".self_attn.q_proj.weight" in param_name:
        return "attn_q"
    if ".self_attn.k_proj.weight" in param_name:
        return "attn_k"
    if ".self_attn.v_proj.weight" in param_name:
        return "attn_v"
    if ".self_attn.o_proj.weight" in param_name:
        return "attn_o"
    if ".mlp.gate_proj.weight" in param_name:
        return "mlp_gate"
    if ".mlp.up_proj.weight" in param_name:
        return "mlp_up"
    if ".mlp.down_proj.weight" in param_name:
        return "mlp_down"
    return "other"


def layer_group(layer: int, num_groups: int = 3, max_layer: int = 27) -> str:
    width = (max_layer + 1) / float(num_groups)
    if layer < width:
        return "early"
    if layer < 2.0 * width:
        return "middle"
    return "late"


def count_families(param_names: list[str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for param_name in param_names:
        counts[module_family(param_name)] += 1
    return dict(sorted(counts.items()))


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


def collect_activation_diags(
    *,
    base_model: str,
    rows: list[dict[str, Any]],
    target_params: list[str],
    spans: tuple[str, ...],
    device: str,
    dtype: str,
    max_seq_length: int,
    prompt_tail_tokens: int,
    response_tail_tokens: int,
) -> tuple[dict[str, dict[str, dict[str, torch.Tensor]]], dict[str, dict[str, dict[str, dict[str, float]]]]]:
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

    target_set = set(target_params)
    sums: dict[str, dict[str, dict[str, torch.Tensor]]] = defaultdict(lambda: defaultdict(dict))
    counts: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    activation_stats: dict[str, dict[str, dict[str, dict[str, float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    )
    current = {"task": "", "masks": {}}
    hooks = []
    for module_name, module in model.named_modules():
        param_name = f"{module_name}.weight"
        if param_name not in target_set:
            continue

        def hook(_module: Any, inputs: tuple[Any, ...], _output: Any, *, name: str = param_name) -> None:
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                return
            hidden = inputs[0].detach().float()
            if hidden.ndim != 3:
                return
            task = str(current["task"])
            family = module_family(name)
            group = layer_group(parse_layer_index(name))
            for span in spans:
                mask = current["masks"].get(span)
                if mask is None:
                    continue
                mask = mask.to(hidden.device, dtype=torch.bool)
                if not bool(mask.any()):
                    continue
                selected = hidden[:, mask, :]
                diag = selected.pow(2).mean(dim=(0, 1)).cpu()
                if name not in sums[task][span]:
                    sums[task][span][name] = torch.zeros_like(diag)
                sums[task][span][name] += diag
                counts[task][span][name] += 1
                stats = activation_stats[task][span][f"{group}:{family}"]
                stats["count"] += 1.0
                stats["diag_mean_sum"] += float(diag.mean().item())
                stats["diag_max_sum"] += float(diag.max().item())

        hooks.append(module.register_forward_hook(hook))
    if not hooks:
        raise ValueError("No model modules matched OP-VEC target params.")

    try:
        with torch.no_grad():
            for row in rows:
                encoded = encode_text(
                    tokenizer,
                    row,
                    max_seq_length=max_seq_length,
                    prompt_tail_tokens=prompt_tail_tokens,
                    response_tail_tokens=response_tail_tokens,
                )
                current["task"] = str(row["task"])
                current["masks"] = encoded.pop("masks")
                current["masks"] = {key: value.to(device) for key, value in current["masks"].items()}
                model_inputs = {key: value.to(device) for key, value in encoded.items()}
                model(**model_inputs, use_cache=False)
                del encoded, model_inputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
    finally:
        for handle in hooks:
            handle.remove()
        model.cpu()
        del model

    averaged: dict[str, dict[str, dict[str, torch.Tensor]]] = defaultdict(lambda: defaultdict(dict))
    for task, span_map in sums.items():
        for span, param_map in span_map.items():
            for param_name, total in param_map.items():
                averaged[task][span][param_name] = total / float(max(counts[task][span][param_name], 1))
    return averaged, activation_stats


def encode_text(
    tokenizer: Any,
    row: Mapping[str, Any],
    *,
    max_seq_length: int,
    prompt_tail_tokens: int,
    response_tail_tokens: int,
) -> dict[str, Any]:
    prompt = render_prompt(tokenizer, row)
    response = str(row.get("response") or row.get("completion") or row.get("expert_response") or row.get("chosen_response"))
    full_text = prompt + response
    encoded = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(encoded.input_ids)
    offsets = list(encoded.offset_mapping)
    prompt_char_len = len(prompt)
    if len(ids) > max_seq_length:
        overflow = len(ids) - int(max_seq_length)
        ids = ids[overflow:]
        offsets = offsets[overflow:]
    masks = build_span_masks(
        offsets=offsets,
        prompt_char_len=prompt_char_len,
        prompt_tail_tokens=prompt_tail_tokens,
        response_tail_tokens=response_tail_tokens,
    )
    input_ids = torch.tensor([ids], dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
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
    offsets: list[tuple[int, int]],
    prompt_char_len: int,
    prompt_tail_tokens: int,
    response_tail_tokens: int,
) -> dict[str, torch.Tensor]:
    prompt = torch.tensor([start < prompt_char_len for start, _end in offsets], dtype=torch.bool)
    response = ~prompt
    if not bool(prompt.any()):
        prompt[0] = True
    if not bool(response.any()):
        response[-1] = True
    prompt_tail = keep_last_true(prompt, prompt_tail_tokens)
    response_tail = keep_last_true(response, response_tail_tokens)
    all_tokens = torch.ones_like(prompt)
    return {
        "prompt": prompt,
        "prompt_tail": prompt_tail,
        "response": response,
        "response_tail": response_tail,
        "all": all_tokens,
    }


def keep_last_true(mask: torch.Tensor, limit: int) -> torch.Tensor:
    limited = mask.clone()
    indices = torch.nonzero(limited, as_tuple=False).view(-1)
    if limit > 0 and indices.numel() > limit:
        keep = indices[-int(limit) :]
        limited[:] = False
        limited[keep] = True
    return limited


def finalize_activation_stats(
    activation_stats: Mapping[str, Mapping[str, Mapping[str, Mapping[str, float]]]]
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    output: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for task, span_map in activation_stats.items():
        output[task] = {}
        for span, bucket_map in span_map.items():
            output[task][span] = {}
            for bucket, stats in sorted(bucket_map.items()):
                count = max(float(stats.get("count", 0.0)), 1.0)
                output[task][span][bucket] = {
                    "count": float(stats.get("count", 0.0)),
                    "diag_mean": float(stats.get("diag_mean_sum", 0.0)) / count,
                    "diag_max": float(stats.get("diag_max_sum", 0.0)) / count,
                }
    return output


def score_delta_exposure(
    *,
    entries: list[dict[str, str]],
    mode_dir: Path,
    activation_diags: Mapping[str, Mapping[str, Mapping[str, torch.Tensor]]],
    normalization: str,
) -> dict[str, Any]:
    stats: dict[str, dict[str, dict[str, dict[str, dict[str, float]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float))))
    )
    layer_stats: dict[str, dict[str, dict[str, dict[int, dict[str, float]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float))))
    )
    for entry in entries:
        param_name = entry["param_name"]
        expert = entry["expert"]
        family = entry["family"]
        layer = int(entry["layer"])
        group = layer_group(layer)
        delta = torch.load(mode_dir / entry["storage_path"], map_location="cpu")
        for task, span_map in activation_diags.items():
            for span, param_map in span_map.items():
                diag = param_map.get(param_name)
                if diag is None:
                    continue
                energy = entry_activation_energy(delta, diag, normalization=normalization)
                update_energy_stats(stats[expert][task][span][f"{group}:{family}"], energy)
                update_energy_stats(stats[expert][task][span][f"{group}:all"], energy)
                update_energy_stats(stats[expert][task][span][f"all:{family}"], energy)
                update_energy_stats(stats[expert][task][span]["all:all"], energy)
                update_energy_stats(layer_stats[expert][task][span][layer], energy)
        del delta
    energy_summary = finalize_energy_stats(stats)
    layer_summary = finalize_layer_energy_stats(layer_stats)
    return {
        "energy_summary": energy_summary,
        "layer_energy_summary": layer_summary,
        "owner_protected_summary": owner_protected_summary(energy_summary),
        "top_owner_exposure": top_owner_exposure(energy_summary),
    }


def update_energy_stats(stats: dict[str, float], energy: float) -> None:
    stats["count"] += 1.0
    stats["sum"] += float(energy)
    stats["max"] = max(float(stats.get("max", 0.0)), float(energy))


def finalize_energy_stats(
    raw: Mapping[str, Mapping[str, Mapping[str, Mapping[str, Mapping[str, float]]]]]
) -> dict[str, dict[str, dict[str, dict[str, dict[str, float]]]]]:
    output: dict[str, dict[str, dict[str, dict[str, dict[str, float]]]]] = {}
    for expert, task_map in raw.items():
        output[expert] = {}
        for task, span_map in task_map.items():
            output[expert][task] = {}
            for span, bucket_map in span_map.items():
                output[expert][task][span] = {}
                for bucket, stats in sorted(bucket_map.items()):
                    count = max(float(stats.get("count", 0.0)), 1.0)
                    output[expert][task][span][bucket] = {
                        "count": float(stats.get("count", 0.0)),
                        "mean": float(stats.get("sum", 0.0)) / count,
                        "sum": float(stats.get("sum", 0.0)),
                        "max": float(stats.get("max", 0.0)),
                    }
    return output


def finalize_layer_energy_stats(
    raw: Mapping[str, Mapping[str, Mapping[str, Mapping[int, Mapping[str, float]]]]]
) -> dict[str, dict[str, dict[str, dict[str, dict[str, float]]]]]:
    output: dict[str, dict[str, dict[str, dict[str, dict[str, float]]]]] = {}
    for expert, task_map in raw.items():
        output[expert] = {}
        for task, span_map in task_map.items():
            output[expert][task] = {}
            for span, layer_map in span_map.items():
                output[expert][task][span] = {}
                for layer, stats in sorted(layer_map.items()):
                    count = max(float(stats.get("count", 0.0)), 1.0)
                    output[expert][task][span][str(layer)] = {
                        "count": float(stats.get("count", 0.0)),
                        "mean": float(stats.get("sum", 0.0)) / count,
                        "sum": float(stats.get("sum", 0.0)),
                        "max": float(stats.get("max", 0.0)),
                    }
    return output


def owner_protected_summary(
    energy_summary: Mapping[str, Mapping[str, Mapping[str, Mapping[str, Mapping[str, float]]]]]
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    output: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    tasks = sorted({task for task_map in energy_summary.values() for task in task_map})
    for expert, task_map in energy_summary.items():
        owner = default_owner_task(expert)
        output[expert] = {}
        for span in sorted({span for span_map in task_map.values() for span in span_map}):
            output[expert][span] = {}
            buckets = sorted({bucket for span_map in task_map.values() for bucket in span_map.get(span, {})})
            for bucket in buckets:
                owner_value = task_map.get(owner, {}).get(span, {}).get(bucket, {}).get("mean", 0.0)
                protected_values = [
                    task_map.get(task, {}).get(span, {}).get(bucket, {}).get("mean", 0.0)
                    for task in tasks
                    if task != owner
                ]
                protected = sum(protected_values) / float(len(protected_values)) if protected_values else 0.0
                output[expert][span][bucket] = {
                    "owner_task": owner,
                    "owner_mean": owner_value,
                    "protected_mean": protected,
                    "owner_minus_protected": owner_value - protected,
                    "owner_over_protected": owner_value / max(protected, 1.0e-12),
                }
    return output


def top_owner_exposure(
    energy_summary: Mapping[str, Mapping[str, Mapping[str, Mapping[str, Mapping[str, float]]]]]
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for expert, task_map in energy_summary.items():
        owner = default_owner_task(expert)
        rows = []
        for span, bucket_map in task_map.get(owner, {}).items():
            for bucket, stats in bucket_map.items():
                if bucket.endswith(":all") or bucket.startswith("all:"):
                    rows.append({"span": span, "bucket": bucket, "mean": stats["mean"], "sum": stats["sum"]})
        rows.sort(key=lambda item: item["mean"], reverse=True)
        output[expert] = rows[:12]
    return output


def write_markdown_summary(path: Path, payload: Mapping[str, Any]) -> None:
    lines = [
        "# Linear Module Exposure Probe",
        "",
        "## Config",
        "",
        f"- base_model: `{payload['config']['base_model']}`",
        f"- mode_manifest: `{payload['config']['mode_manifest']}`",
        f"- data: `{payload['config']['trajectory_jsonl']}`",
        f"- samples_per_task: `{payload['config']['samples_per_task']}`",
        f"- scope/spans: `{payload['config']['scope']}` / `{payload['config']['spans']}`",
        f"- energy_normalization: `{payload['config']['energy_normalization']}`",
        "",
        "## Owner Exposure Top Buckets",
        "",
    ]
    for expert, rows in sorted(payload["top_owner_exposure"].items()):
        lines.append(f"### {expert}")
        lines.append("")
        lines.append("| span | bucket | mean | sum |")
        lines.append("| --- | --- | ---: | ---: |")
        for item in rows:
            lines.append(f"| {item['span']} | {item['bucket']} | {item['mean']:.6g} | {item['sum']:.6g} |")
        lines.append("")
    lines.extend(["## Owner vs Protected: all:all", ""])
    lines.append("| expert | span | owner | protected | owner-protected | ratio |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for expert, span_map in sorted(payload["owner_protected_summary"].items()):
        for span, bucket_map in sorted(span_map.items()):
            stats = bucket_map.get("all:all")
            if not stats:
                continue
            lines.append(
                f"| {expert} | {span} | {stats['owner_mean']:.6g} | {stats['protected_mean']:.6g} | "
                f"{stats['owner_minus_protected']:.6g} | {stats['owner_over_protected']:.4f} |"
            )
    lines.extend(["", "## Task Activation Diag Mean by Span / Family", ""])
    lines.append("| task | span | bucket | diag mean | diag max |")
    lines.append("| --- | --- | --- | ---: | ---: |")
    for task, span_map in sorted(payload["activation_summary"].items()):
        for span, bucket_map in sorted(span_map.items()):
            for bucket, stats in sorted(bucket_map.items()):
                if bucket.endswith(":all"):
                    continue
                lines.append(
                    f"| {task} | {span} | {bucket} | {stats['diag_mean']:.6g} | {stats['diag_max']:.6g} |"
                )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
