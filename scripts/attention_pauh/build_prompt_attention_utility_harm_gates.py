#!/usr/bin/env python3
"""Build PromptAttention-UtilityHarm OP-VEC gate coefficients.

The script only uses prompt forward passes on the base model. It captures the
input activation diagonals of attention q/k/v/o projections, scores OP-VEC
expert-layer modes by owner utility vs protected-task harm, and writes a
parameter-level gate checkpoint consumable by scripts/eval/opvec_bake_checkpoint.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.attention_pauh.core import (  # noqa: E402
    aggregate_layer_energy_scores,
    gate_values_from_layer_coefficients,
    is_attention_param,
    layer_coefficients_from_scores,
    manifest_expert_names,
    manifest_layers,
    normalize_task_name,
    summarize_coefficients,
    task_layer_energy_from_entries,
)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.mode_manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    experts = tuple(args.experts.split(",")) if args.experts else manifest_expert_names(manifest)
    tasks = tuple(normalize_task_name(task) for task in args.tasks.split(","))
    alpha_by_expert = parse_alpha(args.alpha, experts=experts, default=args.default_alpha)
    prompt_rows = load_prompt_rows([Path(path).expanduser() for path in args.prompt_jsonl], tasks, args.samples_per_task)

    run_config = {
        "format": "prompt_attention_utility_harm_config_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode_manifest": str(manifest_path),
        "base_model": args.base_model or manifest["base_model"],
        "prompt_jsonl": [str(Path(path).expanduser()) for path in args.prompt_jsonl],
        "tasks": list(tasks),
        "experts": list(experts),
        "samples_per_task": args.samples_per_task,
        "prompt_tail_tokens": args.prompt_tail_tokens,
        "max_seq_length": args.max_seq_length,
        "beta": args.beta,
        "alpha_by_expert": alpha_by_expert,
        "min_coeff": args.min_coeff,
        "max_coeff": args.max_coeff,
        "scope": args.scope,
        "energy_normalization": args.energy_normalization,
    }
    write_json(output_dir / "pauh_config.json", run_config)

    if args.plan_only:
        summary = {"config": run_config, "prompt_counts": count_rows_by_task(prompt_rows)}
        write_json(output_dir / "pauh_plan.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    activation_diags = collect_attention_activation_diags(
        base_model=args.base_model or manifest["base_model"],
        manifest=manifest,
        rows=prompt_rows,
        tasks=tasks,
        device=args.device,
        dtype=args.torch_dtype,
        max_seq_length=args.max_seq_length,
        prompt_tail_tokens=args.prompt_tail_tokens,
    )
    save_activation_summary(output_dir / "activation_diag_summary.json", activation_diags)
    layer_task_energy = task_layer_energy_from_entries(
        manifest=manifest,
        activation_diags=activation_diags,
        mode_dir=str(manifest_path.parent),
        normalization=args.energy_normalization,
    )
    scores = aggregate_layer_energy_scores(
        layer_task_energy=layer_task_energy,
        experts=experts,
        layers=manifest_layers(manifest),
        tasks=tasks,
    )
    coefficients = layer_coefficients_from_scores(
        scores,
        alpha_by_expert=alpha_by_expert,
        beta=args.beta,
        min_coeff=args.min_coeff,
        max_coeff=args.max_coeff,
    )
    gates = gate_values_from_layer_coefficients(manifest, coefficients, scope=args.scope)
    payload = {
        "format": "prompt_attention_utility_harm_gates_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": run_config,
        "gates": gates,
        "coefficients": {expert: {str(layer): value for layer, value in sorted(values.items())} for expert, values in coefficients.items()},
        "coefficient_summary": summarize_coefficients(coefficients),
        "scores": serialize_scores(scores),
    }
    write_json(output_dir / "pauh_gates.json", payload)
    write_json(output_dir / "pauh_scores.json", payload["scores"])
    write_markdown_summary(output_dir / "pauh_summary.md", payload)
    print(json.dumps({"gate_checkpoint": str(output_dir / "pauh_gates.json"), "num_gates": len(gates)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode-manifest", required=True)
    parser.add_argument("--prompt-jsonl", action="append", required=True, help="JSON/JSONL prompt rows with task labels.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--experts", default="tool,memory,code")
    parser.add_argument("--tasks", default="tool,memory,code")
    parser.add_argument("--samples-per-task", type=int, default=32)
    parser.add_argument("--prompt-tail-tokens", type=int, default=256)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--beta", type=float, default=0.7)
    parser.add_argument("--default-alpha", type=float, default=0.75)
    parser.add_argument("--alpha", default="", help="Comma-separated overrides, e.g. tool=0.5,memory=0.75,code=0.75")
    parser.add_argument("--min-coeff", type=float, default=0.25)
    parser.add_argument("--max-coeff", type=float, default=1.25)
    parser.add_argument("--scope", choices=["layer-all", "attn-only"], default="layer-all")
    parser.add_argument("--energy-normalization", choices=["delta-norm", "none"], default="delta-norm")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def parse_alpha(raw: str, *, experts: tuple[str, ...], default: float) -> dict[str, float]:
    values = {expert: float(default) for expert in experts}
    for item in [part.strip() for part in str(raw or "").split(",") if part.strip()]:
        if "=" not in item:
            raise ValueError(f"Invalid --alpha item: {item}")
        key, value = item.split("=", 1)
        values[str(key).strip()] = float(value)
    return values


def load_prompt_rows(paths: list[Path], tasks: tuple[str, ...], samples_per_task: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = defaultdict(int)
    task_set = set(tasks)
    for path in paths:
        rows = read_json_rows(path)
        for row in rows:
            task = normalize_task_name(str(row.get("task") or row.get("ability") or row.get("data_source") or ""))
            if task not in task_set:
                continue
            if counts[task] >= samples_per_task:
                continue
            if not row_has_prompt(row):
                continue
            item = dict(row)
            item["task"] = task
            selected.append(item)
            counts[task] += 1
    missing = {task: samples_per_task - counts.get(task, 0) for task in tasks if counts.get(task, 0) < samples_per_task}
    if missing:
        raise ValueError(f"Not enough prompt rows for tasks: {missing}")
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
        raise ValueError(f"Unsupported JSON prompt file shape: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def row_has_prompt(row: Mapping[str, Any]) -> bool:
    return bool(row.get("rendered_prompt") or row.get("prompt") or row.get("messages"))


def count_rows_by_task(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row["task"])] += 1
    return dict(sorted(counts.items()))


def collect_attention_activation_diags(
    *,
    base_model: str,
    manifest: Mapping[str, Any],
    rows: list[dict[str, Any]],
    tasks: tuple[str, ...],
    device: str,
    dtype: str,
    max_seq_length: int,
    prompt_tail_tokens: int,
) -> dict[str, dict[str, torch.Tensor]]:
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

    attention_params = {
        str(entry["param_name"])
        for entry in manifest.get("basis_entries", [])
        if is_attention_param(str(entry["param_name"]))
    }
    sums: dict[str, dict[str, torch.Tensor]] = {task: {} for task in tasks}
    counts: dict[str, dict[str, int]] = {task: defaultdict(int) for task in tasks}
    current_task = {"name": ""}

    hooks = []
    for module_name, module in model.named_modules():
        param_name = f"{module_name}.weight"
        if param_name not in attention_params:
            continue

        def hook(_module: Any, inputs: tuple[Any, ...], _output: Any, *, name: str = param_name) -> None:
            hidden = inputs[0].detach().float()
            if hidden.ndim != 3:
                return
            tail = hidden[:, -int(prompt_tail_tokens) :, :]
            diag = tail.pow(2).mean(dim=(0, 1)).cpu()
            task = current_task["name"]
            if name not in sums[task]:
                sums[task][name] = torch.zeros_like(diag)
            sums[task][name] += diag
            counts[task][name] += 1

        hooks.append(module.register_forward_hook(hook))

    if not hooks:
        raise ValueError("No attention modules from the mode manifest were found in the base model.")

    try:
        with torch.no_grad():
            for row in rows:
                task = str(row["task"])
                current_task["name"] = task
                encoded = encode_prompt(tokenizer, row, max_seq_length=max_seq_length)
                encoded = {key: value.to(device) for key, value in encoded.items()}
                model(**encoded, use_cache=False)
    finally:
        for handle in hooks:
            handle.remove()
        model.cpu()
        del model

    averaged: dict[str, dict[str, torch.Tensor]] = {}
    for task, param_sums in sums.items():
        averaged[task] = {}
        for param_name, total in param_sums.items():
            averaged[task][param_name] = total / float(max(counts[task][param_name], 1))
    return averaged


def encode_prompt(tokenizer: Any, row: Mapping[str, Any], *, max_seq_length: int) -> dict[str, torch.Tensor]:
    if row.get("rendered_prompt"):
        prompt = str(row["rendered_prompt"])
    elif row.get("prompt"):
        prompt = str(row["prompt"])
    elif row.get("messages"):
        prompt = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True)
    else:
        raise ValueError("Prompt row has no rendered_prompt, prompt, or messages")
    ids = tokenizer(prompt, add_special_tokens=False).input_ids
    if not ids:
        raise ValueError("Prompt tokenization produced no tokens")
    if len(ids) > max_seq_length:
        ids = ids[-max_seq_length:]
    tensor = torch.tensor([ids], dtype=torch.long)
    return {"input_ids": tensor, "attention_mask": torch.ones_like(tensor)}


def serialize_scores(scores: Mapping[str, Mapping[int, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    payload: dict[str, dict[str, dict[str, float]]] = {}
    for expert, layer_scores in scores.items():
        payload[expert] = {}
        for layer, score in sorted(layer_scores.items()):
            payload[expert][str(layer)] = {
                "utility": float(score.utility),
                "harm": float(score.harm),
                "raw_score": float(score.raw_score),
                "score": float(score.score),
            }
    return payload


def save_activation_summary(path: Path, activation_diags: Mapping[str, Mapping[str, torch.Tensor]]) -> None:
    summary = {
        task: {
            param: {
                "dim": int(value.numel()),
                "mean": float(value.mean().item()),
                "max": float(value.max().item()),
            }
            for param, value in sorted(params.items())
        }
        for task, params in activation_diags.items()
    }
    write_json(path, summary)


def write_markdown_summary(path: Path, payload: Mapping[str, Any]) -> None:
    lines = [
        "# PromptAttention-UtilityHarm Summary",
        "",
        "## Coefficients",
        "",
        "| expert | layers | mean | min | max |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for expert, stats in sorted(payload["coefficient_summary"].items()):
        lines.append(
            f"| {expert} | {int(stats['count'])} | {stats['mean']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} |"
        )
    lines.extend(["", "## Top Utility-Harm Layers", ""])
    for expert, layer_scores in sorted(payload["scores"].items()):
        ranked = sorted(layer_scores.items(), key=lambda item: item[1]["score"], reverse=True)
        lines.append(f"### {expert}")
        lines.append("")
        lines.append("| layer | score | utility | harm |")
        lines.append("| ---: | ---: | ---: | ---: |")
        for layer, score in ranked[:8]:
            lines.append(f"| {layer} | {score['score']:.4f} | {score['utility']:.6g} | {score['harm']:.6g} |")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

