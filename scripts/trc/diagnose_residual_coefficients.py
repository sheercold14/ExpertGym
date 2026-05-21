#!/usr/bin/env python3
"""Diagnose trajectory residuals as coefficients over expert task-vector bases.

This script is intentionally read-only with respect to training.  It answers a
different question from the current positive-residual TRC loss:

    Given one successful trajectory, how much of each expert basis is needed to
    explain a chosen target residual on the same tokens/layers?

The target can be the row expert, a fixed coefficient mixture, or a gate
checkpoint.  The result is a local alpha* estimate that can be below the
current Code gate, providing a concrete "push down" signal for later losses.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from opvec.config import load_config, write_json
from opvec.data.io import read_jsonl, write_jsonl
from opvec.modeling.apply_gates import install_gated_linears_from_manifest
from opvec.modeling.devices import model_input_device, model_load_device_kwargs
from opvec.modeling.gate_parameters import make_torch_gate_manager
from scripts.trc.train_trc_layer_gates import (
    encode_prompt_response,
    parse_int_list,
    response_token_span,
    selected_hidden_states,
    temporary_direct_coefficients,
    with_constant_init,
)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    expert_names = tuple(str(name) for name in config.get("models", {}).get("experts", {}).keys())
    if not expert_names:
        raise ValueError("No experts found in config.models.experts")
    if args.gate_parameterization not in {"layer-band-coefficient", "layer_band_coefficient"}:
        raise ValueError("This diagnostic currently supports direct layer-band coefficients only.")

    rows = read_jsonl(args.calibration)
    rows = filter_rows(rows, task_allowlist=set(args.task or []), max_rows=int(args.max_rows), max_rows_per_task=int(args.max_rows_per_task))
    if not rows:
        raise ValueError("No calibration rows selected for diagnostic.")
    hidden_layers = parse_int_list(args.hidden_layers)
    if not hidden_layers:
        raise ValueError("--hidden-layers must not be empty")

    manifest = {
        "format": "trc_residual_coefficient_diagnostic_manifest_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": str(Path(args.config).expanduser().resolve()),
        "mode_manifest": str(Path(args.mode_manifest).expanduser().resolve()),
        "calibration": str(Path(args.calibration).expanduser().resolve()),
        "output_dir": str(output_dir),
        "args": vars(args),
        "num_rows": len(rows),
        "task_counts": dict(Counter(str(row.get("task")) for row in rows)),
        "expert_names": list(expert_names),
    }
    write_json(output_dir / "diagnostic_manifest.json", manifest)
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = resolve_torch_dtype(torch, args.torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(config["models"]["base"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        config["models"]["base"],
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        **model_load_device_kwargs(device_map=args.device_map, max_memory=args.max_memory),
    )
    if not args.device_map:
        model.to(args.device)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    device = model_input_device(model, torch, args.device)
    gate_manager = make_torch_gate_manager(
        torch,
        with_constant_init(config, init_value=0.0, expert_names=expert_names),
        parameterization=args.gate_parameterization,
    ).to(device)
    install_gated_linears_from_manifest(
        torch,
        model,
        mode_manifest_path=args.mode_manifest,
        gate_manager=gate_manager,
        max_modules=None,
        device=None if args.device_map else str(device),
    )

    target_coefficients = parse_coefficients(args.target_coefficients, expert_names=expert_names)
    alpha_prior = parse_coefficients(args.alpha_prior, expert_names=expert_names, default=0.0)
    target_gate_values = None
    if args.target_gate_checkpoint:
        target_gate_values = load_gate_checkpoint(Path(args.target_gate_checkpoint).expanduser())

    row_outputs: list[dict[str, Any]] = []
    start_time = time.time()
    for row_index, row in enumerate(rows):
        item_start = time.time()
        result = diagnose_row(
            torch=torch,
            model=model,
            tokenizer=tokenizer,
            gate_manager=gate_manager,
            row=row,
            row_index=row_index,
            expert_names=expert_names,
            hidden_layers=hidden_layers,
            device=device,
            max_seq_length=int(args.max_seq_length),
            max_response_tokens=int(args.max_response_tokens),
            topk_tokens=int(args.topk_tokens),
            response_span_mode=str(args.response_span_mode),
            target_source=str(args.target_source),
            target_coefficients=target_coefficients,
            target_gate_values=target_gate_values,
            alpha_prior=alpha_prior,
            ridge=float(args.ridge),
            alpha_max=float(args.alpha_max),
            target_normalize_eps=float(args.target_normalize_eps),
        )
        if result is None:
            continue
        result["elapsed_seconds"] = time.time() - item_start
        row_outputs.append(result)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize_outputs(row_outputs, expert_names=expert_names, elapsed=time.time() - start_time)
    write_jsonl(output_dir / "residual_coefficients.jsonl", row_outputs)
    write_json(output_dir / "summary.json", summary)
    (output_dir / "README.md").write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def diagnose_row(
    *,
    torch: Any,
    model: Any,
    tokenizer: Any,
    gate_manager: Any,
    row: dict[str, Any],
    row_index: int,
    expert_names: tuple[str, ...],
    hidden_layers: list[int],
    device: Any,
    max_seq_length: int,
    max_response_tokens: int,
    topk_tokens: int,
    response_span_mode: str,
    target_source: str,
    target_coefficients: dict[str, float],
    target_gate_values: dict[str, float] | None,
    alpha_prior: dict[str, float],
    ridge: float,
    alpha_max: float,
    target_normalize_eps: float,
) -> dict[str, Any] | None:
    response = str(row.get("response") or "")
    encoded = encode_prompt_response(
        tokenizer,
        prompt_text=str(row.get("rendered_prompt") or row.get("prompt") or ""),
        response_text=response,
        max_seq_length=max_seq_length,
        max_response_tokens=max_response_tokens,
    )
    if encoded is None:
        return None
    input_ids, attention_mask, prompt_len, response_len = encoded
    span_start, span_end, resolved_span_mode = response_token_span(
        tokenizer,
        task=str(row.get("task") or ""),
        response_text=response,
        response_len=response_len,
        max_response_tokens=max_response_tokens,
        mode=response_span_mode,
    )
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    positions = torch.arange(prompt_len + span_start, prompt_len + span_end, device=device)
    if positions.numel() <= 0:
        return None

    base_values = torch.zeros_like(gate_manager.raw_coefficients)
    with torch.no_grad():
        with temporary_direct_coefficients(torch, gate_manager, base_values):
            base_hidden = selected_hidden_states(model, input_ids=input_ids, attention_mask=attention_mask, hidden_layers=hidden_layers)

        basis_hidden: dict[str, dict[int, Any]] = {}
        for expert in expert_names:
            with temporary_direct_coefficients(torch, gate_manager, expert_basis_values(torch, gate_manager, expert)):
                basis_hidden[expert] = selected_hidden_states(
                    model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    hidden_layers=hidden_layers,
                )

        target_values = resolve_target_values(
            torch=torch,
            gate_manager=gate_manager,
            row=row,
            target_source=target_source,
            target_coefficients=target_coefficients,
            target_gate_values=target_gate_values,
        )
        with temporary_direct_coefficients(torch, gate_manager, target_values):
            target_hidden = selected_hidden_states(model, input_ids=input_ids, attention_mask=attention_mask, hidden_layers=hidden_layers)

    y, basis = flatten_target_and_basis(
        torch=torch,
        base_hidden=base_hidden,
        target_hidden=target_hidden,
        basis_hidden=basis_hidden,
        expert_names=expert_names,
        positions=positions,
        topk_tokens=topk_tokens,
    )
    if y.numel() == 0:
        return None
    alpha, fit = solve_alpha(
        torch=torch,
        y=y,
        basis=basis,
        expert_names=expert_names,
        alpha_prior=alpha_prior,
        ridge=ridge,
        alpha_max=alpha_max,
        eps=target_normalize_eps,
    )
    return {
        "format": "trc_residual_coefficient_row_v1",
        "row_index": row_index,
        "task": row.get("task"),
        "expert": row.get("expert"),
        "prompt_id": row.get("prompt_id"),
        "trajectory_id": row.get("trajectory_id"),
        "source_name": row.get("source_name"),
        "reward_train": row.get("reward_train"),
        "target_source": target_source,
        "target_coefficients": target_values_to_mean_dict(target_values, gate_manager),
        "alpha_unconstrained": alpha["unconstrained"],
        "alpha_clipped": alpha["clipped"],
        "fit": fit,
        "response_tokens": response_len,
        "span_mode": resolved_span_mode,
        "span_tokens": int(positions.numel()),
        "selected_vectors": int(y.numel()),
    }


def flatten_target_and_basis(
    *,
    torch: Any,
    base_hidden: dict[int, Any],
    target_hidden: dict[int, Any],
    basis_hidden: dict[str, dict[int, Any]],
    expert_names: tuple[str, ...],
    positions: Any,
    topk_tokens: int,
) -> tuple[Any, Any]:
    y_chunks = []
    basis_chunks = []
    for layer_index in sorted(base_hidden):
        layer_positions = positions.to(device=base_hidden[layer_index].device)
        base = base_hidden[layer_index][:, layer_positions, :].float()
        target = target_hidden[layer_index][:, layer_positions, :].float() - base
        target_vec = target.squeeze(0)
        if target_vec.numel() == 0:
            continue
        weights = target_vec.norm(dim=-1)
        if topk_tokens > 0 and weights.numel() > topk_tokens:
            _, top_indices = torch.topk(weights, k=int(topk_tokens), largest=True, sorted=False)
            target_vec = target_vec[top_indices]
        else:
            top_indices = None
        expert_residuals = []
        for expert in expert_names:
            residual = basis_hidden[expert][layer_index][:, layer_positions, :].float() - base
            residual_vec = residual.squeeze(0)
            if top_indices is not None:
                residual_vec = residual_vec[top_indices]
            expert_residuals.append(residual_vec.reshape(-1))
        y_chunks.append(target_vec.reshape(-1))
        basis_chunks.append(torch.stack(expert_residuals, dim=1))
    if not y_chunks:
        return torch.empty(0), torch.empty(0)
    return torch.cat(y_chunks, dim=0), torch.cat(basis_chunks, dim=0)


def solve_alpha(
    *,
    torch: Any,
    y: Any,
    basis: Any,
    expert_names: tuple[str, ...],
    alpha_prior: dict[str, float],
    ridge: float,
    alpha_max: float,
    eps: float,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    basis = basis.float()
    y = y.float()
    prior = torch.tensor([float(alpha_prior.get(expert, 0.0)) for expert in expert_names], device=basis.device, dtype=torch.float32)
    gram = basis.T @ basis
    rhs = basis.T @ y
    if ridge > 0.0:
        eye = torch.eye(len(expert_names), device=basis.device, dtype=torch.float32)
        gram = gram + float(ridge) * eye
        rhs = rhs + float(ridge) * prior
    try:
        alpha_vec = torch.linalg.solve(gram, rhs)
    except RuntimeError:
        alpha_vec = torch.linalg.lstsq(gram, rhs.unsqueeze(1)).solution.squeeze(1)
    clipped = alpha_vec.clamp(0.0, float(alpha_max))
    pred_unconstrained = basis @ alpha_vec
    pred_clipped = basis @ clipped
    target_norm = y.norm().clamp_min(float(eps))
    fit = {
        "target_norm": float(target_norm.detach().cpu().item()),
        "basis_norm_mean": float(basis.norm(dim=0).mean().detach().cpu().item()),
        "relative_error_unconstrained": float(((pred_unconstrained - y).norm() / target_norm).detach().cpu().item()),
        "relative_error_clipped": float(((pred_clipped - y).norm() / target_norm).detach().cpu().item()),
        "solution_l2": float(alpha_vec.norm().detach().cpu().item()),
    }
    return {
        "unconstrained": {expert: float(alpha_vec[index].detach().cpu().item()) for index, expert in enumerate(expert_names)},
        "clipped": {expert: float(clipped[index].detach().cpu().item()) for index, expert in enumerate(expert_names)},
    }, fit


def resolve_target_values(
    *,
    torch: Any,
    gate_manager: Any,
    row: dict[str, Any],
    target_source: str,
    target_coefficients: dict[str, float],
    target_gate_values: dict[str, float] | None,
) -> Any:
    source = str(target_source).strip().lower()
    if source == "row-expert":
        return expert_basis_values(torch, gate_manager, str(row.get("expert") or ""))
    if source == "coefficients":
        values = torch.zeros_like(gate_manager.raw_coefficients)
        for expert_index, expert in enumerate(gate_manager.expert_names):
            values[:, expert_index] = float(target_coefficients.get(str(expert), 0.0))
        return values
    if source == "gate-checkpoint":
        if not target_gate_values:
            raise ValueError("--target-source gate-checkpoint requires --target-gate-checkpoint")
        return values_from_gate_map(torch, gate_manager, target_gate_values)
    raise ValueError(f"Unsupported target source: {target_source!r}")


def expert_basis_values(torch: Any, gate_manager: Any, expert: str) -> Any:
    values = torch.zeros_like(gate_manager.raw_coefficients)
    try:
        expert_idx = list(gate_manager.expert_names).index(str(expert))
    except ValueError as exc:
        raise ValueError(f"Unknown expert {expert!r}; expected one of {list(gate_manager.expert_names)}") from exc
    values[:, expert_idx] = 1.0
    return values


def values_from_gate_map(torch: Any, gate_manager: Any, gate_map: dict[str, float]) -> Any:
    values = torch.zeros_like(gate_manager.raw_coefficients)
    band_names = list(gate_manager.band_names)
    expert_names = list(gate_manager.expert_names)
    for band_index, band in enumerate(band_names):
        for expert_index, expert in enumerate(expert_names):
            values[band_index, expert_index] = float(
                gate_map.get(f"{band}.{expert}", gate_map.get(str(expert), 0.0))
            )
    return values


def target_values_to_mean_dict(values: Any, gate_manager: Any) -> dict[str, float]:
    detached = values.detach().cpu()
    return {
        str(expert): float(detached[:, expert_index].mean().item())
        for expert_index, expert in enumerate(gate_manager.expert_names)
    }


def load_gate_checkpoint(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    gates = data.get("gates") or data.get("final_gates") or data
    return {str(key): float(value) for key, value in dict(gates).items()}


def parse_coefficients(raw: str, *, expert_names: tuple[str, ...], default: float | None = None) -> dict[str, float]:
    coeffs = {expert: float(default) for expert in expert_names} if default is not None else {}
    if not raw:
        return coeffs
    for chunk in str(raw).split(","):
        if not chunk.strip():
            continue
        if "=" not in chunk:
            raise ValueError(f"Coefficient chunk must be expert=value: {chunk!r}")
        expert, value = chunk.split("=", 1)
        expert = expert.strip()
        if expert not in expert_names:
            raise ValueError(f"Unknown expert in coefficients: {expert!r}; expected {expert_names}")
        coeffs[expert] = float(value)
    return coeffs


def filter_rows(rows: list[dict[str, Any]], *, task_allowlist: set[str], max_rows: int, max_rows_per_task: int) -> list[dict[str, Any]]:
    out = []
    counts: Counter[str] = Counter()
    for row in rows:
        task = str(row.get("task") or "")
        if task_allowlist and task not in task_allowlist:
            continue
        if max_rows_per_task > 0 and counts[task] >= max_rows_per_task:
            continue
        out.append(row)
        counts[task] += 1
        if max_rows > 0 and len(out) >= max_rows:
            break
    return out


def summarize_outputs(rows: list[dict[str, Any]], *, expert_names: tuple[str, ...], elapsed: float) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row.get("task"))].append(row)
    return {
        "format": "trc_residual_coefficient_diagnostic_summary_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rows": len(rows),
        "elapsed_seconds": elapsed,
        "task_counts": {task: len(items) for task, items in sorted(by_task.items())},
        "by_task": {
            task: summarize_group(items, expert_names=expert_names)
            for task, items in sorted(by_task.items())
        },
        "overall": summarize_group(rows, expert_names=expert_names),
    }


def summarize_group(rows: list[dict[str, Any]], *, expert_names: tuple[str, ...]) -> dict[str, Any]:
    if not rows:
        return {}
    out: dict[str, Any] = {
        "rows": len(rows),
        "relative_error_clipped_mean": safe_mean(row["fit"]["relative_error_clipped"] for row in rows),
    }
    for expert in expert_names:
        values = [float(row["alpha_clipped"][expert]) for row in rows]
        out[f"alpha_{expert}_mean"] = safe_mean(values)
        out[f"alpha_{expert}_median"] = safe_median(values)
        out[f"alpha_{expert}_lt_0.75_rate"] = safe_mean(1.0 if value < 0.75 else 0.0 for value in values)
        out[f"alpha_{expert}_lt_1.00_rate"] = safe_mean(1.0 if value < 1.0 else 0.0 for value in values)
    return out


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# TRC Residual Coefficient Diagnostic",
        "",
        "This diagnostic estimates local alpha* coefficients by fitting target hidden residuals with expert task-vector residual bases.",
        "",
        f"- Rows: `{summary['rows']}`",
        f"- Elapsed seconds: `{summary['elapsed_seconds']:.2f}`",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
    ]
    return "\n".join(lines)


def safe_mean(values: Any) -> float:
    vals = [float(value) for value in values]
    return sum(vals) / len(vals) if vals else math.nan


def safe_median(values: Any) -> float:
    vals = [float(value) for value in values]
    return statistics.median(vals) if vals else math.nan


def resolve_torch_dtype(torch: Any, name: str) -> Any:
    if str(name) == "auto":
        return torch.bfloat16
    return getattr(torch, str(name))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/gated_grpo_layer28.yaml")
    parser.add_argument("--mode-manifest", default="/tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json")
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gate-parameterization", default="layer-band-coefficient")
    parser.add_argument("--target-source", choices=["row-expert", "coefficients", "gate-checkpoint"], default="coefficients")
    parser.add_argument("--target-coefficients", default="tool=0.75,memory=0.75,code=0.75")
    parser.add_argument("--target-gate-checkpoint", default="")
    parser.add_argument("--alpha-prior", default="")
    parser.add_argument("--ridge", type=float, default=1.0e-4)
    parser.add_argument("--alpha-max", type=float, default=1.50)
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--max-rows-per-task", type=int, default=0)
    parser.add_argument("--hidden-layers", default="8,16,24,28")
    parser.add_argument("--max-seq-length", type=int, default=1536)
    parser.add_argument("--max-response-tokens", type=int, default=512)
    parser.add_argument("--topk-tokens", type=int, default=128)
    parser.add_argument("--response-span-mode", choices=["response", "auto", "tool-call", "code-block"], default="response")
    parser.add_argument("--target-normalize-eps", type=float, default=1.0e-6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--max-memory", action="append", default=[])
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
