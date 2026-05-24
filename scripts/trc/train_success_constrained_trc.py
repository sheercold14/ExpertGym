#!/usr/bin/env python3
"""Train OP-VEC layer gates with init-anchored success hidden-residual constraints.

This is a conservative TRC variant: successful trajectories are witnesses for
hidden residual directions, not token-level targets to imitate.  The optimizer
starts from an anchor gate, preserves the anchor's projection onto every
success direction, and only pushes rows whose anchor projection is deficient.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from opvec.config import load_config, write_json
from opvec.data.io import read_jsonl, write_jsonl
from opvec.modeling.apply_gates import install_gated_linears_from_manifest
from opvec.modeling.devices import model_input_device, model_load_device_kwargs
from opvec.modeling.gate_parameters import make_torch_gate_manager
from scripts.trc.train_trc_layer_gates import (
    append_jsonl_row,
    encode_prompt_response,
    gate_means,
    hidden_match_loss,
    load_gate_checkpoint,
    make_optimizer,
    mask_gate_gradients_for_trainable_experts,
    parse_int_list,
    parse_optional_name_set,
    parse_task_int_list_overrides,
    parse_task_value_overrides,
    prompt_position_tensor,
    response_token_span,
    resolve_torch_dtype,
    selected_hidden_states,
    task_balanced_row_scales,
    task_counts,
    temporary_direct_coefficients,
    tensor_item,
    validate_calibration_rows,
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
        raise ValueError("Success-constrained TRC currently supports only layer-band-coefficient gates")

    calibration_rows = read_jsonl(args.calibration)
    train_task_allowlist = parse_optional_name_set(args.train_tasks, label="train task")
    if train_task_allowlist:
        calibration_rows = [row for row in calibration_rows if str(row.get("task") or "") in train_task_allowlist]
        if not calibration_rows:
            raise ValueError(f"--train-tasks kept no rows; requested {sorted(train_task_allowlist)}")
    if args.success_only:
        calibration_rows = [row for row in calibration_rows if bool(row.get("success", True))]
    if args.max_rows_per_task:
        calibration_rows = limit_rows_per_task(calibration_rows, per_task=int(args.max_rows_per_task))
    elif args.max_rows:
        calibration_rows = calibration_rows[: int(args.max_rows)]
    validate_calibration_rows(calibration_rows, expert_names=expert_names)

    hidden_layers = parse_int_list(args.hidden_layers)
    if not hidden_layers:
        raise ValueError("--hidden-layers must contain at least one layer index")
    task_hidden_layers = parse_task_int_list_overrides(args.task_hidden_layers)
    task_response_span_modes = parse_task_value_overrides(args.task_response_span_mode, str)
    task_topk_tokens = parse_task_value_overrides(args.task_topk_tokens, int)
    task_loss_multiplier = parse_task_value_overrides(args.task_loss_multiplier, float)
    trainable_experts = parse_optional_name_set(args.trainable_experts, label="trainable expert")
    unknown_trainable_experts = sorted(trainable_experts.difference(expert_names))
    if unknown_trainable_experts:
        raise ValueError(f"Unknown --trainable-experts entries {unknown_trainable_experts}; expected {list(expert_names)}")

    anchor_gate_values = load_gate_checkpoint(args.anchor_gate_checkpoint) if args.anchor_gate_checkpoint else {}
    run_config = with_anchor_init(
        config,
        init_value=float(args.init_value),
        expert_names=expert_names,
        anchor_gate_values=anchor_gate_values,
    )
    run_manifest = {
        "format": "success_constrained_trc_run_manifest_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": str(Path(args.config).expanduser().resolve()),
        "mode_manifest": str(Path(args.mode_manifest).expanduser().resolve()),
        "calibration": str(Path(args.calibration).expanduser().resolve()),
        "output_dir": str(output_dir),
        "args": vars(args),
        "num_rows": len(calibration_rows),
        "task_counts": task_counts(calibration_rows),
        "train_tasks": sorted(train_task_allowlist),
        "trainable_experts": sorted(trainable_experts),
        "hidden_layers": hidden_layers,
        "task_hidden_layers": task_hidden_layers,
        "task_response_span_modes": task_response_span_modes,
        "task_topk_tokens": task_topk_tokens,
        "task_loss_multiplier": task_loss_multiplier,
        "anchor_gate_checkpoint": str(args.anchor_gate_checkpoint or ""),
        "expert_names": list(expert_names),
        "status": "initialized",
    }
    write_json(output_dir / "success_trc_run_manifest.json", run_manifest)
    if args.dry_run:
        run_manifest["status"] = "dry_run_ok"
        write_json(output_dir / "success_trc_run_manifest.json", run_manifest)
        print(json.dumps(run_manifest, ensure_ascii=False, indent=2, sort_keys=True))
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
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
    device = model_input_device(model, torch, args.device)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if args.gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        else:
            raise ValueError("--gradient-checkpointing requested but model does not support gradient_checkpointing_enable()")
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    gate_manager = make_torch_gate_manager(
        torch,
        run_config,
        parameterization=args.gate_parameterization,
    ).to(device)
    installed = install_gated_linears_from_manifest(
        torch,
        model,
        mode_manifest_path=args.mode_manifest,
        gate_manager=gate_manager,
        max_modules=None,
        device=None,
    )
    anchor_values = gate_manager.initial_coefficients.detach().clone()
    optimizer = make_optimizer(torch, args.optimizer, gate_manager.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    metrics_path = output_dir / "success_trc_metrics.jsonl"
    progress_path = output_dir / "success_trc_progress.jsonl"
    metrics_rows: list[dict[str, Any]] = []
    train_rows = list(calibration_rows)
    row_scales = task_balanced_row_scales(train_rows) if args.task_balanced_loss else {}
    start_time = time.time()

    for epoch in range(1, int(args.epochs) + 1):
        if args.shuffle:
            random.Random(int(args.seed) + epoch).shuffle(train_rows)
        epoch_start = time.time()
        epoch_metrics: list[dict[str, Any]] = []
        optimizer.zero_grad(set_to_none=True)
        pending_backward = 0
        for row_index, row in enumerate(train_rows, start=1):
            row_start = time.time()
            task = str(row.get("task") or "")
            losses = compute_success_constrained_row_loss(
                torch=torch,
                model=model,
                tokenizer=tokenizer,
                gate_manager=gate_manager,
                anchor_values=anchor_values,
                row=row,
                hidden_layers=task_hidden_layers.get(task, hidden_layers),
                device=device,
                max_seq_length=int(args.max_seq_length),
                max_response_tokens=int(args.max_response_tokens),
                topk_tokens=int(task_topk_tokens.get(task, int(args.topk_tokens))),
                prompt_drift_tokens=int(args.prompt_drift_tokens),
                beta_base=float(args.beta_base),
                projection_preserve_margin=float(args.projection_preserve_margin),
                projection_preserve_weight=float(args.projection_preserve_weight),
                deficit_projection_floor=float(args.deficit_projection_floor),
                max_projection_improve=float(args.max_projection_improve),
                deficit_projection_weight=float(args.deficit_projection_weight),
                deficit_directional_weight=float(args.deficit_directional_weight),
                gate_anchor_weight=float(args.gate_anchor_weight),
                response_span_mode=str(task_response_span_modes.get(task, str(args.response_span_mode))),
                target_normalize_eps=float(args.target_normalize_eps),
            )
            if losses is None:
                continue
            loss_scale = float(row_scales.get(task, 1.0)) * float(task_loss_multiplier.get(task, 1.0))
            scaled_loss = losses["total_loss"] * loss_scale / float(max(1, int(args.accumulation_steps)))
            scaled_loss.backward()
            pending_backward += 1
            metric = {
                "event": "row",
                "epoch": epoch,
                "row_index": row_index,
                "task": row.get("task"),
                "expert": row.get("expert"),
                "prompt_id": row.get("prompt_id"),
                "trajectory_id": row.get("trajectory_id"),
                "total_loss": tensor_item(losses["total_loss"]),
                "projection_preserve_loss": tensor_item(losses["projection_preserve_loss"]),
                "deficit_projection_loss": tensor_item(losses["deficit_projection_loss"]),
                "deficit_directional_loss": tensor_item(losses["deficit_directional_loss"]),
                "base_drift_loss": tensor_item(losses["base_drift_loss"]),
                "gate_anchor_loss": tensor_item(losses["gate_anchor_loss"]),
                "mean_anchor_projection": float(losses["mean_anchor_projection"]),
                "mean_current_projection": float(losses["mean_current_projection"]),
                "mean_deficit": float(losses["mean_deficit"]),
                "preserve_active_rate": float(losses["preserve_active_rate"]),
                "deficit_active_rate": float(losses["deficit_active_rate"]),
                "response_tokens": int(losses["response_tokens"]),
                "prompt_tokens": int(losses["prompt_tokens"]),
                "span_mode": str(losses["span_mode"]),
                "span_start_token": int(losses["span_start_token"]),
                "span_end_token": int(losses["span_end_token"]),
                "span_tokens": int(losses["span_tokens"]),
                "hidden_layers": losses["hidden_layers"],
                "loss_scale": loss_scale,
                "elapsed_seconds": time.time() - row_start,
            }
            epoch_metrics.append(metric)
            metrics_rows.append(metric)
            if int(args.progress_every_rows) > 0 and row_index % int(args.progress_every_rows) == 0:
                append_jsonl_row(
                    progress_path,
                    {
                        "event": "progress",
                        "epoch": epoch,
                        "row_index": row_index,
                        "rows_seen": len(epoch_metrics),
                        "task": row.get("task"),
                        "prompt_id": row.get("prompt_id"),
                        "total_loss": metric["total_loss"],
                        "mean_anchor_projection": metric["mean_anchor_projection"],
                        "mean_current_projection": metric["mean_current_projection"],
                        "mean_deficit": metric["mean_deficit"],
                        "elapsed_seconds": time.time() - epoch_start,
                    },
                )
            if pending_backward >= int(args.accumulation_steps):
                step_optimizer_with_anchor_cap(
                    torch,
                    gate_manager,
                    optimizer,
                    anchor_values=anchor_values,
                    max_delta=float(args.max_delta_from_anchor),
                    grad_clip_norm=float(args.grad_clip_norm),
                    trainable_experts=trainable_experts,
                )
                optimizer.zero_grad(set_to_none=True)
                pending_backward = 0
        if pending_backward:
            step_optimizer_with_anchor_cap(
                torch,
                gate_manager,
                optimizer,
                anchor_values=anchor_values,
                max_delta=float(args.max_delta_from_anchor),
                grad_clip_norm=float(args.grad_clip_norm),
                trainable_experts=trainable_experts,
            )
            optimizer.zero_grad(set_to_none=True)
        summary = summarize_epoch(epoch, epoch_metrics, gate_manager, anchor_values=anchor_values, elapsed=time.time() - epoch_start)
        metrics_rows.append(summary)
        write_json(output_dir / f"epoch_{epoch:03d}.gates.json", {"gates": gate_manager.gate_values(), "epoch_summary": summary})
        write_jsonl(metrics_path, metrics_rows)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))

    final_summary = {
        "format": "success_constrained_trc_summary_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - start_time,
        "installed_modules": len(installed),
        "final_gates": gate_manager.gate_values(),
        "gate_delta_summary": gate_delta_summary(gate_manager, anchor_values),
        "metrics": str(metrics_path),
    }
    write_json(output_dir / "success_trc_gates.json", {"gates": gate_manager.gate_values(), "summary": final_summary})
    write_json(output_dir / "success_trc_summary.json", final_summary)
    run_manifest["status"] = "completed"
    run_manifest["final_gate_checkpoint"] = str(output_dir / "success_trc_gates.json")
    write_json(output_dir / "success_trc_run_manifest.json", run_manifest)


def compute_success_constrained_row_loss(
    *,
    torch: Any,
    model: Any,
    tokenizer: Any,
    gate_manager: Any,
    anchor_values: Any,
    row: dict[str, Any],
    hidden_layers: list[int],
    device: Any,
    max_seq_length: int,
    max_response_tokens: int,
    topk_tokens: int,
    prompt_drift_tokens: int,
    beta_base: float,
    projection_preserve_margin: float,
    projection_preserve_weight: float,
    deficit_projection_floor: float,
    max_projection_improve: float,
    deficit_projection_weight: float,
    deficit_directional_weight: float,
    gate_anchor_weight: float,
    response_span_mode: str,
    target_normalize_eps: float,
) -> dict[str, Any] | None:
    encoded = encode_prompt_response(
        tokenizer,
        prompt_text=str(row.get("rendered_prompt") or row.get("prompt") or ""),
        response_text=str(row.get("response") or ""),
        max_seq_length=max_seq_length,
        max_response_tokens=max_response_tokens,
    )
    if encoded is None:
        return None
    input_ids, attention_mask, prompt_len, response_len = encoded
    span_start, span_end, resolved_span_mode = response_token_span(
        tokenizer,
        task=str(row.get("task") or ""),
        response_text=str(row.get("response") or ""),
        response_len=response_len,
        max_response_tokens=max_response_tokens,
        mode=response_span_mode,
    )
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    base_values = torch.zeros_like(gate_manager.raw_coefficients)
    expert_values = torch.zeros_like(gate_manager.raw_coefficients)
    expert_idx = list(gate_manager.expert_names).index(str(row["expert"]))
    expert_values[:, expert_idx] = 1.0

    with torch.no_grad():
        with temporary_direct_coefficients(torch, gate_manager, base_values):
            base_hidden = selected_hidden_states(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                hidden_layers=hidden_layers,
            )
        with temporary_direct_coefficients(torch, gate_manager, expert_values):
            source_hidden = selected_hidden_states(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                hidden_layers=hidden_layers,
            )
        with temporary_direct_coefficients(torch, gate_manager, anchor_values):
            anchor_hidden = selected_hidden_states(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                hidden_layers=hidden_layers,
            )
    merged_hidden = selected_hidden_states(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        hidden_layers=hidden_layers,
    )

    response_positions = torch.arange(prompt_len + span_start, prompt_len + span_end, device=device)
    drift_positions = prompt_position_tensor(torch, prompt_len=prompt_len, limit=prompt_drift_tokens, device=device)
    projection_loss = success_projection_loss(
        torch,
        base_hidden=base_hidden,
        source_hidden=source_hidden,
        anchor_hidden=anchor_hidden,
        merged_hidden=merged_hidden,
        positions=response_positions,
        topk_tokens=topk_tokens,
        projection_preserve_margin=projection_preserve_margin,
        deficit_projection_floor=deficit_projection_floor,
        max_projection_improve=max_projection_improve,
        target_normalize_eps=target_normalize_eps,
    )
    if beta_base > 0.0 and drift_positions.numel() > 0:
        base_drift_loss = hidden_match_loss(base_hidden=base_hidden, merged_hidden=merged_hidden, positions=drift_positions)
    else:
        base_drift_loss = torch.zeros((), device=device, dtype=torch.float32)
    gate_anchor_loss = (gate_manager.raw_coefficients - anchor_values.to(device=gate_manager.raw_coefficients.device)).pow(2).mean()
    total_loss = (
        float(projection_preserve_weight) * projection_loss["projection_preserve_loss"]
        + float(deficit_projection_weight) * projection_loss["deficit_projection_loss"]
        + float(deficit_directional_weight) * projection_loss["deficit_directional_loss"]
        + float(beta_base) * base_drift_loss
        + float(gate_anchor_weight) * gate_anchor_loss
    )
    return {
        "total_loss": total_loss,
        "projection_preserve_loss": projection_loss["projection_preserve_loss"].detach(),
        "deficit_projection_loss": projection_loss["deficit_projection_loss"].detach(),
        "deficit_directional_loss": projection_loss["deficit_directional_loss"].detach(),
        "base_drift_loss": base_drift_loss.detach(),
        "gate_anchor_loss": gate_anchor_loss.detach(),
        "mean_anchor_projection": projection_loss["mean_anchor_projection"],
        "mean_current_projection": projection_loss["mean_current_projection"],
        "mean_deficit": projection_loss["mean_deficit"],
        "preserve_active_rate": projection_loss["preserve_active_rate"],
        "deficit_active_rate": projection_loss["deficit_active_rate"],
        "response_tokens": response_len,
        "prompt_tokens": prompt_len,
        "span_mode": resolved_span_mode,
        "span_start_token": span_start,
        "span_end_token": span_end,
        "span_tokens": max(0, span_end - span_start),
        "hidden_layers": list(hidden_layers),
    }


def success_projection_loss(
    torch: Any,
    *,
    base_hidden: dict[int, Any],
    source_hidden: dict[int, Any],
    anchor_hidden: dict[int, Any],
    merged_hidden: dict[int, Any],
    positions: Any,
    topk_tokens: int,
    projection_preserve_margin: float,
    deficit_projection_floor: float,
    max_projection_improve: float,
    target_normalize_eps: float,
) -> dict[str, Any]:
    preserve_losses = []
    deficit_projection_losses = []
    deficit_directional_losses = []
    anchor_projection_values = []
    current_projection_values = []
    deficit_values = []
    preserve_active_values = []
    deficit_active_values = []
    for layer_index in sorted(base_hidden):
        layer_positions = positions.to(device=base_hidden[layer_index].device)
        base = base_hidden[layer_index][:, layer_positions, :].detach()
        target = source_hidden[layer_index][:, layer_positions, :].detach() - base
        anchor = anchor_hidden[layer_index][:, layer_positions, :].detach() - base
        pred = merged_hidden[layer_index][:, layer_positions, :] - base

        weights = target.norm(dim=-1).squeeze(0).clamp_min(1.0e-8)
        if topk_tokens > 0 and weights.numel() > topk_tokens:
            _, top_indices = torch.topk(weights, k=int(topk_tokens), largest=True, sorted=False)
            target = target[:, top_indices, :]
            anchor = anchor[:, top_indices, :]
            pred = pred[:, top_indices, :]
            weights = weights[top_indices]
        weights = weights / weights.mean().clamp_min(1.0e-8)

        target_vec = target.squeeze(0)
        anchor_vec = anchor.squeeze(0)
        pred_vec = pred.squeeze(0)
        denom = target_vec.pow(2).sum(dim=-1).clamp_min(float(target_normalize_eps))
        current_projection = (pred_vec * target_vec).sum(dim=-1) / denom
        anchor_projection = ((anchor_vec * target_vec).sum(dim=-1) / denom).detach()
        preserve_gap = anchor_projection - float(projection_preserve_margin) - current_projection
        preserve_loss = torch.relu(preserve_gap).pow(2)

        floor = torch.full_like(anchor_projection, float(deficit_projection_floor))
        target_projection = torch.minimum(floor, anchor_projection + float(max_projection_improve)).detach()
        deficit = torch.relu(target_projection - anchor_projection).detach()
        current_floor_gap = torch.relu(target_projection - current_projection)
        deficit_projection_loss = deficit * current_floor_gap.pow(2)

        pred_norm = pred_vec.norm(dim=-1).clamp_min(float(target_normalize_eps))
        target_norm = target_vec.norm(dim=-1).clamp_min(float(target_normalize_eps))
        cosine = ((pred_vec * target_vec).sum(dim=-1) / (pred_norm * target_norm)).clamp(min=-1.0, max=1.0)
        deficit_directional_loss = deficit * (1.0 - cosine)

        preserve_losses.append((preserve_loss * weights).mean())
        deficit_projection_losses.append((deficit_projection_loss * weights).mean())
        deficit_directional_losses.append((deficit_directional_loss * weights).mean())
        anchor_projection_values.append(anchor_projection.detach().mean())
        current_projection_values.append(current_projection.detach().mean())
        deficit_values.append(deficit.detach().mean())
        preserve_active_values.append((preserve_gap.detach() > 0).float().mean())
        deficit_active_values.append((deficit.detach() > 0).float().mean())

    return {
        "projection_preserve_loss": torch.stack(preserve_losses).mean(),
        "deficit_projection_loss": torch.stack(deficit_projection_losses).mean(),
        "deficit_directional_loss": torch.stack(deficit_directional_losses).mean(),
        "mean_anchor_projection": mean_tensor_items(anchor_projection_values),
        "mean_current_projection": mean_tensor_items(current_projection_values),
        "mean_deficit": mean_tensor_items(deficit_values),
        "preserve_active_rate": mean_tensor_items(preserve_active_values),
        "deficit_active_rate": mean_tensor_items(deficit_active_values),
    }


def step_optimizer_with_anchor_cap(
    torch: Any,
    gate_manager: Any,
    optimizer: Any,
    *,
    anchor_values: Any,
    max_delta: float,
    grad_clip_norm: float,
    trainable_experts: set[str],
) -> None:
    mask_gate_gradients_for_trainable_experts(gate_manager, trainable_experts)
    if grad_clip_norm > 0.0:
        torch.nn.utils.clip_grad_norm_(list(gate_manager.parameters()), float(grad_clip_norm))
    optimizer.step()
    if hasattr(gate_manager, "project_"):
        gate_manager.project_()
    if max_delta > 0.0:
        with torch.no_grad():
            anchor = anchor_values.to(device=gate_manager.raw_coefficients.device)
            gate_manager.raw_coefficients.copy_(
                gate_manager.raw_coefficients.clamp(
                    min=anchor - float(max_delta),
                    max=anchor + float(max_delta),
                )
            )


def summarize_epoch(epoch: int, metrics: list[dict[str, Any]], gate_manager: Any, *, anchor_values: Any, elapsed: float) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in metrics:
        by_task.setdefault(str(row.get("task")), []).append(row)
    summary = {
        "event": "epoch",
        "epoch": epoch,
        "rows": len(metrics),
        "elapsed_seconds": elapsed,
        "mean_total_loss": mean(item["total_loss"] for item in metrics),
        "mean_projection_preserve_loss": mean(item["projection_preserve_loss"] for item in metrics),
        "mean_deficit_projection_loss": mean(item["deficit_projection_loss"] for item in metrics),
        "mean_deficit_directional_loss": mean(item["deficit_directional_loss"] for item in metrics),
        "mean_base_drift_loss": mean(item["base_drift_loss"] for item in metrics),
        "mean_gate_anchor_loss": mean(item["gate_anchor_loss"] for item in metrics),
        "mean_anchor_projection": mean(item["mean_anchor_projection"] for item in metrics),
        "mean_current_projection": mean(item["mean_current_projection"] for item in metrics),
        "mean_deficit": mean(item["mean_deficit"] for item in metrics),
        "preserve_active_rate": mean(item["preserve_active_rate"] for item in metrics),
        "deficit_active_rate": mean(item["deficit_active_rate"] for item in metrics),
        "task_loss": {
            task: {
                "rows": len(rows),
                "total_loss": mean(item["total_loss"] for item in rows),
                "anchor_projection": mean(item["mean_anchor_projection"] for item in rows),
                "current_projection": mean(item["mean_current_projection"] for item in rows),
                "deficit": mean(item["mean_deficit"] for item in rows),
                "deficit_active_rate": mean(item["deficit_active_rate"] for item in rows),
            }
            for task, rows in sorted(by_task.items())
        },
        "gate_means": gate_means(gate_manager),
        "gate_values": gate_manager.gate_values(),
        "gate_delta_summary": gate_delta_summary(gate_manager, anchor_values),
    }
    return summary


def gate_delta_summary(gate_manager: Any, anchor_values: Any) -> dict[str, Any]:
    delta = (gate_manager.raw_coefficients.detach().cpu() - anchor_values.detach().cpu()).float()
    experts = list(getattr(gate_manager, "expert_names", ()))
    bands = list(getattr(gate_manager, "band_names", ()))
    out: dict[str, Any] = {
        "mean_abs": float(delta.abs().mean().item()),
        "max_abs": float(delta.abs().max().item()),
        "l2": float(delta.norm().item()),
        "by_expert": {},
    }
    for expert_idx, expert in enumerate(experts):
        values = delta[:, expert_idx]
        out["by_expert"][str(expert)] = {
            "mean": float(values.mean().item()),
            "mean_abs": float(values.abs().mean().item()),
            "max_abs": float(values.abs().max().item()),
            "min": float(values.min().item()),
            "max": float(values.max().item()),
            "by_band": {str(band): float(values[idx].item()) for idx, band in enumerate(bands[: values.numel()])},
        }
    return out


def with_anchor_init(
    config: dict[str, Any],
    *,
    init_value: float,
    expert_names: tuple[str, ...],
    anchor_gate_values: dict[str, float],
) -> dict[str, Any]:
    if not anchor_gate_values:
        return with_constant_init(config, init_value=init_value, expert_names=expert_names)
    copied = dict(config)
    band_names = list((config.get("layer_bands") or {}).keys())
    initial: dict[str, float] = {"common": float(init_value)}
    direct_coefficients = global_expert_coefficients(anchor_gate_values, expert_names=expert_names, default=float(init_value))
    for expert, value in direct_coefficients.items():
        initial[expert] = float(value)
        initial[f"global.{expert}"] = float(value)
        initial[f"{expert}_residual"] = float(value) - float(sum(direct_coefficients.values()) / max(1, len(direct_coefficients)))
    for band in band_names:
        for expert in expert_names:
            initial[f"{band}.{expert}"] = float(
                anchor_gate_values.get(f"{band}.{expert}", anchor_gate_values.get(expert, direct_coefficients[expert]))
            )
    copied["initial_gates"] = initial
    return copied


def global_expert_coefficients(gates: dict[str, float], *, expert_names: tuple[str, ...], default: float) -> dict[str, float]:
    if "common" in gates or any(f"{expert}_residual" in gates for expert in expert_names):
        common = float(gates.get("common", default))
        residuals = {expert: float(gates.get(f"{expert}_residual", 0.0)) for expert in expert_names}
        return {expert: common + residuals[expert] for expert in expert_names}
    return {expert: float(gates.get(expert, default)) for expert in expert_names}


def limit_rows_per_task(rows: list[dict[str, Any]], *, per_task: int) -> list[dict[str, Any]]:
    if per_task <= 0:
        return list(rows)
    counts: dict[str, int] = {}
    limited: list[dict[str, Any]] = []
    for row in rows:
        task = str(row.get("task"))
        current = counts.get(task, 0)
        if current >= per_task:
            continue
        limited.append(row)
        counts[task] = current + 1
    return limited


def mean(values: Iterable[float]) -> float:
    vals = [float(value) for value in values]
    if not vals:
        return math.nan
    return sum(vals) / len(vals)


def mean_tensor_items(values: Iterable[Any]) -> float:
    vals = [tensor_item(value) for value in values]
    if not vals:
        return math.nan
    return sum(vals) / len(vals)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/gated_grpo_layer28.yaml")
    parser.add_argument("--mode-manifest", default="/tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json")
    parser.add_argument(
        "--calibration",
        default="/tmp/shared-storage/OnPolicy/data/calibration/20260520_trc_round16_nonleak_code_contrast_v1/trc96_expert_trajectories.jsonl",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gate-parameterization", default="layer-band-coefficient")
    parser.add_argument("--anchor-gate-checkpoint", default="/tmp/shared-storage/OnPolicy/checkpoints/ta_init1_global_20260517/gate_values.json")
    parser.add_argument("--init-value", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--max-rows-per-task", type=int, default=0)
    parser.add_argument("--success-only", action="store_true", default=True)
    parser.add_argument("--train-tasks", default="")
    parser.add_argument("--trainable-experts", default="")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--accumulation-steps", type=int, default=8)
    parser.add_argument("--progress-every-rows", type=int, default=0)
    parser.add_argument("--hidden-layers", default="8,16,24,28")
    parser.add_argument("--task-hidden-layers", action="append", default=[])
    parser.add_argument("--max-seq-length", type=int, default=1536)
    parser.add_argument("--max-response-tokens", type=int, default=512)
    parser.add_argument("--topk-tokens", type=int, default=128)
    parser.add_argument("--task-topk-tokens", action="append", default=[])
    parser.add_argument("--prompt-drift-tokens", type=int, default=256)
    parser.add_argument("--response-span-mode", choices=["response", "auto", "tool-call", "code-block"], default="auto")
    parser.add_argument("--task-response-span-mode", action="append", default=[])
    parser.add_argument("--task-loss-multiplier", action="append", default=[])
    parser.add_argument("--task-balanced-loss", action="store_true")
    parser.add_argument("--projection-preserve-margin", type=float, default=0.02)
    parser.add_argument("--projection-preserve-weight", type=float, default=1.0)
    parser.add_argument("--deficit-projection-floor", type=float, default=0.85)
    parser.add_argument("--max-projection-improve", type=float, default=0.10)
    parser.add_argument("--deficit-projection-weight", type=float, default=0.4)
    parser.add_argument("--deficit-directional-weight", type=float, default=0.05)
    parser.add_argument("--gate-anchor-weight", type=float, default=0.05)
    parser.add_argument("--max-delta-from-anchor", type=float, default=0.08)
    parser.add_argument("--target-normalize-eps", type=float, default=1.0e-6)
    parser.add_argument("--beta-base", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--max-memory", action="append", default=[])
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
