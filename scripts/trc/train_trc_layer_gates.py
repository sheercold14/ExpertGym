#!/usr/bin/env python3
"""Train layer-band OP-VEC gates with TRC hidden-residual alignment.

This is a standalone prototype for TRC-Merging. It does not call or modify the
existing GRPO/OPD update path.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from contextlib import contextmanager
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


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    expert_names = tuple(str(name) for name in config.get("models", {}).get("experts", {}).keys())
    if not expert_names:
        raise ValueError("No experts found in config.models.experts")
    if args.gate_parameterization not in {"layer-band-coefficient", "layer_band_coefficient"}:
        raise ValueError("TRC v1 only supports --gate-parameterization layer-band-coefficient")

    calibration_rows = read_jsonl(args.calibration)
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
    task_residual_weight_power = parse_task_value_overrides(args.task_residual_weight_power, float)
    task_projection_floor = parse_task_value_overrides(args.task_directional_projection_floor, float)
    task_projection_weight = parse_task_value_overrides(args.task_directional_projection_weight, float)
    task_loss_multiplier = parse_task_value_overrides(args.task_loss_multiplier, float)
    trajectory_turn_loss_tasks = {str(task).strip() for task in args.trajectory_turn_loss_task if str(task).strip()}
    contrastive_negative_tasks = {str(task).strip() for task in args.contrastive_negative_task if str(task).strip()}
    run_manifest = {
        "format": "trc_layer_gate_run_manifest_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": str(Path(args.config).expanduser().resolve()),
        "mode_manifest": str(Path(args.mode_manifest).expanduser().resolve()),
        "calibration": str(Path(args.calibration).expanduser().resolve()),
        "output_dir": str(output_dir),
        "args": vars(args),
        "num_rows": len(calibration_rows),
        "task_counts": task_counts(calibration_rows),
        "hidden_layers": hidden_layers,
        "task_hidden_layers": task_hidden_layers,
        "task_response_span_modes": task_response_span_modes,
        "task_topk_tokens": task_topk_tokens,
        "task_residual_weight_power": task_residual_weight_power,
        "task_directional_projection_floor": task_projection_floor,
        "task_directional_projection_weight": task_projection_weight,
        "task_loss_multiplier": task_loss_multiplier,
        "trajectory_turn_loss_tasks": sorted(trajectory_turn_loss_tasks),
        "contrastive_negative_tasks": sorted(contrastive_negative_tasks),
        "expert_names": list(expert_names),
        "status": "initialized",
    }
    write_json(output_dir / "trc_run_manifest.json", run_manifest)
    if args.dry_run:
        run_manifest["status"] = "dry_run_ok"
        write_json(output_dir / "trc_run_manifest.json", run_manifest)
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

    trc_config = with_constant_init(config, init_value=float(args.init_value), expert_names=expert_names)
    gate_manager = make_torch_gate_manager(
        torch,
        trc_config,
        parameterization=args.gate_parameterization,
    ).to(device)
    installed = install_gated_linears_from_manifest(
        torch,
        model,
        mode_manifest_path=args.mode_manifest,
        gate_manager=gate_manager,
        max_modules=None,
        device=None if args.device_map else str(device),
    )
    optimizer = make_optimizer(torch, args.optimizer, gate_manager.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    metrics_path = output_dir / "trc_metrics.jsonl"
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
            loss_kwargs = dict(
                torch=torch,
                model=model,
                tokenizer=tokenizer,
                gate_manager=gate_manager,
                row=row,
                hidden_layers=task_hidden_layers.get(task, hidden_layers),
                device=device,
                max_seq_length=int(args.max_seq_length),
                max_response_tokens=int(args.max_response_tokens),
                topk_tokens=int(task_topk_tokens.get(task, int(args.topk_tokens))),
                prompt_drift_tokens=int(args.prompt_drift_tokens),
                prompt_residual_tokens=int(args.prompt_residual_tokens),
                beta_base=float(args.beta_base),
                prompt_residual_weight=float(args.prompt_residual_weight),
                gamma_gate=float(args.gamma_gate),
                residual_weight_power=float(task_residual_weight_power.get(task, float(args.residual_weight_power))),
                residual_objective=str(args.residual_objective),
                normalize_residual_by_target=bool(args.normalize_residual_by_target),
                target_normalize_eps=float(args.target_normalize_eps),
                directional_projection_floor=float(task_projection_floor.get(task, float(args.directional_projection_floor))),
                directional_projection_weight=float(task_projection_weight.get(task, float(args.directional_projection_weight))),
                coefficient_floor=float(args.coefficient_floor),
                coefficient_floor_weight=float(args.coefficient_floor_weight),
                task_expert_coefficient_floor=float(args.task_expert_coefficient_floor),
                task_expert_coefficient_floor_weight=float(args.task_expert_coefficient_floor_weight),
                response_span_mode=str(task_response_span_modes.get(task, str(args.response_span_mode))),
                contrastive_negative_loss_weight=(
                    float(args.contrastive_negative_loss_weight)
                    if float(args.contrastive_negative_loss_weight) > 0.0
                    and (not contrastive_negative_tasks or task in contrastive_negative_tasks)
                    else 0.0
                ),
                contrastive_negative_margin=float(args.contrastive_negative_margin),
                contrastive_negative_response_key=str(args.contrastive_negative_response_key),
            )
            if task in trajectory_turn_loss_tasks and row.get("trajectory_turns"):
                losses = compute_trc_trajectory_turn_loss(**loss_kwargs)
            else:
                losses = compute_trc_row_loss(**loss_kwargs)
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
                "residual_loss": tensor_item(losses["residual_loss"]),
                "prompt_residual_loss": tensor_item(losses["prompt_residual_loss"]),
                "base_drift_loss": tensor_item(losses["base_drift_loss"]),
                "gate_anchor_loss": tensor_item(losses["gate_anchor_loss"]),
                "coefficient_floor_loss": tensor_item(losses["coefficient_floor_loss"]),
                "task_expert_coefficient_floor_loss": tensor_item(losses["task_expert_coefficient_floor_loss"]),
                "contrastive_negative_loss": tensor_item(losses.get("contrastive_negative_loss", 0.0)),
                "negative_residual_loss": tensor_item(losses.get("negative_residual_loss", 0.0)),
                "negative_response_tokens": int(losses.get("negative_response_tokens", 0)),
                "contrastive_negative_active": bool(losses.get("contrastive_negative_active", False)),
                "loss_scale": loss_scale,
                "response_tokens": int(losses["response_tokens"]),
                "prompt_tokens": int(losses["prompt_tokens"]),
                "span_mode": str(losses["span_mode"]),
                "span_start_token": int(losses["span_start_token"]),
                "span_end_token": int(losses["span_end_token"]),
                "span_tokens": int(losses["span_tokens"]),
                "selected_response_tokens": int(losses["selected_response_tokens"]),
                "hidden_layers": losses["hidden_layers"],
                "trajectory_turns": int(losses.get("trajectory_turns", 1)),
                "elapsed_seconds": time.time() - row_start,
            }
            epoch_metrics.append(metric)
            metrics_rows.append(metric)
            if pending_backward >= int(args.accumulation_steps):
                step_optimizer(torch, gate_manager, optimizer, grad_clip_norm=float(args.grad_clip_norm))
                optimizer.zero_grad(set_to_none=True)
                pending_backward = 0
        if pending_backward:
            step_optimizer(torch, gate_manager, optimizer, grad_clip_norm=float(args.grad_clip_norm))
            optimizer.zero_grad(set_to_none=True)
        summary = summarize_epoch(epoch, epoch_metrics, gate_manager, elapsed=time.time() - epoch_start)
        metrics_rows.append(summary)
        write_json(output_dir / f"epoch_{epoch:03d}.gates.json", {"gates": gate_manager.gate_values(), "epoch_summary": summary})
        write_jsonl(metrics_path, metrics_rows)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))

    final_summary = {
        "format": "trc_layer_gate_summary_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - start_time,
        "installed_modules": len(installed),
        "final_gates": gate_manager.gate_values(),
        "metrics": str(metrics_path),
    }
    write_json(output_dir / "trc_gates.json", {"gates": gate_manager.gate_values(), "summary": final_summary})
    write_json(output_dir / "trc_summary.json", final_summary)
    run_manifest["status"] = "completed"
    run_manifest["final_gate_checkpoint"] = str(output_dir / "trc_gates.json")
    write_json(output_dir / "trc_run_manifest.json", run_manifest)


def validate_calibration_rows(rows: list[dict[str, Any]], *, expert_names: tuple[str, ...]) -> None:
    if not rows:
        raise ValueError("Calibration is empty")
    expert_set = set(expert_names)
    for index, row in enumerate(rows, start=1):
        if row.get("expert") not in expert_set:
            raise ValueError(f"Row {index} has unknown expert {row.get('expert')!r}; expected one of {sorted(expert_set)}")
        if not str(row.get("response") or "").strip():
            raise ValueError(f"Row {index} has empty response")
        if not str(row.get("rendered_prompt") or row.get("prompt") or "").strip():
            raise ValueError(f"Row {index} has empty prompt")


def compute_trc_row_loss(
    *,
    torch: Any,
    model: Any,
    tokenizer: Any,
    gate_manager: Any,
    row: dict[str, Any],
    hidden_layers: list[int],
    device: Any,
    max_seq_length: int,
    max_response_tokens: int,
    topk_tokens: int,
    prompt_drift_tokens: int,
    prompt_residual_tokens: int,
    beta_base: float,
    prompt_residual_weight: float,
    gamma_gate: float,
    residual_weight_power: float,
    residual_objective: str,
    normalize_residual_by_target: bool,
    target_normalize_eps: float,
    directional_projection_floor: float,
    directional_projection_weight: float,
    coefficient_floor: float,
    coefficient_floor_weight: float,
    task_expert_coefficient_floor: float,
    task_expert_coefficient_floor_weight: float,
    response_span_mode: str,
    contrastive_negative_loss_weight: float,
    contrastive_negative_margin: float,
    contrastive_negative_response_key: str,
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
            expert_hidden = selected_hidden_states(
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
    prompt_residual_positions = prompt_position_tensor(torch, prompt_len=prompt_len, limit=prompt_residual_tokens, device=device)
    residual_loss = hidden_residual_loss(
        torch,
        base_hidden=base_hidden,
        expert_hidden=expert_hidden,
        merged_hidden=merged_hidden,
        positions=response_positions,
        topk_tokens=topk_tokens,
        residual_weight_power=residual_weight_power,
        residual_objective=residual_objective,
        normalize_by_target=normalize_residual_by_target,
        target_normalize_eps=target_normalize_eps,
        directional_projection_floor=directional_projection_floor,
        directional_projection_weight=directional_projection_weight,
    )
    if prompt_residual_weight > 0.0 and prompt_residual_positions.numel() > 0:
        prompt_residual_loss = hidden_residual_loss(
            torch,
            base_hidden=base_hidden,
            expert_hidden=expert_hidden,
            merged_hidden=merged_hidden,
            positions=prompt_residual_positions,
            topk_tokens=topk_tokens,
            residual_weight_power=residual_weight_power,
            residual_objective=residual_objective,
            normalize_by_target=normalize_residual_by_target,
            target_normalize_eps=target_normalize_eps,
            directional_projection_floor=directional_projection_floor,
            directional_projection_weight=directional_projection_weight,
        )
    else:
        prompt_residual_loss = torch.zeros((), device=device, dtype=torch.float32)
    if beta_base > 0.0 and drift_positions.numel() > 0:
        base_drift_loss = hidden_match_loss(
            base_hidden=base_hidden,
            merged_hidden=merged_hidden,
            positions=drift_positions,
        )
    else:
        base_drift_loss = torch.zeros((), device=device, dtype=torch.float32)
    gate_anchor_loss = direct_gate_anchor_loss(torch, gate_manager)
    coefficient_floor_loss = direct_gate_floor_loss(torch, gate_manager, coefficient_floor=coefficient_floor)
    task_expert_coefficient_floor_loss = direct_task_expert_gate_floor_loss(
        torch,
        gate_manager,
        expert=str(row["expert"]),
        coefficient_floor=task_expert_coefficient_floor,
    )
    contrastive_negative_loss = torch.zeros((), device=device, dtype=torch.float32)
    negative_residual_loss = torch.zeros((), device=device, dtype=torch.float32)
    negative_response_tokens = 0
    contrastive_negative_active = False
    negative_response = str(row.get(contrastive_negative_response_key) or "").strip()
    if contrastive_negative_loss_weight > 0.0 and negative_response:
        negative_losses = compute_negative_residual_loss(
            torch=torch,
            model=model,
            tokenizer=tokenizer,
            gate_manager=gate_manager,
            prompt_text=str(row.get("rendered_prompt") or row.get("prompt") or ""),
            negative_response=negative_response,
            expert=str(row["expert"]),
            hidden_layers=hidden_layers,
            device=device,
            max_seq_length=max_seq_length,
            max_response_tokens=max_response_tokens,
            topk_tokens=topk_tokens,
            residual_weight_power=residual_weight_power,
            residual_objective=residual_objective,
            normalize_residual_by_target=normalize_residual_by_target,
            target_normalize_eps=target_normalize_eps,
            directional_projection_floor=directional_projection_floor,
            directional_projection_weight=directional_projection_weight,
            response_span_mode=response_span_mode,
            task=str(row.get("task") or ""),
        )
        if negative_losses is not None:
            negative_residual_loss = negative_losses["negative_residual_loss"]
            negative_response_tokens = int(negative_losses["negative_response_tokens"])
            contrastive_negative_loss = torch.relu(
                float(contrastive_negative_margin) + residual_loss - negative_residual_loss
            )
            contrastive_negative_active = bool(tensor_item(contrastive_negative_loss) > 0.0)
    total_loss = (
        residual_loss
        + prompt_residual_weight * prompt_residual_loss
        + beta_base * base_drift_loss
        + gamma_gate * gate_anchor_loss
        + coefficient_floor_weight * coefficient_floor_loss
        + task_expert_coefficient_floor_weight * task_expert_coefficient_floor_loss
        + contrastive_negative_loss_weight * contrastive_negative_loss
    )
    return {
        "total_loss": total_loss,
        "residual_loss": residual_loss.detach(),
        "prompt_residual_loss": prompt_residual_loss.detach(),
        "base_drift_loss": base_drift_loss.detach(),
        "gate_anchor_loss": gate_anchor_loss.detach(),
        "coefficient_floor_loss": coefficient_floor_loss.detach(),
        "task_expert_coefficient_floor_loss": task_expert_coefficient_floor_loss.detach(),
        "contrastive_negative_loss": contrastive_negative_loss.detach(),
        "negative_residual_loss": negative_residual_loss.detach(),
        "negative_response_tokens": negative_response_tokens,
        "contrastive_negative_active": contrastive_negative_active,
        "response_tokens": response_len,
        "prompt_tokens": prompt_len,
        "span_mode": resolved_span_mode,
        "span_start_token": span_start,
        "span_end_token": span_end,
        "span_tokens": max(0, span_end - span_start),
        "selected_response_tokens": min(max(0, span_end - span_start), topk_tokens) if topk_tokens > 0 else max(0, span_end - span_start),
        "hidden_layers": list(hidden_layers),
    }


def compute_trc_trajectory_turn_loss(**kwargs: Any) -> dict[str, Any] | None:
    row = dict(kwargs["row"])
    turns = [turn for turn in row.get("trajectory_turns") or [] if str(turn.get("prompt_text") or "").strip() and str(turn.get("text") or "").strip()]
    if not turns:
        return compute_trc_row_loss(**kwargs)
    turn_losses: list[dict[str, Any]] = []
    for turn in turns:
        turn_row = dict(row)
        turn_row["prompt"] = str(turn.get("prompt_text") or "")
        turn_row["rendered_prompt"] = str(turn.get("prompt_text") or "")
        turn_row["response"] = str(turn.get("text") or "")
        turn_kwargs = dict(kwargs)
        turn_kwargs["row"] = turn_row
        losses = compute_trc_row_loss(**turn_kwargs)
        if losses is not None:
            turn_losses.append(losses)
    if not turn_losses:
        return None
    torch = kwargs["torch"]
    total_loss = torch.stack([item["total_loss"] for item in turn_losses]).mean()
    return {
        "total_loss": total_loss,
        "residual_loss": torch.stack([item["residual_loss"] for item in turn_losses]).mean(),
        "prompt_residual_loss": torch.stack([item["prompt_residual_loss"] for item in turn_losses]).mean(),
        "base_drift_loss": torch.stack([item["base_drift_loss"] for item in turn_losses]).mean(),
        "gate_anchor_loss": torch.stack([item["gate_anchor_loss"] for item in turn_losses]).mean(),
        "coefficient_floor_loss": torch.stack([item["coefficient_floor_loss"] for item in turn_losses]).mean(),
        "task_expert_coefficient_floor_loss": torch.stack([item["task_expert_coefficient_floor_loss"] for item in turn_losses]).mean(),
        "response_tokens": sum(int(item["response_tokens"]) for item in turn_losses),
        "prompt_tokens": sum(int(item["prompt_tokens"]) for item in turn_losses),
        "span_mode": "trajectory-turns",
        "span_start_token": min(int(item["span_start_token"]) for item in turn_losses),
        "span_end_token": max(int(item["span_end_token"]) for item in turn_losses),
        "span_tokens": sum(int(item["span_tokens"]) for item in turn_losses),
        "selected_response_tokens": sum(int(item["selected_response_tokens"]) for item in turn_losses),
        "hidden_layers": list(kwargs["hidden_layers"]),
        "trajectory_turns": len(turn_losses),
    }


def compute_negative_residual_loss(
    *,
    torch: Any,
    model: Any,
    tokenizer: Any,
    gate_manager: Any,
    prompt_text: str,
    negative_response: str,
    expert: str,
    hidden_layers: list[int],
    device: Any,
    max_seq_length: int,
    max_response_tokens: int,
    topk_tokens: int,
    residual_weight_power: float,
    residual_objective: str,
    normalize_residual_by_target: bool,
    target_normalize_eps: float,
    directional_projection_floor: float,
    directional_projection_weight: float,
    response_span_mode: str,
    task: str,
) -> dict[str, Any] | None:
    encoded = encode_prompt_response(
        tokenizer,
        prompt_text=prompt_text,
        response_text=negative_response,
        max_seq_length=max_seq_length,
        max_response_tokens=max_response_tokens,
    )
    if encoded is None:
        return None
    input_ids, attention_mask, prompt_len, response_len = encoded
    span_start, span_end, _ = response_token_span(
        tokenizer,
        task=task,
        response_text=negative_response,
        response_len=response_len,
        max_response_tokens=max_response_tokens,
        mode=response_span_mode,
    )
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    base_values = torch.zeros_like(gate_manager.raw_coefficients)
    expert_values = torch.zeros_like(gate_manager.raw_coefficients)
    expert_idx = list(gate_manager.expert_names).index(str(expert))
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
            expert_hidden = selected_hidden_states(
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
    residual_loss = hidden_residual_loss(
        torch,
        base_hidden=base_hidden,
        expert_hidden=expert_hidden,
        merged_hidden=merged_hidden,
        positions=response_positions,
        topk_tokens=topk_tokens,
        residual_weight_power=residual_weight_power,
        residual_objective=residual_objective,
        normalize_by_target=normalize_residual_by_target,
        target_normalize_eps=target_normalize_eps,
        directional_projection_floor=directional_projection_floor,
        directional_projection_weight=directional_projection_weight,
    )
    return {
        "negative_residual_loss": residual_loss,
        "negative_response_tokens": response_len,
    }


def selected_hidden_states(
    model: Any,
    *,
    input_ids: Any,
    attention_mask: Any,
    hidden_layers: list[int],
) -> dict[int, Any]:
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    hidden_states = outputs.hidden_states
    max_index = len(hidden_states) - 1
    selected = {}
    for layer_index in hidden_layers:
        if layer_index < 0 or layer_index > max_index:
            raise ValueError(f"Hidden layer index {layer_index} out of range [0, {max_index}]")
        selected[layer_index] = hidden_states[layer_index].float()
    return selected


def hidden_residual_loss(
    torch: Any,
    *,
    base_hidden: dict[int, Any],
    expert_hidden: dict[int, Any],
    merged_hidden: dict[int, Any],
    positions: Any,
    topk_tokens: int,
    residual_weight_power: float,
    residual_objective: str,
    normalize_by_target: bool,
    target_normalize_eps: float,
    directional_projection_floor: float,
    directional_projection_weight: float,
) -> Any:
    losses = []
    objective = str(residual_objective).lower()
    for layer_index in sorted(base_hidden):
        layer_positions = positions.to(device=base_hidden[layer_index].device)
        base = base_hidden[layer_index][:, layer_positions, :].detach()
        target = expert_hidden[layer_index][:, layer_positions, :].detach() - base
        pred = merged_hidden[layer_index][:, layer_positions, :] - base
        token_error = (pred - target).pow(2).mean(dim=-1).squeeze(0)
        target_energy = target.pow(2).mean(dim=-1).squeeze(0)
        weights = target.norm(dim=-1).squeeze(0).clamp_min(1.0e-8)
        if residual_weight_power != 1.0:
            weights = weights.pow(float(residual_weight_power))
        if topk_tokens > 0 and token_error.numel() > topk_tokens:
            _, top_indices = torch.topk(weights, k=int(topk_tokens), largest=True, sorted=False)
            token_error = token_error[top_indices]
            target_energy = target_energy[top_indices]
            pred = pred[:, top_indices, :]
            target = target[:, top_indices, :]
            weights = weights[top_indices]
        weights = weights / weights.mean().clamp_min(1.0e-8)
        if objective == "directional":
            pred_vec = pred.squeeze(0)
            target_vec = target.squeeze(0)
            dot = (pred_vec * target_vec).sum(dim=-1)
            pred_norm = pred_vec.norm(dim=-1).clamp_min(float(target_normalize_eps))
            target_norm = target_vec.norm(dim=-1).clamp_min(float(target_normalize_eps))
            cosine_loss = 1.0 - (dot / (pred_norm * target_norm)).clamp(min=-1.0, max=1.0)
            loss = (cosine_loss * weights).mean()
            if directional_projection_floor > 0.0 and directional_projection_weight > 0.0:
                projection_ratio = dot / target_vec.pow(2).sum(dim=-1).clamp_min(float(target_normalize_eps))
                projection_loss = (torch.relu(float(directional_projection_floor) - projection_ratio).pow(2) * weights).mean()
                loss = loss + float(directional_projection_weight) * projection_loss
            losses.append(loss)
        elif normalize_by_target or objective == "relative-mse":
            numerator = (token_error * weights).sum()
            denominator = (target_energy * weights).sum().clamp_min(float(target_normalize_eps))
            losses.append(numerator / denominator)
        else:
            losses.append((token_error * weights).mean())
    return torch.stack(losses).mean()


def hidden_match_loss(*, base_hidden: dict[int, Any], merged_hidden: dict[int, Any], positions: Any) -> Any:
    losses = []
    for layer_index in sorted(base_hidden):
        layer_positions = positions.to(device=base_hidden[layer_index].device)
        base = base_hidden[layer_index][:, layer_positions, :].detach()
        pred = merged_hidden[layer_index][:, layer_positions, :]
        losses.append((pred - base).pow(2).mean())
    return sum(losses) / float(len(losses))


def direct_gate_anchor_loss(torch: Any, gate_manager: Any) -> Any:
    if not hasattr(gate_manager, "raw_coefficients") or not hasattr(gate_manager, "initial_coefficients"):
        return torch.zeros((), device=next(gate_manager.parameters()).device, dtype=torch.float32)
    return (gate_manager.raw_coefficients - gate_manager.initial_coefficients).pow(2).mean()


def direct_gate_floor_loss(torch: Any, gate_manager: Any, *, coefficient_floor: float) -> Any:
    if coefficient_floor <= 0.0 or not hasattr(gate_manager, "raw_coefficients"):
        return torch.zeros((), device=next(gate_manager.parameters()).device, dtype=torch.float32)
    return torch.relu(float(coefficient_floor) - gate_manager.raw_coefficients).pow(2).mean()


def direct_task_expert_gate_floor_loss(torch: Any, gate_manager: Any, *, expert: str, coefficient_floor: float) -> Any:
    if coefficient_floor <= 0.0 or not hasattr(gate_manager, "raw_coefficients"):
        return torch.zeros((), device=next(gate_manager.parameters()).device, dtype=torch.float32)
    try:
        expert_idx = list(gate_manager.expert_names).index(str(expert))
    except ValueError:
        return torch.zeros((), device=next(gate_manager.parameters()).device, dtype=torch.float32)
    expert_coefficients = gate_manager.raw_coefficients[:, expert_idx]
    return torch.relu(float(coefficient_floor) - expert_coefficients).pow(2).mean()


def response_token_span(
    tokenizer: Any,
    *,
    task: str,
    response_text: str,
    response_len: int,
    max_response_tokens: int,
    mode: str,
) -> tuple[int, int, str]:
    if response_len <= 0:
        return 0, 0, "empty"
    normalized = str(mode).strip().lower()
    if normalized == "response":
        return 0, response_len, "response"
    char_span, resolved = response_char_span(task=task, response_text=response_text, mode=normalized)
    if char_span is None:
        return 0, response_len, "response_fallback"
    start_char, end_char = char_span
    start_token = len(tokenizer(response_text[:start_char], add_special_tokens=False).input_ids)
    span_token_len = len(tokenizer(response_text[start_char:end_char], add_special_tokens=False).input_ids)
    end_token = start_token + max(1, span_token_len)
    if max_response_tokens > 0:
        start_token = min(start_token, max_response_tokens)
        end_token = min(end_token, max_response_tokens)
    start_token = max(0, min(start_token, response_len - 1))
    end_token = max(start_token + 1, min(end_token, response_len))
    return start_token, end_token, resolved


def response_char_span(*, task: str, response_text: str, mode: str) -> tuple[tuple[int, int] | None, str]:
    if not response_text:
        return None, "response_fallback"
    if mode == "auto":
        if task == "tool":
            mode = "tool-call"
        elif task == "code":
            mode = "code-block"
        else:
            mode = "response"
    if mode in {"tool-call", "tool_call"}:
        start = response_text.find("<tool_call>")
        end = response_text.find("</tool_call>", start + 1)
        if start >= 0 and end >= 0:
            return (start, end + len("</tool_call>")), "tool-call"
        return None, "response_fallback"
    if mode in {"code-block", "code_block"}:
        matches = list(re.finditer(r"```(?:python|py)?\s*\n?(.*?)```", response_text, flags=re.IGNORECASE | re.DOTALL))
        if matches:
            best = max(matches, key=lambda item: len(item.group(1)))
            return (best.start(1), best.end(1)), "code-block"
        return None, "response_fallback"
    return (0, len(response_text)), "response"


def encode_prompt_response(
    tokenizer: Any,
    *,
    prompt_text: str,
    response_text: str,
    max_seq_length: int,
    max_response_tokens: int,
) -> tuple[Any, Any, int, int] | None:
    import torch

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    response_ids = tokenizer(response_text, add_special_tokens=False).input_ids
    if max_response_tokens > 0:
        response_ids = response_ids[:max_response_tokens]
    if not prompt_ids or not response_ids:
        return None
    budget_for_prompt = max(1, max_seq_length - len(response_ids))
    if len(prompt_ids) > budget_for_prompt:
        prompt_ids = prompt_ids[-budget_for_prompt:]
    input_ids = prompt_ids + response_ids
    if len(input_ids) > max_seq_length:
        input_ids = input_ids[-max_seq_length:]
        prompt_ids = input_ids[: max(1, len(input_ids) - len(response_ids))]
        response_ids = input_ids[len(prompt_ids) :]
    tensor = torch.tensor([input_ids], dtype=torch.long)
    attention = torch.ones_like(tensor)
    return tensor, attention, len(prompt_ids), len(response_ids)


def prompt_position_tensor(torch: Any, *, prompt_len: int, limit: int, device: Any) -> Any:
    if prompt_len <= 0:
        return torch.empty((0,), dtype=torch.long, device=device)
    start = 0 if limit <= 0 or prompt_len <= limit else prompt_len - limit
    return torch.arange(start, prompt_len, device=device)


@contextmanager
def temporary_direct_coefficients(torch: Any, gate_manager: Any, values: Any):
    if not hasattr(gate_manager, "raw_coefficients"):
        raise ValueError("TRC v1 requires a direct raw_coefficients gate manager")
    old_values = gate_manager.raw_coefficients.detach().clone()
    with torch.no_grad():
        gate_manager.raw_coefficients.copy_(values.to(device=gate_manager.raw_coefficients.device))
    try:
        yield
    finally:
        with torch.no_grad():
            gate_manager.raw_coefficients.copy_(old_values)


def step_optimizer(torch: Any, gate_manager: Any, optimizer: Any, *, grad_clip_norm: float) -> None:
    if grad_clip_norm > 0.0:
        torch.nn.utils.clip_grad_norm_(list(gate_manager.parameters()), grad_clip_norm)
    optimizer.step()
    if hasattr(gate_manager, "project_"):
        gate_manager.project_()


def summarize_epoch(epoch: int, metrics: list[dict[str, Any]], gate_manager: Any, *, elapsed: float) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in metrics:
        by_task.setdefault(str(row.get("task")), []).append(row)
    task_loss = {
        task: {
            "rows": len(rows),
            "total_loss": mean(item["total_loss"] for item in rows),
            "residual_loss": mean(item["residual_loss"] for item in rows),
            "prompt_residual_loss": mean(item["prompt_residual_loss"] for item in rows),
            "base_drift_loss": mean(item["base_drift_loss"] for item in rows),
            "gate_anchor_loss": mean(item["gate_anchor_loss"] for item in rows),
            "coefficient_floor_loss": mean(item.get("coefficient_floor_loss", 0.0) for item in rows),
            "task_expert_coefficient_floor_loss": mean(item.get("task_expert_coefficient_floor_loss", 0.0) for item in rows),
            "contrastive_negative_loss": mean(item.get("contrastive_negative_loss", 0.0) for item in rows),
            "negative_residual_loss": mean(item.get("negative_residual_loss", 0.0) for item in rows),
            "contrastive_negative_active_rate": mean(1.0 if item.get("contrastive_negative_active") else 0.0 for item in rows),
            "span_tokens": mean(item.get("span_tokens", item.get("selected_response_tokens", 0)) for item in rows),
            "trajectory_turns": mean(item.get("trajectory_turns", 1) for item in rows),
            "loss_scale": mean(item.get("loss_scale", 1.0) for item in rows),
        }
        for task, rows in sorted(by_task.items())
    }
    return {
        "event": "epoch",
        "epoch": epoch,
        "rows": len(metrics),
        "elapsed_seconds": elapsed,
        "mean_total_loss": mean(item["total_loss"] for item in metrics),
        "mean_residual_loss": mean(item["residual_loss"] for item in metrics),
        "mean_prompt_residual_loss": mean(item["prompt_residual_loss"] for item in metrics),
        "mean_base_drift_loss": mean(item["base_drift_loss"] for item in metrics),
        "mean_gate_anchor_loss": mean(item["gate_anchor_loss"] for item in metrics),
        "mean_coefficient_floor_loss": mean(item.get("coefficient_floor_loss", 0.0) for item in metrics),
        "mean_task_expert_coefficient_floor_loss": mean(item.get("task_expert_coefficient_floor_loss", 0.0) for item in metrics),
        "mean_contrastive_negative_loss": mean(item.get("contrastive_negative_loss", 0.0) for item in metrics),
        "mean_negative_residual_loss": mean(item.get("negative_residual_loss", 0.0) for item in metrics),
        "contrastive_negative_active_rate": mean(1.0 if item.get("contrastive_negative_active") else 0.0 for item in metrics),
        "task_loss": task_loss,
        "gate_means": gate_means(gate_manager),
        "gate_values": gate_manager.gate_values(),
    }


def gate_means(gate_manager: Any) -> dict[str, float]:
    values = gate_manager.gate_values()
    experts = tuple(getattr(gate_manager, "expert_names", ()))
    out: dict[str, float] = {}
    for expert in experts:
        expert_values = [value for key, value in values.items() if key.endswith(f".{expert}") or key == expert]
        if expert_values:
            out[expert] = sum(expert_values) / len(expert_values)
    return out


def make_optimizer(torch: Any, name: str, params: Iterable[Any], *, lr: float, weight_decay: float) -> Any:
    lowered = str(name).lower()
    if lowered == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    if lowered == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def with_constant_init(config: dict[str, Any], *, init_value: float, expert_names: tuple[str, ...]) -> dict[str, Any]:
    copied = dict(config)
    initial = {"common": float(init_value)}
    for expert in expert_names:
        initial[expert] = float(init_value)
        initial[f"global.{expert}"] = float(init_value)
        initial[f"{expert}_residual"] = 0.0
    copied["initial_gates"] = initial
    return copied


def resolve_torch_dtype(torch: Any, name: str) -> Any:
    if str(name) == "auto":
        return torch.bfloat16
    return getattr(torch, str(name))


def tensor_item(value: Any) -> float:
    try:
        return float(value.detach().cpu().item())
    except AttributeError:
        return float(value)


def mean(values: Iterable[float]) -> float:
    vals = [float(value) for value in values]
    if not vals:
        return math.nan
    return sum(vals) / len(vals)


def task_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        task = str(row.get("task"))
        counts[task] = counts.get(task, 0) + 1
    return counts


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


def task_balanced_row_scales(rows: list[dict[str, Any]]) -> dict[str, float]:
    counts = task_counts(rows)
    if not counts:
        return {}
    total = float(sum(counts.values()))
    num_tasks = float(len(counts))
    return {task: total / (num_tasks * float(count)) for task, count in counts.items() if count > 0}


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_task_value_overrides(items: list[str] | None, cast: Any) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for raw in items or []:
        if "=" not in raw:
            raise ValueError(f"Task override must use task=value format: {raw!r}")
        task, value = raw.split("=", 1)
        task = task.strip()
        if not task:
            raise ValueError(f"Task override has empty task: {raw!r}")
        overrides[task] = cast(value.strip())
    return overrides


def parse_task_int_list_overrides(items: list[str] | None) -> dict[str, list[int]]:
    overrides: dict[str, list[int]] = {}
    for raw in items or []:
        if "=" not in raw:
            raise ValueError(f"Task hidden layer override must use task=1,2,3 format: {raw!r}")
        task, value = raw.split("=", 1)
        task = task.strip()
        if not task:
            raise ValueError(f"Task hidden layer override has empty task: {raw!r}")
        layers = parse_int_list(value)
        if not layers:
            raise ValueError(f"Task {task!r} hidden layer override is empty")
        overrides[task] = layers
    return overrides


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/gated_grpo_layer28.yaml")
    parser.add_argument("--mode-manifest", default="/tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json")
    parser.add_argument(
        "--calibration",
        default="/tmp/shared-storage/OnPolicy/data/calibration/20260519_trc_v1/trc96_expert_trajectories.jsonl",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gate-parameterization", default="layer-band-coefficient")
    parser.add_argument("--init-value", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument(
        "--max-rows-per-task",
        type=int,
        default=0,
        help="Balanced probe mode: keep at most N rows per task. Default 0 uses all rows unless --max-rows is set.",
    )
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--accumulation-steps", type=int, default=8)
    parser.add_argument("--hidden-layers", default="8,16,24,28")
    parser.add_argument(
        "--task-hidden-layers",
        action="append",
        default=[],
        help="Override hidden layers for a task, e.g. --task-hidden-layers code=4,8,12,16,20,24,28.",
    )
    parser.add_argument("--max-seq-length", type=int, default=1536)
    parser.add_argument("--max-response-tokens", type=int, default=512)
    parser.add_argument("--topk-tokens", type=int, default=128)
    parser.add_argument(
        "--task-topk-tokens",
        action="append",
        default=[],
        help="Override top-k residual tokens for a task, e.g. --task-topk-tokens tool=64.",
    )
    parser.add_argument("--prompt-drift-tokens", type=int, default=256)
    parser.add_argument(
        "--prompt-residual-tokens",
        type=int,
        default=256,
        help="Prompt-tail token budget for optional expert-residual prompt alignment. Used only when --prompt-residual-weight > 0.",
    )
    parser.add_argument(
        "--prompt-residual-weight",
        type=float,
        default=0.0,
        help="Optional expert-residual alignment loss on prompt tokens. Default 0 keeps the old output-span-only path.",
    )
    parser.add_argument("--residual-weight-power", type=float, default=1.0)
    parser.add_argument(
        "--task-residual-weight-power",
        action="append",
        default=[],
        help="Override residual token weight power for a task, e.g. --task-residual-weight-power code=0.75.",
    )
    parser.add_argument(
        "--residual-objective",
        choices=["mse", "relative-mse", "directional"],
        default="mse",
        help="TRC residual objective. directional matches expert residual direction without penalizing extra orthogonal expert components.",
    )
    parser.add_argument(
        "--normalize-residual-by-target",
        action="store_true",
        help="Use relative residual MSE: weighted ||r_merge-r_expert||^2 / weighted ||r_expert||^2.",
    )
    parser.add_argument("--target-normalize-eps", type=float, default=1.0e-6)
    parser.add_argument("--directional-projection-floor", type=float, default=0.0)
    parser.add_argument("--directional-projection-weight", type=float, default=0.1)
    parser.add_argument(
        "--task-directional-projection-floor",
        action="append",
        default=[],
        help="Override directional projection floor for a task, e.g. --task-directional-projection-floor code=0.9.",
    )
    parser.add_argument(
        "--task-directional-projection-weight",
        action="append",
        default=[],
        help="Override directional projection weight for a task, e.g. --task-directional-projection-weight code=0.2.",
    )
    parser.add_argument("--coefficient-floor", type=float, default=0.0)
    parser.add_argument("--coefficient-floor-weight", type=float, default=0.0)
    parser.add_argument(
        "--task-expert-coefficient-floor",
        type=float,
        default=0.0,
        help="Optional row-task expert coefficient floor. Applies only to the row expert and is off by default.",
    )
    parser.add_argument(
        "--task-expert-coefficient-floor-weight",
        type=float,
        default=0.0,
        help="Weight for --task-expert-coefficient-floor. Default 0 leaves old behavior unchanged.",
    )
    parser.add_argument(
        "--response-span-mode",
        choices=["response", "auto", "tool-call", "code-block"],
        default="response",
        help="Which response span to align. auto uses tool_call span for tool and code block for code.",
    )
    parser.add_argument(
        "--task-response-span-mode",
        action="append",
        default=[],
        help="Override response span mode for a task, e.g. --task-response-span-mode memory=response.",
    )
    parser.add_argument(
        "--task-loss-multiplier",
        action="append",
        default=[],
        help="Multiply total row loss for a task after task-balancing, e.g. --task-loss-multiplier code=1.2.",
    )
    parser.add_argument(
        "--trajectory-turn-loss-task",
        action="append",
        default=[],
        help="Task name whose rows should average TRC loss over row['trajectory_turns']; repeatable. Default keeps old single-response path.",
    )
    parser.add_argument(
        "--task-balanced-loss",
        action="store_true",
        help="Scale row losses so each task contributes equally when task counts are imbalanced.",
    )
    parser.add_argument(
        "--contrastive-negative-loss-weight",
        type=float,
        default=0.0,
        help="Optional hinge loss weight for rows with a negative response. Default 0 disables the branch.",
    )
    parser.add_argument(
        "--contrastive-negative-margin",
        type=float,
        default=0.05,
        help="Require negative residual loss to exceed positive residual loss by this margin when contrastive loss is enabled.",
    )
    parser.add_argument(
        "--contrastive-negative-response-key",
        default="negative_response",
        help="Row field containing a failed response used by the optional contrastive branch.",
    )
    parser.add_argument(
        "--contrastive-negative-task",
        action="append",
        default=[],
        help="Task allowlist for optional negative contrastive loss, e.g. --contrastive-negative-task code. Empty means all tasks.",
    )
    parser.add_argument("--beta-base", type=float, default=0.02)
    parser.add_argument("--gamma-gate", type=float, default=0.001)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--max-memory", action="append", default=[])
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable model gradient checkpointing for memory-heavy TRC gate-gradient runs. Default is off.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
