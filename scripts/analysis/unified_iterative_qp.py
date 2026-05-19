#!/usr/bin/env python3
"""Unified Iterative Utility/Harm QP over all experts.

Solves for ALL expert coefficients (compatible + incompatible) in one
unified optimization, starting from z=0. This removes the dependency on
a grid-searched TA-0.75 baseline.

Algorithm:
    Initialize z = 0 for all modes (all experts)
    For t = 1..T:
        1. Apply current z to base model: W_p += sum_e z_{e,p} * delta_{e,p}
        2. Forward+backward on calibration data per task
        3. Compute utility u_j = -<grad, delta_j> for each mode j
        4. Compute harm h_j = max(<grad_protected, delta_j>, 0)
        5. QP step: z_new = soft_threshold(u, lambda_sparse) / (h + lambda_trust)
        6. Optional: clip z to budget
        7. Restore base model

Usage:
    PY=/path/to/python GPU=0 python scripts/analysis/unified_iterative_qp.py \
        --num-iters 5 \
        --samples-per-task 5 \
        --output-dir /tmp/shared-storage/ExpertGym/analysis/unified_qp_v1
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Model paths
BASE_MODEL = "/mnt/cache/wuruixiao/models/Qwen2.5-7b-instruct"
EXPERTS = {
    "tool": "/mnt/cache/wuruixiao/models/Qwen2.5-7B-Instruct-ToolRL-grpo-cold",
    "memory": "/mnt/cache/wuruixiao/models/RL-MemoryAgent-7B",
    "code": "/mnt/cache/wuruixiao/models/ReasonFlux-Coder-7B",
    "r1": "/mnt/cache/wuruixiao/models/DeepSeek-R1-Distill-Qwen-7B",
}
EXPERT_TASK_MAP = {
    "tool": "ToolCall",
    "memory": "Memory",
    "code": "Code",
    "r1": "Code",  # R1's target skill is Code
}
PROTECTED_TASKS = {"ToolCall", "Memory", "Code"}

DATASET_DIR = Path("/mnt/cache/wuruixiao/users/lsc/AgentMerging/datasets")

EXCLUDE_REGEX = [
    r".*lm_head.*",
    r".*norm.*",
    r".*embed_tokens.*",
    r".*bias.*",
]

INCLUDE_REGEX = [
    r"^model\.layers\.[0-9]+\.(self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.weight$",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-iters", type=int, default=15, help="Number of iterations")
    p.add_argument("--samples-per-task", type=int, default=10, help="Calibration samples per task")
    p.add_argument("--max-length", type=int, default=4096, help="Max token length for calibration")
    p.add_argument("--lambda-sparse", type=float, default=0.0, help="L1 sparsity penalty")
    p.add_argument("--lambda-trust", type=float, default=0.1, help="Weight decay / trust region")
    p.add_argument("--lambda-harm", type=float, default=1.0, help="Harm penalty weight")
    p.add_argument("--max-abs-z", type=float, default=2.0, help="Max |z| per mode")
    p.add_argument("--max-abs-z-r1", type=float, default=0.01, help="Max |z| for R1 modes")
    p.add_argument("--lr", type=float, default=0.0, help="Learning rate (0=auto-calibrate)")
    p.add_argument("--target-max-step", type=float, default=0.15, help="Max z step per iter (for auto-lr)")
    p.add_argument("--update-rule", default="gd", choices=["gd", "qp"],
                    help="gd=gradient descent (stable), qp=analytical QP replacement (legacy)")
    p.add_argument("--normalize-by-delta-norm", type=int, default=1,
                    help="Normalize utility/harm by ||delta||^2 (natural gradient)")
    p.add_argument("--momentum", type=float, default=0.0, help="Momentum for z updates")
    p.add_argument("--device", default="cuda", help="Device for computation")
    p.add_argument("--dtype", default="bfloat16", help="Model dtype")
    p.add_argument("--output-dir", required=True, help="Output directory")
    p.add_argument("--experts", default="tool,memory,code,r1", help="Comma-separated experts to include")
    return p.parse_args()


import re


def get_merge_param_names(model):
    """Get parameter names that match include and exclude filters."""
    names = []
    for name in dict(model.named_parameters()).keys():
        if any(re.match(pat, name) for pat in EXCLUDE_REGEX):
            continue
        if INCLUDE_REGEX and not any(re.match(pat, name) for pat in INCLUDE_REGEX):
            continue
        names.append(name)
    return names


def load_datasets(samples_per_task: int) -> dict[str, list[dict]]:
    """Load calibration samples from each task dataset."""
    datasets = {}
    for task_file in ["ToolCall.json", "Memory.json", "Code.json"]:
        path = DATASET_DIR / task_file
        if not path.exists():
            print(f"  [warn] dataset not found: {path}")
            continue
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        task_name = task_file.replace(".json", "")
        datasets[task_name] = data[:samples_per_task]
        print(f"  [data] {task_name}: {len(datasets[task_name])} samples")
    return datasets


def tokenize_sample(tokenizer, sample: dict, max_length: int):
    """Tokenize a supervised sample into input_ids + labels."""
    messages = sample.get("messages", [])
    response = sample.get("response", "")
    if not messages or not response:
        return None

    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt = "\n".join(f"{m.get('role','user')}: {m.get('content','')}" for m in messages) + "\nassistant:"

    eos = tokenizer.eos_token or ""
    full = prompt + response + eos
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    full_ids = tokenizer(full, add_special_tokens=False).input_ids

    if len(full_ids) <= len(prompt_ids):
        return None

    resp_len = len(full_ids) - len(prompt_ids)

    # Left-truncate: keep response intact, trim prompt from left
    if len(full_ids) > max_length:
        cut = len(full_ids) - max_length
        full_ids = full_ids[cut:]
        new_prompt_len = max(len(prompt_ids) - cut, 0)
    else:
        new_prompt_len = len(prompt_ids)

    labels = [-100] * len(full_ids)
    resp_start = new_prompt_len
    for idx in range(resp_start, len(full_ids)):
        labels[idx] = full_ids[idx]

    if not any(l != -100 for l in labels):
        return None

    import torch
    return {
        "input_ids": torch.tensor([full_ids], dtype=torch.long),
        "labels": torch.tensor([labels], dtype=torch.long),
    }


def soft_threshold(u: float, lam: float) -> float:
    if u > lam:
        return u - lam
    if u < -lam:
        return u + lam
    return 0.0


def compute_expert_level_coefficients(z_dict: dict[str, dict[str, float]], expert_names: list[str]) -> dict[str, float]:
    """Aggregate per-mode z into mean expert-level coefficient."""
    result = {}
    for expert in expert_names:
        values = [v for (e, _), v in z_dict.items() if e == expert]
        if values:
            result[expert] = sum(abs(v) for v in values) / len(values)
        else:
            result[expert] = 0.0
    return result


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    expert_list = [e.strip() for e in args.experts.split(",")]
    print(f"=== Unified Iterative Utility/Harm QP ===")
    print(f"Experts: {expert_list}")
    print(f"Iterations: {args.num_iters}")
    print(f"Lambda sparse: {args.lambda_sparse}, Lambda trust: {args.lambda_trust}")
    print(f"Output: {output_dir}")
    print()

    import torch
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device)
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    model_dtype = dtype_map.get(args.dtype, torch.bfloat16)

    # ---- Load base model ----
    print("[1/4] Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=model_dtype, trust_remote_code=True)
    model.to(device)
    model.eval()

    param_names = get_merge_param_names(model)
    print(f"  Merge parameters: {len(param_names)}")

    # ---- Load task vector deltas ----
    print("[2/4] Computing task vector deltas...")
    deltas: dict[tuple[str, str], torch.Tensor] = {}  # (expert, param_name) -> delta tensor (CPU)

    base_state = {}
    for pname in param_names:
        parts = pname.split(".")
        param = model
        for part in parts:
            param = getattr(param, part)
        base_state[pname] = param.data.clone().cpu()

    for expert_name in expert_list:
        expert_path = EXPERTS[expert_name]
        print(f"  Loading expert: {expert_name} ({expert_path})")
        expert_model = AutoModelForCausalLM.from_pretrained(expert_path, torch_dtype=model_dtype, trust_remote_code=True)
        expert_state = dict(expert_model.named_parameters())
        for pname in param_names:
            delta = (expert_state[pname].data.cpu().float() - base_state[pname].float())
            deltas[(expert_name, pname)] = delta
        del expert_model
        torch.cuda.empty_cache()

    total_modes = len(deltas)
    print(f"  Total modes: {total_modes} ({len(expert_list)} experts x {len(param_names)} params)")

    # Pre-compute delta norms (for natural gradient normalization)
    delta_norm_sq: dict[tuple[str, str], float] = {}
    for key, delta in deltas.items():
        delta_norm_sq[key] = float((delta ** 2).sum().item())
    # Log per-expert aggregate norms
    for expert_name in expert_list:
        norms = [delta_norm_sq[(expert_name, pn)] for pn in param_names]
        total_norm = sum(norms)
        mean_norm = total_norm / len(norms)
        print(f"  {expert_name}: total ||delta||^2 = {total_norm:.2f}, mean per-param = {mean_norm:.4f}")

    # ---- Load calibration data ----
    print("[3/4] Loading calibration data...")
    datasets = load_datasets(args.samples_per_task)

    # ---- Iterative QP ----
    print(f"[4/4] Running {args.num_iters} QP iterations...")

    # Current z values: (expert, param_name) -> float
    z = {key: 0.0 for key in deltas}
    z_momentum = {key: 0.0 for key in deltas}
    effective_lr = args.lr if args.lr > 0 else 1.0  # will be auto-calibrated

    params_dict = dict(model.named_parameters())
    iteration_log = []

    for iteration in range(1, args.num_iters + 1):
        t0 = time.time()
        print(f"\n--- Iteration {iteration}/{args.num_iters} ---")

        # Step 1: Apply current z to model (in-place)
        with torch.no_grad():
            for pname in param_names:
                param = params_dict[pname]
                base_val = base_state[pname].to(device=device, dtype=model_dtype)
                delta_sum = torch.zeros_like(base_val, dtype=torch.float32)
                for expert_name in expert_list:
                    z_val = z[(expert_name, pname)]
                    if z_val != 0.0:
                        delta_sum += deltas[(expert_name, pname)].to(device) * z_val
                param.data.copy_((base_val.float() + delta_sum).to(model_dtype))

        # Step 2: Compute utility and harm via gradient projection
        utility = defaultdict(float)  # (expert, pname) -> float
        utility_count = defaultdict(int)
        harm = defaultdict(float)
        harm_count = defaultdict(int)

        # Disable grad on all params, then selectively enable
        for p in model.parameters():
            p.requires_grad_(False)
        for pname in param_names:
            params_dict[pname].requires_grad_(True)

        for task_name, samples in datasets.items():
            for sample in samples:
                batch = tokenize_sample(tokenizer, sample, args.max_length)
                if batch is None:
                    continue

                model.zero_grad(set_to_none=True)
                inputs = {k: v.to(device) for k, v in batch.items()}
                loss = model(**inputs).loss
                if not torch.isfinite(loss):
                    continue
                loss.backward()

                # Project gradients onto deltas
                for expert_name in expert_list:
                    owner_task = EXPERT_TASK_MAP[expert_name]
                    for pname in param_names:
                        grad = params_dict[pname].grad
                        if grad is None:
                            continue
                        key = (expert_name, pname)
                        delta = deltas[key]
                        proj = float((grad.detach().float().cpu() * delta).sum().item())

                        # Utility: -projection when task matches expert's owner task
                        if task_name == owner_task:
                            utility[key] += -proj
                            utility_count[key] += 1

                        # Harm: max(projection, 0) on protected tasks that are NOT the expert's own
                        if task_name in PROTECTED_TASKS and task_name != owner_task:
                            harm[key] += max(proj, 0.0)
                            harm_count[key] += 1

        # Step 3: Compute gradient and update z
        gradients = {}
        for key in deltas:
            u = utility[key] / max(utility_count[key], 1)
            h = harm[key] / max(harm_count[key], 1)
            h = max(h, 0.0)

            # Normalize by delta norm squared (natural gradient in z-space)
            if args.normalize_by_delta_norm and delta_norm_sq[key] > 1e-12:
                u = u / delta_norm_sq[key]
                h = h / delta_norm_sq[key]

            if args.update_rule == "gd":
                # Gradient descent: grad of [u*z - (lambda_harm*h + lambda_trust)*z^2/2 - lambda_sparse*|z|]
                grad = u - (args.lambda_harm * h + args.lambda_trust) * z[key]
                if args.lambda_sparse > 0:
                    grad -= args.lambda_sparse * (1.0 if z[key] > 0 else (-1.0 if z[key] < 0 else 0.0))
                gradients[key] = grad
            else:  # qp
                z_opt = soft_threshold(u, args.lambda_sparse) / max(args.lambda_harm * h + args.lambda_trust, 1e-12)
                gradients[key] = z_opt - z[key]

        # Auto-calibrate lr on first iteration
        if iteration == 1 and args.lr <= 0:
            max_grad = max(abs(g) for g in gradients.values()) if gradients else 1.0
            auto_lr = args.target_max_step / max(max_grad, 1e-12)
            print(f"  Auto-calibrated LR: {auto_lr:.6f} (max_grad={max_grad:.6f}, target_step={args.target_max_step})")
            effective_lr = auto_lr
        elif args.lr > 0:
            effective_lr = args.lr
        # else: reuse previous auto_lr

        # Apply gradient update
        new_z = {}
        for key in deltas:
            grad = gradients[key]

            # Momentum
            if args.momentum > 0:
                z_momentum[key] = args.momentum * z_momentum[key] + grad
                step = effective_lr * z_momentum[key]
            else:
                step = effective_lr * grad

            z_new = z[key] + step

            # Clip per-expert
            expert_name = key[0]
            max_z = args.max_abs_z_r1 if expert_name == "r1" else args.max_abs_z
            if max_z is not None:
                z_new = max(-max_z, min(max_z, z_new))

            new_z[key] = z_new

        z = new_z

        # Step 4: Log iteration stats
        elapsed = time.time() - t0
        expert_stats = {}
        for expert_name in expert_list:
            expert_z = [z[(expert_name, pname)] for pname in param_names]
            expert_stats[expert_name] = {
                "mean": sum(expert_z) / len(expert_z),
                "abs_mean": sum(abs(v) for v in expert_z) / len(expert_z),
                "min": min(expert_z),
                "max": max(expert_z),
                "nonzero": sum(1 for v in expert_z if abs(v) > 1e-8),
                "total": len(expert_z),
            }

        # Gradient stats per expert
        grad_stats = {}
        for expert_name in expert_list:
            expert_grads = [gradients[(expert_name, pname)] for pname in param_names]
            grad_stats[expert_name] = {
                "mean_grad": sum(expert_grads) / len(expert_grads),
                "max_abs_grad": max(abs(g) for g in expert_grads),
            }

        iter_record = {
            "iteration": iteration,
            "elapsed_sec": round(elapsed, 1),
            "effective_lr": effective_lr,
            "expert_stats": expert_stats,
            "grad_stats": grad_stats,
        }
        iteration_log.append(iter_record)

        # Print summary
        print(f"  Time: {elapsed:.1f}s  LR: {effective_lr:.6f}")
        print(f"  {'Expert':>10}  {'mean z':>10}  {'|z| mean':>10}  {'min z':>10}  {'max z':>10}  {'mean_grad':>10}  {'max|grad|':>10}")
        print(f"  {'-'*80}")
        for expert_name in expert_list:
            s = expert_stats[expert_name]
            g = grad_stats[expert_name]
            print(f"  {expert_name:>10}  {s['mean']:>10.6f}  {s['abs_mean']:>10.6f}  {s['min']:>10.6f}  {s['max']:>10.6f}  {g['mean_grad']:>10.6f}  {g['max_abs_grad']:>10.6f}")

    # ---- Restore base model ----
    with torch.no_grad():
        for pname in param_names:
            params_dict[pname].data.copy_(base_state[pname].to(device=device, dtype=model_dtype))

    # ---- Save results ----
    print("\n=== Saving results ===")

    # Save z coefficients
    z_output = {}
    for (expert_name, pname), val in sorted(z.items()):
        z_output.setdefault(expert_name, {})[pname] = val

    z_path = output_dir / "z_coefficients.json"
    z_path.write_text(json.dumps(z_output, indent=2) + "\n", encoding="utf-8")
    print(f"  z coefficients: {z_path}")

    # Save expert-level summary (mean coefficient per expert)
    summary = {}
    for expert_name in expert_list:
        expert_z = [z[(expert_name, pname)] for pname in param_names]
        summary[expert_name] = {
            "mean_z": sum(expert_z) / len(expert_z),
            "abs_mean_z": sum(abs(v) for v in expert_z) / len(expert_z),
            "median_z": sorted(expert_z)[len(expert_z) // 2],
            "min_z": min(expert_z),
            "max_z": max(expert_z),
            "nonzero_modes": sum(1 for v in expert_z if abs(v) > 1e-8),
            "total_modes": len(expert_z),
        }

    summary_path = output_dir / "expert_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"  Expert summary: {summary_path}")

    # Save iteration log
    log_path = output_dir / "iteration_log.json"
    log_path.write_text(json.dumps(iteration_log, indent=2) + "\n", encoding="utf-8")
    print(f"  Iteration log: {log_path}")

    # Save gate format (for bake_checkpoint compatibility)
    # Global coefficient format: average z per expert
    gate_coefficients = {expert: summary[expert]["mean_z"] for expert in expert_list}
    gate_path = output_dir / "gate_coefficients.json"
    gate_path.write_text(json.dumps(gate_coefficients, indent=2) + "\n", encoding="utf-8")
    print(f"  Gate coefficients: {gate_path}")

    # Per-parameter coefficient format (for opvec bake)
    param_coefficients = {}
    for pname in param_names:
        param_coefficients[pname] = {expert: z[(expert, pname)] for expert in expert_list}
    param_path = output_dir / "parameter_coefficients.json"
    param_path.write_text(json.dumps(param_coefficients, indent=2) + "\n", encoding="utf-8")
    print(f"  Parameter coefficients: {param_path}")

    # Config used
    config = {
        "num_iters": args.num_iters,
        "samples_per_task": args.samples_per_task,
        "max_length": args.max_length,
        "lambda_sparse": args.lambda_sparse,
        "lambda_trust": args.lambda_trust,
        "lambda_harm": args.lambda_harm,
        "max_abs_z": args.max_abs_z,
        "max_abs_z_r1": args.max_abs_z_r1,
        "lr": args.lr,
        "effective_lr": effective_lr,
        "target_max_step": args.target_max_step,
        "update_rule": args.update_rule,
        "normalize_by_delta_norm": args.normalize_by_delta_norm,
        "momentum": args.momentum,
        "experts": expert_list,
        "base_model": BASE_MODEL,
        "expert_paths": {e: EXPERTS[e] for e in expert_list},
        "expert_task_map": {e: EXPERT_TASK_MAP[e] for e in expert_list},
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    # Print final expert-level coefficients
    print(f"\n=== Final Expert Coefficients (mean z) ===")
    print(f"{'Expert':>10}  {'mean z':>10}  {'equivalent TA coeff':>20}")
    print(f"{'-'*45}")
    for expert_name in expert_list:
        mean_z = summary[expert_name]["mean_z"]
        print(f"{expert_name:>10}  {mean_z:>10.6f}  {mean_z:>20.4f}")

    print(f"\nCompare with grid-searched TA-0.75:")
    print(f"  tool=0.50, memory=0.75, code=0.75 (from manual sweep)")
    print(f"\nDone. Results in {output_dir}")


if __name__ == "__main__":
    main()
