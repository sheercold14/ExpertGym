#!/usr/bin/env python3
"""Loss-based optimal coefficient discovery.

Instead of using Fisher (quadratic approximation at theta_0),
directly evaluate actual loss on calibration data for various alpha configs.
This is the ground truth that Fisher tries to approximate.

Strategy:
  1. Evaluate base model loss on all tasks
  2. For each expert individually, sweep alpha and measure loss on all tasks
  3. For combined configs, evaluate loss on all tasks
  4. Build a loss-based M,r and solve QP (using actual Hessian from loss samples)
  5. Also do grid search over alpha to find empirical optimum

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/analysis/loss_based_coefficients.py \
        --samples 20 --output-dir /tmp/shared-storage/ExpertGym/analysis/loss_based_v1
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import time
from pathlib import Path

import numpy as np

BASE_MODEL = "/mnt/cache/wuruixiao/models/Qwen2.5-7b-instruct"
EXPERTS = {
    "tool": "/mnt/cache/wuruixiao/models/Qwen2.5-7B-Instruct-ToolRL-grpo-cold",
    "memory": "/mnt/cache/wuruixiao/models/RL-MemoryAgent-7B",
    "code": "/mnt/cache/wuruixiao/models/ReasonFlux-Coder-7B",
}
EXPERT_TASK_MAP = {"tool": "ToolCall", "memory": "Memory", "code": "Code"}
DATASET_DIR = Path("/mnt/cache/wuruixiao/users/lsc/AgentMerging/datasets")

INCLUDE_REGEX = [
    r"^model\.layers\.[0-9]+\.(self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.weight$",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--samples", type=int, default=20, help="Calibration samples per task")
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def get_merge_param_names(model):
    names = []
    for name in dict(model.named_parameters()).keys():
        if INCLUDE_REGEX and not any(re.match(pat, name) for pat in INCLUDE_REGEX):
            continue
        names.append(name)
    return names


def load_datasets(samples_per_task):
    datasets = {}
    for task_file in ["ToolCall.json", "Memory.json", "Code.json"]:
        path = DATASET_DIR / task_file
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        task_name = task_file.replace(".json", "")
        datasets[task_name] = data[:samples_per_task]
        print(f"  [data] {task_name}: {len(datasets[task_name])} samples")
    return datasets


def tokenize_sample(tokenizer, sample, max_length):
    """Left-truncation to preserve response tokens."""
    import torch
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
    if len(full_ids) > max_length:
        cut = len(full_ids) - max_length
        full_ids = full_ids[cut:]
        new_prompt_len = max(len(prompt_ids) - cut, 0)
    else:
        new_prompt_len = len(prompt_ids)
    labels = [-100] * len(full_ids)
    for idx in range(new_prompt_len, len(full_ids)):
        labels[idx] = full_ids[idx]
    if not any(l != -100 for l in labels):
        return None
    return {
        "input_ids": torch.tensor([full_ids], dtype=torch.long),
        "labels": torch.tensor([labels], dtype=torch.long),
    }


def evaluate_loss(model, tokenizer, samples, max_length, device):
    """Evaluate average loss on samples."""
    import torch
    total_loss = 0.0
    n_valid = 0
    with torch.no_grad():
        for sample in samples:
            batch = tokenize_sample(tokenizer, sample, max_length)
            if batch is None:
                continue
            inputs = {k: v.to(device) for k, v in batch.items()}
            output = model(**inputs)
            if torch.isfinite(output.loss):
                total_loss += output.loss.item()
                n_valid += 1
    return total_loss / max(n_valid, 1), n_valid


def apply_merge(model, base_state, tau, alpha, param_names):
    """Apply merge: theta = base + sum(alpha_i * tau_i) in-place."""
    import torch
    params = dict(model.named_parameters())
    with torch.no_grad():
        for pname in param_names:
            merged = base_state[pname].clone()
            for expert_name, coeff in alpha.items():
                if abs(coeff) > 1e-10:
                    merged += coeff * tau[expert_name][pname]
            params[pname].data.copy_(merged.to(params[pname].device))


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    expert_list = ["tool", "memory", "code"]

    print("=" * 70)
    print("Loss-Based Coefficient Discovery")
    print("=" * 70)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device)
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    model_dtype = dtype_map.get(args.dtype, torch.bfloat16)

    # Step 1: Load base model
    print("\n[1/5] Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=model_dtype, trust_remote_code=True)
    model.to(device)
    model.eval()

    param_names = get_merge_param_names(model)
    print(f"  Merge parameters: {len(param_names)}")

    # Step 2: Compute task vectors
    print("\n[2/5] Computing task vectors...")
    base_state = {}
    for pname in param_names:
        parts = pname.split(".")
        param = model
        for part in parts:
            param = getattr(param, part)
        base_state[pname] = param.data.clone().cpu().float()

    tau = {}
    for expert_name in expert_list:
        expert_path = EXPERTS[expert_name]
        print(f"  Loading expert: {expert_name} ({expert_path})")
        expert_model = AutoModelForCausalLM.from_pretrained(expert_path, torch_dtype=model_dtype, trust_remote_code=True)
        expert_state = dict(expert_model.named_parameters())
        tau[expert_name] = {}
        for pname in param_names:
            tau[expert_name][pname] = expert_state[pname].data.cpu().float() - base_state[pname]
        del expert_model
        torch.cuda.empty_cache()

    # Report norms
    for e in expert_list:
        norm_sq = sum(float((tau[e][p] ** 2).sum()) for p in param_names)
        print(f"  {e}: ||tau|| = {norm_sq**0.5:.4f}")

    # Step 3: Load calibration data
    print("\n[3/5] Loading calibration data...")
    datasets = load_datasets(args.samples)

    # Step 4: Evaluate base model loss
    print("\n[4/5] Evaluating base model loss on all tasks...")
    base_losses = {}
    for task_name, samples in datasets.items():
        # Restore base model params
        apply_merge(model, base_state, tau, {}, param_names)
        loss, n = evaluate_loss(model, tokenizer, samples, args.max_length, device)
        base_losses[task_name] = loss
        print(f"  Base on {task_name}: loss={loss:.4f} ({n} valid)")

    # Step 5: Sweep alpha configs and evaluate
    print("\n[5/5] Sweeping alpha configurations...")

    # Define sweep configs
    alpha_values = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
    results = []

    # A) Individual expert sweeps
    print("\n  --- Individual Expert Sweeps ---")
    for expert_name in expert_list:
        print(f"\n  Expert: {expert_name}")
        for alpha_val in alpha_values:
            alpha_config = {e: 0.0 for e in expert_list}
            alpha_config[expert_name] = alpha_val

            apply_merge(model, base_state, tau, alpha_config, param_names)

            losses = {}
            for task_name, samples in datasets.items():
                loss, n = evaluate_loss(model, tokenizer, samples, args.max_length, device)
                losses[task_name] = loss

            result = {
                "config": f"{expert_name}={alpha_val}",
                "alpha": alpha_config,
                "losses": losses,
                "delta_losses": {t: losses[t] - base_losses[t] for t in losses},
            }
            results.append(result)
            dl = result["delta_losses"]
            print(f"    alpha={alpha_val:.2f}: " + ", ".join(f"{t}={dl[t]:+.4f}" for t in sorted(dl)))

    # B) Combined configs (grid over coarser alpha)
    print("\n  --- Combined Configs ---")
    combined_alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    n_configs = len(combined_alphas) ** 3
    print(f"  Total combined configs: {n_configs}")

    best_config = None
    best_avg_loss = float("inf")

    for i, a_tool in enumerate(combined_alphas):
        for a_mem in combined_alphas:
            for a_code in combined_alphas:
                if a_tool == 0 and a_mem == 0 and a_code == 0:
                    # Already have base model losses
                    losses = dict(base_losses)
                else:
                    alpha_config = {"tool": a_tool, "memory": a_mem, "code": a_code}
                    apply_merge(model, base_state, tau, alpha_config, param_names)
                    losses = {}
                    for task_name, samples in datasets.items():
                        loss, n = evaluate_loss(model, tokenizer, samples, args.max_length, device)
                        losses[task_name] = loss

                alpha_config = {"tool": a_tool, "memory": a_mem, "code": a_code}
                result = {
                    "config": f"tool={a_tool},mem={a_mem},code={a_code}",
                    "alpha": alpha_config,
                    "losses": losses,
                    "delta_losses": {t: losses[t] - base_losses[t] for t in losses},
                }
                results.append(result)

                avg_loss = sum(losses.values()) / len(losses)
                if avg_loss < best_avg_loss:
                    best_avg_loss = avg_loss
                    best_config = result

    print(f"\n  Best combined config (avg loss): {best_config['config']}")
    print(f"    Avg loss: {best_avg_loss:.4f}")
    print(f"    Losses: {best_config['losses']}")

    # Find best per-task configs
    print("\n  --- Best Per-Task Configs ---")
    for task in datasets:
        best_task = min(results, key=lambda r: r["losses"].get(task, float("inf")))
        print(f"  Best for {task}: {best_task['config']} -> loss={best_task['losses'][task]:.4f}")

    # Find best weighted-sum configs
    print("\n  --- Best Weighted-Sum Configs ---")
    # Equal weight
    best_eq = min(results, key=lambda r: sum(r["losses"].values()))
    print(f"  Equal weight: {best_eq['config']} -> losses={best_eq['losses']}")

    # Build loss-based quadratic model from individual sweeps
    print("\n  --- Loss-Based QP ---")
    # For each expert and task, fit L(alpha) = a + b*alpha + c*alpha^2
    # Then the optimal alpha minimizes sum of weighted losses
    individual_fits = {}
    for expert_name in expert_list:
        individual_fits[expert_name] = {}
        expert_results = [r for r in results if r["config"].startswith(f"{expert_name}=")]
        for task in datasets:
            alphas = []
            losses = []
            for r in expert_results:
                alphas.append(r["alpha"][expert_name])
                losses.append(r["losses"][task])
            alphas = np.array(alphas)
            losses = np.array(losses)
            # Fit quadratic: L = a + b*alpha + c*alpha^2
            A = np.column_stack([np.ones_like(alphas), alphas, alphas**2])
            coeffs, _, _, _ = np.linalg.lstsq(A, losses, rcond=None)
            individual_fits[expert_name][task] = {
                "a": float(coeffs[0]),
                "b": float(coeffs[1]),
                "c": float(coeffs[2]),
                "r2": float(1 - np.sum((losses - A @ coeffs)**2) / np.sum((losses - losses.mean())**2)) if np.var(losses) > 0 else 0,
            }
            print(f"    {expert_name} on {task}: L = {coeffs[0]:.4f} + {coeffs[1]:+.4f}*a + {coeffs[2]:+.4f}*a^2  (R^2={individual_fits[expert_name][task]['r2']:.4f})")

    # From individual fits, compute one-expert optimal alpha per expert
    print("\n  --- One-Expert Optimal (from loss fits) ---")
    for expert_name in expert_list:
        # sum_task [b_task + 2*c_task*alpha] = 0
        # alpha* = -sum(b) / (2*sum(c))
        b_sum = sum(individual_fits[expert_name][t]["b"] for t in datasets)
        c_sum = sum(individual_fits[expert_name][t]["c"] for t in datasets)
        if abs(c_sum) > 1e-12:
            alpha_opt = -b_sum / (2 * c_sum)
        else:
            alpha_opt = 0.0
        alpha_opt_clipped = max(0, min(1.25, alpha_opt))
        # Also compute per-task optimal
        owner_task = EXPERT_TASK_MAP[expert_name]
        b_own = individual_fits[expert_name][owner_task]["b"]
        c_own = individual_fits[expert_name][owner_task]["c"]
        alpha_own = -b_own / (2 * c_own) if abs(c_own) > 1e-12 else 0.0
        print(f"  {expert_name}: alpha_opt_all={alpha_opt:.4f} (clipped={alpha_opt_clipped:.4f}), alpha_opt_own_task={alpha_own:.4f}")

    # Save results
    output = {
        "base_losses": base_losses,
        "sweep_results": results,
        "individual_fits": individual_fits,
        "best_config": best_config,
        "config": {
            "samples": args.samples,
            "max_length": args.max_length,
            "experts": expert_list,
        },
    }
    output_path = output_dir / "loss_based_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
