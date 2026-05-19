#!/usr/bin/env python3
"""Iterative Fisher-Regression QP: compute Fisher at the merged model.

Unlike the original FRS-QP (Fisher at base θ₀), this script iteratively:
  1. Apply current α to get θ_merge = θ₀ + Σ α_i τ_i
  2. Compute Fisher at θ_merge (captures cross-expert interactions)
  3. Solve QP → new α
  4. Repeat until convergence

This captures non-linear cross-expert interactions that Fisher at θ₀ misses.
No model checkpointing needed — all done in-place in memory.

Usage:
    CUDA_VISIBLE_DEVICES=2 python scripts/analysis/iterative_fisher_qp.py \
        --fisher-samples 20 --max-iters 5 \
        --output-dir /tmp/shared-storage/ExpertGym/analysis/iterative_fisher_v1
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
    p.add_argument("--fisher-samples", type=int, default=20)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--rho", type=float, default=0.1, help="Base retention weight")
    p.add_argument("--lambda-alpha", type=float, default=1e-10)
    p.add_argument("--alpha-max", type=float, default=1.25)
    p.add_argument("--task-weights", default="1,1,1")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--experts", default="tool,memory,code")
    p.add_argument("--max-iters", type=int, default=5, help="Max iterations")
    p.add_argument("--init-alpha", default="0.75,0.75,0.75", help="Initial alpha (comma-separated)")
    p.add_argument("--tol", type=float, default=0.01, help="Convergence tolerance on alpha change")
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
            print(f"  [warn] dataset not found: {path}")
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


def apply_merge_inplace(model, base_state, tau, alpha, param_names):
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


def estimate_fisher_and_reduce(model, tokenizer, samples, max_length, param_names, device, tau, expert_list):
    """Estimate diagonal Fisher at current model weights and reduce to quadratic forms.

    Returns: quad_forms {(ei, ej) -> tau_i^T F tau_j}, total_fisher
    """
    import torch

    quad_forms = {(ei, ej): 0.0 for ei in expert_list for ej in expert_list}
    total_fisher = 0.0

    params_dict = dict(model.named_parameters())
    for p in model.parameters():
        p.requires_grad_(False)
    for name in param_names:
        params_dict[name].requires_grad_(True)

    n_valid = 0
    for sample in samples:
        batch = tokenize_sample(tokenizer, sample, max_length)
        if batch is None:
            continue
        model.zero_grad(set_to_none=True)
        inputs = {k: v.to(device) for k, v in batch.items()}
        loss = model(**inputs).loss
        if not torch.isfinite(loss):
            continue
        loss.backward()

        for name in param_names:
            grad = params_dict[name].grad
            if grad is None:
                continue
            f_p = (grad.detach().float() ** 2).cpu()
            total_fisher += float(f_p.sum().item())
            for ei in expert_list:
                fi_tau_i = f_p * tau[ei][name]
                for ej in expert_list:
                    quad_forms[(ei, ej)] += float((fi_tau_i * tau[ej][name]).sum().item())

        n_valid += 1

    if n_valid > 0:
        for key in quad_forms:
            quad_forms[key] /= n_valid
        total_fisher /= n_valid

    print(f"    {n_valid} valid samples, total Fisher: {total_fisher:.4f}")
    return quad_forms, total_fisher


def build_and_solve_qp(quad_forms_per_task, expert_list, task_weights, rho, lambda_alpha, alpha_max):
    """Build M, r from per-task quadratic forms and solve box-constrained QP."""
    from scipy.optimize import minimize as scipy_minimize

    m = len(expert_list)
    M = np.zeros((m, m))
    r = np.zeros(m)

    task_to_owner = {EXPERT_TASK_MAP[e]: e for e in expert_list}

    # Skill alignment: sum_c w_c * tau_i^T F_c tau_j  (M)
    #                  sum_c w_c * tau_c^T F_c tau_i    (r)
    for task_name, qf in quad_forms_per_task.items():
        owner = task_to_owner.get(task_name)
        if owner is None:
            continue
        w_c = task_weights.get(task_name, 1.0)
        for i in range(m):
            for j in range(m):
                M[i, j] += w_c * qf[(expert_list[i], expert_list[j])]
            r[i] += w_c * qf[(owner, expert_list[i])]

    # Base retention: rho * avg_task(B^T F_task B)
    num_tasks = len(quad_forms_per_task)
    if rho > 0 and num_tasks > 0:
        for task_name, qf in quad_forms_per_task.items():
            for i in range(m):
                for j in range(m):
                    M[i, j] += rho * qf[(expert_list[i], expert_list[j])] / num_tasks

    # Tikhonov
    M += lambda_alpha * np.eye(m)

    print(f"  M condition: {np.linalg.cond(M):.2f}")

    # Unconstrained solution
    try:
        alpha_unc = np.linalg.solve(M, r)
        print(f"  Unconstrained: {dict(zip(expert_list, [f'{v:.4f}' for v in alpha_unc]))}")
    except np.linalg.LinAlgError:
        alpha_unc = np.zeros(m)

    # Constrained
    def obj(x):
        return 0.5 * x @ M @ x - r @ x

    def grad(x):
        return M @ x - r

    bounds = [(0.0, alpha_max)] * m
    try:
        x0 = np.clip(alpha_unc, 0.0, alpha_max)
    except Exception:
        x0 = np.ones(m) * 0.5

    res = scipy_minimize(obj, x0, jac=grad, bounds=bounds, method="L-BFGS-B",
                         options={"ftol": 1e-30, "gtol": 1e-15, "maxiter": 1000})
    print(f"  QP converged: {res.success}")

    return res.x, res.fun, M, r, alpha_unc


def one_expert_shrinkage(quad_forms_per_task, expert_list):
    """Compute one-expert shrinkage alpha for each expert."""
    results = {}
    for e in expert_list:
        owner_task = EXPERT_TASK_MAP[e]
        A_e = quad_forms_per_task[owner_task][(e, e)]
        protected_tasks = [t for t in quad_forms_per_task if t != owner_task]
        B_e = sum(quad_forms_per_task[t][(e, e)] for t in protected_tasks) / max(len(protected_tasks), 1)
        alpha_shrinkage = A_e / (A_e + B_e) if (A_e + B_e) > 0 else 0.0
        results[e] = {
            "A_e": A_e,
            "B_e": B_e,
            "B/A": B_e / A_e if A_e > 0 else float("inf"),
            "alpha_shrinkage": alpha_shrinkage,
        }
    return results


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    expert_list = [e.strip() for e in args.experts.split(",")]
    task_weights_raw = [float(w) for w in args.task_weights.split(",")]
    task_weights = {EXPERT_TASK_MAP[e]: w for e, w in zip(expert_list, task_weights_raw)}
    init_alpha = [float(a) for a in args.init_alpha.split(",")]

    print("=" * 70)
    print("Iterative Fisher-Regression QP")
    print("=" * 70)
    print(f"Experts: {expert_list}")
    print(f"Task weights: {task_weights}")
    print(f"rho: {args.rho}, lambda_alpha: {args.lambda_alpha}")
    print(f"Init alpha: {dict(zip(expert_list, init_alpha))}")
    print(f"Max iters: {args.max_iters}, tol: {args.tol}")
    print()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device)
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    model_dtype = dtype_map.get(args.dtype, torch.bfloat16)

    # Step 1: Load base model
    print("[1/4] Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=model_dtype, trust_remote_code=True)
    model.to(device)
    model.eval()

    param_names = get_merge_param_names(model)
    print(f"  Merge parameters: {len(param_names)}")

    # Step 2: Compute task vectors & save base state
    print("[2/4] Computing task vectors...")
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

    for e in expert_list:
        norm_sq = sum(float((tau[e][p] ** 2).sum()) for p in param_names)
        print(f"  {e}: ||tau|| = {norm_sq**0.5:.4f}")

    # Step 3: Load calibration data
    print("[3/4] Loading calibration data...")
    datasets = load_datasets(args.fisher_samples)

    # Step 4: Iterative Fisher-QP
    print("\n[4/4] Iterative Fisher-QP loop")
    print("=" * 70)

    current_alpha = {e: a for e, a in zip(expert_list, init_alpha)}
    iteration_history = []

    # Also compute Fisher at base model (iteration 0) for comparison
    for iteration in range(args.max_iters + 1):
        if iteration == 0:
            # Iteration 0: Fisher at base model (for comparison)
            print(f"\n--- Iteration 0: Fisher at BASE model (θ₀) ---")
            apply_merge_inplace(model, base_state, tau, {e: 0.0 for e in expert_list}, param_names)
        else:
            print(f"\n--- Iteration {iteration}: Fisher at merged model (α = {current_alpha}) ---")
            apply_merge_inplace(model, base_state, tau, current_alpha, param_names)

        # Compute Fisher at current model weights
        quad_forms_per_task = {}
        fisher_totals = {}
        for task_name, samples in datasets.items():
            print(f"  Task: {task_name}")
            t0 = time.time()
            qf, total_f = estimate_fisher_and_reduce(
                model, tokenizer, samples, args.max_length, param_names, device, tau, expert_list
            )
            quad_forms_per_task[task_name] = qf
            fisher_totals[task_name] = total_f
            print(f"    Time: {time.time() - t0:.1f}s")

        # One-expert shrinkage
        shrinkage = one_expert_shrinkage(quad_forms_per_task, expert_list)
        print(f"  One-expert shrinkage:")
        for e in expert_list:
            s = shrinkage[e]
            print(f"    {e}: A={s['A_e']:.6e}, B={s['B_e']:.6e}, B/A={s['B/A']:.4f}, α*={s['alpha_shrinkage']:.4f}")

        # Solve QP
        alpha_star, obj_star, M, r_vec, alpha_unc = build_and_solve_qp(
            quad_forms_per_task, expert_list, task_weights,
            args.rho, args.lambda_alpha, args.alpha_max
        )

        new_alpha = {e: float(alpha_star[i]) for i, e in enumerate(expert_list)}
        print(f"  QP solution: {new_alpha}")
        print(f"  Objective: {obj_star:.8e}")

        # Record
        record = {
            "iteration": iteration,
            "eval_point": {e: 0.0 for e in expert_list} if iteration == 0 else dict(current_alpha),
            "fisher_totals": dict(fisher_totals),
            "quad_forms_per_task": {
                t: {f"{k[0]},{k[1]}": v for k, v in qf.items()}
                for t, qf in quad_forms_per_task.items()
            },
            "one_expert_shrinkage": shrinkage,
            "alpha_qp": new_alpha,
            "alpha_unconstrained": {e: float(alpha_unc[i]) for i, e in enumerate(expert_list)},
            "M_matrix": M.tolist(),
            "r_vector": r_vec.tolist(),
            "M_condition": float(np.linalg.cond(M)),
            "objective": float(obj_star),
        }
        iteration_history.append(record)

        if iteration == 0:
            # Use init_alpha for iteration 1 (don't update from base-model QP)
            print(f"  (Base model reference — will use init_alpha for iteration 1)")
            continue

        # Check convergence
        delta = max(abs(new_alpha[e] - current_alpha[e]) for e in expert_list)
        print(f"  Max alpha change: {delta:.6f} (tol={args.tol})")

        if delta < args.tol and iteration > 1:
            print(f"\n  Converged at iteration {iteration}!")
            current_alpha = new_alpha
            break

        current_alpha = new_alpha

    # Restore base model
    apply_merge_inplace(model, base_state, tau, {e: 0.0 for e in expert_list}, param_names)

    # Summary
    print("\n" + "=" * 70)
    print("ITERATION SUMMARY")
    print("=" * 70)
    print(f"{'Iter':>4}  {'Eval Point':>35}  {'QP Solution':>35}")
    print("-" * 80)
    for rec in iteration_history:
        ep = rec["eval_point"]
        qp = rec["alpha_qp"]
        ep_str = ", ".join(f"{e}={ep[e]:.3f}" for e in expert_list)
        qp_str = ", ".join(f"{e}={qp[e]:.3f}" for e in expert_list)
        print(f"{rec['iteration']:>4}  {ep_str:>35}  {qp_str:>35}")

    print(f"\nFinal α: {current_alpha}")

    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"{'Method':>30}  {'tool':>8}  {'memory':>8}  {'code':>8}")
    print("-" * 60)
    base_qp = iteration_history[0]["alpha_qp"]
    print(f"{'Fisher@θ₀ (original FRS-QP)':>30}  {base_qp['tool']:8.4f}  {base_qp['memory']:8.4f}  {base_qp['code']:8.4f}")
    print(f"{'Fisher@θ_merge (iterative)':>30}  {current_alpha['tool']:8.4f}  {current_alpha['memory']:8.4f}  {current_alpha['code']:8.4f}")
    print(f"{'TA-0.75 (baseline)':>30}  {'0.7500':>8}  {'0.7500':>8}  {'0.7500':>8}")

    # Save
    output = {
        "final_alpha": current_alpha,
        "iteration_history": iteration_history,
        "config": {
            "fisher_samples": args.fisher_samples,
            "max_length": args.max_length,
            "rho": args.rho,
            "lambda_alpha": args.lambda_alpha,
            "alpha_max": args.alpha_max,
            "init_alpha": dict(zip(expert_list, init_alpha)),
            "tol": args.tol,
            "max_iters": args.max_iters,
            "task_weights": task_weights,
            "experts": expert_list,
        },
    }
    output_path = output_dir / "iterative_fisher_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
