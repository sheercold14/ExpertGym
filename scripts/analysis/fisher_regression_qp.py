#!/usr/bin/env python3
"""Fisher-Regression Shrinkage QP for compatible expert coefficients.

Memory-optimized: computes Fisher per task and immediately reduces to
scalar quadratic forms, never storing full Fisher matrices in memory.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/analysis/fisher_regression_qp.py \
        --fisher-samples 20 \
        --output-dir /tmp/shared-storage/ExpertGym/analysis/frs_qp_v1
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

BASE_MODEL = "/mnt/cache/wuruixiao/models/Qwen2.5-7b-instruct"
EXPERTS = {
    "tool": "/mnt/cache/wuruixiao/models/Qwen2.5-7B-Instruct-ToolRL-grpo-cold",
    "memory": "/mnt/cache/wuruixiao/models/RL-MemoryAgent-7B",
    "code": "/mnt/cache/wuruixiao/models/ReasonFlux-Coder-7B",
}
EXPERT_TASK_MAP = {
    "tool": "ToolCall",
    "memory": "Memory",
    "code": "Code",
}
DATASET_DIR = Path("/mnt/cache/wuruixiao/users/lsc/AgentMerging/datasets")

INCLUDE_REGEX = [
    r"^model\.layers\.[0-9]+\.(self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.weight$",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fisher-samples", type=int, default=20)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--rho", type=float, default=0.1, help="Base retention weight")
    p.add_argument("--lambda-alpha", type=float, default=1e-4)
    p.add_argument("--alpha-max", type=float, default=1.25)
    p.add_argument("--eta", type=float, default=0.01, help="Objective tolerance for interval")
    p.add_argument("--task-weights", default="1,1,1")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--experts", default="tool,memory,code")
    return p.parse_args()


def get_merge_param_names(model):
    names = []
    for name in dict(model.named_parameters()).keys():
        if INCLUDE_REGEX and not any(re.match(pat, name) for pat in INCLUDE_REGEX):
            continue
        names.append(name)
    return names


def load_datasets(samples_per_task: int) -> dict[str, list[dict]]:
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
    import torch
    return {
        "input_ids": torch.tensor([full_ids], dtype=torch.long),
        "labels": torch.tensor([labels], dtype=torch.long),
    }


def estimate_fisher_and_reduce(model, tokenizer, samples, max_length, param_names, device, tau_dict, expert_list):
    """Estimate diagonal Fisher and immediately reduce to quadratic forms.

    Returns dict: {(expert_i, expert_j) -> tau_i^T F tau_j} for all pairs i,j.
    Also returns total Fisher for logging. Does NOT store the full Fisher matrix.
    """
    import torch

    m = len(expert_list)
    # Accumulators for tau_i^T F tau_j
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

        # Accumulate Fisher quadratic forms immediately
        for name in param_names:
            grad = params_dict[name].grad
            if grad is None:
                continue
            f_p = (grad.detach().float() ** 2).cpu()  # diagonal Fisher for this param
            total_fisher += float(f_p.sum().item())

            for ei in expert_list:
                fi_tau_i = f_p * tau_dict[ei][name]  # F[p] * tau_i[p]
                for ej in expert_list:
                    # tau_i^T F tau_j = sum_p F[p] * tau_i[p] * tau_j[p]
                    quad_forms[(ei, ej)] += float((fi_tau_i * tau_dict[ej][name]).sum().item())

        n_valid += 1

    if n_valid > 0:
        for key in quad_forms:
            quad_forms[key] /= n_valid
        total_fisher /= n_valid

    print(f"    {n_valid} valid samples, total Fisher: {total_fisher:.4f}")
    return quad_forms, total_fisher


def solve_box_qp(M, r, lower=0.0, upper=1.25):
    from scipy.optimize import minimize as scipy_minimize
    m = len(r)
    def obj(x):
        return 0.5 * x @ M @ x - r @ x
    def grad(x):
        return M @ x - r
    bounds = [(lower, upper)] * m
    # Try unconstrained first to get a good starting point
    try:
        x0 = np.clip(np.linalg.solve(M, r), lower, upper)
    except np.linalg.LinAlgError:
        x0 = np.ones(m) * 0.5
    res = scipy_minimize(obj, x0, jac=grad, bounds=bounds, method="L-BFGS-B",
                         options={"ftol": 1e-30, "gtol": 1e-15, "maxiter": 1000})
    print(f"  L-BFGS-B converged: {res.success}, nit={res.nit}, message={res.message}")
    return res.x, res.fun


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    expert_list = [e.strip() for e in args.experts.split(",")]
    task_weights_raw = [float(w) for w in args.task_weights.split(",")]
    task_weights = {EXPERT_TASK_MAP[e]: w for e, w in zip(expert_list, task_weights_raw)}

    m = len(expert_list)

    print("=" * 60)
    print("Fisher-Regression Shrinkage QP (CAEC) [memory-optimized]")
    print("=" * 60)
    print(f"Experts: {expert_list}")
    print(f"Task weights: {task_weights}")
    print(f"rho: {args.rho}, lambda_alpha: {args.lambda_alpha}, alpha_max: {args.alpha_max}")
    print(f"Fisher samples: {args.fisher_samples}, max_length: {args.max_length}")
    print(f"Output: {output_dir}")
    print()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device)
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    model_dtype = dtype_map.get(args.dtype, torch.bfloat16)

    # ---- Step 1: Load base model ----
    print("[1/5] Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=model_dtype, trust_remote_code=True)
    model.to(device)
    model.eval()

    param_names = get_merge_param_names(model)
    print(f"  Merge parameters: {len(param_names)}")

    # ---- Step 2: Compute task vectors ----
    print("[2/5] Computing task vectors...")
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

    # Free base_state now (we don't need it anymore since model is on GPU)
    del base_state
    gc.collect()

    # Report delta norms
    for expert_name in expert_list:
        total_norm_sq = sum(float((tau[expert_name][p] ** 2).sum()) for p in param_names)
        print(f"  {expert_name}: ||tau||^2 = {total_norm_sq:.4f}, ||tau|| = {total_norm_sq**0.5:.4f}")

    # ---- Step 3: Load calibration data ----
    print("[3/5] Loading calibration data...")
    datasets = load_datasets(args.fisher_samples)

    # ---- Step 4: Estimate Fisher quadratic forms per task ----
    print("[4/5] Estimating Fisher quadratic forms per task...")
    # For each task, compute tau_i^T F_task tau_j for all expert pairs (i,j)
    # This avoids storing full Fisher matrices in memory.
    quad_forms_per_task = {}  # task_name -> {(ei, ej) -> float}
    fisher_totals = {}

    for task_name, samples in datasets.items():
        print(f"  Task: {task_name}")
        t0 = time.time()
        qf, total_f = estimate_fisher_and_reduce(
            model, tokenizer, samples, args.max_length, param_names, device, tau, expert_list
        )
        quad_forms_per_task[task_name] = qf
        fisher_totals[task_name] = total_f
        elapsed = time.time() - t0
        print(f"    Time: {elapsed:.1f}s")

    # ---- Step 5: Build and solve QP ----
    print("\n[5/5] Building and solving QP...")

    # One-expert shrinkage
    print("\n  === One-Expert Shrinkage Analysis ===")
    one_expert_results = {}
    for expert_name in expert_list:
        owner_task = EXPERT_TASK_MAP[expert_name]

        # A_e = tau_e^T F_owner tau_e
        A_e = quad_forms_per_task[owner_task][(expert_name, expert_name)]

        # B_e = tau_e^T F_prot tau_e (average of non-owner tasks)
        protected_tasks = [t for t in datasets if t != owner_task]
        B_e = sum(quad_forms_per_task[pt][(expert_name, expert_name)] for pt in protected_tasks) / max(len(protected_tasks), 1)

        # Base retention (average Fisher across all tasks)
        F_base_form = sum(quad_forms_per_task[t][(expert_name, expert_name)] for t in datasets) / len(datasets)

        alpha_shrinkage = A_e / (A_e + B_e) if (A_e + B_e) > 0 else 0.0
        alpha_with_base = A_e / (A_e + B_e + args.rho * F_base_form) if (A_e + B_e + args.rho * F_base_form) > 0 else 0.0

        one_expert_results[expert_name] = {
            "A_e": A_e,
            "B_e": B_e,
            "F_base": F_base_form,
            "B_e/A_e": B_e / A_e if A_e > 0 else float("inf"),
            "alpha_shrinkage": alpha_shrinkage,
            "alpha_with_base_retention": alpha_with_base,
        }

        print(f"  {expert_name:>8}: A_e={A_e:.6f}, B_e={B_e:.6f}, B/A={B_e/A_e if A_e > 0 else float('inf'):.4f}")
        print(f"           alpha*={alpha_shrinkage:.4f} (no base), alpha*={alpha_with_base:.4f} (with rho={args.rho})")

    # Multi-expert QP
    print("\n  === Multi-Expert QP ===")

    M = np.zeros((m, m))
    r = np.zeros(m)

    for c_idx, expert_c in enumerate(expert_list):
        task_c = EXPERT_TASK_MAP[expert_c]
        if task_c not in quad_forms_per_task:
            continue
        w_c = task_weights.get(task_c, 1.0)
        qf = quad_forms_per_task[task_c]

        for i in range(m):
            for j in range(m):
                M[i, j] += w_c * qf[(expert_list[i], expert_list[j])]
            r[i] += w_c * qf[(expert_list[i], expert_c)]

    # Base retention: rho * sum_task B^T F_task B / num_tasks
    for task_name in datasets:
        qf = quad_forms_per_task[task_name]
        for i in range(m):
            for j in range(m):
                M[i, j] += args.rho * qf[(expert_list[i], expert_list[j])] / len(datasets)

    # Tikhonov
    M += args.lambda_alpha * np.eye(m)

    print(f"  M matrix:\n    {np.array2string(M, precision=6, separator=', ')}")
    print(f"  r vector: {np.array2string(r, precision=6)}")
    print(f"  M condition: {np.linalg.cond(M):.2f}")

    # Unconstrained
    try:
        alpha_unconstrained = np.linalg.solve(M, r)
        print(f"  Unconstrained: {dict(zip(expert_list, [f'{v:.4f}' for v in alpha_unconstrained]))}")
    except np.linalg.LinAlgError:
        alpha_unconstrained = np.zeros(m)
        print("  [warn] M singular")

    # Constrained
    alpha_star, obj_star = solve_box_qp(M, r, lower=0.0, upper=args.alpha_max)
    print(f"  Constrained:   {dict(zip(expert_list, [f'{v:.4f}' for v in alpha_star]))}")
    print(f"  Objective: {obj_star:.8f}")

    # Active bounds
    for i, e in enumerate(expert_list):
        if alpha_star[i] <= 1e-6:
            print(f"  {e}: at lower bound")
        elif alpha_star[i] >= args.alpha_max - 1e-6:
            print(f"  {e}: at upper bound ({args.alpha_max})")

    # Intervals
    print("\n  === Coefficient Intervals (eta={}) ===".format(args.eta))
    try:
        M_inv = np.linalg.inv(M)
        intervals = {}
        for i, expert_name in enumerate(expert_list):
            width = np.sqrt(2 * args.eta * abs(M_inv[i, i]))
            lo = max(0.0, alpha_star[i] - width)
            hi = min(args.alpha_max, alpha_star[i] + width)
            intervals[expert_name] = {"point": float(alpha_star[i]), "lower": float(lo), "upper": float(hi), "width": float(width)}
            print(f"  {expert_name:>8}: alpha={alpha_star[i]:.4f}, interval=[{lo:.4f}, {hi:.4f}]")
    except np.linalg.LinAlgError:
        intervals = {e: {"point": float(alpha_star[i]), "lower": 0.0, "upper": float(args.alpha_max)} for i, e in enumerate(expert_list)}
        print("  [warn] Could not invert M for intervals")

    # ---- Save results ----
    results = {
        "alpha": {e: float(alpha_star[i]) for i, e in enumerate(expert_list)},
        "alpha_unconstrained": {e: float(alpha_unconstrained[i]) for i, e in enumerate(expert_list)},
        "intervals": intervals,
        "one_expert_shrinkage": one_expert_results,
        "M_matrix": M.tolist(),
        "r_vector": r.tolist(),
        "M_condition": float(np.linalg.cond(M)),
        "objective": float(obj_star),
        "fisher_totals": fisher_totals,
        "quad_forms_per_task": {t: {f"{k[0]},{k[1]}": v for k, v in qf.items()} for t, qf in quad_forms_per_task.items()},
        "config": {
            "fisher_samples": args.fisher_samples,
            "max_length": args.max_length,
            "rho": args.rho,
            "lambda_alpha": args.lambda_alpha,
            "alpha_max": args.alpha_max,
            "eta": args.eta,
            "task_weights": task_weights,
            "experts": expert_list,
        },
    }

    results_path = output_dir / "frs_qp_results.json"
    results_path.write_text(json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\n  Results saved to: {results_path}")

    # Comparison table
    print("\n" + "=" * 60)
    print("COMPARISON WITH TA-0.75")
    print("=" * 60)
    ta_grid = {"tool": 0.50, "memory": 0.75, "code": 0.75}
    print(f"  {'Expert':>8}  {'FRS-QP':>8}  {'1-expert':>10}  {'TA-grid':>8}  {'In interval?':>12}")
    for expert_name in expert_list:
        frs = alpha_star[expert_list.index(expert_name)]
        one = one_expert_results[expert_name]["alpha_shrinkage"]
        ta = ta_grid.get(expert_name, "?")
        intv = intervals[expert_name]
        in_intv = "YES" if intv["lower"] <= ta <= intv["upper"] else "NO"
        print(f"  {expert_name:>8}  {frs:>8.4f}  {one:>10.4f}  {ta:>8}  {in_intv:>12}")

    print(f"\nDone. Results in {output_dir}")


if __name__ == "__main__":
    main()
