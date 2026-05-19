#!/usr/bin/env python3
"""FRS-QP with per-sample storage for bootstrap confidence intervals.

Extends fisher_regression_qp.py:
1. Stores per-sample quadratic form contributions
2. After Fisher estimation, runs B=200 bootstrap resamples offline
3. Reports bootstrap 5/50/95 percentile intervals for each coefficient

Usage:
    CUDA_VISIBLE_DEVICES=3 python scripts/analysis/frs_qp_bootstrap.py \
        --fisher-samples 20 \
        --bootstrap-resamples 200 \
        --output-dir /tmp/shared-storage/ExpertGym/analysis/frs_qp_bootstrap_v1
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
    p.add_argument("--rho", type=float, default=0.1)
    p.add_argument("--lambda-alpha", type=float, default=1e-10)
    p.add_argument("--alpha-max", type=float, default=1.25)
    p.add_argument("--experts", type=str, default="tool,memory,code")
    p.add_argument("--task-weights", type=str, default="1.0,1.0,1.0")
    p.add_argument("--bootstrap-resamples", type=int, default=200)
    p.add_argument("--bootstrap-seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default="/tmp/shared-storage/ExpertGym/analysis/frs_qp_bootstrap_v1")
    return p.parse_args()


def tokenize_sample(tokenizer, sample, max_length):
    """Tokenize sample with left-truncation to preserve response tokens."""
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
    # Left-truncation: keep response, trim prompt prefix
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


def estimate_fisher_per_sample(model, tokenizer, samples, max_length, param_names, device, tau_dict, expert_list):
    """Like estimate_fisher_and_reduce but stores per-sample contributions."""
    import torch

    m = len(expert_list)
    per_sample_quads = []  # list of dicts {(ei,ej) -> float}
    per_sample_fisher = []  # list of float (total Fisher for that sample)

    params_dict = dict(model.named_parameters())
    for p in model.parameters():
        p.requires_grad_(False)
    for name in param_names:
        params_dict[name].requires_grad_(True)

    for sample_idx, sample in enumerate(samples):
        batch = tokenize_sample(tokenizer, sample, max_length)
        if batch is None:
            continue

        model.zero_grad(set_to_none=True)
        inputs = {k: v.to(device) for k, v in batch.items()}
        loss = model(**inputs).loss
        if not torch.isfinite(loss):
            continue
        loss.backward()

        # Compute this sample's quadratic forms
        sample_quads = {(ei, ej): 0.0 for ei in expert_list for ej in expert_list}
        sample_fisher = 0.0

        for name in param_names:
            grad = params_dict[name].grad
            if grad is None:
                continue
            f_p = (grad.detach().float() ** 2).cpu()
            sample_fisher += float(f_p.sum().item())

            for ei in expert_list:
                fi_tau_i = f_p * tau_dict[ei][name]
                for ej in expert_list:
                    sample_quads[(ei, ej)] += float((fi_tau_i * tau_dict[ej][name]).sum().item())

        per_sample_quads.append(sample_quads)
        per_sample_fisher.append(sample_fisher)

        if (sample_idx + 1) % 5 == 0:
            print(f"    sample {sample_idx + 1}/{len(samples)} done")

    return per_sample_quads, per_sample_fisher


def solve_box_qp(M, r, lower=0.0, upper=1.25):
    from scipy.optimize import minimize as scipy_minimize
    m = len(r)

    def obj(x):
        return 0.5 * x @ M @ x - r @ x

    def grad(x):
        return M @ x - r

    bounds = [(lower, upper)] * m
    try:
        x0 = np.clip(np.linalg.solve(M, r), lower, upper)
    except np.linalg.LinAlgError:
        x0 = np.ones(m) * 0.5

    res = scipy_minimize(obj, x0, jac=grad, bounds=bounds, method="L-BFGS-B",
                         options={"ftol": 1e-30, "gtol": 1e-15, "maxiter": 1000})
    return res.x, res.fun, res.success


def build_M_r_from_samples(per_sample_quads_per_task, expert_list, task_weights, rho, lambda_alpha):
    """Build M, r from per-sample quad forms (with resampling support)."""
    m = len(expert_list)
    M = np.zeros((m, m))
    r = np.zeros(m)
    task_to_owner = {"ToolCall": "tool", "Memory": "memory", "Code": "code"}
    num_tasks = len(per_sample_quads_per_task)

    for task, samples_quads in per_sample_quads_per_task.items():
        w_c = task_weights.get(task, 1.0)
        owner = task_to_owner[task]
        n = len(samples_quads)
        if n == 0:
            continue

        # Average over samples
        for sq in samples_quads:
            for i, ei in enumerate(expert_list):
                for j, ej in enumerate(expert_list):
                    M[i, j] += w_c * sq[(ei, ej)] / n
                r[i] += w_c * sq[(owner, ei)] / n

    # Base retention: rho * avg_task(B^T F_task B)
    if rho > 0 and num_tasks > 0:
        for task, samples_quads in per_sample_quads_per_task.items():
            n = len(samples_quads)
            if n == 0:
                continue
            for sq in samples_quads:
                for i, ei in enumerate(expert_list):
                    for j, ej in enumerate(expert_list):
                        M[i, j] += rho * sq[(ei, ej)] / (n * num_tasks)

    M += lambda_alpha * np.eye(m)
    return M, r


def bootstrap_qp(per_sample_quads_per_task, expert_list, task_weights, rho, lambda_alpha,
                  alpha_max, n_bootstrap, rng):
    """Run bootstrap resamples and return alpha distributions."""
    m = len(expert_list)
    alphas = np.zeros((n_bootstrap, m))

    for b in range(n_bootstrap):
        # Resample per-task sample indices with replacement
        resampled = {}
        for task, samples_quads in per_sample_quads_per_task.items():
            n = len(samples_quads)
            if n == 0:
                resampled[task] = []
                continue
            indices = rng.integers(0, n, size=n)
            resampled[task] = [samples_quads[i] for i in indices]

        M, r_vec = build_M_r_from_samples(resampled, expert_list, task_weights, rho, lambda_alpha)
        alpha, _, _ = solve_box_qp(M, r_vec, lower=0.0, upper=alpha_max)
        alphas[b] = alpha

        if (b + 1) % 50 == 0:
            print(f"  Bootstrap {b + 1}/{n_bootstrap} done")

    return alphas


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    expert_list = [e.strip() for e in args.experts.split(",")]
    task_weights_raw = [float(w) for w in args.task_weights.split(",")]
    task_weights = {EXPERT_TASK_MAP[e]: w for e, w in zip(expert_list, task_weights_raw)}
    m = len(expert_list)

    print("=" * 60)
    print("FRS-QP with Bootstrap Confidence Intervals")
    print("=" * 60)
    print(f"Experts: {expert_list}")
    print(f"Task weights: {task_weights}")
    print(f"rho: {args.rho}, lambda_alpha: {args.lambda_alpha}")
    print(f"Fisher samples: {args.fisher_samples}, max_length: {args.max_length}")
    print(f"Bootstrap resamples: {args.bootstrap_resamples}")
    print(f"Output: {output_dir}")
    print()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # [1] Load base model
    print("[1/6] Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True)

    param_names = [n for n in base_model.state_dict().keys()
                   if any(re.match(rx, n) for rx in INCLUDE_REGEX)]
    print(f"  Merge parameters: {len(param_names)}")

    base_state = {n: base_model.state_dict()[n].float().cpu() for n in param_names}

    # [2] Compute task vectors
    print("[2/6] Computing task vectors...")
    tau_dict = {}
    for expert_name in expert_list:
        expert_path = EXPERTS[expert_name]
        print(f"  Loading expert: {expert_name} ({expert_path})")
        expert_model = AutoModelForCausalLM.from_pretrained(expert_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
        expert_state = expert_model.state_dict()
        tau = {}
        for n in param_names:
            tau[n] = (expert_state[n].float().cpu() - base_state[n])
        tau_dict[expert_name] = tau
        del expert_model, expert_state
        gc.collect()

    for en in expert_list:
        norm_sq = sum(float((tau_dict[en][n] ** 2).sum()) for n in param_names)
        print(f"  {en}: ||tau||^2 = {norm_sq:.4f}, ||tau|| = {norm_sq**0.5:.4f}")

    # [3] Load calibration data
    print("[3/6] Loading calibration data...")
    datasets = {}
    for expert_name in expert_list:
        task_name = EXPERT_TASK_MAP[expert_name]
        data_path = DATASET_DIR / f"{task_name}.json"
        with open(data_path) as f:
            all_data = json.load(f)
        n = min(args.fisher_samples, len(all_data))
        datasets[task_name] = all_data[:n]
        print(f"  [data] {task_name}: {n} samples")

    # [4] Estimate Fisher per sample
    print("[4/6] Estimating per-sample Fisher quadratic forms...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = base_model.to(device)
    base_model.eval()

    per_sample_quads_per_task = {}
    per_sample_fisher_per_task = {}

    for task_name, samples in datasets.items():
        print(f"  Task: {task_name}")
        t0 = time.time()
        sq, sf = estimate_fisher_per_sample(
            base_model, tokenizer, samples, args.max_length,
            param_names, device, tau_dict, expert_list
        )
        elapsed = time.time() - t0
        per_sample_quads_per_task[task_name] = sq
        per_sample_fisher_per_task[task_name] = sf

        n = len(sq)
        avg_fisher = sum(sf) / max(n, 1)
        print(f"    {n} valid samples, avg Fisher: {avg_fisher:.4f}, time: {elapsed:.1f}s")

    # Free GPU memory
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    # [5] Solve QP with point estimate
    print("\n[5/6] Solving point estimate QP...")
    M, r_vec = build_M_r_from_samples(
        per_sample_quads_per_task, expert_list, task_weights,
        args.rho, args.lambda_alpha
    )

    print(f"  M matrix:\n    {np.array2string(M, precision=6, separator=', ')}")
    print(f"  r vector: {np.array2string(r_vec, precision=6)}")
    print(f"  M condition: {np.linalg.cond(M):.2f}")

    # Unconstrained
    try:
        alpha_unc = np.linalg.solve(M, r_vec)
        print(f"  Unconstrained: {dict(zip(expert_list, [f'{a:.4f}' for a in alpha_unc]))}")
    except np.linalg.LinAlgError:
        alpha_unc = np.ones(m) * 0.5

    # Constrained
    alpha_point, obj_point, success = solve_box_qp(M, r_vec, lower=0.0, upper=args.alpha_max)
    print(f"  Constrained:   {dict(zip(expert_list, [f'{a:.4f}' for a in alpha_point]))}")
    print(f"  Objective: {obj_point:.8e}")

    # One-expert shrinkage
    print("\n  === One-Expert Shrinkage ===")
    task_to_owner = {"ToolCall": "tool", "Memory": "memory", "Code": "code"}
    owner_to_task = {v: k for k, v in task_to_owner.items()}
    one_expert_results = {}
    for e in expert_list:
        task_e = owner_to_task[e]
        sq_list = per_sample_quads_per_task[task_e]
        n = len(sq_list)
        A_e = sum(sq[(e, e)] for sq in sq_list) / max(n, 1)

        protected_tasks = [t for t in per_sample_quads_per_task if t != task_e]
        B_e = sum(
            sum(sq[(e, e)] for sq in per_sample_quads_per_task[pt]) / max(len(per_sample_quads_per_task[pt]), 1)
            for pt in protected_tasks
        ) / max(len(protected_tasks), 1)

        alpha_shrinkage = A_e / (A_e + B_e) if (A_e + B_e) > 0 else 0.0
        one_expert_results[e] = {
            "A_e": A_e, "B_e": B_e,
            "B/A": B_e / A_e if A_e > 0 else float("inf"),
            "alpha": alpha_shrinkage,
        }
        print(f"    {e:>8}: A={A_e:.6e}, B={B_e:.6e}, B/A={B_e/A_e if A_e > 0 else float('inf'):.4f}, alpha*={alpha_shrinkage:.4f}")

    # [6] Bootstrap
    print(f"\n[6/6] Running {args.bootstrap_resamples} bootstrap resamples...")
    rng = np.random.default_rng(args.bootstrap_seed)
    bootstrap_alphas = bootstrap_qp(
        per_sample_quads_per_task, expert_list, task_weights,
        args.rho, args.lambda_alpha, args.alpha_max,
        args.bootstrap_resamples, rng
    )

    print("\n  === Bootstrap Results ===")
    print(f"  {'expert':>8}  {'point':>8}  {'p5':>8}  {'p25':>8}  {'p50':>8}  {'p75':>8}  {'p95':>8}  {'std':>8}")
    print("  " + "-" * 72)

    bootstrap_results = {}
    for i, e in enumerate(expert_list):
        vals = bootstrap_alphas[:, i]
        p5, p25, p50, p75, p95 = np.percentile(vals, [5, 25, 50, 75, 95])
        std = np.std(vals)
        bootstrap_results[e] = {
            "point": float(alpha_point[i]),
            "p5": float(p5), "p25": float(p25), "p50": float(p50),
            "p75": float(p75), "p95": float(p95),
            "std": float(std), "mean": float(np.mean(vals)),
        }
        print(f"  {e:>8}  {alpha_point[i]:8.4f}  {p5:8.4f}  {p25:8.4f}  {p50:8.4f}  {p75:8.4f}  {p95:8.4f}  {std:8.4f}")

    # Also bootstrap one-expert shrinkage
    print("\n  === One-Expert Shrinkage Bootstrap ===")
    n_boot = args.bootstrap_resamples
    shrinkage_boot = {e: np.zeros(n_boot) for e in expert_list}

    for b in range(n_boot):
        for e in expert_list:
            task_e = owner_to_task[e]
            sq_list = per_sample_quads_per_task[task_e]
            n = len(sq_list)
            if n == 0:
                continue
            indices = rng.integers(0, n, size=n)
            A_e = sum(sq_list[idx][(e, e)] for idx in indices) / n

            protected_tasks = [t for t in per_sample_quads_per_task if t != task_e]
            B_e = 0.0
            for pt in protected_tasks:
                pt_list = per_sample_quads_per_task[pt]
                np_ = len(pt_list)
                if np_ == 0:
                    continue
                pt_indices = rng.integers(0, np_, size=np_)
                B_e += sum(pt_list[idx][(e, e)] for idx in pt_indices) / np_
            B_e /= max(len(protected_tasks), 1)

            shrinkage_boot[e][b] = A_e / (A_e + B_e) if (A_e + B_e) > 0 else 0.0

    print(f"  {'expert':>8}  {'point':>8}  {'p5':>8}  {'p25':>8}  {'p50':>8}  {'p75':>8}  {'p95':>8}  {'std':>8}")
    print("  " + "-" * 72)

    shrinkage_bootstrap_results = {}
    for e in expert_list:
        vals = shrinkage_boot[e]
        p5, p25, p50, p75, p95 = np.percentile(vals, [5, 25, 50, 75, 95])
        std = np.std(vals)
        point_alpha = one_expert_results[e]["alpha"]
        shrinkage_bootstrap_results[e] = {
            "point": float(point_alpha),
            "p5": float(p5), "p25": float(p25), "p50": float(p50),
            "p75": float(p75), "p95": float(p95),
            "std": float(std), "mean": float(np.mean(vals)),
        }
        print(f"  {e:>8}  {point_alpha:8.4f}  {p5:8.4f}  {p25:8.4f}  {p50:8.4f}  {p75:8.4f}  {p95:8.4f}  {std:8.4f}")

    # Save all results
    all_results = {
        "point_alpha": {e: float(alpha_point[i]) for i, e in enumerate(expert_list)},
        "point_alpha_unconstrained": {e: float(alpha_unc[i]) for i, e in enumerate(expert_list)},
        "one_expert_shrinkage": one_expert_results,
        "bootstrap_multi_expert": bootstrap_results,
        "bootstrap_one_expert_shrinkage": shrinkage_bootstrap_results,
        "M_matrix": M.tolist(),
        "r_vector": r_vec.tolist(),
        "M_condition": float(np.linalg.cond(M)),
        "objective": float(obj_point),
        "per_sample_fisher_totals": {
            task: [float(f) for f in fishers]
            for task, fishers in per_sample_fisher_per_task.items()
        },
        "config": {
            "fisher_samples": args.fisher_samples,
            "max_length": args.max_length,
            "rho": args.rho,
            "lambda_alpha": args.lambda_alpha,
            "alpha_max": args.alpha_max,
            "bootstrap_resamples": args.bootstrap_resamples,
            "bootstrap_seed": args.bootstrap_seed,
            "task_weights": task_weights,
            "experts": expert_list,
        },
    }

    # Save per-sample quad forms for future analysis
    serializable_quads = {}
    for task, sq_list in per_sample_quads_per_task.items():
        serializable_quads[task] = [
            {f"{ei},{ej}": sq[(ei, ej)] for ei in expert_list for ej in expert_list}
            for sq in sq_list
        ]
    all_results["per_sample_quad_forms"] = serializable_quads

    output_path = output_dir / "bootstrap_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY: FRS-QP Coefficients with Bootstrap CIs")
    print("=" * 70)
    print(f"  {'Expert':>8}  {'Point':>8}  {'Bootstrap 90% CI':>22}  {'1-expert':>10}  {'1-expert 90% CI':>22}")
    print("  " + "-" * 76)
    for e in expert_list:
        br = bootstrap_results[e]
        sr = shrinkage_bootstrap_results[e]
        oe = one_expert_results[e]["alpha"]
        print(f"  {e:>8}  {br['point']:8.4f}  [{br['p5']:8.4f}, {br['p95']:8.4f}]  {oe:10.4f}  [{sr['p5']:8.4f}, {sr['p95']:8.4f}]")

    print(f"\n  Comparison with TA baselines:")
    print(f"  {'Method':>30s}  {'tool':>8s}  {'memory':>8s}  {'code':>8s}")
    print("  " + "-" * 58)
    print(f"  {'TA-0.75 (grid)':>30s}  {'0.5000':>8s}  {'0.7500':>8s}  {'0.7500':>8s}")
    print(f"  {'TA-1/3 (uniform)':>30s}  {'0.3333':>8s}  {'0.3333':>8s}  {'0.3333':>8s}")
    br_tool = bootstrap_results['tool']
    br_mem = bootstrap_results['memory']
    br_code = bootstrap_results['code']
    print(f"  {'FRS-QP point':>30s}  {br_tool['point']:8.4f}  {br_mem['point']:8.4f}  {br_code['point']:8.4f}")
    print(f"  {'FRS-QP bootstrap median':>30s}  {br_tool['p50']:8.4f}  {br_mem['p50']:8.4f}  {br_code['p50']:8.4f}")

    # Check if TA-0.75 falls within bootstrap 90% CI
    ta_grid = {"tool": 0.5, "memory": 0.75, "code": 0.75}
    for e in expert_list:
        br = bootstrap_results[e]
        in_ci = br["p5"] <= ta_grid[e] <= br["p95"]
        print(f"  TA-grid {e}={ta_grid[e]} in 90% CI [{br['p5']:.4f}, {br['p95']:.4f}]? {'YES' if in_ci else 'NO'}")


if __name__ == "__main__":
    main()
