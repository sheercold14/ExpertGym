#!/usr/bin/env python3
"""
FRS-QP Sensitivity Analysis: sweep rho and task_weights.

Uses pre-computed quadratic forms from frs_qp_v3 — no GPU needed.
Rebuilds M, r for each configuration and re-solves the box-constrained QP.
"""

import json
import sys
import os
import numpy as np
from scipy.optimize import minimize as scipy_minimize
from itertools import product

def load_results(path):
    with open(path) as f:
        return json.load(f)

def build_M_r(quad_forms, experts, task_weights, rho, lambda_alpha, fisher_totals):
    """Build M matrix and r vector from per-task quadratic forms."""
    m = len(experts)
    M = np.zeros((m, m))
    r = np.zeros(m)

    task_to_owner = {"ToolCall": "tool", "Memory": "memory", "Code": "code"}

    for task, w_c in task_weights.items():
        qf = quad_forms[task]
        for i, ei in enumerate(experts):
            for j, ej in enumerate(experts):
                key = f"{ei},{ej}"
                M[i, j] += w_c * qf[key]
            # r_i = sum_c w_c * tau_c^T F_c tau_i
            # The owner of task c is expert c, so we need tau_c^T F_c tau_i
            owner = task_to_owner[task]
            key_owner_i = f"{owner},{ei}"
            r[i] += w_c * qf[key_owner_i]

    # Base retention: rho * avg_task(B^T F_task B)
    # Original FRS-QP code divides by num_tasks to get average Fisher
    num_tasks = len(quad_forms)
    if rho > 0 and num_tasks > 0:
        for task in quad_forms:
            qf = quad_forms[task]
            for i, ei in enumerate(experts):
                for j, ej in enumerate(experts):
                    key = f"{ei},{ej}"
                    M[i, j] += rho * qf[key] / num_tasks

    # Tikhonov
    M += lambda_alpha * np.eye(m)

    return M, r

def solve_box_qp(M, r, lower=0.0, upper=1.25):
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

    res = scipy_minimize(
        obj, x0, jac=grad, bounds=bounds, method="L-BFGS-B",
        options={"ftol": 1e-30, "gtol": 1e-15, "maxiter": 1000}
    )
    return res.x, res.fun, res.success

def one_expert_shrinkage(quad_forms, experts, task_weights, rho):
    """Compute one-expert shrinkage alpha for each expert."""
    task_to_owner = {"ToolCall": "tool", "Memory": "memory", "Code": "code"}
    owner_to_task = {v: k for k, v in task_to_owner.items()}

    results = {}
    for e in experts:
        task_e = owner_to_task[e]
        # A_e = tau_e^T F_owner tau_e (using owner task Fisher)
        A_e = quad_forms[task_e][f"{e},{e}"]

        # B_e = average of non-owner task Fishers (matches original FRS-QP code)
        protected_tasks = [t for t in quad_forms if t != task_e]
        B_e = sum(quad_forms[t][f"{e},{e}"] for t in protected_tasks) / max(len(protected_tasks), 1)

        # Base retention: average Fisher across ALL tasks
        F_base_ee = sum(quad_forms[task][f"{e},{e}"] for task in quad_forms) / len(quad_forms)

        alpha_no_base = A_e / (A_e + B_e) if (A_e + B_e) > 0 else 0.0
        alpha_with_base = A_e / (A_e + B_e + rho * F_base_ee) if (A_e + B_e + rho * F_base_ee) > 0 else 0.0

        results[e] = {
            "A_e": A_e,
            "B_e": B_e,
            "B/A": B_e / A_e if A_e > 0 else float("inf"),
            "alpha_no_base": alpha_no_base,
            "alpha_with_base": alpha_with_base,
        }
    return results

def main():
    results_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/shared-storage/ExpertGym/analysis/frs_qp_v3/frs_qp_results.json"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "/tmp/shared-storage/ExpertGym/analysis/frs_qp_sensitivity"

    os.makedirs(output_dir, exist_ok=True)

    data = load_results(results_path)
    quad_forms = data["quad_forms_per_task"]
    experts = data["config"]["experts"]
    fisher_totals = data["fisher_totals"]

    print("=" * 70)
    print("FRS-QP Sensitivity Analysis")
    print("=" * 70)
    print(f"Experts: {experts}")
    print(f"Fisher totals: {fisher_totals}")
    print()

    # ===== Sweep 1: rho sensitivity =====
    rho_values = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
    lambda_alpha = 1e-10
    task_weights_default = {"ToolCall": 1.0, "Memory": 1.0, "Code": 1.0}

    print("=" * 70)
    print("SWEEP 1: rho sensitivity (equal task weights)")
    print("=" * 70)
    print(f"{'rho':>8s}  {'tool':>8s}  {'memory':>8s}  {'code':>8s}  {'objective':>12s}")
    print("-" * 50)

    rho_results = []
    for rho in rho_values:
        M, r = build_M_r(quad_forms, experts, task_weights_default, rho, lambda_alpha, fisher_totals)
        alpha, obj, success = solve_box_qp(M, r)
        result = {
            "rho": rho,
            "alpha": {e: float(alpha[i]) for i, e in enumerate(experts)},
            "objective": float(obj),
            "converged": bool(success),
        }
        rho_results.append(result)
        print(f"{rho:8.2f}  {alpha[0]:8.4f}  {alpha[1]:8.4f}  {alpha[2]:8.4f}  {obj:12.6e}")

    # ===== Sweep 2: task weight sensitivity =====
    print()
    print("=" * 70)
    print("SWEEP 2: task weight sensitivity (rho=0.1)")
    print("=" * 70)

    weight_configs = [
        {"label": "equal (1:1:1)", "weights": {"ToolCall": 1.0, "Memory": 1.0, "Code": 1.0}},
        {"label": "code 3x (1:1:3)", "weights": {"ToolCall": 1.0, "Memory": 1.0, "Code": 3.0}},
        {"label": "code 10x (1:1:10)", "weights": {"ToolCall": 1.0, "Memory": 1.0, "Code": 10.0}},
        {"label": "code 30x (1:1:30)", "weights": {"ToolCall": 1.0, "Memory": 1.0, "Code": 30.0}},
        {"label": "code 100x (1:1:100)", "weights": {"ToolCall": 1.0, "Memory": 1.0, "Code": 100.0}},
        {"label": "code 250x (1:1:250)", "weights": {"ToolCall": 1.0, "Memory": 1.0, "Code": 250.0}},
        {"label": "mem 3x (1:3:1)", "weights": {"ToolCall": 1.0, "Memory": 3.0, "Code": 1.0}},
        {"label": "tool 3x (3:1:1)", "weights": {"ToolCall": 3.0, "Memory": 1.0, "Code": 1.0}},
        {"label": "no code (1:1:0)", "weights": {"ToolCall": 1.0, "Memory": 1.0, "Code": 0.0}},
        {"label": "fisher-balanced", "weights": {
            "ToolCall": 1.0 / max(fisher_totals["ToolCall"], 1e-12),
            "Memory": 1.0 / max(fisher_totals["Memory"], 1e-12),
            "Code": 1.0 / max(fisher_totals["Code"], 1e-12),
        }},
    ]

    # Normalize fisher-balanced weights so they sum to 3
    fb = weight_configs[-1]["weights"]
    total_fb = sum(fb.values())
    for k in fb:
        fb[k] = fb[k] / total_fb * 3.0

    print(f"{'config':>25s}  {'tool':>8s}  {'memory':>8s}  {'code':>8s}  {'objective':>12s}")
    print("-" * 70)

    weight_results = []
    for cfg in weight_configs:
        M, r = build_M_r(quad_forms, experts, cfg["weights"], 0.1, lambda_alpha, fisher_totals)
        alpha, obj, success = solve_box_qp(M, r)
        result = {
            "label": cfg["label"],
            "weights": cfg["weights"],
            "alpha": {e: float(alpha[i]) for i, e in enumerate(experts)},
            "objective": float(obj),
            "converged": bool(success),
        }
        weight_results.append(result)
        print(f"{cfg['label']:>25s}  {alpha[0]:8.4f}  {alpha[1]:8.4f}  {alpha[2]:8.4f}  {obj:12.6e}")

    # ===== Sweep 3: lambda_alpha sensitivity =====
    print()
    print("=" * 70)
    print("SWEEP 3: lambda_alpha sensitivity (rho=0.1, equal weights)")
    print("=" * 70)

    lambda_values = [0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

    print(f"{'lambda':>12s}  {'tool':>8s}  {'memory':>8s}  {'code':>8s}  {'objective':>12s}")
    print("-" * 55)

    lambda_results = []
    for lam in lambda_values:
        M, r = build_M_r(quad_forms, experts, task_weights_default, 0.1, lam, fisher_totals)
        alpha, obj, success = solve_box_qp(M, r)
        result = {
            "lambda_alpha": lam,
            "alpha": {e: float(alpha[i]) for i, e in enumerate(experts)},
            "objective": float(obj),
        }
        lambda_results.append(result)
        print(f"{lam:12.1e}  {alpha[0]:8.4f}  {alpha[1]:8.4f}  {alpha[2]:8.4f}  {obj:12.6e}")

    # ===== One-expert shrinkage sensitivity to rho =====
    print()
    print("=" * 70)
    print("SWEEP 4: One-expert shrinkage vs rho")
    print("=" * 70)
    print(f"{'rho':>8s}  {'tool':>10s}  {'memory':>10s}  {'code':>10s}")
    print("-" * 45)

    shrinkage_results = []
    for rho in rho_values:
        sr = one_expert_shrinkage(quad_forms, experts, task_weights_default, rho)
        shrinkage_results.append({"rho": rho, "shrinkage": sr})
        print(f"{rho:8.2f}  {sr['tool']['alpha_with_base']:10.4f}  {sr['memory']['alpha_with_base']:10.4f}  {sr['code']['alpha_with_base']:10.4f}")

    # ===== Summary: comparison with TA baselines =====
    print()
    print("=" * 70)
    print("SUMMARY: Comparison with baselines")
    print("=" * 70)

    # Find default config (rho=0.1, equal weights)
    default_alpha = rho_results[3]["alpha"]  # rho=0.1

    print(f"{'Method':>30s}  {'tool':>8s}  {'memory':>8s}  {'code':>8s}")
    print("-" * 60)
    print(f"{'TA-0.75 (grid)':>30s}  {'0.5000':>8s}  {'0.7500':>8s}  {'0.7500':>8s}")
    print(f"{'TA-1/3 (uniform)':>30s}  {'0.3333':>8s}  {'0.3333':>8s}  {'0.3333':>8s}")
    print(f"{'1-expert shrinkage':>30s}  {0.7430:8.4f}  {0.5212:8.4f}  {0.0067:8.4f}")
    print(f"{'FRS-QP rho=0.1 (default)':>30s}  {default_alpha['tool']:8.4f}  {default_alpha['memory']:8.4f}  {default_alpha['code']:8.4f}")

    # Best code-boosted config
    for wr in weight_results:
        if wr["label"] == "code 250x (1:1:250)":
            a = wr["alpha"]
            print(f"{'FRS-QP code 250x':>30s}  {a['tool']:8.4f}  {a['memory']:8.4f}  {a['code']:8.4f}")
    for wr in weight_results:
        if wr["label"] == "fisher-balanced":
            a = wr["alpha"]
            print(f"{'FRS-QP fisher-balanced':>30s}  {a['tool']:8.4f}  {a['memory']:8.4f}  {a['code']:8.4f}")

    # No-base config
    rho0_alpha = rho_results[0]["alpha"]  # rho=0
    print(f"{'FRS-QP rho=0 (no base)':>30s}  {rho0_alpha['tool']:8.4f}  {rho0_alpha['memory']:8.4f}  {rho0_alpha['code']:8.4f}")

    # Save all results
    all_results = {
        "rho_sweep": rho_results,
        "weight_sweep": weight_results,
        "lambda_sweep": lambda_results,
        "shrinkage_sweep": shrinkage_results,
        "source": results_path,
    }

    output_path = os.path.join(output_dir, "sensitivity_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()
