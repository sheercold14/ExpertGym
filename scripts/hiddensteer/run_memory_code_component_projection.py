#!/usr/bin/env python3
"""Closed-form Memory prompt code-component projection smoke.

This is the no-training counterpart to train_memory_residual_immunization.py.
It builds a code-on-memory interference basis from train Memory prompts only,
then directly projects that component out at inference time:

    h' = h - alpha * B_bad B_bad^T (h - center)

The purpose is to test whether direct geometric removal improves a heldout
Memory subset before trying any benchmark-scale evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.hiddensteer import train_memory_residual_immunization as base


DEFAULT_OUTPUT_DIR = "/tmp/shared-storage/ExpertGym/hiddensteer/memory_code_component_projection_smoke_20260525"


class DirectCodeComponentProjector:
    def __init__(
        self,
        *,
        layers: list[int],
        geometry: dict[int, dict[str, torch.Tensor]],
        alpha: float,
        threshold_quantile: float,
        device: str,
        dtype: torch.dtype,
    ) -> None:
        self.layers = list(layers)
        self.geometry = geometry
        self.alpha = float(alpha)
        self.threshold_quantile = float(threshold_quantile)
        self.device = device
        self.dtype = dtype
        self.enabled = True
        self.stats: dict[str, float] = {}
        self.reset_stats()

    def reset_stats(self) -> None:
        self.stats = {
            "hook_calls": 0.0,
            "tokens_seen": 0.0,
            "tokens_projected": 0.0,
            "projection_norm_sum": 0.0,
            "centered_norm_sum": 0.0,
            "energy_sum": 0.0,
        }

    def clear_step(self, positions: torch.Tensor | None) -> None:
        del positions

    def set_config(self, *, alpha: float, threshold_quantile: float) -> None:
        self.alpha = float(alpha)
        self.threshold_quantile = float(threshold_quantile)
        self.reset_stats()

    def forward_layer(self, layer: int, hidden: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return hidden
        payload = self.geometry[layer]
        basis = payload["code_bad_basis"].to(device=hidden.device, dtype=hidden.dtype)
        if basis.numel() == 0:
            return hidden
        center = payload["merged_center"].to(device=hidden.device, dtype=hidden.dtype).view(1, 1, -1)
        threshold = float(payload[energy_key(self.threshold_quantile)].item())
        centered = hidden - center
        coeff = torch.matmul(centered, basis)
        projection = torch.matmul(coeff, basis.transpose(0, 1))
        numerator = coeff.float().pow(2).sum(dim=-1)
        denominator = centered.float().pow(2).sum(dim=-1).clamp_min(1.0e-12)
        energy = numerator / denominator
        mask = (energy > threshold).to(dtype=hidden.dtype).unsqueeze(-1)
        correction = self.alpha * projection * mask
        projected = hidden - correction

        self.stats["hook_calls"] += 1.0
        self.stats["tokens_seen"] += float(hidden.shape[0] * hidden.shape[1])
        self.stats["tokens_projected"] += float(mask.detach().float().sum().item())
        self.stats["projection_norm_sum"] += float(correction.detach().float().norm(dim=-1).sum().item())
        self.stats["centered_norm_sum"] += float(centered.detach().float().norm(dim=-1).sum().item())
        self.stats["energy_sum"] += float(energy.detach().float().sum().item())
        return projected


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base.seed_everything(int(args.seed))

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[str(args.torch_dtype)]
    layers = base.parse_layers(args.layers)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    examples = base.load_memory_examples(
        Path(args.rollout_path).expanduser().resolve(),
        turn_kinds=base.split_csv(args.turn_kinds),
        max_examples=int(args.max_examples),
        max_turns_per_prompt=int(args.max_turns_per_prompt),
        prefer_success=not bool(args.allow_failed_teacher_samples),
    )
    train_examples, heldout_examples = base.split_examples_by_prompt(
        examples,
        train_count=int(args.train_examples),
        heldout_count=int(args.heldout_examples),
        seed=int(args.seed),
    )
    encoded_train = [
        base.encode_example(
            tokenizer,
            item,
            max_seq_length=int(args.max_seq_length),
            response_tail_tokens=int(args.response_tail_tokens),
            device=str(args.device),
        )
        for item in train_examples
    ]
    encoded_heldout = [
        base.encode_example(
            tokenizer,
            item,
            max_seq_length=int(args.max_seq_length),
            response_tail_tokens=int(args.response_tail_tokens),
            device=str(args.device),
        )
        for item in heldout_examples
    ]

    geometry_path = Path(args.geometry_path).expanduser().resolve() if args.geometry_path else output_dir / "memory_code_bad_centered_basis.pt"
    if args.geometry_path:
        geometry = torch.load(geometry_path, map_location="cpu")
    else:
        geometry = build_centered_code_bad_geometry(
            args=args,
            encoded_examples=encoded_train,
            layers=layers,
            dtype=dtype,
            AutoModelForCausalLM=AutoModelForCausalLM,
        )
        torch.save(geometry, geometry_path)

    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype, trust_remote_code=True)
    model.to(args.device)
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    projector = DirectCodeComponentProjector(
        layers=layers,
        geometry=geometry,
        alpha=0.0,
        threshold_quantile=0.95,
        device=str(args.device),
        dtype=dtype,
    )
    hooks = base.install_corrector_hooks(model, projector)
    retention_args = SimpleNamespace(
        tool_retention_rollout=args.tool_retention_rollout,
        code_retention_rollout=args.code_retention_rollout,
        retention_examples=int(args.retention_examples),
        max_seq_length=int(args.max_seq_length),
        retention_response_tail_tokens=int(args.retention_response_tail_tokens),
        device=str(args.device),
    )

    try:
        baseline_heldout = base.evaluate_nll(model, projector, encoded_heldout, geometry, enabled=False)
        baseline_train = base.evaluate_nll(model, projector, encoded_train, geometry, enabled=False)
        runs = []
        for alpha in parse_float_list(args.alphas):
            for threshold_quantile in parse_float_list(args.threshold_quantiles):
                projector.set_config(alpha=alpha, threshold_quantile=threshold_quantile)
                started = time.perf_counter()
                corrected_heldout = base.evaluate_nll(model, projector, encoded_heldout, geometry, enabled=True)
                corrected_train = base.evaluate_nll(model, projector, encoded_train, geometry, enabled=True)
                retention = base.evaluate_retention(
                    model=model,
                    tokenizer=tokenizer,
                    corrector=projector,
                    geometry=geometry,
                    args=retention_args,
                )
                wall = time.perf_counter() - started
                runs.append(
                    {
                        "alpha": alpha,
                        "threshold_quantile": threshold_quantile,
                        "threshold_name": f"p{int(round(threshold_quantile * 100)):02d}",
                        "corrected_train": corrected_train,
                        "corrected_heldout": corrected_heldout,
                        "heldout_delta_corrected_minus_baseline": corrected_heldout["mean_nll"] - baseline_heldout["mean_nll"],
                        "train_delta_corrected_minus_baseline": corrected_train["mean_nll"] - baseline_train["mean_nll"],
                        "retention": retention,
                        "stats": dict(projector.stats),
                        "wall_time_sec": wall,
                        "interpretation": interpret_projection_result(baseline_heldout, corrected_heldout, retention),
                    }
                )
        summary = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "objective": "Direct Memory-prompt code-component projection smoke",
            "model_path": str(args.model_path),
            "memory_model_path": str(args.memory_model_path),
            "code_model_path": str(args.code_model_path),
            "rollout_path": str(args.rollout_path),
            "layers": layers,
            "basis_rank": int(args.basis_rank),
            "train_examples": [base.compact_example(item) for item in train_examples],
            "heldout_examples": [base.compact_example(item) for item in heldout_examples],
            "baseline_train": baseline_train,
            "baseline_heldout": baseline_heldout,
            "runs": runs,
            "geometry_path": str(geometry_path),
        }
        base.write_json(output_dir / "run_summary.json", summary)
        write_markdown(output_dir / "README.md", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        for handle in hooks:
            handle.remove()
        model.cpu()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_centered_code_bad_geometry(
    *,
    args: argparse.Namespace,
    encoded_examples: list[base.EncodedExample],
    layers: list[int],
    dtype: torch.dtype,
    AutoModelForCausalLM: Any,
) -> dict[int, dict[str, torch.Tensor]]:
    merged = base.collect_hidden_vectors(
        model_path=str(args.model_path),
        encoded_examples=encoded_examples,
        layers=layers,
        dtype=dtype,
        device=str(args.device),
        AutoModelForCausalLM=AutoModelForCausalLM,
    )
    memory = base.collect_hidden_vectors(
        model_path=str(args.memory_model_path),
        encoded_examples=encoded_examples,
        layers=layers,
        dtype=dtype,
        device=str(args.device),
        AutoModelForCausalLM=AutoModelForCausalLM,
    )
    code = base.collect_hidden_vectors(
        model_path=str(args.code_model_path),
        encoded_examples=encoded_examples,
        layers=layers,
        dtype=dtype,
        device=str(args.device),
        AutoModelForCausalLM=AutoModelForCausalLM,
    )
    geometry: dict[int, dict[str, torch.Tensor]] = {}
    for layer in layers:
        merged_center = merged[layer].float().mean(dim=0)
        centered_merged = merged[layer].float() - merged_center
        memory_delta = base.normalize_rows(memory[layer] - merged[layer])
        code_delta = base.normalize_rows(code[layer] - merged[layer])
        memory_basis = base.fit_basis(memory_delta, rank=int(args.basis_rank))
        code_basis = base.fit_basis(code_delta, rank=int(args.basis_rank))
        code_bad = base.orthogonalize_against(code_basis, memory_basis)
        energy = component_energy(centered_merged, code_bad)
        payload = {
            "merged_center": merged_center.cpu(),
            "memory_basis": memory_basis.cpu(),
            "code_basis": code_basis.cpu(),
            "code_bad_basis": code_bad.cpu(),
            "energy_p90": torch.quantile(energy, 0.90).cpu(),
            "energy_p95": torch.quantile(energy, 0.95).cpu(),
            "energy_p99": torch.quantile(energy, 0.99).cpu(),
            "train_energy_mean": energy.mean().cpu(),
            "num_vectors": torch.tensor([memory_delta.shape[0]], dtype=torch.long),
        }
        for quantile in parse_float_list(args.threshold_quantiles):
            payload[energy_key(quantile)] = torch.quantile(energy, quantile).cpu()
        geometry[layer] = payload
    return geometry


def component_energy(centered_hidden: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    if basis.numel() == 0:
        return centered_hidden.new_zeros((centered_hidden.shape[0],))
    coeff = torch.matmul(centered_hidden.float(), basis.float())
    numerator = coeff.pow(2).sum(dim=-1)
    denominator = centered_hidden.float().pow(2).sum(dim=-1).clamp_min(1.0e-12)
    return numerator / denominator


def interpret_projection_result(
    baseline_heldout: dict[str, float],
    corrected_heldout: dict[str, float],
    retention: dict[str, dict[str, Any]],
) -> str:
    heldout_delta = corrected_heldout["mean_nll"] - baseline_heldout["mean_nll"]
    retention_ok = all(payload["nll_delta"] <= 0.0 for payload in retention.values())
    if heldout_delta < 0.0 and retention_ok:
        return "positive_memory_signal_with_retention"
    if heldout_delta < 0.0:
        return "positive_memory_signal_retention_risk"
    return "no_memory_nll_gain"


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Direct Memory Code-Component Projection Smoke",
        "",
        f"- Created: `{summary['created_at']}`",
        f"- Model: `{summary['model_path']}`",
        f"- Layers: `{summary['layers']}`",
        f"- Basis rank: `{summary['basis_rank']}`",
        f"- Geometry: `{summary['geometry_path']}`",
        "",
        f"- Baseline train NLL: `{summary['baseline_train']['mean_nll']:.4f}`",
        f"- Baseline heldout NLL: `{summary['baseline_heldout']['mean_nll']:.4f}`",
        "",
        "| alpha | threshold | heldout NLL | heldout delta | train delta | tool delta | code delta | projected/token | interpretation |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for run in summary["runs"]:
        stats = run["stats"]
        projected_per_token = stats["tokens_projected"] / max(stats["tokens_seen"], 1.0)
        retention = run["retention"]
        lines.append(
            f"| {run['alpha']:.2f} | {run['threshold_name']} | "
            f"{run['corrected_heldout']['mean_nll']:.4f} | "
            f"{run['heldout_delta_corrected_minus_baseline']:+.4f} | "
            f"{run['train_delta_corrected_minus_baseline']:+.4f} | "
            f"{retention['tool']['nll_delta']:+.4f} | "
            f"{retention['code']['nll_delta']:+.4f} | "
            f"{projected_per_token:.4f} | {run['interpretation']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=base.DEFAULT_MODEL_PATH)
    parser.add_argument("--memory-model-path", default=base.DEFAULT_MEMORY_MODEL_PATH)
    parser.add_argument("--code-model-path", default=base.DEFAULT_CODE_MODEL_PATH)
    parser.add_argument("--rollout-path", default=base.DEFAULT_ROLLOUT_PATH)
    parser.add_argument("--tool-retention-rollout", default=base.DEFAULT_TOOL_RETENTION_ROLLOUT)
    parser.add_argument("--code-retention-rollout", default=base.DEFAULT_CODE_RETENTION_ROLLOUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--geometry-path", default="")
    parser.add_argument("--turn-kinds", default="memory_update")
    parser.add_argument("--layers", default="26")
    parser.add_argument("--basis-rank", type=int, default=2)
    parser.add_argument("--max-examples", type=int, default=32)
    parser.add_argument("--max-turns-per-prompt", type=int, default=1)
    parser.add_argument("--train-examples", type=int, default=20)
    parser.add_argument("--heldout-examples", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=1536)
    parser.add_argument("--response-tail-tokens", type=int, default=64)
    parser.add_argument("--retention-examples", type=int, default=4)
    parser.add_argument("--retention-response-tail-tokens", type=int, default=96)
    parser.add_argument("--alphas", default="0.25,0.5,1.0")
    parser.add_argument("--threshold-quantiles", default="0.90,0.95")
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--allow-failed-teacher-samples", action="store_true")
    return parser.parse_args()


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in str(raw).split(",") if item.strip()]


def energy_key(quantile: float) -> str:
    return f"energy_p{int(round(float(quantile) * 100)):02d}"


if __name__ == "__main__":
    main()
