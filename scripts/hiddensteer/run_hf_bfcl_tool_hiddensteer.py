#!/usr/bin/env python3
"""Run a small BFCL Tool generation pass with optional HiddenSteer HF hooks.

This is a feasibility harness, not a replacement for the official vLLM BFCL
run. It keeps the comparison local: same HF model, same prompts, hook disabled
vs enabled, and records wall-clock/tokens so we can decide whether vLLM
engineering is justified.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_BFCL_ROOT = "/mnt/cache/wuruixiao/users/lsc/gorilla/berkeley-function-call-leaderboard"
DEFAULT_PYTHON = "/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python"
DEFAULT_CATEGORIES = "parallel,parallel_multiple,live_parallel,live_parallel_multiple"
DEFAULT_OUTPUT_DIR = "/tmp/shared-storage/ExpertGym/hiddensteer/hf_tool_smoke_20260524"


@dataclass
class HookStats:
    hook_calls: int = 0
    tokens_seen: int = 0
    conflict_tokens: int = 0
    correction_norm_sum: float = 0.0
    owner_norm_sum: float = 0.0
    residual_norm_sum: float = 0.0
    output_norm_sum: float = 0.0
    module_calls: dict[str, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "hook_calls": self.hook_calls,
            "tokens_seen": self.tokens_seen,
            "conflict_tokens": self.conflict_tokens,
            "conflict_token_fraction": self.conflict_tokens / max(self.tokens_seen, 1),
            "correction_norm_mean": self.correction_norm_sum / max(self.tokens_seen, 1),
            "owner_norm_mean": self.owner_norm_sum / max(self.tokens_seen, 1),
            "residual_norm_mean": self.residual_norm_sum / max(self.tokens_seen, 1),
            "output_norm_mean": self.output_norm_sum / max(self.tokens_seen, 1),
            "correction_to_output_norm": self.correction_norm_sum / max(self.output_norm_sum, 1.0e-12),
            "module_calls": dict(sorted((self.module_calls or {}).items())),
        }


@dataclass
class ResidualHookStats:
    hook_calls: int = 0
    tokens_seen: int = 0
    capped_tokens: int = 0
    correction_norm_sum: float = 0.0
    projection_norm_sum: float = 0.0
    orthogonal_norm_sum: float = 0.0
    hidden_norm_sum: float = 0.0
    layer_calls: dict[int, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "hook_calls": self.hook_calls,
            "tokens_seen": self.tokens_seen,
            "capped_tokens": self.capped_tokens,
            "capped_token_fraction": self.capped_tokens / max(self.tokens_seen, 1),
            "correction_norm_mean": self.correction_norm_sum / max(self.tokens_seen, 1),
            "projection_norm_mean": self.projection_norm_sum / max(self.tokens_seen, 1),
            "orthogonal_norm_mean": self.orthogonal_norm_sum / max(self.tokens_seen, 1),
            "hidden_norm_mean": self.hidden_norm_sum / max(self.tokens_seen, 1),
            "correction_to_hidden_norm": self.correction_norm_sum / max(self.hidden_norm_sum, 1.0e-12),
            "projection_to_hidden_norm": self.projection_norm_sum / max(self.hidden_norm_sum, 1.0e-12),
            "layer_calls": {str(key): value for key, value in sorted((self.layer_calls or {}).items())},
        }


class HiddenSteerToolProjector:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        basis_manifest: Path,
        strength: float,
        max_alpha: float,
        dtype: torch.dtype,
    ) -> None:
        self.model = model
        self.basis_manifest_path = basis_manifest
        self.manifest = json.loads(basis_manifest.read_text(encoding="utf-8"))
        factor_path = Path(self.manifest["factor_path"])
        self.raw_factors = torch.load(factor_path, map_location="cpu")
        self.strength = float(strength)
        self.max_alpha = float(max_alpha)
        self.dtype = dtype
        self.stats = HookStats(module_calls={})
        self.handles: list[Any] = []
        self.factors: dict[str, dict[str, dict[str, torch.Tensor]]] = {}

    def install(self) -> None:
        target_names = set(self.raw_factors)
        for module_name, module in self.model.named_modules():
            param_name = f"{module_name}.weight"
            if param_name not in target_names:
                continue
            self.handles.append(module.register_forward_hook(self._make_hook(param_name)))
        if not self.handles:
            raise RuntimeError("No HiddenSteer target modules were found in the HF model.")

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _module_factors(self, param_name: str, device: torch.device) -> dict[str, dict[str, torch.Tensor]]:
        cached = self.factors.get(param_name)
        if cached is not None and next(iter(cached.values()))["u"].device == device:
            return cached
        module_factors: dict[str, dict[str, torch.Tensor]] = {}
        for expert, factor in self.raw_factors[param_name].items():
            module_factors[expert] = {
                "u": factor["u"].to(device=device, dtype=self.dtype),
                "s": factor["s"].to(device=device, dtype=self.dtype),
                "v": factor["v"].to(device=device, dtype=self.dtype),
            }
        self.factors[param_name] = module_factors
        return module_factors

    def _lowrank_update(self, x: torch.Tensor, factor: Mapping[str, torch.Tensor]) -> torch.Tensor:
        z = torch.matmul(x.to(dtype=self.dtype), factor["v"])
        z = z * factor["s"]
        return torch.matmul(z, factor["u"].transpose(0, 1))

    def _make_hook(self, param_name: str):
        module_info = {item["param_name"]: item for item in self.manifest["modules"]}[param_name]
        coeff_memory = float(module_info.get("coeff_memory", 1.0))
        coeff_code = float(module_info.get("coeff_code", 1.0))

        def hook(_module: Any, inputs: tuple[Any, ...], output: Any) -> Any:
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                return output
            if not isinstance(output, torch.Tensor) or output.ndim != 3:
                return output
            x = inputs[0]
            if x.ndim != 3:
                return output
            factors = self._module_factors(param_name, output.device)
            owner = self._lowrank_update(x, factors["tool"]).to(dtype=output.dtype)
            residual = (
                coeff_memory * self._lowrank_update(x, factors["memory"])
                + coeff_code * self._lowrank_update(x, factors["code"])
            ).to(dtype=output.dtype)
            denom = owner.float().pow(2).sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
            alpha = (residual.float() * owner.float()).sum(dim=-1, keepdim=True) / denom
            alpha = alpha.clamp(min=-self.max_alpha, max=0.0)
            correction = (-self.strength * alpha).to(dtype=output.dtype) * owner
            token_count = int(output.shape[0] * output.shape[1])
            self.stats.hook_calls += 1
            self.stats.tokens_seen += token_count
            self.stats.conflict_tokens += int((alpha < 0.0).sum().item())
            self.stats.correction_norm_sum += float(correction.float().norm(dim=-1).sum().item())
            self.stats.owner_norm_sum += float(owner.float().norm(dim=-1).sum().item())
            self.stats.residual_norm_sum += float(residual.float().norm(dim=-1).sum().item())
            self.stats.output_norm_sum += float(output.float().norm(dim=-1).sum().item())
            assert self.stats.module_calls is not None
            self.stats.module_calls[param_name] = self.stats.module_calls.get(param_name, 0) + 1
            return output + correction

        return hook


class HiddenSteerResidualProjector:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        basis_manifest: Path,
        strength: float,
        max_correction_ratio: float,
        mode: str,
        dtype: torch.dtype,
    ) -> None:
        self.model = model
        self.basis_manifest_path = basis_manifest
        self.manifest = json.loads(basis_manifest.read_text(encoding="utf-8"))
        basis_path = Path(self.manifest["basis_path"])
        self.raw_basis = torch.load(basis_path, map_location="cpu")
        self.strength = float(strength)
        self.max_correction_ratio = float(max_correction_ratio)
        self.mode = str(mode)
        self.dtype = dtype
        self.stats = ResidualHookStats(layer_calls={})
        self.handles: list[Any] = []
        self.basis_cache: dict[tuple[int, str], torch.Tensor] = {}

    def install(self) -> None:
        target = {f"model.layers.{int(item['layer'])}": int(item["layer"]) for item in self.manifest["layers"]}
        for module_name, module in self.model.named_modules():
            if module_name not in target:
                continue
            self.handles.append(module.register_forward_hook(self._make_hook(target[module_name])))
        if not self.handles:
            raise RuntimeError("No residual-stream HiddenSteer target layers were found in the HF model.")

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _basis_for_layer(self, layer: int, key: str, device: torch.device) -> torch.Tensor:
        cache_key = (layer, key)
        cached = self.basis_cache.get(cache_key)
        if cached is not None and cached.device == device and cached.dtype == self.dtype:
            return cached
        raw = self.raw_basis.get(layer, self.raw_basis.get(str(layer)))
        if raw is None:
            raise KeyError(f"Residual basis does not contain layer {layer}")
        if isinstance(raw, dict):
            basis = raw.get(key)
            if basis is None and key == "success_basis":
                basis = raw.get("basis")
        else:
            basis = raw
        if basis is None:
            raise KeyError(f"Residual basis for layer {layer} does not contain key {key!r}")
        basis = basis.to(device=device, dtype=self.dtype).contiguous()
        self.basis_cache[cache_key] = basis
        return basis

    def _project(self, hidden: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
        work = hidden.to(dtype=self.dtype)
        return torch.matmul(torch.matmul(work, basis), basis.transpose(0, 1))

    def _make_hook(self, layer: int):
        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
            hidden = output[0] if isinstance(output, tuple) else output
            if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
                return output
            success_basis = self._basis_for_layer(layer, "success_basis", hidden.device)
            success_projection = self._project(hidden, success_basis)
            if self.mode == "anchor_boost":
                direction = success_projection
            elif self.mode == "remove_failure":
                failure_basis = self._basis_for_layer(layer, "failure_basis", hidden.device)
                direction = -self._project(hidden, failure_basis)
            elif self.mode == "remove_failure_orthogonal":
                failure_basis = self._basis_for_layer(layer, "failure_orthogonal_basis", hidden.device)
                direction = -self._project(hidden, failure_basis)
            elif self.mode == "boost_success_remove_failure":
                failure_basis = self._basis_for_layer(layer, "failure_orthogonal_basis", hidden.device)
                direction = success_projection - self._project(hidden, failure_basis)
            else:
                raise RuntimeError(f"Unknown residual mode: {self.mode}")
            raw_correction = (self.strength * direction).to(dtype=hidden.dtype)
            hidden_norm = hidden.float().norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
            raw_norm = raw_correction.float().norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
            if self.max_correction_ratio > 0.0:
                limit = self.max_correction_ratio * hidden_norm
                scale = torch.minimum(torch.ones_like(raw_norm), limit / raw_norm)
                correction = (raw_correction.float() * scale).to(dtype=hidden.dtype)
                capped = raw_norm > limit
            else:
                correction = raw_correction
                capped = torch.zeros_like(raw_norm, dtype=torch.bool)
            new_hidden = hidden + correction

            token_count = int(hidden.shape[0] * hidden.shape[1])
            direction_norm = direction.float().norm(dim=-1)
            orthogonal_norm = (hidden.float() - success_projection.float()).norm(dim=-1)
            self.stats.hook_calls += 1
            self.stats.tokens_seen += token_count
            self.stats.capped_tokens += int(capped.sum().item())
            self.stats.correction_norm_sum += float(correction.float().norm(dim=-1).sum().item())
            self.stats.projection_norm_sum += float(direction_norm.sum().item())
            self.stats.orthogonal_norm_sum += float(orthogonal_norm.sum().item())
            self.stats.hidden_norm_sum += float(hidden_norm.squeeze(-1).sum().item())
            assert self.stats.layer_calls is not None
            self.stats.layer_calls[layer] = self.stats.layer_calls.get(layer, 0) + 1
            if isinstance(output, tuple):
                return (new_hidden,) + output[1:]
            return new_hidden

        return hook


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    bfcl_root = Path(args.bfcl_root).expanduser().resolve()
    sys.path.insert(0, str(bfcl_root))

    from bfcl_eval.constants.model_config import ModelConfig
    from bfcl_eval.model_handler.local_inference.qwen import QwenHandler
    from bfcl_eval.model_handler.utils import system_prompt_pre_processing_chat_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[str(args.torch_dtype)]
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype, trust_remote_code=True)
    model.to(args.device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    handler = QwenHandler(
        model_name=str(args.model_path),
        temperature=float(args.temperature),
        registry_name=args.model_name,
        is_fc_model=False,
    )
    handler.tokenizer = tokenizer
    handler.max_context_length = int(getattr(model.config, "max_position_embeddings", tokenizer.model_max_length))

    if args.basis_manifest and args.residual_basis_manifest:
        raise SystemExit("--basis-manifest and --residual-basis-manifest are mutually exclusive.")

    projector = None
    if args.basis_manifest:
        projector = HiddenSteerToolProjector(
            model=model,
            basis_manifest=Path(args.basis_manifest).expanduser().resolve(),
            strength=float(args.projection_strength),
            max_alpha=float(args.max_alpha),
            dtype=dtype,
        )
        projector.install()
    elif args.residual_basis_manifest:
        projector = HiddenSteerResidualProjector(
            model=model,
            basis_manifest=Path(args.residual_basis_manifest).expanduser().resolve(),
            strength=float(args.residual_strength),
            max_correction_ratio=float(args.residual_max_correction_ratio),
            mode=str(args.residual_mode),
            dtype=dtype,
        )
        projector.install()

    summary: dict[str, Any] = {
        "model_path": str(Path(args.model_path).expanduser().resolve()),
        "basis_manifest": str(Path(args.basis_manifest).expanduser().resolve()) if args.basis_manifest else None,
        "residual_basis_manifest": (
            str(Path(args.residual_basis_manifest).expanduser().resolve()) if args.residual_basis_manifest else None
        ),
        "categories": split_csv(args.categories),
        "max_samples_per_category": int(args.max_samples_per_category),
        "max_new_tokens": int(args.max_new_tokens),
        "temperature": float(args.temperature),
        "projection_strength": float(args.projection_strength),
        "max_alpha": float(args.max_alpha),
        "residual_strength": float(args.residual_strength),
        "residual_max_correction_ratio": float(args.residual_max_correction_ratio),
        "residual_mode": str(args.residual_mode),
        "results": {},
    }

    result_dir_arg = output_dir / "result"
    score_dir_arg = output_dir / "score"
    result_root = result_dir_arg / args.model_name
    score_root = score_dir_arg / args.model_name
    try:
        for category in split_csv(args.categories):
            split = "live" if category.startswith("live_") else "non_live"
            result_dir = result_root / split
            result_dir.mkdir(parents=True, exist_ok=True)
            data_path = bfcl_root / "bfcl_eval" / "data" / f"BFCL_v4_{category}.json"
            result_path = result_dir / f"BFCL_v4_{category}_result.json"
            rows = read_jsonl(data_path)[: int(args.max_samples_per_category)]
            category_stats = run_category(
                rows=rows,
                category=category,
                handler=handler,
                tokenizer=tokenizer,
                model=model,
                system_prompt_pre_processing_chat_model=system_prompt_pre_processing_chat_model,
                max_new_tokens=int(args.max_new_tokens),
                temperature=float(args.temperature),
                device=str(args.device),
                result_path=result_path,
            )
            summary["results"][category] = category_stats
        if projector is not None:
            summary["hook_stats"] = projector.stats.to_dict()
    finally:
        if projector is not None:
            projector.remove()
        model.cpu()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.evaluate:
        ensure_model_config(bfcl_root, args.model_name)
        evaluate_summary = run_bfcl_evaluate(
            bfcl_root=bfcl_root,
            model_name=args.model_name,
            categories=str(args.categories),
            result_dir=str(result_dir_arg),
            score_dir=str(score_dir_arg),
            python_bin=Path(args.python_bin).expanduser().resolve(),
        )
        summary["bfcl_evaluate"] = evaluate_summary

    write_json(output_dir / "run_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", default="hiddensteer-hf-tool")
    parser.add_argument("--basis-manifest", default="")
    parser.add_argument("--residual-basis-manifest", default="")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bfcl-root", default=DEFAULT_BFCL_ROOT)
    parser.add_argument("--python-bin", default=DEFAULT_PYTHON)
    parser.add_argument("--categories", default=DEFAULT_CATEGORIES)
    parser.add_argument("--max-samples-per-category", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--projection-strength", type=float, default=1.0)
    parser.add_argument("--max-alpha", type=float, default=0.25)
    parser.add_argument("--residual-strength", type=float, default=0.05)
    parser.add_argument("--residual-max-correction-ratio", type=float, default=0.05)
    parser.add_argument(
        "--residual-mode",
        choices=["anchor_boost", "remove_failure", "remove_failure_orthogonal", "boost_success_remove_failure"],
        default="anchor_boost",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--evaluate", action="store_true")
    return parser.parse_args()


def run_category(
    *,
    rows: list[dict[str, Any]],
    category: str,
    handler: Any,
    tokenizer: Any,
    model: torch.nn.Module,
    system_prompt_pre_processing_chat_model: Any,
    max_new_tokens: int,
    temperature: float,
    device: str,
    result_path: Path,
) -> dict[str, Any]:
    total_input = 0
    total_output = 0
    total_latency = 0.0
    started = time.perf_counter()
    with result_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            test_entry = copy.deepcopy(row)
            test_entry["question"][0] = system_prompt_pre_processing_chat_model(
                test_entry["question"][0], test_entry["function"], test_entry["id"]
            )
            prompt = handler._format_prompt(test_entry["question"][0], test_entry["function"])
            input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
            attention_mask = torch.ones_like(input_ids)
            gen_started = time.perf_counter()
            generate_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0.0,
                "pad_token_id": tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if temperature > 0.0:
                generate_kwargs["temperature"] = temperature
            with torch.no_grad():
                generated = model.generate(**generate_kwargs)
            latency = time.perf_counter() - gen_started
            output_ids = generated[0, input_ids.shape[1] :]
            text = tokenizer.decode(output_ids, skip_special_tokens=True)
            input_count = int(input_ids.numel())
            output_count = int(output_ids.numel())
            total_input += input_count
            total_output += output_count
            total_latency += latency
            handle.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "result": text,
                        "input_token_count": input_count,
                        "output_token_count": output_count,
                        "latency": latency,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    wall = time.perf_counter() - started
    return {
        "category": category,
        "samples": len(rows),
        "input_tokens": total_input,
        "output_tokens": total_output,
        "generation_latency_sec": total_latency,
        "wall_time_sec": wall,
        "samples_per_min": len(rows) / max(wall, 1.0e-12) * 60.0,
        "output_tokens_per_sec": total_output / max(total_latency, 1.0e-12),
        "result_path": str(result_path),
    }


def run_bfcl_evaluate(
    *,
    bfcl_root: Path,
    model_name: str,
    categories: str,
    result_dir: str,
    score_dir: str,
    python_bin: Path,
) -> dict[str, Any]:
    env = os.environ.copy()
    conda_bin = str(python_bin.parent)
    env["PATH"] = conda_bin + os.pathsep + env.get("PATH", "")
    cmd = [
        "bfcl",
        "evaluate",
        "--model",
        model_name,
        "--test-category",
        categories,
        "--result-dir",
        result_dir,
        "--score-dir",
        score_dir,
        "--partial-eval",
    ]
    completed = subprocess.run(cmd, cwd=bfcl_root, env=env, text=True, capture_output=True, check=False)
    summary = {"cmd": cmd, "returncode": completed.returncode, "stdout_tail": completed.stdout[-4000:], "stderr_tail": completed.stderr[-4000:]}
    score_root = Path(score_dir)
    if not score_root.is_absolute():
        score_root = bfcl_root / score_root
    model_score_root = score_root / model_name
    scores = {}
    for category in split_csv(categories):
        split = "live" if category.startswith("live_") else "non_live"
        path = model_score_root / split / f"BFCL_v4_{category}_score.json"
        if not path.exists():
            continue
        lines = read_jsonl(path)
        if lines and "accuracy" in lines[0]:
            scores[category] = lines[0]
    summary["scores"] = scores
    return summary


def ensure_model_config(bfcl_root: Path, model_name: str) -> None:
    path = bfcl_root / "bfcl_eval" / "constants" / "model_config.py"
    text = path.read_text(encoding="utf-8")
    if f'"{model_name}": ModelConfig(' in text or f"'{model_name}': ModelConfig(" in text:
        return
    marker = "\nMODEL_CONFIG_MAPPING ="
    marker_idx = text.find(marker)
    if marker_idx < 0:
        raise RuntimeError(f"Could not find MODEL_CONFIG_MAPPING marker in {path}")
    insert_idx = text.rfind("\n}", 0, marker_idx)
    if insert_idx < 0:
        raise RuntimeError(f"Could not find local_inference_model_map closing brace in {path}")
    block = f'''
    "{model_name}": ModelConfig(
      model_name="{model_name}",
      display_name="{model_name}",
      url="",
      org="local",
      license="",
      model_handler=QwenHandler,
      input_price=None,
      output_price=None,
      is_fc_model=False,
      underscore_to_dot=False,
  ),
'''
    path.write_text(text[:insert_idx] + block + text[insert_idx:], encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def split_csv(raw: str) -> list[str]:
    return [item.strip() for item in str(raw or "").split(",") if item.strip()]


if __name__ == "__main__":
    main()
