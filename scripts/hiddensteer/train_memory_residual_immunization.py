#!/usr/bin/env python3
"""Small Memory-Conditioned Residual Immunization prototype.

This script is intentionally a feasibility harness.  It trains a tiny
zero-output low-rank residual corrector on Memory prompts only:

    h' = h + U_l V_l h

The offline geometry is built on the same Memory teacher trajectories:

    memory_anchor = h_memory_expert - h_merged
    code_on_memory = h_code_expert - h_merged
    code_bad = code_on_memory orthogonalized against memory_anchor

At training time the corrector minimizes teacher-output NLL on Memory
trajectory responses while penalizing corrected activations that remain in the
code_bad subspace.  The goal is not to replace the full HotpotQA harness; it is
to check whether this method has a small-scale signal before any vLLM work.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


DEFAULT_MODEL_PATH = "/tmp/shared-storage/OnPolicy/checkpoints/rcrf_code_spanaware_conservative_v2"
DEFAULT_MEMORY_MODEL_PATH = "/mnt/cache/wuruixiao/models/RL-MemoryAgent-7B"
DEFAULT_CODE_MODEL_PATH = "/mnt/cache/wuruixiao/models/ReasonFlux-Coder-7B"
DEFAULT_ROLLOUT_PATH = (
    "/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/"
    "memory_expert_paper96_s2_seed20260514.jsonl"
)
DEFAULT_TOOL_RETENTION_ROLLOUT = (
    "/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/"
    "tool_expert_paper96_s2_seed20260514.jsonl"
)
DEFAULT_CODE_RETENTION_ROLLOUT = (
    "/tmp/shared-storage/OnPolicy/data/calibration/paper96_expert_rollouts_seed20260514/"
    "code_expert_paper96_s2_seed20260514.jsonl"
)
DEFAULT_OUTPUT_DIR = "/tmp/shared-storage/ExpertGym/hiddensteer/memory_residual_immunization_smoke_20260525"


@dataclass
class Example:
    prompt_id: str
    sample_id: str
    turn: int
    kind: str
    prompt_text: str
    response_text: str


@dataclass
class EncodedExample:
    example: Example
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    predict_positions: torch.Tensor
    response_tokens: int


class LowRankResidualCorrector(torch.nn.Module):
    def __init__(self, *, layers: list[int], hidden_size: int, rank: int, scale: float) -> None:
        super().__init__()
        self.layers = list(layers)
        self.rank = int(rank)
        self.scale = float(scale)
        self.down = torch.nn.ParameterDict()
        self.up = torch.nn.ParameterDict()
        for layer in self.layers:
            down = torch.empty(self.rank, hidden_size)
            torch.nn.init.normal_(down, mean=0.0, std=0.02)
            up = torch.zeros(hidden_size, self.rank)
            self.down[str(layer)] = torch.nn.Parameter(down)
            self.up[str(layer)] = torch.nn.Parameter(up)
        self.current_positions: torch.Tensor | None = None
        self.last_hidden: dict[int, torch.Tensor] = {}
        self.last_correction: dict[int, torch.Tensor] = {}
        self.stats: dict[str, float] = defaultdict(float)
        self.enabled = True

    def clear_step(self, positions: torch.Tensor | None) -> None:
        self.current_positions = positions
        self.last_hidden.clear()
        self.last_correction.clear()

    def forward_layer(self, layer: int, hidden: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return hidden
        down = self.down[str(layer)].to(device=hidden.device, dtype=hidden.dtype)
        up = self.up[str(layer)].to(device=hidden.device, dtype=hidden.dtype)
        low = torch.matmul(hidden, down.transpose(0, 1))
        correction = self.scale * torch.matmul(low, up.transpose(0, 1))
        out = hidden + correction
        positions = self.current_positions
        if positions is not None and positions.numel() > 0:
            pos = positions.to(hidden.device)
            self.last_hidden[layer] = out[0, pos, :]
            self.last_correction[layer] = correction[0, pos, :]
        self.stats["hook_calls"] += 1.0
        self.stats["tokens_seen"] += float(hidden.shape[0] * hidden.shape[1])
        self.stats["correction_norm_sum"] += float(correction.detach().float().norm(dim=-1).sum().item())
        self.stats["hidden_norm_sum"] += float(hidden.detach().float().norm(dim=-1).sum().item())
        return out


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(int(args.seed))

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[str(args.torch_dtype)]
    layers = parse_layers(args.layers)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    examples = load_memory_examples(
        Path(args.rollout_path).expanduser().resolve(),
        turn_kinds=split_csv(args.turn_kinds),
        max_examples=int(args.max_examples),
        max_turns_per_prompt=int(args.max_turns_per_prompt),
        prefer_success=not bool(args.allow_failed_teacher_samples),
    )
    if len(examples) < 2:
        raise SystemExit("Need at least two Memory teacher examples for train/heldout split.")
    train_examples, heldout_examples = split_examples_by_prompt(
        examples,
        train_count=int(args.train_examples),
        heldout_count=int(args.heldout_examples),
        seed=int(args.seed),
    )

    encoded_train = [
        encode_example(
            tokenizer,
            ex,
            max_seq_length=int(args.max_seq_length),
            response_tail_tokens=int(args.response_tail_tokens),
            device=str(args.device),
        )
        for ex in train_examples
    ]
    encoded_heldout = [
        encode_example(
            tokenizer,
            ex,
            max_seq_length=int(args.max_seq_length),
            response_tail_tokens=int(args.response_tail_tokens),
            device=str(args.device),
        )
        for ex in heldout_examples
    ]

    geometry_path = Path(args.geometry_path).expanduser().resolve() if args.geometry_path else output_dir / "memory_code_bad_basis.pt"
    geometry_examples = encoded_train + encoded_heldout
    if args.geometry_path:
        geometry = torch.load(geometry_path, map_location="cpu")
    else:
        geometry = build_code_bad_geometry(
            args=args,
            tokenizer=tokenizer,
            encoded_examples=geometry_examples,
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

    hidden_size = int(getattr(model.config, "hidden_size"))
    corrector = LowRankResidualCorrector(
        layers=layers,
        hidden_size=hidden_size,
        rank=int(args.corrector_rank),
        scale=float(args.corrector_scale),
    ).to(args.device)
    hooks = install_corrector_hooks(model, corrector)
    optimizer = torch.optim.AdamW(corrector.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    baseline_train = evaluate_nll(model, corrector, encoded_train, geometry, enabled=False)
    baseline_heldout = evaluate_nll(model, corrector, encoded_heldout, geometry, enabled=False)

    train_history = []
    started_train = time.perf_counter()
    try:
        for step in range(1, int(args.steps) + 1):
            encoded = encoded_train[(step - 1) % len(encoded_train)]
            optimizer.zero_grad(set_to_none=True)
            loss_payload = forward_loss(
                model=model,
                corrector=corrector,
                encoded=encoded,
                geometry=geometry,
                enabled=True,
                requires_grad=True,
                code_bad_weight=float(args.code_bad_weight),
                correction_norm_weight=float(args.correction_norm_weight),
            )
            loss_payload["loss"].backward()
            torch.nn.utils.clip_grad_norm_(corrector.parameters(), max_norm=float(args.max_grad_norm))
            optimizer.step()
            if step == 1 or step % int(args.log_every) == 0 or step == int(args.steps):
                train_history.append(
                    {
                        "step": step,
                        "loss": float(loss_payload["loss"].detach().item()),
                        "nll": float(loss_payload["nll"].detach().item()),
                        "code_bad_penalty": float(loss_payload["code_bad_penalty"].detach().item()),
                        "correction_norm_penalty": float(loss_payload["correction_norm_penalty"].detach().item()),
                    }
                )
    finally:
        train_wall = time.perf_counter() - started_train

    corrected_train = evaluate_nll(model, corrector, encoded_train, geometry, enabled=True)
    corrected_heldout = evaluate_nll(model, corrector, encoded_heldout, geometry, enabled=True)
    retention = evaluate_retention(
        model=model,
        tokenizer=tokenizer,
        corrector=corrector,
        geometry=geometry,
        args=args,
    )
    generation = generation_smoke(
        model=model,
        tokenizer=tokenizer,
        corrector=corrector,
        examples=encoded_heldout[: int(args.generation_smoke)],
        max_new_tokens=int(args.generation_max_new_tokens),
        device=str(args.device),
    )

    for handle in hooks:
        handle.remove()
    model.cpu()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    corrector_path = output_dir / "memory_corrector.pt"
    torch.save(
        {
            "state_dict": {key: value.detach().cpu() for key, value in corrector.state_dict().items()},
            "layers": layers,
            "rank": int(args.corrector_rank),
            "scale": float(args.corrector_scale),
            "model_path": str(Path(args.model_path).expanduser().resolve()),
        },
        corrector_path,
    )
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": "Memory-Conditioned Residual Immunization smoke",
        "model_path": str(Path(args.model_path).expanduser().resolve()),
        "memory_model_path": str(Path(args.memory_model_path).expanduser().resolve()),
        "code_model_path": str(Path(args.code_model_path).expanduser().resolve()),
        "rollout_path": str(Path(args.rollout_path).expanduser().resolve()),
        "layers": layers,
        "basis_rank": int(args.basis_rank),
        "corrector_rank": int(args.corrector_rank),
        "train_examples": [compact_example(item.example) for item in encoded_train],
        "heldout_examples": [compact_example(item.example) for item in encoded_heldout],
        "baseline_train": baseline_train,
        "baseline_heldout": baseline_heldout,
        "corrected_train": corrected_train,
        "corrected_heldout": corrected_heldout,
        "train_history": train_history,
        "train_wall_time_sec": train_wall,
        "retention": retention,
        "generation_smoke": generation,
        "corrector_path": str(corrector_path),
        "geometry_path": str(geometry_path),
        "interpretation": interpret_result(baseline_train, baseline_heldout, corrected_train, corrected_heldout),
    }
    write_json(output_dir / "run_summary.json", summary)
    write_markdown(output_dir / "README.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--memory-model-path", default=DEFAULT_MEMORY_MODEL_PATH)
    parser.add_argument("--code-model-path", default=DEFAULT_CODE_MODEL_PATH)
    parser.add_argument("--rollout-path", default=DEFAULT_ROLLOUT_PATH)
    parser.add_argument("--geometry-path", default="")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--layers", default="24-27")
    parser.add_argument("--turn-kinds", default="final_answer")
    parser.add_argument("--max-examples", type=int, default=10)
    parser.add_argument("--max-turns-per-prompt", type=int, default=1)
    parser.add_argument("--train-examples", type=int, default=6)
    parser.add_argument("--heldout-examples", type=int, default=4)
    parser.add_argument("--basis-rank", type=int, default=4)
    parser.add_argument("--corrector-rank", type=int, default=4)
    parser.add_argument("--corrector-scale", type=float, default=1.0)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--response-tail-tokens", type=int, default=128)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2.0e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--code-bad-weight", type=float, default=0.05)
    parser.add_argument("--correction-norm-weight", type=float, default=0.01)
    parser.add_argument("--generation-smoke", type=int, default=2)
    parser.add_argument("--generation-max-new-tokens", type=int, default=64)
    parser.add_argument("--tool-retention-rollout", default=DEFAULT_TOOL_RETENTION_ROLLOUT)
    parser.add_argument("--code-retention-rollout", default=DEFAULT_CODE_RETENTION_ROLLOUT)
    parser.add_argument("--retention-examples", type=int, default=4)
    parser.add_argument("--retention-response-tail-tokens", type=int, default=128)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--allow-failed-teacher-samples", action="store_true")
    return parser.parse_args()


def load_memory_examples(
    path: Path,
    *,
    turn_kinds: list[str],
    max_examples: int,
    max_turns_per_prompt: int,
    prefer_success: bool,
) -> list[Example]:
    output = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            samples = row.get("samples") or []
            if prefer_success:
                samples = [sample for sample in samples if sample.get("success")]
            if not samples:
                continue
            sample = samples[0]
            turns_for_prompt = 0
            for turn in sample.get("trajectory") or []:
                if turn.get("kind") not in turn_kinds:
                    continue
                prompt_text = str(turn.get("prompt_text") or "")
                response_text = str(turn.get("text") or "")
                if not prompt_text.strip() or not response_text.strip():
                    continue
                output.append(
                    Example(
                        prompt_id=str(row.get("prompt_id")),
                        sample_id=str(sample.get("sample_id")),
                        turn=int(turn.get("turn") or 0),
                        kind=str(turn.get("kind")),
                        prompt_text=prompt_text,
                        response_text=response_text,
                    )
                )
                turns_for_prompt += 1
                if len(output) >= max_examples:
                    return output
                if max_turns_per_prompt > 0 and turns_for_prompt >= max_turns_per_prompt:
                    break
    return output


def load_rollout_response_examples(
    path: Path,
    *,
    task: str,
    max_examples: int,
    prefer_success: bool,
) -> list[Example]:
    output = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            samples = row.get("samples") or []
            if prefer_success:
                samples = [sample for sample in samples if sample.get("success")]
            if not samples:
                continue
            sample = samples[0]
            prompt_text = str(row.get("rendered_prompt") or "")
            response_text = str(sample.get("text") or "")
            if not prompt_text.strip() or not response_text.strip():
                continue
            output.append(
                Example(
                    prompt_id=str(row.get("prompt_id")),
                    sample_id=str(sample.get("sample_id")),
                    turn=0,
                    kind=f"{task}_retention",
                    prompt_text=prompt_text,
                    response_text=response_text,
                )
            )
            if len(output) >= max_examples:
                break
    return output


def split_examples_by_prompt(
    examples: list[Example],
    *,
    train_count: int,
    heldout_count: int,
    seed: int,
) -> tuple[list[Example], list[Example]]:
    grouped: dict[str, list[Example]] = defaultdict(list)
    for example in examples:
        grouped[example.prompt_id].append(example)
    groups = list(grouped.values())
    random.Random(seed).shuffle(groups)
    train: list[Example] = []
    heldout: list[Example] = []
    for group in groups:
        if len(train) < train_count:
            train.extend(group[: max(train_count - len(train), 0)])
        elif len(heldout) < heldout_count:
            heldout.extend(group[: max(heldout_count - len(heldout), 0)])
        if len(train) >= train_count and len(heldout) >= heldout_count:
            break
    if not heldout:
        raise SystemExit("No heldout examples after prompt-group split; increase --max-examples.")
    return train, heldout


def encode_example(
    tokenizer: Any,
    example: Example,
    *,
    max_seq_length: int,
    response_tail_tokens: int,
    device: str,
) -> EncodedExample:
    prompt_ids = tokenizer(example.prompt_text, add_special_tokens=False).input_ids
    response_ids = tokenizer(example.response_text, add_special_tokens=False).input_ids
    if not prompt_ids or not response_ids:
        raise ValueError(f"Empty prompt or response for {example.prompt_id}")
    ids = prompt_ids + response_ids
    prompt_len = len(prompt_ids)
    label_positions = list(range(prompt_len, len(ids)))
    predict_positions = [pos - 1 for pos in label_positions if pos > 0]
    if response_tail_tokens > 0 and len(label_positions) > response_tail_tokens:
        label_positions = label_positions[-response_tail_tokens:]
        predict_positions = predict_positions[-response_tail_tokens:]
    if len(ids) > max_seq_length:
        overflow = len(ids) - max_seq_length
        ids = ids[overflow:]
        label_positions = [pos - overflow for pos in label_positions if pos >= overflow]
        predict_positions = [pos - overflow for pos in predict_positions if pos >= overflow]
    if not label_positions:
        label_positions = [len(ids) - 1]
        predict_positions = [max(len(ids) - 2, 0)]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    labels = torch.full_like(input_ids, -100)
    for pos in label_positions:
        if 0 <= pos < labels.shape[1]:
            labels[0, pos] = input_ids[0, pos]
    attention_mask = torch.ones_like(input_ids)
    return EncodedExample(
        example=example,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        predict_positions=torch.tensor(predict_positions, dtype=torch.long, device=device),
        response_tokens=int((labels != -100).sum().item()),
    )


def build_code_bad_geometry(
    *,
    args: argparse.Namespace,
    tokenizer: Any,
    encoded_examples: list[EncodedExample],
    layers: list[int],
    dtype: torch.dtype,
    AutoModelForCausalLM: Any,
) -> dict[int, dict[str, torch.Tensor]]:
    del tokenizer
    merged = collect_hidden_vectors(
        model_path=str(args.model_path),
        encoded_examples=encoded_examples,
        layers=layers,
        dtype=dtype,
        device=str(args.device),
        AutoModelForCausalLM=AutoModelForCausalLM,
    )
    memory = collect_hidden_vectors(
        model_path=str(args.memory_model_path),
        encoded_examples=encoded_examples,
        layers=layers,
        dtype=dtype,
        device=str(args.device),
        AutoModelForCausalLM=AutoModelForCausalLM,
    )
    code = collect_hidden_vectors(
        model_path=str(args.code_model_path),
        encoded_examples=encoded_examples,
        layers=layers,
        dtype=dtype,
        device=str(args.device),
        AutoModelForCausalLM=AutoModelForCausalLM,
    )
    geometry: dict[int, dict[str, torch.Tensor]] = {}
    for layer in layers:
        memory_delta = normalize_rows(memory[layer] - merged[layer])
        code_delta = normalize_rows(code[layer] - merged[layer])
        memory_basis = fit_basis(memory_delta, rank=int(args.basis_rank))
        code_basis = fit_basis(code_delta, rank=int(args.basis_rank))
        code_bad = orthogonalize_against(code_basis, memory_basis)
        geometry[layer] = {
            "memory_basis": memory_basis.cpu(),
            "code_basis": code_basis.cpu(),
            "code_bad_basis": code_bad.cpu(),
            "num_vectors": torch.tensor([memory_delta.shape[0]], dtype=torch.long),
        }
    return geometry


def collect_hidden_vectors(
    *,
    model_path: str,
    encoded_examples: list[EncodedExample],
    layers: list[int],
    dtype: torch.dtype,
    device: str,
    AutoModelForCausalLM: Any,
) -> dict[int, torch.Tensor]:
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
    model.to(device)
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    captured: dict[int, torch.Tensor] = {}
    chunks: dict[int, list[torch.Tensor]] = defaultdict(list)
    hooks = install_capture_hooks(model, layers, captured)
    try:
        with torch.no_grad():
            for encoded in encoded_examples:
                captured.clear()
                _ = model(
                    input_ids=encoded.input_ids,
                    attention_mask=encoded.attention_mask,
                    use_cache=False,
                )
                positions = encoded.predict_positions
                for layer in layers:
                    hidden = captured[layer]
                    chunks[layer].append(hidden[0, positions.to(hidden.device), :].detach().float().cpu())
    finally:
        for handle in hooks:
            handle.remove()
        model.cpu()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {layer: torch.cat(chunks[layer], dim=0) for layer in layers}


def install_capture_hooks(model: torch.nn.Module, layers: list[int], captured: dict[int, torch.Tensor]) -> list[Any]:
    target = {f"model.layers.{layer}": layer for layer in layers}
    hooks = []
    for module_name, module in model.named_modules():
        if module_name not in target:
            continue
        layer = target[module_name]

        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any, *, layer_index: int = layer) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            if isinstance(hidden, torch.Tensor) and hidden.ndim == 3:
                captured[layer_index] = hidden.detach()

        hooks.append(module.register_forward_hook(hook))
    if len(hooks) != len(layers):
        raise RuntimeError(f"Expected {len(layers)} layer hooks, installed {len(hooks)}")
    return hooks


def install_corrector_hooks(model: torch.nn.Module, corrector: LowRankResidualCorrector) -> list[Any]:
    target = {f"model.layers.{layer}": layer for layer in corrector.layers}
    hooks = []
    for module_name, module in model.named_modules():
        if module_name not in target:
            continue
        layer = target[module_name]

        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any, *, layer_index: int = layer) -> Any:
            hidden = output[0] if isinstance(output, tuple) else output
            if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
                return output
            corrected = corrector.forward_layer(layer_index, hidden)
            if isinstance(output, tuple):
                return (corrected,) + output[1:]
            return corrected

        hooks.append(module.register_forward_hook(hook))
    if len(hooks) != len(corrector.layers):
        raise RuntimeError(f"Expected {len(corrector.layers)} corrector hooks, installed {len(hooks)}")
    return hooks


def forward_loss(
    *,
    model: torch.nn.Module,
    corrector: LowRankResidualCorrector,
    encoded: EncodedExample,
    geometry: dict[int, dict[str, torch.Tensor]],
    enabled: bool,
    requires_grad: bool,
    code_bad_weight: float,
    correction_norm_weight: float,
) -> dict[str, torch.Tensor]:
    positions = encoded.predict_positions if enabled else None
    corrector.clear_step(positions)
    old_enabled = corrector.enabled
    corrector.enabled = enabled
    with torch.set_grad_enabled(requires_grad):
        try:
            outputs = model(
                input_ids=encoded.input_ids,
                attention_mask=encoded.attention_mask,
                use_cache=False,
            )
            logits = outputs.logits
            nll = shifted_cross_entropy(logits, encoded.labels)
            if enabled and code_bad_weight != 0.0:
                code_bad_penalty = compute_code_bad_penalty(corrector, geometry)
            else:
                code_bad_penalty = logits.new_tensor(0.0)
            if enabled and correction_norm_weight != 0.0:
                correction_norm_penalty = compute_correction_norm_penalty(corrector)
            else:
                correction_norm_penalty = logits.new_tensor(0.0)
            loss = nll + code_bad_weight * code_bad_penalty + correction_norm_weight * correction_norm_penalty
        finally:
            corrector.enabled = old_enabled
    return {
        "loss": loss,
        "nll": nll,
        "code_bad_penalty": code_bad_penalty,
        "correction_norm_penalty": correction_norm_penalty,
    }


def shifted_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def compute_code_bad_penalty(
    corrector: LowRankResidualCorrector,
    geometry: dict[int, dict[str, torch.Tensor]],
) -> torch.Tensor:
    penalties = []
    for layer, hidden in corrector.last_hidden.items():
        basis = geometry[layer]["code_bad_basis"].to(device=hidden.device, dtype=hidden.dtype)
        if basis.numel() == 0:
            continue
        projection = torch.matmul(torch.matmul(hidden, basis), basis.transpose(0, 1))
        numerator = projection.float().pow(2).sum(dim=-1)
        denominator = hidden.float().pow(2).sum(dim=-1).clamp_min(1.0e-12)
        penalties.append((numerator / denominator).mean())
    if not penalties:
        return next(iter(corrector.parameters())).new_tensor(0.0)
    return torch.stack(penalties).mean()


def compute_correction_norm_penalty(corrector: LowRankResidualCorrector) -> torch.Tensor:
    penalties = []
    for layer, correction in corrector.last_correction.items():
        hidden = corrector.last_hidden[layer]
        numerator = correction.float().pow(2).sum(dim=-1)
        denominator = hidden.float().pow(2).sum(dim=-1).clamp_min(1.0e-12)
        penalties.append((numerator / denominator).mean())
    if not penalties:
        return next(iter(corrector.parameters())).new_tensor(0.0)
    return torch.stack(penalties).mean()


def evaluate_nll(
    model: torch.nn.Module,
    corrector: LowRankResidualCorrector,
    encoded_examples: list[EncodedExample],
    geometry: dict[int, dict[str, torch.Tensor]],
    *,
    enabled: bool,
) -> dict[str, float]:
    total_loss = 0.0
    total_tokens = 0
    wall_started = time.perf_counter()
    with torch.no_grad():
        for encoded in encoded_examples:
            payload = forward_loss(
                model=model,
                corrector=corrector,
                encoded=encoded,
                geometry=geometry,
                enabled=enabled,
                requires_grad=False,
                code_bad_weight=0.0,
                correction_norm_weight=0.0,
            )
            total_loss += float(payload["nll"].detach().item()) * encoded.response_tokens
            total_tokens += encoded.response_tokens
    wall = time.perf_counter() - wall_started
    mean_nll = total_loss / max(total_tokens, 1)
    return {
        "examples": len(encoded_examples),
        "response_tokens": total_tokens,
        "mean_nll": mean_nll,
        "perplexity": float(torch.exp(torch.tensor(min(mean_nll, 20.0))).item()),
        "wall_time_sec": wall,
        "response_tokens_per_sec": total_tokens / max(wall, 1.0e-12),
    }


def evaluate_retention(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    corrector: LowRankResidualCorrector,
    geometry: dict[int, dict[str, torch.Tensor]],
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    specs = [
        ("tool", Path(args.tool_retention_rollout).expanduser().resolve()),
        ("code", Path(args.code_retention_rollout).expanduser().resolve()),
    ]
    for task, path in specs:
        if int(args.retention_examples) <= 0:
            continue
        examples = load_rollout_response_examples(
            path,
            task=task,
            max_examples=int(args.retention_examples),
            prefer_success=True,
        )
        encoded = [
            encode_example(
                tokenizer,
                example,
                max_seq_length=int(args.max_seq_length),
                response_tail_tokens=int(args.retention_response_tail_tokens),
                device=str(args.device),
            )
            for example in examples
        ]
        baseline = evaluate_nll(model, corrector, encoded, geometry, enabled=False)
        corrected = evaluate_nll(model, corrector, encoded, geometry, enabled=True)
        output[task] = {
            "rollout_path": str(path),
            "examples": [compact_example(item.example) for item in encoded],
            "baseline": baseline,
            "corrected": corrected,
            "nll_delta": corrected["mean_nll"] - baseline["mean_nll"],
            "wall_time_ratio": corrected["wall_time_sec"] / max(baseline["wall_time_sec"], 1.0e-12),
            "retention_ok_by_nll": corrected["mean_nll"] <= baseline["mean_nll"] + 0.02,
        }
    return output


def generation_smoke(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    corrector: LowRankResidualCorrector,
    examples: list[EncodedExample],
    max_new_tokens: int,
    device: str,
) -> list[dict[str, Any]]:
    output = []
    for encoded in examples:
        prompt_ids = encoded.input_ids[:, : first_label_position(encoded.labels)]
        attention_mask = torch.ones_like(prompt_ids)
        baseline = run_generate_once(
            model=model,
            tokenizer=tokenizer,
            corrector=corrector,
            prompt_ids=prompt_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            device=device,
            enabled=False,
        )
        corrected = run_generate_once(
            model=model,
            tokenizer=tokenizer,
            corrector=corrector,
            prompt_ids=prompt_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            device=device,
            enabled=True,
        )
        output.append(
            {
                "prompt_id": encoded.example.prompt_id,
                "kind": encoded.example.kind,
                "teacher_preview": encoded.example.response_text[:300],
                "baseline_preview": baseline["text"][:500],
                "corrected_preview": corrected["text"][:500],
                "same_output": baseline["text"] == corrected["text"],
                "baseline_output_tokens": baseline["output_tokens"],
                "corrected_output_tokens": corrected["output_tokens"],
                "baseline_wall_time_sec": baseline["wall_time_sec"],
                "corrected_wall_time_sec": corrected["wall_time_sec"],
                "wall_time_ratio": corrected["wall_time_sec"] / max(baseline["wall_time_sec"], 1.0e-12),
            }
        )
    return output


def run_generate_once(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    corrector: LowRankResidualCorrector,
    prompt_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    device: str,
    enabled: bool,
) -> dict[str, Any]:
    corrector.clear_step(None)
    old_enabled = corrector.enabled
    corrector.enabled = enabled
    started = time.perf_counter()
    try:
        with torch.no_grad():
            generated = model.generate(
                input_ids=prompt_ids.to(device),
                attention_mask=attention_mask.to(device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    finally:
        corrector.enabled = old_enabled
    wall = time.perf_counter() - started
    text = tokenizer.decode(generated[0, prompt_ids.shape[1] :], skip_special_tokens=True)
    return {
        "text": text,
        "output_tokens": int(generated.shape[1] - prompt_ids.shape[1]),
        "wall_time_sec": wall,
    }


def first_label_position(labels: torch.Tensor) -> int:
    positions = (labels[0] != -100).nonzero(as_tuple=False)
    if positions.numel() == 0:
        return labels.shape[1]
    return int(positions[0].item())


def normalize_rows(matrix: torch.Tensor) -> torch.Tensor:
    return matrix / matrix.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)


def fit_basis(matrix: torch.Tensor, *, rank: int) -> torch.Tensor:
    actual_rank = min(rank, matrix.shape[0], matrix.shape[1])
    if actual_rank <= 0:
        return matrix.new_zeros((matrix.shape[1], 0))
    q = min(actual_rank + 4, matrix.shape[0], matrix.shape[1])
    _u, _s, v = torch.pca_lowrank(matrix.float(), q=q, center=False, niter=2)
    basis = v[:, :actual_rank].contiguous()
    basis, _ = torch.linalg.qr(basis, mode="reduced")
    return basis


def orthogonalize_against(source_basis: torch.Tensor, reference_basis: torch.Tensor) -> torch.Tensor:
    if source_basis.numel() == 0:
        return source_basis
    residual = source_basis - reference_basis @ (reference_basis.transpose(0, 1) @ source_basis)
    keep = residual.norm(dim=0) > 1.0e-6
    residual = residual[:, keep]
    if residual.numel() == 0:
        return residual.reshape(source_basis.shape[0], 0)
    residual, _ = torch.linalg.qr(residual.contiguous(), mode="reduced")
    return residual


def parse_layers(raw: str) -> list[int]:
    layers = []
    for item in split_csv(raw):
        if "-" in item:
            lo, hi = item.split("-", 1)
            layers.extend(range(int(lo), int(hi) + 1))
        else:
            layers.append(int(item))
    return sorted(dict.fromkeys(layers))


def split_csv(raw: str) -> list[str]:
    return [item.strip() for item in str(raw or "").split(",") if item.strip()]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compact_example(example: Example) -> dict[str, Any]:
    return {
        "prompt_id": example.prompt_id,
        "sample_id": example.sample_id,
        "turn": example.turn,
        "kind": example.kind,
        "response_preview": example.response_text[:160],
    }


def interpret_result(
    baseline_train: dict[str, float],
    baseline_heldout: dict[str, float],
    corrected_train: dict[str, float],
    corrected_heldout: dict[str, float],
) -> str:
    train_delta = baseline_train["mean_nll"] - corrected_train["mean_nll"]
    heldout_delta = baseline_heldout["mean_nll"] - corrected_heldout["mean_nll"]
    if train_delta > 0.0 and heldout_delta > 0.0:
        return "positive_teacher_loss_signal"
    if train_delta > 0.0 and heldout_delta <= 0.0:
        return "train_only_signal_overfit_risk"
    return "no_teacher_loss_signal"


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Memory-Conditioned Residual Immunization Smoke",
        "",
        f"- Created: `{summary['created_at']}`",
        f"- Interpretation: `{summary['interpretation']}`",
        f"- Model: `{summary['model_path']}`",
        f"- Memory teacher: `{summary['memory_model_path']}`",
        f"- Code expert: `{summary['code_model_path']}`",
        f"- Layers: `{summary['layers']}`",
        f"- Corrector: rank `{summary['corrector_rank']}`",
        "",
        "| split | baseline NLL | corrected NLL | delta | baseline tok/s | corrected tok/s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split in ["train", "heldout"]:
        base = summary[f"baseline_{split}"]
        corr = summary[f"corrected_{split}"]
        lines.append(
            f"| {split} | {base['mean_nll']:.4f} | {corr['mean_nll']:.4f} | "
            f"{base['mean_nll'] - corr['mean_nll']:.4f} | "
            f"{base['response_tokens_per_sec']:.2f} | {corr['response_tokens_per_sec']:.2f} |"
        )
    if summary.get("retention"):
        lines.extend(
            [
                "",
                "## Retention",
                "",
                "| task | baseline NLL | corrected NLL | corrected - baseline | wall ratio | retention ok |",
                "|---|---:|---:|---:|---:|---|",
            ]
        )
        for task, payload in summary["retention"].items():
            base = payload["baseline"]
            corr = payload["corrected"]
            lines.append(
                f"| {task} | {base['mean_nll']:.4f} | {corr['mean_nll']:.4f} | "
                f"{payload['nll_delta']:.4f} | {payload['wall_time_ratio']:.3f} | "
                f"{payload['retention_ok_by_nll']} |"
            )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This is a teacher-loss smoke, not a full HotpotQA score.",
            "- A positive result only permits a heldout generation/eval_50 follow-up.",
            "- A train-only result should be treated as overfit risk, not method success.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
