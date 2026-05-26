#!/usr/bin/env python3
"""Build residual-stream Tool activation bases from successful BFCL outputs.

This is the second HiddenSteer prototype after module-output task-vector
projection proved too weak.  It uses successful Tool generations from the same
HF model, teacher-forces prompt + response, captures transformer block residual
outputs, and stores a compact orthonormal basis per selected layer.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_BFCL_ROOT = "/mnt/cache/wuruixiao/users/lsc/gorilla/berkeley-function-call-leaderboard"
DEFAULT_OUTPUT_DIR = "/tmp/shared-storage/ExpertGym/hiddensteer/tool_residual_basis_rank4_l24-27_20260525"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    bfcl_root = Path(args.bfcl_root).expanduser().resolve()
    sys.path.insert(0, str(bfcl_root))

    from bfcl_eval.model_handler.local_inference.qwen import QwenHandler
    from bfcl_eval.model_handler.utils import system_prompt_pre_processing_chat_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[str(args.torch_dtype)]
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype, trust_remote_code=True)
    model.to(args.device)
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    handler = QwenHandler(
        model_name=str(args.model_path),
        temperature=0.0,
        registry_name="hiddensteer-residual-basis-builder",
        is_fc_model=False,
    )
    layers = parse_layers(args.layers)
    success_rows, failure_rows = collect_labeled_rows(
        bfcl_root=bfcl_root,
        result_root=Path(args.result_root).expanduser().resolve(),
        score_root=Path(args.score_root).expanduser().resolve(),
        categories=split_csv(args.categories),
        max_success_per_category=int(args.max_success_per_category),
        max_failure_per_category=int(args.max_failure_per_category),
    )
    rows = success_rows + failure_rows
    if not rows:
        raise SystemExit("No successful BFCL rows found for residual basis construction.")

    captured: dict[int, torch.Tensor] = {}
    hooks = register_layer_hooks(model, layers, captured)
    if not hooks:
        raise RuntimeError(f"No model layers matched requested layers: {layers}")

    layer_success_vectors: dict[int, list[torch.Tensor]] = defaultdict(list)
    layer_failure_vectors: dict[int, list[torch.Tensor]] = defaultdict(list)
    layer_norm_sums: dict[int, float] = defaultdict(float)
    layer_token_counts: dict[int, int] = defaultdict(int)
    try:
        with torch.no_grad():
            for row in rows:
                captured.clear()
                prompt = build_prompt(
                    handler=handler,
                    tokenizer=tokenizer,
                    system_prompt_pre_processing_chat_model=system_prompt_pre_processing_chat_model,
                    row=row["prompt"],
                )
                response = str(row["response"])
                encoded, positions = encode_prompt_response(
                    tokenizer=tokenizer,
                    prompt=prompt,
                    response=response,
                    max_seq_length=int(args.max_seq_length),
                    response_tail_tokens=int(args.response_tail_tokens),
                    device=str(args.device),
                )
                _ = model(**encoded, use_cache=False)
                for layer in layers:
                    hidden = captured.get(layer)
                    if hidden is None:
                        continue
                    selected = hidden[0, positions.to(hidden.device), :].detach().float().cpu()
                    if selected.numel() == 0:
                        continue
                    norms = selected.norm(dim=-1).clamp_min(1.0e-12)
                    layer_norm_sums[layer] += float(norms.sum().item())
                    layer_token_counts[layer] += int(selected.shape[0])
                    normalized = selected / norms.unsqueeze(-1)
                    if row["label"] == "success":
                        layer_success_vectors[layer].append(normalized)
                    else:
                        layer_failure_vectors[layer].append(normalized)
                if torch.cuda.is_available() and str(args.device).startswith("cuda"):
                    torch.cuda.empty_cache()
    finally:
        for handle in hooks:
            handle.remove()
        model.cpu()
        del model

    basis_payload: dict[int, dict[str, Any]] = {}
    manifest_layers = []
    for layer in layers:
        success_chunks = layer_success_vectors.get(layer, [])
        if not success_chunks:
            continue
        success_matrix = torch.cat(success_chunks, dim=0)
        success_basis = fit_basis(
            success_matrix,
            rank=int(args.rank),
            oversample=int(args.oversample),
            niter=int(args.niter),
        )
        failure_basis = None
        failure_orthogonal_basis = None
        failure_token_count = 0
        failure_chunks = layer_failure_vectors.get(layer, [])
        if failure_chunks:
            failure_matrix = torch.cat(failure_chunks, dim=0)
            failure_token_count = int(failure_matrix.shape[0])
            failure_basis = fit_basis(
                failure_matrix,
                rank=int(args.rank),
                oversample=int(args.oversample),
                niter=int(args.niter),
            )
            failure_orthogonal_basis = orthogonalize_against(failure_basis, success_basis)
        basis_payload[layer] = {
            "basis": success_basis.to(dtype_map[str(args.basis_dtype)]),
            "success_basis": success_basis.to(dtype_map[str(args.basis_dtype)]),
            "failure_basis": failure_basis.to(dtype_map[str(args.basis_dtype)]) if failure_basis is not None else None,
            "failure_orthogonal_basis": (
                failure_orthogonal_basis.to(dtype_map[str(args.basis_dtype)])
                if failure_orthogonal_basis is not None
                else None
            ),
            "rank": int(success_basis.shape[1]),
            "token_count": int(layer_token_counts[layer]),
            "success_token_count": int(success_matrix.shape[0]),
            "failure_token_count": int(failure_token_count),
            "mean_hidden_norm": layer_norm_sums[layer] / max(layer_token_counts[layer], 1),
        }
        manifest_layers.append(
            {
                "layer": int(layer),
                "rank": int(success_basis.shape[1]),
                "token_count": int(layer_token_counts[layer]),
                "success_token_count": int(success_matrix.shape[0]),
                "failure_token_count": int(failure_token_count),
                "failure_orthogonal_rank": (
                    int(failure_orthogonal_basis.shape[1]) if failure_orthogonal_basis is not None else 0
                ),
                "mean_hidden_norm": basis_payload[layer]["mean_hidden_norm"],
            }
        )

    basis_path = output_dir / "residual_basis.pt"
    torch.save(basis_payload, basis_path)
    manifest = {
        "format": "hiddensteer_tool_residual_basis_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_path": str(Path(args.model_path).expanduser().resolve()),
        "bfcl_root": str(bfcl_root),
        "result_root": str(Path(args.result_root).expanduser().resolve()),
        "score_root": str(Path(args.score_root).expanduser().resolve()),
        "basis_path": str(basis_path),
        "categories": split_csv(args.categories),
        "num_success_rows": len(success_rows),
        "num_failure_rows": len(failure_rows),
        "rank": int(args.rank),
        "layers": manifest_layers,
        "runtime_semantics": (
            "Hook selected transformer block residual outputs. The payload stores success and failure "
            "bases; runtime can either boost success directions or remove failure-only directions."
        ),
    }
    write_json(output_dir / "basis_manifest.json", manifest)
    write_json(output_dir / "success_rows.json", compact_rows(success_rows))
    write_json(output_dir / "failure_rows.json", compact_rows(failure_rows))
    write_markdown(output_dir / "README.md", manifest)
    print(
        json.dumps(
            {
                "basis_manifest": str(output_dir / "basis_manifest.json"),
                "success_rows": len(success_rows),
                "failure_rows": len(failure_rows),
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--bfcl-root", default=DEFAULT_BFCL_ROOT)
    parser.add_argument("--result-root", required=True, help="Path to result/<model_name> containing live/non_live files.")
    parser.add_argument("--score-root", required=True, help="Path to score/<model_name> containing live/non_live files.")
    parser.add_argument("--categories", default="live_parallel_multiple")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--layers", default="24-27")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--oversample", type=int, default=4)
    parser.add_argument("--niter", type=int, default=2)
    parser.add_argument("--max-success-per-category", type=int, default=64)
    parser.add_argument("--max-failure-per-category", type=int, default=64)
    parser.add_argument("--response-tail-tokens", type=int, default=128)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--basis-dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    return parser.parse_args()


def collect_labeled_rows(
    *,
    bfcl_root: Path,
    result_root: Path,
    score_root: Path,
    categories: list[str],
    max_success_per_category: int,
    max_failure_per_category: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    success_rows = []
    failure_rows = []
    for category in categories:
        split = "live" if category.startswith("live_") else "non_live"
        prompt_path = bfcl_root / "bfcl_eval" / "data" / f"BFCL_v4_{category}.json"
        result_path = result_root / split / f"BFCL_v4_{category}_result.json"
        score_path = score_root / split / f"BFCL_v4_{category}_score.json"
        prompts = {str(row["id"]): row for row in read_jsonl(prompt_path)}
        results = {str(row["id"]): row for row in read_jsonl(result_path)}
        invalid_ids = set()
        if score_path.exists():
            for index, row in enumerate(read_jsonl(score_path)):
                if index == 0 and "accuracy" in row:
                    continue
                if row.get("valid") is False:
                    invalid_ids.add(str(row["id"]))
        success_ids = [item for item in results if item in prompts and item not in invalid_ids]
        for item_id in success_ids[:max_success_per_category]:
            success_rows.append(
                {
                    "category": category,
                    "id": item_id,
                    "label": "success",
                    "prompt": prompts[item_id],
                    "response": str(results[item_id].get("result", "")),
                }
            )
        failure_ids = [item for item in results if item in prompts and item in invalid_ids]
        for item_id in failure_ids[:max_failure_per_category]:
            failure_rows.append(
                {
                    "category": category,
                    "id": item_id,
                    "label": "failure",
                    "prompt": prompts[item_id],
                    "response": str(results[item_id].get("result", "")),
                }
            )
    return success_rows, failure_rows


def fit_basis(matrix: torch.Tensor, *, rank: int, oversample: int, niter: int) -> torch.Tensor:
    actual_rank = min(rank, matrix.shape[0], matrix.shape[1])
    q = min(actual_rank + oversample, matrix.shape[0], matrix.shape[1])
    _u, _s, v = torch.pca_lowrank(matrix, q=q, center=False, niter=niter)
    basis = v[:, :actual_rank].contiguous()
    basis, _ = torch.linalg.qr(basis, mode="reduced")
    return basis


def orthogonalize_against(source_basis: torch.Tensor, reference_basis: torch.Tensor) -> torch.Tensor:
    residual = source_basis - reference_basis @ (reference_basis.transpose(0, 1) @ source_basis)
    keep = residual.norm(dim=0) > 1.0e-6
    residual = residual[:, keep]
    if residual.numel() == 0:
        return residual.reshape(source_basis.shape[0], 0)
    residual, _ = torch.linalg.qr(residual.contiguous(), mode="reduced")
    return residual


def build_prompt(*, handler: Any, tokenizer: Any, system_prompt_pre_processing_chat_model: Any, row: dict[str, Any]) -> str:
    item = copy.deepcopy(row)
    item["question"][0] = system_prompt_pre_processing_chat_model(item["question"][0], item["function"], item["id"])
    return handler._format_prompt(item["question"][0], item["function"])


def encode_prompt_response(
    *,
    tokenizer: Any,
    prompt: str,
    response: str,
    max_seq_length: int,
    response_tail_tokens: int,
    device: str,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    response_ids = tokenizer(response, add_special_tokens=False).input_ids
    if not prompt_ids or not response_ids:
        raise ValueError("Prompt and response must both be non-empty.")
    ids = prompt_ids + response_ids
    prompt_len = len(prompt_ids)
    positions = []
    for local_index in range(len(response_ids)):
        pos = prompt_len + local_index - 1
        if 0 <= pos < len(ids):
            positions.append(pos)
    if response_tail_tokens > 0 and len(positions) > response_tail_tokens:
        positions = positions[-response_tail_tokens:]
    if len(ids) > max_seq_length:
        overflow = len(ids) - max_seq_length
        ids = ids[overflow:]
        positions = [pos - overflow for pos in positions if pos >= overflow]
    if not positions:
        positions = [len(ids) - 1]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask}, torch.tensor(positions, dtype=torch.long)


def register_layer_hooks(model: torch.nn.Module, layers: list[int], captured: dict[int, torch.Tensor]) -> list[Any]:
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
    return hooks


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def compact_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"category": row["category"], "id": row["id"], "response_preview": row["response"][:300]} for row in rows]


def write_markdown(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# HiddenSteer Tool Residual Basis",
        "",
        f"- Created: `{manifest['created_at']}`",
        f"- Model: `{manifest['model_path']}`",
        f"- Success rows: `{manifest['num_success_rows']}`",
        f"- Failure rows: `{manifest['num_failure_rows']}`",
        f"- Basis path: `{manifest['basis_path']}`",
        "",
        "| layer | rank | success tokens | failure tokens | failure-orth rank | mean hidden norm |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for item in manifest["layers"]:
        lines.append(
            f"| {item['layer']} | {item['rank']} | {item['success_token_count']} | "
            f"{item['failure_token_count']} | {item['failure_orthogonal_rank']} | "
            f"{float(item['mean_hidden_norm']):.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
