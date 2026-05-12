#!/usr/bin/env python3
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm


SKIP_SUFFIXES = ("inv_freq",)


def index_path(model_dir: Path) -> Path:
    return model_dir / "model.safetensors.index.json"


def tensor_index(model_dir: Path) -> dict[str, str]:
    idx = index_path(model_dir)
    if idx.exists():
        return json.loads(idx.read_text())["weight_map"]
    files = sorted(model_dir.glob("*.safetensors"))
    if len(files) != 1:
        raise FileNotFoundError(f"Cannot infer safetensors index in {model_dir}")
    with safe_open(files[0], framework="pt", device="cpu") as f:
        return {k: files[0].name for k in f.keys()}


def read_tensor(model_dir: Path, weight_map: dict[str, str], name: str) -> torch.Tensor:
    with safe_open(model_dir / weight_map[name], framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def copy_metadata(base_dir: Path, out_dir: Path) -> None:
    keep = {
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "README.md",
    }
    for src in base_dir.iterdir():
        if src.name in keep and src.is_file():
            shutil.copy2(src, out_dir / src.name)


def probe_lambdas(
    base_dir: Path,
    base_map: dict[str, str],
    expert_dirs: list[Path],
    expert_maps: list[dict[str, str]],
    names: list[str],
    eps: float,
    r: float,
    alpha: float,
) -> tuple[list[float], list[dict[str, float | int]]]:
    shared_counts = [0] * len(expert_dirs)
    unique_counts = [0] * len(expert_dirs)

    for name in tqdm(names, desc="Probe RAM distribution"):
        if any(name.endswith(s) for s in SKIP_SUFFIXES):
            continue
        base = read_tensor(base_dir, base_map, name)
        if not torch.is_floating_point(base):
            continue
        base_f = base.to(torch.float32)
        masks = []
        count = torch.zeros_like(base_f, dtype=torch.int16)
        for model_dir, weight_map_i in zip(expert_dirs, expert_maps):
            delta = read_tensor(model_dir, weight_map_i, name).to(torch.float32) - base_f
            mask = delta.abs() > eps
            masks.append(mask)
            count += mask.to(torch.int16)

        shared = count >= 2
        unique = count == 1
        for i, mask in enumerate(masks):
            shared_counts[i] += int((shared & mask).sum().item())
            unique_counts[i] += int((unique & mask).sum().item())

    lambdas = []
    stats = []
    for n_shared, n_unique in zip(shared_counts, unique_counts):
        rho = n_shared / max(n_unique, 1)
        lam = 1.0 + r * min(max(rho, 0.0), alpha)
        lambdas.append(lam)
        stats.append({"shared": n_shared, "unique": n_unique, "rho": rho, "lambda": lam})
    print("RAM unique-region lambdas:", ", ".join(f"{v:.6f}" for v in lambdas))
    return lambdas, stats


def ram_tensor(base: torch.Tensor, experts: list[torch.Tensor], lambdas: list[float], eps: float) -> torch.Tensor:
    if not torch.is_floating_point(base):
        return base

    dtype = base.dtype
    base_f = base.to(torch.float32)
    deltas = [expert.to(torch.float32) - base_f for expert in experts]
    masks = [delta.abs() > eps for delta in deltas]
    count = torch.zeros_like(base_f, dtype=torch.int16)
    for mask in masks:
        count += mask.to(torch.int16)

    merged = torch.zeros_like(base_f)
    shared = count >= 2
    if shared.any():
        acc = torch.zeros_like(base_f)
        for delta, mask in zip(deltas, masks):
            acc += torch.where(mask, delta, torch.zeros_like(delta))
        merged = torch.where(shared, acc / count.clamp_min(1).to(torch.float32), merged)

    unique = count == 1
    for delta, mask, lam in zip(deltas, masks, lambdas):
        unique_t = unique & mask
        if not unique_t.any():
            continue
        merged = torch.where(unique_t, delta * lam, merged)

    return (base_f + merged).to(dtype)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge RL experts with RAM/RAM+.")
    parser.add_argument("--base", required=True)
    parser.add_argument("--experts", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--r", type=float, default=0.10, help="0 gives RAM; 0.10 is the paper's best RAM+ setting.")
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--shard-size", type=int, default=256, help="Number of tensors per output safetensors shard.")
    args = parser.parse_args()

    base_dir = Path(args.base)
    expert_dirs = [Path(p) for p in args.experts]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_map = tensor_index(base_dir)
    expert_maps = [tensor_index(p) for p in expert_dirs]
    names = sorted(base_map)
    for model_dir, weight_map in zip(expert_dirs, expert_maps):
        missing = set(names) - set(weight_map)
        if missing:
            raise ValueError(f"{model_dir} misses {len(missing)} base tensors, e.g. {sorted(missing)[:3]}")

    copy_metadata(base_dir, out_dir)
    lambdas, merge_stats = probe_lambdas(base_dir, base_map, expert_dirs, expert_maps, names, args.eps, args.r, args.alpha)
    metadata = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": sys.argv,
        "method": "RAM" if args.r == 0 else "RAM+",
        "formula": "shared regions are averaged; unique regions use lambda_t = 1 + r * min(rho_t, alpha)",
        "base": str(base_dir),
        "experts": [str(p) for p in expert_dirs],
        "eps": args.eps,
        "r": args.r,
        "alpha": args.alpha,
        "expert_stats": merge_stats,
    }
    (out_dir / "ram_merge_config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    weight_map: dict[str, str] = {}
    shard: dict[str, torch.Tensor] = {}
    shard_id = 1

    def flush() -> None:
        nonlocal shard, shard_id
        if not shard:
            return
        name = f"model-{shard_id:05d}.safetensors"
        save_file(shard, out_dir / name)
        for key in shard:
            weight_map[key] = name
        shard = {}
        shard_id += 1

    for name in tqdm(names, desc="RAM merge"):
        base = read_tensor(base_dir, base_map, name)
        if any(name.endswith(s) for s in SKIP_SUFFIXES):
            merged = base
        else:
            experts = [read_tensor(model_dir, weight_map_i, name) for model_dir, weight_map_i in zip(expert_dirs, expert_maps)]
            merged = ram_tensor(base, experts, lambdas, args.eps)
        shard[name] = merged.contiguous()
        if len(shard) >= args.shard_size:
            flush()
    flush()

    total_size = sum((out_dir / f).stat().st_size for f in set(weight_map.values()))
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (out_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")


if __name__ == "__main__":
    main()
