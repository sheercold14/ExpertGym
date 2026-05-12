#!/usr/bin/env python3
import argparse
import os
import subprocess
from pathlib import Path
from typing import Optional

import yaml
from huggingface_hub import HfFolder, snapshot_download


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def hf_download(repo_id: str, repo_type: str, local_dir: Path, endpoint: str, token: Optional[str]) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
        endpoint=endpoint,
        token=token,
    )


def sparse_clone(repo_url: str, local_dir: Path, sparse_paths: list[str]) -> None:
    if (local_dir / ".git").exists():
        subprocess.run(["git", "-C", str(local_dir), "pull", "--ff-only"], check=True)
        return

    local_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", "--filter=blob:none", "--no-checkout", repo_url, str(local_dir)],
        check=True,
    )
    subprocess.run(["git", "-C", str(local_dir), "sparse-checkout", "init", "--cone"], check=True)
    subprocess.run(["git", "-C", str(local_dir), "sparse-checkout", "set", *sparse_paths], check=True)
    subprocess.run(["git", "-C", str(local_dir), "checkout"], check=True)


def selected_keys(items: dict, requested: list[str]) -> list[str]:
    return list(items.keys()) if requested == ["all"] else requested


def main() -> None:
    parser = argparse.ArgumentParser(description="Download RAM Llama reproduction assets.")
    parser.add_argument("--config", default="reproduce/ram_llama/config/assets.yaml")
    parser.add_argument("--root", default=None, help="Override storage_root from config.")
    parser.add_argument("--models", nargs="+", default=["all"], help="Model keys or all.")
    parser.add_argument("--datasets", nargs="+", default=["all"], help="Dataset keys or all.")
    parser.add_argument("--eval-datasets", nargs="+", default=[], help="Evaluation dataset keys, all, or empty.")
    parser.add_argument("--skip-models", action="store_true")
    parser.add_argument("--skip-datasets", action="store_true")
    parser.add_argument("--skip-gated", action="store_true", help="Skip gated models such as Meta Llama.")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    root = Path(args.root or cfg["storage_root"])
    endpoint = cfg.get("hf_endpoint", "https://hf-mirror.com")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or HfFolder.get_token()

    os.environ.setdefault("HF_ENDPOINT", endpoint)
    os.environ.setdefault("HF_HOME", str(root / ".hf_home"))

    if not args.skip_models:
        for key in selected_keys(cfg["models"], args.models):
            model = cfg["models"][key]
            if args.skip_gated and model.get("gated"):
                print(f"[skip] gated model {key}: {model['repo_id']}")
                continue
            print(f"[model] {key}: {model['repo_id']}")
            hf_download(model["repo_id"], "model", root / model["local_dir"], endpoint, token)

    if not args.skip_datasets:
        for key in selected_keys(cfg["datasets"], args.datasets):
            data = cfg["datasets"][key]
            print(f"[dataset] {key}")
            if data.get("source") == "github":
                sparse_clone(data["repo_url"], root / data["local_dir"], data["sparse_paths"])
                continue
            try:
                hf_download(data["repo_id"], data.get("repo_type", "dataset"), root / data["local_dir"], endpoint, token)
            except Exception:
                fallback = data.get("fallback_repo_id")
                if not fallback:
                    raise
                print(f"[fallback] {fallback}")
                hf_download(fallback, data.get("repo_type", "dataset"), root / data["local_dir"], endpoint, token)

    if args.eval_datasets:
        for key in selected_keys(cfg["eval_datasets"], args.eval_datasets):
            data = cfg["eval_datasets"][key]
            print(f"[eval_dataset] {key}: {data['repo_id']}")
            hf_download(data["repo_id"], data.get("repo_type", "dataset"), root / data["local_dir"], endpoint, token)


if __name__ == "__main__":
    main()
