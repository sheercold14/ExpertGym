"""Configuration helpers for OP-VEC runs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping


DEFAULT_STORAGE_ROOT = Path("/tmp/shared-storage/OnPolicy")


class ConfigError(ValueError):
    """Raised when an OP-VEC config is invalid."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config and expand environment-style variables."""

    try:
        import yaml
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("OP-VEC config loading requires PyYAML") from error

    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ConfigError(f"Config root must be a mapping: {config_path}")
    expanded = _expand_value(payload, _variables(config_path))
    apply_defaults(expanded, config_path)
    validate_config(expanded)
    return expanded


def apply_defaults(config: dict[str, Any], config_path: Path | None = None) -> None:
    """Apply conservative defaults used by scripts and tests."""

    config.setdefault("run", {})
    config["run"].setdefault("name", "opvec4")
    config["run"].setdefault("seed", 42)

    config.setdefault("storage", {})
    config["storage"].setdefault("root", str(DEFAULT_STORAGE_ROOT))

    config.setdefault("modes", {})
    config["modes"].setdefault("mode_set", "opvec4")
    config["modes"].setdefault("artifact_dir", "${ONPOLICY_STORAGE_ROOT}/modes/opvec4")
    config["modes"].setdefault(
        "include_regex",
        [
            r"^model\.layers\.[0-9]+\.(self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.weight$"
        ],
    )
    config["modes"].setdefault(
        "exclude_regex",
        [".*lm_head.*", ".*norm.*", ".*embed_tokens.*", ".*bias.*"],
    )
    config["modes"].setdefault("residual_zero_mean", True)

    config.setdefault("initial_gates", {})
    config["initial_gates"].setdefault("common", 0.50)
    config["initial_gates"].setdefault("tool_residual", 0.0)
    config["initial_gates"].setdefault("memory_residual", 0.0)
    config["initial_gates"].setdefault("code_residual", 0.0)

    config.setdefault("gate_bounds", {})
    config["gate_bounds"].setdefault("common", [-0.10, 1.50])
    config["gate_bounds"].setdefault("residual", [-0.50, 0.50])

    config.setdefault("data", {})
    config["data"].setdefault("seed_manifest", "${ONPOLICY_STORAGE_ROOT}/data/seed_prompt_manifest.jsonl")
    config["data"].setdefault("behavior_matrix", "${ONPOLICY_STORAGE_ROOT}/data/behavior_matrix.jsonl")
    config["data"].setdefault("rollout_store", "${ONPOLICY_STORAGE_ROOT}/runs/opvec4/onpolicy_rollouts.jsonl")

    config.setdefault("frontier", {})
    config["frontier"].setdefault("min_frontier_weight", 0.20)
    config["frontier"].setdefault("min_reward_std", 0.05)

    config.setdefault("loss", {})
    config["loss"].setdefault("eps_clip", 0.2)
    config["loss"].setdefault("advantage_eps", 1.0e-6)

    config.setdefault("optimizer", {})
    config["optimizer"].setdefault("lr", 0.01)
    config["optimizer"].setdefault("grad_clip_norm", 1.0)

    config.setdefault("evaluation", {})
    config["evaluation"].setdefault("model_name", config["run"].get("name", "opvec-gated-grpo"))
    if config_path is not None:
        config["run"].setdefault("config_path", str(config_path))


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate fields that would otherwise fail after expensive model loading."""

    models = config.get("models", {})
    if not models.get("base"):
        raise ConfigError("models.base is required")
    experts = models.get("experts", {})
    for name in ["tool", "memory", "code"]:
        if not isinstance(experts, Mapping) or not experts.get(name):
            raise ConfigError(f"models.experts.{name} is required")
    common_bounds = config.get("gate_bounds", {}).get("common", [])
    residual_bounds = config.get("gate_bounds", {}).get("residual", [])
    if len(common_bounds) != 2 or len(residual_bounds) != 2:
        raise ConfigError("gate_bounds.common and gate_bounds.residual must have two values")


def write_json(path: str | Path, payload: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def _variables(config_path: Path) -> dict[str, str]:
    repo_root = _discover_repo_root(config_path)
    storage_root = os.environ.get("ONPOLICY_STORAGE_ROOT", str(DEFAULT_STORAGE_ROOT))
    variables = {
        "REPO_ROOT": str(repo_root),
        "PROJECT_ROOT": str(repo_root),
        "ONPOLICY_STORAGE_ROOT": storage_root,
    }
    variables.update(os.environ)
    return variables


def _discover_repo_root(path: Path) -> Path:
    for candidate in [path.parent, *path.parents]:
        if (candidate / ".git").exists():
            return candidate
    return Path.cwd().resolve()


def _expand_value(value: Any, variables: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        expanded = os.path.expanduser(value)
        for key, replacement in variables.items():
            expanded = expanded.replace("${" + key + "}", str(replacement))
        return expanded
    if isinstance(value, list):
        return [_expand_value(item, variables) for item in value]
    if isinstance(value, dict):
        return {key: _expand_value(item, variables) for key, item in value.items()}
    return value
