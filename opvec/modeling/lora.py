"""Minimal LoRA adapters for local calibration experiments."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable


LAYER_RE = re.compile(r"(?:^|\.)model\.layers\.(\d+)\.")


class LoRALinear:
    """Wrap a Linear layer with a trainable low-rank residual."""

    def __init__(
        self,
        torch_module: Any,
        base_linear: Any,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ):
        if rank <= 0:
            raise ValueError("rank must be positive")
        nn = torch_module.nn

        class _LoRALinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.base_linear = base_linear
                self.rank = int(rank)
                self.alpha = float(alpha)
                self.scaling = float(alpha) / float(rank)
                self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
                for param in self.base_linear.parameters():
                    param.requires_grad_(False)
                self.lora_A = nn.Parameter(torch_module.empty((rank, base_linear.in_features), dtype=torch_module.float32))
                self.lora_B = nn.Parameter(torch_module.zeros((base_linear.out_features, rank), dtype=torch_module.float32))
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

            def forward(self, x):
                out = self.base_linear(x)
                lora_input = self.dropout(x).to(dtype=self.lora_A.dtype)
                hidden = torch_module.nn.functional.linear(lora_input, self.lora_A)
                delta = torch_module.nn.functional.linear(hidden, self.lora_B) * self.scaling
                return out + delta.to(dtype=out.dtype)

            def merge_into_base(self):
                delta = self.lora_B @ self.lora_A
                delta = delta * self.scaling
                with torch_module.no_grad():
                    updated = self.base_linear.weight.detach().to(dtype=torch_module.float32) + delta.detach()
                    self.base_linear.weight.copy_(updated.to(dtype=self.base_linear.weight.dtype))
                return self.base_linear

        self.module = _LoRALinear()

    def __getattr__(self, name: str):
        return getattr(self.module, name)


def install_lora_adapters(
    torch_module: Any,
    model: Any,
    *,
    target_module_names: Iterable[str],
    layers: set[int] | None,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
) -> list[str]:
    """Install LoRA adapters on matching Linear modules and return module names."""

    target_names = {item.strip() for item in target_module_names if item.strip()}
    installed = []
    for module_name, module in list(model.named_modules()):
        if not isinstance(module, torch_module.nn.Linear):
            continue
        leaf_name = module_name.rsplit(".", 1)[-1]
        if leaf_name not in target_names:
            continue
        layer_idx = layer_index(module_name)
        if layers is not None and layer_idx not in layers:
            continue
        parent, child_name = parent_module(model, module_name)
        wrapped = LoRALinear(torch_module, module, rank=rank, alpha=alpha, dropout=dropout).module
        wrapped.to(device=module.weight.device)
        setattr(parent, child_name, wrapped)
        installed.append(module_name)
    if not installed:
        raise ValueError(f"No LoRA target modules matched: {sorted(target_names)} layers={sorted(layers) if layers is not None else 'all'}")
    return installed


def merge_lora_adapters(torch_module: Any, model: Any) -> list[str]:
    """Merge all LoRA adapters into their base Linear modules in place."""

    del torch_module
    merged = []
    for module_name, module in list(model.named_modules()):
        if not hasattr(module, "lora_A") or not hasattr(module, "merge_into_base"):
            continue
        parent, child_name = parent_module(model, module_name)
        setattr(parent, child_name, module.merge_into_base())
        merged.append(module_name)
    return merged


def trainable_lora_parameters(model: Any) -> list[Any]:
    return [param for name, param in model.named_parameters() if "lora_A" in name or "lora_B" in name]


def lora_state_dict(model: Any) -> dict[str, Any]:
    return {name: param.detach().cpu() for name, param in model.named_parameters() if "lora_A" in name or "lora_B" in name}


def save_lora_adapter(torch_module: Any, model: Any, output_dir: str | Path, config: dict[str, Any]) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch_module.save(lora_state_dict(model), output / "adapter_model.pt")
    (output / "adapter_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_lora_adapter(torch_module: Any, model: Any, adapter_dir: str | Path, *, device: str | None = None) -> dict[str, Any]:
    output = Path(adapter_dir)
    config = json.loads((output / "adapter_config.json").read_text(encoding="utf-8"))
    installed = install_lora_adapters(
        torch_module,
        model,
        target_module_names=config["target_modules"],
        layers=parse_layers(config.get("layers")),
        rank=int(config["rank"]),
        alpha=float(config["alpha"]),
        dropout=float(config.get("dropout", 0.0)),
    )
    state = torch_module.load(output / "adapter_model.pt", map_location=device or "cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected_lora = [name for name in unexpected if "lora_" in name]
    missing_lora = [name for name in missing if "lora_" in name]
    if unexpected_lora or missing_lora:
        raise RuntimeError(f"LoRA adapter load mismatch: missing={missing_lora} unexpected={unexpected_lora}")
    return {**config, "installed_modules": installed}


def parse_layers(spec: str | None) -> set[int] | None:
    """Parse comma-separated layer ids/ranges. None or 'all' means all layers."""

    if spec is None or spec.strip().lower() in {"", "all", "*"}:
        return None
    layers: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            layers.update(range(int(start), int(end) + 1))
        else:
            layers.add(int(part))
    return layers


def layer_index(module_name: str) -> int | None:
    match = LAYER_RE.search(module_name)
    if not match:
        return None
    return int(match.group(1))


def parent_module(model: Any, module_name: str) -> tuple[Any, str]:
    parent_name, child_name = module_name.rsplit(".", 1) if "." in module_name else ("", module_name)
    parent = model.get_submodule(parent_name) if parent_name else model
    return parent, child_name
