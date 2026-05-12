"""Device helpers for single-GPU and sharded HF models."""

from __future__ import annotations

from typing import Any


def parse_max_memory(items: list[str] | None) -> dict[Any, str] | None:
    if not items:
        return None
    out: dict[Any, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --max-memory value: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        out[int(key) if key.isdigit() else key] = value.strip()
    return out


def model_input_device(model: Any, torch_module: Any, fallback: str = "cuda"):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch_module.device(fallback)


def model_load_device_kwargs(*, device_map: str | None, max_memory: list[str] | None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if device_map:
        kwargs["device_map"] = device_map
        parsed = parse_max_memory(max_memory)
        if parsed:
            kwargs["max_memory"] = parsed
    return kwargs
