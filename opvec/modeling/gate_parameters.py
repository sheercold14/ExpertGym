"""Torch gate managers for OP-VEC."""

from __future__ import annotations

import re
from typing import Any, Mapping


EXPERT_NAMES = ("tool", "memory", "code")
DEFAULT_LAYER_BANDS = {
    "early": (0, 9),
    "mid": (10, 19),
    "late": (20, 10_000),
}


class TorchGateManager:
    """Small nn.Module that owns the only trainable OP-VEC parameters."""

    def __init__(
        self,
        torch_module: Any,
        init_values: Mapping[str, float],
        *,
        common_bounds: tuple[float, float] = (-0.10, 1.50),
        residual_bounds: tuple[float, float] = (-0.50, 0.50),
    ):
        nn = torch_module.nn

        class _Manager(nn.Module):
            def __init__(self):
                super().__init__()
                initial_common = float(init_values.get("common", 0.5))
                self.raw_common = nn.Parameter(torch_module.tensor([initial_common], dtype=torch_module.float32))
                initial_residual = [
                    float(init_values.get("tool_residual", 0.0)),
                    float(init_values.get("memory_residual", 0.0)),
                    float(init_values.get("code_residual", 0.0)),
                ]
                self.raw_residual = nn.Parameter(
                    torch_module.tensor(
                        initial_residual,
                        dtype=torch_module.float32,
                    )
                )
                self.register_buffer("initial_raw_common", torch_module.tensor([initial_common], dtype=torch_module.float32))
                self.register_buffer("initial_raw_residual", torch_module.tensor(initial_residual, dtype=torch_module.float32))

            def effective_gates(self):
                residual = self.raw_residual - self.raw_residual.mean()
                return {
                    "common": self.raw_common[0],
                    "tool_residual": residual[0],
                    "memory_residual": residual[1],
                    "code_residual": residual[2],
                }

            def expert_coefficients(self):
                gates = self.effective_gates()
                return {
                    "tool": gates["common"] + gates["tool_residual"],
                    "memory": gates["common"] + gates["memory_residual"],
                    "code": gates["common"] + gates["code_residual"],
                }

            def gate_values(self):
                gates = self.effective_gates()
                return {key: float(value.detach().cpu().item()) for key, value in gates.items()}

            def project_(self):
                with torch_module.no_grad():
                    self.raw_common.clamp_(float(common_bounds[0]), float(common_bounds[1]))
                    self.raw_residual.clamp_(float(residual_bounds[0]), float(residual_bounds[1]))
                    self.raw_residual.sub_(self.raw_residual.mean())

        self.module = _Manager()

    def __getattr__(self, name: str):
        return getattr(self.module, name)


class TorchLayerBandGateManager:
    """nn.Module gate manager with one OP-VEC gate set per layer band."""

    def __init__(
        self,
        torch_module: Any,
        init_values: Mapping[str, float],
        *,
        layer_bands: Mapping[str, tuple[int, int]] | None = None,
        common_bounds: tuple[float, float] = (-0.10, 1.50),
        residual_bounds: tuple[float, float] = (-0.50, 0.50),
    ):
        nn = torch_module.nn
        bands = dict(layer_bands or DEFAULT_LAYER_BANDS)
        band_names = tuple(bands.keys())

        class _Manager(nn.Module):
            def __init__(self):
                super().__init__()
                self.band_names = band_names
                self.layer_bands = bands
                initial_common = [_init_value(init_values, band, "common", 0.5) for band in self.band_names]
                self.raw_common = nn.Parameter(
                    torch_module.tensor(
                        initial_common,
                        dtype=torch_module.float32,
                    )
                )
                initial_residual = [
                    [
                        _init_value(init_values, band, f"{expert}_residual", 0.0)
                        for expert in EXPERT_NAMES
                    ]
                    for band in self.band_names
                ]
                self.raw_residual = nn.Parameter(
                    torch_module.tensor(
                        initial_residual,
                        dtype=torch_module.float32,
                    )
                )
                self.register_buffer("initial_raw_common", torch_module.tensor(initial_common, dtype=torch_module.float32))
                self.register_buffer("initial_raw_residual", torch_module.tensor(initial_residual, dtype=torch_module.float32))

            def band_for_param(self, param_name: str | None) -> str:
                return layer_band_for_param(param_name, self.layer_bands, default=self.band_names[0])

            def effective_gates(self, *, band: str | None = None, param_name: str | None = None):
                residual = self.raw_residual - self.raw_residual.mean(dim=1, keepdim=True)
                if band is None and param_name is not None:
                    band = self.band_for_param(param_name)
                if band is not None:
                    idx = self.band_names.index(band)
                    return {
                        "common": self.raw_common[idx],
                        "tool_residual": residual[idx, 0],
                        "memory_residual": residual[idx, 1],
                        "code_residual": residual[idx, 2],
                    }
                values = {}
                for idx, band_name in enumerate(self.band_names):
                    values[f"{band_name}.common"] = self.raw_common[idx]
                    values[f"{band_name}.tool_residual"] = residual[idx, 0]
                    values[f"{band_name}.memory_residual"] = residual[idx, 1]
                    values[f"{band_name}.code_residual"] = residual[idx, 2]
                return values

            def expert_coefficients(self, *, band: str | None = None, param_name: str | None = None):
                gates = self.effective_gates(band=band, param_name=param_name)
                return {
                    "tool": gates["common"] + gates["tool_residual"],
                    "memory": gates["common"] + gates["memory_residual"],
                    "code": gates["common"] + gates["code_residual"],
                }

            def gate_values(self):
                gates = self.effective_gates()
                return {key: float(value.detach().cpu().item()) for key, value in gates.items()}

            def project_(self):
                with torch_module.no_grad():
                    self.raw_common.clamp_(float(common_bounds[0]), float(common_bounds[1]))
                    self.raw_residual.clamp_(float(residual_bounds[0]), float(residual_bounds[1]))
                    self.raw_residual.sub_(self.raw_residual.mean(dim=1, keepdim=True))

        self.module = _Manager()

    def __getattr__(self, name: str):
        return getattr(self.module, name)


class TorchParameterCoefficientManager:
    """nn.Module that learns one coefficient per mergeable parameter and expert."""

    def __init__(
        self,
        torch_module: Any,
        init_values: Mapping[str, float],
        *,
        param_names: list[str],
        coefficient_bounds: tuple[float, float] = (-0.50, 1.50),
    ):
        if not param_names:
            raise ValueError("TorchParameterCoefficientManager requires non-empty param_names")
        nn = torch_module.nn
        names = tuple(str(name) for name in param_names)
        initial = _initial_parameter_coefficients(init_values, names)

        class _Manager(nn.Module):
            def __init__(self):
                super().__init__()
                self.param_names = names
                self.param_to_index = {name: index for index, name in enumerate(self.param_names)}
                self.raw_coefficients = nn.Parameter(torch_module.tensor(initial, dtype=torch_module.float32))
                self.register_buffer(
                    "initial_coefficients",
                    torch_module.tensor(initial, dtype=torch_module.float32),
                )

            def effective_gates(self, *, param_name: str | None = None):
                if param_name is None:
                    values = {}
                    for param_idx, name in enumerate(self.param_names):
                        for expert_idx, expert in enumerate(EXPERT_NAMES):
                            values[f"{name}::{expert}"] = self.raw_coefficients[param_idx, expert_idx]
                    return values
                if str(param_name) not in self.param_to_index:
                    raise KeyError(f"Unknown parameter coefficient name: {param_name}")
                row = self.raw_coefficients[self.param_to_index[str(param_name)]]
                return {expert: row[index] for index, expert in enumerate(EXPERT_NAMES)}

            def expert_coefficients(self, *, param_name: str | None = None):
                if param_name is None:
                    values = self.raw_coefficients.mean(dim=0)
                else:
                    if str(param_name) not in self.param_to_index:
                        raise KeyError(f"Unknown parameter coefficient name: {param_name}")
                    values = self.raw_coefficients[self.param_to_index[str(param_name)]]
                return {expert: values[index] for index, expert in enumerate(EXPERT_NAMES)}

            def gate_values(self):
                values = {}
                detached = self.raw_coefficients.detach().cpu()
                for param_idx, param_name in enumerate(self.param_names):
                    for expert_idx, expert in enumerate(EXPERT_NAMES):
                        values[f"{param_name}::{expert}"] = float(detached[param_idx, expert_idx].item())
                return values

            def project_(self):
                with torch_module.no_grad():
                    self.raw_coefficients.clamp_(float(coefficient_bounds[0]), float(coefficient_bounds[1]))

        self.module = _Manager()

    def __getattr__(self, name: str):
        return getattr(self.module, name)


class TorchGlobalCoefficientManager:
    """nn.Module that learns exactly one direct coefficient per expert."""

    def __init__(
        self,
        torch_module: Any,
        init_values: Mapping[str, float],
        *,
        coefficient_bounds: tuple[float, float] = (-0.50, 1.50),
    ):
        nn = torch_module.nn
        initial = _initial_global_coefficients(init_values)

        class _Manager(nn.Module):
            def __init__(self):
                super().__init__()
                self.raw_coefficients = nn.Parameter(torch_module.tensor(initial, dtype=torch_module.float32))
                self.register_buffer(
                    "initial_coefficients",
                    torch_module.tensor(initial, dtype=torch_module.float32),
                )

            def effective_gates(self):
                return {expert: self.raw_coefficients[index] for index, expert in enumerate(EXPERT_NAMES)}

            def expert_coefficients(self, *, param_name: str | None = None):
                return self.effective_gates()

            def gate_values(self):
                detached = self.raw_coefficients.detach().cpu()
                return {expert: float(detached[index].item()) for index, expert in enumerate(EXPERT_NAMES)}

            def project_(self):
                with torch_module.no_grad():
                    self.raw_coefficients.clamp_(float(coefficient_bounds[0]), float(coefficient_bounds[1]))

        self.module = _Manager()

    def __getattr__(self, name: str):
        return getattr(self.module, name)


class TorchGlobalParameterCoefficientManager:
    """Learn global expert strengths plus small parameter-specific residuals."""

    def __init__(
        self,
        torch_module: Any,
        init_values: Mapping[str, float],
        *,
        param_names: list[str],
        global_bounds: tuple[float, float] = (0.00, 1.20),
        residual_bounds: tuple[float, float] = (-0.15, 0.15),
        coefficient_bounds: tuple[float, float] = (-0.50, 1.50),
        global_prior_scale: float = 0.10,
        residual_prior_scale: float = 1.00,
    ):
        if not param_names:
            raise ValueError("TorchGlobalParameterCoefficientManager requires non-empty param_names")
        nn = torch_module.nn
        names = tuple(str(name) for name in param_names)
        initial_global, initial_residual = _initial_global_parameter_coefficients(
            init_values,
            names,
            residual_bounds=residual_bounds,
        )

        class _Manager(nn.Module):
            def __init__(self):
                super().__init__()
                self.param_names = names
                self.param_to_index = {name: index for index, name in enumerate(self.param_names)}
                self.global_prior_scale = float(global_prior_scale)
                self.residual_prior_scale = float(residual_prior_scale)
                self.raw_global_coefficients = nn.Parameter(torch_module.tensor(initial_global, dtype=torch_module.float32))
                self.raw_residual_coefficients = nn.Parameter(torch_module.tensor(initial_residual, dtype=torch_module.float32))
                self.register_buffer(
                    "initial_global_coefficients",
                    torch_module.tensor(initial_global, dtype=torch_module.float32),
                )
                self.register_buffer(
                    "initial_residual_coefficients",
                    torch_module.tensor(initial_residual, dtype=torch_module.float32),
                )

            def effective_coefficients(self):
                return self.raw_global_coefficients.unsqueeze(0) + self.raw_residual_coefficients

            def effective_gates(self, *, param_name: str | None = None):
                coefficients = self.effective_coefficients()
                if param_name is None:
                    values = {}
                    for expert_idx, expert in enumerate(EXPERT_NAMES):
                        values[f"__global__::{expert}"] = self.raw_global_coefficients[expert_idx]
                    for param_idx, name in enumerate(self.param_names):
                        for expert_idx, expert in enumerate(EXPERT_NAMES):
                            values[f"{name}::{expert}"] = coefficients[param_idx, expert_idx]
                    return values
                if str(param_name) not in self.param_to_index:
                    raise KeyError(f"Unknown parameter coefficient name: {param_name}")
                row = coefficients[self.param_to_index[str(param_name)]]
                return {expert: row[index] for index, expert in enumerate(EXPERT_NAMES)}

            def expert_coefficients(self, *, param_name: str | None = None):
                if param_name is None:
                    values = self.raw_global_coefficients
                else:
                    if str(param_name) not in self.param_to_index:
                        raise KeyError(f"Unknown parameter coefficient name: {param_name}")
                    values = self.effective_coefficients()[self.param_to_index[str(param_name)]]
                return {expert: values[index] for index, expert in enumerate(EXPERT_NAMES)}

            def gate_values(self):
                values = {}
                detached_global = self.raw_global_coefficients.detach().cpu()
                detached_effective = self.effective_coefficients().detach().cpu()
                for expert_idx, expert in enumerate(EXPERT_NAMES):
                    values[f"__global__::{expert}"] = float(detached_global[expert_idx].item())
                for param_idx, param_name in enumerate(self.param_names):
                    for expert_idx, expert in enumerate(EXPERT_NAMES):
                        values[f"{param_name}::{expert}"] = float(detached_effective[param_idx, expert_idx].item())
                return values

            def project_(self):
                with torch_module.no_grad():
                    self.raw_global_coefficients.clamp_(float(global_bounds[0]), float(global_bounds[1]))
                    self.raw_residual_coefficients.clamp_(float(residual_bounds[0]), float(residual_bounds[1]))
                    effective = (self.raw_global_coefficients.unsqueeze(0) + self.raw_residual_coefficients).clamp(
                        float(coefficient_bounds[0]),
                        float(coefficient_bounds[1]),
                    )
                    self.raw_residual_coefficients.copy_(
                        (effective - self.raw_global_coefficients.unsqueeze(0)).clamp(
                            float(residual_bounds[0]),
                            float(residual_bounds[1]),
                        )
                    )

        self.module = _Manager()

    def __getattr__(self, name: str):
        return getattr(self.module, name)


def layer_band_for_param(
    param_name: str | None,
    layer_bands: Mapping[str, tuple[int, int]] | None = None,
    *,
    default: str = "early",
) -> str:
    """Map a Qwen-style parameter name to a coarse layer band."""

    if not param_name:
        return default
    match = re.search(r"model\.layers\.(\d+)\.", str(param_name))
    if not match:
        return default
    layer_idx = int(match.group(1))
    for band, (start, end) in (layer_bands or DEFAULT_LAYER_BANDS).items():
        if int(start) <= layer_idx <= int(end):
            return band
    return default


def make_torch_gate_manager(
    torch_module: Any,
    config: Mapping[str, Any],
    *,
    parameterization: str = "global",
    param_names: list[str] | None = None,
):
    """Construct a gate manager module from run config."""

    common_bounds = tuple(float(item) for item in config.get("gate_bounds", {}).get("common", (-0.10, 1.50)))
    residual_bounds = tuple(float(item) for item in config.get("gate_bounds", {}).get("residual", (-0.50, 0.50)))
    if parameterization == "global":
        return TorchGateManager(
            torch_module,
            config.get("initial_gates", {}),
            common_bounds=common_bounds,  # type: ignore[arg-type]
            residual_bounds=residual_bounds,  # type: ignore[arg-type]
        ).module
    if parameterization in {"layer-band", "layer_band"}:
        return TorchLayerBandGateManager(
            torch_module,
            config.get("initial_gates", {}),
            layer_bands=_config_layer_bands(config),
            common_bounds=common_bounds,  # type: ignore[arg-type]
            residual_bounds=residual_bounds,  # type: ignore[arg-type]
        ).module
    coefficient_bounds = tuple(
        float(item)
        for item in config.get("gate_bounds", {}).get(
            "coefficient",
            config.get("gate_bounds", {}).get("common", (-0.50, 1.50)),
        )
    )
    if parameterization in {
        "global-coefficient",
        "global_coefficient",
        "global-coefficients",
        "global_coefficients",
        "global-direct",
        "global_direct",
        "expert-coefficient",
        "expert_coefficient",
    }:
        return TorchGlobalCoefficientManager(
            torch_module,
            config.get("initial_gates", {}),
            coefficient_bounds=coefficient_bounds,  # type: ignore[arg-type]
        ).module
    if parameterization in {"parameter", "param", "param-coefficients", "parameter-coefficients"}:
        return TorchParameterCoefficientManager(
            torch_module,
            config.get("initial_gates", {}),
            param_names=list(param_names or []),
            coefficient_bounds=coefficient_bounds,  # type: ignore[arg-type]
        ).module
    if parameterization in {
        "global-parameter",
        "global_parameter",
        "global-param",
        "global_param",
        "global-residual",
        "global_residual",
    }:
        global_bounds = tuple(
            float(item)
            for item in config.get("gate_bounds", {}).get(
                "global_coefficient",
                (0.00, 1.20),
            )
        )
        parameter_residual_bounds = tuple(
            float(item)
            for item in config.get("gate_bounds", {}).get(
                "parameter_residual",
                (-0.15, 0.15),
            )
        )
        loss_config = config.get("loss", {})
        return TorchGlobalParameterCoefficientManager(
            torch_module,
            config.get("initial_gates", {}),
            param_names=list(param_names or []),
            global_bounds=global_bounds,  # type: ignore[arg-type]
            residual_bounds=parameter_residual_bounds,  # type: ignore[arg-type]
            coefficient_bounds=coefficient_bounds,  # type: ignore[arg-type]
            global_prior_scale=float(loss_config.get("global_coefficient_prior_scale", 0.10)),
            residual_prior_scale=float(loss_config.get("parameter_residual_prior_scale", 1.00)),
        ).module
    raise ValueError(f"Unknown gate parameterization: {parameterization}")


def _init_value(init_values: Mapping[str, float], band: str, name: str, default: float) -> float:
    for key in (f"{band}.{name}", f"{band}_{name}", name):
        if key in init_values:
            return float(init_values[key])
    return float(default)


def _config_layer_bands(config: Mapping[str, Any]) -> dict[str, tuple[int, int]]:
    raw = config.get("layer_bands") or config.get("modes", {}).get("layer_bands") or DEFAULT_LAYER_BANDS
    bands = {}
    for name, bounds in raw.items():
        if len(bounds) != 2:
            raise ValueError(f"Layer band {name} must have [start, end] bounds")
        bands[str(name)] = (int(bounds[0]), int(bounds[1]))
    return bands


def _initial_parameter_coefficients(init_values: Mapping[str, float], param_names: tuple[str, ...]) -> list[list[float]]:
    fallback = _fallback_expert_coefficients(init_values)
    rows = []
    for param_name in param_names:
        row = []
        for expert in EXPERT_NAMES:
            row.append(float(init_values.get(f"{param_name}::{expert}", fallback[expert])))
        rows.append(row)
    return rows


def _initial_global_coefficients(init_values: Mapping[str, float]) -> list[float]:
    fallback = _fallback_expert_coefficients(init_values)
    values = []
    for expert in EXPERT_NAMES:
        values.append(float(init_values.get(expert, init_values.get(f"global.{expert}", fallback[expert]))))
    return values


def _initial_global_parameter_coefficients(
    init_values: Mapping[str, float],
    param_names: tuple[str, ...],
    *,
    residual_bounds: tuple[float, float],
) -> tuple[list[float], list[list[float]]]:
    rows = _initial_parameter_coefficients(init_values, param_names)
    explicit_global = [init_values.get(f"__global__::{expert}") for expert in EXPERT_NAMES]
    if any(value is not None for value in explicit_global):
        global_values = [
            float(value) if value is not None else sum(row[expert_idx] for row in rows) / len(rows)
            for expert_idx, value in enumerate(explicit_global)
        ]
    else:
        global_values = [sum(row[expert_idx] for row in rows) / len(rows) for expert_idx in range(len(EXPERT_NAMES))]
    residuals = []
    for param_name, row in zip(param_names, rows):
        residual_row = []
        for expert_idx, expert in enumerate(EXPERT_NAMES):
            residual_key = f"__residual__::{param_name}::{expert}"
            raw_residual = init_values.get(residual_key)
            if raw_residual is None:
                raw_residual = row[expert_idx] - global_values[expert_idx]
            residual_row.append(
                max(float(residual_bounds[0]), min(float(raw_residual), float(residual_bounds[1])))
            )
        residuals.append(residual_row)
    return global_values, residuals


def _fallback_expert_coefficients(init_values: Mapping[str, float]) -> dict[str, float]:
    common = float(init_values.get("common", 0.5))
    residuals = {
        "tool": float(init_values.get("tool_residual", 0.0)),
        "memory": float(init_values.get("memory_residual", 0.0)),
        "code": float(init_values.get("code_residual", 0.0)),
    }
    mean_residual = sum(residuals.values()) / len(residuals)
    return {expert: common + residual - mean_residual for expert, residual in residuals.items()}
