"""OP-VEC-4 scalar gate algebra."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


TASK_ORDER = ("tool", "memory", "code")


@dataclass(frozen=True)
class GateValues:
    common: float
    tool_residual: float = 0.0
    memory_residual: float = 0.0
    code_residual: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "common": float(self.common),
            "tool_residual": float(self.tool_residual),
            "memory_residual": float(self.memory_residual),
            "code_residual": float(self.code_residual),
        }


def effective_gates(values: GateValues | Mapping[str, float]) -> dict[str, float]:
    """Return common plus zero-mean residual gates."""

    gate = _coerce(values)
    residuals = [gate.tool_residual, gate.memory_residual, gate.code_residual]
    mean_residual = sum(residuals) / 3.0
    return {
        "common": float(gate.common),
        "tool_residual": float(gate.tool_residual - mean_residual),
        "memory_residual": float(gate.memory_residual - mean_residual),
        "code_residual": float(gate.code_residual - mean_residual),
    }


def expert_coefficients(values: GateValues | Mapping[str, float]) -> dict[str, float]:
    """Map OP-VEC gates to coefficients on physical expert deltas."""

    gates = effective_gates(values)
    return {
        "tool": gates["common"] + gates["tool_residual"],
        "memory": gates["common"] + gates["memory_residual"],
        "code": gates["common"] + gates["code_residual"],
    }


def project_gates(
    values: GateValues | Mapping[str, float],
    *,
    common_bounds: tuple[float, float] = (-0.10, 1.50),
    residual_bounds: tuple[float, float] = (-0.50, 0.50),
) -> GateValues:
    """Clamp gates and project residuals back to zero mean."""

    gates = effective_gates(values)
    common = min(max(gates["common"], common_bounds[0]), common_bounds[1])
    residuals = [
        min(max(gates["tool_residual"], residual_bounds[0]), residual_bounds[1]),
        min(max(gates["memory_residual"], residual_bounds[0]), residual_bounds[1]),
        min(max(gates["code_residual"], residual_bounds[0]), residual_bounds[1]),
    ]
    mean_residual = sum(residuals) / 3.0
    residuals = [value - mean_residual for value in residuals]
    return GateValues(common=common, tool_residual=residuals[0], memory_residual=residuals[1], code_residual=residuals[2])


def _coerce(values: GateValues | Mapping[str, float]) -> GateValues:
    if isinstance(values, GateValues):
        return values
    return GateValues(
        common=float(values.get("common", values.get("a_common", 0.0))),
        tool_residual=float(values.get("tool_residual", values.get("a_tool", 0.0))),
        memory_residual=float(values.get("memory_residual", values.get("a_memory", 0.0))),
        code_residual=float(values.get("code_residual", values.get("a_code", 0.0))),
    )
