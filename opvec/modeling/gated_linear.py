"""Functional linear layer with frozen OP-VEC expert deltas."""

from __future__ import annotations

from typing import Any, Mapping


class GatedLinear:
    """Wrap a torch.nn.Linear as W = W0 + sum_i coeff_i * Delta_i."""

    def __init__(
        self,
        torch_module: Any,
        base_linear: Any,
        delta_tensors: Mapping[str, Any],
        gate_manager: Any,
        *,
        param_name: str | None = None,
    ):
        nn = torch_module.nn
        functional = torch_module.nn.functional

        class _GatedLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.base_linear = base_linear
                object.__setattr__(self, "gate_manager", gate_manager)
                self.param_name = param_name
                for param in self.base_linear.parameters():
                    param.requires_grad_(False)
                target_weight = self.base_linear.weight
                self.delta_expert_names = tuple(str(name) for name in delta_tensors)
                for expert_name, delta in delta_tensors.items():
                    if list(delta.shape) != list(self.base_linear.weight.shape):
                        raise ValueError(f"Delta shape mismatch for {expert_name}")
                    frozen_delta = delta.detach().to(device=target_weight.device, dtype=target_weight.dtype).contiguous()
                    self.register_buffer(f"delta_{expert_name}", frozen_delta)

            def forward(self, x):
                out = self.base_linear(x)
                try:
                    coeffs = self.gate_manager.expert_coefficients(param_name=self.param_name)
                except TypeError:
                    coeffs = self.gate_manager.expert_coefficients()
                for expert_name in self.delta_expert_names:
                    buffer_name = f"delta_{expert_name}"
                    if not hasattr(self, buffer_name):
                        continue
                    if expert_name not in coeffs:
                        continue
                    coeff = coeffs[expert_name]
                    delta = getattr(self, buffer_name)
                    delta_out = functional.linear(x, delta, None)
                    out = out + coeff.to(device=delta_out.device, dtype=delta_out.dtype) * delta_out
                return out

        self.module = _GatedLinear()

    def __getattr__(self, name: str):
        return getattr(self.module, name)
