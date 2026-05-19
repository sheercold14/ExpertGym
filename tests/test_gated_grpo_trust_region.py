import unittest

import torch

from opvec.modeling.gate_parameters import make_torch_gate_manager
from scripts.train.opvec_update_gates_from_rollouts import (
    _parse_expert_bounds_items,
    _project_effective_coefficient_bounds_,
    _project_max_delta_from_initial_,
)


class GatedGrpoTrustRegionTest(unittest.TestCase):
    def test_global_gate_projection_limits_common_and_residual(self):
        config = {
            "initial_gates": {"common": 0.75, "tool_residual": 0.0, "memory_residual": 0.0, "code_residual": 0.0},
            "gate_bounds": {"common": [0.0, 1.5], "residual": [-1.0, 1.0]},
        }
        manager = make_torch_gate_manager(torch, config, parameterization="global")
        with torch.no_grad():
            manager.raw_common.fill_(1.40)
            manager.raw_residual.copy_(torch.tensor([0.60, -0.40, -0.20]))
        _project_max_delta_from_initial_(torch, manager, 0.10)
        self.assertLessEqual(abs(float(manager.raw_common.item()) - 0.75), 0.100001)
        deltas = (manager.raw_residual - manager.initial_raw_residual).detach().abs()
        self.assertTrue(bool((deltas <= 0.100001).all()))

    def test_parameter_gate_projection_limits_coefficients(self):
        config = {
            "initial_gates": {"common": 0.75},
            "gate_bounds": {"coefficient": [0.0, 1.5]},
        }
        manager = make_torch_gate_manager(torch, config, parameterization="parameter", param_names=["p0", "p1"])
        with torch.no_grad():
            manager.raw_coefficients.fill_(1.30)
        _project_max_delta_from_initial_(torch, manager, 0.20)
        deltas = (manager.raw_coefficients - manager.initial_coefficients).detach().abs()
        self.assertTrue(bool((deltas <= 0.200001).all()))

    def test_parameter_gate_projection_supports_expert_specific_delta(self):
        config = {
            "modes": {"expert_names": ["tool", "memory", "code", "reasoning"]},
            "initial_gates": {
                "p0::tool": 1.0,
                "p0::memory": 1.0,
                "p0::code": 1.0,
                "p0::reasoning": 0.001,
                "p1::tool": 1.0,
                "p1::memory": 1.0,
                "p1::code": 1.0,
                "p1::reasoning": 0.001,
            },
            "gate_bounds": {"coefficient": [0.0, 1.5]},
        }
        manager = make_torch_gate_manager(torch, config, parameterization="parameter", param_names=["p0", "p1"])
        with torch.no_grad():
            manager.raw_coefficients[:, 0].fill_(1.20)
            manager.raw_coefficients[:, 1].fill_(1.20)
            manager.raw_coefficients[:, 2].fill_(1.20)
            manager.raw_coefficients[:, 3].fill_(0.20)
        _project_max_delta_from_initial_(torch, manager, 0.20, max_delta_by_expert={"reasoning": 0.002})
        deltas = (manager.raw_coefficients - manager.initial_coefficients).detach().abs()
        self.assertTrue(bool((deltas[:, :3] <= 0.200001).all()))
        self.assertTrue(bool((deltas[:, 3] <= 0.002001).all()))
        self.assertAlmostEqual(float(manager.raw_coefficients[0, 3].item()), 0.003, places=6)

    def test_layer_band_projection_supports_absolute_expert_bounds(self):
        config = {
            "modes": {"expert_names": ["tool", "memory", "code", "reasoning"]},
            "layer_bands": {"layer0": [0, 0], "layer1": [1, 1]},
            "initial_gates": {
                "layer0.common": 0.75025,
                "layer0.tool_residual": 0.24975,
                "layer0.memory_residual": 0.24975,
                "layer0.code_residual": 0.24975,
                "layer0.reasoning_residual": -0.74925,
                "layer1.common": 0.75025,
                "layer1.tool_residual": 0.24975,
                "layer1.memory_residual": 0.24975,
                "layer1.code_residual": 0.24975,
                "layer1.reasoning_residual": -0.74925,
            },
            "gate_bounds": {"common": [-1.0, 2.0], "residual": [-2.0, 2.0]},
        }
        manager = make_torch_gate_manager(torch, config, parameterization="layer-band")
        with torch.no_grad():
            manager.raw_common.copy_(torch.tensor([0.75, 0.75]))
            manager.raw_residual.copy_(
                torch.tensor(
                    [
                        [0.25, 0.25, 0.25, -0.80],
                        [0.25, 0.25, 0.25, -0.70],
                    ]
                )
            )
        _project_effective_coefficient_bounds_(
            torch,
            manager,
            _parse_expert_bounds_items(["reasoning=0.0:0.003"]),
        )
        layer0 = manager.expert_coefficients(band="layer0")
        layer1 = manager.expert_coefficients(band="layer1")
        layer0_reasoning = float(layer0["reasoning"].detach().item())
        layer1_reasoning = float(layer1["reasoning"].detach().item())
        self.assertGreaterEqual(layer0_reasoning, -1e-7)
        self.assertLessEqual(layer0_reasoning, 0.003001)
        self.assertGreaterEqual(layer1_reasoning, -1e-7)
        self.assertLessEqual(layer1_reasoning, 0.003001)
        self.assertGreater(float(layer0["code"].detach().item()), 0.9)

    def test_parse_expert_bounds_items_supports_one_sided_bounds(self):
        bounds = _parse_expert_bounds_items(["reasoning=:0.01", "code=0.5:"])
        self.assertEqual(bounds["reasoning"], (None, 0.01))
        self.assertEqual(bounds["code"], (0.5, None))


if __name__ == "__main__":
    unittest.main()
