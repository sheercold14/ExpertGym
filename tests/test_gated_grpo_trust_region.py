import unittest

import torch

from opvec.modeling.gate_parameters import make_torch_gate_manager
from scripts.train.opvec_update_gates_from_rollouts import _project_max_delta_from_initial_


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


if __name__ == "__main__":
    unittest.main()
