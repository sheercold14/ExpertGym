import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import torch

from opvec.modeling.gate_parameters import make_torch_gate_manager


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts/modes/build_zero_gate_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("build_zero_gate_checkpoint", SCRIPT_PATH)
zero_gate_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(zero_gate_module)


class ZeroGateCheckpointTest(unittest.TestCase):
    def test_global_parameter_checkpoint_starts_all_coefficients_at_zero(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest_path = Path(temp) / "mode_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "basis_entries": [
                            {"param_name": "model.layers.0.self_attn.q_proj.weight", "expert": "tool"},
                            {"param_name": "model.layers.1.mlp.down_proj.weight", "expert": "memory"},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            gates, metadata = zero_gate_module.build_zero_gate_checkpoint(
                mode_manifest=manifest_path,
                parameterization="global-parameter",
            )

        self.assertEqual(metadata["num_mergeable_params"], 2)
        self.assertEqual(gates["common"], 0.0)
        self.assertEqual(gates["__global__::tool"], 0.0)
        self.assertEqual(gates["model.layers.0.self_attn.q_proj.weight::memory"], 0.0)

        manager = make_torch_gate_manager(
            torch,
            {
                "initial_gates": gates,
                "gate_bounds": {
                    "coefficient": [-0.5, 1.5],
                    "global_coefficient": [0.0, 1.2],
                    "parameter_residual": [-0.15, 0.15],
                },
            },
            parameterization="global-parameter",
            param_names=[
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.1.mlp.down_proj.weight",
            ],
        )
        effective = manager.effective_coefficients().detach()
        self.assertTrue(torch.equal(effective, torch.zeros_like(effective)))
        self.assertTrue(torch.equal(manager.raw_global_coefficients.detach(), torch.zeros_like(manager.raw_global_coefficients)))

    def test_plain_global_checkpoint_overrides_common_default(self):
        gates, metadata = zero_gate_module.build_zero_gate_checkpoint(
            mode_manifest="unused-for-global.json",
            parameterization="global",
        )

        self.assertEqual(metadata["num_mergeable_params"], 0)
        manager = make_torch_gate_manager(
            torch,
            {"initial_gates": gates, "gate_bounds": {"common": [-0.1, 1.5], "residual": [-0.5, 0.5]}},
            parameterization="global",
        )
        coefficients = manager.expert_coefficients()
        self.assertEqual(float(coefficients["tool"].detach().item()), 0.0)
        self.assertEqual(float(coefficients["memory"].detach().item()), 0.0)
        self.assertEqual(float(coefficients["code"].detach().item()), 0.0)


if __name__ == "__main__":
    unittest.main()
