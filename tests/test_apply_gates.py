import json
import tempfile
import unittest
from pathlib import Path


class ApplyGatesTest(unittest.TestCase):
    def test_install_gated_linears_from_manifest_on_toy_model(self):
        import torch

        from opvec.modeling.apply_gates import install_gated_linears_from_manifest
        from opvec.modeling.gate_parameters import TorchGateManager

        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(2, 1, bias=False)

            def forward(self, x):
                return self.proj(x)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for expert, value in [("tool", 1.0), ("memory", 0.0), ("code", 0.0)]:
                path = root / f"{expert}.pt"
                torch.save(torch.full((1, 2), value), path)
            manifest = {
                "basis_entries": [
                    {"expert": "tool", "param_name": "proj.weight", "storage_path": "tool.pt"},
                    {"expert": "memory", "param_name": "proj.weight", "storage_path": "memory.pt"},
                    {"expert": "code", "param_name": "proj.weight", "storage_path": "code.pt"},
                ]
            }
            manifest_path = root / "mode_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            model = Toy()
            with torch.no_grad():
                model.proj.weight.zero_()
            gates = TorchGateManager(torch, {"common": 0.0, "tool_residual": 0.3}).module

            installed = install_gated_linears_from_manifest(
                torch,
                model,
                mode_manifest_path=manifest_path,
                gate_manager=gates,
            )

            self.assertEqual(installed, ["proj"])
            out = model(torch.ones(1, 2))
            self.assertGreater(float(out.item()), 0.0)


if __name__ == "__main__":
    unittest.main()
