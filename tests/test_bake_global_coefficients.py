import json
import tempfile
import unittest
from pathlib import Path

from opvec.modeling.bake import create_bake_plan, load_gate_values


class BakeGlobalCoefficientsTest(unittest.TestCase):
    def test_direct_global_coefficients_are_not_interpreted_as_common_residual(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            gate_path = root / "gates.json"
            gate_path.write_text(
                json.dumps({"gates": {"tool": 0.1, "memory": 0.2, "code": 0.3}}),
                encoding="utf-8",
            )
            manifest_path = root / "mode_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "base_model": str(root / "base"),
                        "basis_entries": [
                            {"param_name": "p0", "expert": "tool", "storage_path": "tool.pt"},
                            {"param_name": "p0", "expert": "memory", "storage_path": "memory.pt"},
                            {"param_name": "p1", "expert": "code", "storage_path": "code.pt"},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            gate_values = load_gate_values(gate_path)
            plan = create_bake_plan(
                mode_manifest_path=manifest_path,
                gate_values=gate_values,
                output_dir=root / "baked",
            )

        self.assertEqual(gate_values, {"tool": 0.1, "memory": 0.2, "code": 0.3})
        self.assertEqual(plan["gate_parameterization"], "global-coefficient")
        self.assertEqual(plan["expert_coefficients"], {"tool": 0.1, "memory": 0.2, "code": 0.3})


if __name__ == "__main__":
    unittest.main()
