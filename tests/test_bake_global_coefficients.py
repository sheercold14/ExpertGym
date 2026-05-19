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

    def test_layer_band_bake_keeps_fourth_reasoning_expert(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest_path = root / "mode_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "base_model": str(root / "base"),
                        "expert_names": ["tool", "memory", "code", "reasoning"],
                        "basis_entries": [
                            {"param_name": "model.layers.0.mlp.down_proj.weight", "expert": "tool", "storage_path": "tool.pt"},
                            {"param_name": "model.layers.0.mlp.down_proj.weight", "expert": "memory", "storage_path": "memory.pt"},
                            {"param_name": "model.layers.0.mlp.down_proj.weight", "expert": "code", "storage_path": "code.pt"},
                            {"param_name": "model.layers.0.mlp.down_proj.weight", "expert": "reasoning", "storage_path": "reasoning.pt"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            gate_values = {
                "early.common": 0.25,
                "early.tool_residual": 0.0,
                "early.memory_residual": 0.1,
                "early.code_residual": 0.0,
                "early.reasoning_residual": -0.1,
            }

            plan = create_bake_plan(
                mode_manifest_path=manifest_path,
                gate_values=gate_values,
                output_dir=root / "baked",
                layer_bands={"early": (0, 9)},
            )

        self.assertEqual(plan["gate_parameterization"], "layer-band")
        self.assertIn("reasoning", plan["expert_coefficients"]["early"])
        self.assertAlmostEqual(plan["expert_coefficients"]["early"]["tool"], 0.25)
        self.assertAlmostEqual(plan["expert_coefficients"]["early"]["memory"], 0.35)
        self.assertAlmostEqual(plan["expert_coefficients"]["early"]["code"], 0.25)
        self.assertAlmostEqual(plan["expert_coefficients"]["early"]["reasoning"], 0.15)

    def test_layer_band_direct_coefficients_are_baked_directly(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest_path = root / "mode_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "base_model": str(root / "base"),
                        "basis_entries": [
                            {"param_name": "model.layers.0.mlp.down_proj.weight", "expert": "tool", "storage_path": "tool.pt"},
                            {"param_name": "model.layers.0.mlp.down_proj.weight", "expert": "memory", "storage_path": "memory.pt"},
                            {"param_name": "model.layers.0.mlp.down_proj.weight", "expert": "code", "storage_path": "code.pt"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            gate_values = {
                "early.tool": 0.1,
                "early.memory": 0.2,
                "early.code": 0.3,
            }

            plan = create_bake_plan(
                mode_manifest_path=manifest_path,
                gate_values=gate_values,
                output_dir=root / "baked",
                layer_bands={"early": (0, 9)},
            )

        self.assertEqual(plan["gate_parameterization"], "layer-band")
        self.assertEqual(plan["expert_coefficients"]["early"], {"tool": 0.1, "memory": 0.2, "code": 0.3})

    def test_layer_band_parameter_globals_do_not_trigger_parameter_bake(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest_path = root / "mode_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "base_model": str(root / "base"),
                        "basis_entries": [
                            {"param_name": "model.layers.0.mlp.down_proj.weight", "expert": "tool", "storage_path": "tool.pt"},
                            {"param_name": "model.layers.0.mlp.down_proj.weight", "expert": "memory", "storage_path": "memory.pt"},
                            {"param_name": "model.layers.0.mlp.down_proj.weight", "expert": "code", "storage_path": "code.pt"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            gate_values = {
                "__global__::tool": 0.5,
                "__global__::memory": 0.6,
                "__global__::code": 0.7,
                "early.tool": 0.1,
                "early.memory": 0.2,
                "early.code": 0.3,
            }

            plan = create_bake_plan(
                mode_manifest_path=manifest_path,
                gate_values=gate_values,
                output_dir=root / "baked",
                layer_bands={"early": (0, 9)},
            )

        self.assertEqual(plan["gate_parameterization"], "layer-band-parameter")
        self.assertEqual(plan["expert_coefficients"]["early"], {"tool": 0.1, "memory": 0.2, "code": 0.3})


if __name__ == "__main__":
    unittest.main()
