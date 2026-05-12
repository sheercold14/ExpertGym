import unittest


class GatedLinearTest(unittest.TestCase):
    def test_zero_and_common_gates_match_manual_forward(self):
        import torch

        from opvec.modeling.gate_parameters import TorchGateManager
        from opvec.modeling.gated_linear import GatedLinear

        base = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            base.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        deltas = {
            "tool": torch.tensor([[0.1, 0.0], [0.0, 0.1]]),
            "memory": torch.tensor([[0.0, 0.2], [0.2, 0.0]]),
            "code": torch.tensor([[0.3, 0.0], [0.0, 0.3]]),
        }
        gates = TorchGateManager(torch, {"common": 0.0}).module
        wrapped = GatedLinear(torch, base, deltas, gates).module
        x = torch.tensor([[1.0, 2.0]])
        self.assertTrue(torch.allclose(wrapped(x), base(x)))

        with torch.no_grad():
            gates.raw_common.fill_(0.5)
        manual_weight = base.weight + 0.5 * (deltas["tool"] + deltas["memory"] + deltas["code"])
        self.assertTrue(torch.allclose(wrapped(x), torch.nn.functional.linear(x, manual_weight)))

    def test_only_gate_parameters_receive_gradients(self):
        import torch

        from opvec.modeling.gate_parameters import TorchGateManager
        from opvec.modeling.gated_linear import GatedLinear

        base = torch.nn.Linear(2, 1, bias=False)
        deltas = {
            "tool": torch.ones_like(base.weight),
            "memory": torch.zeros_like(base.weight),
            "code": torch.zeros_like(base.weight),
        }
        gates = TorchGateManager(torch, {"common": 0.1, "tool_residual": 0.1}).module
        wrapped = GatedLinear(torch, base, deltas, gates).module
        loss = wrapped(torch.ones(1, 2)).sum()
        loss.backward()
        self.assertIsNone(base.weight.grad)
        self.assertIsNotNone(gates.raw_common.grad)
        self.assertIsNotNone(gates.raw_residual.grad)

    def test_layer_band_gate_uses_param_name(self):
        import torch

        from opvec.modeling.gate_parameters import TorchLayerBandGateManager, layer_band_for_param
        from opvec.modeling.gated_linear import GatedLinear

        self.assertEqual(layer_band_for_param("model.layers.0.mlp.down_proj.weight"), "early")
        self.assertEqual(layer_band_for_param("model.layers.14.mlp.down_proj.weight"), "mid")
        self.assertEqual(layer_band_for_param("model.layers.27.mlp.down_proj.weight"), "late")

        base = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            base.weight.zero_()
        deltas = {
            "tool": torch.ones_like(base.weight),
            "memory": torch.zeros_like(base.weight),
            "code": torch.zeros_like(base.weight),
        }
        gates = TorchLayerBandGateManager(
            torch,
            {
                "common": 0.0,
                "early_tool_residual": 0.0,
                "late_tool_residual": 0.6,
                "late_memory_residual": -0.3,
                "late_code_residual": -0.3,
            },
        ).module
        early = GatedLinear(
            torch,
            base,
            deltas,
            gates,
            param_name="model.layers.0.mlp.down_proj.weight",
        ).module
        late = GatedLinear(
            torch,
            base,
            deltas,
            gates,
            param_name="model.layers.27.mlp.down_proj.weight",
        ).module

        x = torch.ones(1, 1)
        self.assertAlmostEqual(float(early(x).item()), 0.0, places=6)
        self.assertGreater(float(late(x).item()), 0.0)

    def test_parameter_coefficients_are_param_specific_and_trainable(self):
        import torch

        from opvec.modeling.gate_parameters import TorchParameterCoefficientManager
        from opvec.modeling.gated_linear import GatedLinear

        param_a = "model.layers.0.mlp.down_proj.weight"
        param_b = "model.layers.1.mlp.down_proj.weight"
        base_a = torch.nn.Linear(1, 1, bias=False)
        base_b = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            base_a.weight.zero_()
            base_b.weight.zero_()
        deltas = {
            "tool": torch.ones_like(base_a.weight),
            "memory": torch.zeros_like(base_a.weight),
            "code": torch.zeros_like(base_a.weight),
        }
        gates = TorchParameterCoefficientManager(
            torch,
            {
                f"{param_a}::tool": 0.2,
                f"{param_b}::tool": 0.8,
                "common": 0.0,
            },
            param_names=[param_a, param_b],
        ).module
        wrapped_a = GatedLinear(torch, base_a, deltas, gates, param_name=param_a).module
        wrapped_b = GatedLinear(torch, base_b, deltas, gates, param_name=param_b).module

        x = torch.ones(1, 1)
        self.assertAlmostEqual(float(wrapped_a(x).item()), 0.2, places=6)
        self.assertAlmostEqual(float(wrapped_b(x).item()), 0.8, places=6)

        loss = (wrapped_a(x) + wrapped_b(x)).sum()
        loss.backward()
        self.assertIsNone(base_a.weight.grad)
        self.assertIsNone(base_b.weight.grad)
        self.assertIsNotNone(gates.raw_coefficients.grad)
        self.assertGreater(float(gates.raw_coefficients.grad[:, 0].abs().sum().item()), 0.0)

    def test_global_parameter_coefficients_use_global_plus_residual(self):
        import torch

        from opvec.modeling.gate_parameters import TorchGlobalParameterCoefficientManager
        from opvec.modeling.gated_linear import GatedLinear

        param_a = "model.layers.0.mlp.down_proj.weight"
        param_b = "model.layers.1.mlp.down_proj.weight"
        base_a = torch.nn.Linear(1, 1, bias=False)
        base_b = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            base_a.weight.zero_()
            base_b.weight.zero_()
        deltas = {
            "tool": torch.ones_like(base_a.weight),
            "memory": torch.zeros_like(base_a.weight),
            "code": torch.zeros_like(base_a.weight),
        }
        gates = TorchGlobalParameterCoefficientManager(
            torch,
            {
                "__global__::tool": 0.7,
                f"{param_a}::tool": 0.75,
                f"{param_b}::tool": 0.65,
                "common": 0.5,
            },
            param_names=[param_a, param_b],
        ).module
        wrapped_a = GatedLinear(torch, base_a, deltas, gates, param_name=param_a).module
        wrapped_b = GatedLinear(torch, base_b, deltas, gates, param_name=param_b).module

        x = torch.ones(1, 1)
        self.assertAlmostEqual(float(gates.expert_coefficients()["tool"].detach().item()), 0.7, places=6)
        self.assertAlmostEqual(float(wrapped_a(x).item()), 0.75, places=6)
        self.assertAlmostEqual(float(wrapped_b(x).item()), 0.65, places=6)

        loss = (wrapped_a(x) + wrapped_b(x)).sum()
        loss.backward()
        self.assertIsNotNone(gates.raw_global_coefficients.grad)
        self.assertIsNotNone(gates.raw_residual_coefficients.grad)
        self.assertGreater(float(gates.raw_global_coefficients.grad.abs().sum().item()), 0.0)
        self.assertGreater(float(gates.raw_residual_coefficients.grad.abs().sum().item()), 0.0)


if __name__ == "__main__":
    unittest.main()
