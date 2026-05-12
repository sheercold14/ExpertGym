import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import torch

from opvec.modeling.gate_parameters import make_torch_gate_manager


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts/train/opvec_update_gates_from_rollouts.py"
SPEC = importlib.util.spec_from_file_location("opvec_update_gates_from_rollouts", SCRIPT_PATH)
update_gates_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(update_gates_module)


class UpdateGatesObjectivesTest(unittest.TestCase):
    def test_pairwise_best_response_loss_pushes_positive_above_negative(self):
        positive = torch.tensor(-2.0, requires_grad=True)
        negative = torch.tensor(-1.0, requires_grad=True)
        loss = update_gates_module._pairwise_best_response_loss(
            torch,
            [
                {"current": positive, "reward": 1.0, "length": 10},
                {"current": negative, "reward": 0.0, "length": 10},
            ],
            margin=0.0,
            length_normalize=False,
            positive_reward_threshold=None,
        )
        self.assertGreater(float(loss.detach().item()), 0.0)
        loss.backward()
        self.assertLess(float(positive.grad.item()), 0.0)
        self.assertGreater(float(negative.grad.item()), 0.0)

    def test_coefficient_projection_freezes_unlisted_coefficients(self):
        config = {
            "initial_gates": {
                "early.common": 0.5,
                "early.tool_residual": 0.0,
                "early.memory_residual": 0.1,
                "early.code_residual": -0.1,
                "mid.common": 0.6,
                "mid.tool_residual": -0.2,
                "mid.memory_residual": 0.2,
                "mid.code_residual": 0.0,
                "late.common": 0.4,
                "late.tool_residual": 0.1,
                "late.memory_residual": -0.1,
                "late.code_residual": 0.0,
            },
            "gate_bounds": {"common": [-0.1, 1.5], "residual": [-0.5, 0.5]},
        }
        manager = make_torch_gate_manager(torch, config, parameterization="layer-band")
        with torch.no_grad():
            manager.raw_common.add_(0.05)
            manager.raw_residual.add_(torch.tensor([[0.2, -0.1, -0.1], [0.3, -0.2, -0.1], [0.2, -0.1, -0.1]]))
        update_gates_module._project_trainable_coefficients_(
            torch,
            manager,
            {"mid.tool", "late.tool"},
            dict(config["initial_gates"]),
        )

        early = {key: float(value.detach().item()) for key, value in manager.expert_coefficients(band="early").items()}
        mid = {key: float(value.detach().item()) for key, value in manager.expert_coefficients(band="mid").items()}
        late = {key: float(value.detach().item()) for key, value in manager.expert_coefficients(band="late").items()}

        self.assertAlmostEqual(early["tool"], 0.5, places=6)
        self.assertAlmostEqual(early["memory"], 0.6, places=6)
        self.assertAlmostEqual(early["code"], 0.4, places=6)
        self.assertAlmostEqual(mid["memory"], 0.8, places=6)
        self.assertAlmostEqual(mid["code"], 0.6, places=6)
        self.assertAlmostEqual(late["memory"], 0.3, places=6)
        self.assertAlmostEqual(late["code"], 0.4, places=6)
        self.assertNotAlmostEqual(mid["tool"], 0.4, places=3)
        self.assertNotAlmostEqual(late["tool"], 0.5, places=3)

    def test_manifest_param_names_drive_parameter_coefficients(self):
        from opvec.modeling.manifest import manifest_param_names

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "mode_manifest.json"
            path.write_text(
                json.dumps(
                    {
                        "basis_entries": [
                            {"param_name": "model.layers.1.mlp.down_proj.weight", "expert": "tool"},
                            {"param_name": "model.layers.0.mlp.down_proj.weight", "expert": "memory"},
                            {"param_name": "lm_head.bias", "expert": "code"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                manifest_param_names(path),
                [
                    "model.layers.0.mlp.down_proj.weight",
                    "model.layers.1.mlp.down_proj.weight",
                ],
            )

    def test_parameter_prior_anchors_initial_coefficients(self):
        param = "model.layers.0.mlp.down_proj.weight"
        manager = make_torch_gate_manager(
            torch,
            {
                "initial_gates": {f"{param}::tool": 0.75, "common": 0.5},
                "gate_bounds": {"coefficient": [-0.5, 1.5]},
            },
            parameterization="parameter",
            param_names=[param],
        )
        self.assertAlmostEqual(float(update_gates_module._gate_prior_loss(torch, manager).detach().item()), 0.0, places=6)
        with torch.no_grad():
            manager.raw_coefficients[0, 0].add_(0.3)
        self.assertGreater(float(update_gates_module._gate_prior_loss(torch, manager).detach().item()), 0.0)

    def test_global_parameter_prior_separates_global_and_residual(self):
        param_a = "model.layers.0.mlp.down_proj.weight"
        param_b = "model.layers.1.mlp.down_proj.weight"
        manager = make_torch_gate_manager(
            torch,
            {
                "initial_gates": {f"{param_a}::tool": 0.8, f"{param_b}::tool": 0.6, "common": 0.5},
                "gate_bounds": {"coefficient": [-0.5, 1.5], "global_coefficient": [0.0, 1.2], "parameter_residual": [-0.15, 0.15]},
                "loss": {"global_coefficient_prior_scale": 0.1, "parameter_residual_prior_scale": 1.0},
            },
            parameterization="global-parameter",
            param_names=[param_a, param_b],
        )
        self.assertAlmostEqual(float(manager.expert_coefficients()["tool"].detach().item()), 0.7, places=6)
        self.assertAlmostEqual(float(update_gates_module._gate_prior_loss(torch, manager).detach().item()), 0.0, places=6)
        with torch.no_grad():
            manager.raw_global_coefficients[0].add_(0.2)
        self.assertAlmostEqual(float(update_gates_module._gate_prior_loss(torch, manager).detach().item()), (0.2**2) / 3.0 * 0.1, places=6)

    def test_max_delta_projection_bounds_global_parameter_manager(self):
        param = "model.layers.0.mlp.down_proj.weight"
        manager = make_torch_gate_manager(
            torch,
            {
                "initial_gates": {f"{param}::tool": 0.6, f"{param}::memory": 0.5, f"{param}::code": 0.7},
                "gate_bounds": {"coefficient": [-0.5, 1.5], "global_coefficient": [0.0, 1.2], "parameter_residual": [-0.15, 0.15]},
            },
            parameterization="global-parameter",
            param_names=[param],
        )
        initial_global = manager.initial_global_coefficients.detach().clone()
        initial_residual = manager.initial_residual_coefficients.detach().clone()
        with torch.no_grad():
            manager.raw_global_coefficients.add_(0.4)
            manager.raw_residual_coefficients.add_(0.4)
        update_gates_module._project_max_delta_from_initial_(torch, manager, 0.03)
        self.assertLessEqual(float((manager.raw_global_coefficients - initial_global).abs().max().item()), 0.030001)
        self.assertLessEqual(float((manager.raw_residual_coefficients - initial_residual).abs().max().item()), 0.030001)

    def test_frontier_task_quota_caps_memory_rows(self):
        rows = [
            {"prompt_id": "m0", "task": "memory"},
            {"prompt_id": "m1", "task": "memory"},
            {"prompt_id": "m2", "task": "memory"},
            {"prompt_id": "c0", "task": "code"},
            {"prompt_id": "t0", "task": "tool"},
        ]
        limited = update_gates_module._limit_frontier_rows(rows, task_quota={"memory": 1}, max_per_task=None)
        self.assertEqual([row["prompt_id"] for row in limited], ["m0", "c0", "t0"])
        counts = update_gates_module._task_counts(limited)
        self.assertEqual(counts, {"memory": 1, "code": 1, "tool": 1})

    def test_config_task_weights_are_overridden_by_cli(self):
        weights = update_gates_module._merged_float_mapping({"memory": 0.5, "code": 2.0}, ["memory=0.25"])
        self.assertEqual(weights, {"memory": 0.25, "code": 2.0})

    def test_best_response_samples_do_not_require_old_logprob(self):
        samples = [
            {"text": "good answer", "reward": 1.0},
            {"text": "bad answer", "reward": 0.0, "old_logprob": None},
            {"text": "", "reward": 1.0},
        ]
        self.assertEqual(len(update_gates_module._objective_samples(samples, require_old_logprob=False)), 2)
        self.assertEqual(len(update_gates_module._objective_samples(samples, require_old_logprob=True)), 0)

    def test_trajectory_logprob_sums_turns(self):
        original = update_gates_module.response_logprob_tensor_from_text

        def fake_logprob(torch_module, model, tokenizer, *, prompt_text, response_text, device, max_length):
            return torch.tensor(float(len(prompt_text) + len(response_text)), requires_grad=True)

        update_gates_module.response_logprob_tensor_from_text = fake_logprob
        try:
            value = update_gates_module._sample_response_logprob_tensor(
                torch,
                None,
                None,
                prompt_text="unused",
                sample={
                    "text": "final",
                    "trajectory": [
                        {"prompt_text": "aa", "text": "b"},
                        {"prompt_text": "c", "text": "dddd"},
                    ],
                },
                device="cpu",
                max_length=128,
            )
        finally:
            update_gates_module.response_logprob_tensor_from_text = original

        self.assertAlmostEqual(float(value.detach().item()), 8.0, places=6)

    def test_bounded_pairwise_samples_prefers_high_positive_low_negative(self):
        positives = [
            {"sample_id": "p_low", "reward": 0.8},
            {"sample_id": "p_high", "reward": 1.0},
        ]
        negatives = [
            {"sample_id": "n_mid", "reward": 0.4},
            {"sample_id": "n_low", "reward": 0.0},
        ]

        pairs = update_gates_module._bounded_pairwise_samples(positives, negatives, max_pairs=2)

        self.assertEqual(
            [(positive["sample_id"], negative["sample_id"]) for positive, negative in pairs],
            [("p_high", "n_low"), ("p_high", "n_mid")],
        )


if __name__ == "__main__":
    unittest.main()
