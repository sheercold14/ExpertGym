import importlib.util
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch

from opvec.modeling.gate_parameters import make_torch_gate_manager


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts/train/opvec_update_gates_from_rollouts.py"
SPEC = importlib.util.spec_from_file_location("opvec_update_gates_from_rollouts", SCRIPT_PATH)
update_gates_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(update_gates_module)


class UpdateGatesObjectivesTest(unittest.TestCase):
    def test_update_batcher_accumulates_rows_before_step(self):
        class TinyGateManager(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))
                self.project_calls = 0

            def project_(self):
                self.project_calls += 1

            def gate_values(self):
                return {"weight": float(self.weight.detach().item())}

        manager = TinyGateManager()
        optimizer = torch.optim.SGD(manager.parameters(), lr=1.0)
        batcher = update_gates_module._UpdateBatcher(
            torch=torch,
            optimizer=optimizer,
            gate_manager=manager,
            grad_clip_norm=10.0,
            min_grad_norm_for_step=0.0,
            update_batch_size=2,
            batch_loss_reduction="mean",
            optimizer_step_scope="batch",
            loss_normalizer=2,
            train_coefficients=set(),
            coefficient_anchor_gates={},
            args=Namespace(
                max_coefficient_delta_from_init=None,
                max_coefficient_delta_from_init_by_expert=[],
                coefficient_bound_by_expert=[],
                tool_min_margin_over_memory=0.0,
                tool_min_margin_over_code=0.0,
            ),
        )
        log_rows = [{"prompt_id": "a"}, {"prompt_id": "b"}]
        (manager.weight * batcher.loss_scale).backward()
        batcher.add(log_rows, 0)
        self.assertAlmostEqual(float(manager.weight.detach().item()), 1.0, places=6)
        (manager.weight * batcher.loss_scale).backward()
        batcher.add(log_rows, 1)

        self.assertAlmostEqual(float(manager.weight.detach().item()), 0.0, places=6)
        self.assertEqual(batcher.optimizer_steps, 1)
        self.assertEqual(manager.project_calls, 1)
        self.assertEqual(log_rows[0]["optimizer_step_index"], 1)
        self.assertEqual(log_rows[1]["optimizer_step_index"], 1)
        self.assertAlmostEqual(log_rows[0]["batch_loss_scale"], 0.5, places=6)

    def test_update_batcher_can_defer_optimizer_step_to_epoch(self):
        class TinyGateManager(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))

            def gate_values(self):
                return {"weight": float(self.weight.detach().item())}

            def project_(self):
                pass

        manager = TinyGateManager()
        optimizer = torch.optim.SGD(manager.parameters(), lr=1.0)
        batcher = update_gates_module._UpdateBatcher(
            torch=torch,
            optimizer=optimizer,
            gate_manager=manager,
            grad_clip_norm=10.0,
            min_grad_norm_for_step=0.0,
            update_batch_size=2,
            batch_loss_reduction="mean",
            optimizer_step_scope="epoch",
            loss_normalizer=4,
            train_coefficients=set(),
            coefficient_anchor_gates={},
            args=Namespace(
                max_coefficient_delta_from_init=None,
                max_coefficient_delta_from_init_by_expert=[],
                coefficient_bound_by_expert=[],
                tool_min_margin_over_memory=0.0,
                tool_min_margin_over_code=0.0,
            ),
        )
        log_rows = [{"prompt_id": str(idx)} for idx in range(4)]
        for idx in range(4):
            (manager.weight * batcher.loss_scale).backward()
            batcher.add(log_rows, idx)
        self.assertAlmostEqual(float(manager.weight.detach().item()), 1.0, places=6)
        self.assertEqual(batcher.optimizer_steps, 0)

        batcher.flush(log_rows, force=True)

        self.assertAlmostEqual(float(manager.weight.detach().item()), 0.0, places=6)
        self.assertEqual(batcher.optimizer_steps, 1)
        self.assertAlmostEqual(log_rows[0]["batch_loss_scale"], 0.25, places=6)
        self.assertEqual(log_rows[0]["optimizer_step_scope"], "epoch")
        self.assertEqual(log_rows[-1]["optimizer_step_index"], 1)

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

    def test_token_logprob_entry_uses_response_mask(self):
        original = update_gates_module.response_logprob_tensor_details_from_text

        def fake_details(torch_module, model, tokenizer, *, prompt_text, response_text, device, max_length):
            return {
                "response_token_ids": [1, 2],
                "logprobs": torch.tensor([-0.2, -0.3], requires_grad=True),
                "response_mask": [1, 1],
            }

        update_gates_module.response_logprob_tensor_details_from_text = fake_details
        try:
            entry = update_gates_module._sample_response_logprob_entry(
                torch,
                None,
                None,
                prompt_text="prompt",
                sample={
                    "sample_id": "s0",
                    "text": "answer",
                    "old_logprobs": [-0.1, -0.2],
                    "response_mask": [1, 0],
                },
                device="cpu",
                max_length=128,
                loss_granularity="token",
            )
        finally:
            update_gates_module.response_logprob_tensor_details_from_text = original

        self.assertIsNotNone(entry)
        self.assertAlmostEqual(float(entry["current"].detach().item()), -0.2, places=6)
        self.assertAlmostEqual(float(entry["old"].detach().item()), -0.1, places=6)
        self.assertEqual(int(entry["current_logprobs"].numel()), 2)

    def test_token_logprob_entry_prefers_stored_response_token_ids(self):
        original_ids = update_gates_module.response_logprob_tensor_details_from_token_ids
        original_text = update_gates_module.response_logprob_tensor_details_from_text
        calls = []

        def fake_details_from_ids(torch_module, model, tokenizer, *, prompt_text, response_token_ids, device, max_length):
            calls.append(("ids", list(response_token_ids)))
            return {
                "response_token_ids": list(response_token_ids),
                "logprobs": torch.tensor([-0.2, -0.3], requires_grad=True),
                "response_mask": [1, 1],
            }

        def fake_details_from_text(*args, **kwargs):
            calls.append(("text", None))
            raise AssertionError("text tokenizer path should not be used")

        update_gates_module.response_logprob_tensor_details_from_token_ids = fake_details_from_ids
        update_gates_module.response_logprob_tensor_details_from_text = fake_details_from_text
        try:
            entry = update_gates_module._sample_response_logprob_entry(
                torch,
                None,
                None,
                prompt_text="prompt",
                sample={
                    "sample_id": "s0",
                    "text": "answer",
                    "response_token_ids": [7, 8],
                    "old_logprobs": [-0.1, -0.2],
                    "response_mask": [1, 1],
                },
                device="cpu",
                max_length=128,
                loss_granularity="token",
            )
        finally:
            update_gates_module.response_logprob_tensor_details_from_token_ids = original_ids
            update_gates_module.response_logprob_tensor_details_from_text = original_text

        self.assertIsNotNone(entry)
        self.assertEqual(calls, [("ids", [7, 8])])

    def test_token_policy_metrics_respect_mask(self):
        metrics = update_gates_module._entry_policy_metrics(
            torch,
            {
                "current_logprobs": torch.tensor([-0.2, -10.0]),
                "old_logprobs": torch.tensor([-0.1, -0.1]),
                "response_mask": torch.tensor([1, 0]),
            },
            {},
            length_normalize=False,
            loss_granularity="token",
            eps_clip=0.05,
        )

        self.assertEqual(metrics["clip_frac"], 1.0)
        self.assertAlmostEqual(metrics["approx_kl"], 0.1, places=6)

    def test_fill_missing_old_logprobs_adds_token_payload_for_trajectory(self):
        original = update_gates_module.response_logprob_tensor_details_from_text

        class Model:
            def eval(self):
                pass

            def train(self):
                pass

        def fake_details(torch_module, model, tokenizer, *, prompt_text, response_text, device, max_length):
            length = len(response_text)
            return {
                "response_token_ids": list(range(length)),
                "logprobs": torch.full((length,), -1.0),
                "response_mask": [1 for _ in range(length)],
            }

        rows = [
            {
                "rendered_prompt": "unused",
                "samples": [
                    {
                        "sample_id": "memory__k0",
                        "text": "final",
                        "trajectory": [
                            {"prompt_text": "p0", "text": "a"},
                            {"prompt_text": "p1", "text": "bc"},
                        ],
                    }
                ],
            }
        ]
        update_gates_module.response_logprob_tensor_details_from_text = fake_details
        try:
            filled = update_gates_module._fill_missing_old_logprobs(
                torch,
                Model(),
                None,
                rows,
                device="cpu",
                max_logprob_tokens=128,
                fill_token_logprobs=True,
            )
        finally:
            update_gates_module.response_logprob_tensor_details_from_text = original

        sample = rows[0]["samples"][0]
        self.assertEqual(filled, 1)
        self.assertAlmostEqual(sample["old_logprob"], -3.0, places=6)
        self.assertEqual(sample["old_logprobs"], [-1.0, -1.0, -1.0])
        self.assertEqual(sample["response_mask"], [1, 1, 1])
        self.assertAlmostEqual(sample["trajectory"][0]["old_logprob"], -1.0, places=6)
        self.assertAlmostEqual(sample["trajectory"][1]["old_logprob"], -2.0, places=6)

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

    def test_best_response_nll_gives_nonzero_gradient_for_success_sample(self):
        original = update_gates_module._sample_response_logprob_tensor
        logp = torch.tensor(0.5, requires_grad=True)

        def fake_logprob(*args, **kwargs):
            return logp

        update_gates_module._sample_response_logprob_tensor = fake_logprob
        try:
            stats = update_gates_module._backward_incremental_best_response_losses(
                torch,
                model=None,
                tokenizer=None,
                prompt_text="prompt",
                valid_samples=[
                    {"sample_id": "ok", "text": "good", "reward_train": 1.0, "length": 4},
                    {"sample_id": "bad", "text": "bad", "reward_train": 0.0, "length": 4},
                ],
                task="tool",
                device="cpu",
                max_logprob_tokens=32,
                task_weight=1.0,
                best_response_loss_weight=0.2,
                pairwise_loss_weight=0.0,
                pairwise_margin=0.0,
                length_normalize=False,
                positive_reward_threshold=1.0,
                max_pairwise_pairs_per_row=0,
                loss_scale=1.0,
            )
        finally:
            update_gates_module._sample_response_logprob_tensor = original

        self.assertEqual(stats["processed"], 1.0)
        self.assertLess(float(logp.grad.item()), 0.0)


if __name__ == "__main__":
    unittest.main()
