import importlib.util
import unittest
from argparse import Namespace
from pathlib import Path

import torch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts/train/opvec_update_gates_from_rollouts.py"
SPEC = importlib.util.spec_from_file_location("opvec_update_gates_from_rollouts", SCRIPT_PATH)
update_gates_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(update_gates_module)


class PCGradGateGradientsTest(unittest.TestCase):
    def test_disabled_update_batcher_uses_existing_gradient(self):
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
            update_batch_size=1,
            batch_loss_reduction="sum",
            optimizer_step_scope="epoch",
            loss_normalizer=1,
            train_coefficients=set(),
            coefficient_anchor_gates={},
            args=Namespace(
                pcgrad_gate_gradients=False,
                max_coefficient_delta_from_init=None,
                max_coefficient_delta_from_init_by_expert=[],
                coefficient_bound_by_expert=[],
                tool_min_margin_over_memory=0.0,
                tool_min_margin_over_code=0.0,
            ),
        )
        batcher.pcgrad_recompute_fn = lambda: (_ for _ in ()).throw(AssertionError("PCGrad should be disabled"))
        log_rows = [{"prompt_id": "toy"}]
        (manager.weight * 2.0).backward()
        batcher.add(log_rows, 0)
        batcher.flush(log_rows, force=True)

        self.assertAlmostEqual(float(manager.weight.detach().item()), -1.0, places=6)
        self.assertNotIn("pcgrad_enabled", log_rows[0])

    def test_pcgrad_projects_conflicting_gradients(self):
        task_grads = {
            "tool": torch.tensor([1.0, 0.0]),
            "memory": torch.tensor([-1.0, 0.0]),
        }
        projected, stats = update_gates_module._pcgrad_project(task_grads, eps=1e-12)

        self.assertGreater(stats["conflict_count"], 0)
        self.assertAlmostEqual(float(torch.dot(projected["tool"], task_grads["memory"]).item()), 0.0, places=6)
        self.assertAlmostEqual(float(torch.dot(projected["memory"], task_grads["tool"]).item()), 0.0, places=6)

    def test_pcgrad_keeps_non_conflicting_gradients_unchanged(self):
        task_grads = {
            "tool": torch.tensor([1.0, 0.0]),
            "memory": torch.tensor([1.0, 0.0]),
        }
        projected, stats = update_gates_module._pcgrad_project(task_grads, eps=1e-12)

        self.assertEqual(stats["conflict_count"], 0)
        self.assertTrue(torch.allclose(projected["tool"], task_grads["tool"]))
        self.assertTrue(torch.allclose(projected["memory"], task_grads["memory"]))

    def test_regularizer_is_added_after_projection(self):
        task_grads = {
            "tool": torch.tensor([1.0, 0.0]),
            "memory": torch.tensor([-1.0, 0.0]),
        }
        regularizer_grad = torch.tensor([0.0, 1.0])

        final_grad, projected, _stats = update_gates_module._combine_pcgrad_task_and_regularizer_grads(
            task_grads,
            regularizer_grad,
            eps=1e-12,
        )

        expected = projected["tool"] + projected["memory"] + regularizer_grad
        self.assertTrue(torch.allclose(final_grad, expected, atol=1e-6))
        self.assertTrue(torch.allclose(final_grad, torch.tensor([0.0, 1.0]), atol=1e-6))

    def test_write_flat_grad_overwrites_existing_gradients(self):
        param = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        param.grad = torch.tensor([100.0, 100.0])

        update_gates_module._write_flat_grad_to_gate_params_(torch.tensor([0.25, -0.5]), [param])

        self.assertTrue(torch.allclose(param.grad, torch.tensor([0.25, -0.5])))
        param.grad.zero_()
        self.assertTrue(torch.allclose(param.grad, torch.zeros_like(param)))


class ToolNullspaceGateGradientsTest(unittest.TestCase):
    def test_projection_removes_protected_direction(self):
        total_grad = torch.tensor([3.0, 4.0])
        basis_grads = [torch.tensor([1.0, 0.0]), torch.tensor([2.0, 0.0])]

        projected, stats = update_gates_module._project_flat_grad_away_from_basis_grads(
            torch,
            total_grad,
            basis_grads,
            eps=1e-12,
            max_rank=0,
        )

        self.assertTrue(torch.allclose(projected, torch.tensor([0.0, 4.0]), atol=1e-6))
        self.assertEqual(stats["rank"], 1)
        self.assertAlmostEqual(stats["grad_norm_before"], 5.0, places=6)
        self.assertAlmostEqual(stats["grad_norm_after"], 4.0, places=6)
        self.assertAlmostEqual(stats["removed_norm"], 3.0, places=6)

    def test_projection_respects_rank_limit(self):
        total_grad = torch.tensor([3.0, 4.0])
        basis_grads = [torch.tensor([2.0, 0.0]), torch.tensor([0.0, 1.0])]

        projected, stats = update_gates_module._project_flat_grad_away_from_basis_grads(
            torch,
            total_grad,
            basis_grads,
            eps=1e-12,
            max_rank=1,
        )

        self.assertTrue(torch.allclose(projected, torch.tensor([0.0, 4.0]), atol=1e-6))
        self.assertEqual(stats["rank"], 1)

    def test_tool_behavior_char_spans_prefer_tool_call_markup(self):
        text = 'prefix <tool_call>{"name":"x"}</tool_call> suffix'
        spans = update_gates_module._tool_behavior_char_spans(text)

        self.assertEqual(len(spans), 1)
        protected = text[spans[0][0] : spans[0][1]]
        self.assertTrue(protected.startswith("<tool_call>"))
        self.assertTrue(protected.endswith("</tool_call>"))

    def test_tool_behavior_char_spans_cover_bfcl_bracket_call(self):
        text = "[get_weather(location='Boston')]"
        spans = update_gates_module._tool_behavior_char_spans(text)

        self.assertEqual(spans, [(0, len(text))])


if __name__ == "__main__":
    unittest.main()
