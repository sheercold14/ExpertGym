import unittest

from opvec.data.schema import make_gate_id, validate_rollout_row, validate_token_level_sample


class RolloutSchemaTest(unittest.TestCase):
    def test_validate_current_rollout_row(self):
        row = {
            "run_id": "run",
            "step": 1,
            "policy_id": "gate_t",
            "prompt_id": "tool__abc",
            "task": "tool",
            "samples": [
                {
                    "sample_id": "tool__abc__k0",
                    "text": "answer",
                    "reward": 1.0,
                    "task_reward": 1.0,
                }
            ],
        }

        validate_rollout_row(row)

    def test_validate_token_level_lengths(self):
        sample = {
            "response_token_ids": [1, 2],
            "old_logprobs": [-0.1, -0.2],
            "response_mask": [1, 1],
        }

        validate_token_level_sample(sample)

        sample["response_mask"] = [1]
        with self.assertRaisesRegex(ValueError, "same length"):
            validate_token_level_sample(sample)

        sample["response_mask"] = [1, 0.5]
        with self.assertRaisesRegex(ValueError, "0/1"):
            validate_token_level_sample(sample)

    def test_validate_memory_trajectory_requires_turn_fields(self):
        row = {
            "run_id": "run",
            "step": 1,
            "policy_id": "gate_t",
            "prompt_id": "memory__abc",
            "task": "memory",
            "samples": [
                {
                    "sample_id": "memory__abc__k0",
                    "text": "final",
                    "reward": 1.0,
                    "task_reward": 1.0,
                    "trajectory": [{"turn": 1, "kind": "memory_update", "text": "updated"}],
                }
            ],
        }

        with self.assertRaisesRegex(ValueError, "prompt_text"):
            validate_rollout_row(row)

    def test_gate_id_is_stable_over_key_order(self):
        self.assertEqual(make_gate_id({"b": 2, "a": 1}), make_gate_id({"a": 1, "b": 2}))


if __name__ == "__main__":
    unittest.main()
