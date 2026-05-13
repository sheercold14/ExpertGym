import importlib.util
import unittest
from dataclasses import dataclass
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts/train/opvec_collect_vllm_rollouts.py"
SPEC = importlib.util.spec_from_file_location("opvec_collect_vllm_rollouts", SCRIPT_PATH)
collect_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(collect_module)


@dataclass
class FakeLogprob:
    logprob: float


class FakeCompletion:
    text = "hello"
    token_ids = [10, 11]
    logprobs = [{10: FakeLogprob(-0.1)}, {11: FakeLogprob(-0.2)}]


class VllmRolloutLogprobsTest(unittest.TestCase):
    def test_vllm_completion_payload_stores_sampled_token_logprobs(self):
        payload = collect_module._vllm_completion_payload(FakeCompletion(), store_token_logprobs=True)

        self.assertEqual(payload["text"], "hello")
        self.assertEqual(payload["response_token_ids"], [10, 11])
        self.assertEqual(payload["old_logprobs"], [-0.1, -0.2])
        self.assertEqual(payload["response_mask"], [1, 1])
        self.assertAlmostEqual(payload["old_logprob"], -0.3, places=6)

    def test_aggregate_trajectory_token_payload(self):
        payload = collect_module._aggregate_trajectory_token_payload(
            [
                {"response_token_ids": [1], "old_logprobs": [-0.5], "response_mask": [1]},
                {"response_token_ids": [2, 3], "old_logprobs": [-0.2, -0.3], "response_mask": [1, 1]},
            ]
        )

        self.assertEqual(payload["response_token_ids"], [1, 2, 3])
        self.assertEqual(payload["old_logprobs"], [-0.5, -0.2, -0.3])
        self.assertEqual(payload["response_mask"], [1, 1, 1])
        self.assertAlmostEqual(payload["old_logprob"], -1.0, places=6)


if __name__ == "__main__":
    unittest.main()
