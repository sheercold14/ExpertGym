import importlib.util
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts/train/opvec_collect_hf_rollouts.py"
SPEC = importlib.util.spec_from_file_location("opvec_collect_hf_rollouts", SCRIPT_PATH)
collect_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(collect_module)


class CollectHFRolloutsRewardTest(unittest.TestCase):
    def test_behavior_span_mix_keeps_correctness_dominant(self):
        scored = {
            "reward": 0.2,
            "task_reward": 0.2,
            "contract_reward": 0.1,
            "details": {"token_f1": 1.0},
            "success": False,
        }
        mixed = collect_module._apply_behavior_span_mix(scored, behavior_span_weight=0.05)

        self.assertAlmostEqual(mixed["reward"], 0.24, places=6)
        self.assertEqual(mixed["task_reward"], 0.2)
        self.assertEqual(mixed["behavior_span_reward"], 1.0)
        self.assertEqual(mixed["details"]["behavior_span_weight"], 0.05)


if __name__ == "__main__":
    unittest.main()
