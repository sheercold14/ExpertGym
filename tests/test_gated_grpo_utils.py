import unittest

from opvec.train.gated_grpo import normalize_rewards_to_advantages, valid_policy_samples


class GatedGrpoUtilsTest(unittest.TestCase):
    def test_advantages_are_group_relative(self):
        advantages, stats = normalize_rewards_to_advantages([0.0, 1.0, 1.0], frontier_weight=0.5)
        self.assertEqual(stats.num_samples, 3)
        self.assertGreater(stats.std_reward, 0.0)
        self.assertAlmostEqual(sum(advantages), 0.0, places=5)
        self.assertGreater(advantages[1], advantages[0])
        self.assertGreater(advantages[2], advantages[0])

    def test_zero_variance_group_has_zero_advantage(self):
        advantages, stats = normalize_rewards_to_advantages([1.0, 1.0, 1.0])
        self.assertEqual(advantages, [0.0, 0.0, 0.0])
        self.assertEqual(stats.nonzero_advantages, 0)

    def test_valid_policy_samples_require_text_and_old_logprob(self):
        samples = [
            {"text": "ok", "old_logprob": -1.0},
            {"text": "", "old_logprob": -1.0},
            {"text": "no logp", "old_logprob": None},
        ]
        self.assertEqual(len(valid_policy_samples(samples)), 1)
        self.assertEqual(len(valid_policy_samples(samples, require_old_logprob=False)), 2)


if __name__ == "__main__":
    unittest.main()
