import unittest

import torch

from opvec.train.gated_grpo import (
    clipped_grpo_sequence_loss,
    clipped_grpo_token_loss,
    normalize_rewards_to_advantages,
    reverse_kl_token_penalty,
    valid_policy_samples,
)


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

    def test_token_loss_matches_sequence_loss_for_one_token(self):
        current = torch.tensor([-0.2])
        old = torch.tensor([-0.3])
        advantage = torch.tensor(1.5)
        token_loss = clipped_grpo_token_loss(
            torch,
            current_logprobs=current,
            old_logprobs=old,
            response_mask=torch.tensor([1]),
            advantage=advantage,
            clip_epsilon=0.2,
        )
        sequence_loss = clipped_grpo_sequence_loss(
            torch,
            current_logp=current.sum(),
            old_logp=old.sum(),
            advantage=advantage,
            clip_epsilon=0.2,
        )

        self.assertAlmostEqual(float(token_loss.item()), float(sequence_loss.item()), places=6)

    def test_token_loss_ignores_masked_tokens(self):
        loss = clipped_grpo_token_loss(
            torch,
            current_logprobs=torch.tensor([-0.2, -10.0]),
            old_logprobs=torch.tensor([-0.3, -0.1]),
            response_mask=torch.tensor([1, 0]),
            advantage=torch.tensor(1.0),
            clip_epsilon=0.2,
        )
        expected = clipped_grpo_token_loss(
            torch,
            current_logprobs=torch.tensor([-0.2]),
            old_logprobs=torch.tensor([-0.3]),
            response_mask=torch.tensor([1]),
            advantage=torch.tensor(1.0),
            clip_epsilon=0.2,
        )

        self.assertAlmostEqual(float(loss.item()), float(expected.item()), places=6)

    def test_token_reverse_kl_is_zero_when_logprobs_match(self):
        penalty = reverse_kl_token_penalty(
            torch,
            current_logprobs=torch.tensor([-0.1, -0.2]),
            old_logprobs=torch.tensor([-0.1, -0.2]),
            response_mask=torch.tensor([1, 1]),
        )

        self.assertAlmostEqual(float(penalty.item()), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
