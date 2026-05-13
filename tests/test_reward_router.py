import unittest

from opvec.rewards.router import RewardRouter


class RewardRouterBatchTest(unittest.TestCase):
    def test_batch_score_matches_single_score(self):
        router = RewardRouter()
        tool_reference = (
            "<think> use tool </think>\n"
            "<tool_call>\n"
            '{"name": "lookup", "parameters": {"x": 1}}\n'
            "</tool_call>"
        )
        records = [
            {"task": "tool", "reference": {"response": tool_reference}},
            {"task": "memory", "reference": {"answer": ["Bob"], "metadata": {"round_type": "final"}}},
        ]
        outputs = [tool_reference, "The answer is \\boxed{Bob}."]

        batch = router.batch_score(records, outputs)
        singles = [router.score(record, output) for record, output in zip(records, outputs)]

        self.assertEqual(batch, singles)

    def test_batch_score_broadcasts_one_prompt_record(self):
        router = RewardRouter()
        record = {"task": "memory", "reference": {"answer": ["Bob"], "metadata": {"round_type": "final"}}}

        scores = router.batch_score(record, ["\\boxed{Bob}", "\\boxed{Alice}"])

        self.assertEqual(len(scores), 2)
        self.assertEqual(scores[0]["reward"], 1.0)
        self.assertEqual(scores[1]["reward"], 0.0)

    def test_batch_score_rejects_mismatched_lengths(self):
        router = RewardRouter()

        with self.assertRaisesRegex(ValueError, "length mismatch"):
            router.batch_score([{"task": "memory", "reference": {"answer": ["Bob"]}}], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
