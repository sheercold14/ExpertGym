import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opvec.rewards.simple import MemoryRewardAdapter, ToolRewardAdapter


COLLECT_PATH = Path(__file__).resolve().parents[1] / "scripts/train/opvec_collect_hf_rollouts.py"
COLLECT_SPEC = importlib.util.spec_from_file_location("opvec_collect_hf_rollouts", COLLECT_PATH)
collect_module = importlib.util.module_from_spec(COLLECT_SPEC)
assert COLLECT_SPEC.loader is not None
COLLECT_SPEC.loader.exec_module(collect_module)

BUILD_PATH = Path(__file__).resolve().parents[1] / "scripts/data/build_routed_correct_seed_manifest.py"
BUILD_SPEC = importlib.util.spec_from_file_location("build_routed_correct_seed_manifest", BUILD_PATH)
build_module = importlib.util.module_from_spec(BUILD_SPEC)
assert BUILD_SPEC.loader is not None
BUILD_SPEC.loader.exec_module(build_module)


class OfficialRewardAlignmentTest(unittest.TestCase):
    def test_toolrl_score_uses_official_raw_range(self):
        reference = (
            "<think> use tool </think>\n"
            "<tool_call>\n"
            '{"name": "lookup", "parameters": {"x": 1}}\n'
            "</tool_call>"
        )
        record = {"task": "tool", "reference": {"response": reference}}
        result = ToolRewardAdapter().score(record, reference).as_dict()

        self.assertEqual(result["reward"], 4.0)
        self.assertEqual(result["task_reward"], 4.0)
        self.assertTrue(result["success"])
        self.assertEqual(result["details"]["toolrl_score_range"], [-3.0, 4.0])

    def test_toolrl_correctness_uses_first_tool_call_block_like_official(self):
        reference = (
            "<think> use tool </think>\n"
            "<tool_call>\n"
            '{"name": "lookup", "parameters": {"x": 1}}\n'
            "</tool_call>"
        )
        response = (
            reference
            + "\n<tool_call>\n"
            + '{"name": "extra", "parameters": {}}\n'
            + "</tool_call>"
        )
        record = {"task": "tool", "reference": {"response": reference}}
        result = ToolRewardAdapter().score(record, response).as_dict()

        self.assertEqual(result["details"]["format_score"], 0.0)
        self.assertEqual(result["details"]["toolrl_correctness_raw"], 3.0)
        self.assertEqual(result["reward"], 3.0)

    def test_toolrl_invalid_json_line_gets_min_correctness_like_official(self):
        reference = (
            "<think> use tool </think>\n"
            "<tool_call>\n"
            '{"name": "lookup", "parameters": {"x": 1}}\n'
            "</tool_call>"
        )
        response = (
            "<think> use tool </think>\n"
            "<tool_call>\n"
            '{"name": "lookup", "parameters": {"x": 1}}\n'
            "not json\n"
            "</tool_call>"
        )
        record = {"task": "tool", "reference": {"response": reference}}
        result = ToolRewardAdapter().score(record, response).as_dict()

        self.assertEqual(result["details"]["format_score"], 1.0)
        self.assertEqual(result["details"]["toolrl_correctness_raw"], -3.0)
        self.assertEqual(result["reward"], -2.0)

    def test_toolrl_missing_parameters_gets_min_correctness_like_official(self):
        reference = (
            "<think> use tool </think>\n"
            "<tool_call>\n"
            '{"name": "lookup", "parameters": {"x": 1}}\n'
            "</tool_call>"
        )
        response = (
            "<think> use tool </think>\n"
            "<tool_call>\n"
            '{"name": "lookup"}\n'
            "</tool_call>"
        )
        record = {"task": "tool", "reference": {"response": reference}}
        result = ToolRewardAdapter().score(record, response).as_dict()

        self.assertEqual(result["details"]["format_score"], 1.0)
        self.assertEqual(result["details"]["toolrl_correctness_raw"], -3.0)
        self.assertEqual(result["reward"], -2.0)

    def test_memagent_final_answer_matches_official_boxed_exact(self):
        record = {"task": "memory", "reference": {"answer": ["President Richard Nixon"], "metadata": {"round_type": "final"}}}
        result = MemoryRewardAdapter().score(record, "The answer is \\boxed{President Richard Nixon}.").as_dict()

        self.assertEqual(result["reward"], 1.0)
        self.assertTrue(result["success"])
        self.assertEqual(result["details"]["reward_source"], "MemAgent/verl/utils/reward_score/hotpotqa.py")

    def test_behavior_mix_zero_preserves_raw_official_reward(self):
        scored = {"reward": 4.0, "task_reward": 4.0, "contract_reward": 0.1, "details": {}, "success": True}
        mixed = collect_module._apply_behavior_span_mix(scored, behavior_span_weight=0.0)

        self.assertEqual(mixed["reward"], 4.0)
        self.assertEqual(mixed["task_reward"], 4.0)
        self.assertEqual(mixed["details"]["behavior_span_weight"], 0.0)

    def test_build_routed_manifest_groups_memory_chunks_into_trajectory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "ToolCall.json").write_text("[]", encoding="utf-8")
            (root / "Code.json").write_text("[]", encoding="utf-8")
            memory_payload = [
                {
                    "question_id": 7,
                    "round_type": "chunk",
                    "round_idx": 0,
                    "messages": [
                        {
                            "role": "user",
                            "content": "<problem>Who?</problem><memory>No previous memory</memory><section>Alice knows Bob.</section>",
                        }
                    ],
                    "response": "Updated memory: Alice knows Bob.",
                },
                {
                    "question_id": 7,
                    "round_type": "final",
                    "messages": [{"role": "user", "content": "<problem>Who?</problem><memory>Alice knows Bob.</memory>"}],
                    "response": "\\boxed{Bob}",
                },
            ]
            (root / "Memory.json").write_text(json.dumps(memory_payload), encoding="utf-8")
            args = type(
                "Args",
                (),
                {
                    "input_root": str(root),
                    "tool_limit": 0,
                    "code_limit": 0,
                    "memory_limit": -1,
                    "split": "test",
                    "dry_run": True,
                    "output": str(root / "out.jsonl"),
                },
            )()

            rows, summary = build_module.build_manifest(args)

        self.assertEqual(summary["task_counts"], {"memory": 1})
        self.assertEqual(rows[0]["reference"]["metadata"]["memagent_chunks"], ["Alice knows Bob."])
        self.assertEqual(rows[0]["reference"]["answer"], ["Bob"])


if __name__ == "__main__":
    unittest.main()
