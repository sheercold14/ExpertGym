#!/usr/bin/env python3
"""Build BFCL tool-call calibration prompts and official-answer OPD anchors."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from opvec.rewards.bfcl import BFCLToolRewardAdapter


DEFAULT_BFCL_ROOT = Path("/mnt/cache/wuruixiao/users/lsc/gorilla/berkeley-function-call-leaderboard")
DEFAULT_BASE_MANIFEST = Path("/tmp/shared-storage/OnPolicy/data/calibration/qbank_c033333_paper96_balanced32_seed20260514.prompts.jsonl")
DEFAULT_OUTPUT_DIR = Path("/tmp/shared-storage/OnPolicy/data/calibration/20260519_l4_bfcl_tool_aug")
DEFAULT_HARD_RESULT_ROOT = DEFAULT_BFCL_ROOT / (
    "result/expertgym-p1-main-gc-i7-tool-expertgym-p1-main-gc-i7-eval6-tool-20260518-"
    "expertgym_p1_main_gc_i7_eval6_tool_20260518/"
    "expertgym-p1-main-gc-i7-tool-expertgym-p1-main-gc-i7-eval6-tool-20260518"
)

CALIBRATION_CATEGORIES = {
    "parallel": {
        "data": "BFCL_v4_parallel.json",
        "answer": "possible_answer/BFCL_v4_parallel.json",
        "eval_group": "non_live",
        "quota": 4,
    },
    "parallel_multiple": {
        "data": "BFCL_v4_parallel_multiple.json",
        "answer": "possible_answer/BFCL_v4_parallel_multiple.json",
        "eval_group": "non_live",
        "quota": 4,
    },
    "live_parallel": {
        "data": "BFCL_v4_live_parallel.json",
        "answer": "possible_answer/BFCL_v4_live_parallel.json",
        "eval_group": "live",
        "quota": 4,
    },
    "live_parallel_multiple": {
        "data": "BFCL_v4_live_parallel_multiple.json",
        "answer": "possible_answer/BFCL_v4_live_parallel_multiple.json",
        "eval_group": "live",
        "quota": 4,
    },
}


def main() -> None:
    args = parse_args()
    created_at = datetime.now(timezone.utc).isoformat()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_rows, expert_rows, selection_summary = build_bfcl_rows(args, created_at=created_at)
    base_rows = read_jsonl(Path(args.base_manifest))
    merged_rows = base_rows + selected_rows

    tool_output = output_dir / "bfcl_tool16_nonlive8_live8_seed20260519.prompts.jsonl"
    expert_output = output_dir / "bfcl_tool16_official_answer_expert_rollouts_seed20260519.jsonl"
    merged_output = output_dir / "qbank_c033333_paper96_plus_bfcl_tool16_seed20260519.prompts.jsonl"
    summary_output = output_dir / "qbank_c033333_paper96_plus_bfcl_tool16_seed20260519.summary.json"

    write_jsonl(tool_output, selected_rows)
    write_jsonl(expert_output, expert_rows)
    write_jsonl(merged_output, merged_rows)
    summary = {
        "format": "bfcl_tool_calibration_v1",
        "created_at": created_at,
        "base_manifest": str(Path(args.base_manifest)),
        "base_rows": len(base_rows),
        "tool_output": str(tool_output),
        "expert_rollout_output": str(expert_output),
        "merged_output": str(merged_output),
        "merged_rows": len(merged_rows),
        "selected_rows": len(selected_rows),
        "selected_task_counts": dict(Counter(row.get("task") for row in selected_rows)),
        "selected_eval_groups": dict(Counter(row["reference"]["bfcl"]["eval_group"] for row in selected_rows)),
        "selected_categories": dict(Counter(row["reference"]["bfcl"]["category"] for row in selected_rows)),
        "hard_live_rows": sum(1 for row in selected_rows if row["reference"]["bfcl"].get("hard_selection")),
        "selection_summary": selection_summary,
        "leakage_policy": (
            "This calibration set intentionally uses official BFCL evaluation prompts and possible answers "
            "as requested for tool-distribution coverage; all source ids and paths are recorded."
        ),
    }
    summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def build_bfcl_rows(args: argparse.Namespace, *, created_at: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    bfcl_root = Path(args.bfcl_root)
    data_root = bfcl_root / "bfcl_eval" / "data"
    hard_failures = load_hard_failures(Path(args.hard_result_root), data_root)
    adapter = BFCLToolRewardAdapter()

    selected: list[dict[str, Any]] = []
    expert_rows: list[dict[str, Any]] = []
    per_category_summary: dict[str, Any] = {}
    for category, spec in CALIBRATION_CATEGORIES.items():
        examples = load_bfcl_examples(data_root, category, spec)
        failure_ids = hard_failures.get(category, set())
        ordered = sorted(
            examples,
            key=lambda item: (
                0 if item["id"] in failure_ids else 1,
                _stable_hash([category, item["id"], args.seed]),
            ),
        )
        chosen = ordered[: int(spec["quota"])]
        per_category_summary[category] = {
            "available": len(examples),
            "hard_failure_ids_available": len(failure_ids),
            "selected": [item["id"] for item in chosen],
            "selected_hard_failures": [item["id"] for item in chosen if item["id"] in failure_ids],
        }
        for index, item in enumerate(chosen):
            hard_selection = item["id"] in failure_ids and category.startswith("live_")
            row, expert = make_prompt_and_expert_row(
                item,
                category=category,
                eval_group=str(spec["eval_group"]),
                source_data_path=str(data_root / spec["data"]),
                source_answer_path=str(data_root / spec["answer"]),
                created_at=created_at,
                hard_selection=hard_selection,
                selection_rank=index,
            )
            score = adapter.score(row, expert["samples"][0]["text"]).as_dict()
            if not score.get("success"):
                raise RuntimeError(
                    f"Canonical BFCL response failed verifier for {row['prompt_id']}: {score}"
                )
            expert["samples"][0].update(
                {
                    "reward": float(score["reward"]),
                    "task_reward": float(score["task_reward"]),
                    "contract_reward": float(score.get("contract_reward", 0.0)),
                    "reward_train": float(score["reward"]),
                    "success": bool(score["success"]),
                    "details": score.get("details", {}),
                }
            )
            selected.append(row)
            expert_rows.append(expert)

    live_hard_count = sum(
        1
        for row in selected
        if row["reference"]["bfcl"]["eval_group"] == "live" and row["reference"]["bfcl"].get("hard_selection")
    )
    if live_hard_count < int(args.min_live_hard):
        raise RuntimeError(f"Selected only {live_hard_count} live hard rows; need >= {args.min_live_hard}")

    return selected, expert_rows, {
        "hard_result_root": str(Path(args.hard_result_root)),
        "min_live_hard": int(args.min_live_hard),
        "live_hard_selected": live_hard_count,
        "per_category": per_category_summary,
    }


def load_bfcl_examples(data_root: Path, category: str, spec: dict[str, Any]) -> list[dict[str, Any]]:
    examples = {row["id"]: row for row in read_jsonl(data_root / spec["data"])}
    answers = {row["id"]: row["ground_truth"] for row in read_jsonl(data_root / spec["answer"])}
    merged = []
    for row_id, row in examples.items():
        if row_id not in answers:
            continue
        item = dict(row)
        item["category"] = category
        item["possible_answer"] = answers[row_id]
        merged.append(item)
    return merged


def load_hard_failures(result_root: Path, data_root: Path) -> dict[str, set[str]]:
    failures: dict[str, set[str]] = defaultdict(set)
    adapter = BFCLToolRewardAdapter()
    for category, spec in CALIBRATION_CATEGORIES.items():
        result_path = result_root / spec["eval_group"] / f"BFCL_v4_{category}_result.json"
        if not result_path.exists():
            continue
        examples = {item["id"]: item for item in load_bfcl_examples(data_root, category, spec)}
        for result in read_jsonl(result_path):
            row_id = str(result.get("id") or "")
            if row_id not in examples:
                continue
            prompt_row = make_prompt_record(
                examples[row_id],
                category=category,
                eval_group=str(spec["eval_group"]),
                source_data_path=str(data_root / spec["data"]),
                source_answer_path=str(data_root / spec["answer"]),
                created_at="hard-scan",
                hard_selection=False,
                selection_rank=-1,
            )
            score = adapter.score(prompt_row, str(result.get("result") or "")).as_dict()
            if not bool(score.get("success")):
                failures[category].add(row_id)
    return failures


def make_prompt_and_expert_row(
    item: dict[str, Any],
    *,
    category: str,
    eval_group: str,
    source_data_path: str,
    source_answer_path: str,
    created_at: str,
    hard_selection: bool,
    selection_rank: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    row = make_prompt_record(
        item,
        category=category,
        eval_group=eval_group,
        source_data_path=source_data_path,
        source_answer_path=source_answer_path,
        created_at=created_at,
        hard_selection=hard_selection,
        selection_rank=selection_rank,
    )
    response = canonical_bfcl_response(item["possible_answer"])
    expert = {
        "run_id": "bfcl_official_answer_expert_rollout",
        "created_at": created_at,
        "policy_id": "bfcl_official_possible_answer",
        "gate_checkpoint": None,
        "gate_values": {},
        "gate_id": "bfcl_official_possible_answer",
        "group_id": row["group_id"],
        "prompt_id": row["prompt_id"],
        "task": "tool",
        "prompt": row["prompt"],
        "reference": row["reference"],
        "rendered_prompt": row["prompt"],
        "samples": [
            {
                "sample_id": f"{row['prompt_id']}__official_pos0",
                "text": response,
                "old_logprob": None,
                "old_logprob_max_length": None,
                "length": len(response.split()),
                "opd_role": "positive",
                "opd_source": "bfcl_official_possible_answer",
            }
        ],
        "frontier": {
            "all_failure": False,
            "all_success": True,
            "num_success": 1,
            "num_failure": 0,
            "mean_reward": 1.0,
            "std_reward": 0.0,
            "reward_field": "reward_train",
        },
        "keep_for_policy_loss": False,
        "skip_reason": "official_answer_anchor_only",
    }
    return row, expert


def make_prompt_record(
    item: dict[str, Any],
    *,
    category: str,
    eval_group: str,
    source_data_path: str,
    source_answer_path: str,
    created_at: str,
    hard_selection: bool,
    selection_rank: int,
) -> dict[str, Any]:
    prompt = bfcl_prompt(item["function"], _question_text(item["question"]))
    prompt_hash = hashlib.sha256(f"{category}:{item['id']}:{prompt}".encode("utf-8")).hexdigest()
    prompt_id = f"tool__bfclv4_{_safe_id(category)}_{_safe_id(item['id'])}_{prompt_hash[:8]}"
    return {
        "task": "tool",
        "prompt_id": prompt_id,
        "group_id": prompt_id,
        "prompt_hash": prompt_hash,
        "prompt": prompt,
        "messages": [],
        "reference": {
            "answer": None,
            "response": canonical_bfcl_response(item["possible_answer"]),
            "bfcl": {
                "id": item["id"],
                "category": category,
                "eval_group": eval_group,
                "function": item["function"],
                "possible_answer": item["possible_answer"],
                "model_name": "opvec-bfcl-offline",
                "has_tool_call_tag": False,
                "source_data_path": source_data_path,
                "source_answer_path": source_answer_path,
                "hard_selection": bool(hard_selection),
            },
            "metadata": {
                "source_dataset": "BFCL_v4_eval",
                "source_id": item["id"],
                "source_category": category,
                "source_eval_group": eval_group,
                "has_tool_call": True,
            },
        },
        "source": source_data_path,
        "source_row": item["id"],
        "split": "bfcl_v4_eval_calibration",
        "verifier": {"name": "bfcl_ast", "config": {"source": "BFCL_v4_eval"}},
        "tags": [
            "tool_call",
            "bfcl",
            "bfcl_v4",
            f"bfcl_category:{category}",
            f"bfcl_group:{eval_group}",
            "hard_live" if hard_selection else "coverage",
        ],
        "difficulty": "hard_live" if hard_selection else "coverage",
        "bfcl_tool_calibration": {
            "created_at": created_at,
            "selection_rank": int(selection_rank),
            "selection_policy": "prefer model failures for live hard rows; balanced BFCL live/non-live coverage",
        },
    }


def bfcl_prompt(functions: list[dict[str, Any]], user_text: str) -> str:
    function_json = json.dumps(functions, ensure_ascii=False, indent=4, sort_keys=True)
    system = (
        "You are an expert in composing functions.You are given a question and a set of possible functions. "
        "Based on the question, you will need to make one or more function/tool calls to achieve the purpose. "
        "If none of the functions can be used, point it out. If the given question lacks the parameters required "
        "by the function, also point it out.\n\n"
        "You should only return the function calls in your response.\n\n"
        "If you decide to invoke any of the function(s), you MUST put it in the format of "
        "[func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)].  "
        "You SHOULD NOT include any other text in the response.\n\n"
        "At each turn, you should try your best to complete the tasks requested by the user within the current turn. "
        "Continue to output functions to call until you have fulfilled the user's request to the best of your ability. "
        "Once you have no more functions to call, the system will consider the current turn complete and proceed to "
        "the next turn or task.\n\n"
        "Here is a list of functions in json format that you can invoke.\n"
        f"{function_json}"
    )
    return f"<|im_start|>system\n{system}\n<|im_end|>\n<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"


def canonical_bfcl_response(possible_answer: list[dict[str, Any]]) -> str:
    calls = []
    for call in possible_answer:
        if not isinstance(call, dict) or not call:
            continue
        name, params = next(iter(call.items()))
        params = params if isinstance(params, dict) else {}
        args = ", ".join(f"{key}={_py_literal(_choose_allowed(value))}" for key, value in params.items())
        calls.append(f"{name}({args})")
    return "[" + ", ".join(calls) + "]"


def _choose_allowed(value: Any) -> Any:
    if isinstance(value, list):
        if not value:
            return ""
        if any(isinstance(item, list) for item in value):
            preferred = [item for item in value if isinstance(item, list) and item]
            chosen = preferred[0] if preferred else value[0]
            return [_choose_allowed(item) for item in chosen]
        if any(isinstance(item, dict) for item in value):
            preferred = [item for item in value if isinstance(item, dict) and item]
            chosen = preferred[0] if preferred else value[0]
            return _choose_allowed(chosen)
        preferred = [item for item in value if item not in ("", None)]
        return _choose_allowed(preferred[0] if preferred else value[0])
    if isinstance(value, dict):
        return {key: _choose_allowed(item) for key, item in value.items()}
    return value


def _py_literal(value: Any) -> str:
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{json.dumps(str(key), ensure_ascii=False)}: {_py_literal(item)}" for key, item in value.items()) + "}"
    if isinstance(value, list):
        return "[" + ", ".join(_py_literal(item) for item in value) + "]"
    return repr(value)


def _question_text(question: Any) -> str:
    if isinstance(question, list) and question:
        first_turn = question[0]
        if isinstance(first_turn, list) and first_turn:
            return str(first_turn[0].get("content", ""))
        if isinstance(first_turn, dict):
            return str(first_turn.get("content", ""))
    return str(question)


def _safe_id(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_").lower()


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bfcl-root", type=Path, default=DEFAULT_BFCL_ROOT)
    parser.add_argument("--base-manifest", type=Path, default=DEFAULT_BASE_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--hard-result-root", type=Path, default=DEFAULT_HARD_RESULT_ROOT)
    parser.add_argument("--min-live-hard", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260519)
    return parser.parse_args()


if __name__ == "__main__":
    main()
