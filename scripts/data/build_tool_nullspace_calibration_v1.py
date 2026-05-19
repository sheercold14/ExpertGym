#!/usr/bin/env python3
"""Build Tool-nullspace v1 calibration and expert anchors.

Output layout:
- Tool: keep 16 original paper96 Tool rows, add 16 BFCL live rows with
  historical model-success trajectories.
- Memory: keep all 32 original paper96 Memory rows.
- Code: keep all 32 original paper96 Code rows, add 8 formal
  LiveBench/LiveCodeBench rows where TA-1/3 fails and ReasonFlux/R1 succeeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from opvec.data.io import read_jsonl, write_jsonl
from opvec.data.schema import validate_rollout_row, validate_seed_record
from opvec.rewards.bfcl import BFCLToolRewardAdapter
from opvec.rewards.simple import CodeRewardAdapter
from scripts.data.build_bfcl_tool_calibration import (
    CALIBRATION_CATEGORIES,
    bfcl_prompt,
    canonical_bfcl_response,
    load_bfcl_examples,
    make_prompt_record,
)
from scripts.data.build_cure_eval_code16_calibration import (
    EXPERTS as CURE_EXPERTS,
    cure_formal_prompt,
    _load_official_rows,
)


DEFAULT_BASE_MANIFEST = Path("/tmp/shared-storage/OnPolicy/data/calibration/qbank_c033333_paper96_balanced32_seed20260514.prompts.jsonl")
DEFAULT_OUTPUT_DIR = Path("/tmp/shared-storage/OnPolicy/data/calibration/20260519_tool_nullspace_v1")
DEFAULT_BFCL_ROOT = Path("/mnt/cache/wuruixiao/users/lsc/gorilla/berkeley-function-call-leaderboard")
DEFAULT_CURE_DATA_ROOT = Path("/mnt/cache/wuruixiao/users/lsc/CURE/data")
DEFAULT_CURE_TEMP_ROOT = Path("/tmp/shared-storage/AgentMerging_plan/evaluation_artifacts/cure_temp_backup_20260503")
DEFAULT_CURE_TA_TEMP_ROOT = Path("/tmp/shared-storage/AgentMerging_plan/evaluation_artifacts/cure_workspace/temp_data")
DEFAULT_TA13_BFCL_RESULT_ROOT = DEFAULT_BFCL_ROOT / "result/ta-c033333-global-20260517-tool-20260517-p0-ta13-eval6-20260517_p0_ta13_eval6/ta-c033333-global-20260517-tool-20260517-p0-ta13-eval6"
DEFAULT_TOOL_RESULT_ROOTS = [
    DEFAULT_BFCL_ROOT / "result/expertgym-p1-main-global-i3-tool-expertgym-p1-main-global-i3-eval6-tool-20260518-expertgym_p1_main_global_i3_eval6_tool_20260518/expertgym-p1-main-global-i3-tool-expertgym-p1-main-global-i3-eval6-tool-20260518",
    DEFAULT_BFCL_ROOT / "result/expertgym-p1-main-global-i7-tool-expertgym-p1-main-global-i7-eval6-tool-20260518-expertgym_p1_main_global_i7_eval6_tool_20260518/expertgym-p1-main-global-i7-tool-expertgym-p1-main-global-i7-eval6-tool-20260518",
    DEFAULT_BFCL_ROOT / "result/expertgym-p1-main-gc-i5-tool-expertgym-p1-main-gc-i5-eval6-tool-20260518-expertgym_p1_main_gc_i5_eval6_tool_20260518/expertgym-p1-main-gc-i5-tool-expertgym-p1-main-gc-i5-eval6-tool-20260518",
    DEFAULT_BFCL_ROOT / "result/expertgym-p1-main-gc-i7-tool-expertgym-p1-main-gc-i7-eval6-tool-20260518-expertgym_p1_main_gc_i7_eval6_tool_20260518/expertgym-p1-main-gc-i7-tool-expertgym-p1-main-gc-i7-eval6-tool-20260518",
]
DEFAULT_CODE_SELECTIONS = {
    "LiveBench": list(range(0, 64)),
    "LiveCodeBench": list(range(0, 64)),
}
TA13_CODE_MODEL_NAME = "tmp.shared-storage.OnPolicy.checkpoints.ta_c033333_global_20260517"


def main() -> None:
    args = parse_args()
    created_at = datetime.now(timezone.utc).isoformat()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_rows = read_jsonl(Path(args.base_manifest))
    for row in base_rows:
        validate_seed_record(row)
    base_by_task = _group_by_task(base_rows)
    tool_original = base_by_task["tool"][: int(args.keep_original_tool)]
    memory_rows = base_by_task["memory"]
    code_original = base_by_task["code"]

    bfcl_rows, bfcl_experts, bfcl_blueprints = _build_bfcl_live_rows(args, created_at=created_at)
    code_rows, code_experts, code_blueprints = _build_code_live_rows(args, created_at=created_at)

    merged_rows = _dedupe_seed_rows(tool_original + bfcl_rows + memory_rows + code_original + code_rows)
    extra_experts = _dedupe_rollout_rows(bfcl_experts + code_experts)
    _assert_target_layout(
        base_by_task=base_by_task,
        tool_original=tool_original,
        bfcl_rows=bfcl_rows,
        memory_rows=memory_rows,
        code_original=code_original,
        code_rows=code_rows,
        merged_rows=merged_rows,
        extra_experts=extra_experts,
    )

    prompts_out = output_dir / "tool32_memory32_code40_toolnullspace_seed20260519.prompts.jsonl"
    expert_out = output_dir / "toolnullspace_extra_expert_rollouts_seed20260519.jsonl"
    tool_replay_out = output_dir / "toolnullspace_tool_replay_rollouts_seed20260519.jsonl"
    blueprint_out = output_dir / "toolnullspace_selection_blueprints.jsonl"
    summary_out = output_dir / "toolnullspace_seed20260519.summary.json"
    readme_out = output_dir / "README.md"

    write_jsonl(prompts_out, merged_rows)
    write_jsonl(expert_out, extra_experts)
    write_jsonl(tool_replay_out, bfcl_experts)
    write_jsonl(blueprint_out, bfcl_blueprints + code_blueprints)

    summary = {
        "format": "tool_nullspace_calibration_v1",
        "created_at": created_at,
        "inputs": {
            "base_manifest": str(Path(args.base_manifest).expanduser().resolve()),
            "bfcl_root": str(Path(args.bfcl_root).expanduser().resolve()),
            "cure_data_root": str(Path(args.cure_data_root).expanduser().resolve()),
            "cure_temp_root": str(Path(args.cure_temp_root).expanduser().resolve()),
            "cure_ta_temp_root": str(Path(args.cure_ta_temp_root).expanduser().resolve()),
            "tool_ta13_result_root": str(Path(args.tool_ta13_result_root).expanduser().resolve()),
            "tool_historical_result_roots": [str(Path(item).expanduser().resolve()) for item in args.tool_result_root],
        },
        "outputs": {
            "prompts": str(prompts_out),
            "extra_expert_rollouts": str(expert_out),
            "tool_nullspace_replay_rollout": str(tool_replay_out),
            "blueprints": str(blueprint_out),
            "summary": str(summary_out),
            "readme": str(readme_out),
        },
        "counts": {
            "base_task_counts": dict(Counter(row["task"] for row in base_rows)),
            "kept_original_tool": len(tool_original),
            "kept_original_memory": len(memory_rows),
            "kept_original_code": len(code_original),
            "added_bfcl_tool_live": len(bfcl_rows),
            "added_code_live": len(code_rows),
            "merged_rows": len(merged_rows),
            "merged_task_counts": dict(Counter(row["task"] for row in merged_rows)),
            "extra_expert_rows": len(extra_experts),
            "extra_expert_task_counts": dict(Counter(row["task"] for row in extra_experts)),
            "extra_positive_samples": sum(len(row.get("samples") or []) for row in extra_experts),
        },
        "tool_selection": {
            "policy": "BFCL live prompts with historical model-success trajectories; prefer TA-1/3 failures when available.",
            "categories": dict(Counter(item["category"] for item in bfcl_blueprints)),
            "ta13_failure_count": sum(1 for item in bfcl_blueprints if not item.get("ta13_success")),
            "historical_policies": dict(Counter(item.get("selected_policy") for item in bfcl_blueprints)),
        },
        "code_selection": {
            "policy": "TA-1/3 fails, ReasonFlux preferred, R1 fallback; reward uses CodeRewardAdapter with first-8 official CURE tests.",
            "datasets": dict(Counter(item["dataset"] for item in code_blueprints)),
            "selected_experts": dict(Counter(item.get("selected_expert") for item in code_blueprints)),
        },
        "training_intent": {
            "tool": "Preserve behavior span via a fixed replay bank and dynamic all-success rows; Tool final count stays 32.",
            "memory": "Keep the existing paper96 memory 32 unchanged.",
            "code": "Keep paper96 code 32 and add 8 eval-aligned hard-vs-TA code anchors.",
        },
    }
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    readme_out.write_text(_render_readme(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def _build_bfcl_live_rows(args: argparse.Namespace, *, created_at: str):
    bfcl_root = Path(args.bfcl_root)
    data_root = bfcl_root / "bfcl_eval" / "data"
    adapter = BFCLToolRewardAdapter()
    candidates = []
    for category in ("live_parallel", "live_parallel_multiple"):
        spec = CALIBRATION_CATEGORIES[category]
        examples = {item["id"]: item for item in load_bfcl_examples(data_root, category, spec)}
        ta_success = _bfcl_success_by_id(
            adapter,
            examples,
            result_root=Path(args.tool_ta13_result_root),
            data_root=data_root,
            category=category,
            spec=spec,
        )
        historical = []
        for result_root in args.tool_result_root:
            historical.extend(
                _bfcl_success_rows(
                    adapter,
                    examples,
                    result_root=Path(result_root),
                    data_root=data_root,
                    category=category,
                    spec=spec,
                )
            )
        best_by_id: dict[str, dict[str, Any]] = {}
        for item in historical:
            previous = best_by_id.get(item["id"])
            if previous is None or item["policy_id"] < previous["policy_id"]:
                best_by_id[item["id"]] = item
        for row_id, success in best_by_id.items():
            candidates.append(
                {
                    **success,
                    "ta13_success": bool(ta_success.get(row_id, False)),
                    "priority": (
                        0 if not bool(ta_success.get(row_id, False)) else 1,
                        0 if category == "live_parallel_multiple" else 1,
                        _stable_int([category, row_id, args.seed]),
                    ),
                }
            )
    candidates.sort(key=lambda item: item["priority"])
    selected = _balanced_bfcl_live_selection(candidates, target=int(args.add_bfcl_tool))
    if len(selected) < int(args.add_bfcl_tool):
        raise RuntimeError(f"Only found {len(selected)} BFCL live historical successes; need {args.add_bfcl_tool}")

    rows = []
    expert_rows = []
    blueprints = []
    for rank, item in enumerate(selected):
        spec = CALIBRATION_CATEGORIES[item["category"]]
        prompt_row = make_prompt_record(
            item["example"],
            category=item["category"],
            eval_group=str(spec["eval_group"]),
            source_data_path=str(data_root / spec["data"]),
            source_answer_path=str(data_root / spec["answer"]),
            created_at=created_at,
            hard_selection=not bool(item["ta13_success"]),
            selection_rank=rank,
        )
        prompt_row["split"] = "tool_nullspace_v1_bfcl_live"
        prompt_row.setdefault("tags", []).append("tool_nullspace_v1")
        sample = {
            "sample_id": f"{prompt_row['prompt_id']}__{item['policy_id']}__histpos0",
            "text": item["text"],
            "old_logprob": None,
            "old_logprob_max_length": None,
            "length": len(str(item["text"]).split()),
            "opd_role": "positive",
            "opd_source": "bfcl_historical_model_success",
            "opd_source_policy_id": item["policy_id"],
            "reward": float(item["score"]["reward"]),
            "task_reward": float(item["score"]["task_reward"]),
            "contract_reward": float(item["score"].get("contract_reward", 0.0)),
            "reward_train": float(item["score"]["reward"]),
            "success": bool(item["score"]["success"]),
            "details": item["score"].get("details", {}),
        }
        expert_row = _rollout_row(
            run_id="toolnullspace_bfcl_historical_tool_success",
            policy_id=item["policy_id"],
            prompt_row=prompt_row,
            sample=sample,
            created_at=created_at,
        )
        validate_seed_record(prompt_row)
        validate_rollout_row(expert_row)
        rows.append(prompt_row)
        expert_rows.append(expert_row)
        blueprints.append(
            {
                "format": "tool_nullspace_bfcl_blueprint_v1",
                "task": "tool",
                "prompt_id": prompt_row["prompt_id"],
                "bfcl_id": item["id"],
                "category": item["category"],
                "ta13_success": bool(item["ta13_success"]),
                "selected_policy": item["policy_id"],
                "selection_rank": rank,
                "response_preview": str(item["text"])[:200],
            }
        )
    return rows, expert_rows, blueprints


def _assert_target_layout(
    *,
    base_by_task: dict[str, list[dict]],
    tool_original: list[dict],
    bfcl_rows: list[dict],
    memory_rows: list[dict],
    code_original: list[dict],
    code_rows: list[dict],
    merged_rows: list[dict],
    extra_experts: list[dict],
) -> None:
    expected_base = {"tool": 32, "memory": 32, "code": 32}
    observed_base = {task: len(base_by_task.get(task, [])) for task in expected_base}
    if observed_base != expected_base:
        raise RuntimeError(f"Expected paper96 base layout {expected_base}, got {observed_base}")
    expected_parts = {
        "tool_original": 16,
        "bfcl_rows": 16,
        "memory_rows": 32,
        "code_original": 32,
        "code_rows": 8,
    }
    observed_parts = {
        "tool_original": len(tool_original),
        "bfcl_rows": len(bfcl_rows),
        "memory_rows": len(memory_rows),
        "code_original": len(code_original),
        "code_rows": len(code_rows),
    }
    if observed_parts != expected_parts:
        raise RuntimeError(f"Unexpected calibration parts: expected {expected_parts}, got {observed_parts}")
    merged_counts = dict(Counter(row["task"] for row in merged_rows))
    expected_merged = {"tool": 32, "memory": 32, "code": 40}
    if merged_counts != expected_merged:
        raise RuntimeError(f"Unexpected merged task counts: expected {expected_merged}, got {merged_counts}")
    expert_counts = dict(Counter(row["task"] for row in extra_experts))
    expected_experts = {"tool": 16, "code": 8}
    if expert_counts != expected_experts:
        raise RuntimeError(f"Unexpected extra expert counts: expected {expected_experts}, got {expert_counts}")


def _balanced_bfcl_live_selection(candidates: list[dict[str, Any]], *, target: int) -> list[dict[str, Any]]:
    """Prefer an even split across BFCL live categories, then fill by global priority."""

    categories = ["live_parallel", "live_parallel_multiple"]
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        by_category[str(item.get("category") or "")].append(item)
    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[str, str]] = set()
    base_quota = int(target) // len(categories)
    remainder = int(target) % len(categories)
    for index, category in enumerate(categories):
        quota = base_quota + (1 if index < remainder else 0)
        for item in by_category.get(category, [])[:quota]:
            key = (str(item.get("category") or ""), str(item.get("id") or ""))
            selected.append(item)
            selected_keys.add(key)
    if len(selected) < int(target):
        for item in candidates:
            key = (str(item.get("category") or ""), str(item.get("id") or ""))
            if key in selected_keys:
                continue
            selected.append(item)
            selected_keys.add(key)
            if len(selected) >= int(target):
                break
    return selected[: int(target)]


def _bfcl_success_by_id(adapter, examples, *, result_root: Path, data_root: Path, category: str, spec: dict[str, Any]):
    return {item["id"]: True for item in _bfcl_success_rows(adapter, examples, result_root=result_root, data_root=data_root, category=category, spec=spec)}


def _bfcl_success_rows(adapter, examples, *, result_root: Path, data_root: Path, category: str, spec: dict[str, Any]):
    result_path = result_root / spec["eval_group"] / f"BFCL_v4_{category}_result.json"
    if not result_path.exists():
        return []
    output = []
    policy_id = _safe_policy_id(result_root.name)
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
            created_at="scan",
            hard_selection=False,
            selection_rank=-1,
        )
        text = str(result.get("result") or "")
        score = adapter.score(prompt_row, text).as_dict()
        if bool(score.get("success")):
            output.append(
                {
                    "id": row_id,
                    "category": category,
                    "example": examples[row_id],
                    "text": text,
                    "score": score,
                    "policy_id": policy_id,
                }
            )
    return output


def _build_code_live_rows(args: argparse.Namespace, *, created_at: str):
    selected_indices = DEFAULT_CODE_SELECTIONS
    official_rows = _load_official_rows(Path(args.cure_data_root), selected_indices)
    ta_status = {
        dataset: _load_cure_model_success_status(
            Path(args.cure_ta_temp_root),
            model_name=TA13_CODE_MODEL_NAME,
            dataset=dataset,
            indices=indices,
            max_tests=int(args.code_max_tests),
            max_positives=0,
        )
        for dataset, indices in selected_indices.items()
    }
    expert_status = {
        expert_name: {
            dataset: _load_cure_model_success_status(
                Path(args.cure_temp_root),
                model_name=model_name,
                dataset=dataset,
                indices=indices,
                max_tests=int(args.code_max_tests),
                max_positives=int(args.code_max_positives_per_expert),
            )
            for dataset, indices in selected_indices.items()
        }
        for expert_name, model_name in CURE_EXPERTS.items()
    }
    adapter = CodeRewardAdapter()
    candidates = []
    expert_priority = {"reasonflux": 0, "deepseek_r1_distill": 1, "memory_agent": 2}
    for dataset, by_index in official_rows.items():
        for source_row, raw in by_index.items():
            if int(ta_status[dataset].get(source_row, {}).get("success_count", 0)) > 0:
                continue
            positives = []
            for expert_name in ("reasonflux", "deepseek_r1_distill", "memory_agent"):
                samples = expert_status[expert_name][dataset].get(source_row, {}).get("success_samples", [])
                positives.extend((expert_name, sample) for sample in samples)
            if not positives:
                continue
            positives.sort(key=lambda item: (expert_priority.get(item[0], 99), item[1]["sample_index"]))
            selected_expert, selected_sample = positives[0]
            candidates.append(
                {
                    "dataset": dataset,
                    "source_row": source_row,
                    "raw": raw,
                    "expert": selected_expert,
                    "sample": selected_sample,
                    "priority": (
                        expert_priority.get(selected_expert, 99),
                        0 if dataset == "LiveCodeBench" else 1,
                        _stable_int([dataset, source_row, args.seed]),
                    ),
                }
            )
    candidates.sort(key=lambda item: item["priority"])
    selected = candidates[: int(args.add_code_live)]
    if len(selected) < int(args.add_code_live):
        raise RuntimeError(f"Only found {len(selected)} TA-fail expert-success code rows; need {args.add_code_live}")

    rows = []
    expert_rows = []
    blueprints = []
    for rank, item in enumerate(selected):
        prompt_row = _make_code_prompt_row(item["raw"], dataset=item["dataset"], source_row=item["source_row"], args=args, created_at=created_at)
        text = str(item["sample"]["text"] or "")
        score = adapter.score(prompt_row, text).as_dict()
        if float(score.get("reward", 0.0)) < float(args.code_positive_threshold):
            raise RuntimeError(f"Selected code sample failed CodeRewardAdapter: {prompt_row['prompt_id']} {score}")
        sample = {
            "sample_id": f"{prompt_row['prompt_id']}__{item['expert']}__k{item['sample']['sample_index']}",
            "text": text,
            "reward": float(score["reward"]),
            "task_reward": float(score["task_reward"]),
            "contract_reward": float(score.get("contract_reward", 0.0)),
            "reward_train": float(score["reward"]),
            "success": bool(score.get("success")),
            "old_logprob": None,
            "old_logprob_max_length": None,
            "length": len(text.split()),
            "opd_role": "positive",
            "opd_source": "cure_formal_eval_expert_success",
            "opd_source_policy_id": item["expert"],
            "details": {
                **(score.get("details") or {}),
                "expert_name": item["expert"],
                "expert_model": item["sample"].get("model_name"),
                "expert_temp_file": item["sample"].get("temp_file"),
                "expert_sample_index": item["sample"].get("sample_index"),
                "selection_rank": rank,
            },
        }
        expert_row = _rollout_row(
            run_id="toolnullspace_code_live_expert_success",
            policy_id=item["expert"],
            prompt_row=prompt_row,
            sample=sample,
            created_at=created_at,
        )
        validate_seed_record(prompt_row)
        validate_rollout_row(expert_row)
        rows.append(prompt_row)
        expert_rows.append(expert_row)
        blueprints.append(
            {
                "format": "tool_nullspace_code_blueprint_v1",
                "task": "code",
                "prompt_id": prompt_row["prompt_id"],
                "dataset": item["dataset"],
                "source_row": int(item["source_row"]),
                "ta13_success": False,
                "selected_expert": item["expert"],
                "selection_rank": rank,
                "question_preview": str(item["raw"].get("question") or "")[:220],
            }
        )
    return rows, expert_rows, blueprints


def _make_code_prompt_row(raw: dict[str, Any], *, dataset: str, source_row: int, args: argparse.Namespace, created_at: str) -> dict[str, Any]:
    question = str(raw.get("question") or "").strip()
    test_input = list(raw.get("test_input") or [])
    test_output = list(raw.get("test_output") or [])
    total = min(len(test_input), len(test_output), int(args.code_max_tests))
    reward_inputs = test_input[:total]
    reward_outputs = test_output[:total]
    prompt_hash = hashlib.sha256(
        json.dumps(
            {
                "task": "code",
                "source_dataset": dataset,
                "source_row": source_row,
                "question": question,
                "test_slice": f"first_{int(args.code_max_tests)}",
                "builder": "tool_nullspace_v1",
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    prompt_id = f"code__tns_{dataset.lower()}_{source_row:04d}_{prompt_hash[:8]}"
    metadata = {
        "source_dataset": dataset,
        "source_path": str(Path(args.cure_data_root).expanduser().resolve() / f"{dataset}.json"),
        "source_row": int(source_row),
        "task_id": f"{dataset}:{source_row}",
        "question_id": f"{dataset}:{source_row}",
        "test_method": raw.get("test_method") or "stdio",
        "exe_method": raw.get("test_method") or "stdio",
        "test_time_limit": raw.get("test_time_limit", 8),
        "test_input": reward_inputs,
        "test_output": reward_outputs,
        "reward_test_input": reward_inputs,
        "reward_test_output": reward_outputs,
        "reward_test_indices": list(range(total)),
        "all_test_count": min(len(test_input), len(test_output)),
        "code_bank_role": "formal_cure_eval_anchor",
        "prompt_template": "CURE/evaluation/evaluation_config.py::system_prompts",
    }
    return {
        "prompt_id": prompt_id,
        "group_id": prompt_id,
        "task": "code",
        "source": metadata["source_path"],
        "source_row": int(source_row),
        "split": "tool_nullspace_v1_code_live",
        "prompt": cure_formal_prompt(question),
        "messages": [],
        "reference": {"answer": None, "response": "", "metadata": metadata},
        "verifier": {"name": "cure_code_pass_rate", "config": {"source": dataset, "test_slice": f"first_{int(args.code_max_tests)}_formal_cure"}},
        "tags": ["code", "cure", "formal_eval_calibration", "tool_nullspace_v1", f"cure_dataset:{dataset}"],
        "difficulty": "hard_vs_ta13",
        "prompt_hash": prompt_hash,
        "tool_nullspace_v1": {
            "created_at": created_at,
            "selection_policy": "TA-1/3 fail and ReasonFlux/R1/MemoryAgent expert success",
        },
    }


def _load_cure_model_success_status(temp_root: Path, *, model_name: str, dataset: str, indices: list[int], max_tests: int, max_positives: int):
    path = _resolve_cure_temp_path(temp_root, model_name=model_name, dataset=dataset)
    selected = set(int(index) for index in indices)
    status = {}
    for idx, row in _iter_selected_temp_rows(path, selected):
        successes = _success_samples(
            row,
            dataset=dataset,
            source_row=idx,
            model_name=model_name,
            temp_file=path,
            max_tests=max_tests,
            max_positives=max_positives,
        )
        status[idx] = {"success_count": len(successes), "success_samples": successes, "temp_file": str(path)}
    return status


def _resolve_cure_temp_path(temp_root: Path, *, model_name: str, dataset: str) -> Path:
    candidates = [
        temp_root / f"outputs-eval-.{model_name}-{dataset}.json",
        temp_root / f"outputs-eval-.mnt.cache.wuruixiao.models.{model_name}-{dataset}.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(candidates[0])


def _iter_selected_temp_rows(path: Path, selected: set[int]) -> Iterable[tuple[int, dict[str, Any]]]:
    try:
        import ijson
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("ijson is required to stream CURE temp outputs") from error
    if not path.exists():
        raise FileNotFoundError(path)
    max_index = max(selected) if selected else -1
    with path.open("rb") as handle:
        for idx, row in enumerate(ijson.items(handle, "item")):
            if idx in selected:
                yield idx, row
            if idx >= max_index:
                break


def _success_samples(row: dict[str, Any], *, dataset: str, source_row: int, model_name: str, temp_file: Path, max_tests: int, max_positives: int):
    bool_table = list(row.get("test_bool_table") or [])
    full_generations = list(row.get("full_code_generation") or [])
    generated_codes = list(row.get("generated_code") or [])
    successes = []
    for sample_index, case_bools in enumerate(bool_table):
        flags = [bool(item) for item in list(case_bools or [])[:max_tests]]
        if not flags or not all(flags):
            continue
        text = str(full_generations[sample_index]) if sample_index < len(full_generations) else ""
        text_source = "full_code_generation"
        if "```python" not in text and sample_index < len(generated_codes):
            text = "```python\n" + str(generated_codes[sample_index]).strip() + "\n```"
            text_source = "generated_code_wrapped"
        if not text.strip():
            continue
        successes.append(
            {
                "dataset": dataset,
                "source_row": int(source_row),
                "model_name": model_name,
                "temp_file": str(temp_file),
                "sample_index": int(sample_index),
                "text": text,
                "text_source": text_source,
                "pass_count": sum(flags),
                "test_count": len(flags),
            }
        )
    successes.sort(key=lambda item: (item["sample_index"], _stable_int([model_name, dataset, source_row, item["sample_index"]])))
    if max_positives <= 0:
        return successes
    return successes[:max_positives]


def _rollout_row(*, run_id: str, policy_id: str, prompt_row: dict[str, Any], sample: dict[str, Any], created_at: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "created_at": created_at,
        "step": 0,
        "policy_id": policy_id,
        "gate_checkpoint": None,
        "gate_values": {},
        "gate_id": policy_id,
        "group_id": prompt_row["group_id"],
        "prompt_id": prompt_row["prompt_id"],
        "task": prompt_row["task"],
        "prompt": prompt_row["prompt"],
        "reference": prompt_row["reference"],
        "rendered_prompt": prompt_row["prompt"],
        "samples": [sample],
        "frontier": {
            "all_failure": False,
            "all_success": True,
            "num_success": 1,
            "num_failure": 0,
            "mean_reward": float(sample["reward_train"]),
            "std_reward": 0.0,
            "reward_field": "reward_train",
        },
        "keep_for_policy_loss": False,
        "skip_reason": "expert_positive_anchor_only",
    }


def _group_by_task(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        output[str(row.get("task") or "")].append(row)
    return output


def _dedupe_seed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    output = []
    for row in rows:
        key = str(row.get("prompt_id") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def _dedupe_rollout_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    output = []
    for row in rows:
        key = (str(row.get("prompt_id") or ""), str(row.get("policy_id") or ""))
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def _safe_policy_id(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(value))[:96]


def _stable_int(value: Any) -> int:
    return int(hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16], 16)


def _render_readme(summary: dict[str, Any]) -> str:
    counts = summary["counts"]
    return "\n".join(
        [
            "# Tool Nullspace v1 Calibration",
            "",
            f"生成时间：`{summary['created_at']}`",
            "",
            "## Counts",
            "",
            f"- merged rows: `{counts['merged_rows']}`",
            f"- task counts: `{counts['merged_task_counts']}`",
            f"- extra expert rows: `{counts['extra_expert_rows']}`",
            f"- extra positive samples: `{counts['extra_positive_samples']}`",
            "",
            "## Outputs",
            "",
            *[f"- `{key}`: `{path}`" for key, path in summary["outputs"].items()],
            "",
            "## Notes",
            "",
            "- Tool final count is 32: 16 original paper96 Tool + 16 BFCL live historical-success rows.",
            "- Code keeps original paper96 32 rows and adds 8 formal LiveBench/LiveCodeBench rows.",
            "- `tool_nullspace_replay_rollout` is the fixed replay bank for Tool behavior-span nullspace construction.",
        ]
    ) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, default=DEFAULT_BASE_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bfcl-root", type=Path, default=DEFAULT_BFCL_ROOT)
    parser.add_argument("--tool-ta13-result-root", type=Path, default=DEFAULT_TA13_BFCL_RESULT_ROOT)
    parser.add_argument("--tool-result-root", action="append", default=[str(path) for path in DEFAULT_TOOL_RESULT_ROOTS])
    parser.add_argument("--cure-data-root", type=Path, default=DEFAULT_CURE_DATA_ROOT)
    parser.add_argument("--cure-temp-root", type=Path, default=DEFAULT_CURE_TEMP_ROOT)
    parser.add_argument("--cure-ta-temp-root", type=Path, default=DEFAULT_CURE_TA_TEMP_ROOT)
    parser.add_argument("--keep-original-tool", type=int, default=16)
    parser.add_argument("--add-bfcl-tool", type=int, default=16)
    parser.add_argument("--add-code-live", type=int, default=8)
    parser.add_argument("--code-max-tests", type=int, default=8)
    parser.add_argument("--code-max-positives-per-expert", type=int, default=1)
    parser.add_argument("--code-positive-threshold", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260519)
    return parser.parse_args()


if __name__ == "__main__":
    main()
