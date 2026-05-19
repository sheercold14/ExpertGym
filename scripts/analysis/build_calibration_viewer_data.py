#!/usr/bin/env python3
"""Build embedded data for the calibration viewer."""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
from pathlib import Path
from typing import Any


TASK_REWARD_EXPLANATIONS = {
    "toolrl_source_reward": {
        "title": "ToolRL source reward",
        "adapter": "ToolRewardAdapter",
        "code": "opvec/rewards/simple.py",
        "route": "RewardRouter -> adapter_for_task('tool')",
        "formula": "raw = format_score + tool_call_correctness_reward; reward_train = clip((raw + 3) / 7, 0, 1)",
        "success": "format_score == 1 and exact_tool_match == true",
        "fields": [
            "reference.response: canonical <tool_call> answer",
            "output_text: model response parsed for <tool_call> payloads",
            "reference.metadata.source_dataset/source_index: ToolRL row provenance",
        ],
    },
    "bfcl_ast": {
        "title": "BFCL AST reward",
        "adapter": "BFCLToolRewardAdapter",
        "code": "opvec/rewards/bfcl.py",
        "route": "RewardRouter -> BFCLToolRewardAdapter when reference.bfcl exists",
        "formula": "full AST success => 1.0; otherwise min(0.90, 0.20 + 0.50*partial_call_match + 0.15*name_recall + 0.05*count_match)",
        "success": "BFCL ast_checker(valid=true)",
        "fields": [
            "reference.bfcl.function: available function schemas",
            "reference.bfcl.possible_answer: expected function calls",
            "reference.bfcl.category: BFCL-style category",
        ],
    },
    "memagent_source_reward": {
        "title": "MemAgent / HotpotQA reward",
        "adapter": "MemoryRewardAdapter",
        "code": "opvec/rewards/simple.py",
        "route": "RewardRouter -> adapter_for_task('memory')",
        "formula": "final answer: compute_score(solution[-300:], ground_truth_list); update turns are intermediate and get reward 0",
        "success": "score >= 1.0 for final-answer prompts",
        "fields": [
            "reference.answer / reference.metadata.ground_truth: accepted answers",
            "model output final boxed answer: prediction region",
            "reference.metadata.memagent_chunks: retrieval context used by the prompt",
        ],
    },
    "cure_code_pass_rate": {
        "title": "CURE code pass-rate reward",
        "adapter": "CodeRewardAdapter",
        "code": "opvec/rewards/simple.py",
        "route": "RewardRouter -> adapter_for_task('code')",
        "formula": "if source tests exist: reward = passed_tests / total_tests; otherwise syntax/input/output/public examples fallback",
        "success": "reward >= 0.95",
        "fields": [
            "reference.metadata.test_input/test_output: executable tests when present",
            "prompt examples: fallback public examples",
            "model output code block: extracted Python code",
        ],
    },
}


LIVECODEBENCH_FLOW = [
    {
        "title": "Load Dataset",
        "description": "CURE reads ../data/LiveCodeBench.json. Each row contains question, public examples, hidden-style test_input/test_output, and test_time_limit.",
    },
    {
        "title": "Sample Code",
        "description": "For each task, the evaluator builds a code-generation prompt and samples k_code candidate programs from the evaluated model.",
    },
    {
        "title": "Sample Unit Tests",
        "description": "When single_eval=false, the same model also receives a unit-test-generation prompt and samples k_case generated tests.",
    },
    {
        "title": "Execute Matrix",
        "description": "Every candidate program is executed against generated tests and ground-truth tests, producing case_bool_table and test_bool_table.",
    },
    {
        "title": "Compute Metrics",
        "description": "code_acc counts fully-passing sampled programs; hidden_test_acc averages all hidden-test pass cells; BoN selects the candidate with the most generated-test passes, then scores it on hidden tests.",
    },
]


LIVECODEBENCH_FORMULAS = {
    "code_acc": "sum_i num_candidates_passing_all_hidden_tests(i) / sum_i num_candidates(i)",
    "hidden_test_acc": "mean over tasks of passed_hidden_test_cells / total_hidden_test_cells",
    "estimated_unit_test_acc": "for tasks with at least one correct code, fraction of generated tests passed by all correct codes",
    "bon_4x4_acc": "for each task, argmax over first 4 candidates by first 4 generated-test pass count; score 1 if selected candidate passes all hidden tests",
    "bon_4x4_hidden_test_acc": "hidden-test pass rate of the BoN-selected candidate, averaged over tasks",
}


def _shorten_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[:limit].rstrip() + f"\n...[truncated {len(value) - limit} chars]"


def _compact(value: Any, *, text_limit: int = 5000, depth: int = 0) -> Any:
    if depth > 5:
        return "[max depth]"
    if isinstance(value, str):
        return _shorten_text(value, text_limit)
    if isinstance(value, list):
        max_items = 18 if depth < 3 else 8
        compacted = [_compact(item, text_limit=text_limit, depth=depth + 1) for item in value[:max_items]]
        if len(value) > max_items:
            compacted.append(f"...[{len(value) - max_items} more items]")
        return compacted
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            limit = text_limit
            if key in {"memagent_chunks"}:
                limit = 1600
            elif key in {"prompt", "response"}:
                limit = 8000
            result[key] = _compact(item, text_limit=limit, depth=depth + 1)
        return result
    return value


def _bool_table(value: Any) -> list[list[bool]]:
    if not isinstance(value, list):
        return []
    table: list[list[bool]] = []
    for row in value:
        if isinstance(row, list):
            table.append([bool(item) for item in row])
    return table


def _compact_sample(sample: dict[str, Any], *, text_limit: int) -> dict[str, Any]:
    details = sample.get("details") or {}
    return {
        "sample_id": sample.get("sample_id"),
        "success": sample.get("success"),
        "reward": sample.get("reward"),
        "reward_train": sample.get("reward_train"),
        "task_reward": sample.get("task_reward"),
        "contract_reward": sample.get("contract_reward"),
        "length": sample.get("length"),
        "text": _compact(sample.get("text") or "", text_limit=text_limit),
        "details": _compact(details, text_limit=3500),
    }


def _read_rollouts(
    specs: list[str],
    *,
    tasks: set[str],
    max_samples_per_source: int,
    sample_text_limit: int,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    by_prompt: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    source_summaries: list[dict[str, Any]] = []
    for spec in specs:
        label, raw_path = _split_labeled_path(spec)
        path = Path(raw_path).expanduser()
        rows_read = 0
        rows_kept = 0
        if not path.exists():
            source_summaries.append({"label": label, "path": str(path), "missing": True})
            continue
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                rows_read += 1
                row = json.loads(line)
                if tasks and str(row.get("task")) not in tasks:
                    continue
                rows_kept += 1
                prompt_id = str(row.get("prompt_id") or "")
                samples = list(row.get("samples") or [])
                source = {
                    "label": label,
                    "path": str(path),
                    "run_id": row.get("run_id"),
                    "policy_id": row.get("policy_id"),
                    "gate_id": row.get("gate_id"),
                    "gate_checkpoint": row.get("gate_checkpoint"),
                    "gate_values": _compact(row.get("gate_values") or {}, text_limit=1200),
                    "frontier": _compact(row.get("frontier") or {}, text_limit=1800),
                    "keep_for_policy_loss": row.get("keep_for_policy_loss"),
                    "sample_count": len(samples),
                    "samples": [
                        _compact_sample(sample, text_limit=sample_text_limit)
                        for sample in samples[:max_samples_per_source]
                    ],
                }
                by_prompt[prompt_id].append(source)
        source_summaries.append(
            {"label": label, "path": str(path), "rows_read": rows_read, "rows_kept": rows_kept}
        )
    return dict(by_prompt), source_summaries


def _read_livecodebench_eval(
    app_data_path: Path | None,
    *,
    model_id: str,
    result_path: Path | None,
    max_examples: int,
) -> dict[str, Any] | None:
    if not app_data_path:
        return None
    app_data_path = app_data_path.expanduser()
    if not app_data_path.exists():
        return {
            "missing": True,
            "app_data_path": str(app_data_path),
            "model_id": model_id,
        }
    app_data = json.loads(app_data_path.read_text(encoding="utf-8"))
    model_info = next(
        (model for model in app_data.get("models", []) if model.get("model_id") == model_id),
        {"model_id": model_id},
    )
    metrics = (
        app_data.get("metrics", {})
        .get("models", {})
        .get(model_id, {})
        .get("code", {})
        .get("datasets", {})
        .get("LiveCodeBench", {})
    )
    cases = [
        case
        for case in app_data.get("cases", [])
        if case.get("benchmark") == "CURE" and case.get("category") == "LiveCodeBench"
    ]
    examples = _select_livecode_examples(cases, model_id=model_id, max_examples=max_examples)
    result_text = ""
    if result_path:
        result_path = result_path.expanduser()
        if result_path.exists():
            result_text = _shorten_text(result_path.read_text(encoding="utf-8"), 5000)
    source_files = sorted(
        {
            str(case.get("prompt_source_file"))
            for case in cases
            if case.get("prompt_source_file")
        }
    )
    return {
        "format": "livecodebench_eval_audit_v1",
        "model_id": model_id,
        "display_name": model_info.get("display_name", model_id),
        "app_data_path": str(app_data_path),
        "result_path": str(result_path) if result_path else None,
        "raw_result_text": result_text,
        "cure_eval_script": "/mnt/cache/wuruixiao/users/lsc/CURE/evaluation/eval.py",
        "cure_eval_config": "/mnt/cache/wuruixiao/users/lsc/CURE/evaluation/evaluation_config.py",
        "case_browser_builder": "scripts/analysis/build_eval_case_browser.py",
        "output_files": source_files[:8],
        "output_file_count": len(source_files),
        "dataset": "LiveCodeBench",
        "total_cases": len(cases),
        "metrics": metrics,
        "flow": LIVECODEBENCH_FLOW,
        "formulas": LIVECODEBENCH_FORMULAS,
        "examples": examples,
    }


def _select_livecode_examples(
    cases: list[dict[str, Any]],
    *,
    model_id: str,
    max_examples: int,
) -> list[dict[str, Any]]:
    selected: list[tuple[str, dict[str, Any]]] = []

    def add(label: str, predicate) -> None:
        if len(selected) >= max_examples:
            return
        for case in cases:
            if any(case.get("case_id") == row.get("case_id") for _, row in selected):
                continue
            row = case.get("models", {}).get(model_id, {})
            if predicate(row):
                selected.append((label, case))
                return

    add(
        "BoN 成功样例",
        lambda row: bool(row.get("bon_success"))
        and int(row.get("code_success_count") or 0) > 0
        and _has_eval_tables(row),
    )
    add(
        "Selection 失败样例",
        lambda row: bool(row.get("any_code_success"))
        and not bool(row.get("bon_success"))
        and int(row.get("code_success_count") or 0) > 0
        and _has_eval_tables(row),
    )
    add(
        "无完全正确候选样例",
        lambda row: not bool(row.get("any_code_success"))
        and not bool(row.get("bon_success"))
        and _has_eval_tables(row),
    )

    if len(selected) < max_examples:
        for case in cases:
            if len(selected) >= max_examples:
                break
            if any(case.get("case_id") == row.get("case_id") for _, row in selected):
                continue
            row = case.get("models", {}).get(model_id, {})
            if _has_eval_tables(row):
                selected.append(("普通样例", case))

    return [
        _livecode_example_payload(case, model_id=model_id, label=label)
        for label, case in selected[:max_examples]
    ]


def _has_eval_tables(row: dict[str, Any]) -> bool:
    return bool(row.get("case_bool_table")) and bool(row.get("test_bool_table"))


def _livecode_example_payload(
    case: dict[str, Any],
    *,
    model_id: str,
    label: str,
) -> dict[str, Any]:
    row = case.get("models", {}).get(model_id, {})
    case_table = _bool_table(row.get("case_bool_table"))
    test_table = _bool_table(row.get("test_bool_table"))
    case_scores = [sum(1 for item in values[:4] if item) for values in case_table[:4]]
    bon_index = row.get("bon_code_index")
    if bon_index is None and case_scores:
        bon_index = max(range(len(case_scores)), key=lambda idx: case_scores[idx])
    correct_indices = [
        idx for idx, values in enumerate(test_table)
        if values and all(values)
    ]
    wrong_indices = [
        idx for idx in range(len(test_table))
        if idx not in correct_indices
    ]
    candidates = _livecode_candidate_payloads(
        row,
        case_table=case_table,
        test_table=test_table,
        bon_index=bon_index,
    )
    generated_tests = _livecode_generated_test_payloads(
        row,
        case_table=case_table,
        correct_indices=correct_indices,
        wrong_indices=wrong_indices,
    )
    selected_hidden_vector = (
        test_table[int(bon_index)]
        if bon_index is not None and int(bon_index) < len(test_table)
        else []
    )
    hidden_cells = sum(len(values) for values in test_table)
    hidden_passed = sum(1 for values in test_table for item in values if item)
    selected_case_score = (
        case_scores[int(bon_index)]
        if bon_index is not None and int(bon_index) < len(case_scores)
        else None
    )
    return {
        "label": label,
        "case_id": case.get("case_id"),
        "question": _shorten_text(str(case.get("question") or ""), 3000),
        "code_tags": case.get("code_tags") or [],
        "failure_tags": row.get("failure_tags") or [],
        "output_file": row.get("output_file") or case.get("prompt_source_file"),
        "metrics": {
            "code_sample_count": row.get("code_sample_count"),
            "generated_test_count": row.get("generated_test_count"),
            "hidden_test_count": row.get("hidden_test_count"),
            "code_success_count": row.get("code_success_count"),
            "code_success_rate": row.get("code_success_rate"),
            "hidden_test_acc": row.get("hidden_test_acc"),
            "any_code_success": row.get("any_code_success"),
            "bon_code_index": bon_index,
            "bon_success": row.get("bon_success"),
            "bon_hidden_test_acc": row.get("bon_hidden_test_acc"),
        },
        "metric_calculation": {
            "correct_candidate_indices": correct_indices,
            "wrong_candidate_indices": wrong_indices,
            "case_scores_first_4x4": case_scores,
            "bon_selected_index": bon_index,
            "selected_generated_test_passes": selected_case_score,
            "selected_hidden_test_vector": selected_hidden_vector,
            "selected_hidden_passes": sum(1 for item in selected_hidden_vector if item),
            "selected_hidden_total": len(selected_hidden_vector),
            "bon_success_contribution": int(bool(selected_hidden_vector and all(selected_hidden_vector))),
            "case_hidden_test_acc": hidden_passed / hidden_cells if hidden_cells else None,
        },
        "hidden_tests_preview": {
            "input": row.get("test_input_preview") or [],
            "output": row.get("test_output_preview") or [],
        },
        "full_code_generation_preview": row.get("full_code_generation_preview") or [],
        "full_case_generation_preview": row.get("full_case_generation_preview") or [],
        "candidates": candidates,
        "generated_tests": generated_tests,
        "case_bool_table_first_4x4": [values[:4] for values in case_table[:4]],
        "test_bool_table": test_table,
    }


def _livecode_candidate_payloads(
    row: dict[str, Any],
    *,
    case_table: list[list[bool]],
    test_table: list[list[bool]],
    bon_index: int | None,
) -> list[dict[str, Any]]:
    generated_code = row.get("generated_code_preview") or []
    candidates = []
    max_rows = max(len(generated_code), len(test_table), len(case_table))
    for idx in range(max_rows):
        hidden_vector = test_table[idx] if idx < len(test_table) else []
        generated_vector = case_table[idx] if idx < len(case_table) else []
        candidates.append(
            {
                "index": idx,
                "selected_by_bon": bon_index is not None and idx == int(bon_index),
                "passes_all_hidden_tests": bool(hidden_vector and all(hidden_vector)),
                "hidden_test_vector": hidden_vector,
                "hidden_passes": sum(1 for item in hidden_vector if item),
                "hidden_total": len(hidden_vector),
                "generated_test_vector_first_4": generated_vector[:4],
                "generated_test_passes_first_4": sum(1 for item in generated_vector[:4] if item),
                "code_preview": _compact(generated_code[idx] if idx < len(generated_code) else "", text_limit=2200),
            }
        )
    return candidates


def _livecode_generated_test_payloads(
    row: dict[str, Any],
    *,
    case_table: list[list[bool]],
    correct_indices: list[int],
    wrong_indices: list[int],
) -> list[dict[str, Any]]:
    case_input = row.get("case_input") or []
    case_output = row.get("case_output") or []
    if not case_table:
        return []
    max_tests = min(4, max(len(values) for values in case_table if values))
    tests = []
    for idx in range(max_tests):
        column = [values[idx] for values in case_table if idx < len(values)]
        correct_values = [
            case_table[code_idx][idx]
            for code_idx in correct_indices
            if code_idx < len(case_table) and idx < len(case_table[code_idx])
        ]
        wrong_values = [
            case_table[code_idx][idx]
            for code_idx in wrong_indices
            if code_idx < len(case_table) and idx < len(case_table[code_idx])
        ]
        tests.append(
            {
                "index": idx,
                "input": _compact(case_input[idx] if idx < len(case_input) else "", text_limit=1000),
                "output": _compact(case_output[idx] if idx < len(case_output) else "", text_limit=1000),
                "pass_vector_by_candidate": column,
                "passed_by_all_correct_candidates": bool(correct_values) and all(correct_values),
                "wrong_candidate_reject_count": sum(1 for item in wrong_values if not item),
                "wrong_candidate_count": len(wrong_values),
            }
        )
    return tests


def _split_labeled_path(spec: str) -> tuple[str, str]:
    if "=" in spec:
        label, raw_path = spec.split("=", 1)
        label = label.strip()
    else:
        raw_path = spec
        label = Path(raw_path).parent.name
    return label or Path(raw_path).parent.name, raw_path


def _reference_summary(row: dict[str, Any]) -> dict[str, Any]:
    ref = row.get("reference") or {}
    meta = ref.get("metadata") or {}
    task = row.get("task")
    if task == "tool" and ref.get("bfcl"):
        bfcl = ref["bfcl"]
        return {
            "kind": "bfcl",
            "category": bfcl.get("category"),
            "functions": len(bfcl.get("function") or []),
            "expected_calls": len(bfcl.get("possible_answer") or []),
            "target_tags": meta.get("failure_tags_targeted") or row.get("eval_targeted_calibration", {}).get("failure_tags_targeted"),
        }
    if task == "tool":
        return {
            "kind": "toolrl",
            "source_dataset": meta.get("source_dataset"),
            "has_tool_call": meta.get("has_tool_call"),
            "reference_calls": str(ref.get("response") or "").count("<tool_call>"),
        }
    if task == "memory":
        return {
            "kind": "hotpotqa",
            "question_id": meta.get("question_id"),
            "answers": ref.get("answer") or meta.get("ground_truth"),
            "num_docs": meta.get("num_docs"),
            "num_chunks": meta.get("num_chunks"),
            "context_tokens": meta.get("context_token_count"),
        }
    if task == "code":
        tests = meta.get("test_input") or []
        return {
            "kind": "codecontests",
            "task_id": meta.get("task_id") or meta.get("question_id"),
            "code_tags": meta.get("code_tags") or row.get("eval_targeted_calibration", {}).get("code_tags"),
            "exe_method": meta.get("exe_method"),
            "test_count": len(tests) if isinstance(tests, list) else 0,
            "time_limit": meta.get("test_time_limit"),
        }
    return {}


def _row_payload(row: dict[str, Any], index: int, rollouts_by_prompt: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    verifier = row.get("verifier") or {}
    verifier_name = verifier.get("name") or ""
    eval_info = row.get("eval_targeted_calibration") or {}
    selection = row.get("question_bank_selection") or {}
    return {
        "index": index,
        "prompt_id": row.get("prompt_id"),
        "prompt_hash": row.get("prompt_hash"),
        "task": row.get("task"),
        "role": eval_info.get("role") or "",
        "verifier": verifier_name,
        "reward": TASK_REWARD_EXPLANATIONS.get(verifier_name, {}),
        "source": row.get("source"),
        "source_row": row.get("source_row"),
        "split": row.get("split"),
        "tags": row.get("tags") or [],
        "prompt": _compact(row.get("prompt") or "", text_limit=12000),
        "messages": _compact(row.get("messages") or [], text_limit=5000),
        "reference": _compact(row.get("reference") or {}, text_limit=7000),
        "reference_summary": _reference_summary(row),
        "eval_targeted_calibration": _compact(eval_info, text_limit=3500),
        "question_bank_selection": _compact(selection, text_limit=2500),
        "rollouts": rollouts_by_prompt.get(str(row.get("prompt_id") or ""), []),
        "raw_preview": _compact(row, text_limit=3500),
    }


def build(
    input_path: Path,
    output_path: Path,
    *,
    name: str,
    summary_path: Path | None,
    rollout_specs: list[str],
    rollout_tasks: set[str],
    max_samples_per_source: int,
    rollout_text_limit: int,
    livecode_app_data: Path | None,
    livecode_model_id: str,
    livecode_result: Path | None,
    livecode_max_examples: int,
) -> None:
    rows = [json.loads(line) for line in input_path.read_text().splitlines() if line.strip()]
    rollouts_by_prompt, rollout_sources = _read_rollouts(
        rollout_specs,
        tasks=rollout_tasks,
        max_samples_per_source=max_samples_per_source,
        sample_text_limit=rollout_text_limit,
    )
    task_counts = collections.Counter(str(row.get("task")) for row in rows)
    role_counts = collections.Counter(str((row.get("eval_targeted_calibration") or {}).get("role") or "") for row in rows)
    verifier_counts = collections.Counter(str((row.get("verifier") or {}).get("name") or "") for row in rows)
    summary: dict[str, Any] = {}
    if summary_path and summary_path.exists():
        summary = json.loads(summary_path.read_text())
    livecodebench_eval = _read_livecodebench_eval(
        livecode_app_data,
        model_id=livecode_model_id,
        result_path=livecode_result,
        max_examples=livecode_max_examples,
    )
    payload = {
        "dataset": {
            "name": name,
            "path": str(input_path),
            "summary_path": str(summary_path) if summary_path else None,
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "row_count": len(rows),
            "task_counts": dict(task_counts),
            "role_counts": dict(role_counts),
            "verifier_counts": dict(verifier_counts),
            "source_files": sorted({str(row.get("source")) for row in rows if row.get("source")}),
            "rollout_sources": rollout_sources,
            "rollout_task_filter": sorted(rollout_tasks),
            "rollout_max_samples_per_source": max_samples_per_source,
        },
        "reward_explanations": TASK_REWARD_EXPLANATIONS,
        "livecodebench_eval": livecodebench_eval,
        "source_summary": _compact(summary, text_limit=3000),
        "rows": [_row_payload(row, index, rollouts_by_prompt) for index, row in enumerate(rows, start=1)],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "window.CALIBRATION_VIEWER_DATA = "
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + ";\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--name", default="eval_targeted96_cure_aligned_20260517")
    parser.add_argument("--summary", type=Path)
    parser.add_argument(
        "--rollout",
        action="append",
        default=[],
        help="Optional labeled rollout JSONL, e.g. policy_iter001=/path/rollouts.jsonl. Repeatable.",
    )
    parser.add_argument("--rollout-task", action="append", default=["code"], help="Task to keep from rollouts.")
    parser.add_argument("--max-samples-per-source", type=int, default=4)
    parser.add_argument("--rollout-text-limit", type=int, default=10000)
    parser.add_argument(
        "--livecode-app-data",
        type=Path,
        help="Optional eval_case_browser app_data.json with compact real CURE LiveCodeBench outputs.",
    )
    parser.add_argument("--livecode-model-id", default="expertgym-B-codeaug-opd-i18")
    parser.add_argument("--livecode-result", type=Path, help="Optional CURE results text file.")
    parser.add_argument("--livecode-max-examples", type=int, default=2)
    args = parser.parse_args()
    build(
        args.input,
        args.output,
        name=args.name,
        summary_path=args.summary,
        rollout_specs=args.rollout,
        rollout_tasks=set(args.rollout_task),
        max_samples_per_source=args.max_samples_per_source,
        rollout_text_limit=args.rollout_text_limit,
        livecode_app_data=args.livecode_app_data,
        livecode_model_id=args.livecode_model_id,
        livecode_result=args.livecode_result,
        livecode_max_examples=args.livecode_max_examples,
    )


if __name__ == "__main__":
    main()
