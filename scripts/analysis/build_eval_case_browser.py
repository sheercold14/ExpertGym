#!/usr/bin/env python3
"""Build a model-comparison case database from BFCL score/result files."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

try:
    import ijson
except ImportError:  # pragma: no cover - fallback for minimal environments
    ijson = None


BFCL_CATEGORIES: dict[str, tuple[str, str]] = {
    "parallel": ("non_live", "BFCL_v4_parallel"),
    "parallel_multiple": ("non_live", "BFCL_v4_parallel_multiple"),
    "live_parallel": ("live", "BFCL_v4_live_parallel"),
    "live_parallel_multiple": ("live", "BFCL_v4_live_parallel_multiple"),
}

CODE_DATASETS = ["LiveBench", "LiveCodeBench"]


def main() -> None:
    args = parse_args()
    registry_path = Path(args.registry).expanduser().resolve()
    registry = _read_json(registry_path)
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    bfcl_data_root = Path(registry["bfcl_data_root"]).expanduser()
    models = registry["models"]
    model_ids = [model["model_id"] for model in models]
    pair = registry.get("default_pair", {})
    left_id = args.left or pair.get("left") or model_ids[0]
    right_id = args.right or pair.get("right") or model_ids[1]
    if left_id not in model_ids or right_id not in model_ids:
        raise SystemExit(f"unknown pair: {left_id=} {right_id=} known={model_ids}")

    prompts = _load_bfcl_prompts(bfcl_data_root)
    answers = _load_bfcl_answers(bfcl_data_root)
    tool_outputs = {
        model["model_id"]: _load_model_bfcl_outputs(model)
        for model in models
    }
    code_outputs = {
        model["model_id"]: _load_model_code_outputs(model)
        for model in models
    }

    cases = []
    cases.extend(_build_bfcl_cases(prompts, answers, models, tool_outputs))
    cases.extend(_build_code_cases(models, code_outputs))
    pairwise = _build_pairwise(cases, left_id=left_id, right_id=right_id)
    metrics = _build_metrics(cases, pairwise, models, left_id=left_id, right_id=right_id)
    calibration_candidates = _build_calibration_candidates(
        cases,
        pairwise,
        left_id=left_id,
        right_id=right_id,
    )

    _write_jsonl(out_dir / "cases.jsonl", cases)
    _write_jsonl(out_dir / "pairwise_diffs.jsonl", pairwise)
    _write_jsonl(out_dir / "bfcl_live_calibration_candidates.jsonl", calibration_candidates)
    _write_json(out_dir / "model_metrics.json", metrics)
    _write_json(
        out_dir / "app_data.json",
        {
            "format": "eval_case_browser_app_data_v1",
            "registry_path": str(registry_path),
            "models": [
                {
                    "model_id": model["model_id"],
                    "display_name": model.get("display_name", model["model_id"]),
                    "tags": model.get("tags", []),
                }
                for model in models
            ],
            "default_pair": {"left": left_id, "right": right_id},
            "benchmarks": sorted(set(case["benchmark"] for case in cases)),
            "categories": sorted(set(case["category"] for case in cases)),
            "tool_categories": list(BFCL_CATEGORIES),
            "code_datasets": CODE_DATASETS,
            "metrics": metrics,
            "cases": cases,
            "pairwise": pairwise,
            "calibration_candidates": calibration_candidates,
        },
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[done] wrote {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--registry",
        default="docs/evaluation/eval_case_browser/models.json",
        help="Model registry JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/shared-storage/OnPolicy/analysis/eval_case_browser",
    )
    parser.add_argument("--left", default=None)
    parser.add_argument("--right", default=None)
    return parser.parse_args()


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _iter_json_array(path: Path):
    if ijson is not None:
        with path.open("rb") as f:
            yield from ijson.items(f, "item")
        return

    decoder = json.JSONDecoder()
    buffer = ""
    eof = False
    with path.open("r", encoding="utf-8") as f:
        while True:
            if not eof and len(buffer) < 1_000_000:
                chunk = f.read(1_000_000)
                if chunk:
                    buffer += chunk
                else:
                    eof = True
            buffer = buffer.lstrip()
            if buffer.startswith("["):
                buffer = buffer[1:]
                continue
            if buffer.startswith(","):
                buffer = buffer[1:]
                continue
            if buffer.startswith("]"):
                return
            if not buffer:
                if eof:
                    return
                continue
            try:
                obj, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                if eof:
                    raise
                chunk = f.read(1_000_000)
                if chunk:
                    buffer += chunk
                    continue
                eof = True
                continue
            yield obj
            buffer = buffer[end:]


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _load_bfcl_prompts(root: Path) -> dict[str, dict[str, Any]]:
    output = {}
    for category, (_, stem) in BFCL_CATEGORIES.items():
        path = root / f"{stem}.json"
        rows = _iter_jsonl(path)
        for row in rows:
            output[row["id"]] = {
                "category": category,
                "prompt": row,
                "source_file": str(path),
            }
    return output


def _load_bfcl_answers(root: Path) -> dict[str, dict[str, Any]]:
    output = {}
    answer_root = root / "possible_answer"
    for category, (_, stem) in BFCL_CATEGORIES.items():
        path = answer_root / f"{stem}.json"
        rows = _iter_jsonl(path)
        for row in rows:
            output[row["id"]] = {
                "category": category,
                "ground_truth": row.get("ground_truth"),
                "source_file": str(path),
            }
    return output


def _load_model_bfcl_outputs(model: dict[str, Any]) -> dict[str, dict[str, Any]]:
    score_root = Path(model["tool_score_root"])
    result_root = Path(model["tool_result_root"])
    output: dict[str, dict[str, Any]] = {}
    for category, (split, stem) in BFCL_CATEGORIES.items():
        score_path = score_root / split / f"{stem}_score.json"
        result_path = result_root / split / f"{stem}_result.json"
        summary, failures = _load_score_file(score_path)
        results = {row["id"]: row for row in _iter_jsonl(result_path)}
        for case_id, result in results.items():
            failure = failures.get(case_id)
            valid = failure is None
            decoded = failure.get("model_result_decoded") if failure else None
            error = failure.get("error") if failure else None
            error_type = failure.get("error_type") if failure else None
            output[case_id] = {
                "model_id": model["model_id"],
                "category": category,
                "valid": valid,
                "error_type": error_type,
                "error": error,
                "failure_tags": _failure_tags(
                    error_type=error_type,
                    error=error,
                    decoded_calls=decoded,
                    raw_output=result.get("result"),
                    ground_truth=None,
                ),
                "raw_output": result.get("result"),
                "decoded_calls": decoded,
                "input_token_count": result.get("input_token_count"),
                "output_token_count": result.get("output_token_count"),
                "latency": result.get("latency"),
                "score_file": str(score_path),
                "result_file": str(result_path),
            }
        expected_total = int(summary.get("total_count", len(results)))
        if len(results) != expected_total:
            raise SystemExit(
                f"{model['model_id']} {category}: result rows {len(results)} != score total {expected_total}"
            )
    return output


def _load_model_code_outputs(model: dict[str, Any]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for dataset, path_str in model.get("code_outputs", {}).items():
        path = Path(path_str).expanduser()
        for idx, row in enumerate(_iter_cure_code_rows(path)):
            case_id = f"code::{dataset}::{idx:04d}"
            metrics = _code_row_metrics(row)
            output[case_id] = {
                "model_id": model["model_id"],
                "category": dataset,
                "question": row.get("question"),
                "valid": metrics["bon_success"],
                "primary_success_metric": "cure_bon_4x4_success",
                "failure_tags": _code_failure_tags(row=row, metrics=metrics),
                "generated_code_preview": _preview_texts(row.get("generated_code", []), limit=4, chars=900),
                "full_code_generation_preview": _preview_texts(row.get("full_code_generation", []), limit=2, chars=900),
                "full_case_generation_preview": _preview_texts(row.get("full_case_generation", []), limit=2, chars=700),
                "case_input": row.get("case_input", []),
                "case_output": row.get("case_output", []),
                "case_bool_table": row.get("case_bool_table"),
                "test_bool_table": row.get("test_bool_table"),
                "test_input_preview": _preview_texts(row.get("test_input", []), limit=3, chars=400),
                "test_output_preview": _preview_texts(row.get("test_output", []), limit=3, chars=400),
                "code_success_rate": metrics["code_success_rate"],
                "code_success_count": metrics["code_success_count"],
                "code_sample_count": metrics["code_sample_count"],
                "any_code_success": metrics["any_code_success"],
                "hidden_test_acc": metrics["hidden_test_acc"],
                "bon_code_index": metrics["bon_code_index"],
                "bon_success": metrics["bon_success"],
                "bon_hidden_test_acc": metrics["bon_hidden_test_acc"],
                "generated_test_count": metrics["generated_test_count"],
                "hidden_test_count": metrics["hidden_test_count"],
                "output_file": str(path),
            }
    return output


def _iter_cure_code_rows(path: Path):
    if ijson is None:
        yield from _iter_json_array(path)
        return
    with path.open("rb") as f:
        row: dict[str, Any] | None = None
        current_test_bool_row: list[bool] | None = None
        current_case_bool_row: list[bool] | None = None
        for prefix, event, value in ijson.parse(f):
            if prefix == "item" and event == "start_map":
                row = {
                    "generated_code": [],
                    "full_code_generation": [],
                    "full_case_generation": [],
                    "case_input": [],
                    "case_output": [],
                    "test_input": [],
                    "test_output": [],
                    "case_bool_table": [],
                    "test_bool_table": [],
                }
                continue
            if row is None:
                continue
            if prefix == "item" and event == "end_map":
                yield row
                row = None
                continue
            if prefix == "item.question" and event == "string":
                row["question"] = value
            elif prefix == "item.generated_code.item" and event == "string":
                if len(row["generated_code"]) < 4:
                    row["generated_code"].append(value)
            elif prefix == "item.full_code_generation.item" and event == "string":
                if len(row["full_code_generation"]) < 2:
                    row["full_code_generation"].append(value)
            elif prefix == "item.full_case_generation.item" and event == "string":
                if len(row["full_case_generation"]) < 2:
                    row["full_case_generation"].append(value)
            elif prefix == "item.case_input.item" and event == "string":
                if len(row["case_input"]) < 4:
                    row["case_input"].append(value)
            elif prefix == "item.case_output.item" and event == "string":
                if len(row["case_output"]) < 4:
                    row["case_output"].append(value)
            elif prefix == "item.test_input.item" and event == "string":
                if len(row["test_input"]) < 3:
                    row["test_input"].append(value[:400])
            elif prefix == "item.test_output.item" and event == "string":
                if len(row["test_output"]) < 3:
                    row["test_output"].append(value[:400])
            elif prefix == "item.case_bool_table.item" and event == "start_array":
                current_case_bool_row = []
            elif prefix == "item.case_bool_table.item.item" and event == "boolean":
                if current_case_bool_row is not None:
                    current_case_bool_row.append(bool(value))
            elif prefix == "item.case_bool_table.item" and event == "end_array":
                row["case_bool_table"].append(current_case_bool_row or [])
                current_case_bool_row = None
            elif prefix == "item.test_bool_table.item" and event == "start_array":
                current_test_bool_row = []
            elif prefix == "item.test_bool_table.item.item" and event == "boolean":
                if current_test_bool_row is not None:
                    current_test_bool_row.append(bool(value))
            elif prefix == "item.test_bool_table.item" and event == "end_array":
                row["test_bool_table"].append(current_test_bool_row or [])
                current_test_bool_row = None


def _code_row_metrics(row: dict[str, Any]) -> dict[str, Any]:
    test_table = _bool_table(row.get("test_bool_table"))
    case_table = _bool_table(row.get("case_bool_table"))
    code_sample_count = len(test_table)
    hidden_test_count = len(test_table[0]) if test_table else 0
    correct_indices = [
        idx for idx, values in enumerate(test_table)
        if values and all(values)
    ]
    total_cells = sum(len(values) for values in test_table)
    correct_cells = sum(1 for values in test_table for item in values if item)
    bon_idx = _cure_bon_index(case_table, max_code=4, max_case=4)
    bon_row = test_table[bon_idx] if bon_idx is not None and bon_idx < len(test_table) else []
    generated_test_count = len(case_table[0]) if case_table else 0
    return {
        "code_sample_count": code_sample_count,
        "hidden_test_count": hidden_test_count,
        "code_success_count": len(correct_indices),
        "code_success_rate": len(correct_indices) / code_sample_count if code_sample_count else None,
        "any_code_success": bool(correct_indices),
        "hidden_test_acc": correct_cells / total_cells if total_cells else None,
        "bon_code_index": bon_idx,
        "bon_success": bool(bon_row and all(bon_row)),
        "bon_hidden_test_acc": sum(1 for item in bon_row if item) / len(bon_row) if bon_row else None,
        "generated_test_count": generated_test_count,
    }


def _bool_table(value: Any) -> list[list[bool]]:
    if not isinstance(value, list):
        return []
    table = []
    for row in value:
        if isinstance(row, list):
            table.append([bool(item) for item in row])
    return table


def _cure_bon_index(case_table: list[list[bool]], *, max_code: int, max_case: int) -> int | None:
    if not case_table:
        return None
    sub = [row[:max_case] for row in case_table[:max_code]]
    if not sub:
        return None
    scores = [sum(1 for item in row if item) for row in sub]
    return max(range(len(scores)), key=lambda idx: scores[idx])


def _preview_list(value: Any, *, limit: int) -> list[Any]:
    return value[:limit] if isinstance(value, list) else []


def _preview_texts(value: Any, *, limit: int, chars: int) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item)[:chars] for item in value[:limit]]


def _load_score_file(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    rows = _iter_jsonl(path)
    if not rows:
        raise SystemExit(f"empty score file: {path}")
    summary = rows[0]
    failures = {row["id"]: row for row in rows[1:]}
    expected_failures = int(summary["total_count"]) - int(summary["correct_count"])
    if expected_failures != len(failures):
        raise SystemExit(
            f"{path}: expected {expected_failures} failure rows but found {len(failures)}"
        )
    return summary, failures


def _build_bfcl_cases(
    prompts: dict[str, dict[str, Any]],
    answers: dict[str, dict[str, Any]],
    models: list[dict[str, Any]],
    model_outputs: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    cases = []
    for case_id in sorted(prompts):
        prompt_info = prompts[case_id]
        prompt = prompt_info["prompt"]
        category = prompt_info["category"]
        answer = answers.get(case_id, {})
        ground_truth = answer.get("ground_truth")
        question_text = _question_text(prompt.get("question"))
        function_names = _function_names(prompt.get("function"))
        model_map = {}
        tags = set()
        for model in models:
            model_id = model["model_id"]
            row = dict(model_outputs[model_id].get(case_id, {}))
            row["failure_tags"] = _failure_tags(
                error_type=row.get("error_type"),
                error=row.get("error"),
                decoded_calls=row.get("decoded_calls"),
                raw_output=row.get("raw_output"),
                ground_truth=ground_truth,
            )
            tags.update(row["failure_tags"])
            model_map[model_id] = row
        cases.append(
            {
                "benchmark": "BFCL",
                "task_family": "tool",
                "case_id": case_id,
                "category": category,
                "is_live": category.startswith("live_"),
                "language": _detect_language(question_text),
                "question": question_text,
                "prompt": prompt,
                "function_names": function_names,
                "num_functions": len(function_names),
                "num_reference_calls": len(ground_truth or []),
                "ground_truth": ground_truth,
                "failure_tags": sorted(tags),
                "models": model_map,
                "prompt_source_file": prompt_info.get("source_file"),
                "answer_source_file": answer.get("source_file"),
            }
        )
    return cases


def _build_code_cases(
    models: list[dict[str, Any]],
    code_outputs: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    case_ids = sorted({case_id for outputs in code_outputs.values() for case_id in outputs})
    cases = []
    for case_id in case_ids:
        dataset = case_id.split("::")[1]
        exemplar = _first_code_row(case_id, code_outputs, models)
        question = str(exemplar.get("question") or "")
        model_map = {}
        tags = set()
        for model in models:
            model_id = model["model_id"]
            row = dict(code_outputs.get(model_id, {}).get(case_id, {}))
            row.setdefault("valid", False)
            row.setdefault("failure_tags", ["missing_code_eval"])
            tags.update(row.get("failure_tags", []))
            model_map[model_id] = row
        cases.append(
            {
                "benchmark": "CURE",
                "task_family": "code",
                "case_id": case_id,
                "category": dataset,
                "is_live": False,
                "language": "code",
                "question": question,
                "prompt": {"question": question},
                "function_names": [],
                "num_functions": 0,
                "num_reference_calls": int(exemplar.get("hidden_test_count") or 0),
                "ground_truth": None,
                "failure_tags": sorted(tags),
                "code_tags": _code_question_tags(question),
                "models": model_map,
                "prompt_source_file": exemplar.get("output_file"),
                "answer_source_file": exemplar.get("output_file"),
            }
        )
    return cases


def _first_code_row(
    case_id: str,
    code_outputs: dict[str, dict[str, dict[str, Any]]],
    models: list[dict[str, Any]],
) -> dict[str, Any]:
    for model in models:
        row = code_outputs.get(model["model_id"], {}).get(case_id)
        if row:
            dataset = case_id.split("::")[1]
            return {
                "question": row.get("question"),
                "hidden_test_count": row.get("hidden_test_count"),
                "output_file": row.get("output_file"),
                "dataset": dataset,
            }
    return {}


def _build_pairwise(cases: list[dict[str, Any]], *, left_id: str, right_id: str) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        left = case["models"][left_id]
        right = case["models"][right_id]
        left_ok = bool(left.get("valid"))
        right_ok = bool(right.get("valid"))
        if left_ok and right_ok:
            status = "both_correct"
        elif left_ok and not right_ok:
            status = "left_correct_right_wrong"
        elif not left_ok and right_ok:
            status = "left_wrong_right_correct"
        else:
            status = "both_wrong"
        rows.append(
            {
                "case_id": case["case_id"],
                "benchmark": case["benchmark"],
                "task_family": case.get("task_family"),
                "category": case["category"],
                "is_live": case["is_live"],
                "language": case["language"],
                "function_names": case["function_names"],
                "num_reference_calls": case["num_reference_calls"],
                "status": status,
                "left_model": left_id,
                "right_model": right_id,
                "left_valid": left_ok,
                "right_valid": right_ok,
                "left_error_type": left.get("error_type"),
                "right_error_type": right.get("error_type"),
                "failure_tags": sorted(set(left.get("failure_tags", [])) | set(right.get("failure_tags", []))),
                "question_preview": case["question"][:240],
            }
        )
    return rows


def _build_metrics(
    cases: list[dict[str, Any]],
    pairwise: list[dict[str, Any]],
    models: list[dict[str, Any]],
    *,
    left_id: str,
    right_id: str,
) -> dict[str, Any]:
    by_model: dict[str, Any] = {}
    for model in models:
        model_id = model["model_id"]
        category_stats: dict[str, dict[str, Any]] = {}
        for category in sorted(set(case["category"] for case in cases)):
            selected = [case for case in cases if case["category"] == category]
            total = len(selected)
            correct = sum(1 for case in selected if case["models"][model_id].get("valid"))
            category_stats[category] = {
                "correct": correct,
                "total": total,
                "accuracy": correct / total if total else None,
            }
        live_cats = ["live_parallel", "live_parallel_multiple"]
        tool_accs = [
            category_stats[name]["accuracy"]
            for name in BFCL_CATEGORIES
            if category_stats.get(name, {}).get("accuracy") is not None
        ]
        live_accs = [
            category_stats[name]["accuracy"]
            for name in live_cats
            if category_stats.get(name, {}).get("accuracy") is not None
        ]
        code_stats = _code_metrics_for_model(cases, model_id)
        by_model[model_id] = {
            "display_name": model.get("display_name", model_id),
            "categories": category_stats,
            "tool_mean": sum(tool_accs) / len(tool_accs) if tool_accs else None,
            "tool_live_mean": sum(live_accs) / len(live_accs) if live_accs else None,
            "code": code_stats,
            "code_mean_bon_acc": _mean([
                item.get("bon_acc")
                for item in code_stats.get("datasets", {}).values()
            ]),
            "code_mean_sample_acc": _mean([
                item.get("sample_acc")
                for item in code_stats.get("datasets", {}).values()
            ]),
            "tags": model.get("tags", []),
        }

    pair_status = Counter(row["status"] for row in pairwise)
    category_pair_status: dict[str, dict[str, int]] = {}
    for category in sorted(set(row["category"] for row in pairwise)):
        category_pair_status[category] = dict(Counter(row["status"] for row in pairwise if row["category"] == category))
    tag_counts = Counter(tag for row in pairwise for tag in row["failure_tags"])
    language_counts = Counter(case["language"] for case in cases)
    category_counts = Counter(case["category"] for case in cases)
    benchmark_counts = Counter(case["benchmark"] for case in cases)
    return {
        "models": by_model,
        "pair": {"left": left_id, "right": right_id},
        "pair_status_counts": dict(pair_status),
        "category_pair_status_counts": category_pair_status,
        "failure_tag_counts": dict(tag_counts),
        "language_counts": dict(language_counts),
        "category_counts": dict(category_counts),
        "benchmark_counts": dict(benchmark_counts),
        "total_cases": len(cases),
    }


def _code_metrics_for_model(cases: list[dict[str, Any]], model_id: str) -> dict[str, Any]:
    datasets: dict[str, dict[str, Any]] = {}
    for dataset in CODE_DATASETS:
        selected = [
            case for case in cases
            if case["benchmark"] == "CURE" and case["category"] == dataset
        ]
        if not selected:
            continue
        sample_success = 0
        sample_total = 0
        hidden_success = 0.0
        hidden_total = 0
        bon_success = 0
        bon_hidden_values = []
        any_success = 0
        for case in selected:
            row = case["models"][model_id]
            sample_success += int(row.get("code_success_count") or 0)
            sample_total += int(row.get("code_sample_count") or 0)
            hidden_acc = row.get("hidden_test_acc")
            if hidden_acc is not None:
                hidden_success += float(hidden_acc)
                hidden_total += 1
            bon_success += int(bool(row.get("bon_success")))
            bon_hidden = row.get("bon_hidden_test_acc")
            if bon_hidden is not None:
                bon_hidden_values.append(float(bon_hidden))
            any_success += int(bool(row.get("any_code_success")))
        datasets[dataset] = {
            "total": len(selected),
            "sample_acc": sample_success / sample_total if sample_total else None,
            "hidden_test_acc": hidden_success / hidden_total if hidden_total else None,
            "bon_acc": bon_success / len(selected) if selected else None,
            "bon_hidden_test_acc": sum(bon_hidden_values) / len(bon_hidden_values) if bon_hidden_values else None,
            "any_success_acc": any_success / len(selected) if selected else None,
        }
    return {"datasets": datasets}


def _mean(values: list[Any]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None


def _build_calibration_candidates(
    cases: list[dict[str, Any]],
    pairwise: list[dict[str, Any]],
    *,
    left_id: str,
    right_id: str,
) -> list[dict[str, Any]]:
    """Create leakage-safe calibration blueprints from right-model failures.

    This intentionally excludes exact prompt text, ground truth answers, and model outputs.
    The output is a design queue for synthetic same-skill calibration data, not a training set.
    """

    cases_by_id = {case["case_id"]: case for case in cases}
    candidates = []
    for row in pairwise:
        if row["right_valid"]:
            continue
        case = cases_by_id[row["case_id"]]
        tags = row["failure_tags"] or [f"unclassified_{case.get('task_family', 'task')}_failure"]
        if case["benchmark"] == "CURE":
            right_row = case["models"][right_id]
            candidates.append(
                {
                    "format": "cure_code_calibration_candidate_v1",
                    "source_case_id": row["case_id"],
                    "source_category": row["category"],
                    "source_benchmark": row["benchmark"],
                    "source_language": row["language"],
                    "status": row["status"],
                    "target_model": right_id,
                    "reference_model": left_id,
                    "priority": _code_candidate_priority(row, right_row),
                    "failure_tags": tags,
                    "num_reference_tests": case["num_reference_calls"],
                    "code_tags": case.get("code_tags", []),
                    "calibration_goal": _code_candidate_goal(row["category"], tags),
                    "synthetic_requirements": _code_candidate_requirements(row["category"], tags, case.get("code_tags", [])),
                    "target_model_sample_acc": right_row.get("code_success_rate"),
                    "target_model_any_success": right_row.get("any_code_success"),
                    "target_model_bon_hidden_test_acc": right_row.get("bon_hidden_test_acc"),
                    "positive_source_suggestion": "Use code expert / stronger reasoning model BoN rollout, execute hidden-style tests, and keep only verified passing code.",
                    "leakage_policy": "Do not copy official CURE/LiveBench prompt, tests, generated code, or outputs. Generate same-skill synthetic programming tasks with fresh statements and tests.",
                }
            )
            continue
        candidates.append(
            {
                "format": "bfcl_live_calibration_candidate_v1",
                "source_case_id": row["case_id"],
                "source_category": row["category"],
                "source_benchmark": row["benchmark"],
                "source_is_live": row["is_live"],
                "source_language": row["language"],
                "status": row["status"],
                "target_model": right_id,
                "reference_model": left_id,
                "priority": _candidate_priority(row),
                "failure_tags": tags,
                "num_functions": case["num_functions"],
                "num_reference_calls": case["num_reference_calls"],
                "function_name_hints": case["function_names"],
                "calibration_goal": _candidate_goal(row["category"], tags),
                "synthetic_requirements": _candidate_requirements(row["category"], tags, row["language"]),
                "positive_source_suggestion": _positive_source_suggestion(row),
                "leakage_policy": "Do not copy official BFCL prompt, entities, schemas, ground truth, or model output. Generate fresh same-skill synthetic cases only.",
            }
        )
    priority_rank = {"high": 0, "medium": 1, "low": 2}
    return sorted(
        candidates,
        key=lambda item: (
            priority_rank.get(item["priority"], 9),
            not bool(item.get("source_is_live", False)),
            item["source_category"],
            item["source_case_id"],
        ),
    )


def _candidate_priority(row: dict[str, Any]) -> str:
    if row["is_live"] and row["status"] == "left_correct_right_wrong":
        return "high"
    if row["is_live"] or row["status"] == "left_correct_right_wrong":
        return "medium"
    return "low"


def _code_candidate_priority(row: dict[str, Any], right_row: dict[str, Any]) -> str:
    if row["status"] == "left_correct_right_wrong":
        return "high"
    if right_row.get("any_code_success") and not right_row.get("bon_success"):
        return "high"
    if (right_row.get("hidden_test_acc") or 0.0) >= 0.5:
        return "medium"
    return "low"


def _candidate_goal(category: str, tags: list[str]) -> str:
    goals = []
    if category.startswith("live_"):
        goals.append("recover BFCL-live style function-call robustness")
    if "parallel_alignment" in tags or "wrong_count" in tags:
        goals.append("produce the correct number of parallel calls and align each call to one reference intent")
    if "enum_exactness" in tags:
        goals.append("emit exact enum values under schema constraints")
    if "default_value" in tags:
        goals.append("respect schema defaults instead of over-specifying optional parameters")
    if "canonicalization" in tags or "parameter_value_error" in tags:
        goals.append("canonicalize entity and parameter values to scorer-accepted forms")
    if "wrong_function" in tags:
        goals.append("select the right function under distractor schemas")
    if "ast_decode_failed" in tags:
        goals.append("emit parseable tool-call syntax")
    return "; ".join(goals) if goals else "recover failed BFCL tool-call behavior"


def _code_candidate_goal(category: str, tags: list[str]) -> str:
    goals = [f"recover CURE/{category} code generation performance"]
    if "unit_test_selection_failure" in tags:
        goals.append("improve generated-test guided selection so BoN picks a hidden-test-passing code sample")
    if "no_correct_code_sample" in tags:
        goals.append("increase probability of at least one correct code sample under rollout")
    if "partial_unit_test_pass" in tags:
        goals.append("fix edge cases that cause partial hidden-test failures")
    if "code_extraction_failed" in tags:
        goals.append("enforce parseable Python code block output")
    return "; ".join(goals)


def _candidate_requirements(category: str, tags: list[str], language: str) -> list[str]:
    requirements = [
        "use fresh entities and fresh function schemas",
        "include deterministic scorer-compatible possible answers",
        "verify with BFCL scorer or an equivalent exact tool-call checker",
    ]
    if category.endswith("multiple"):
        requirements.append("include multiple turns or multiple tool intents")
    if "parallel" in category:
        requirements.append("include more than one tool call in a single response")
    if category.startswith("live_"):
        requirements.append("include live-style defaults, real-world entities, and schema distractors")
    if language != "en":
        requirements.append(f"include {language} user phrasing while keeping tool arguments canonical")
    if "enum_exactness" in tags:
        requirements.append("include close enum distractors and plural/singular variants")
    if "default_value" in tags:
        requirements.append("include optional parameters with defaults and penalize unnecessary overrides")
    if "canonicalization" in tags:
        requirements.append("include cross-lingual or alias entity names that require canonical output")
    if "wrong_function" in tags:
        requirements.append("include semantically similar functions to test function selection")
    return requirements


def _code_candidate_requirements(category: str, tags: list[str], code_tags: list[str]) -> list[str]:
    requirements = [
        "use fresh programming problems and fresh hidden tests",
        "include executable scorer with per-test pass/fail feedback",
        "store all sampled code, test results, and selected BoN index for audit",
    ]
    if category == "LiveCodeBench":
        requirements.append("include contest-style stdin/stdout tasks with strict formatting")
    if category == "LiveBench":
        requirements.append("include LeetCode-style algorithmic tasks with function-signature discipline")
    if "unit_test_selection_failure" in tags:
        requirements.append("generate counterexample tests that distinguish correct and plausible wrong solutions")
    if "no_correct_code_sample" in tags:
        requirements.append("use expert OPD positives from verified passing code before relying on GRPO only")
    if "partial_unit_test_pass" in tags:
        requirements.append("include edge cases beyond public examples and easy random tests")
    for tag in code_tags[:4]:
        requirements.append(f"cover code skill tag: {tag}")
    return requirements


def _positive_source_suggestion(row: dict[str, Any]) -> str:
    if row["status"] == "left_correct_right_wrong":
        return "Use the reference model family as one positive generator, then verify by scorer."
    return "Use tool expert / stronger model BoN rollout, then keep only scorer-passing trajectories."


def _question_text(question: Any) -> str:
    parts = []
    if isinstance(question, list):
        for turn in question:
            if isinstance(turn, list):
                for item in turn:
                    if isinstance(item, dict) and item.get("content"):
                        parts.append(str(item["content"]))
            elif isinstance(turn, dict) and turn.get("content"):
                parts.append(str(turn["content"]))
    return "\n".join(parts)


def _function_names(functions: Any) -> list[str]:
    if not isinstance(functions, list):
        return []
    return [str(item.get("name")) for item in functions if isinstance(item, dict) and item.get("name")]


def _detect_language(text: str) -> str:
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh"
    if re.search(r"[\uac00-\ud7af]", text):
        return "ko"
    lowered = text.lower()
    spanish_markers = ["¿", "podr", "clima", "condiciones", "actuales", "ciudad"]
    if any(marker in lowered for marker in spanish_markers):
        return "es"
    return "en"


def _failure_tags(
    *,
    error_type: Any,
    error: Any,
    decoded_calls: Any,
    raw_output: Any,
    ground_truth: Any,
) -> list[str]:
    if not error_type and not error:
        return []
    tags = set()
    text = " ".join(_flatten_strings([error_type, error]))
    lower = text.lower()
    if "ast_decoder" in lower or "invalid syntax" in lower:
        tags.add("ast_decode_failed")
    if "wrong_count" in lower:
        tags.add("wrong_count")
    if "cannot_find_match" in lower:
        tags.add("parallel_alignment")
    if "invalid value for parameter" in lower:
        tags.add("parameter_value_error")
    if "expected one of" in lower:
        tags.add("enum_exactness")
    if "location" in lower and "invalid value for parameter" in lower:
        tags.add("canonicalization")
    if "unit" in lower and "invalid value for parameter" in lower:
        tags.add("default_value")
    if "meal_type" in lower or "portion_unit" in lower:
        tags.add("enum_exactness")
    if "function" in lower and ("wrong" in lower or "missing" in lower):
        tags.add("wrong_function")

    decoded_n = len(decoded_calls) if isinstance(decoded_calls, list) else None
    ref_n = len(ground_truth) if isinstance(ground_truth, list) else None
    if decoded_n is not None and ref_n is not None:
        if decoded_n < ref_n:
            tags.add("missing_call")
        elif decoded_n > ref_n:
            tags.add("extra_call")
    raw_text = str(raw_output or "")
    if raw_text and re.search(r"[\u4e00-\u9fff\uac00-\ud7af]", raw_text) and "invalid value" in lower:
        tags.add("canonicalization")
    return sorted(tags)


def _code_failure_tags(*, row: dict[str, Any], metrics: dict[str, Any]) -> list[str]:
    tags = set(_code_question_tags(str(row.get("question") or "")))
    if not metrics.get("any_code_success"):
        tags.add("no_correct_code_sample")
    elif not metrics.get("bon_success"):
        tags.add("unit_test_selection_failure")
    if not metrics.get("bon_success") and (metrics.get("bon_hidden_test_acc") or 0.0) > 0:
        tags.add("partial_unit_test_pass")
    generated_code = row.get("generated_code", [])
    if isinstance(generated_code, list) and any("can not extract" in str(item).lower() for item in generated_code):
        tags.add("code_extraction_failed")
    if (metrics.get("hidden_test_count") or 0) >= 8:
        tags.add("multi_test_hidden_eval")
    return sorted(tags)


def _code_question_tags(text: str) -> list[str]:
    lower = text.lower()
    tags = set()
    keyword_map = {
        "array": ["array", "nums", "list"],
        "string": ["string", "substring", "subsequence", "character"],
        "graph": ["graph", "tree", "node", "edge", "dfs", "bfs"],
        "dynamic_programming": ["dynamic programming", "dp", "memo"],
        "math": ["gcd", "modulo", "prime", "probability", "integer", "math"],
        "greedy": ["minimum", "maximum", "minimize", "maximize", "optimal"],
        "simulation": ["simulate", "operation", "process"],
        "stdin_stdout": ["input", "output", "test case"],
        "format_sensitive": ["print", "output \"yes\"", "output \"no\"", "case-insensitive"],
    }
    for tag, needles in keyword_map.items():
        if any(needle in lower for needle in needles):
            tags.add(tag)
    return sorted(tags)


def _flatten_strings(value: Any) -> list[str]:
    out: list[str] = []
    if value is None:
        return out
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        for key, val in value.items():
            out.extend(_flatten_strings(key))
            out.extend(_flatten_strings(val))
        return out
    if isinstance(value, list):
        for item in value:
            out.extend(_flatten_strings(item))
        return out
    return [str(value)]


if __name__ == "__main__":
    main()
