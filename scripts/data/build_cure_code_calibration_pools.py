#!/usr/bin/env python3
"""Build CURE-style Code calibration blueprints from case-level eval results.

The output is an audit/candidate pool, not a leakage-safe training set.  By
default it stores only previews and tags so follow-up data generation can make
fresh programming tasks with the same failure pattern.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_CASES = "/tmp/shared-storage/OnPolicy/analysis/eval_case_browser/cases.jsonl"


def main() -> None:
    args = parse_args()
    cases_path = Path(args.cases).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = [row for row in _iter_jsonl(cases_path) if row.get("benchmark") == "CURE"]
    strong_models = _unique([item.strip() for item in args.strong_model if item.strip()])
    target_model = str(args.target_model)
    generation, selection, partial = build_pools(
        cases,
        target_model=target_model,
        strong_models=strong_models,
        include_question=bool(args.include_question),
    )
    generation = _limit_rows(generation, int(args.generation_limit))
    selection = _limit_rows(selection, int(args.selection_limit))
    partial = _limit_rows(partial, int(args.partial_limit))
    prompts = _interleave([generation, selection, partial], limit=int(args.prompt_limit))

    _write_jsonl(out_dir / "code_generation_pool.jsonl", generation)
    _write_jsonl(out_dir / "code_selection_pool.jsonl", selection)
    _write_jsonl(out_dir / "code_partial_edge_pool.jsonl", partial)
    _write_jsonl(out_dir / "code_calibration_blueprints.jsonl", prompts)
    summary = summarize(
        cases=cases,
        generation=generation,
        selection=selection,
        partial=partial,
        prompts=prompts,
        target_model=target_model,
        strong_models=strong_models,
        cases_path=cases_path,
        out_dir=out_dir,
        include_question=bool(args.include_question),
    )
    _write_json(out_dir / "summary.json", summary)
    _write_text(out_dir / "README.md", render_readme(summary))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def build_pools(
    cases: list[dict[str, Any]],
    *,
    target_model: str,
    strong_models: list[str],
    include_question: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    generation = []
    selection = []
    partial = []
    for case in cases:
        models = case.get("models") or {}
        target = models.get(target_model) or {}
        if not target:
            continue
        positives = _positive_models(models, strong_models)
        base = _base_candidate(case, target_model=target_model, target=target, positives=positives, include_question=include_question)
        if _is_generation_failure(target) and positives:
            generation.append(
                {
                    **base,
                    "pool": "generation",
                    "failure_type": "no_correct_code_sample",
                    "training_intent": "OPD/NLL on verified passing code from stronger models; target is higher sample acc and any-pass.",
                    "priority_score": _priority(case, target, positives, pool="generation"),
                }
            )
        if _is_selection_failure(target):
            selection.append(
                {
                    **base,
                    "pool": "selection",
                    "failure_type": "unit_test_selection_failure",
                    "training_intent": "Train generated tests / selection behavior so BoN chooses an already-present passing code sample.",
                    "priority_score": _priority(case, target, positives, pool="selection"),
                }
            )
        if _is_partial_failure(target):
            partial.append(
                {
                    **base,
                    "pool": "partial_edge",
                    "failure_type": "partial_unit_test_pass",
                    "training_intent": "Generate fresh hidden-edge tests and verified solutions for boundary cases.",
                    "priority_score": _priority(case, target, positives, pool="partial_edge"),
                }
            )
    return (
        sorted(generation, key=lambda row: (-row["priority_score"], row["case_id"])),
        sorted(selection, key=lambda row: (-row["priority_score"], row["case_id"])),
        sorted(partial, key=lambda row: (-row["priority_score"], row["case_id"])),
    )


def summarize(
    *,
    cases: list[dict[str, Any]],
    generation: list[dict[str, Any]],
    selection: list[dict[str, Any]],
    partial: list[dict[str, Any]],
    prompts: list[dict[str, Any]],
    target_model: str,
    strong_models: list[str],
    cases_path: Path,
    out_dir: Path,
    include_question: bool,
) -> dict[str, Any]:
    return {
        "format": "cure_code_calibration_blueprints_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cases": str(cases_path),
        "output_dir": str(out_dir),
        "target_model": target_model,
        "strong_models": strong_models,
        "include_question": include_question,
        "counts": {
            "cure_cases": len(cases),
            "generation_pool": len(generation),
            "selection_pool": len(selection),
            "partial_edge_pool": len(partial),
            "blueprints": len(prompts),
        },
        "by_dataset": {
            "generation": _count_by(generation, "dataset"),
            "selection": _count_by(selection, "dataset"),
            "partial_edge": _count_by(partial, "dataset"),
            "blueprints": _count_by(prompts, "dataset"),
        },
        "top_tags": {
            "generation": _top_tags(generation),
            "selection": _top_tags(selection),
            "partial_edge": _top_tags(partial),
        },
        "outputs": {
            "generation": str(out_dir / "code_generation_pool.jsonl"),
            "selection": str(out_dir / "code_selection_pool.jsonl"),
            "partial_edge": str(out_dir / "code_partial_edge_pool.jsonl"),
            "blueprints": str(out_dir / "code_calibration_blueprints.jsonl"),
            "summary": str(out_dir / "summary.json"),
        },
        "leakage_policy": (
            "Do not train directly on official CURE/LiveBench/LiveCodeBench prompts or hidden tests. "
            "Use these rows as failure-pattern blueprints to generate fresh tasks, fresh tests, and verified solutions."
        ),
    }


def render_readme(summary: dict[str, Any]) -> str:
    counts = summary["counts"]
    lines = [
        "# CURE Code Calibration Blueprints",
        "",
        f"生成时间：`{summary['created_at']}`",
        "",
        "## 目的",
        "",
        "把正式 CURE 评测失败拆成可优化的 Code calibration 候选：generation、selection、partial edge。默认输出是 blueprint，不是可直接训练的官方评测题。",
        "",
        "## 计数",
        "",
        "| pool | rows |",
        "|---|---:|",
        f"| generation | {counts['generation_pool']} |",
        f"| selection | {counts['selection_pool']} |",
        f"| partial_edge | {counts['partial_edge_pool']} |",
        f"| mixed blueprints | {counts['blueprints']} |",
        "",
        "## 使用原则",
        "",
        "- `generation`: 当前模型没有任何通过 hidden tests 的 code sample，适合 expert-verified code OPD。",
        "- `selection`: 当前模型已有正确 code sample，但 BoN 选错，适合 generated-test / selector 修复。",
        "- `partial_edge`: 代码过部分 hidden tests，适合构造 fresh edge-case tests。",
        "- 不要直接把官方题面、hidden tests 或官方生成结果混入训练；应按 tag 生成新题和新测试。",
        "",
        "## 输出",
        "",
    ]
    for key, path in summary["outputs"].items():
        lines.append(f"- `{key}`: `{path}`")
    return "\n".join(lines) + "\n"


def _base_candidate(
    case: dict[str, Any],
    *,
    target_model: str,
    target: dict[str, Any],
    positives: list[dict[str, Any]],
    include_question: bool,
) -> dict[str, Any]:
    row = {
        "case_id": case["case_id"],
        "dataset": case["category"],
        "code_tags": case.get("code_tags", []),
        "failure_tags": target.get("failure_tags", case.get("failure_tags", [])),
        "question_preview": str(case.get("question") or "")[:360],
        "target_model": target_model,
        "target_metrics": _code_metrics(target),
        "positive_models": positives,
        "num_reference_tests": case.get("num_reference_calls"),
        "prompt_source_file": case.get("prompt_source_file"),
        "leakage_policy": "Use as blueprint only; generate fresh task statement/tests before training.",
    }
    if include_question:
        row["question"] = case.get("question")
    return row


def _positive_models(models: dict[str, dict[str, Any]], strong_models: list[str]) -> list[dict[str, Any]]:
    positives = []
    for model_id in strong_models:
        row = models.get(model_id) or {}
        if row.get("any_code_success") or row.get("bon_success"):
            positives.append({"model_id": model_id, **_code_metrics(row)})
    return positives


def _code_metrics(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "bon_success": bool(row.get("bon_success")),
        "any_code_success": bool(row.get("any_code_success")),
        "code_success_count": int(row.get("code_success_count") or 0),
        "code_sample_count": int(row.get("code_sample_count") or 0),
        "code_success_rate": row.get("code_success_rate"),
        "bon_hidden_test_acc": row.get("bon_hidden_test_acc"),
        "hidden_test_acc": row.get("hidden_test_acc"),
        "bon_code_index": row.get("bon_code_index"),
    }


def _is_generation_failure(target: dict[str, Any]) -> bool:
    tags = set(target.get("failure_tags") or [])
    return (not bool(target.get("any_code_success"))) or "no_correct_code_sample" in tags


def _is_selection_failure(target: dict[str, Any]) -> bool:
    tags = set(target.get("failure_tags") or [])
    return bool(target.get("any_code_success")) and not bool(target.get("bon_success")) or "unit_test_selection_failure" in tags


def _is_partial_failure(target: dict[str, Any]) -> bool:
    tags = set(target.get("failure_tags") or [])
    return "partial_unit_test_pass" in tags or (target.get("bon_hidden_test_acc") or 0.0) > 0.0 and not target.get("bon_success")


def _priority(case: dict[str, Any], target: dict[str, Any], positives: list[dict[str, Any]], *, pool: str) -> float:
    score = 0.0
    tags = set(target.get("failure_tags") or []) | set(case.get("code_tags") or [])
    if case.get("category") == "LiveCodeBench":
        score += 1.0
    if "math" in tags:
        score += 0.5
    if "format_sensitive" in tags or "stdin_stdout" in tags:
        score += 0.5
    if "multi_test_hidden_eval" in tags:
        score += 0.5
    if pool == "selection":
        score += 1.5
        score += float(target.get("code_success_count") or 0) * 0.5
    if pool == "generation":
        score += 1.0
        score += len(positives) * 0.5
    if pool == "partial_edge":
        score += float(target.get("bon_hidden_test_acc") or 0.0)
    return score


def _limit_rows(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    return rows if limit < 0 else rows[:limit]


def _interleave(pools: list[list[dict[str, Any]]], *, limit: int) -> list[dict[str, Any]]:
    output = []
    idx = 0
    while any(idx < len(pool) for pool in pools):
        for pool in pools:
            if idx < len(pool):
                output.append(pool[idx])
                if 0 <= limit <= len(output):
                    return output
        idx += 1
    return output


def _count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key)) for row in rows).items()))


def _unique(items: list[str]) -> list[str]:
    seen = set()
    output = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def _top_tags(rows: list[dict[str, Any]], *, limit: int = 20) -> dict[str, int]:
    counter = Counter()
    for row in rows:
        counter.update(row.get("failure_tags") or [])
        counter.update(row.get("code_tags") or [])
    return dict(counter.most_common(limit))


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default=DEFAULT_CASES)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-model", default="expertgym-B-codeaug-opd-i18")
    parser.add_argument("--strong-model", action="append", default=["best-ever-tame-cg-r1calib-global-v2", "ta-c075"])
    parser.add_argument("--generation-limit", type=int, default=-1)
    parser.add_argument("--selection-limit", type=int, default=-1)
    parser.add_argument("--partial-limit", type=int, default=-1)
    parser.add_argument("--prompt-limit", type=int, default=96)
    parser.add_argument("--include-question", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
