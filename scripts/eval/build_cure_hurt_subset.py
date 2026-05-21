#!/usr/bin/env python3
"""Build a small CURE subset where a target model is hurt vs references."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


def parse_reference(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--reference must use name=/path/to/output.json")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("reference name is empty")
    return name, Path(path)


def load_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"{path} row {idx} is not an object")
        rows.append(item)
    return rows


def short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def bool_table_metrics(row: dict[str, Any]) -> dict[str, Any]:
    raw_table = row.get("test_bool_table")
    if not isinstance(raw_table, list):
        return {
            "candidate_count": 0,
            "test_count": 0,
            "pass_count": 0,
            "pass_any": False,
            "pass_rate": 0.0,
            "test_point_rate": 0.0,
        }

    candidates: list[list[bool]] = []
    for candidate in raw_table:
        if not isinstance(candidate, list):
            continue
        candidates.append([bool(value) for value in candidate])

    candidate_count = len(candidates)
    pass_by_candidate = [bool(candidate) and all(candidate) for candidate in candidates]
    pass_count = sum(1 for passed in pass_by_candidate if passed)
    total_tests = sum(len(candidate) for candidate in candidates)
    total_passed_tests = sum(sum(1 for value in candidate if value) for candidate in candidates)
    return {
        "candidate_count": candidate_count,
        "test_count": total_tests,
        "pass_count": pass_count,
        "pass_any": pass_count > 0,
        "pass_rate": pass_count / candidate_count if candidate_count else 0.0,
        "test_point_rate": total_passed_tests / total_tests if total_tests else 0.0,
    }


def validate_alignment(dataset_rows: list[dict[str, Any]], named_rows: dict[str, list[dict[str, Any]]]) -> None:
    expected_len = len(dataset_rows)
    for name, rows in named_rows.items():
        if len(rows) != expected_len:
            raise ValueError(f"{name} length mismatch: got {len(rows)}, expected {expected_len}")
        checked = 0
        for idx, (dataset_row, output_row) in enumerate(zip(dataset_rows, rows, strict=True)):
            dataset_question = str(dataset_row.get("question", ""))
            output_question = str(output_row.get("question", ""))
            if dataset_question and output_question:
                checked += 1
                if dataset_question != output_question:
                    raise ValueError(f"{name} question mismatch at index {idx}")
        if checked == 0:
            raise ValueError(f"{name} has no comparable question field")


def build_cases(
    *,
    dataset_name: str,
    dataset_rows: list[dict[str, Any]],
    target_name: str,
    target_rows: list[dict[str, Any]],
    references: dict[str, list[dict[str, Any]]],
    min_point_delta: float,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for idx, target_row in enumerate(target_rows):
        target_metrics = bool_table_metrics(target_row)
        ref_metrics = {name: bool_table_metrics(rows[idx]) for name, rows in references.items()}

        best_ref_name, best_ref = max(
            ref_metrics.items(),
            key=lambda item: (item[1]["pass_any"], item[1]["test_point_rate"], item[1]["pass_count"]),
        )
        any_ref_pass = any(metrics["pass_any"] for metrics in ref_metrics.values())
        all_models_pass = target_metrics["pass_any"] and all(metrics["pass_any"] for metrics in ref_metrics.values())
        all_models_fail = (not target_metrics["pass_any"]) and not any_ref_pass
        point_delta = best_ref["test_point_rate"] - target_metrics["test_point_rate"]

        strict_hurt = any_ref_pass and not target_metrics["pass_any"]
        soft_hurt = any_ref_pass and point_delta >= min_point_delta and not all_models_pass
        if all_models_pass or all_models_fail or not (strict_hurt or soft_hurt):
            continue

        question = str(dataset_rows[idx].get("question", ""))
        cases.append(
            {
                "dataset": dataset_name,
                "dataset_index": idx,
                "question_hash": short_hash(question),
                "question_preview": " ".join(question.split())[:240],
                "target_name": target_name,
                "target": target_metrics,
                "references": ref_metrics,
                "best_reference_name": best_ref_name,
                "best_reference": best_ref,
                "point_delta": point_delta,
                "hurt_score": (1.0 if strict_hurt else 0.0) + max(0.0, point_delta),
                "reason": "reference_pass_target_fail" if strict_hurt else "reference_point_advantage",
            }
        )

    return sorted(
        cases,
        key=lambda item: (item["hurt_score"], item["point_delta"], item["best_reference"]["pass_count"]),
        reverse=True,
    )


def write_markdown(
    *,
    output_path: Path,
    dataset_name: str,
    subset_name: str,
    target_name: str,
    references: list[str],
    selected_cases: list[dict[str, Any]],
    total_candidates: int,
    subset_json: Path,
    installed_json: Path | None,
) -> None:
    lines = [
        f"# CURE Code Hurt Subset `{subset_name}`",
        "",
        f"- Dataset: `{dataset_name}`",
        f"- Target: `{target_name}`",
        f"- References: {', '.join(f'`{name}`' for name in references)}",
        f"- Selected / candidate hurt cases: `{len(selected_cases)}` / `{total_candidates}`",
        f"- Subset JSON: `{subset_json}`",
    ]
    if installed_json is not None:
        lines.append(f"- Installed CURE dataset: `{installed_json}`")
        lines.append(f"- CURE dataset name: `{installed_json.stem}`")
    lines.extend(
        [
            "",
            "## Rule",
            "",
            "保留参考模型至少一个 hidden-test pass、目标模型未 pass 或测试点明显掉分的题；排除所有模型都 pass 和所有模型都 fail 的题。",
            "",
            "## Cases",
            "",
            "| Rank | Index | Hash | Reason | Best Ref | Target Pass | Ref Pass | Target Test Acc | Ref Test Acc | Delta | Prompt |",
            "| ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for rank, case in enumerate(selected_cases, start=1):
        target = case["target"]
        best_ref = case["best_reference"]
        prompt = str(case["question_preview"]).replace("|", "\\|")
        lines.append(
            "| {rank} | {idx} | `{hash}` | {reason} | `{ref}` | {tpass} | {rpass} | {tacc:.3f} | {racc:.3f} | {delta:.3f} | {prompt} |".format(
                rank=rank,
                idx=case["dataset_index"],
                hash=case["question_hash"],
                reason=case["reason"],
                ref=case["best_reference_name"],
                tpass=target["pass_count"],
                rpass=best_ref["pass_count"],
                tacc=target["test_point_rate"],
                racc=best_ref["test_point_rate"],
                delta=case["point_delta"],
                prompt=prompt,
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--target-output", type=Path, required=True)
    parser.add_argument("--reference", action="append", type=parse_reference, default=[], help="name=/path/to/output.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subset-name", required=True, help="Output dataset stem, e.g. LiveBenchRcrfHurt16")
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--min-point-delta", type=float, default=0.25)
    parser.add_argument("--install-to-cure-data", type=Path, default=None)
    args = parser.parse_args()

    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if not args.reference:
        raise ValueError("At least one --reference is required")

    dataset_rows = load_json_list(args.dataset_json)
    target_rows = load_json_list(args.target_output)
    references = {name: load_json_list(path) for name, path in args.reference}
    validate_alignment(dataset_rows, {args.target_name: target_rows, **references})

    cases = build_cases(
        dataset_name=args.dataset_name,
        dataset_rows=dataset_rows,
        target_name=args.target_name,
        target_rows=target_rows,
        references=references,
        min_point_delta=args.min_point_delta,
    )
    selected = cases[: args.top_k]
    selected_indices = [case["dataset_index"] for case in selected]
    subset_rows = [dataset_rows[idx] for idx in selected_indices]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    subset_json = args.output_dir / f"{args.subset_name}.json"
    cases_json = args.output_dir / f"{args.subset_name}.cases.json"
    report_md = args.output_dir / f"{args.subset_name}.md"
    manifest_json = args.output_dir / f"{args.subset_name}.manifest.json"

    subset_json.write_text(json.dumps(subset_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    cases_json.write_text(json.dumps(selected, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "dataset_name": args.dataset_name,
        "subset_name": args.subset_name,
        "dataset_json": str(args.dataset_json),
        "target_name": args.target_name,
        "target_output": str(args.target_output),
        "references": {name: str(path) for name, path in args.reference},
        "top_k": args.top_k,
        "min_point_delta": args.min_point_delta,
        "candidate_hurt_count": len(cases),
        "selected_count": len(selected),
        "selected_indices": selected_indices,
        "subset_json": str(subset_json),
        "cases_json": str(cases_json),
    }

    installed_json = None
    if args.install_to_cure_data is not None:
        args.install_to_cure_data.mkdir(parents=True, exist_ok=True)
        installed_json = args.install_to_cure_data / f"{args.subset_name}.json"
        shutil.copy2(subset_json, installed_json)
        manifest["installed_cure_dataset_json"] = str(installed_json)
        manifest["installed_cure_dataset_name"] = installed_json.stem

    manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(
        output_path=report_md,
        dataset_name=args.dataset_name,
        subset_name=args.subset_name,
        target_name=args.target_name,
        references=[name for name, _ in args.reference],
        selected_cases=selected,
        total_candidates=len(cases),
        subset_json=subset_json,
        installed_json=installed_json,
    )

    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
