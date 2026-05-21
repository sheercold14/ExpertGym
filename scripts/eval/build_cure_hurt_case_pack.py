#!/usr/bin/env python3
"""Build a readable case pack for CURE hurt subsets."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def bools(values: Any) -> list[bool]:
    if not isinstance(values, list):
        return []
    return [bool(value) for value in values]


def candidate_rate(values: list[bool]) -> float:
    return sum(1 for value in values if value) / len(values) if values else 0.0


def best_candidate(row: dict[str, Any]) -> dict[str, Any]:
    table = row.get("test_bool_table")
    if not isinstance(table, list):
        table = []
    code_list = row.get("generated_code")
    if not isinstance(code_list, list):
        code_list = []
    full_list = row.get("full_code_generation")
    if not isinstance(full_list, list):
        full_list = []

    candidates: list[dict[str, Any]] = []
    for idx, raw_passes in enumerate(table):
        passes = bools(raw_passes)
        candidates.append(
            {
                "index": idx,
                "test_passes": passes,
                "passed_tests": sum(1 for value in passes if value),
                "test_count": len(passes),
                "test_point_rate": candidate_rate(passes),
                "pass_all": bool(passes) and all(passes),
                "generated_code": code_list[idx] if idx < len(code_list) else "",
                "full_generation": full_list[idx] if idx < len(full_list) else "",
            }
        )
    if not candidates:
        return {
            "index": None,
            "test_passes": [],
            "passed_tests": 0,
            "test_count": 0,
            "test_point_rate": 0.0,
            "pass_all": False,
            "generated_code": "",
            "full_generation": "",
        }
    return max(candidates, key=lambda item: (item["pass_all"], item["test_point_rate"], item["passed_tests"]))


def row_metrics(row: dict[str, Any]) -> dict[str, Any]:
    table = row.get("test_bool_table")
    if not isinstance(table, list):
        table = []
    candidates = [bools(raw) for raw in table if isinstance(raw, list)]
    pass_all = [bool(candidate) and all(candidate) for candidate in candidates]
    total_tests = sum(len(candidate) for candidate in candidates)
    total_passed = sum(sum(1 for value in candidate if value) for candidate in candidates)
    best = best_candidate(row)
    return {
        "candidate_count": len(candidates),
        "test_count": total_tests,
        "pass_count": sum(1 for value in pass_all if value),
        "pass_any": any(pass_all),
        "test_point_rate": total_passed / total_tests if total_tests else 0.0,
        "best_candidate": {
            key: value
            for key, value in best.items()
            if key not in {"generated_code", "full_generation"}
        },
    }


ABILITY_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("bitwise", (" bit", "xor", " x-or", " k-or", "binary representation", "set bit")),
    ("grid", ("grid", "row", "column", "cell", "matrix")),
    ("string", ("string", "substring", "subsequence", "lexicographically", "characters", "lowercase", "prefix", "suffix")),
    ("dp_counting", ("number of ways", "modulo", "return the number", "count the number", "digit_sum", "longest subsequence", "partition")),
    ("greedy_operations", ("operation", "swap", "remove", "choose", "at most", "minimum", "maximum", "minimize", "maximize")),
    ("math_set", ("subset", "set", "frequency", "permutation", "combination", "digits", "sum")),
]


def ability_tags(question: str) -> list[str]:
    text = " " + question.lower()
    tags = [name for name, patterns in ABILITY_PATTERNS if any(pattern in text for pattern in patterns)]
    return tags or ["unclassified"]


def code_signature(code: str) -> dict[str, Any]:
    lines = [line for line in code.splitlines() if line.strip()]
    return {
        "nonempty_lines": len(lines),
        "has_def": bool(re.search(r"^\s*def\s+", code, flags=re.M)),
        "has_stdin": "input(" in code or "sys.stdin" in code,
        "has_print": "print(" in code,
        "has_mod": "%" in code or "mod" in code.lower(),
        "imports": sorted(set(re.findall(r"^\s*(?:from\s+(\S+)\s+import|import\s+(\S+))", code, flags=re.M))),
    }


def compact_code(code: str, max_chars: int) -> str:
    code = code.strip()
    if len(code) <= max_chars:
        return code
    return code[:max_chars].rstrip() + "\n# ... truncated ..."


def select_best_reference(model_rows: dict[str, list[dict[str, Any]]], dataset_index: int) -> tuple[str, dict[str, Any], dict[str, Any]]:
    scored: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for name, rows in model_rows.items():
        row = rows[dataset_index]
        metrics = row_metrics(row)
        candidate = best_candidate(row)
        scored.append((name, metrics, candidate))
    return max(scored, key=lambda item: (item[1]["pass_any"], item[1]["test_point_rate"], item[1]["pass_count"]))


def build_pack(
    *,
    manifest_path: Path,
    output_dir: Path,
    max_code_chars: int,
) -> tuple[dict[str, Any], str]:
    manifest = load_json(manifest_path)
    cases_path = Path(str(manifest["cases_json"]))
    selected_cases = load_json(cases_path)
    target_rows = load_json(Path(str(manifest["target_output"])))
    reference_rows = {name: load_json(Path(path)) for name, path in manifest["references"].items()}

    packed_cases: list[dict[str, Any]] = []
    tag_counts: dict[str, int] = {}
    failure_buckets: dict[str, int] = {}
    for rank, case in enumerate(selected_cases, start=1):
        idx = int(case["dataset_index"])
        target_row = target_rows[idx]
        target_metrics = row_metrics(target_row)
        target_best = best_candidate(target_row)
        ref_name, ref_metrics, ref_best = select_best_reference(reference_rows, idx)
        tags = ability_tags(str(target_row.get("question", "")))
        for tag in tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

        if target_best["test_point_rate"] == 0.0:
            failure_bucket = "target_zero_hidden_tests"
        elif target_best["test_point_rate"] < 0.5:
            failure_bucket = "target_low_partial"
        else:
            failure_bucket = "target_near_miss"
        failure_buckets[failure_bucket] = failure_buckets.get(failure_bucket, 0) + 1

        packed_cases.append(
            {
                "rank": rank,
                "dataset_index": idx,
                "question_hash": case.get("question_hash"),
                "question": target_row.get("question", ""),
                "ability_tags": tags,
                "failure_bucket": failure_bucket,
                "hurt_score": case.get("hurt_score"),
                "point_delta": case.get("point_delta"),
                "target": {
                    "name": manifest["target_name"],
                    "metrics": target_metrics,
                    "best_candidate": target_best,
                    "code_signature": code_signature(str(target_best.get("generated_code", ""))),
                },
                "best_reference": {
                    "name": ref_name,
                    "metrics": ref_metrics,
                    "best_candidate": ref_best,
                    "code_signature": code_signature(str(ref_best.get("generated_code", ""))),
                },
                "all_references": {
                    name: row_metrics(rows[idx]) for name, rows in reference_rows.items()
                },
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    pack_json = output_dir / f"{manifest['subset_name']}.case_pack.json"
    pack_md = output_dir / f"{manifest['subset_name']}.case_pack.md"
    summary = {
        "subset_name": manifest["subset_name"],
        "dataset_name": manifest["dataset_name"],
        "manifest": str(manifest_path),
        "case_count": len(packed_cases),
        "tag_counts": dict(sorted(tag_counts.items())),
        "failure_buckets": dict(sorted(failure_buckets.items())),
        "cases": packed_cases,
        "pack_json": str(pack_json),
        "pack_md": str(pack_md),
    }
    pack_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        f"# CURE Hurt Case Pack `{manifest['subset_name']}`",
        "",
        f"- Dataset: `{manifest['dataset_name']}`",
        f"- Target: `{manifest['target_name']}`",
        f"- Cases: `{len(packed_cases)}`",
        f"- Source manifest: `{manifest_path}`",
        f"- JSON pack: `{pack_json}`",
        "",
        "## Ability Buckets",
        "",
        "| tag | count |",
        "| --- | ---: |",
    ]
    for tag, count in sorted(tag_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{tag}` | {count} |")
    lines.extend(["", "## Failure Buckets", "", "| bucket | count |", "| --- | ---: |"])
    for bucket, count in sorted(failure_buckets.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{bucket}` | {count} |")
    lines.extend(
        [
            "",
            "## Case Index",
            "",
            "| Rank | Index | Tags | Failure | Ref | Target best | Ref best | Delta | Question |",
            "| ---: | ---: | --- | --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for item in packed_cases:
        target_best = item["target"]["best_candidate"]
        ref_best = item["best_reference"]["best_candidate"]
        question = " ".join(str(item["question"]).split())[:180].replace("|", "\\|")
        lines.append(
            "| {rank} | {idx} | {tags} | `{failure}` | `{ref}` | {tp}/{tc} | {rp}/{rc} | {delta:.3f} | {question} |".format(
                rank=item["rank"],
                idx=item["dataset_index"],
                tags=", ".join(f"`{tag}`" for tag in item["ability_tags"]),
                failure=item["failure_bucket"],
                ref=item["best_reference"]["name"],
                tp=target_best["passed_tests"],
                tc=target_best["test_count"],
                rp=ref_best["passed_tests"],
                rc=ref_best["test_count"],
                delta=float(item.get("point_delta") or 0.0),
                question=question,
            )
        )

    lines.extend(["", "## Representative Code Pairs", ""])
    for item in packed_cases[:8]:
        target_best = item["target"]["best_candidate"]
        ref_best = item["best_reference"]["best_candidate"]
        lines.extend(
            [
                f"### Rank {item['rank']} / Index {item['dataset_index']} / `{item['question_hash']}`",
                "",
                f"- Tags: {', '.join(f'`{tag}`' for tag in item['ability_tags'])}",
                f"- Target best hidden tests: `{target_best['passed_tests']}/{target_best['test_count']}`",
                f"- Reference `{item['best_reference']['name']}` best hidden tests: `{ref_best['passed_tests']}/{ref_best['test_count']}`",
                "",
                "Question:",
                "",
                str(item["question"]).strip()[:1200],
                "",
                "Target best code:",
                "",
                "```python",
                compact_code(str(target_best.get("generated_code", "")), max_code_chars),
                "```",
                "",
                "Reference best code:",
                "",
                "```python",
                compact_code(str(ref_best.get("generated_code", "")), max_code_chars),
                "```",
                "",
            ]
        )
    pack_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary, str(pack_md)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-code-chars", type=int, default=2400)
    args = parser.parse_args()
    summary, _ = build_pack(manifest_path=args.manifest, output_dir=args.output_dir, max_code_chars=args.max_code_chars)
    print(
        json.dumps(
            {
                "subset_name": summary["subset_name"],
                "case_count": summary["case_count"],
                "tag_counts": summary["tag_counts"],
                "failure_buckets": summary["failure_buckets"],
                "pack_json": summary["pack_json"],
                "pack_md": summary["pack_md"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
