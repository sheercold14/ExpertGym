#!/usr/bin/env python3
"""Build positive/negative span-pair manifests from CURE hurt cases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.build_cure_hurt_case_pack import (
    ability_tags,
    best_candidate,
    load_json,
    row_metrics,
    select_best_reference,
)


def jsonl_write(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def make_probe_row(
    *,
    pair_id: str,
    role: str,
    response_kind: str,
    dataset: str,
    dataset_index: int,
    model_name: str,
    source_row: dict[str, Any],
    candidate: dict[str, Any],
    tags: list[str],
) -> dict[str, Any]:
    if response_kind == "full":
        response = str(candidate.get("full_generation") or candidate.get("generated_code") or "")
    elif response_kind == "code":
        response = str(candidate.get("generated_code") or "")
    else:
        raise ValueError(f"Unsupported response_kind: {response_kind}")
    return {
        "format": "cure_hurt_span_probe_row_v1",
        "task": "code",
        "ability": "code",
        "data_source": "cure_hurt_subset",
        "split": "diagnostic",
        "pair_id": pair_id,
        "prompt_id": f"{dataset.lower()}__{dataset_index:05d}",
        "sample_id": f"{pair_id}__{role}__{response_kind}",
        "dataset": dataset,
        "dataset_index": dataset_index,
        "role": role,
        "response_kind": response_kind,
        "model_name": model_name,
        "ability_tags": tags,
        "prompt": str(source_row.get("question") or ""),
        "rendered_prompt": str(source_row.get("code_generation_prompt") or source_row.get("question") or ""),
        "response": response,
        "generated_code": str(candidate.get("generated_code") or ""),
        "full_generation": str(candidate.get("full_generation") or ""),
        "candidate_index": candidate.get("index"),
        "test_passes": candidate.get("test_passes") or [],
        "passed_tests": int(candidate.get("passed_tests") or 0),
        "test_count": int(candidate.get("test_count") or 0),
        "test_point_rate": float(candidate.get("test_point_rate") or 0.0),
        "pass_all": bool(candidate.get("pass_all")),
    }


def build_span_pair_manifest(*, manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    selected_cases = load_json(Path(str(manifest["cases_json"])))
    target_rows = load_json(Path(str(manifest["target_output"])))
    reference_rows = {name: load_json(Path(path)) for name, path in manifest["references"].items()}

    subset_name = str(manifest["subset_name"])
    dataset = str(manifest["dataset_name"])
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs: list[dict[str, Any]] = []
    positive_full: list[dict[str, Any]] = []
    negative_full: list[dict[str, Any]] = []
    positive_code: list[dict[str, Any]] = []
    negative_code: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []

    for case in selected_cases:
        dataset_index = int(case["dataset_index"])
        pair_id = f"{subset_name}__{dataset_index:05d}"
        target_row = target_rows[dataset_index]
        target_candidate = best_candidate(target_row)
        ref_name, ref_metrics, ref_candidate = select_best_reference(reference_rows, dataset_index)
        tags = ability_tags(str(target_row.get("question") or ""))

        pos_full = make_probe_row(
            pair_id=pair_id,
            role="reference_pass",
            response_kind="full",
            dataset=dataset,
            dataset_index=dataset_index,
            model_name=ref_name,
            source_row=target_row,
            candidate=ref_candidate,
            tags=tags,
        )
        neg_full = make_probe_row(
            pair_id=pair_id,
            role="target_hurt",
            response_kind="full",
            dataset=dataset,
            dataset_index=dataset_index,
            model_name=str(manifest["target_name"]),
            source_row=target_row,
            candidate=target_candidate,
            tags=tags,
        )
        pos_code = make_probe_row(
            pair_id=pair_id,
            role="reference_pass",
            response_kind="code",
            dataset=dataset,
            dataset_index=dataset_index,
            model_name=ref_name,
            source_row=target_row,
            candidate=ref_candidate,
            tags=tags,
        )
        neg_code = make_probe_row(
            pair_id=pair_id,
            role="target_hurt",
            response_kind="code",
            dataset=dataset,
            dataset_index=dataset_index,
            model_name=str(manifest["target_name"]),
            source_row=target_row,
            candidate=target_candidate,
            tags=tags,
        )
        positive_full.append(pos_full)
        negative_full.append(neg_full)
        positive_code.append(pos_code)
        negative_code.append(neg_code)
        contrast = dict(pos_code)
        contrast.update(
            {
                "format": "cure_hurt_code_contrast_row_v1",
                "negative_response": neg_code["response"],
                "negative_model_name": neg_code["model_name"],
                "negative_candidate_index": neg_code["candidate_index"],
                "negative_passed_tests": neg_code["passed_tests"],
                "negative_test_count": neg_code["test_count"],
                "negative_test_point_rate": neg_code["test_point_rate"],
            }
        )
        contrast_rows.append(contrast)
        pairs.append(
            {
                "format": "cure_hurt_span_pair_v1",
                "pair_id": pair_id,
                "dataset": dataset,
                "dataset_index": dataset_index,
                "question_hash": case.get("question_hash"),
                "ability_tags": tags,
                "question": str(target_row.get("question") or ""),
                "rendered_prompt": str(target_row.get("code_generation_prompt") or target_row.get("question") or ""),
                "target_name": str(manifest["target_name"]),
                "target_metrics": row_metrics(target_row),
                "target_best_candidate": {
                    key: value for key, value in target_candidate.items() if key not in {"full_generation", "generated_code"}
                },
                "best_reference_name": ref_name,
                "best_reference_metrics": ref_metrics,
                "best_reference_candidate": {
                    key: value for key, value in ref_candidate.items() if key not in {"full_generation", "generated_code"}
                },
                "positive_full_sample_id": pos_full["sample_id"],
                "negative_full_sample_id": neg_full["sample_id"],
                "positive_code_sample_id": pos_code["sample_id"],
                "negative_code_sample_id": neg_code["sample_id"],
            }
        )

    outputs = {
        "pairs_jsonl": output_dir / f"{subset_name}.span_pairs.jsonl",
        "positive_full_jsonl": output_dir / f"{subset_name}.positive_full.jsonl",
        "negative_full_jsonl": output_dir / f"{subset_name}.negative_full.jsonl",
        "positive_code_jsonl": output_dir / f"{subset_name}.positive_code.jsonl",
        "negative_code_jsonl": output_dir / f"{subset_name}.negative_code.jsonl",
        "contrast_code_jsonl": output_dir / f"{subset_name}.contrast_code.jsonl",
    }
    jsonl_write(outputs["pairs_jsonl"], pairs)
    jsonl_write(outputs["positive_full_jsonl"], positive_full)
    jsonl_write(outputs["negative_full_jsonl"], negative_full)
    jsonl_write(outputs["positive_code_jsonl"], positive_code)
    jsonl_write(outputs["negative_code_jsonl"], negative_code)
    jsonl_write(outputs["contrast_code_jsonl"], contrast_rows)

    summary = {
        "format": "cure_hurt_span_pair_manifest_summary_v1",
        "subset_name": subset_name,
        "dataset_name": dataset,
        "source_manifest": str(manifest_path),
        "case_count": len(pairs),
        "outputs": {name: str(path) for name, path in outputs.items()},
        "intended_probes": {
            "prompt_span": "Use positive_full/negative_full with probe_signed_utility.py --span prompt.",
            "reasoning_span": "Use positive_full/negative_full with --span reasoning; inspect because CURE generations may not always contain fenced code.",
            "code_span": "Use positive_code/negative_code with --span response to isolate extracted executable code.",
            "contrast": "Use contrast_code_jsonl for pass-vs-fail code residual comparisons.",
        },
    }
    summary_path = output_dir / f"{subset_name}.span_pair_summary.json"
    summary["summary_json"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(output_dir / f"{subset_name}.span_pair_summary.md", summary)
    return summary


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        f"# CURE Hurt Span Pair Manifest `{summary['subset_name']}`",
        "",
        f"- Dataset: `{summary['dataset_name']}`",
        f"- Cases: `{summary['case_count']}`",
        f"- Source manifest: `{summary['source_manifest']}`",
        "",
        "## Files",
        "",
        "| file | purpose |",
        "| --- | --- |",
    ]
    purpose = {
        "pairs_jsonl": "pair-level metadata; no duplicated full code text beyond row references",
        "positive_full_jsonl": "reference successful full generations for prompt/reasoning-span probe",
        "negative_full_jsonl": "target hurt full generations for prompt/reasoning-span probe",
        "positive_code_jsonl": "reference successful extracted code for final-code-span probe",
        "negative_code_jsonl": "target hurt extracted code for final-code-span probe",
        "contrast_code_jsonl": "same-prompt pass/fail code rows with `negative_response`",
    }
    for name, output in summary["outputs"].items():
        lines.append(f"| `{output}` | {purpose.get(name, '')} |")
    lines.extend(
        [
            "",
            "## Probe Commands",
            "",
            "Prompt-span positive utility:",
            "",
            "```bash",
            "PYTHONDONTWRITEBYTECODE=1 python scripts/attention_pauh/probe_signed_utility.py \\",
            "  --mode-manifest /tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json \\",
            f"  --trajectory-jsonl {summary['outputs']['positive_full_jsonl']} \\",
            "  --output-dir /tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/probes/<run_id> \\",
            "  --tasks code --experts tool,memory,code --scope all-linear \\",
            f"  --samples-per-task {summary['case_count']} --span prompt --write-row-details",
            "```",
            "",
            "Code-span positive utility:",
            "",
            "```bash",
            "PYTHONDONTWRITEBYTECODE=1 python scripts/attention_pauh/probe_signed_utility.py \\",
            "  --mode-manifest /tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json \\",
            f"  --trajectory-jsonl {summary['outputs']['positive_code_jsonl']} \\",
            "  --output-dir /tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/probes/<run_id> \\",
            "  --tasks code --experts tool,memory,code --scope all-linear \\",
            f"  --samples-per-task {summary['case_count']} --span response --write-row-details",
            "```",
            "",
            "Run the same commands on `negative_full_jsonl` / `negative_code_jsonl` to get fail-trajectory utility, then compare positive minus negative by `pair_id`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = build_span_pair_manifest(manifest_path=args.manifest, output_dir=args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
