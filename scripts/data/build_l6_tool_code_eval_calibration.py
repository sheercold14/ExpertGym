#!/usr/bin/env python3
"""Merge BFCL Tool16 and CURE Code16 into one auditable calibration manifest."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from opvec.data.io import read_jsonl, write_jsonl
from opvec.data.schema import validate_rollout_row, validate_seed_record


DEFAULT_BASE_MANIFEST = Path("/tmp/shared-storage/OnPolicy/data/calibration/qbank_c033333_paper96_balanced32_seed20260514.prompts.jsonl")
DEFAULT_TOOL_PROMPTS = Path("/tmp/shared-storage/OnPolicy/data/calibration/20260519_l4_bfcl_tool_aug/bfcl_tool16_nonlive8_live8_seed20260519.prompts.jsonl")
DEFAULT_TOOL_EXPERT = Path("/tmp/shared-storage/OnPolicy/data/calibration/20260519_l4_bfcl_tool_aug/bfcl_tool16_official_answer_expert_rollouts_seed20260519.jsonl")
DEFAULT_CODE_PROMPTS = Path("/tmp/shared-storage/OnPolicy/data/calibration/20260519_l5_cure_eval_code16/cure_eval_code16_livebench8_livecodebench8_seed20260519.prompts.jsonl")
DEFAULT_CODE_EXPERT = Path("/tmp/shared-storage/OnPolicy/data/calibration/20260519_l5_cure_eval_code16/cure_eval_code16_expert_success_rollouts_seed20260519.jsonl")
DEFAULT_OUTPUT_DIR = Path("/tmp/shared-storage/OnPolicy/data/calibration/20260519_l6_bfcl_tool16_cure_code16")


def main() -> None:
    args = parse_args()
    created_at = datetime.now(timezone.utc).isoformat()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_rows = _read_seed_rows(Path(args.base_manifest))
    tool_rows = _read_seed_rows(Path(args.tool_prompts))
    code_rows = _read_seed_rows(Path(args.code_prompts))
    tool_expert_rows = _read_rollout_rows(Path(args.tool_expert_rollout))
    code_expert_rows = _read_rollout_rows(Path(args.code_expert_rollout))

    merged_rows = _dedupe_seed_rows(base_rows + tool_rows + code_rows)
    merged_expert_rows = _dedupe_rollout_rows(tool_expert_rows + code_expert_rows)

    prompts_out = output_dir / "qbank_c033333_paper96_plus_bfcl_tool16_cure_code16_seed20260519.prompts.jsonl"
    expert_out = output_dir / "bfcl_tool16_cure_code16_extra_expert_rollouts_seed20260519.jsonl"
    summary_out = output_dir / "qbank_c033333_paper96_plus_bfcl_tool16_cure_code16_seed20260519.summary.json"
    readme_out = output_dir / "README.md"

    write_jsonl(prompts_out, merged_rows)
    write_jsonl(expert_out, merged_expert_rows)

    summary = {
        "format": "l6_tool_code_eval_calibration_v1",
        "created_at": created_at,
        "inputs": {
            "base_manifest": str(Path(args.base_manifest).expanduser().resolve()),
            "tool_prompts": str(Path(args.tool_prompts).expanduser().resolve()),
            "tool_expert_rollout": str(Path(args.tool_expert_rollout).expanduser().resolve()),
            "code_prompts": str(Path(args.code_prompts).expanduser().resolve()),
            "code_expert_rollout": str(Path(args.code_expert_rollout).expanduser().resolve()),
        },
        "outputs": {
            "merged_prompts": str(prompts_out),
            "extra_expert_rollouts": str(expert_out),
            "summary": str(summary_out),
            "readme": str(readme_out),
        },
        "counts": {
            "base_rows": len(base_rows),
            "tool_added_rows": len(tool_rows),
            "code_added_rows": len(code_rows),
            "merged_rows": len(merged_rows),
            "merged_task_counts": dict(Counter(str(row.get("task")) for row in merged_rows)),
            "tool_extra_expert_rows": len(tool_expert_rows),
            "code_extra_expert_rows": len(code_expert_rows),
            "extra_expert_rows": len(merged_expert_rows),
            "extra_expert_task_counts": dict(Counter(str(row.get("task")) for row in merged_expert_rows)),
            "extra_expert_positive_samples": sum(len(row.get("samples") or []) for row in merged_expert_rows),
        },
        "training_intent": {
            "tool": "Add formal BFCL live/non-live tool anchors so Tool does not rely only on paper96 ToolRL source prompts.",
            "code": "Add formal CURE LiveBench/LiveCodeBench anchors with reward aligned to official first-8 tests.",
            "memory": "Leave memory at paper96 32 rows to keep the previous memory anchor unchanged.",
        },
        "audit_note": (
            "The merged prompt file is independent from L4/L5 and does not overwrite them. "
            "The tool expert rollout is an official-answer anchor; the code expert rollout is model-generated verified positives."
        ),
    }
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    readme_out.write_text(_render_readme(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def _read_seed_rows(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    for row in rows:
        validate_seed_record(row)
    return rows


def _read_rollout_rows(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    for row in rows:
        row.setdefault("step", 0)
        validate_rollout_row(row)
    return rows


def _dedupe_seed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    seen: set[str] = set()
    for row in rows:
        key = str(row.get("prompt_id") or "")
        if not key:
            raise ValueError("Seed row without prompt_id")
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def _dedupe_rollout_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (str(row.get("prompt_id") or ""), str(row.get("policy_id") or ""))
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def _render_readme(summary: dict[str, Any]) -> str:
    counts = summary["counts"]
    lines = [
        "# L6 Tool16 + Code16 Eval Calibration",
        "",
        f"生成时间：`{summary['created_at']}`",
        "",
        "## 目的",
        "",
        "把 L4 的正式 BFCL Tool16 和 L5 的正式 CURE Code16 同时加入 paper96 calibration，形成一个不覆盖旧数据的合并版实验入口。",
        "",
        "## 计数",
        "",
        f"- base rows: `{counts['base_rows']}`",
        f"- tool added rows: `{counts['tool_added_rows']}`",
        f"- code added rows: `{counts['code_added_rows']}`",
        f"- merged rows: `{counts['merged_rows']}`",
        f"- merged task counts: `{counts['merged_task_counts']}`",
        f"- extra expert rows: `{counts['extra_expert_rows']}`",
        f"- extra positive samples: `{counts['extra_expert_positive_samples']}`",
        "",
        "## 输出",
        "",
    ]
    for key, path in summary["outputs"].items():
        lines.append(f"- `{key}`: `{path}`")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, default=DEFAULT_BASE_MANIFEST)
    parser.add_argument("--tool-prompts", type=Path, default=DEFAULT_TOOL_PROMPTS)
    parser.add_argument("--tool-expert-rollout", type=Path, default=DEFAULT_TOOL_EXPERT)
    parser.add_argument("--code-prompts", type=Path, default=DEFAULT_CODE_PROMPTS)
    parser.add_argument("--code-expert-rollout", type=Path, default=DEFAULT_CODE_EXPERT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    main()
