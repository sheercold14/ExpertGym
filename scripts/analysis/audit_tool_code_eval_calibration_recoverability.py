#!/usr/bin/env python3
"""Audit model-positive recoverability for the added Tool16 and Code16 rows."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from opvec.data.io import read_jsonl, write_jsonl
from opvec.rewards.bfcl import BFCLToolRewardAdapter


DEFAULT_TOOL_PROMPTS = Path("/tmp/shared-storage/OnPolicy/data/calibration/20260519_l4_bfcl_tool_aug/bfcl_tool16_nonlive8_live8_seed20260519.prompts.jsonl")
DEFAULT_CODE_EXPERT = Path("/tmp/shared-storage/OnPolicy/data/calibration/20260519_l5_cure_eval_code16/cure_eval_code16_expert_success_rollouts_seed20260519.jsonl")
DEFAULT_CODE_SUMMARY = Path("/tmp/shared-storage/OnPolicy/data/calibration/20260519_l5_cure_eval_code16/qbank_c033333_paper96_plus_cure_code16_seed20260519.summary.json")
DEFAULT_BFCL_RESULT_ROOT = Path("/mnt/cache/wuruixiao/users/lsc/gorilla/berkeley-function-call-leaderboard/result")
DEFAULT_OUTPUT_DIR = Path("/tmp/shared-storage/OnPolicy/data/calibration/20260519_l6_bfcl_tool16_cure_code16/audit")
TOOL_RESULT_DIRS = {
    "qwen25_instruct": "qwen25-instruct",
    "toolrl": "toolrl",
    "reasonflux": "reasonflux",
    "r1_injected_alpha001": "arm-r-v2-plus-r1-alpha0.001-tool-20260518-r1-inject-eval6-20260518_r1_inject_eval6/arm-r-v2-plus-r1-alpha0.001-tool-20260518-r1-inject-eval6",
}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tool_rows = read_jsonl(args.tool_prompts)
    tool_audit_rows, tool_summary = audit_tool_rows(tool_rows, result_root=Path(args.bfcl_result_root))
    code_summary = audit_code_rows(Path(args.code_expert_rollout), Path(args.code_summary))

    row_path = output_dir / "tool16_model_recoverability_rows.jsonl"
    summary_path = output_dir / "tool_code_eval_calibration_recoverability_summary.json"
    report_path = output_dir / "tool_code_eval_calibration_recoverability_report.md"
    write_jsonl(row_path, tool_audit_rows)
    summary = {
        "format": "tool_code_eval_calibration_recoverability_v1",
        "inputs": {
            "tool_prompts": str(Path(args.tool_prompts).expanduser().resolve()),
            "code_expert_rollout": str(Path(args.code_expert_rollout).expanduser().resolve()),
            "code_summary": str(Path(args.code_summary).expanduser().resolve()),
            "bfcl_result_root": str(Path(args.bfcl_result_root).expanduser().resolve()),
        },
        "outputs": {
            "tool_rows": str(row_path),
            "summary": str(summary_path),
            "report": str(report_path),
        },
        "tool": tool_summary,
        "code": code_summary,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def audit_tool_rows(tool_rows: list[dict[str, Any]], *, result_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    adapter = BFCLToolRewardAdapter()
    result_indices = {
        model: _load_bfcl_result_index(result_root / rel_path)
        for model, rel_path in TOOL_RESULT_DIRS.items()
        if (result_root / rel_path).exists()
    }
    audit_rows = []
    success_by_model: dict[str, int] = defaultdict(int)
    missing_by_model: dict[str, int] = defaultdict(int)
    reward_by_model: dict[str, list[float]] = defaultdict(list)
    recoverable_by_id: dict[str, list[str]] = defaultdict(list)
    for prompt in tool_rows:
        bfcl = prompt["reference"]["bfcl"]
        row_id = str(bfcl["id"])
        category = str(bfcl["category"])
        eval_group = str(bfcl["eval_group"])
        for model, index in result_indices.items():
            result = index.get(category, {}).get(row_id)
            if result is None:
                missing_by_model[model] += 1
                audit_rows.append(
                    {
                        "prompt_id": prompt["prompt_id"],
                        "bfcl_id": row_id,
                        "category": category,
                        "eval_group": eval_group,
                        "model": model,
                        "missing_result": True,
                        "success": False,
                        "reward": 0.0,
                    }
                )
                continue
            score = adapter.score(prompt, str(result.get("result") or "")).as_dict()
            success = bool(score.get("success"))
            reward = float(score.get("reward", 0.0))
            success_by_model[model] += int(success)
            reward_by_model[model].append(reward)
            if success:
                recoverable_by_id[row_id].append(model)
            audit_rows.append(
                {
                    "prompt_id": prompt["prompt_id"],
                    "bfcl_id": row_id,
                    "category": category,
                    "eval_group": eval_group,
                    "model": model,
                    "missing_result": False,
                    "success": success,
                    "reward": reward,
                    "result": str(result.get("result") or ""),
                    "details": score.get("details") or {},
                }
            )
    official_anchor_rows = 16
    by_category = Counter(row["reference"]["bfcl"]["category"] for row in tool_rows)
    by_eval_group = Counter(row["reference"]["bfcl"]["eval_group"] for row in tool_rows)
    model_stats = {}
    for model in sorted(result_indices):
        rewards = reward_by_model.get(model, [])
        model_stats[model] = {
            "available_results": len(rewards),
            "missing_results": int(missing_by_model.get(model, 0)),
            "success_rows": int(success_by_model.get(model, 0)),
            "success_rate": round(success_by_model.get(model, 0) / max(1, len(tool_rows)), 4),
            "mean_reward": round(sum(rewards) / max(1, len(rewards)), 4),
        }
    return audit_rows, {
        "prompt_rows": len(tool_rows),
        "by_category": dict(by_category),
        "by_eval_group": dict(by_eval_group),
        "official_answer_anchor_success_rows": official_anchor_rows,
        "official_answer_anchor_note": "The L4/L6 tool OPD file uses canonical BFCL possible_answer, not a model rollout.",
        "model_stats": model_stats,
        "rows_with_any_model_success": sum(1 for row in tool_rows if recoverable_by_id.get(str(row["reference"]["bfcl"]["id"]))),
        "rows_with_toolrl_success": int(success_by_model.get("toolrl", 0)),
        "pure_r1_result_available": False,
        "pure_r1_result_note": "No standalone DeepSeek-R1-Distill-Qwen-7B BFCL result directory was found locally; only an R1-injected merged model result is present.",
    }


def audit_code_rows(code_expert_path: Path, code_summary_path: Path) -> dict[str, Any]:
    expert_rows = read_jsonl(code_expert_path)
    summary = json.loads(code_summary_path.read_text(encoding="utf-8"))
    rows = summary.get("selection", {}).get("rows", [])
    positive_rows_by_expert = Counter()
    positive_samples_by_expert = Counter()
    for row in rows:
        for expert in row.get("positive_experts") or []:
            positive_rows_by_expert[expert] += 1
    for row in expert_rows:
        for sample in row.get("samples") or []:
            expert = str(sample.get("opd_source_policy_id") or sample.get("details", {}).get("expert_name") or "unknown")
            positive_samples_by_expert[expert] += 1
    return {
        "prompt_rows": len(rows),
        "expert_rollout_rows": len(expert_rows),
        "positive_samples": sum(len(row.get("samples") or []) for row in expert_rows),
        "base_success_rows": sum(1 for row in rows if int(row.get("base_success_count", 0)) > 0),
        "rows_with_any_expert_success": sum(1 for row in rows if row.get("positive_experts")),
        "positive_rows_by_expert": dict(sorted(positive_rows_by_expert.items())),
        "positive_samples_by_expert": dict(sorted(positive_samples_by_expert.items())),
        "r1_positive_rows": int(positive_rows_by_expert.get("deepseek_r1_distill", 0)),
        "r1_positive_samples": int(positive_samples_by_expert.get("deepseek_r1_distill", 0)),
        "note": "Code16 positives are real model generations from formal CURE temp outputs and re-verified by local CodeRewardAdapter.",
    }


def _load_bfcl_result_index(model_result_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    output: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for group in ("live", "non_live"):
        group_dir = model_result_dir / group
        if not group_dir.exists():
            continue
        for path in group_dir.glob("BFCL_v4_*_result.json"):
            category = path.name.removeprefix("BFCL_v4_").removesuffix("_result.json")
            for row in read_jsonl(path):
                row_id = str(row.get("id") or "")
                if row_id:
                    output[category][row_id] = row
    return output


def render_report(summary: dict[str, Any]) -> str:
    tool = summary["tool"]
    code = summary["code"]
    lines = [
        "# Tool16 + Code16 Calibration Recoverability Audit",
        "",
        "## 结论",
        "",
        "- Tool16 已覆盖 BFCL non-live 8 条、live 8 条，但当前 OPD 正样本是官方 possible_answer，不是模型专家真实 rollout。",
        f"- ToolRL 在 Tool16 上真实做对 `{tool['rows_with_toolrl_success']}/{tool['prompt_rows']}`；如果只依赖模型 rollout，Tool16 可恢复信号偏少。",
        f"- Code16 的正轨迹来自真实模型 rollout，`{code['rows_with_any_expert_success']}/{code['prompt_rows']}` 条都有至少一个专家成功；R1 做对 `{code['r1_positive_rows']}/{code['prompt_rows']}` 条。",
        "- 本机没有找到纯 DeepSeek-R1-Distill-Qwen-7B 的 BFCL result；只找到 R1-injected merged model 的 Tool 结果，因此 Tool 侧不能证明纯 R1 有正确 tool trajectory。",
        "",
        "## Tool16",
        "",
        f"- rows: `{tool['prompt_rows']}`",
        f"- eval group: `{tool['by_eval_group']}`",
        f"- category: `{tool['by_category']}`",
        f"- official answer anchor: `{tool['official_answer_anchor_success_rows']}/{tool['prompt_rows']}`",
        "",
        "| model | success | mean reward | missing |",
        "|---|---:|---:|---:|",
    ]
    for model, stats in sorted(tool["model_stats"].items()):
        lines.append(
            f"| {model} | {stats['success_rows']}/{tool['prompt_rows']} | {stats['mean_reward']:.4f} | {stats['missing_results']} |"
        )
    lines.extend(
        [
            "",
            "## Code16",
            "",
            f"- rows: `{code['prompt_rows']}`",
            f"- expert rollout rows: `{code['expert_rollout_rows']}`",
            f"- positive samples: `{code['positive_samples']}`",
            f"- base success rows: `{code['base_success_rows']}`",
            f"- rows with any expert success: `{code['rows_with_any_expert_success']}/{code['prompt_rows']}`",
            f"- positive rows by expert: `{code['positive_rows_by_expert']}`",
            f"- positive samples by expert: `{code['positive_samples_by_expert']}`",
            "",
            "## 产物",
            "",
        ]
    )
    for key, path in summary["outputs"].items():
        lines.append(f"- `{key}`: `{path}`")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool-prompts", type=Path, default=DEFAULT_TOOL_PROMPTS)
    parser.add_argument("--code-expert-rollout", type=Path, default=DEFAULT_CODE_EXPERT)
    parser.add_argument("--code-summary", type=Path, default=DEFAULT_CODE_SUMMARY)
    parser.add_argument("--bfcl-result-root", type=Path, default=DEFAULT_BFCL_RESULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    main()
