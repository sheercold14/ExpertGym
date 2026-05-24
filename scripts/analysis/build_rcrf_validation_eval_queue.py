#!/usr/bin/env python3
"""Build bake/evaluation queues for RCF-BC validation interventions.

The validation interventions are small OP-VEC gate probes.  This script turns
their manifest into auditable shell queues for:

1. baking each gate to a HF checkpoint;
2. running Tool+Memory quick evaluation;
3. running Code hurt evaluation after behavior checks pass.

It only writes queue files and reports.  It does not execute bake or evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path("/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521")
DEFAULT_INTERVENTIONS = (
    ROOT
    / "contrast_gates"
    / "validation_card_interventions_20260522"
    / "validation_interventions_manifest.json"
)
DEFAULT_OUTPUT_DIR = ROOT / "analysis" / "rcrf_validation_eval_queue_20260522"
DEFAULT_DOC_REPORT = REPO_ROOT / "docs" / "report" / "RCRF" / "20260522_rcrf_validation_eval_queue.md"
DEFAULT_CHECKPOINT_ROOT = Path("/tmp/shared-storage/OnPolicy/checkpoints/rcrf_validation_interventions_20260522")
DEFAULT_EVAL_ROOT = Path("/tmp/shared-storage/ExpertGym/rcrf/eval/validation_interventions_20260522")
DEFAULT_CONFIG = "configs/gated_grpo.yaml"
DEFAULT_MODE_MANIFEST = "/tmp/shared-storage/OnPolicy/modes/opvec4/mode_manifest.json"
DEFAULT_PY = "/mnt/cache/wuruixiao/miniconda3/envs/BFCL/bin/python"
DEFAULT_CURE_PY = "/mnt/cache/wuruixiao/miniconda3/envs/CURE/bin/python"


QUEUE_FIELDS = [
    "candidate_id",
    "card_id",
    "operation",
    "validation_type",
    "gate_path",
    "checkpoint_path",
    "quick_eval_summary_dir",
    "code_eval_result_prefix",
    "row_count",
    "changed_rows",
    "hypothesis",
    "success_criterion",
]


def main() -> None:
    args = parse_args()
    interventions = load_json(Path(args.interventions).expanduser().resolve())
    generated = interventions.get("generated", [])
    if not generated:
        raise ValueError(f"No generated interventions found in {args.interventions}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    checkpoint_root = Path(args.checkpoint_root).expanduser().resolve()
    eval_root = Path(args.eval_root).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = [build_entry(row, checkpoint_root=checkpoint_root, eval_root=eval_root) for row in generated]
    manifest = {
        "format": "rcrf_validation_eval_queue_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "interventions_manifest": str(Path(args.interventions).expanduser().resolve()),
        "output_dir": str(output_dir),
        "checkpoint_root": str(checkpoint_root),
        "eval_root": str(eval_root),
        "config": args.config,
        "mode_manifest": args.mode_manifest,
        "candidate_count": len(entries),
        "entries": entries,
        "execution_order": [
            "bash bake_queue.sh",
            "bash quick_tool_memory_queue.sh",
            "inspect Tool/Memory guardrails",
            "bash code_hurt_queue.sh for candidates that pass guardrails",
        ],
    }
    write_json(output_dir / "eval_queue_manifest.json", manifest)
    write_csv(output_dir / "eval_queue_candidates.csv", entries, QUEUE_FIELDS)
    write_text(output_dir / "bake_queue.sh", render_bake_queue(entries, args))
    write_text(output_dir / "quick_tool_memory_queue.sh", render_quick_eval_queue(entries, args))
    write_text(output_dir / "code_hurt_queue.sh", render_code_hurt_queue(entries, args))
    write_text(output_dir / "README.md", render_markdown(manifest))
    make_executable(output_dir / "bake_queue.sh")
    make_executable(output_dir / "quick_tool_memory_queue.sh")
    make_executable(output_dir / "code_hurt_queue.sh")

    if args.doc_report:
        doc_report = Path(args.doc_report).expanduser().resolve()
        doc_report.parent.mkdir(parents=True, exist_ok=True)
        doc_report.write_text(render_markdown(manifest), encoding="utf-8")

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "candidates": len(entries),
                "manifest": str(output_dir / "eval_queue_manifest.json"),
                "doc_report": str(Path(args.doc_report).expanduser().resolve()) if args.doc_report else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interventions", type=Path, default=DEFAULT_INTERVENTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--doc-report", type=Path, default=DEFAULT_DOC_REPORT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--mode-manifest", default=DEFAULT_MODE_MANIFEST)
    parser.add_argument("--default-py", default=DEFAULT_PY)
    parser.add_argument("--default-cure-py", default=DEFAULT_CURE_PY)
    return parser.parse_args()


def build_entry(row: dict[str, Any], *, checkpoint_root: Path, eval_root: Path) -> dict[str, Any]:
    candidate_id = str(row["candidate_id"])
    checkpoint_path = checkpoint_root / candidate_id
    quick_eval_summary_dir = eval_root / candidate_id / "quick_tool_memory"
    return {
        "candidate_id": candidate_id,
        "card_id": str(row["card_id"]),
        "operation": str(row["operation"]),
        "validation_type": str(row["validation_type"]),
        "gate_path": str(row["gate_path"]),
        "checkpoint_path": str(checkpoint_path),
        "quick_eval_summary_dir": str(quick_eval_summary_dir),
        "code_eval_result_prefix": str(checkpoint_path),
        "row_count": int(row["row_count"]),
        "changed_rows": int(row["changed_rows"]),
        "hypothesis": str(row["hypothesis"]),
        "success_criterion": str(row["success_criterion"]),
    }


def render_header(args: argparse.Namespace) -> list[str]:
    return [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f'PY="${{PY:-{args.default_py}}}"',
        f'CURE_PY="${{CURE_PY:-{args.default_cure_py}}}"',
        f'CONFIG="${{CONFIG:-{args.config}}}"',
        f'MODE_MANIFEST="${{MODE_MANIFEST:-{args.mode_manifest}}}"',
        'TOOL_GPU="${TOOL_GPU:-0}"',
        'TOOL_PORT="${TOOL_PORT:-8151}"',
        'MEMORY_GPU_IDS="${MEMORY_GPU_IDS:-$TOOL_GPU}"',
        'MEMORY_DATASETS="${MEMORY_DATASETS:-eval_50}"',
        'CODE_GPU="${CODE_GPU:-2}"',
        'CODE_DATASETS="${CODE_DATASETS:-LiveBenchCodeHurtRcrfVsTa16 LiveCodeBenchCodeHurtRcrfVsTa16}"',
        "",
        "cd " + q(str(REPO_ROOT)),
        "",
    ]


def render_bake_queue(entries: list[dict[str, Any]], args: argparse.Namespace) -> str:
    lines = render_header(args)
    for row in entries:
        lines.extend(
            [
                f"echo '[bake] {row['candidate_id']}'",
                '"$PY" scripts/eval/opvec_bake_checkpoint.py \\',
                '  --config "$CONFIG" \\',
                '  --mode-manifest "$MODE_MANIFEST" \\',
                f"  --gate-checkpoint {q(row['gate_path'])} \\",
                f"  --output {q(row['checkpoint_path'])}",
                "",
            ]
        )
    return "\n".join(lines)


def render_quick_eval_queue(entries: list[dict[str, Any]], args: argparse.Namespace) -> str:
    lines = render_header(args)
    for row in entries:
        lines.extend(
            [
                f"echo '[quick_eval] {row['candidate_id']}'",
                "RUN_TOOL=1 \\",
                "RUN_MEMORY=1 \\",
                "RUN_CODE=0 \\",
                'TOOL_GPU="$TOOL_GPU" \\',
                'TOOL_PORT="$TOOL_PORT" \\',
                'MEMORY_GPU_IDS="$MEMORY_GPU_IDS" \\',
                'MEMORY_DATASETS="$MEMORY_DATASETS" \\',
                "RUN_ID=quick_tool_memory \\",
                "EXPERIMENT_NAME=rcrf-validation \\",
                "ROOT=/tmp/shared-storage/ExpertGym/rcrf \\",
                f"SUMMARY_DIR={q(row['quick_eval_summary_dir'])} \\",
                f"bash skill/command/run_full_eval_suite.sh {q(row['checkpoint_path'])} {q(row['candidate_id'])}",
                "",
            ]
        )
    return "\n".join(lines)


def render_code_hurt_queue(entries: list[dict[str, Any]], args: argparse.Namespace) -> str:
    lines = render_header(args)
    for row in entries:
        lines.extend(
            [
                f"echo '[code_hurt] {row['candidate_id']}'",
                f"MODEL={q(row['checkpoint_path'])} \\",
                'GPU="$CODE_GPU" \\',
                'PYTHON_BIN="$CURE_PY" \\',
                'DATASETS="$CODE_DATASETS" \\',
                "bash scripts/eval/run_cure_code_hurt_eval.sh",
                "",
            ]
        )
    return "\n".join(lines)


def render_markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# 2026-05-22 RCF-BC Validation Eval Queue",
        "",
        "## 目的",
        "",
        "这个队列把 validation interventions 展开成可执行但默认不自动运行的 bake / quick eval / code hurt 命令。它用于保证每个反事实 probe 的评测流程可回溯、可审查。",
        "",
        "## 执行顺序",
        "",
    ]
    for step in manifest["execution_order"]:
        lines.append(f"- `{step}`")
    lines.extend(
        [
            "",
            "## Candidates",
            "",
            "| candidate | operation | validation | rows | changed |",
            "|---|---|---|---:|---:|",
        ]
    )
    for row in manifest["entries"]:
        lines.append(
            f"| `{row['candidate_id']}` | `{row['operation']}` | `{row['validation_type']}` | "
            f"{row['row_count']} | {row['changed_rows']} |"
        )
    lines.extend(
        [
            "",
            "## 队列脚本",
            "",
            f"- bake: `{Path(manifest['output_dir']) / 'bake_queue.sh'}`",
            f"- Tool/Memory quick: `{Path(manifest['output_dir']) / 'quick_tool_memory_queue.sh'}`",
            f"- Code hurt: `{Path(manifest['output_dir']) / 'code_hurt_queue.sh'}`",
            f"- manifest: `{Path(manifest['output_dir']) / 'eval_queue_manifest.json'}`",
            f"- candidates CSV: `{Path(manifest['output_dir']) / 'eval_queue_candidates.csv'}`",
            "",
            "## Guardrail",
            "",
            "先跑 `bake_queue.sh` 和 `quick_tool_memory_queue.sh`。只有 Tool/Memory quick 没有明显失败的候选，才进入 Code hurt。这个顺序避免在行为已经失败的 probe 上浪费 Code 评测时间。",
            "",
        ]
    )
    return "\n".join(lines)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content + "\n", encoding="utf-8")


def make_executable(path: Path) -> None:
    path.chmod(path.stat().st_mode | 0o111)


def q(value: str) -> str:
    return shlex.quote(value)


if __name__ == "__main__":
    main()
