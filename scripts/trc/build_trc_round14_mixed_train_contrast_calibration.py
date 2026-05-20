#!/usr/bin/env python3
"""Build Round14 mixed Code calibration: train CodeP0 + contrastive hard anchors."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from opvec.data.io import read_jsonl, write_jsonl


DEFAULT_STABLE_TRAIN_BANK = Path(
    "/tmp/shared-storage/OnPolicy/data/calibration/20260520_trc_round11_hybrid_v1/"
    "r11b_r11g_r10tag24_rf8_stablelate3/trc96_expert_trajectories.jsonl"
)
DEFAULT_CONTRAST_BANK = Path(
    "/tmp/shared-storage/OnPolicy/data/calibration/20260520_trc_round14_code_contrast_v1/"
    "trc_expert_trajectories.jsonl"
)
DEFAULT_OUTPUT_DIR = Path(
    "/tmp/shared-storage/OnPolicy/data/calibration/20260520_trc_round14_mixed_train_contrast_v1"
)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stable_rows = read_jsonl(Path(args.stable_train_bank).expanduser())
    contrast_rows = read_jsonl(Path(args.contrast_bank).expanduser())
    tool_rows = select_task_rows(stable_rows, "tool", int(args.tool_count))
    memory_rows = select_task_rows(stable_rows, "memory", int(args.memory_count))
    train_code_rows = select_task_rows(stable_rows, "code", int(args.train_code_count))
    eval_code_rows = select_eval_contrast_rows(
        [row for row in contrast_rows if row.get("task") == "code"],
        int(args.eval_code_count),
    )

    rows: list[dict[str, Any]] = []
    rows.extend(with_provenance(row, component="stable_tool", source="stable_train_bank") for row in tool_rows)
    rows.extend(with_provenance(row, component="stable_memory", source="stable_train_bank") for row in memory_rows)
    rows.extend(with_provenance(row, component="codep0_train_code", source="stable_train_bank") for row in train_code_rows)
    rows.extend(with_provenance(row, component="formal_contrast_code", source="contrast_bank") for row in eval_code_rows)
    renumber(rows)
    validate(rows, expected_total=int(args.tool_count) + int(args.memory_count) + int(args.train_code_count) + int(args.eval_code_count))

    out_jsonl = output_dir / "trc96_expert_trajectories.jsonl"
    write_jsonl(out_jsonl, rows)
    summary = {
        "format": "trc_round14_mixed_train_contrast_summary_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output": str(out_jsonl),
        "input_banks": {
            "stable_train_bank": str(Path(args.stable_train_bank).expanduser().resolve()),
            "contrast_bank": str(Path(args.contrast_bank).expanduser().resolve()),
        },
        "num_rows": len(rows),
        "task_counts": dict(sorted(Counter(row.get("task") for row in rows).items())),
        "expert_counts": dict(sorted(Counter(row.get("expert") for row in rows).items())),
        "code_policy": {
            "train_code_count": int(args.train_code_count),
            "eval_contrast_code_count": int(args.eval_code_count),
            "train_code_source": "CodeContests_train / CodeP0 original train successful trajectories",
            "eval_contrast_source": "formal CURE hard anchors with negative_response; diagnostic only",
            "eval_selection": "Prefer rows whose gate expert is code, then fill deterministically.",
        },
        "code_distribution": summarize_code(rows),
        "notes": [
            "This bank mixes original train CodeP0 prompts with a small formal contrastive hard-anchor slice.",
            "Rows with negative_response only affect training when contrastive-negative loss is explicitly enabled.",
            "Because formal hard anchors are included, this specific bank is diagnostic and not a non-leak paper-main dataset.",
        ],
    }
    write_json(output_dir / "summary.json", summary)
    (output_dir / "README.md").write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def select_task_rows(rows: list[dict[str, Any]], task: str, count: int) -> list[dict[str, Any]]:
    selected = [deepcopy(row) for row in rows if row.get("task") == task]
    if len(selected) < count:
        raise ValueError(f"Need {count} {task} rows, found {len(selected)}")
    return selected[:count]


def select_eval_contrast_rows(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    usable = [deepcopy(row) for row in rows if str(row.get("negative_response") or "").strip()]
    usable.sort(
        key=lambda row: (
            0 if row.get("expert") == "code" else 1,
            str(((row.get("reference") or {}).get("metadata") or {}).get("source_dataset") or ""),
            str(row.get("prompt_id") or ""),
            str(row.get("trajectory_id") or ""),
        )
    )
    if len(usable) < count:
        raise ValueError(f"Need {count} contrast Code rows with negative_response, found {len(usable)}")
    return usable[:count]


def with_provenance(row: dict[str, Any], *, component: str, source: str) -> dict[str, Any]:
    copied = deepcopy(row)
    metadata = dict(copied.get("row_metadata") or {})
    metadata["round14_mixed_train_contrast"] = {
        "component": component,
        "source": source,
    }
    copied["row_metadata"] = metadata
    return copied


def renumber(rows: list[dict[str, Any]]) -> None:
    for index, row in enumerate(rows):
        old = str(row.get("trajectory_id") or row.get("sample_id") or index)
        row["trajectory_id"] = f"trc14mixed__{index:03d}__{row.get('task')}__{row.get('expert')}__{stable_suffix(old)}"


def stable_suffix(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[-80:] or "row"


def validate(rows: list[dict[str, Any]], *, expected_total: int) -> None:
    if len(rows) != expected_total:
        raise ValueError(f"Expected {expected_total} rows, found {len(rows)}")
    for index, row in enumerate(rows):
        if not str(row.get("rendered_prompt") or row.get("prompt") or "").strip():
            raise ValueError(f"row {index} has blank prompt")
        if not str(row.get("response") or "").strip():
            raise ValueError(f"row {index} has blank response")
    counts = Counter(row.get("task") for row in rows)
    if counts.get("tool") != 32 or counts.get("memory") != 32 or counts.get("code") != 32:
        raise ValueError(f"Expected 32 rows per task, got {dict(counts)}")


def summarize_code(rows: list[dict[str, Any]]) -> dict[str, Any]:
    code_rows = [row for row in rows if row.get("task") == "code"]
    source_datasets = Counter()
    components = Counter()
    negative_count = 0
    for row in code_rows:
        metadata = (row.get("reference") or {}).get("metadata") or {}
        source_datasets[str(metadata.get("source_dataset") or "unknown")] += 1
        component = ((row.get("row_metadata") or {}).get("round14_mixed_train_contrast") or {}).get("component")
        components[str(component or "unknown")] += 1
        if str(row.get("negative_response") or "").strip():
            negative_count += 1
    return {
        "rows": len(code_rows),
        "unique_prompts": len({row.get("prompt_id") for row in code_rows}),
        "expert_counts": dict(sorted(Counter(row.get("expert") for row in code_rows).items())),
        "source_dataset_counts": dict(sorted(source_datasets.items())),
        "component_counts": dict(sorted(components.items())),
        "negative_response_rows": negative_count,
    }


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Round14 Mixed Train + Contrast Code Calibration",
            "",
            f"- Output: `{summary['output']}`",
            f"- Rows: `{summary['num_rows']}`",
            f"- Task counts: `{summary['task_counts']}`",
            f"- Code distribution: `{summary['code_distribution']}`",
            "",
            "This is a diagnostic bank: Code mixes original CodeP0 train trajectories and a small formal contrastive anchor slice.",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stable-train-bank", default=str(DEFAULT_STABLE_TRAIN_BANK))
    parser.add_argument("--contrast-bank", default=str(DEFAULT_CONTRAST_BANK))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--tool-count", type=int, default=32)
    parser.add_argument("--memory-count", type=int, default=32)
    parser.add_argument("--train-code-count", type=int, default=24)
    parser.add_argument("--eval-code-count", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    main()
