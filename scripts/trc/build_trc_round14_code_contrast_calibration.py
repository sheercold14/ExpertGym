#!/usr/bin/env python3
"""Build TRC Code contrastive calibration from positive expert rows and CURE failures.

The output keeps normal TRC rows unchanged except Code rows may receive a
`negative_response` field. The trainer ignores that field unless
`--contrastive-negative-loss-weight > 0`, so this builder is safe for normal
positive-only TRC runs too.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from opvec.data.io import read_jsonl, write_jsonl


DEFAULT_INPUT = Path(
    "/tmp/shared-storage/OnPolicy/data/calibration/20260520_trc_round13_evalleak_code16/"
    "rfmem_only/trc_expert_trajectories.jsonl"
)
DEFAULT_OUTPUT_DIR = Path(
    "/tmp/shared-storage/OnPolicy/data/calibration/20260520_trc_round14_code_contrast_v1"
)
DEFAULT_NEGATIVE_TEMPS = [
    Path(
        "/mnt/cache/wuruixiao/users/lsc/CURE/evaluation/temp_data/"
        "outputs-eval-.tmp.shared-storage.OnPolicy.checkpoints."
        "trc_r11b_r8d_codeblock_e08_20260520-selected-LiveBench.json"
    ),
    Path(
        "/mnt/cache/wuruixiao/users/lsc/CURE/evaluation/temp_data/"
        "outputs-eval-.tmp.shared-storage.OnPolicy.checkpoints."
        "trc_r11b_r8d_codeblock_e08_20260520-selected-LiveCodeBench.json"
    ),
]


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(input_path)
    negative_bank = load_negative_bank([Path(item).expanduser() for item in args.negative_temp])

    augmented: list[dict[str, Any]] = []
    stats = Counter()
    dataset_stats = Counter()
    for row in rows:
        copied = deepcopy(row)
        if copied.get("task") == "code":
            stats["code_rows"] += 1
            key = code_eval_key(copied)
            dataset_stats[str(key[0])] += 1
            negative = negative_bank.get(key)
            if negative is not None:
                copied["negative_response"] = negative["response"]
                copied["negative_reward_train"] = negative["pass_fraction"]
                copied.setdefault("row_metadata", {})["round14_code_contrast"] = {
                    "negative_source": negative["source_path"],
                    "negative_dataset": negative["dataset"],
                    "negative_source_row": negative["source_row"],
                    "negative_candidate_index": negative["candidate_index"],
                    "negative_pass_fraction": negative["pass_fraction"],
                    "negative_test_count": negative["test_count"],
                    "policy": "Use the lowest-pass failed CURE generation for the same formal prompt.",
                }
                stats["code_rows_with_negative"] += 1
            else:
                stats["code_rows_without_negative"] += 1
        augmented.append(copied)

    out_jsonl = output_dir / "trc_expert_trajectories.jsonl"
    write_jsonl(out_jsonl, augmented)
    summary = {
        "format": "trc_round14_code_contrast_summary_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "output": str(out_jsonl),
        "negative_temp_files": [str(Path(item).expanduser()) for item in args.negative_temp],
        "num_rows": len(augmented),
        "task_counts": dict(sorted(Counter(row.get("task") for row in augmented).items())),
        "stats": dict(sorted(stats.items())),
        "code_dataset_counts": dict(sorted(dataset_stats.items())),
        "notes": [
            "This is an eval-leak diagnostic if the input positive bank is formal Code16.",
            "Rows without negative_response still train with the normal positive TRC objective.",
            "The trainer uses negative_response only when --contrastive-negative-loss-weight is set.",
        ],
    }
    write_json(output_dir / "summary.json", summary)
    (output_dir / "README.md").write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def load_negative_bank(paths: list[Path]) -> dict[tuple[str, int], dict[str, Any]]:
    bank: dict[tuple[str, int], dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        dataset = infer_dataset(path.name)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            continue
        for source_row, item in enumerate(payload):
            negative = select_failed_candidate(item)
            if negative is None:
                continue
            negative["dataset"] = dataset
            negative["source_row"] = source_row
            negative["source_path"] = str(path)
            key = (dataset, source_row)
            current = bank.get(key)
            if current is None or negative["pass_fraction"] < current["pass_fraction"]:
                bank[key] = negative
    return bank


def select_failed_candidate(item: dict[str, Any]) -> dict[str, Any] | None:
    generations = item.get("full_code_generation") or item.get("generated_code") or []
    bool_tables = item.get("test_bool_table") or []
    if not isinstance(generations, list) or not isinstance(bool_tables, list):
        return None
    best: dict[str, Any] | None = None
    for index, response in enumerate(generations):
        text = str(response or "").strip()
        if not text:
            continue
        if item.get("full_code_generation") is None and "```" not in text:
            text = f"```python\n{text}\n```"
        passed = flatten_bools(bool_tables[index] if index < len(bool_tables) else [])
        if not passed:
            continue
        pass_fraction = sum(1.0 for value in passed if value) / float(len(passed))
        if pass_fraction >= 1.0:
            continue
        candidate = {
            "response": text,
            "candidate_index": index,
            "pass_fraction": pass_fraction,
            "test_count": len(passed),
        }
        if best is None or pass_fraction < best["pass_fraction"]:
            best = candidate
    return best


def flatten_bools(value: Any) -> list[bool]:
    if isinstance(value, bool):
        return [value]
    if isinstance(value, list):
        out: list[bool] = []
        for item in value:
            out.extend(flatten_bools(item))
        return out
    return []


def code_eval_key(row: dict[str, Any]) -> tuple[str, int]:
    metadata = (row.get("reference") or {}).get("metadata") or {}
    dataset = str(metadata.get("source_dataset") or infer_dataset(str(row.get("prompt_id") or "")))
    source_row = int(metadata.get("source_row"))
    return dataset, source_row


def infer_dataset(value: str) -> str:
    lowered = str(value).lower()
    if "livecodebench" in lowered:
        return "LiveCodeBench"
    if "livebench" in lowered:
        return "LiveBench"
    return "unknown"


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Round14 Code Contrastive TRC Calibration",
            "",
            f"- Input: `{summary['input']}`",
            f"- Output: `{summary['output']}`",
            f"- Rows: `{summary['num_rows']}`",
            f"- Task counts: `{summary['task_counts']}`",
            f"- Stats: `{summary['stats']}`",
            f"- Code datasets: `{summary['code_dataset_counts']}`",
            "",
            "This bank adds `negative_response` to Code rows when a failed CURE generation exists for the same prompt.",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--negative-temp", action="append", default=[str(path) for path in DEFAULT_NEGATIVE_TEMPS])
    return parser.parse_args()


if __name__ == "__main__":
    main()
