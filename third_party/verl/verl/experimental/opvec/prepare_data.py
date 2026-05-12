#!/usr/bin/env python3
"""Prepare small OP-VEC prompt manifests for verl-side smoke runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    args = parse_args()
    rows = _read_jsonl(Path(args.input))
    if args.tasks:
        tasks = {item.strip() for item in args.tasks.split(",") if item.strip()}
        rows = [row for row in rows if str(row.get("task")) in tasks]
    rows = rows[args.offset : args.offset + args.limit]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(Path(args.output), rows)
    summary = {
        "format": "opvec_verl_prepare_data_v1",
        "input": args.input,
        "output": args.output,
        "rows": len(rows),
        "offset": args.offset,
        "limit": args.limit,
    }
    if args.parquet:
        _write_parquet(rows, Path(args.parquet))
        summary["parquet"] = args.parquet
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    import pandas as pd

    out_rows = []
    for row in rows:
        out_rows.append(
            {
                "data_source": "opvec",
                "agent_name": _agent_name(row),
                "prompt": _verl_messages(row),
                "ability": str(row.get("task", "unknown")),
                "reward_model": {"style": "rule", "ground_truth": _json_dumps(row.get("reference", {}))},
                "extra_info": {
                    "prompt_id": row.get("prompt_id"),
                    "task": row.get("task"),
                    "source": row.get("source"),
                    "split": row.get("split"),
                    "prompt": row.get("prompt"),
                    "difficulty": row.get("difficulty"),
                    "merged_seed_source": row.get("merged_seed_source"),
                    "messages_json": _json_dumps(row.get("messages")),
                    "reference_json": _json_dumps(row.get("reference", {})),
                    "verifier_json": _json_dumps(row.get("verifier")),
                    "tags_json": _json_dumps(row.get("tags", [])),
                    "question_bank_selection_json": _json_dumps(row.get("question_bank_selection")),
                },
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_parquet(path, index=False)


def _verl_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        return [
            {
                "role": str(message.get("role", "user")),
                "content": str(message.get("content", "")),
            }
            for message in messages
            if isinstance(message, dict)
        ]
    return [{"role": "user", "content": str(row.get("prompt") or "")}]


def _agent_name(row: dict[str, Any]) -> str:
    reference = row.get("reference") if isinstance(row.get("reference"), dict) else {}
    metadata = reference.get("metadata", {}) if isinstance(reference.get("metadata"), dict) else {}
    chunks = metadata.get("memagent_chunks") or reference.get("memagent_chunks")
    if row.get("task") == "memory" and isinstance(chunks, list) and chunks:
        return "opvec_memagent"
    return "single_turn_agent"


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--parquet", default=None)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--tasks", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
