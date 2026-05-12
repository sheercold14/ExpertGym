#!/usr/bin/env python3
"""Convert an OP-VEC seed manifest into a VeRL/slime-friendly prompt dataset.

The output JSONL always works.  If pandas/pyarrow are installed, ``--parquet``
also writes a Parquet file with the common VeRL columns: data_source, prompt,
ability, reward_model, and extra_info.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from opvec.data.io import read_jsonl, write_jsonl


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.seed_manifest)
    if args.tasks:
        tasks = {item.strip() for item in args.tasks.split(",") if item.strip()}
        rows = [row for row in rows if str(row.get("task")) in tasks]
    if args.limit is not None:
        rows = rows[: args.limit]
    out_rows = [convert_row(row) for row in rows]
    count = write_jsonl(args.output, out_rows)
    summary = {"format": "opvec_verl_prompt_dataset_v1", "rows": count, "output": args.output}
    if args.parquet:
        try:
            import pandas as pd
        except ImportError as error:  # pragma: no cover
            raise SystemExit("--parquet requires pandas and pyarrow/fastparquet") from error
        Path(args.parquet).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(out_rows).to_parquet(args.parquet, index=False)
        summary["parquet"] = args.parquet
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def convert_row(row: dict) -> dict:
    prompt = row.get("prompt") or ""
    messages = row.get("messages")
    if messages:
        # VeRL examples typically store either a string prompt or chat messages.
        # Keeping both prompt and messages in extra_info avoids forcing one chat
        # template at dataset-build time.
        prompt = prompt or "\n".join(f"{item.get('role','user')}: {item.get('content','')}" for item in messages)
    return {
        "data_source": "opvec",
        "prompt": prompt,
        "ability": str(row.get("task", "unknown")),
        "reward_model": {"style": "rule", "ground_truth": row.get("reference", {})},
        "extra_info": {
            "prompt_id": row.get("prompt_id"),
            "task": row.get("task"),
            "source": row.get("source"),
            "split": row.get("split"),
            "messages": messages,
            "reference": row.get("reference", {}),
            "verifier": row.get("verifier"),
            "tags": row.get("tags", []),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--parquet", default=None)
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
