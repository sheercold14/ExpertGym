#!/usr/bin/env python3
"""Build MemAgent-style trajectory seed manifests from HotpotQA parquet data."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from opvec.data.io import write_jsonl
from opvec.data.schema import make_prompt_id, stable_hash, validate_seed_record


DEFAULT_HOTPOTQA_TRAIN = "/mnt/cache/wuruixiao/users/lsc/ExpertMerging/dataset/hotpotqa/hotpotqa_train_32k.parquet"
DEFAULT_TOKENIZER = "/mnt/cache/wuruixiao/models/Qwen2.5-7b-instruct"


def main() -> None:
    args = parse_args()
    rows, summary = build_manifest(args)
    if not args.dry_run:
        count = write_jsonl(args.output, rows)
        summary.update({"output": str(args.output), "written": count})
        Path(args.output).with_suffix(".summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def build_manifest(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import pandas as pd
    from transformers import AutoTokenizer

    df = pd.read_parquet(args.input)
    candidates: list[int] = []
    skipped = Counter()
    for source_row, (_, row) in enumerate(df.iterrows()):
        question = _question_from_prompt(row.get("prompt"))
        context = str(row.get("context") or "")
        answers = _answers_from_reward_model(row.get("reward_model"))
        if not question:
            skipped["missing_question"] += 1
            continue
        if not context:
            skipped["missing_context"] += 1
            continue
        if not answers:
            skipped["missing_ground_truth"] += 1
            continue
        candidates.append(int(source_row))

    selected_indices = _sample(candidates, limit=args.limit, seed=args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.chunk_tokenizer, trust_remote_code=True)

    rows: list[dict[str, Any]] = []
    chunk_counts = Counter()
    context_token_lengths: list[int] = []
    for source_row in selected_indices:
        row = df.iloc[source_row]
        question = _question_from_prompt(row.get("prompt"))
        context = str(row.get("context") or "")
        answers = _answers_from_reward_model(row.get("reward_model"))
        extra_info = _jsonable(row.get("extra_info") or {})
        chunks, token_count = _token_chunks(
            tokenizer,
            context,
            chunk_size_tokens=args.chunk_size_tokens,
            max_chunks=args.max_chunks,
        )
        if not chunks:
            skipped["empty_chunks_after_tokenization"] += 1
            continue
        chunk_counts[len(chunks)] += 1
        context_token_lengths.append(token_count)
        context_hash = stable_hash({"context": context})
        prompt_hash = stable_hash(
            {
                "task": "memory",
                "source": str(args.input),
                "source_row": int(source_row),
                "question": question,
                "answers": answers,
                "context_hash": context_hash,
                "chunk_size_tokens": int(args.chunk_size_tokens),
                "max_chunks": int(args.max_chunks),
            }
        )
        hotpot_index = extra_info.get("index", source_row) if isinstance(extra_info, Mapping) else source_row
        record = {
            "prompt_id": make_prompt_id("memory", prompt_hash),
            "task": "memory",
            "source": str(args.input),
            "source_row": int(source_row),
            "split": args.split,
            "prompt": question,
            "messages": [],
            "reference": {
                "answer": answers,
                "response": f"\\boxed{{{answers[0]}}}",
                "metadata": {
                    "source_dataset": "MemAgent/HotpotQA train parquet",
                    "source_split": args.split,
                    "question_id": str(hotpot_index),
                    "hotpotqa_index": hotpot_index,
                    "round_type": "trajectory",
                    "memagent_prompt": question,
                    "memagent_chunks": chunks,
                    "num_chunks": len(chunks),
                    "answers": answers,
                    "ground_truth": answers,
                    "num_docs": extra_info.get("num_docs") if isinstance(extra_info, Mapping) else None,
                    "context_token_count": token_count,
                    "context_hash": context_hash,
                    "chunk_size_tokens": int(args.chunk_size_tokens),
                    "chunk_tokenizer": str(args.chunk_tokenizer),
                    "max_chunks": int(args.max_chunks),
                },
            },
            "verifier": {"name": "memagent_source_reward", "config": {"source": "MemAgent"}},
            "tags": ["hotpotqa", "memagent", "memory_trajectory", "source_reward", "train_parquet"],
            "difficulty": None,
            "prompt_hash": prompt_hash,
        }
        validate_seed_record(record)
        rows.append(record)

    summary = {
        "format": "opvec_hotpotqa_memory_seed_manifest_v1",
        "input": str(args.input),
        "rows": len(rows),
        "seed": int(args.seed),
        "split": args.split,
        "task_counts": {"memory": len(rows)},
        "source_summary": {
            "available": len(candidates),
            "selected": len(rows),
            "skipped": dict(sorted(skipped.items())),
            "chunk_count_distribution": {str(k): v for k, v in sorted(chunk_counts.items())},
            "context_token_count": _basic_stats(context_token_lengths),
        },
        "chunking": {
            "tokenizer": str(args.chunk_tokenizer),
            "chunk_size_tokens": int(args.chunk_size_tokens),
            "max_chunks": int(args.max_chunks),
        },
    }
    return rows, summary


def _token_chunks(tokenizer: Any, text: str, *, chunk_size_tokens: int, max_chunks: int) -> tuple[list[str], int]:
    token_ids = tokenizer(text, add_special_tokens=False).input_ids
    chunks: list[str] = []
    for start in range(0, len(token_ids), chunk_size_tokens):
        if max_chunks > 0 and len(chunks) >= max_chunks:
            break
        piece = token_ids[start : start + chunk_size_tokens]
        decoded = tokenizer.decode(piece, skip_special_tokens=True).strip()
        if decoded:
            chunks.append(decoded)
    return chunks, len(token_ids)


def _question_from_prompt(value: Any) -> str:
    value = _jsonable(value)
    if isinstance(value, list):
        for item in value:
            if isinstance(item, Mapping) and str(item.get("role") or "") == "user":
                content = str(item.get("content") or "").strip()
                if content:
                    return content
        for item in value:
            if isinstance(item, Mapping):
                content = str(item.get("content") or "").strip()
                if content:
                    return content
    if isinstance(value, str):
        return value.strip()
    return ""


def _answers_from_reward_model(value: Any) -> list[str]:
    value = _jsonable(value)
    if not isinstance(value, Mapping):
        return []
    raw = value.get("ground_truth") or value.get("answers") or value.get("answer")
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    text = str(raw).strip()
    return [text] if text else []


def _sample(rows: list[int], *, limit: int, seed: int) -> list[int]:
    pool = list(rows)
    random.Random(seed).shuffle(pool)
    if limit < 0:
        return pool
    return pool[: min(int(limit), len(pool))]


def _basic_stats(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "max": None, "mean": None}
    return {
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_HOTPOTQA_TRAIN)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=500, help="-1 means all")
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--split", default="source_reward_train")
    parser.add_argument("--chunk-tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--chunk-size-tokens", type=int, default=5000)
    parser.add_argument("--max-chunks", type=int, default=0, help="0 means no cap")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
