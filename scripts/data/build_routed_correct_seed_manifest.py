#!/usr/bin/env python3
"""Build official-aligned seed manifests from routed correct-sample pools."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from opvec.data.io import write_jsonl
from opvec.data.manifest import normalize_seed_record
from opvec.data.schema import make_prompt_id, stable_hash, validate_seed_record


DEFAULT_INPUT_ROOT = "/mnt/cache/wuruixiao/users/lsc/ExpertMerging/datasets_2/correct_samples/routed_1"


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
    root = Path(args.input_root)
    rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    if args.tool_limit != 0:
        tool_rows, summaries["tool"] = _single_turn_rows(root / "ToolCall.json", task="tool", limit=args.tool_limit, split=args.split)
        rows.extend(tool_rows)
    if args.code_limit != 0:
        code_rows, summaries["code"] = _single_turn_rows(root / "Code.json", task="code", limit=args.code_limit, split=args.split)
        rows.extend(code_rows)
    if args.memory_limit != 0:
        memory_rows, summaries["memory"] = _memory_trajectory_rows(root / "Memory.json", limit=args.memory_limit, split=args.split)
        rows.extend(memory_rows)
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    skipped_duplicates = 0
    for row in rows:
        key = str(row["prompt_hash"])
        if key in seen:
            skipped_duplicates += 1
            continue
        seen.add(key)
        validate_seed_record(row)
        deduped.append(row)
    summary = {
        "format": "opvec_routed_correct_seed_manifest_v1",
        "input_root": str(root),
        "rows": len(deduped),
        "task_counts": dict(sorted(Counter(row["task"] for row in deduped).items())),
        "skipped_duplicates": skipped_duplicates,
        "source_summaries": summaries,
    }
    return deduped, summary


def _single_turn_rows(path: Path, *, task: str, limit: int, split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = _read_list(path)
    rows: list[dict[str, Any]] = []
    skipped = Counter()
    for index, raw in enumerate(payload):
        if not isinstance(raw, Mapping):
            skipped["non_object"] += 1
            continue
        try:
            row = normalize_seed_record(raw, task=task, source_path=path, row_index=index, split=split)
        except ValueError:
            skipped["invalid_record"] += 1
            continue
        tags = set(row.get("tags", []))
        tags.update({"routed_correct", "official_reward"})
        row["tags"] = sorted(tags)
        rows.append(row)
        if limit > 0 and len(rows) >= limit:
            break
    return rows, {"available": len(payload), "selected": len(rows), "skipped": dict(sorted(skipped.items()))}


def _memory_trajectory_rows(path: Path, *, limit: int, split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = _read_list(path)
    chunks_by_qid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    finals_by_qid: dict[str, dict[str, Any]] = {}
    skipped = Counter()
    for index, raw in enumerate(payload):
        if not isinstance(raw, Mapping):
            skipped["non_object"] += 1
            continue
        qid = str(raw.get("question_id") or "")
        if not qid:
            skipped["missing_question_id"] += 1
            continue
        round_type = str(raw.get("round_type") or "")
        if round_type == "chunk":
            parsed = _parse_memory_update_message(raw)
            if parsed is None:
                skipped["unparseable_chunk"] += 1
                continue
            parsed.update({"source_row": index, "round_idx": raw.get("round_idx")})
            chunks_by_qid[qid].append(parsed)
        elif round_type == "final":
            finals_by_qid[qid] = {**dict(raw), "source_row": index}
    rows: list[dict[str, Any]] = []
    for qid in sorted(finals_by_qid, key=lambda item: (int(item) if item.isdigit() else item)):
        if qid not in chunks_by_qid:
            skipped["final_without_chunks"] += 1
            continue
        final = finals_by_qid[qid]
        final_prompt = _parse_final_prompt(final)
        chunks = sorted(chunks_by_qid[qid], key=lambda item: (item.get("round_idx") is None, item.get("round_idx") or 0))
        prompt = final_prompt or chunks[0]["problem"]
        sections = [item["section"] for item in chunks]
        answer = _boxed_answer(str(final.get("response") or ""))
        if not prompt or not sections or not answer:
            skipped["missing_prompt_chunks_or_answer"] += 1
            continue
        prompt_hash = stable_hash({"task": "memory", "question_id": qid, "prompt": prompt, "chunks": sections})
        row = {
            "prompt_id": make_prompt_id("memory", prompt_hash),
            "task": "memory",
            "source": str(path),
            "source_row": int(final.get("source_row", -1)),
            "split": split,
            "prompt": prompt,
            "messages": [],
            "reference": {
                "answer": [answer],
                "response": str(final.get("response") or ""),
                "metadata": {
                    "source_dataset": "MemAgent/HotpotQA trajectory",
                    "question_id": qid,
                    "round_type": "trajectory",
                    "memagent_prompt": prompt,
                    "memagent_chunks": sections,
                    "num_chunks": len(sections),
                    "final_source_row": final.get("source_row"),
                    "chunk_source_rows": [item["source_row"] for item in chunks],
                },
            },
            "verifier": {"name": "memagent_source_reward", "config": {"source": "MemAgent"}},
            "tags": ["memagent", "memory_trajectory", "official_reward", "routed_correct"],
            "difficulty": None,
            "prompt_hash": prompt_hash,
        }
        rows.append(row)
        if limit > 0 and len(rows) >= limit:
            break
    return rows, {
        "available_questions": len(finals_by_qid),
        "selected": len(rows),
        "chunk_rows": sum(len(items) for items in chunks_by_qid.values()),
        "skipped": dict(sorted(skipped.items())),
    }


def _parse_memory_update_message(raw: Mapping[str, Any]) -> dict[str, str] | None:
    messages = raw.get("messages") if isinstance(raw.get("messages"), list) else []
    content = str(messages[0].get("content") or "") if messages and isinstance(messages[0], Mapping) else ""
    problem = _tag_content(content, "problem")
    section = _tag_content(content, "section")
    if not problem or not section:
        return None
    return {"problem": problem, "section": section}


def _parse_final_prompt(raw: Mapping[str, Any]) -> str:
    messages = raw.get("messages") if isinstance(raw.get("messages"), list) else []
    content = str(messages[0].get("content") or "") if messages and isinstance(messages[0], Mapping) else ""
    return _tag_content(content, "problem")


def _tag_content(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text or "", flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _boxed_answer(text: str) -> str:
    match = re.search(r"\\boxed\{(.*?)\}", text or "", flags=re.DOTALL)
    return match.group(1).strip() if match else str(text or "").strip()


def _read_list(path: Path) -> list[Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected list JSON: {path}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tool-limit", type=int, default=-1, help="-1 means all; 0 disables task")
    parser.add_argument("--memory-limit", type=int, default=-1, help="-1 means all; 0 disables task")
    parser.add_argument("--code-limit", type=int, default=-1, help="-1 means all; 0 disables task")
    parser.add_argument("--split", default="routed_correct_train")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
