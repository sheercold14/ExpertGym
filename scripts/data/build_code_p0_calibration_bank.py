#!/usr/bin/env python3
"""Build a leakage-safe Code P0 calibration bank from CodeContests train.

The output is intentionally code-only and disjoint across train/monitor/guard.
Each row stores the tests used by the training reward as ``metadata.test_input``
and keeps an additional held-out test slice in ``metadata.guard_test_input`` for
auditing.  No LiveBench/LiveCodeBench/CURE official eval prompts or hidden tests
are copied.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from opvec.data.io import read_json, write_jsonl
from opvec.data.schema import stable_hash, validate_seed_record


DEFAULT_CODECONTESTS = "/mnt/cache/wuruixiao/users/lsc/ExpertMerging/dataset/CodeContests_train/train/CodeContests_train.json"
DEFAULT_OUTPUT_DIR = "/tmp/shared-storage/OnPolicy/data/calibration/code_p0_v3_20260518"
SPLITS = ("train", "monitor", "guard")
ROLE_ORDER = ("generation", "frontier", "partial_edge", "stable")
DEFAULT_SPLIT_ROLE_COUNTS = {
    "train": {"generation": 24, "frontier": 20, "partial_edge": 12, "stable": 8},
    "monitor": {"generation": 10, "frontier": 10, "partial_edge": 8, "stable": 4},
    "guard": {"generation": 10, "frontier": 10, "partial_edge": 8, "stable": 4},
}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(args.codecontests).expanduser().resolve()
    source_rows = read_json(source_path)
    if not isinstance(source_rows, list):
        raise SystemExit(f"CodeContests file must be a JSON list: {source_path}")

    split_role_counts = _parse_split_role_counts(args.split_role_count)
    created_at = datetime.now(timezone.utc).isoformat()
    exclude_task_ids = _load_excluded_task_ids(args.exclude_manifest)
    candidates = _build_candidates(
        source_rows,
        source_path=source_path,
        min_tests=int(args.min_tests),
        exclude_task_ids=exclude_task_ids,
        seed=int(args.seed),
    )
    selected = _select_by_split_role(candidates, split_role_counts=split_role_counts, seed=int(args.seed))

    outputs: dict[str, str] = {}
    blueprints = []
    for split in SPLITS:
        rows = [
            _candidate_to_seed_record(
                candidate,
                split=split,
                tag=args.tag,
                created_at=created_at,
                reward_tests=int(args.reward_tests),
                guard_tests=int(args.guard_tests),
                seed=int(args.seed),
            )
            for candidate in selected[split]
        ]
        for row in rows:
            validate_seed_record(row)
        out_path = output_dir / f"{split}_code{len(rows)}.prompts.jsonl"
        write_jsonl(out_path, rows)
        outputs[split] = str(out_path)
        blueprints.extend(_row_blueprint(row) for row in rows)

    blueprint_path = output_dir / "code_p0_blueprints.jsonl"
    write_jsonl(blueprint_path, blueprints)
    summary = _summary(
        args=args,
        created_at=created_at,
        source_path=source_path,
        output_dir=output_dir,
        candidates=candidates,
        selected=selected,
        outputs=outputs,
        blueprint_path=blueprint_path,
        exclude_task_ids=exclude_task_ids,
        split_role_counts=split_role_counts,
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    readme_path = output_dir / "README.md"
    readme_path.write_text(_render_readme(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def _build_candidates(
    source_rows: list[Any],
    *,
    source_path: Path,
    min_tests: int,
    exclude_task_ids: set[str],
    seed: int,
) -> list[dict[str, Any]]:
    candidates = []
    for idx, raw in enumerate(source_rows):
        if not isinstance(raw, dict):
            continue
        task_id = str(raw.get("task_id") or "")
        if not task_id or task_id in exclude_task_ids:
            continue
        question = str(raw.get("question") or "").strip()
        inputs = list(raw.get("test_input") or [])
        outputs = list(raw.get("test_output") or [])
        total_tests = min(len(inputs), len(outputs))
        if not question or total_tests < min_tests:
            continue
        tags = _code_tags(question)
        if not tags:
            continue
        score = _code_priority_score(question, tags, total_tests)
        roles = _candidate_roles(question, tags, score)
        candidates.append(
            {
                "idx": idx,
                "raw": raw,
                "source_path": str(source_path),
                "task_id": task_id,
                "question": question,
                "tags": tags,
                "roles": roles,
                "score": score,
                "total_tests": total_tests,
                "sort_key": stable_hash({"seed": seed, "task_id": task_id, "question": question}),
            }
        )
    if not candidates:
        raise SystemExit("No Code P0 candidates found.")
    return candidates


def _select_by_split_role(
    candidates: list[dict[str, Any]],
    *,
    split_role_counts: dict[str, dict[str, int]],
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    selected: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    used_task_ids: set[str] = set()
    pools: dict[str, list[dict[str, Any]]] = {}
    for role in ROLE_ORDER:
        role_rows = [row for row in candidates if role in row["roles"]]
        role_rows.sort(key=lambda row: _role_sort_key(row, role=role, seed=seed))
        pools[role] = role_rows

    for split in SPLITS:
        for role in ROLE_ORDER:
            need = int(split_role_counts[split].get(role, 0))
            taken = []
            for row in pools[role]:
                if len(taken) >= need:
                    break
                if row["task_id"] in used_task_ids:
                    continue
                used_task_ids.add(row["task_id"])
                copied = dict(row)
                copied["role"] = role
                taken.append(copied)
            if len(taken) < need:
                raise SystemExit(
                    f"Not enough Code P0 rows for split={split} role={role}: need={need}, got={len(taken)}"
                )
            selected[split].extend(taken)
        selected[split].sort(key=lambda row: stable_hash({"seed": seed, "split": split, "task_id": row["task_id"]}))
    return selected


def _role_sort_key(row: dict[str, Any], *, role: str, seed: int) -> tuple[Any, ...]:
    score = float(row["score"])
    if role == "generation":
        rank_score = -score
    elif role == "frontier":
        rank_score = abs(score - 4.0)
    elif role == "partial_edge":
        rank_score = abs(score - 3.2)
    else:
        rank_score = score
    primary = _primary_code_tag(row["tags"])
    return (rank_score, primary, stable_hash({"seed": seed, "role": role, "task_id": row["task_id"]}))


def _candidate_roles(question: str, tags: list[str], score: float) -> list[str]:
    tag_set = set(tags)
    roles = []
    if score >= 4.6 or tag_set & {"graph", "dynamic_programming"}:
        roles.append("generation")
    if 2.8 <= score <= 5.2:
        roles.append("frontier")
    if tag_set & {"format_sensitive", "string", "simulation"}:
        roles.append("partial_edge")
    if score <= 3.4 and len(question) <= 2200:
        roles.append("stable")
    return roles or ["frontier"]


def _candidate_to_seed_record(
    candidate: dict[str, Any],
    *,
    split: str,
    tag: str,
    created_at: str,
    reward_tests: int,
    guard_tests: int,
    seed: int,
) -> dict[str, Any]:
    raw = candidate["raw"]
    task_id = candidate["task_id"]
    question = candidate["question"]
    inputs = list(raw.get("test_input") or [])
    outputs = list(raw.get("test_output") or [])
    total = min(len(inputs), len(outputs))
    order = list(range(total))
    rng = random.Random(int(stable_hash({"seed": seed, "split": split, "task_id": task_id})[:16], 16))
    rng.shuffle(order)
    reward_indices = sorted(order[: min(reward_tests, total)])
    guard_start = min(reward_tests, total)
    guard_indices = sorted(order[guard_start : min(guard_start + guard_tests, total)])
    if not guard_indices:
        guard_indices = reward_indices[-min(len(reward_indices), guard_tests) :]

    reward_inputs = [inputs[idx] for idx in reward_indices]
    reward_outputs = [outputs[idx] for idx in reward_indices]
    guard_inputs = [inputs[idx] for idx in guard_indices]
    guard_outputs = [outputs[idx] for idx in guard_indices]
    prompt_hash = stable_hash({"task": "code", "source": candidate["source_path"], "task_id": task_id, "question": question})
    prompt_id = f"code_p0v3__{prompt_hash[:16]}"
    messages = _cure_code_messages(question)
    metadata = {
        "source_dataset": "CodeContests_train",
        "source_path": candidate["source_path"],
        "source_row": candidate["idx"],
        "task_id": task_id,
        "question_id": task_id,
        "exe_method": raw.get("exe_method") or "stdin",
        "test_time_limit": raw.get("test_time_limit", 1),
        "test_input": reward_inputs,
        "test_output": reward_outputs,
        "reward_test_input": reward_inputs,
        "reward_test_output": reward_outputs,
        "reward_test_indices": reward_indices,
        "guard_test_input": guard_inputs,
        "guard_test_output": guard_outputs,
        "guard_test_indices": guard_indices,
        "all_test_count": total,
        "code_tags": candidate["tags"],
        "primary_code_tag": _primary_code_tag(candidate["tags"]),
        "code_bank_role": candidate["role"],
        "selection_score": candidate["score"],
    }
    return {
        "prompt_id": prompt_id,
        "task": "code",
        "source": candidate["source_path"],
        "source_row": candidate["idx"],
        "split": f"{tag}_{split}",
        "prompt": question,
        "messages": messages,
        "reference": {"answer": None, "response": "", "metadata": metadata},
        "verifier": {"name": "cure_code_pass_rate", "config": {"source": "CodeContests_train", "test_slice": "reward"}},
        "tags": sorted(
            {
                tag,
                f"{tag}:{split}",
                "code_p0_v3",
                "codecontests_train",
                "cure_style",
                f"code_role:{candidate['role']}",
                *[f"code_tag:{item}" for item in candidate["tags"]],
            }
        ),
        "difficulty": None,
        "prompt_hash": prompt_hash,
        "code_p0_calibration": {
            "format": "code_p0_v3",
            "created_at": created_at,
            "split": split,
            "role": candidate["role"],
            "role_semantics": _role_semantics(candidate["role"]),
            "leakage_policy": "Selected from CodeContests train only; no LiveBench/LiveCodeBench/CURE official eval prompt, hidden test, generated code, or output is copied.",
            "reward_policy": "metadata.test_input/test_output is the train reward slice; guard_test_input/output is held out for audit/monitoring.",
        },
    }


def _row_blueprint(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row["reference"]["metadata"]
    return {
        "format": "code_p0_v3_blueprint",
        "prompt_id": row["prompt_id"],
        "split": row["code_p0_calibration"]["split"],
        "role": row["code_p0_calibration"]["role"],
        "task_id": metadata["task_id"],
        "source_row": metadata["source_row"],
        "primary_code_tag": metadata["primary_code_tag"],
        "code_tags": metadata["code_tags"],
        "reward_test_count": len(metadata["reward_test_input"]),
        "guard_test_count": len(metadata["guard_test_input"]),
        "all_test_count": metadata["all_test_count"],
        "selection_score": metadata["selection_score"],
        "source": row["source"],
    }


def _summary(
    *,
    args: argparse.Namespace,
    created_at: str,
    source_path: Path,
    output_dir: Path,
    candidates: list[dict[str, Any]],
    selected: dict[str, list[dict[str, Any]]],
    outputs: dict[str, str],
    blueprint_path: Path,
    exclude_task_ids: set[str],
    split_role_counts: dict[str, dict[str, int]],
) -> dict[str, Any]:
    split_summary = {}
    for split, rows in selected.items():
        split_summary[split] = {
            "rows": len(rows),
            "role_counts": dict(sorted(Counter(row["role"] for row in rows).items())),
            "tag_counts": _top_counter(tag for row in rows for tag in row["tags"]),
            "primary_tag_counts": dict(sorted(Counter(_primary_code_tag(row["tags"]) for row in rows).items())),
            "mean_score": sum(float(row["score"]) for row in rows) / len(rows) if rows else 0.0,
        }
    return {
        "format": "code_p0_calibration_bank_v3",
        "created_at": created_at,
        "seed": int(args.seed),
        "tag": args.tag,
        "inputs": {"codecontests_train": str(source_path), "exclude_manifest": [str(Path(path).expanduser().resolve()) for path in args.exclude_manifest]},
        "outputs": {**outputs, "blueprints": str(blueprint_path), "summary": str(output_dir / "summary.json"), "readme": str(output_dir / "README.md")},
        "source_counts": {
            "source_rows": len(read_json(source_path)),
            "excluded_task_ids": len(exclude_task_ids),
            "candidate_rows": len(candidates),
            "candidate_role_memberships": dict(sorted(Counter(role for row in candidates for role in row["roles"]).items())),
            "candidate_tags": _top_counter(tag for row in candidates for tag in row["tags"]),
        },
        "split_role_counts_requested": split_role_counts,
        "split_summary": split_summary,
        "reward_policy": {
            "reward_tests_per_prompt": int(args.reward_tests),
            "guard_tests_per_prompt": int(args.guard_tests),
            "min_tests": int(args.min_tests),
            "training_reward_adapter": "CodeRewardAdapter reads reference.metadata.test_input/test_output, which this builder sets to reward_test_input/output.",
        },
        "leakage_policy": "Only CodeContests train rows are used. Formal CURE/LiveBench/LiveCodeBench case studies may motivate tags, but their prompts/tests/outputs are not read by this builder.",
    }


def _load_excluded_task_ids(paths: list[str]) -> set[str]:
    excluded = set()
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                metadata = row.get("reference", {}).get("metadata", {}) if isinstance(row.get("reference"), dict) else {}
                for key in ("task_id", "question_id"):
                    value = metadata.get(key)
                    if value is not None:
                        excluded.add(str(value))
                value = row.get("source_row")
                if value is not None and str(row.get("source", "")).endswith("CodeContests_train.json"):
                    excluded.add(str(value))
    return excluded


def _parse_split_role_counts(items: list[str]) -> dict[str, dict[str, int]]:
    result = {split: dict(DEFAULT_SPLIT_ROLE_COUNTS[split]) for split in SPLITS}
    for item in items:
        if ":" not in item:
            raise SystemExit(f"Invalid --split-role-count {item!r}; expected split:role=count,...")
        split, raw_counts = item.split(":", 1)
        split = split.strip()
        if split not in result:
            raise SystemExit(f"Unsupported split in --split-role-count: {split!r}")
        counts = {role: 0 for role in ROLE_ORDER}
        for part in raw_counts.split(","):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                raise SystemExit(f"Invalid role count {part!r}")
            role, raw_value = part.split("=", 1)
            role = role.strip()
            if role not in counts:
                raise SystemExit(f"Unsupported Code P0 role: {role!r}")
            counts[role] = int(raw_value)
        result[split] = counts
    return result


def _code_priority_score(question: str, tags: list[str], total_tests: int) -> float:
    tag_weights = {
        "math": 1.4,
        "format_sensitive": 1.2,
        "greedy": 1.2,
        "array": 1.0,
        "string": 1.0,
        "simulation": 1.0,
        "graph": 1.6,
        "dynamic_programming": 1.8,
        "stdin_stdout": 0.2,
    }
    lowered = question.lower()
    score = sum(tag_weights.get(tag, 0.2) for tag in tags)
    score += min(1.5, total_tests / 10.0)
    if len(question) > 1600:
        score += 0.4
    if any(marker in lowered for marker in ("mod", "minimum", "maximum", "shortest", "lexicographic")):
        score += 0.4
    if any(marker in lowered for marker in ("multiple test", "test cases", "each test")):
        score += 0.3
    return score


def _code_tags(text: str) -> list[str]:
    lowered = text.lower()
    tags = {"stdin_stdout"}
    if any(word in lowered for word in ("sum", "product", "gcd", "mod", "integer", "prime", "divisible", "number", "formula", "probability")):
        tags.add("math")
    if any(word in lowered for word in ("array", "sequence", "list", "permutation", "subarray", "prefix")):
        tags.add("array")
    if any(word in lowered for word in ("string", "substring", "character", "lexicographic", "palindrome")):
        tags.add("string")
    if any(word in lowered for word in ("greedy", "minimum", "maximum", "at most", "at least", "choose", "remove", "operation")):
        tags.add("greedy")
    if any(word in lowered for word in ("simulate", "simulation", "move", "turn", "round", "step", "game")):
        tags.add("simulation")
    if any(word in lowered for word in ("graph", "tree", "edge", "vertex", "node", "path", "connected", "shortest")):
        tags.add("graph")
    if any(word in lowered for word in ("dynamic programming", "dp", "subsequence")):
        tags.add("dynamic_programming")
    if any(word in lowered for word in ("print", "output", "format", "yes", "no", "case")):
        tags.add("format_sensitive")
    return sorted(tags)


def _primary_code_tag(tags: list[str]) -> str:
    for tag in ("dynamic_programming", "graph", "math", "greedy", "array", "string", "simulation", "format_sensitive"):
        if tag in tags:
            return tag
    return "stdin_stdout"


def _cure_code_messages(question: str) -> list[dict[str, str]]:
    content = (
        "You need to think first then write python script. You should use input() to input and print() to output in your script. "
        "Your code should output the results based on the input read in, rather than generating the given test example.\n"
        "This is the problem:\n"
        f"{question}"
    )
    return [
        {"role": "system", "content": "You are a helpful assistant help user solve problems."},
        {"role": "user", "content": content},
    ]


def _role_semantics(role: str) -> str:
    return {
        "generation": "Intended hard generation rows for OPD recoverability and pass@1 improvement.",
        "frontier": "Intended non-saturated rows for GRPO relative advantage.",
        "partial_edge": "Format/boundary-sensitive rows for hidden-like unit-test pressure.",
        "stable": "Easier executable rows reserved for non-collapse and retention diagnostics.",
    }[role]


def _top_counter(values: Any, *, limit: int = 20) -> list[list[Any]]:
    return [[key, value] for key, value in Counter(str(value) for value in values if value is not None).most_common(limit)]


def _render_readme(summary: dict[str, Any]) -> str:
    lines = [
        "# Code P0 v3 Calibration Bank",
        "",
        f"created_at: `{summary['created_at']}`",
        f"seed: `{summary['seed']}`",
        "",
        "## Purpose",
        "",
        "Code-only train/monitor/guard bank for checking whether ExpertGym can turn Code reward gains into formal CURE gains.",
        "",
        "## Outputs",
        "",
    ]
    for key, path in summary["outputs"].items():
        lines.append(f"- `{key}`: `{path}`")
    lines.extend(["", "## Split Summary", "", "| split | rows | generation | frontier | partial_edge | stable |", "|---|---:|---:|---:|---:|---:|"])
    for split in SPLITS:
        item = summary["split_summary"][split]
        roles = item["role_counts"]
        lines.append(
            f"| {split} | {item['rows']} | {roles.get('generation', 0)} | {roles.get('frontier', 0)} | {roles.get('partial_edge', 0)} | {roles.get('stable', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Policy",
            "",
            "- Source is CodeContests train only.",
            "- `metadata.test_input/test_output` is the reward slice used by the current CodeRewardAdapter.",
            "- `metadata.guard_test_input/output` is held out for auditing and future guard reward work.",
            "- Splits are disjoint by CodeContests `task_id`.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codecontests", default=DEFAULT_CODECONTESTS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=20260518)
    parser.add_argument("--tag", default="code_p0_v3_20260518")
    parser.add_argument("--min-tests", type=int, default=10)
    parser.add_argument("--reward-tests", type=int, default=6)
    parser.add_argument("--guard-tests", type=int, default=4)
    parser.add_argument("--exclude-manifest", action="append", default=[], help="Existing prompt manifest whose CodeContests task ids should be excluded.")
    parser.add_argument(
        "--split-role-count",
        action="append",
        default=[],
        help="Override split counts, e.g. train:generation=24,frontier=20,partial_edge=12,stable=8",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
