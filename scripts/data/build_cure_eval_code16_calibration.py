#!/usr/bin/env python3
"""Build 16 formal CURE eval code prompts plus verified expert OPD anchors.

This builder intentionally imports a small number of LiveBench/LiveCodeBench
formal-eval prompts.  The reward slice is the same first-8 official CURE tests
used by the local CodeRewardAdapter and by CURE's formal eval config.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from opvec.data.io import read_jsonl, write_jsonl
from opvec.data.schema import stable_hash, validate_rollout_row, validate_seed_record
from opvec.rewards.simple import CodeRewardAdapter


DEFAULT_BASE_MANIFEST = Path("/tmp/shared-storage/OnPolicy/data/calibration/qbank_c033333_paper96_balanced32_seed20260514.prompts.jsonl")
DEFAULT_CURE_DATA_ROOT = Path("/mnt/cache/wuruixiao/users/lsc/CURE/data")
DEFAULT_TEMP_ROOT = Path("/tmp/shared-storage/AgentMerging_plan/evaluation_artifacts/cure_temp_backup_20260503")
DEFAULT_OUTPUT_DIR = Path("/tmp/shared-storage/OnPolicy/data/calibration/20260519_l5_cure_eval_code16")
DEFAULT_SELECTION = {
    "LiveBench": [2, 8, 12, 14, 15, 23, 35, 38],
    "LiveCodeBench": [1, 4, 6, 7, 10, 11, 12, 15],
}
EXPERTS = {
    "reasonflux": "ReasonFlux-Coder-7B",
    "deepseek_r1_distill": "DeepSeek-R1-Distill-Qwen-7B",
    "memory_agent": "RL-MemoryAgent-7B",
}
BASE_MODEL_NAME = "Qwen2.5-7b-instruct"


def main() -> None:
    args = parse_args()
    created_at = datetime.now(timezone.utc).isoformat()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_indices = _parse_selection(args.selection)
    official_rows = _load_official_rows(Path(args.cure_data_root), selected_indices)
    base_status = _load_model_success_status(
        Path(args.temp_root),
        model_name=BASE_MODEL_NAME,
        selected_indices=selected_indices,
        max_tests=int(args.max_tests),
        max_positives_per_expert=0,
    )
    expert_status = {
        expert_name: _load_model_success_status(
            Path(args.temp_root),
            model_name=model_name,
            selected_indices=selected_indices,
            max_tests=int(args.max_tests),
            max_positives_per_expert=int(args.max_positives_per_expert),
        )
        for expert_name, model_name in EXPERTS.items()
    }

    prompt_rows, expert_rows, selection_blueprints = build_rows(
        official_rows,
        base_status=base_status,
        expert_status=expert_status,
        args=args,
        created_at=created_at,
    )
    base_rows = read_jsonl(Path(args.base_manifest))
    merged_rows = base_rows + prompt_rows

    prompt_path = output_dir / "cure_eval_code16_livebench8_livecodebench8_seed20260519.prompts.jsonl"
    expert_path = output_dir / "cure_eval_code16_expert_success_rollouts_seed20260519.jsonl"
    merged_path = output_dir / "qbank_c033333_paper96_plus_cure_code16_seed20260519.prompts.jsonl"
    blueprint_path = output_dir / "cure_eval_code16_selection_blueprints.jsonl"
    summary_path = output_dir / "qbank_c033333_paper96_plus_cure_code16_seed20260519.summary.json"
    readme_path = output_dir / "README.md"

    write_jsonl(prompt_path, prompt_rows)
    write_jsonl(expert_path, expert_rows)
    write_jsonl(merged_path, merged_rows)
    write_jsonl(blueprint_path, selection_blueprints)
    summary = {
        "format": "cure_eval_code16_calibration_v1",
        "created_at": created_at,
        "seed": int(args.seed),
        "inputs": {
            "base_manifest": str(Path(args.base_manifest).expanduser().resolve()),
            "cure_data_root": str(Path(args.cure_data_root).expanduser().resolve()),
            "temp_root": str(Path(args.temp_root).expanduser().resolve()),
            "base_model": BASE_MODEL_NAME,
            "experts": EXPERTS,
        },
        "outputs": {
            "prompts": str(prompt_path),
            "expert_rollouts": str(expert_path),
            "merged_prompts": str(merged_path),
            "blueprints": str(blueprint_path),
            "summary": str(summary_path),
            "readme": str(readme_path),
        },
        "counts": {
            "base_manifest_rows": len(base_rows),
            "added_prompts": len(prompt_rows),
            "merged_rows": len(merged_rows),
            "added_task_counts": dict(Counter(row["task"] for row in prompt_rows)),
            "added_dataset_counts": dict(Counter(row["reference"]["metadata"]["source_dataset"] for row in prompt_rows)),
            "expert_rollout_rows": len(expert_rows),
            "expert_positive_samples": sum(len(row.get("samples") or []) for row in expert_rows),
        },
        "selection": {
            "requested_indices": selected_indices,
            "require_base_fail": bool(args.require_base_fail),
            "min_positive_experts": int(args.min_positive_experts),
            "max_tests": int(args.max_tests),
            "source_policy": (
                "Rows are formal CURE eval prompts selected from hard-vs-base indices: "
                "base Qwen2.5-7B-Instruct has no passing sample, while at least one "
                "code-capable expert has a passing formal-eval trajectory."
            ),
            "rows": selection_blueprints,
        },
        "reward_alignment": {
            "training_adapter": "CodeRewardAdapter",
            "metadata_fields": "reference.metadata.test_input/test_output/test_time_limit",
            "test_slice": f"first_{int(args.max_tests)}_official_cure_tests",
            "formal_eval_config": "CURE/evaluation/evaluation_config.py uses max_test=8 for final eval.",
        },
        "leakage_policy": (
            "This file intentionally imports 16 formal LiveBench/LiveCodeBench prompts/tests/model-output positives "
            "as requested for eval-distribution calibration.  Every source row, temp file, and selected expert "
            "sample is recorded for audit."
        ),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    readme_path.write_text(render_readme(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def build_rows(
    official_rows: dict[str, dict[int, dict[str, Any]]],
    *,
    base_status: dict[str, dict[int, dict[str, Any]]],
    expert_status: dict[str, dict[str, dict[int, dict[str, Any]]]],
    args: argparse.Namespace,
    created_at: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    adapter = CodeRewardAdapter()
    prompt_rows: list[dict[str, Any]] = []
    expert_rows: list[dict[str, Any]] = []
    blueprints: list[dict[str, Any]] = []
    for dataset in sorted(official_rows):
        for source_row in sorted(official_rows[dataset]):
            raw = official_rows[dataset][source_row]
            positives = []
            positive_experts = []
            for expert_name, by_dataset in expert_status.items():
                status = by_dataset.get(dataset, {}).get(source_row, {})
                samples = list(status.get("success_samples") or [])
                if samples:
                    positive_experts.append(expert_name)
                    for sample in samples:
                        positives.append((expert_name, sample))
            base_success_count = int(base_status.get(dataset, {}).get(source_row, {}).get("success_count", 0))
            if args.require_base_fail and base_success_count > 0:
                raise RuntimeError(f"{dataset}[{source_row}] is not base-fail: base_success_count={base_success_count}")
            if len(positive_experts) < int(args.min_positive_experts):
                raise RuntimeError(
                    f"{dataset}[{source_row}] has only {len(positive_experts)} positive experts: {positive_experts}"
                )

            prompt_row = _make_prompt_row(raw, dataset=dataset, source_row=source_row, args=args, created_at=created_at)
            validate_seed_record(prompt_row)
            expert_row = _make_expert_rollout_row(prompt_row, positives, adapter=adapter, args=args, created_at=created_at)
            validate_rollout_row(expert_row)
            prompt_rows.append(prompt_row)
            expert_rows.append(expert_row)
            blueprints.append(
                {
                    "format": "cure_eval_code16_selection_blueprint_v1",
                    "prompt_id": prompt_row["prompt_id"],
                    "dataset": dataset,
                    "source_row": source_row,
                    "base_success_count": base_success_count,
                    "positive_experts": positive_experts,
                    "positive_sample_count": len(expert_row["samples"]),
                    "test_count_total": min(len(raw.get("test_input") or []), len(raw.get("test_output") or [])),
                    "reward_test_count": len(prompt_row["reference"]["metadata"]["test_input"]),
                    "question_preview": str(raw.get("question") or "")[:240],
                    "selection_reason": "formal CURE hard-vs-base code anchor with verified expert success",
                }
            )
    return prompt_rows, expert_rows, blueprints


def _make_prompt_row(
    raw: dict[str, Any],
    *,
    dataset: str,
    source_row: int,
    args: argparse.Namespace,
    created_at: str,
) -> dict[str, Any]:
    question = str(raw.get("question") or "").strip()
    test_input = list(raw.get("test_input") or [])
    test_output = list(raw.get("test_output") or [])
    total = min(len(test_input), len(test_output), int(args.max_tests))
    reward_inputs = test_input[:total]
    reward_outputs = test_output[:total]
    prompt = cure_formal_prompt(question)
    prompt_hash = stable_hash(
        {
            "task": "code",
            "source_dataset": dataset,
            "source_row": source_row,
            "question": question,
            "test_slice": f"first_{int(args.max_tests)}",
        }
    )
    prompt_id = f"code__cure_{dataset.lower()}_{source_row:04d}_{prompt_hash[:8]}"
    metadata = {
        "source_dataset": dataset,
        "source_path": str(Path(args.cure_data_root).expanduser().resolve() / f"{dataset}.json"),
        "source_row": int(source_row),
        "task_id": f"{dataset}:{source_row}",
        "question_id": f"{dataset}:{source_row}",
        "test_method": raw.get("test_method") or "stdio",
        "exe_method": raw.get("test_method") or "stdio",
        "test_time_limit": raw.get("test_time_limit", 8),
        "test_input": reward_inputs,
        "test_output": reward_outputs,
        "reward_test_input": reward_inputs,
        "reward_test_output": reward_outputs,
        "reward_test_indices": list(range(total)),
        "all_test_count": min(len(test_input), len(test_output)),
        "code_bank_role": "formal_cure_eval_anchor",
        "prompt_template": "CURE/evaluation/evaluation_config.py::system_prompts",
    }
    return {
        "prompt_id": prompt_id,
        "group_id": prompt_id,
        "task": "code",
        "source": metadata["source_path"],
        "source_row": int(source_row),
        "split": "l5_cure_eval_code16_calibration",
        "prompt": prompt,
        "messages": [],
        "reference": {"answer": None, "response": "", "metadata": metadata},
        "verifier": {
            "name": "cure_code_pass_rate",
            "config": {"source": dataset, "test_slice": f"first_{int(args.max_tests)}_formal_cure"},
        },
        "tags": [
            "code",
            "cure",
            "formal_eval_calibration",
            "l5_cure_eval_code16",
            f"cure_dataset:{dataset}",
            "code_role:formal_cure_eval_anchor",
        ],
        "difficulty": "hard_vs_base",
        "prompt_hash": prompt_hash,
        "cure_eval_code16_calibration": {
            "format": "cure_eval_code16_calibration_v1",
            "created_at": created_at,
            "selection_policy": "formal eval hard-vs-base row with expert positive trajectory",
            "reward_policy": "CodeRewardAdapter executes reference.metadata.test_input/test_output, capped at first official CURE tests.",
        },
    }


def _make_expert_rollout_row(
    prompt_row: dict[str, Any],
    positives: list[tuple[str, dict[str, Any]]],
    *,
    adapter: CodeRewardAdapter,
    args: argparse.Namespace,
    created_at: str,
) -> dict[str, Any]:
    samples = []
    for index, (expert_name, source_sample) in enumerate(positives):
        text = str(source_sample.get("text") or "")
        score = adapter.score(prompt_row, text).as_dict()
        if float(score.get("reward", 0.0)) < float(args.positive_threshold):
            raise RuntimeError(
                f"Expert sample failed local CodeRewardAdapter check: prompt={prompt_row['prompt_id']} "
                f"expert={expert_name} score={score}"
            )
        sample = {
            "sample_id": f"{prompt_row['prompt_id']}__{expert_name}__k{source_sample['sample_index']}",
            "text": text,
            "reward": float(score["reward"]),
            "task_reward": float(score["task_reward"]),
            "contract_reward": float(score.get("contract_reward", 0.0)),
            "reward_train": float(score["reward"]),
            "success": bool(score.get("success")),
            "old_logprob": None,
            "old_logprob_max_length": None,
            "length": len(text.split()),
            "opd_role": "positive",
            "opd_source": "cure_formal_eval_expert_success",
            "opd_source_policy_id": expert_name,
            "details": {
                **(score.get("details") or {}),
                "source_dataset": source_sample.get("dataset"),
                "source_row": source_sample.get("source_row"),
                "expert_name": expert_name,
                "expert_model": source_sample.get("model_name"),
                "expert_temp_file": source_sample.get("temp_file"),
                "expert_sample_index": source_sample.get("sample_index"),
                "formal_pass_count": source_sample.get("pass_count"),
                "formal_test_count": source_sample.get("test_count"),
                "text_source": source_sample.get("text_source"),
            },
        }
        samples.append(sample)
    if not samples:
        raise RuntimeError(f"No verified expert positives for {prompt_row['prompt_id']}")
    return {
        "run_id": "l5_cure_eval_code16_expert_success",
        "created_at": created_at,
        "step": 0,
        "policy_id": "cure_eval_code_expert_pool",
        "gate_checkpoint": None,
        "gate_values": {},
        "gate_id": "cure_eval_code_expert_pool",
        "group_id": prompt_row["group_id"],
        "prompt_id": prompt_row["prompt_id"],
        "task": "code",
        "prompt": prompt_row["prompt"],
        "reference": prompt_row["reference"],
        "rendered_prompt": prompt_row["prompt"],
        "samples": samples,
        "frontier": {
            "all_failure": False,
            "all_success": True,
            "num_success": len(samples),
            "num_failure": 0,
            "mean_reward": sum(float(sample["reward_train"]) for sample in samples) / len(samples),
            "std_reward": 0.0,
            "reward_field": "reward_train",
        },
        "keep_for_policy_loss": False,
        "skip_reason": "expert_positive_anchor_only",
    }


def _load_official_rows(data_root: Path, selected_indices: dict[str, list[int]]) -> dict[str, dict[int, dict[str, Any]]]:
    output: dict[str, dict[int, dict[str, Any]]] = {}
    for dataset, indices in selected_indices.items():
        path = data_root / f"{dataset}.json"
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        output[dataset] = {}
        for index in indices:
            if index >= len(payload):
                raise IndexError(f"{dataset}[{index}] out of range for {path}")
            output[dataset][index] = payload[index]
    return output


def _load_model_success_status(
    temp_root: Path,
    *,
    model_name: str,
    selected_indices: dict[str, list[int]],
    max_tests: int,
    max_positives_per_expert: int,
) -> dict[str, dict[int, dict[str, Any]]]:
    status: dict[str, dict[int, dict[str, Any]]] = {}
    for dataset, indices in selected_indices.items():
        path = _temp_path(temp_root, model_name=model_name, dataset=dataset)
        selected = set(int(index) for index in indices)
        status[dataset] = {}
        for idx, row in _iter_selected_temp_rows(path, selected):
            successes = _success_samples(
                row,
                dataset=dataset,
                source_row=idx,
                model_name=model_name,
                temp_file=path,
                max_tests=max_tests,
                max_positives=max_positives_per_expert,
            )
            status[dataset][idx] = {
                "success_count": len(successes),
                "success_samples": successes,
                "temp_file": str(path),
            }
        missing = sorted(selected - set(status[dataset]))
        if missing:
            raise RuntimeError(f"Missing selected rows in {path}: {missing}")
    return status


def _iter_selected_temp_rows(path: Path, selected: set[int]) -> Iterable[tuple[int, dict[str, Any]]]:
    try:
        import ijson
    except ImportError as error:  # pragma: no cover - production env has ijson.
        raise RuntimeError("ijson is required to stream CURE temp outputs without loading multi-GB files") from error
    if not path.exists():
        raise FileNotFoundError(path)
    max_index = max(selected) if selected else -1
    with path.open("rb") as handle:
        for idx, row in enumerate(ijson.items(handle, "item")):
            if idx in selected:
                if not isinstance(row, dict):
                    raise RuntimeError(f"Temp row is not an object: {path}[{idx}]")
                yield idx, row
            if idx >= max_index:
                break


def _success_samples(
    row: dict[str, Any],
    *,
    dataset: str,
    source_row: int,
    model_name: str,
    temp_file: Path,
    max_tests: int,
    max_positives: int,
) -> list[dict[str, Any]]:
    bool_table = list(row.get("test_bool_table") or [])
    full_generations = list(row.get("full_code_generation") or [])
    generated_codes = list(row.get("generated_code") or [])
    successes = []
    for sample_index, case_bools in enumerate(bool_table):
        flags = [bool(item) for item in list(case_bools or [])[:max_tests]]
        if not flags or not all(flags):
            continue
        text = str(full_generations[sample_index]) if sample_index < len(full_generations) else ""
        text_source = "full_code_generation"
        if "```python" not in text and sample_index < len(generated_codes):
            text = "```python\n" + str(generated_codes[sample_index]).strip() + "\n```"
            text_source = "generated_code_wrapped"
        if not text.strip():
            continue
        successes.append(
            {
                "dataset": dataset,
                "source_row": int(source_row),
                "model_name": model_name,
                "temp_file": str(temp_file),
                "sample_index": int(sample_index),
                "text": text,
                "text_source": text_source,
                "pass_count": sum(flags),
                "test_count": len(flags),
            }
        )
    successes.sort(key=lambda item: (item["sample_index"], stable_hash([model_name, dataset, source_row, item["sample_index"]])))
    if max_positives <= 0:
        return []
    return successes[:max_positives]


def _temp_path(temp_root: Path, *, model_name: str, dataset: str) -> Path:
    return temp_root / f"outputs-eval-.mnt.cache.wuruixiao.models.{model_name}-{dataset}.json"


def cure_formal_prompt(question: str) -> str:
    special_requirements = (
        "You should use input() to input and print() to output in your script. "
        "Your code should output the results based on the input read in, rather than generating the given test example."
    )
    return (
        "<|im_start|>You are a helpful assistant help user solve problems. "
        "<|im_end|>\n<|im_start|>User: You need to think first then write python script. "
        f"{special_requirements}\nThis is the problem:\n{question} <|im_end|>\n<|im_start|>Assistant: "
    )


def _parse_selection(items: list[str]) -> dict[str, list[int]]:
    if not items:
        return {dataset: list(indices) for dataset, indices in DEFAULT_SELECTION.items()}
    selected: dict[str, list[int]] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --selection {item!r}; expected Dataset=1,2,3")
        dataset, raw_indices = item.split("=", 1)
        indices = [int(part.strip()) for part in raw_indices.split(",") if part.strip()]
        if dataset.strip() not in {"LiveBench", "LiveCodeBench"}:
            raise ValueError(f"Unsupported CURE dataset: {dataset!r}")
        selected[dataset.strip()] = indices
    return selected


def render_readme(summary: dict[str, Any]) -> str:
    counts = summary["counts"]
    lines = [
        "# L5 CURE Eval Code16 Calibration",
        "",
        f"生成时间：`{summary['created_at']}`",
        "",
        "## 目的",
        "",
        "把正式 `LiveBench` / `LiveCodeBench` 各 8 条 hard-vs-base code 题引入 calibration，使用和 CURE formal eval 一致的官方题面 prompt 与前 8 个官方测试作为训练 reward。",
        "",
        "## 产物",
        "",
    ]
    for key, path in summary["outputs"].items():
        lines.append(f"- `{key}`: `{path}`")
    lines.extend(
        [
            "",
            "## 计数",
            "",
            f"- added prompts: `{counts['added_prompts']}`",
            f"- merged rows: `{counts['merged_rows']}`",
            f"- expert rollout rows: `{counts['expert_rollout_rows']}`",
            f"- expert positive samples: `{counts['expert_positive_samples']}`",
            "",
            "## Reward 对齐",
            "",
            "- seed record 的 `reference.metadata.test_input/test_output` 直接来自 CURE 官方数据。",
            "- 当前 `CodeRewardAdapter` 会执行这些测试并返回 pass-rate；默认 cap 是 8，和 formal eval 的 `max_test=8` 一致。",
            "- expert positive 轨迹来自已有 formal eval temp output，脚本会重新用本地 `CodeRewardAdapter` 验证 reward=1.0 后才写入 OPD 文件。",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, default=DEFAULT_BASE_MANIFEST)
    parser.add_argument("--cure-data-root", type=Path, default=DEFAULT_CURE_DATA_ROOT)
    parser.add_argument("--temp-root", type=Path, default=DEFAULT_TEMP_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--selection", action="append", default=[], help="Override rows, e.g. LiveBench=2,8,12")
    parser.add_argument("--max-tests", type=int, default=8)
    parser.add_argument("--max-positives-per-expert", type=int, default=1)
    parser.add_argument("--min-positive-experts", type=int, default=1)
    parser.add_argument("--positive-threshold", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--no-require-base-fail", dest="require_base_fail", action="store_false")
    parser.set_defaults(require_base_fail=True)
    return parser.parse_args()


if __name__ == "__main__":
    os.environ.setdefault("OPVEC_CODE_REWARD_MAX_TESTS", "8")
    main()
