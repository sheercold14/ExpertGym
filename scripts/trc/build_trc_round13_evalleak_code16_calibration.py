#!/usr/bin/env python3
"""Build Round13 TRC formal-code eval-leak diagnostic calibrations.

This builder intentionally creates diagnostic data only.  It uses formal
LiveBench/LiveCodeBench prompts and verified expert-positive trajectories from
the L5 CURE Code16 bank to test whether the current hidden-state TRC objective
can learn code behavior when calibration is aligned with the formal eval
distribution.
"""

from __future__ import annotations

import argparse
import hashlib
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


DEFAULT_OUTPUT_DIR = Path(
    "/tmp/shared-storage/OnPolicy/data/calibration/20260520_trc_round13_evalleak_code16"
)
DEFAULT_STABLE_TM_BANK = Path(
    "/tmp/shared-storage/OnPolicy/data/calibration/20260520_trc_round12_rfonly_tagquota_v1/"
    "r12d_rfonly_primarytag_quota_stablelate3/trc96_expert_trajectories.jsonl"
)
DEFAULT_L5_CODE16 = Path(
    "/tmp/shared-storage/OnPolicy/data/calibration/20260519_l5_cure_eval_code16/"
    "cure_eval_code16_expert_success_rollouts_seed20260519.jsonl"
)

EXPERT_TO_GATE_EXPERT = {
    "reasonflux": "code",
    "memory_agent": "memory",
    "deepseek_r1_distill": "reasoning",
}
VARIANTS = {
    "rfmem_only": ("reasonflux", "memory_agent"),
    "all_with_r1": ("reasonflux", "memory_agent", "deepseek_r1_distill"),
}


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    stable_rows = read_jsonl(Path(args.stable_tm_bank).expanduser())
    tool_rows = select_task_rows(stable_rows, "tool", expected=32)
    memory_rows = select_task_rows(stable_rows, "memory", expected=32)
    l5_rows = read_jsonl(Path(args.l5_code16).expanduser())

    reports: dict[str, Any] = {}
    for variant, expert_allowlist in VARIANTS.items():
        variant_dir = output_root / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        code_rows, code_report = build_code_rows(
            l5_rows,
            allowlist=set(expert_allowlist),
            variant=variant,
            reasoning_context_chars=int(args.reasoning_context_chars),
        )
        rows: list[dict[str, Any]] = []
        rows.extend(with_round13_provenance(row, variant=variant, component="stable_tool") for row in tool_rows)
        rows.extend(with_round13_provenance(row, variant=variant, component="stable_memory") for row in memory_rows)
        rows.extend(code_rows)
        renumber_trajectory_ids(rows, variant=variant)
        validate_rows(rows, variant=variant)

        out_jsonl = variant_dir / "trc_expert_trajectories.jsonl"
        write_jsonl(out_jsonl, rows)
        summary = build_summary(
            rows=rows,
            out_jsonl=out_jsonl,
            variant=variant,
            expert_allowlist=expert_allowlist,
            code_report=code_report,
            args=args,
        )
        write_json(variant_dir / "summary.json", summary)
        (variant_dir / "README.md").write_text(render_markdown(summary), encoding="utf-8")
        reports[variant] = summary

    combined = {
        "format": "trc_round13_evalleak_code16_combined_summary_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_root),
        "variants": {name: report["output"] for name, report in reports.items()},
        "variant_task_counts": {name: report["task_counts"] for name, report in reports.items()},
        "variant_code_source_counts": {
            name: report["code_selection"]["source_expert_counts"] for name, report in reports.items()
        },
        "leakage_policy": "Diagnostic only: formal LiveBench/LiveCodeBench prompts and verified positive expert trajectories.",
    }
    write_json(output_root / "summary.json", combined)
    (output_root / "README.md").write_text(render_combined_markdown(combined, reports), encoding="utf-8")
    print(json.dumps(combined, ensure_ascii=False, indent=2, sort_keys=True))


def build_code_rows(
    l5_rows: list[dict[str, Any]],
    *,
    allowlist: set[str],
    variant: str,
    reasoning_context_chars: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    per_prompt = Counter()
    source_counts = Counter()
    dataset_counts = Counter()
    ability_span_counts = Counter()
    for row_index, source_row in enumerate(l5_rows):
        prompt_id = str(source_row.get("prompt_id") or source_row.get("group_id") or f"code16_{row_index:04d}")
        reference = deepcopy(source_row.get("reference") or {})
        metadata = dict(reference.get("metadata") or {})
        dataset = str(metadata.get("source_dataset") or infer_dataset_from_prompt_id(prompt_id))
        for sample_index, sample in enumerate(source_row.get("samples") or []):
            source_expert = sample_source_expert(sample)
            if source_expert not in allowlist:
                continue
            reward_train = safe_float(sample.get("reward_train", sample.get("reward", 0.0)))
            if not (bool(sample.get("success")) or reward_train >= 1.0):
                continue
            raw_response = str(sample.get("text") or sample.get("response") or "")
            if not raw_response.strip():
                continue
            compact_response, spans = compact_code_response(
                raw_response,
                reasoning_context_chars=reasoning_context_chars,
            )
            if not compact_response.strip():
                continue
            gate_expert = EXPERT_TO_GATE_EXPERT[source_expert]
            sample_id = str(sample.get("sample_id") or f"{prompt_id}__k{sample_index}")
            code_row = {
                "format": "trc_expert_trajectory_v1",
                "trajectory_id": f"round13__{variant}__code__{sample_id}",
                "task": "code",
                "expert": gate_expert,
                "source_name": f"l5_cure_code16_{source_expert}",
                "source_path": str(DEFAULT_L5_CODE16),
                "prompt_id": prompt_id,
                "group_id": str(source_row.get("group_id") or prompt_id),
                "sample_id": sample_id,
                "prompt": str(source_row.get("prompt") or ""),
                "rendered_prompt": str(source_row.get("rendered_prompt") or source_row.get("prompt") or ""),
                "response": compact_response,
                "reference": reference,
                "reward": safe_float(sample.get("reward", reward_train)),
                "reward_train": reward_train,
                "task_reward": safe_float(sample.get("task_reward", reward_train)),
                "length": len(compact_response.split()),
                "success": True,
                "row_metadata": {
                    "round13_evalleak_code16": {
                        "variant": variant,
                        "eval_leak_diagnostic": True,
                        "source_dataset": dataset,
                        "source_prompt_id": prompt_id,
                        "source_sample_id": sample_id,
                        "source_expert": source_expert,
                        "gate_expert": gate_expert,
                        "reward_test_count": len(metadata.get("test_input") or metadata.get("reward_test_input") or []),
                        "prompt_template": metadata.get("prompt_template"),
                        "task_id": metadata.get("task_id"),
                    }
                },
                "sample_metadata": {
                    "details": sample.get("details"),
                    "ability_spans": spans,
                    "raw_response_sha256": hashlib.sha256(raw_response.encode("utf-8")).hexdigest(),
                    "raw_response_chars": len(raw_response),
                    "compact_response_chars": len(compact_response),
                    "compact_policy": "critical reasoning tail plus final fenced code block when available",
                },
            }
            rows.append(code_row)
            per_prompt[prompt_id] += 1
            source_counts[source_expert] += 1
            dataset_counts[dataset] += 1
            ability_span_counts.update(span["kind"] for span in spans)

    rows.sort(
        key=lambda item: (
            str(item["reference"].get("metadata", {}).get("source_dataset") or ""),
            item["prompt_id"],
            item["source_name"],
            item["sample_id"],
        )
    )
    return rows, {
        "selected_rows": len(rows),
        "unique_prompts": len(per_prompt),
        "duplicate_prompt_rows": sum(count - 1 for count in per_prompt.values()),
        "duplicate_prompt_details": {key: value for key, value in sorted(per_prompt.items()) if value > 1},
        "source_expert_counts": dict(sorted(source_counts.items())),
        "gate_expert_counts": dict(sorted(Counter(row["expert"] for row in rows).items())),
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "ability_span_counts": dict(sorted(ability_span_counts.items())),
    }


def compact_code_response(raw_response: str, *, reasoning_context_chars: int) -> tuple[str, list[dict[str, Any]]]:
    matches = list(re.finditer(r"```(?:python|py)?\s*\n?(.*?)```", raw_response, flags=re.IGNORECASE | re.DOTALL))
    if not matches:
        text = raw_response.strip()
        return text, [{"kind": "full_response", "start": 0, "end": len(text)}]
    best = matches[-1]
    reasoning = raw_response[: best.start()].strip()
    reasoning_tail = reasoning[-reasoning_context_chars:].strip() if reasoning_context_chars > 0 else ""
    fenced_code = raw_response[best.start() : best.end()].strip()
    parts: list[str] = []
    spans: list[dict[str, Any]] = []
    if reasoning_tail:
        start = sum(len(part) for part in parts)
        parts.append("<critical_reasoning_span>\n")
        start = sum(len(part) for part in parts)
        parts.append(reasoning_tail)
        end = sum(len(part) for part in parts)
        spans.append({"kind": "critical_reasoning_span", "start": start, "end": end})
        parts.append("\n</critical_reasoning_span>\n\n")
    parts.append("<final_code_span>\n")
    start = sum(len(part) for part in parts)
    parts.append(fenced_code)
    end = sum(len(part) for part in parts)
    spans.append({"kind": "final_code_span", "start": start, "end": end})
    parts.append("\n</final_code_span>")
    return "".join(parts).strip(), spans


def select_task_rows(rows: list[dict[str, Any]], task: str, *, expected: int) -> list[dict[str, Any]]:
    selected = [deepcopy(row) for row in rows if row.get("task") == task]
    if len(selected) != expected:
        raise ValueError(f"Expected {expected} {task} rows, found {len(selected)}")
    return selected


def with_round13_provenance(row: dict[str, Any], *, variant: str, component: str) -> dict[str, Any]:
    copied = deepcopy(row)
    metadata = dict(copied.get("row_metadata") or {})
    metadata["round13_evalleak_code16"] = {
        "variant": variant,
        "component": component,
        "eval_leak_diagnostic": component == "formal_code16",
        "copied_from_stable_tool_memory_bank": component in {"stable_tool", "stable_memory"},
    }
    copied["row_metadata"] = metadata
    return copied


def renumber_trajectory_ids(rows: list[dict[str, Any]], *, variant: str) -> None:
    for index, row in enumerate(rows):
        old = str(row.get("trajectory_id") or row.get("sample_id") or index)
        row["trajectory_id"] = f"trc13__{variant}__{index:03d}__{row.get('task')}__{row.get('expert')}__{stable_suffix(old)}"


def stable_suffix(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[-80:]
    return cleaned or hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def validate_rows(rows: list[dict[str, Any]], *, variant: str) -> None:
    if not rows:
        raise ValueError(f"{variant}: no rows")
    for index, row in enumerate(rows):
        if not str(row.get("response") or "").strip():
            raise ValueError(f"{variant}: row {index} has empty response")
        if not str(row.get("rendered_prompt") or row.get("prompt") or "").strip():
            raise ValueError(f"{variant}: row {index} has empty prompt")
    if variant == "rfmem_only":
        bad = [
            row
            for row in rows
            if row.get("task") == "code"
            and "deepseek" in str(row.get("source_name") or row.get("sample_id") or "").lower()
        ]
        if bad:
            raise ValueError(f"rfmem_only unexpectedly contains DeepSeek/R1 rows: {len(bad)}")


def build_summary(
    *,
    rows: list[dict[str, Any]],
    out_jsonl: Path,
    variant: str,
    expert_allowlist: tuple[str, ...],
    code_report: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "format": "trc_round13_evalleak_code16_summary_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "variant": variant,
        "output": str(out_jsonl),
        "num_rows": len(rows),
        "task_counts": dict(sorted(Counter(row["task"] for row in rows).items())),
        "expert_counts": dict(sorted(Counter(row["expert"] for row in rows).items())),
        "unique_prompt_counts": {
            task: len({row["prompt_id"] for row in rows if row["task"] == task})
            for task in sorted({row["task"] for row in rows})
        },
        "input_banks": {
            "stable_tool_memory_bank": str(Path(args.stable_tm_bank).expanduser().resolve()),
            "l5_cure_code16": str(Path(args.l5_code16).expanduser().resolve()),
        },
        "construction_policy": {
            "tool": "Copy 32 stable Tool rows from Round12/Round11 stable bank.",
            "memory": "Copy 32 stable Memory trajectory rows from Round12/Round11 stable bank.",
            "code": (
                "Use formal Code16 verified-positive trajectories; map trajectory source to the matching gate expert "
                "(ReasonFlux->code, MemoryAgent->memory, DeepSeek->reasoning)."
            ),
            "ability_span": (
                "Code responses are compacted to critical reasoning tail plus final fenced code block; spans are stored "
                "under sample_metadata.ability_spans."
            ),
            "leakage_policy": "Diagnostic only; do not report as a paper main non-leak result.",
        },
        "code_allowed_source_experts": list(expert_allowlist),
        "code_selection": code_report,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"# Round13 Eval-Leak Code16 TRC Calibration: {summary['variant']}",
            "",
            "This is a diagnostic calibration bank, not a paper-main non-leak dataset.",
            "",
            f"- Output: `{summary['output']}`",
            f"- Rows: `{summary['num_rows']}`",
            f"- Task counts: `{summary['task_counts']}`",
            f"- Expert counts: `{summary['expert_counts']}`",
            f"- Code source experts: `{summary['code_selection']['source_expert_counts']}`",
            f"- Code gate experts: `{summary['code_selection']['gate_expert_counts']}`",
            f"- Code datasets: `{summary['code_selection']['dataset_counts']}`",
            f"- Unique code prompts: `{summary['code_selection']['unique_prompts']}`",
            "",
            "Code ability spans are stored in `sample_metadata.ability_spans`.",
            "",
            "```json",
            json.dumps(summary["code_selection"], ensure_ascii=False, indent=2, sort_keys=True),
            "```",
            "",
        ]
    )


def render_combined_markdown(combined: dict[str, Any], reports: dict[str, dict[str, Any]]) -> str:
    lines = [
        "# Round13 Eval-Leak Code16 TRC Calibration",
        "",
        "Diagnostic purpose: test whether formal Code eval ability is learnable when verified positive trajectories come from the eval distribution.",
        "",
        "These banks must be reported separately from leak-safe calibration results.",
        "",
        "## Variants",
        "",
    ]
    for name, report in reports.items():
        lines.extend(
            [
                f"### {name}",
                "",
                f"- Output: `{report['output']}`",
                f"- Task counts: `{report['task_counts']}`",
                f"- Expert counts: `{report['expert_counts']}`",
                f"- Code source experts: `{report['code_selection']['source_expert_counts']}`",
                f"- Code gate experts: `{report['code_selection']['gate_expert_counts']}`",
                f"- Code datasets: `{report['code_selection']['dataset_counts']}`",
                "",
            ]
        )
    lines.extend(["## Combined Summary", "", "```json", json.dumps(combined, ensure_ascii=False, indent=2, sort_keys=True), "```", ""])
    return "\n".join(lines)


def sample_source_expert(sample: dict[str, Any]) -> str:
    details = sample.get("details") if isinstance(sample.get("details"), dict) else {}
    raw = str(details.get("expert_name") or sample.get("expert_name") or sample.get("expert") or "").strip()
    lowered = raw.lower()
    if "reasonflux" in lowered:
        return "reasonflux"
    if "memory" in lowered:
        return "memory_agent"
    if "deepseek" in lowered or "r1" in lowered:
        return "deepseek_r1_distill"
    return lowered or "unknown"


def infer_dataset_from_prompt_id(prompt_id: str) -> str:
    lowered = prompt_id.lower()
    if "livecodebench" in lowered:
        return "LiveCodeBench"
    if "livebench" in lowered:
        return "LiveBench"
    return "unknown"


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--stable-tm-bank", default=str(DEFAULT_STABLE_TM_BANK))
    parser.add_argument("--l5-code16", default=str(DEFAULT_L5_CODE16))
    parser.add_argument(
        "--reasoning-context-chars",
        type=int,
        default=600,
        help="Keep this many chars before the final code block. Keep compact enough that final code is inside 512 response tokens.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
