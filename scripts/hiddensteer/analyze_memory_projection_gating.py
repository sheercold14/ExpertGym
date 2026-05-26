#!/usr/bin/env python3
"""Analyze no-gold accept/reject policies for paired Memory projection runs."""

from __future__ import annotations

import argparse
import json
import re
import string
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in Path(args.paired_results).expanduser().open(encoding="utf-8")]
    policies = build_policies(args)
    rows = []
    for name, chooser in policies:
        rows.append(evaluate_policy(records, name, chooser))
    summary = {
        "paired_results": str(Path(args.paired_results).expanduser().resolve()),
        "num_records": len(records),
        "policies": rows,
        "note": "Policy decisions use only prediction strings and projection runtime stats; gold answers are used only for evaluation.",
    }
    write_json(output_dir / "gating_summary.json", summary)
    write_markdown(output_dir / "README.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_policies(args: argparse.Namespace) -> list[tuple[str, Callable[[dict[str, Any]], bool]]]:
    policies: list[tuple[str, Callable[[dict[str, Any]], bool]]] = [
        ("always_baseline", lambda _r: False),
        ("always_projection", lambda _r: True),
        ("projection_reject_expansion", lambda r: not projected_expands_baseline_answer(r)),
        ("projection_reject_numeric_flip", lambda r: not numeric_answer_flip(r)),
        ("projection_reject_short_to_long_unrelated", lambda r: not short_answer_to_long_unrelated_answer(r)),
        ("projection_reject_answer_shape_risk", lambda r: not answer_shape_risk(r)),
    ]
    for threshold in parse_float_list(args.projection_rate_thresholds):
        policies.append(
            (
                f"projection_rate_lt_{threshold:.3f}",
                lambda r, threshold=threshold: projection_rate(r) < threshold,
            )
        )
        policies.append(
            (
                f"answer_changed_and_projection_rate_lt_{threshold:.3f}",
                lambda r, threshold=threshold: answer_changed(r) and projection_rate(r) < threshold,
            )
        )
    return policies


def evaluate_policy(
    records: list[dict[str, Any]],
    name: str,
    choose_projected: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    f1_values = []
    em_values = []
    sub_em_values = []
    accepted = []
    wins = 0
    losses = 0
    ties = 0
    for record in records:
        use_projected = bool(choose_projected(record))
        accepted.append(use_projected)
        payload = record["projected"] if use_projected else record["baseline"]
        scores = score_prediction(payload.get("pred") or "", record.get("gold") or "")
        f1_values.append(scores["f1"])
        em_values.append(scores["em"])
        sub_em_values.append(scores["sub_em"])
        delta = float(record["projected_scores"]["f1"]) - float(record["baseline_scores"]["f1"]) if use_projected else 0.0
        wins += delta > 1.0e-9
        losses += delta < -1.0e-9
        ties += abs(delta) <= 1.0e-9
    accepted_records = [record for record, use_projected in zip(records, accepted) if use_projected]
    return {
        "policy": name,
        "accepted": len(accepted_records),
        "f1": sum(f1_values) / max(len(f1_values), 1),
        "em": sum(em_values) / max(len(em_values), 1),
        "sub_em": sum(sub_em_values) / max(len(sub_em_values), 1),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "accepted_changed_indices": [
            {
                "index": record.get("index"),
                "projection_rate": projection_rate(record),
                "baseline_pred": record["baseline"].get("pred"),
                "projected_pred": record["projected"].get("pred"),
                "delta_f1": record.get("delta_f1"),
            }
            for record in accepted_records
            if answer_changed(record)
        ],
    }


def answer_changed(record: dict[str, Any]) -> bool:
    return normalize_answer(record["baseline"].get("pred") or "") != normalize_answer(record["projected"].get("pred") or "")


def answer_shape_risk(record: dict[str, Any]) -> bool:
    return (
        projected_expands_baseline_answer(record)
        or numeric_answer_flip(record)
        or short_answer_to_long_unrelated_answer(record)
    )


def projected_expands_baseline_answer(record: dict[str, Any]) -> bool:
    if not answer_changed(record):
        return False
    baseline_tokens = answer_tokens(record["baseline"].get("pred") or "")
    projected_tokens = answer_tokens(record["projected"].get("pred") or "")
    return (
        bool(baseline_tokens)
        and len(projected_tokens) > len(baseline_tokens)
        and set(baseline_tokens).issubset(set(projected_tokens))
    )


def numeric_answer_flip(record: dict[str, Any]) -> bool:
    if not answer_changed(record):
        return False
    baseline = normalize_answer(record["baseline"].get("pred") or "")
    projected = normalize_answer(record["projected"].get("pred") or "")
    return bool(re.fullmatch(r"\d{1,4}", baseline)) and baseline not in projected


def short_answer_to_long_unrelated_answer(record: dict[str, Any]) -> bool:
    if not answer_changed(record):
        return False
    baseline_tokens = set(answer_tokens(record["baseline"].get("pred") or ""))
    projected_tokens = set(answer_tokens(record["projected"].get("pred") or ""))
    return (
        bool(baseline_tokens)
        and bool(projected_tokens)
        and len(baseline_tokens) <= 3
        and len(projected_tokens) >= 5
        and not (baseline_tokens & projected_tokens)
    )


def answer_tokens(text: str) -> list[str]:
    return normalize_answer(text).split()


def projection_rate(record: dict[str, Any]) -> float:
    stats = record.get("projection_stats") or {}
    return float(stats.get("tokens_projected", 0.0)) / max(float(stats.get("tokens_seen", 0.0)), 1.0)


def score_prediction(prediction: str, gold: str) -> dict[str, float]:
    f1, precision, recall = f1_score(prediction, gold)
    normalized_pred = normalize_answer(prediction)
    normalized_gold = normalize_answer(gold)
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "em": float(normalized_pred == normalized_gold),
        "sub_em": float((normalized_gold in normalized_pred) or (normalized_pred in normalized_gold)),
    }


def f1_score(prediction: str, ground_truth: str) -> tuple[float, float, float]:
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
        return 0.0, 0.0, 0.0
    if normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
        return 0.0, 0.0, 0.0
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0 or not prediction_tokens or not ground_truth_tokens:
        return 0.0, 0.0, 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall), precision, recall


def normalize_answer(text: str) -> str:
    value = (text or "").lower()
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    value = "".join(ch for ch in value if ch not in set(string.punctuation))
    return " ".join(value.split())


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Memory Projection Gating Analysis",
        "",
        f"- Paired results: `{summary['paired_results']}`",
        f"- Records: `{summary['num_records']}`",
        "",
        "| policy | accepted | F1 | EM | sub-EM | wins/losses/ties |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["policies"]:
        lines.append(
            f"| {row['policy']} | {row['accepted']} | {row['f1']:.4f} | {row['em']:.4f} | "
            f"{row['sub_em']:.4f} | {row['wins']} / {row['losses']} / {row['ties']} |"
        )
    lines.extend(["", summary["note"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in str(raw).split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paired-results",
        default=(
            "/tmp/shared-storage/ExpertGym/hiddensteer/"
            "memory_hotpotqa_projection_eval_init1_eval50_full128_all_l16-27_r4_a035_p90_20260525/"
            "paired_results.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/shared-storage/ExpertGym/hiddensteer/memory_projection_gating_analysis_init1_eval50_full128_20260525",
    )
    parser.add_argument("--projection-rate-thresholds", default="0.085,0.09,0.095,0.10")
    return parser.parse_args()


if __name__ == "__main__":
    main()
