"""Build a contrast-aware OP-VEC gate overlay from pass/fail residual probes.

This is an offline, training-free candidate generator.  It starts from an
existing parameter-level gate checkpoint, then redistributes a small amount of
coefficient mass toward residual entries whose signed utility is higher on
passing code trajectories than failing trajectories for the same prompt.

The script does not change reward, rollout, training, or baking logic.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable, Mapping


DEFAULT_BASE_GATES = Path("/tmp/shared-storage/ExpertGym/rcrf/rcrf_v1_20260521/rcrf/gates.json")
DEFAULT_CONTRAST_SUMMARIES = (
    Path(
        "/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast/"
        "livebench_code_alllayers_s16_20260521/contrast_module_summary.jsonl"
    ),
    Path(
        "/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast/"
        "livecodebench_code_alllayers_s16_20260521/contrast_module_summary.jsonl"
    ),
)
DEFAULT_OUTPUT_DIR = Path(
    "/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/"
    "rcrf_code_contrast_v1"
)


def main() -> None:
    args = parse_args()
    base_payload = load_json(args.base_gates)
    base_gates = extract_gate_map(base_payload)
    contrast_rows = load_contrast_rows(args.contrast_summary)
    if not contrast_rows:
        raise ValueError("No contrast rows loaded")

    rows_with_scores, scale_info = score_contrast_rows(
        contrast_rows,
        normalization=args.normalization,
        scale_quantile=args.scale_quantile,
        min_scale=args.min_scale,
    )
    overlay = aggregate_overlay_scores(
        rows_with_scores,
        allowed_experts=set(args.expert),
        aggregation=args.aggregation,
        min_abs_score=args.min_abs_score,
        conflict_penalty=args.conflict_penalty,
    )
    preserve_signals, preserve_summary = build_preserve_signals(
        args=args,
        base_gates=base_gates,
    )
    harm_veto_signals, harm_veto_summary = build_harm_veto_signals(
        args=args,
        base_gates=base_gates,
    )
    gates, decision_rows = apply_overlay(
        base_gates=base_gates,
        overlay=overlay,
        max_delta=args.max_delta,
        min_abs_score=args.min_abs_score,
        min_coeff=args.min_coeff,
        max_coeff=args.max_coeff,
        protect_negative_experts=set(args.protect_negative_expert),
        preserve_signals=preserve_signals,
        preserve_negative_scale=args.preserve_negative_scale,
        harm_veto_signals=harm_veto_signals,
        harm_veto_positive_scale=args.harm_veto_positive_scale,
        harm_veto_positive_scale_mode=args.harm_veto_positive_scale_mode,
        harm_veto_task_positive_scales=args.harm_veto_task_positive_scales,
        eps=args.harm_veto_ratio_eps,
    )
    if args.preserve_expert_mean:
        gates = recenter_by_expert(
            gates=gates,
            base_gates=base_gates,
            min_coeff=args.min_coeff,
            max_coeff=args.max_coeff,
            passes=args.recenter_passes,
            skip_experts=set(args.protect_negative_expert),
            skip_keys=set(preserve_signals) | set(harm_veto_signals),
        )
        decision_rows = refresh_decision_deltas(decision_rows, gates=gates, base_gates=base_gates)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "contrast_aware_residual_overlay_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_gate_checkpoint": str(args.base_gates.expanduser().resolve()),
        "mode_manifest": base_payload.get("mode_manifest"),
        "principle": {
            "unit": "parameter-level OP-VEC residual coefficient",
            "signal": "same-prompt pass/fail signed utility contrast on Code hurt subsets",
            "rule": "increase entries whose residual utility is higher on passing code; suppress entries whose utility is higher on failing code",
            "budget": "preserve each expert's mean coefficient by default; protected experts can skip negative overlay and recentering to test capability-preserve floors",
        },
        "config": {
            "contrast_summary": [str(path.expanduser().resolve()) for path in args.contrast_summary],
            "normalization": args.normalization,
            "scale_quantile": args.scale_quantile,
            "min_scale": args.min_scale,
            "max_delta": args.max_delta,
            "min_abs_score": args.min_abs_score,
            "aggregation": args.aggregation,
            "conflict_penalty": args.conflict_penalty,
            "min_coeff": args.min_coeff,
            "max_coeff": args.max_coeff,
            "preserve_expert_mean": args.preserve_expert_mean,
            "recenter_passes": args.recenter_passes,
            "protect_negative_expert": args.protect_negative_expert,
            "preserve_summary": [str(path.expanduser().resolve()) for path in args.preserve_summary],
            "preserve_task": args.preserve_task,
            "preserve_expert": args.preserve_expert,
            "preserve_min_normalized_utility": args.preserve_min_normalized_utility,
            "preserve_min_positive_fraction": args.preserve_min_positive_fraction,
            "preserve_negative_scale": args.preserve_negative_scale,
            "harm_veto_summary": [str(path.expanduser().resolve()) for path in args.harm_veto_summary],
            "harm_veto_task": args.harm_veto_task,
            "harm_veto_expert": args.harm_veto_expert,
            "harm_veto_min_normalized_harm": args.harm_veto_min_normalized_harm,
            "harm_veto_positive_scale": args.harm_veto_positive_scale,
            "harm_veto_positive_scale_mode": args.harm_veto_positive_scale_mode,
            "harm_veto_task_positive_scale": args.harm_veto_task_positive_scale,
            "harm_veto_ratio_eps": args.harm_veto_ratio_eps,
            "experts": args.expert,
        },
        "scale_info": scale_info,
        "preserve_signal_summary": preserve_summary,
        "harm_veto_signal_summary": harm_veto_summary,
        "gates": gates,
        "coefficient_summary": coefficient_summary(gates),
        "delta_summary": delta_summary(base_gates, gates),
        "decision_summary": decision_summary(decision_rows),
        "decision_rows": sorted(decision_rows, key=lambda row: abs(float(row["delta"])), reverse=True),
    }
    write_json(output_dir / "gates.json", payload)
    write_jsonl(output_dir / "decision_rows.jsonl", payload["decision_rows"])
    (output_dir / "summary.md").write_text(render_summary(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "gate_checkpoint": str(output_dir / "gates.json"),
                "num_gates": len(gates),
                "changed": payload["delta_summary"]["overall"]["changed_count"],
                "coefficient_summary": payload["coefficient_summary"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-gates", type=Path, default=DEFAULT_BASE_GATES)
    parser.add_argument(
        "--contrast-summary",
        type=Path,
        action="append",
        default=[],
        help="Path to contrast_module_summary.jsonl. Defaults to LiveBench+LiveCodeBench hurt16 contrasts.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--normalization", choices=("global", "per-file"), default="global")
    parser.add_argument("--scale-quantile", type=float, default=0.90)
    parser.add_argument("--min-scale", type=float, default=1.0e-12)
    parser.add_argument("--max-delta", type=float, default=0.06)
    parser.add_argument("--min-abs-score", type=float, default=0.10)
    parser.add_argument(
        "--aggregation",
        choices=("mean", "conservative"),
        default="mean",
        help="How to combine multiple contrast sources for the same residual entry.",
    )
    parser.add_argument(
        "--conflict-penalty",
        type=float,
        default=0.35,
        help="For conservative aggregation, negative score multiplier for entries with conflicting source signs.",
    )
    parser.add_argument("--min-coeff", type=float, default=0.55)
    parser.add_argument("--max-coeff", type=float, default=1.12)
    parser.add_argument(
        "--expert",
        action="append",
        default=[],
        help="Expert allowlist. Defaults to all experts present in the base gates.",
    )
    parser.add_argument(
        "--no-preserve-expert-mean",
        dest="preserve_expert_mean",
        action="store_false",
        help="Disable per-expert mean recentering after the contrast overlay.",
    )
    parser.set_defaults(preserve_expert_mean=True)
    parser.add_argument("--recenter-passes", type=int, default=3)
    parser.add_argument(
        "--protect-negative-expert",
        action="append",
        default=[],
        help=(
            "Expert whose coefficients should not be decreased by the contrast overlay. "
            "Protected experts are also skipped during mean recentering so the floor is preserved. "
            "Default: no protected experts."
        ),
    )
    parser.add_argument(
        "--preserve-summary",
        type=Path,
        action="append",
        default=[],
        help=(
            "Optional signed_utility_summary.json used to protect residual entries that are useful "
            "on a specified behavior task. Can be passed multiple times. Default: disabled."
        ),
    )
    parser.add_argument(
        "--preserve-task",
        action="append",
        default=[],
        help="Task whose positive signed utility marks a residual as protected, e.g. --preserve-task memory.",
    )
    parser.add_argument(
        "--preserve-expert",
        action="append",
        default=[],
        help="Expert allowlist for preserve signals. Defaults to all experts.",
    )
    parser.add_argument("--preserve-min-normalized-utility", type=float, default=0.40)
    parser.add_argument("--preserve-min-positive-fraction", type=float, default=0.50)
    parser.add_argument(
        "--preserve-negative-scale",
        type=float,
        default=0.0,
        help="Multiplier applied to negative contrast deltas for preserve-protected entries. 0 means floor at base.",
    )
    parser.add_argument(
        "--harm-veto-summary",
        type=Path,
        action="append",
        default=[],
        help=(
            "Optional signed_utility_summary.json used to veto positive contrast deltas on residual "
            "entries that are harmful to specified behavior tasks. Can be passed multiple times. Default: disabled."
        ),
    )
    parser.add_argument(
        "--harm-veto-task",
        action="append",
        default=[],
        help="Task whose harm_mean marks a residual as unsafe to increase, e.g. --harm-veto-task memory.",
    )
    parser.add_argument(
        "--harm-veto-expert",
        action="append",
        default=[],
        help="Expert allowlist for harm veto signals. Defaults to all experts.",
    )
    parser.add_argument("--harm-veto-min-normalized-harm", type=float, default=0.40)
    parser.add_argument(
        "--harm-veto-positive-scale",
        type=float,
        default=0.0,
        help=(
            "Constant multiplier applied to positive contrast deltas for harm-vetoed entries, or the "
            "minimum multiplier when --harm-veto-positive-scale-mode=evidence-ratio. 0 means no floor."
        ),
    )
    parser.add_argument(
        "--harm-veto-positive-scale-mode",
        choices=("constant", "evidence-ratio"),
        default="constant",
        help=(
            "constant keeps the old fixed harm-veto multiplier; evidence-ratio scales positive deltas by "
            "code_utility / (code_utility + protected_harm), with --harm-veto-positive-scale as a floor."
        ),
    )
    parser.add_argument(
        "--harm-veto-task-positive-scale",
        action="append",
        default=[],
        metavar="TASK=SCALE",
        help=(
            "Optional task-specific positive delta multiplier for harm-vetoed entries, e.g. "
            "--harm-veto-task-positive-scale tool=0 --harm-veto-task-positive-scale memory=0.5. "
            "Overrides --harm-veto-positive-scale-mode for the matching harm task."
        ),
    )
    parser.add_argument("--harm-veto-ratio-eps", type=float, default=1e-12)
    args = parser.parse_args()
    args.harm_veto_task_positive_scales = parse_task_scales(args.harm_veto_task_positive_scale)
    if not args.contrast_summary:
        args.contrast_summary = list(DEFAULT_CONTRAST_SUMMARIES)
    if not args.expert:
        args.expert = ["tool", "memory", "code"]
    if not (0.0 < args.scale_quantile <= 1.0):
        raise ValueError("--scale-quantile must be in (0, 1]")
    if args.preserve_summary and not args.preserve_task:
        raise ValueError("--preserve-summary requires at least one --preserve-task")
    if args.harm_veto_summary and not args.harm_veto_task:
        raise ValueError("--harm-veto-summary requires at least one --harm-veto-task")
    if not (0.0 <= args.preserve_negative_scale <= 1.0):
        raise ValueError("--preserve-negative-scale must be in [0, 1]")
    if not (0.0 <= args.harm_veto_positive_scale <= 1.0):
        raise ValueError("--harm-veto-positive-scale must be in [0, 1]")
    if args.harm_veto_ratio_eps <= 0.0:
        raise ValueError("--harm-veto-ratio-eps must be positive")
    return args


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.expanduser().read_text(encoding="utf-8"))


def parse_task_scales(raw_items: Iterable[str]) -> dict[str, float]:
    scales: dict[str, float] = {}
    for raw in raw_items:
        if "=" not in raw:
            raise ValueError(f"--harm-veto-task-positive-scale must be TASK=SCALE, got: {raw}")
        task, value = raw.split("=", 1)
        task = task.strip()
        if not task:
            raise ValueError(f"Missing task in --harm-veto-task-positive-scale: {raw}")
        scale = float(value)
        if not (0.0 <= scale <= 1.0):
            raise ValueError(f"Task-specific harm-veto scale must be in [0, 1], got {raw}")
        scales[task] = scale
    return scales


def iter_summary_modules(paths: Iterable[Path], option_name: str) -> Iterable[tuple[Path, Mapping[str, Any]]]:
    for path in paths:
        resolved = path.expanduser()
        payload = load_json(resolved)
        module_summary = payload.get("module_summary", {})
        if not isinstance(module_summary, Mapping):
            raise ValueError(f"{option_name} must contain a module_summary mapping: {resolved}")
        yield resolved, module_summary


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.expanduser().open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def extract_gate_map(payload: Mapping[str, Any]) -> dict[str, float]:
    raw = payload.get("gates", payload)
    if not isinstance(raw, Mapping):
        raise ValueError("Gate checkpoint must be a mapping or contain a 'gates' mapping")
    gates = {str(key): float(value) for key, value in raw.items()}
    if not gates:
        raise ValueError("Gate checkpoint is empty")
    return gates


def load_contrast_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        loaded = load_jsonl(path)
        for row in loaded:
            row = dict(row)
            row["_contrast_source"] = str(path.expanduser().resolve())
            rows.append(row)
    return rows


def score_contrast_rows(
    rows: list[dict[str, Any]],
    *,
    normalization: str,
    scale_quantile: float,
    min_scale: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if normalization == "global":
        scale = robust_abs_quantile([float(row["contrast_signed_effect_mean"]) for row in rows], scale_quantile, min_scale)
        scales = {str(row["_contrast_source"]): scale for row in rows}
    else:
        source_to_values: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            source_to_values[str(row["_contrast_source"])].append(float(row["contrast_signed_effect_mean"]))
        source_to_scale = {
            source: robust_abs_quantile(values, scale_quantile, min_scale)
            for source, values in source_to_values.items()
        }
        scales = {str(row["_contrast_source"]): source_to_scale[str(row["_contrast_source"])] for row in rows}

    scored: list[dict[str, Any]] = []
    for row in rows:
        contrast = float(row["contrast_signed_effect_mean"])
        positive_fraction = float(row.get("contrast_positive_fraction", 0.5))
        stability = min(1.0, abs(positive_fraction - 0.5) * 2.0)
        confidence = 0.5 + 0.5 * stability
        normalized = clamp(contrast / scales[str(row["_contrast_source"])], -1.0, 1.0)
        scored_row = dict(row)
        scored_row["contrast_scale"] = scales[str(row["_contrast_source"])]
        scored_row["normalized_contrast_score"] = normalized
        scored_row["contrast_confidence"] = confidence
        scored_row["overlay_score"] = normalized * confidence
        scored.append(scored_row)

    scale_info = {
        "normalization": normalization,
        "scale_quantile": scale_quantile,
        "sources": {},
    }
    for source in sorted({str(row["_contrast_source"]) for row in rows}):
        source_rows = [row for row in rows if str(row["_contrast_source"]) == source]
        abs_values = [abs(float(row["contrast_signed_effect_mean"])) for row in source_rows]
        scale_info["sources"][source] = {
            "num_rows": len(source_rows),
            "mean_abs_contrast": mean(abs_values) if abs_values else 0.0,
            "scale": scored[[str(row["_contrast_source"]) for row in scored].index(source)]["contrast_scale"],
        }
    return scored, scale_info


def robust_abs_quantile(values: Iterable[float], quantile: float, min_scale: float) -> float:
    abs_values = sorted(abs(float(value)) for value in values if math.isfinite(float(value)))
    if not abs_values:
        return min_scale
    index = int(round((len(abs_values) - 1) * quantile))
    return max(float(abs_values[index]), min_scale)


def aggregate_overlay_scores(
    rows: list[dict[str, Any]],
    *,
    allowed_experts: set[str],
    aggregation: str,
    min_abs_score: float,
    conflict_penalty: float,
) -> dict[str, dict[str, Any]]:
    support: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        expert = str(row["expert"])
        if expert not in allowed_experts:
            continue
        key = f"{row['param_name']}::{expert}"
        support[key].append(row)

    overlay: dict[str, dict[str, Any]] = {}
    for key in sorted(support):
        rows_for_key = support[key]
        scores = [float(row["overlay_score"]) for row in rows_for_key]
        weights = [max(1.0, float(row.get("pair_count", 1.0))) for row in rows_for_key]
        if aggregation == "mean":
            score = weighted_mean(scores, weights)
            aggregation_reason = "weighted_mean"
            informative_count = sum(1 for value in scores if abs(value) >= min_abs_score)
            positive_informative = sum(1 for value in scores if value >= min_abs_score)
            negative_informative = sum(1 for value in scores if value <= -min_abs_score)
        elif aggregation == "conservative":
            informative = [value for value in scores if abs(value) >= min_abs_score]
            informative_count = len(informative)
            positive_informative = sum(1 for value in informative if value > 0.0)
            negative_informative = sum(1 for value in informative if value < 0.0)
            if not informative:
                score = 0.0
                aggregation_reason = "no_informative_source"
            elif positive_informative and negative_informative:
                score = -abs(float(conflict_penalty)) * mean(abs(value) for value in informative)
                aggregation_reason = "source_sign_conflict_suppress"
            else:
                score = mean(informative)
                aggregation_reason = "source_sign_agreement"
        else:  # pragma: no cover - argparse guards this.
            raise ValueError(f"Unsupported aggregation: {aggregation}")
        overlay[key] = {
            "score": score,
            "aggregation": aggregation,
            "aggregation_reason": aggregation_reason,
            "source_scores": scores,
            "informative_source_count": informative_count,
            "positive_informative_source_count": positive_informative,
            "negative_informative_source_count": negative_informative,
            "weight": sum(weights),
            "support_count": len(rows_for_key),
            "sources": sorted({str(row["_contrast_source"]) for row in rows_for_key}),
            "raw_contrast_mean": mean(float(row["contrast_signed_effect_mean"]) for row in rows_for_key),
            "normalized_score_mean": mean(float(row["normalized_contrast_score"]) for row in rows_for_key),
            "positive_fraction_mean": mean(float(row.get("contrast_positive_fraction", 0.5)) for row in rows_for_key),
            "positive_expression_mean": mean(float(row.get("positive_expression_mean", 0.0)) for row in rows_for_key),
            "negative_expression_mean": mean(float(row.get("negative_expression_mean", 0.0)) for row in rows_for_key),
        }
    return overlay


def build_preserve_signals(
    *,
    args: argparse.Namespace,
    base_gates: Mapping[str, float],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if not args.preserve_summary:
        return {}, {"enabled": False}
    module_summaries = list(iter_summary_modules(args.preserve_summary, "--preserve-summary"))
    allowed_experts = set(args.preserve_expert or [key.rsplit("::", 1)[-1] for key in base_gates])
    preserve_tasks = set(args.preserve_task)

    positive_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for _summary_path, module_summary in module_summaries:
        for expert, param_map in module_summary.items():
            if str(expert) not in allowed_experts:
                continue
            for task_stats in dict(param_map).values():
                for task in preserve_tasks:
                    stats = dict(task_stats).get(task, {})
                    effect = float(stats.get("signed_effect_mean", 0.0))
                    if effect > 0.0:
                        positive_values[(str(expert), str(task))].append(effect)

    scales = {
        key: robust_abs_quantile(values, 0.75, args.min_scale)
        for key, values in positive_values.items()
    }
    signals: dict[str, dict[str, Any]] = {}
    task_counts: dict[str, int] = defaultdict(int)
    expert_counts: dict[str, int] = defaultdict(int)
    for summary_path, module_summary in module_summaries:
        for expert, param_map in module_summary.items():
            expert = str(expert)
            if expert not in allowed_experts:
                continue
            for param_name, task_stats in dict(param_map).items():
                key = f"{param_name}::{expert}"
                if key not in base_gates:
                    continue
                best_signal = signals.get(key)
                for task in preserve_tasks:
                    stats = dict(task_stats).get(task, {})
                    effect = float(stats.get("signed_effect_mean", 0.0))
                    positive_fraction = float(stats.get("positive_fraction", 0.0))
                    scale = max(float(scales.get((expert, str(task)), args.min_scale)), args.min_scale)
                    normalized = effect / scale
                    if effect <= 0.0:
                        continue
                    if normalized < float(args.preserve_min_normalized_utility):
                        continue
                    if positive_fraction < float(args.preserve_min_positive_fraction):
                        continue
                    candidate = {
                        "task": str(task),
                        "expert": expert,
                        "summary_path": str(summary_path.resolve()),
                        "signed_effect_mean": effect,
                        "positive_fraction": positive_fraction,
                        "expression_mean": float(stats.get("expression_mean", 0.0)),
                        "normalized_preserve_utility": normalized,
                        "scale": scale,
                    }
                    if best_signal is None or normalized > float(best_signal["normalized_preserve_utility"]):
                        best_signal = candidate
                if best_signal is not None:
                    signals[key] = best_signal
    for signal in signals.values():
        task_counts[str(signal["task"])] += 1
        expert_counts[str(signal["expert"])] += 1

    summary = {
        "enabled": True,
        "summary_paths": [str(path.expanduser().resolve()) for path in args.preserve_summary],
        "tasks": sorted(preserve_tasks),
        "experts": sorted(allowed_experts),
        "min_normalized_utility": args.preserve_min_normalized_utility,
        "min_positive_fraction": args.preserve_min_positive_fraction,
        "negative_scale": args.preserve_negative_scale,
        "num_preserved_keys": len(signals),
        "task_counts": dict(sorted(task_counts.items())),
        "expert_counts": dict(sorted(expert_counts.items())),
        "scales": {f"{expert}:{task}": value for (expert, task), value in sorted(scales.items())},
    }
    return signals, summary


def build_harm_veto_signals(
    *,
    args: argparse.Namespace,
    base_gates: Mapping[str, float],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if not args.harm_veto_summary:
        return {}, {"enabled": False}
    module_summaries = list(iter_summary_modules(args.harm_veto_summary, "--harm-veto-summary"))
    allowed_experts = set(args.harm_veto_expert or [key.rsplit("::", 1)[-1] for key in base_gates])
    veto_tasks = set(args.harm_veto_task)

    harm_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for _summary_path, module_summary in module_summaries:
        for expert, param_map in module_summary.items():
            if str(expert) not in allowed_experts:
                continue
            for task_stats in dict(param_map).values():
                for task in veto_tasks:
                    stats = dict(task_stats).get(task, {})
                    harm = float(stats.get("harm_mean", 0.0))
                    if harm > 0.0:
                        harm_values[(str(expert), str(task))].append(harm)

    scales = {
        key: robust_abs_quantile(values, 0.75, args.min_scale)
        for key, values in harm_values.items()
    }
    signals: dict[str, dict[str, Any]] = {}
    task_counts: dict[str, int] = defaultdict(int)
    expert_counts: dict[str, int] = defaultdict(int)
    for summary_path, module_summary in module_summaries:
        for expert, param_map in module_summary.items():
            expert = str(expert)
            if expert not in allowed_experts:
                continue
            for param_name, task_stats in dict(param_map).items():
                key = f"{param_name}::{expert}"
                if key not in base_gates:
                    continue
                best_signal = signals.get(key)
                for task in veto_tasks:
                    stats = dict(task_stats).get(task, {})
                    harm = float(stats.get("harm_mean", 0.0))
                    scale = max(float(scales.get((expert, str(task)), args.min_scale)), args.min_scale)
                    normalized = harm / scale
                    if harm <= 0.0:
                        continue
                    if normalized < float(args.harm_veto_min_normalized_harm):
                        continue
                    candidate = {
                        "task": str(task),
                        "expert": expert,
                        "summary_path": str(summary_path.resolve()),
                        "harm_mean": harm,
                        "signed_effect_mean": float(stats.get("signed_effect_mean", 0.0)),
                        "positive_fraction": float(stats.get("positive_fraction", 0.0)),
                        "expression_mean": float(stats.get("expression_mean", 0.0)),
                        "normalized_harm": normalized,
                        "scale": scale,
                    }
                    if best_signal is None or normalized > float(best_signal["normalized_harm"]):
                        best_signal = candidate
                if best_signal is not None:
                    signals[key] = best_signal
    for signal in signals.values():
        task_counts[str(signal["task"])] += 1
        expert_counts[str(signal["expert"])] += 1

    summary = {
        "enabled": True,
        "summary_paths": [str(path.expanduser().resolve()) for path in args.harm_veto_summary],
        "tasks": sorted(veto_tasks),
        "experts": sorted(allowed_experts),
        "min_normalized_harm": args.harm_veto_min_normalized_harm,
        "positive_scale": args.harm_veto_positive_scale,
        "task_positive_scales": dict(sorted(args.harm_veto_task_positive_scales.items())),
        "num_veto_keys": len(signals),
        "task_counts": dict(sorted(task_counts.items())),
        "expert_counts": dict(sorted(expert_counts.items())),
        "scales": {f"{expert}:{task}": value for (expert, task), value in sorted(scales.items())},
    }
    return signals, summary


def weighted_mean(values: list[float], weights: list[float]) -> float:
    total_weight = sum(weights)
    if total_weight <= 0.0:
        return 0.0
    return sum(value * weight for value, weight in zip(values, weights)) / total_weight


def apply_overlay(
    *,
    base_gates: Mapping[str, float],
    overlay: Mapping[str, Mapping[str, Any]],
    max_delta: float,
    min_abs_score: float,
    min_coeff: float,
    max_coeff: float,
    protect_negative_experts: set[str],
    preserve_signals: Mapping[str, Mapping[str, Any]],
    preserve_negative_scale: float,
    harm_veto_signals: Mapping[str, Mapping[str, Any]],
    harm_veto_positive_scale: float,
    harm_veto_positive_scale_mode: str,
    harm_veto_task_positive_scales: Mapping[str, float],
    eps: float,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    gates: dict[str, float] = dict(base_gates)
    decision_rows: list[dict[str, Any]] = []
    for key, base_coeff in base_gates.items():
        expert = key.rsplit("::", 1)[-1]
        metrics = overlay.get(key)
        if not metrics:
            delta = 0.0
            reason = "no_contrast_signal"
            effective_harm_scale = None
        else:
            score = float(metrics["score"])
            effective_harm_scale = None
            if abs(score) < min_abs_score:
                delta = 0.0
                reason = "below_min_abs_score"
            else:
                delta = max_delta * clamp(score, -1.0, 1.0)
                reason = "pass_fail_positive" if delta > 0 else "pass_fail_negative"
                if expert in protect_negative_experts and delta < 0.0:
                    delta = 0.0
                    reason = "protected_negative_overlay"
                preserve_signal = preserve_signals.get(key)
                if preserve_signal and delta < 0.0:
                    delta *= float(preserve_negative_scale)
                    reason = "preserve_utility_floor"
                harm_veto_signal = harm_veto_signals.get(key)
                if harm_veto_signal and delta > 0.0:
                    effective_harm_scale = harm_veto_scale(
                        mode=harm_veto_positive_scale_mode,
                        task=str(harm_veto_signal.get("task", "")),
                        score=score,
                        normalized_harm=float(harm_veto_signal.get("normalized_harm", 0.0)),
                        floor=harm_veto_positive_scale,
                        task_positive_scales=harm_veto_task_positive_scales,
                        eps=eps,
                    )
                    delta *= effective_harm_scale
                    reason = "behavior_harm_veto"
        new_coeff = clamp(float(base_coeff) + delta, min_coeff, max_coeff)
        gates[key] = new_coeff
        param_name = key.rsplit("::", 1)[0]
        decision_rows.append(
            {
                "key": key,
                "param_name": param_name,
                "expert": expert,
                "base_coefficient": float(base_coeff),
                "coefficient": float(new_coeff),
                "delta": float(new_coeff - float(base_coeff)),
                "pre_recenter_delta": float(delta),
                "reason": reason,
                "metrics": dict(metrics or {}),
                "preserve_signal": dict(preserve_signals.get(key, {})),
                "harm_veto_signal": dict(harm_veto_signals.get(key, {})),
                "harm_veto_effective_positive_scale": effective_harm_scale,
            }
        )
    return gates, decision_rows


def harm_veto_scale(
    *,
    mode: str,
    task: str,
    score: float,
    normalized_harm: float,
    floor: float,
    task_positive_scales: Mapping[str, float],
    eps: float,
) -> float:
    if task in task_positive_scales:
        return float(task_positive_scales[task])
    if mode == "constant":
        return float(floor)
    if mode == "evidence-ratio":
        code_utility = max(float(score), 0.0)
        protected_harm = max(float(normalized_harm), 0.0)
        ratio = code_utility / (code_utility + protected_harm + eps)
        return clamp(max(float(floor), ratio), 0.0, 1.0)
    raise ValueError(f"Unsupported harm-veto scale mode: {mode}")


def recenter_by_expert(
    *,
    gates: dict[str, float],
    base_gates: Mapping[str, float],
    min_coeff: float,
    max_coeff: float,
    passes: int,
    skip_experts: set[str] | None = None,
    skip_keys: set[str] | None = None,
) -> dict[str, float]:
    recentered = dict(gates)
    skip_experts = set(skip_experts or set())
    skip_keys = set(skip_keys or set())
    for _ in range(max(0, passes)):
        for expert in sorted({key.rsplit("::", 1)[-1] for key in base_gates}):
            if expert in skip_experts:
                continue
            keys = [key for key in base_gates if key.endswith(f"::{expert}")]
            if not keys:
                continue
            adjustable_keys = [key for key in keys if key not in skip_keys]
            if not adjustable_keys:
                continue
            target_total = sum(float(base_gates[key]) for key in keys)
            current_total = sum(float(recentered[key]) for key in keys)
            shift = (target_total - current_total) / float(len(adjustable_keys))
            for key in adjustable_keys:
                recentered[key] = clamp(float(recentered[key]) + shift, min_coeff, max_coeff)
    return recentered


def refresh_decision_deltas(
    decision_rows: list[dict[str, Any]],
    *,
    gates: Mapping[str, float],
    base_gates: Mapping[str, float],
) -> list[dict[str, Any]]:
    refreshed: list[dict[str, Any]] = []
    for row in decision_rows:
        key = str(row["key"])
        new_row = dict(row)
        new_row["coefficient"] = float(gates[key])
        new_row["delta"] = float(gates[key] - base_gates[key])
        refreshed.append(new_row)
    return refreshed


def coefficient_summary(gates: Mapping[str, float]) -> dict[str, dict[str, float]]:
    by_expert: dict[str, list[float]] = defaultdict(list)
    for key, value in gates.items():
        by_expert[key.rsplit("::", 1)[-1]].append(float(value))
    return {expert: numeric_summary(values) for expert, values in sorted(by_expert.items())}


def delta_summary(base_gates: Mapping[str, float], gates: Mapping[str, float]) -> dict[str, Any]:
    by_expert: dict[str, list[float]] = defaultdict(list)
    overall: list[float] = []
    for key, base_value in base_gates.items():
        delta = float(gates[key]) - float(base_value)
        by_expert[key.rsplit("::", 1)[-1]].append(delta)
        overall.append(delta)
    summary = {"overall": delta_numeric_summary(overall)}
    for expert, values in sorted(by_expert.items()):
        summary[expert] = delta_numeric_summary(values)
    return summary


def decision_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    by_reason: dict[str, int] = defaultdict(int)
    by_expert_reason: dict[str, int] = defaultdict(int)
    for row in rows:
        reason = str(row["reason"])
        expert = str(row["expert"])
        by_reason[reason] += 1
        by_expert_reason[f"{expert}:{reason}"] += 1
    return {
        "reason_counts": dict(sorted(by_reason.items())),
        "expert_reason_counts": dict(sorted(by_expert_reason.items())),
    }


def numeric_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "mean": mean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def delta_numeric_summary(values: list[float]) -> dict[str, float]:
    summary = numeric_summary(values)
    nonzero = [value for value in values if abs(value) > 1.0e-12]
    summary.update(
        {
            "changed_count": len(nonzero),
            "positive_count": sum(1 for value in values if value > 1.0e-12),
            "negative_count": sum(1 for value in values if value < -1.0e-12),
            "mean_abs": mean(abs(value) for value in values) if values else 0.0,
            "max_abs": max((abs(value) for value in values), default=0.0),
        }
    )
    return summary


def render_summary(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Contrast-Aware Residual Gate Overlay",
        "",
        "## Principle",
        "",
        "- Start from the existing RCRF gate checkpoint.",
        "- Use same-prompt Code pass/fail contrast instead of positive imitation alone.",
        "- Increase residual entries whose signed utility is higher on passing code generations.",
        "- Suppress entries whose signed utility is higher on failing code generations.",
        "- Preserve each expert's mean coefficient by default; protected experts can skip negative overlay and recentering to test capability-preserve floors.",
        "",
        "## Inputs",
        "",
        f"- base_gate_checkpoint: `{payload['base_gate_checkpoint']}`",
        f"- mode_manifest: `{payload.get('mode_manifest')}`",
        "",
        "Contrast summaries:",
    ]
    for path in payload["config"]["contrast_summary"]:
        lines.append(f"- `{path}`")
    lines.extend(["", "## Config", "", "```json", json.dumps(payload["config"], indent=2, ensure_ascii=False), "```"])
    preserve_summary = payload.get("preserve_signal_summary", {})
    if preserve_summary.get("enabled"):
        lines.extend(
            [
                "",
                "## Preserve Signals",
                "",
                f"- summaries: `{json.dumps(preserve_summary.get('summary_paths', []), sort_keys=True)}`",
                f"- preserved keys: `{preserve_summary.get('num_preserved_keys', 0)}`",
                f"- task counts: `{json.dumps(preserve_summary.get('task_counts', {}), sort_keys=True)}`",
                f"- expert counts: `{json.dumps(preserve_summary.get('expert_counts', {}), sort_keys=True)}`",
            ]
        )
    harm_veto_summary = payload.get("harm_veto_signal_summary", {})
    if harm_veto_summary.get("enabled"):
        lines.extend(
            [
                "",
                "## Harm Veto Signals",
                "",
                f"- summaries: `{json.dumps(harm_veto_summary.get('summary_paths', []), sort_keys=True)}`",
                f"- veto keys: `{harm_veto_summary.get('num_veto_keys', 0)}`",
                f"- task counts: `{json.dumps(harm_veto_summary.get('task_counts', {}), sort_keys=True)}`",
                f"- expert counts: `{json.dumps(harm_veto_summary.get('expert_counts', {}), sort_keys=True)}`",
            ]
        )
    lines.extend(["", "## Coefficients", "", "| expert | mean | std | min | max |", "|---|---:|---:|---:|---:|"])
    for expert, stats in sorted(payload["coefficient_summary"].items()):
        lines.append(
            f"| {expert} | {stats['mean']:.6f} | {stats['std']:.6f} | {stats['min']:.6f} | {stats['max']:.6f} |"
        )
    lines.extend(["", "## Delta Summary", "", "| group | changed | + | - | mean_abs | max_abs |", "|---|---:|---:|---:|---:|---:|"])
    for group, stats in payload["delta_summary"].items():
        lines.append(
            f"| {group} | {stats['changed_count']} | {stats['positive_count']} | {stats['negative_count']} | "
            f"{stats['mean_abs']:.6f} | {stats['max_abs']:.6f} |"
        )
    lines.extend(["", "## Top Changed Entries", "", "| rank | expert | param | base | new | delta | reason | score |", "|---:|---|---|---:|---:|---:|---|---:|"])
    for idx, row in enumerate(payload["decision_rows"][:30], start=1):
        score = float(row.get("metrics", {}).get("score", 0.0))
        lines.append(
            f"| {idx} | {row['expert']} | `{row['param_name']}` | {row['base_coefficient']:.6f} | "
            f"{row['coefficient']:.6f} | {row['delta']:.6f} | {row['reason']} | {score:.4f} |"
        )
    lines.extend(["", "## Output", "", "- `gates.json`: OP-VEC bake-compatible gate checkpoint.", "- `decision_rows.jsonl`: per-parameter audit rows."])
    return "\n".join(lines) + "\n"


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


if __name__ == "__main__":
    main()
