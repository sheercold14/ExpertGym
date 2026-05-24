#!/usr/bin/env python3
"""Build an expert-coefficient ablation from an existing OP-VEC gate file.

This script is intentionally mechanical: it does not read probes, rewards, or
rollouts.  It only rewrites coefficients for one expert so that capability
changes can be attributed to that expert's residual scale.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


DEFAULT_BASE_GATE = Path(
    "/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/"
    "rcrf_code_spanaware_tmpos_s32_memoryfull_softveto_v9/gates.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates/"
    "rcrf_v9_code_half_v14"
)


def main() -> None:
    args = parse_args()
    if (args.scale is None) == (args.set_value is None):
        raise ValueError("Specify exactly one of --scale or --set-value")

    base_path = Path(args.base_gate).expanduser().resolve()
    base_payload = load_json(base_path)
    base_gates = extract_gate_map(base_payload)
    if not base_gates:
        raise ValueError(f"No gates found in {base_path}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    target_expert = args.expert.strip()
    gates: dict[str, float] = {}
    decision_rows: list[dict[str, Any]] = []
    for key in sorted(base_gates):
        before = float(base_gates[key])
        expert = expert_from_key(key)
        if expert == target_expert:
            after = before * float(args.scale) if args.scale is not None else float(args.set_value)
            reason = "scaled_target_expert" if args.scale is not None else "set_target_expert"
        else:
            after = before
            reason = "unchanged_non_target_expert"
        gates[key] = after
        decision_rows.append(
            {
                "key": key,
                "param_name": key.rsplit("::", 1)[0] if "::" in key else key,
                "expert": expert,
                "before": before,
                "after": after,
                "delta": after - before,
                "reason": reason,
            }
        )

    summary = {
        "variant": args.variant_name,
        "base_gate": str(base_path),
        "target_expert": target_expert,
        "action": "scale" if args.scale is not None else "set_value",
        "scale": args.scale,
        "set_value": args.set_value,
        "num_gates": len(gates),
        "changed_count": sum(1 for row in decision_rows if abs(float(row["delta"])) > 0.0),
        "coefficient_summary_before": coefficient_summary({row["key"]: row["before"] for row in decision_rows}),
        "coefficient_summary_after": coefficient_summary(gates),
        "delta_summary": delta_summary(decision_rows),
    }
    payload = {
        "format": "expert_scaled_gate_ablation_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "variant": args.variant_name,
        "base_gate_checkpoint": str(base_path),
        "principle": {
            "unit": "parameter-level OP-VEC residual coefficient",
            "rule": "mechanically change one expert's coefficients and keep all other experts unchanged",
            "purpose": "test whether one expert residual is over- or under-contributing without changing scoring logic",
        },
        "config": {
            "expert": target_expert,
            "scale": args.scale,
            "set_value": args.set_value,
        },
        "summary": summary,
        "gates": gates,
        "decision_rows": decision_rows,
    }
    write_json(output_dir / "gates.json", payload)
    write_json(output_dir / "expert_scale_summary.json", summary)
    (output_dir / "expert_scale_summary.md").write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps({"gate_checkpoint": str(output_dir / "gates.json"), "summary": summary}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-gate", type=Path, default=DEFAULT_BASE_GATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--variant-name", default="rcrf_v9_code_half_v14")
    parser.add_argument("--expert", default="code")
    parser.add_argument("--scale", type=float, default=None, help="Multiply target expert coefficients by this value.")
    parser.add_argument("--set-value", type=float, default=None, help="Set target expert coefficients to this value.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_gate_map(payload: dict[str, Any]) -> dict[str, float]:
    raw = payload.get("gates", payload)
    if not isinstance(raw, dict):
        raise TypeError("Gate payload must be a dict or contain a dict-valued `gates` field")
    return {str(key): float(value) for key, value in raw.items() if isinstance(value, int | float)}


def expert_from_key(key: str) -> str:
    return key.rsplit("::", 1)[1] if "::" in key else ""


def coefficient_summary(gates: dict[str, float]) -> dict[str, Any]:
    by_expert: dict[str, list[float]] = defaultdict(list)
    for key, value in gates.items():
        by_expert[expert_from_key(key)].append(float(value))
    return {expert: describe(values) for expert, values in sorted(by_expert.items())}


def delta_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_expert: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_expert[str(row["expert"])].append(float(row["delta"]))
    return {expert: describe(values) for expert, values in sorted(by_expert.items())}


def describe(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(values),
        "min": ordered[0],
        "mean": mean(values),
        "max": ordered[-1],
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "changed_count": sum(1 for value in values if abs(float(value)) > 0.0),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        f"# {summary['variant']}",
        "",
        "## Config",
        "",
        f"- Base gate: `{summary['base_gate']}`",
        f"- Target expert: `{summary['target_expert']}`",
        f"- Action: `{summary['action']}`",
        f"- Scale: `{summary['scale']}`",
        f"- Set value: `{summary['set_value']}`",
        f"- Changed gates: {summary['changed_count']} / {summary['num_gates']}",
        "",
        "## Coefficient Summary",
        "",
        "| expert | before mean | after mean | before min/max | after min/max |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    before = summary["coefficient_summary_before"]
    after = summary["coefficient_summary_after"]
    for expert in sorted(after):
        b = before.get(expert, {})
        a = after.get(expert, {})
        lines.append(
            f"| {expert} | {float(b.get('mean', 0.0)):.6f} | {float(a.get('mean', 0.0)):.6f} | "
            f"{float(b.get('min', 0.0)):.6f}/{float(b.get('max', 0.0)):.6f} | "
            f"{float(a.get('min', 0.0)):.6f}/{float(a.get('max', 0.0)):.6f} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
