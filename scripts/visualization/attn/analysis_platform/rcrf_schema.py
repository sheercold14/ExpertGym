"""RCRF mechanism data loading and normalization helpers.

The workbench consumes four long-form JSONL schemas, but most current runs
still expose native artifacts such as signed_utility_summary.json and RCRF
gates.json.  This module keeps the conversion local and robust: missing real
paths produce deterministic example data instead of failing the UI.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd


EXPERTS = ("tool", "memory", "code")
TASKS = ("tool", "memory", "code")
MODULE_ORDER = ("q", "k", "v", "o", "gate", "up", "down")
MODULE_FAMILIES = ("q", "k", "v", "o", "mlp")
RCRF_CANDIDATES = ("owner_only", "energy_only", "no_conflict", "rcrf")

RESIDUAL_COLUMNS = (
    "run_id",
    "expert",
    "task",
    "prompt_id",
    "span_type",
    "layer",
    "module",
    "module_family",
    "signed_effect",
    "expression",
    "positive_fraction",
)
INTERFERENCE_COLUMNS = (
    "expert_a",
    "expert_b",
    "task",
    "layer",
    "module",
    "module_family",
    "cosine",
    "conflict_score",
    "cross_harm",
)
GATE_COLUMNS = (
    "candidate",
    "expert",
    "layer",
    "module",
    "module_family",
    "alpha",
    "owner_signal",
    "synergy_signal",
    "harm_signal",
    "noise_score",
)
EVAL_COLUMNS = ("candidate", "task", "subset", "metric", "score", "n_examples")

DEFAULT_INPUT_PATHS = (
    "/tmp/shared-storage/ExpertGym/attention_matrix/signed_utility_signature_s4_layers8_alllinear_20260521",
    "/tmp/shared-storage/ExpertGym/rcrf/rcrf_v1_20260521",
    "/tmp/shared-storage/ExpertGym/rcrf/eval/rcrf_v1_20260521",
    "/tmp/shared-storage/ExpertGym/rcrf/eval/toolrl_rlla4k_20260521",
    "/tmp/shared-storage/OnPolicy/eval/cure_feedback/rcrf-v1-rcrf-code/rcrf-v1-rcrf/rcrf_v1_rcrf_code_quick_20260521",
)


@dataclass
class MechanismData:
    residual: pd.DataFrame
    interference: pd.DataFrame
    gate: pd.DataFrame
    eval: pd.DataFrame
    source_notes: list[str]
    used_example: bool = False


def module_from_param(param_name: str) -> str:
    text = str(param_name)
    if ".self_attn.q_proj.weight" in text or text in {"attn_q", "q"}:
        return "q"
    if ".self_attn.k_proj.weight" in text or text in {"attn_k", "k"}:
        return "k"
    if ".self_attn.v_proj.weight" in text or text in {"attn_v", "v"}:
        return "v"
    if ".self_attn.o_proj.weight" in text or text in {"attn_o", "o"}:
        return "o"
    if ".mlp.gate_proj.weight" in text or text in {"mlp_gate", "gate"}:
        return "gate"
    if ".mlp.up_proj.weight" in text or text in {"mlp_up", "up"}:
        return "up"
    if ".mlp.down_proj.weight" in text or text in {"mlp_down", "down"}:
        return "down"
    return "other"


def module_family(module: str) -> str:
    module = module_from_param(module)
    if module in {"q", "k", "v", "o"}:
        return module
    if module in {"gate", "up", "down"}:
        return "mlp"
    return "other"


def layer_from_param(param_name: str, default: int | None = None) -> int:
    match = re.search(r"model\.layers\.(\d+)\.", str(param_name))
    if match:
        return int(match.group(1))
    if default is not None:
        return int(default)
    return -1


def normalize_task_name(raw: str) -> str:
    text = str(raw or "").strip().lower().replace("-", "_")
    if text in {"tool", "tool_call", "toolcall", "bfcl", "toolrl"}:
        return "tool"
    if text in {"memory", "mem", "hotpotqa", "memagent"}:
        return "memory"
    if text in {"code", "coding", "cure", "livebench", "livecodebench"}:
        return "code"
    return text


def owner_task(expert: str) -> str:
    expert = str(expert)
    if expert in EXPERTS:
        return expert
    return normalize_task_name(expert)


def resolve_span(span: str, task: str) -> str:
    span = str(span or "response")
    task = normalize_task_name(task)
    if span != "signature":
        return span
    if task == "tool":
        return "tool-call"
    if task == "code":
        return "code-block"
    return "response"


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    if math.isnan(result) or math.isinf(result):
        return float(default)
    return result


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def path_run_id(path: Path) -> str:
    if path.name in {
        "signed_utility_summary.json",
        "signed_utility_rows.jsonl",
        "residual_records.jsonl",
        "interference_records.jsonl",
        "gate_records.jsonl",
        "eval_records.jsonl",
        "summary.json",
        "gates.json",
    }:
        return path.parent.name
    return path.stem


def infer_candidate_from_path(path: Path, payload: Mapping[str, Any] | None = None) -> str:
    if payload:
        profile = payload.get("profile")
        if isinstance(profile, Mapping) and profile.get("name"):
            return str(profile["name"])
        model_name = str(payload.get("model_name") or payload.get("experiment_name") or "")
        for name in RCRF_CANDIDATES:
            if name.replace("_", "-") in model_name or name in model_name:
                return name
    parts = [part.lower().replace("-", "_") for part in path.parts]
    joined = "_".join(parts)
    for name in RCRF_CANDIDATES:
        if name == "rcrf":
            continue
        dashed = name.replace("_", "-")
        if name in parts or dashed in parts or name in joined or dashed in joined:
            return name
    if "ta_c100" in joined or "ta_init1" in joined:
        return "ta_init1"
    if "rcrf" in parts:
        return "rcrf"
    if "rcrf" in joined:
        return "rcrf"
    return path.parent.name


def existing_paths(paths: Iterable[str | os.PathLike[str]]) -> list[Path]:
    result: list[Path] = []
    for raw in paths:
        if raw is None:
            continue
        text = str(raw).strip()
        if not text:
            continue
        path = Path(text).expanduser()
        if path.exists():
            result.append(path.resolve())
    return result


def discover_files(paths: Iterable[Path]) -> dict[str, list[Path]]:
    buckets = {
        "residual_jsonl": [],
        "interference_jsonl": [],
        "gate_jsonl": [],
        "eval_jsonl": [],
        "signed_summary": [],
        "signed_rows": [],
        "gates": [],
        "gate_values": [],
        "summary": [],
        "logs": [],
    }

    def add_file(path: Path) -> None:
        name = path.name
        if name == "residual_records.jsonl":
            buckets["residual_jsonl"].append(path)
        elif name == "interference_records.jsonl":
            buckets["interference_jsonl"].append(path)
        elif name == "gate_records.jsonl":
            buckets["gate_jsonl"].append(path)
        elif name == "eval_records.jsonl":
            buckets["eval_jsonl"].append(path)
        elif name == "signed_utility_summary.json":
            buckets["signed_summary"].append(path)
        elif name == "signed_utility_rows.jsonl":
            buckets["signed_rows"].append(path)
        elif name == "gates.json":
            buckets["gates"].append(path)
        elif name == "gate_values.json":
            buckets["gate_values"].append(path)
        elif name == "summary.json":
            buckets["summary"].append(path)
        elif name.endswith(".log"):
            buckets["logs"].append(path)

    for root in paths:
        if root.is_file():
            add_file(root)
            continue
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            # Keep discovery cheap and ignore model shards/tokenizers.
            if path.suffix in {".safetensors", ".bin", ".pt", ".pth"}:
                continue
            add_file(path)
    return {key: sorted(set(value)) for key, value in buckets.items()}


def ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for column in columns:
        if column not in df.columns:
            df[column] = None
    return df


def normalize_residual_df(df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_columns(df.copy(), RESIDUAL_COLUMNS)
    if "module_family" in df.columns:
        df["module_family"] = df.apply(
            lambda row: row["module_family"] if row["module_family"] not in {None, "", "attn_q", "attn_k", "attn_v", "attn_o"} else module_family(str(row["module"])),
            axis=1,
        )
    df["expert"] = df["expert"].astype(str)
    df["task"] = df["task"].map(normalize_task_name)
    df["module"] = df["module"].map(module_from_param)
    df["module_family"] = df["module"].map(module_family)
    df["layer"] = df["layer"].map(lambda value: safe_int(value, -1))
    for col in ("signed_effect", "expression", "positive_fraction"):
        df[col] = df[col].map(safe_float)
    df["prompt_id"] = df["prompt_id"].fillna("aggregate").astype(str)
    df["span_type"] = df["span_type"].fillna("response").astype(str)
    df["run_id"] = df["run_id"].fillna("unknown").astype(str)
    return df


def normalize_interference_df(df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_columns(df.copy(), INTERFERENCE_COLUMNS)
    df["expert_a"] = df["expert_a"].astype(str)
    df["expert_b"] = df["expert_b"].astype(str)
    df["task"] = df["task"].map(normalize_task_name)
    df["module"] = df["module"].map(module_from_param)
    df["module_family"] = df["module"].map(module_family)
    df["layer"] = df["layer"].map(lambda value: safe_int(value, -1))
    for col in ("cosine", "conflict_score", "cross_harm"):
        df[col] = df[col].map(safe_float)
    if "shared_positive_effect" not in df.columns:
        df["shared_positive_effect"] = 0.0
    df["shared_positive_effect"] = df["shared_positive_effect"].map(safe_float)
    return df


def normalize_gate_df(df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_columns(df.copy(), GATE_COLUMNS)
    df["candidate"] = df["candidate"].fillna("unknown").astype(str)
    df["expert"] = df["expert"].astype(str)
    df["module"] = df["module"].map(module_from_param)
    df["module_family"] = df["module"].map(module_family)
    df["layer"] = df["layer"].map(lambda value: safe_int(value, -1))
    for col in ("alpha", "owner_signal", "synergy_signal", "harm_signal", "noise_score"):
        df[col] = df[col].map(safe_float)
    for optional in ("cross_harm", "conflict_score", "owner_effect", "owner_expression", "positive_fraction"):
        if optional in df.columns:
            df[optional] = df[optional].map(safe_float)
    return df


def normalize_eval_df(df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_columns(df.copy(), EVAL_COLUMNS)
    df["candidate"] = df["candidate"].fillna("unknown").astype(str)
    df["task"] = df["task"].map(normalize_task_name)
    df["subset"] = df["subset"].fillna("unknown").astype(str)
    df["metric"] = df["metric"].fillna("score").astype(str)
    df["score"] = df["score"].map(safe_float)
    df["n_examples"] = df["n_examples"].map(lambda value: safe_int(value, 0))
    return df


def load_mechanism_data(
    input_paths: Iterable[str | os.PathLike[str]] | None = None,
    *,
    include_defaults: bool = True,
    example_on_empty: bool = True,
) -> MechanismData:
    search_paths: list[str | os.PathLike[str]] = []
    if input_paths:
        search_paths.extend(input_paths)
    if include_defaults:
        search_paths.extend(DEFAULT_INPUT_PATHS)

    paths = existing_paths(search_paths)
    files = discover_files(paths)
    notes = [f"search_path: {path}" for path in paths]

    residual_records: list[dict[str, Any]] = []
    interference_records: list[dict[str, Any]] = []
    gate_records: list[dict[str, Any]] = []
    eval_records: list[dict[str, Any]] = []

    for path in files["residual_jsonl"]:
        residual_records.extend(read_jsonl(path))
        notes.append(f"loaded residual_records: {path}")
    for path in files["interference_jsonl"]:
        interference_records.extend(read_jsonl(path))
        notes.append(f"loaded interference_records: {path}")
    for path in files["gate_jsonl"]:
        gate_records.extend(read_jsonl(path))
        notes.append(f"loaded gate_records: {path}")
    for path in files["eval_jsonl"]:
        eval_records.extend(read_jsonl(path))
        notes.append(f"loaded eval_records: {path}")

    signed_summaries = {}
    for path in files["signed_summary"]:
        try:
            signed_summaries[path.parent.resolve()] = (path, read_json(path))
        except Exception as exc:  # noqa: BLE001
            notes.append(f"skip signed summary {path}: {exc}")

    row_dirs = {path.parent.resolve() for path in files["signed_rows"]}
    for path in files["signed_rows"]:
        summary = signed_summaries.get(path.parent.resolve(), (None, {}))[1]
        residual_records.extend(parse_signed_rows(path, summary))
        notes.append(f"parsed signed_utility_rows: {path}")

    for folder, (path, payload) in signed_summaries.items():
        if folder not in row_dirs:
            residual_records.extend(parse_signed_summary(path, payload))
            notes.append(f"parsed signed_utility_summary: {path}")
        interference_records.extend(parse_interference_from_signed_summary(path, payload))
        notes.append(f"parsed conflict_summary: {path}")

    for path in files["gates"]:
        try:
            payload = read_json(path)
            gate_records.extend(parse_gate_payload(path, payload))
            notes.append(f"parsed gates: {path}")
        except Exception as exc:  # noqa: BLE001
            notes.append(f"skip gates {path}: {exc}")
    for path in files["gate_values"]:
        try:
            payload = read_json(path)
            gate_records.extend(parse_gate_values(path, payload))
            notes.append(f"parsed gate_values: {path}")
        except Exception as exc:  # noqa: BLE001
            notes.append(f"skip gate_values {path}: {exc}")

    for path in files["summary"]:
        try:
            eval_records.extend(parse_eval_summary(path, read_json(path)))
            notes.append(f"parsed eval summary: {path}")
        except Exception as exc:  # noqa: BLE001
            notes.append(f"skip eval summary {path}: {exc}")
    for path in files["logs"]:
        try:
            eval_records.extend(parse_eval_log(path))
            notes.append(f"parsed eval log: {path}")
        except Exception as exc:  # noqa: BLE001
            notes.append(f"skip eval log {path}: {exc}")

    data = MechanismData(
        residual=normalize_residual_df(pd.DataFrame(residual_records)) if residual_records else empty_df(RESIDUAL_COLUMNS),
        interference=normalize_interference_df(pd.DataFrame(interference_records)) if interference_records else empty_df(INTERFERENCE_COLUMNS),
        gate=normalize_gate_df(pd.DataFrame(gate_records)) if gate_records else empty_df(GATE_COLUMNS),
        eval=normalize_eval_df(pd.DataFrame(eval_records)) if eval_records else empty_df(EVAL_COLUMNS),
        source_notes=notes,
        used_example=False,
    )

    if example_on_empty and data.residual.empty and data.interference.empty and data.gate.empty:
        example = make_example_data()
        example.source_notes = notes + ["no usable real mechanism files found; using deterministic example data"]
        return example

    data.residual = dedupe(data.residual)
    data.interference = dedupe(data.interference)
    data.gate = dedupe(data.gate)
    data.eval = dedupe(data.eval)
    return data


def empty_df(columns: Iterable[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.drop_duplicates().reset_index(drop=True)


def parse_signed_rows(path: Path, summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    config = summary.get("config", {}) if isinstance(summary, Mapping) else {}
    configured_span = str(config.get("span") or "response")
    run_id = path_run_id(path)
    output: list[dict[str, Any]] = []
    for row in read_jsonl(path):
        details = row.get("details")
        if not isinstance(details, list):
            continue
        for item in details:
            if not isinstance(item, Mapping):
                continue
            param_name = str(item.get("param_name") or "")
            module = module_from_param(param_name or str(item.get("module") or ""))
            task = normalize_task_name(str(item.get("task") or row.get("task") or ""))
            signed_effect = safe_float(item.get("signed_effect"))
            output.append(
                {
                    "run_id": run_id,
                    "expert": str(item.get("expert") or ""),
                    "task": task,
                    "prompt_id": str(item.get("row_id") or row.get("row_id") or "unknown"),
                    "span_type": resolve_span(configured_span, task),
                    "layer": safe_int(item.get("layer"), layer_from_param(param_name, -1)),
                    "module": module,
                    "module_family": module_family(module),
                    "signed_effect": signed_effect,
                    "expression": safe_float(item.get("expression")),
                    "positive_fraction": 1.0 if signed_effect > 0.0 else 0.0,
                    "param_name": param_name,
                    "source_path": str(path),
                }
            )
    return output


def parse_signed_summary(path: Path, payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    config = payload.get("config", {}) if isinstance(payload, Mapping) else {}
    configured_span = str(config.get("span") or "response")
    run_id = path_run_id(path)
    output: list[dict[str, Any]] = []
    module_summary = payload.get("module_summary", {}) if isinstance(payload, Mapping) else {}
    for expert, module_map in module_summary.items():
        if not isinstance(module_map, Mapping):
            continue
        for param_name, task_map in module_map.items():
            module = module_from_param(str(param_name))
            for task, stats in task_map.items():
                if not isinstance(stats, Mapping):
                    continue
                task_name = normalize_task_name(str(task))
                output.append(
                    {
                        "run_id": run_id,
                        "expert": str(expert),
                        "task": task_name,
                        "prompt_id": "aggregate",
                        "span_type": resolve_span(configured_span, task_name),
                        "layer": layer_from_param(str(param_name), -1),
                        "module": module,
                        "module_family": module_family(module),
                        "signed_effect": safe_float(stats.get("signed_effect_mean")),
                        "expression": safe_float(stats.get("expression_mean")),
                        "positive_fraction": safe_float(stats.get("positive_fraction")),
                        "count": safe_float(stats.get("count")),
                        "param_name": str(param_name),
                        "source_path": str(path),
                    }
                )
    return output


def parse_interference_from_signed_summary(path: Path, payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    module_stats = index_module_stats(payload)
    output: list[dict[str, Any]] = []
    conflict_summary = payload.get("conflict_summary", {}) if isinstance(payload, Mapping) else {}
    for task, conflict_map in conflict_summary.items():
        task_name = normalize_task_name(str(task))
        if not isinstance(conflict_map, Mapping):
            continue
        for key, stats in conflict_map.items():
            match = re.match(r"layer_(\d+):([^|]+)\|(.+)$", str(key))
            if not match or not isinstance(stats, Mapping):
                continue
            layer = int(match.group(1))
            expert_a = match.group(2)
            expert_b = match.group(3)
            cosine = safe_float(stats.get("cosine_mean"))
            negative_fraction = safe_float(stats.get("negative_fraction"))
            base_conflict = max(0.0, negative_fraction - 0.5) + max(0.0, -cosine)
            base_conflict = min(base_conflict, 1.0)
            for module in MODULE_ORDER:
                a_stats = module_stats.get(expert_a, {}).get(layer, {}).get(module, {}).get(task_name, {})
                b_stats = module_stats.get(expert_b, {}).get(layer, {}).get(module, {}).get(task_name, {})
                if not a_stats and not b_stats:
                    continue
                signed_a = safe_float(a_stats.get("signed_effect_mean"))
                signed_b = safe_float(b_stats.get("signed_effect_mean"))
                harm_a = max(0.0, -signed_a) + safe_float(a_stats.get("harm_mean"))
                harm_b = max(0.0, -signed_b) + safe_float(b_stats.get("harm_mean"))
                shared_positive = 0.0
                if signed_a > 0.0 and signed_b > 0.0:
                    shared_positive = min(signed_a, signed_b)
                output.append(
                    {
                        "expert_a": expert_a,
                        "expert_b": expert_b,
                        "task": task_name,
                        "layer": layer,
                        "module": module,
                        "module_family": module_family(module),
                        "cosine": cosine,
                        "conflict_score": base_conflict,
                        "cross_harm": (harm_a + harm_b) / 2.0,
                        "shared_positive_effect": shared_positive,
                        "negative_fraction": negative_fraction,
                        "source_path": str(path),
                    }
                )
    return output


def index_module_stats(payload: Mapping[str, Any]) -> dict[str, dict[int, dict[str, dict[str, Any]]]]:
    result: dict[str, dict[int, dict[str, dict[str, Any]]]] = {}
    module_summary = payload.get("module_summary", {}) if isinstance(payload, Mapping) else {}
    for expert, module_map in module_summary.items():
        expert_map = result.setdefault(str(expert), {})
        if not isinstance(module_map, Mapping):
            continue
        for param_name, task_map in module_map.items():
            module = module_from_param(str(param_name))
            layer = layer_from_param(str(param_name), -1)
            expert_map.setdefault(layer, {})[module] = task_map if isinstance(task_map, Mapping) else {}
    return result


def parse_gate_payload(path: Path, payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidate = infer_candidate_from_path(path, payload)
    rows = payload.get("decision_rows") if isinstance(payload, Mapping) else None
    output: list[dict[str, Any]] = []
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            metrics = row.get("metrics", {}) if isinstance(row.get("metrics"), Mapping) else {}
            module = module_from_param(str(row.get("family") or row.get("param_name") or row.get("module") or ""))
            output.append(
                {
                    "candidate": candidate,
                    "expert": str(row.get("expert") or ""),
                    "layer": safe_int(row.get("layer")),
                    "module": module,
                    "module_family": module_family(module),
                    "alpha": safe_float(row.get("coefficient")),
                    "owner_signal": safe_float(metrics.get("owner_signal")),
                    "synergy_signal": safe_float(metrics.get("synergy_signal")),
                    "harm_signal": max(safe_float(metrics.get("direct_harm")), safe_float(metrics.get("conflict_score"))),
                    "noise_score": safe_float(metrics.get("noise_score")),
                    "cross_harm": safe_float(metrics.get("cross_harm")),
                    "conflict_score": safe_float(metrics.get("conflict_score")),
                    "owner_effect": safe_float(metrics.get("owner_effect")),
                    "owner_expression": safe_float(metrics.get("owner_expression")),
                    "positive_fraction": safe_float(metrics.get("positive_fraction")),
                    "reason": str(row.get("reason") or ""),
                    "param_name": str(row.get("param_name") or ""),
                    "source_path": str(path),
                }
            )
        return output
    if isinstance(payload, Mapping) and isinstance(payload.get("gates"), Mapping):
        output.extend(parse_gate_values(path, payload["gates"], candidate=candidate))
    return output


def parse_gate_values(path: Path, payload: Mapping[str, Any], *, candidate: str | None = None) -> list[dict[str, Any]]:
    candidate = candidate or infer_candidate_from_path(path)
    output: list[dict[str, Any]] = []
    for key, value in iter_gate_items(payload):
        param_name = str(key)
        expert = ""
        if "::" in param_name:
            param_name, expert = param_name.rsplit("::", 1)
        module = module_from_param(param_name)
        output.append(
            {
                "candidate": candidate,
                "expert": expert or infer_expert_from_key(str(key)),
                "layer": layer_from_param(param_name, -1),
                "module": module,
                "module_family": module_family(module),
                "alpha": safe_float(value),
                "owner_signal": 0.0,
                "synergy_signal": 0.0,
                "harm_signal": 0.0,
                "noise_score": 0.0,
                "param_name": param_name,
                "source_path": str(path),
            }
        )
    return output


def iter_gate_items(payload: Mapping[str, Any], prefix: str = "") -> Iterable[tuple[str, Any]]:
    for key, value in payload.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            yield from iter_gate_items(value, full_key)
        elif isinstance(value, (int, float)):
            yield full_key, value


def infer_expert_from_key(key: str) -> str:
    for expert in EXPERTS:
        if f"::{expert}" in key or key.endswith(f".{expert}") or expert in key.split("."):
            return expert
    return ""


def parse_eval_summary(path: Path, payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidate = infer_candidate_from_path(path, payload)
    output: list[dict[str, Any]] = []
    if isinstance(payload.get("datasets"), list):
        for dataset in payload["datasets"]:
            if not isinstance(dataset, Mapping):
                continue
            subset = str(dataset.get("dataset") or "code")
            for metric in (
                "code_acc",
                "code_accumulate_acc",
                "estimated_unit_test_acc",
                "estimated_unit_test_accumulate_acc",
                "estimated_p_00",
                "estimated_p_01",
            ):
                if metric in dataset:
                    output.append(
                        {
                            "candidate": candidate,
                            "task": "code",
                            "subset": subset,
                            "metric": metric,
                            "score": safe_float(dataset.get(metric)),
                            "n_examples": safe_int(dataset.get("n_examples"), 0),
                            "source_path": str(path),
                        }
                    )
            bon = dataset.get("bon")
            if isinstance(bon, Mapping):
                for scale, scale_stats in bon.items():
                    if not isinstance(scale_stats, Mapping):
                        continue
                    for metric, value in scale_stats.items():
                        output.append(
                            {
                                "candidate": candidate,
                                "task": "code",
                                "subset": f"{subset}:BoN{scale}",
                                "metric": str(metric),
                                "score": safe_float(value),
                                "n_examples": safe_int(dataset.get("n_examples"), 0),
                                "source_path": str(path),
                            }
                        )
    task_stats = payload.get("task_stats")
    if isinstance(task_stats, Mapping):
        for task, stats in task_stats.items():
            if not isinstance(stats, Mapping):
                continue
            subset = "toolrl80" if normalize_task_name(str(task)) == "tool" else "summary"
            for metric, value in stats.items():
                if isinstance(value, (int, float)) and metric not in {"rows", "samples"}:
                    output.append(
                        {
                            "candidate": candidate,
                            "task": normalize_task_name(str(task)),
                            "subset": subset,
                            "metric": str(metric),
                            "score": safe_float(value),
                            "n_examples": safe_int(stats.get("rows") or stats.get("samples"), 0),
                            "source_path": str(path),
                        }
                    )
    return output


def parse_eval_log(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    candidate = infer_candidate_from_path(path)
    output: list[dict[str, Any]] = []
    for payload in extract_json_objects(text):
        if isinstance(payload.get("scores"), Mapping):
            for subset, stats in payload["scores"].items():
                if not isinstance(stats, Mapping):
                    continue
                output.append(
                    {
                        "candidate": candidate,
                        "task": "tool",
                        "subset": str(subset),
                        "metric": "accuracy",
                        "score": safe_float(stats.get("accuracy")),
                        "n_examples": safe_int(stats.get("total_count")),
                        "source_path": str(path),
                    }
                )
        output.extend(find_memory_eval_records(payload, candidate=candidate, source_path=path))
    return output


def extract_json_objects(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    index = 0
    while index < len(text):
        start = text.find("{", index)
        if start < 0:
            break
        try:
            obj, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            index = start + 1
            continue
        if isinstance(obj, dict):
            objects.append(obj)
        index = start + max(end, 1)
    return objects


def find_memory_eval_records(payload: Any, *, candidate: str, source_path: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    if isinstance(payload, Mapping):
        if "dataset" in payload and "avg_f1" in payload:
            output.append(
                {
                    "candidate": candidate,
                    "task": "memory",
                    "subset": str(payload.get("dataset") or "memory"),
                    "metric": "avg_f1",
                    "score": safe_float(payload.get("avg_f1")),
                    "n_examples": 0,
                    "source_path": str(source_path),
                }
            )
        for value in payload.values():
            output.extend(find_memory_eval_records(value, candidate=candidate, source_path=source_path))
    elif isinstance(payload, list):
        for item in payload:
            output.extend(find_memory_eval_records(item, candidate=candidate, source_path=source_path))
    return output


def write_longform(output_dir: str | os.PathLike[str], data: MechanismData) -> dict[str, str]:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "residual": output / "residual_records.jsonl",
        "interference": output / "interference_records.jsonl",
        "gate": output / "gate_records.jsonl",
        "eval": output / "eval_records.jsonl",
    }
    write_df_jsonl(paths["residual"], data.residual)
    write_df_jsonl(paths["interference"], data.interference)
    write_df_jsonl(paths["gate"], data.gate)
    write_df_jsonl(paths["eval"], data.eval)
    (output / "source_notes.txt").write_text("\n".join(data.source_notes) + "\n", encoding="utf-8")
    return {key: str(value) for key, value in paths.items()}


def write_df_jsonl(path: Path, df: pd.DataFrame) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in df.to_dict(orient="records"):
            clean = {key: sanitize_json_value(value) for key, value in row.items()}
            handle.write(json.dumps(clean, ensure_ascii=False, sort_keys=True) + "\n")


def sanitize_json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def make_example_data(num_prompts: int = 96) -> MechanismData:
    layers = [0, 4, 8, 12, 16, 20, 24, 27]
    residual_rows: list[dict[str, Any]] = []
    for prompt_idx in range(num_prompts):
        task = TASKS[prompt_idx % len(TASKS)]
        prompt_id = f"{task}_example_{prompt_idx:03d}"
        span = resolve_span("signature", task)
        for expert in EXPERTS:
            for layer in layers:
                layer_phase = math.sin((layer + 1) * 0.37 + prompt_idx * 0.11)
                for module in MODULE_ORDER:
                    is_owner = owner_task(expert) == task
                    family = module_family(module)
                    expression = example_expression(expert, task, module, layer, layer_phase)
                    signed = example_signed_effect(expert, task, module, layer, layer_phase)
                    if is_owner and module in {"gate", "up", "down"}:
                        signed += expression * (0.003 if expert != "code" else 0.0006)
                    if not is_owner and expert == "code" and task == "memory" and layer >= 20 and module in {"q", "k"}:
                        signed -= expression * 0.004
                    residual_rows.append(
                        {
                            "run_id": "example_rcrf_mechanism",
                            "expert": expert,
                            "task": task,
                            "prompt_id": prompt_id,
                            "span_type": span,
                            "layer": layer,
                            "module": module,
                            "module_family": family,
                            "signed_effect": signed,
                            "expression": expression,
                            "positive_fraction": 1.0 if signed > 0 else 0.0,
                        }
                    )

    interference_rows: list[dict[str, Any]] = []
    for task in TASKS:
        for expert_a, expert_b in combinations(EXPERTS, 2):
            for layer in layers:
                for module in MODULE_ORDER:
                    late = max(0.0, (layer - 12) / 15.0)
                    routing = 1.0 if module in {"q", "k"} else 0.35
                    pair_is_code_memory = {expert_a, expert_b} == {"code", "memory"}
                    pair_is_code_tool = {expert_a, expert_b} == {"code", "tool"}
                    cosine = 0.05 * math.sin(layer + len(module))
                    if pair_is_code_memory and task in {"code", "memory"}:
                        cosine -= 0.20 * late * routing
                    if pair_is_code_tool and task == "code":
                        cosine -= 0.13 * late * routing
                    conflict = max(0.0, -cosine) + (0.18 * late * routing if cosine < 0 else 0.0)
                    cross_harm = max(0.0, -cosine) * 2.5e-5 + (1.0e-6 if module_family(module) == "mlp" else 3.0e-6)
                    interference_rows.append(
                        {
                            "expert_a": expert_a,
                            "expert_b": expert_b,
                            "task": task,
                            "layer": layer,
                            "module": module,
                            "module_family": module_family(module),
                            "cosine": cosine,
                            "conflict_score": min(conflict, 1.0),
                            "cross_harm": cross_harm,
                            "shared_positive_effect": max(0.0, 2.0e-6 - cross_harm * 0.2),
                        }
                    )

    gate_rows: list[dict[str, Any]] = []
    for candidate in RCRF_CANDIDATES:
        for expert in EXPERTS:
            for layer in range(28):
                for module in MODULE_ORDER:
                    alpha, owner_signal, synergy_signal, harm_signal, noise_score = example_gate(candidate, expert, module, layer)
                    gate_rows.append(
                        {
                            "candidate": candidate,
                            "expert": expert,
                            "layer": layer,
                            "module": module,
                            "module_family": module_family(module),
                            "alpha": alpha,
                            "owner_signal": owner_signal,
                            "synergy_signal": synergy_signal,
                            "harm_signal": harm_signal,
                            "noise_score": noise_score,
                            "cross_harm": harm_signal * 1.0e-5,
                            "conflict_score": harm_signal,
                            "positive_fraction": max(0.0, min(1.0, owner_signal)),
                            "reason": example_gate_reason(alpha, harm_signal, noise_score),
                        }
                    )

    eval_rows = [
        {"candidate": "rcrf", "task": "tool", "subset": "live_parallel", "metric": "accuracy", "score": 0.8125, "n_examples": 16},
        {"candidate": "rcrf", "task": "tool", "subset": "parallel", "metric": "accuracy", "score": 0.8800, "n_examples": 200},
        {"candidate": "energy_only", "task": "tool", "subset": "live_parallel", "metric": "accuracy", "score": 0.7500, "n_examples": 16},
        {"candidate": "owner_only", "task": "memory", "subset": "eval_100", "metric": "avg_f1", "score": 0.7574, "n_examples": 100},
        {"candidate": "no_conflict", "task": "memory", "subset": "eval_100", "metric": "avg_f1", "score": 0.7372, "n_examples": 100},
        {"candidate": "rcrf", "task": "memory", "subset": "eval_100", "metric": "avg_f1", "score": 0.7567, "n_examples": 100},
        {"candidate": "rcrf", "task": "code", "subset": "LiveBench", "metric": "code_acc", "score": 0.3789, "n_examples": 0},
        {"candidate": "rcrf", "task": "code", "subset": "LiveCodeBench", "metric": "code_acc", "score": 0.2862, "n_examples": 0},
        {"candidate": "rcrf", "task": "code", "subset": "LiveCodeBench:BoN(4, 4)", "metric": "acc", "score": 0.3464, "n_examples": 0},
    ]

    return MechanismData(
        residual=normalize_residual_df(pd.DataFrame(residual_rows)),
        interference=normalize_interference_df(pd.DataFrame(interference_rows)),
        gate=normalize_gate_df(pd.DataFrame(gate_rows)),
        eval=normalize_eval_df(pd.DataFrame(eval_rows)),
        source_notes=["deterministic example data generated by analysis_platform.rcrf_schema"],
        used_example=True,
    )


def example_expression(expert: str, task: str, module: str, layer: int, phase: float) -> float:
    base = 1.0e-5
    if expert == "memory":
        base *= 12.0
    if module in {"gate", "up", "down"}:
        base *= 8.0
    if expert == "code" and module in {"gate", "up"}:
        base *= 3.5
    if owner_task(expert) == task:
        base *= 1.35
    if layer >= 16 and module in {"q", "k"}:
        base *= 1.15
    return max(1.0e-9, base * (1.0 + 0.18 * phase))


def example_signed_effect(expert: str, task: str, module: str, layer: int, phase: float) -> float:
    expression = example_expression(expert, task, module, layer, phase)
    signed = expression * 0.0004 * phase
    if owner_task(expert) == task:
        if expert == "memory":
            signed += expression * 0.0025
        elif expert == "tool":
            signed += expression * (0.0018 if module in {"o", "gate", "up", "down"} else 0.0007)
        else:
            signed += expression * (0.0005 if module in {"gate", "up", "down"} else 0.0001)
    else:
        signed -= expression * (0.0008 if module in {"q", "k"} and layer >= 16 else 0.00015)
    return signed


def example_gate(candidate: str, expert: str, module: str, layer: int) -> tuple[float, float, float, float, float]:
    mlp = module_family(module) == "mlp"
    routing = module in {"q", "k"}
    late = max(0.0, (layer - 10) / 17.0)
    owner_signal = 0.75 if (expert in {"tool", "memory"} and mlp) else 0.42
    if expert == "code":
        owner_signal = 0.32 if not mlp else 0.48
    synergy_signal = 0.18 if mlp else 0.08
    harm_signal = (0.16 + 0.30 * late) if routing else 0.06
    if expert == "code" and routing and layer >= 16:
        harm_signal += 0.22
    noise_score = 1.0 if (expert == "code" and routing) else 0.0

    if candidate == "owner_only":
        alpha = 1.0 + 0.10 * owner_signal - 0.04 * noise_score
    elif candidate == "energy_only":
        alpha = 1.0 + (0.05 if mlp else -0.03) - 0.03 * late
    elif candidate == "no_conflict":
        alpha = 1.0 + 0.10 * owner_signal + 0.04 * synergy_signal - 0.03 * noise_score
    else:
        alpha = 1.0 + 0.12 * owner_signal + 0.05 * synergy_signal - 0.18 * harm_signal - 0.10 * noise_score
    return max(0.55, min(1.12, alpha)), owner_signal, synergy_signal, harm_signal, noise_score


def example_gate_reason(alpha: float, harm_signal: float, noise_score: float) -> str:
    if noise_score > 0 and alpha < 1.0:
        return "suppress_low_energy_or_unstable_residual"
    if harm_signal > 0.25 and alpha < 1.0:
        return "suppress_conflict_or_harm"
    if alpha > 1.03:
        return "keep_or_amplify_owner_utility"
    if alpha < 0.97:
        return "mild_filter"
    return "neutral"


def summarize_research_questions(
    residual: pd.DataFrame,
    interference: pd.DataFrame,
    gate: pd.DataFrame,
    *,
    top_k: int = 20,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not residual.empty:
        grouped = (
            residual.groupby(["expert", "task", "layer", "module", "module_family"], as_index=False)
            .agg(
                signed_effect=("signed_effect", "mean"),
                expression=("expression", "mean"),
                positive_fraction=("positive_fraction", "mean"),
                prompt_count=("prompt_id", "nunique"),
            )
        )
        expr_cut = grouped["expression"].quantile(0.80) if len(grouped) else 0.0
        bad = grouped[(grouped["expression"] >= expr_cut) & (grouped["signed_effect"] < 0.0)].copy()
        for item in bad.sort_values(["expression", "signed_effect"], ascending=[False, True]).head(top_k).to_dict("records"):
            rows.append(
                {
                    "question_type": "high_expression_negative_effect",
                    "priority": "P0",
                    "question": "Why is an expressed residual pointing against the teacher-forced trajectory?",
                    "evidence": format_module_evidence(item),
                    "next_action": "Inspect prompt spans and compare whether suppression improves the owner task without hurting protected tasks.",
                }
            )

        prompt_stats = (
            residual[residual["prompt_id"] != "aggregate"]
            .groupby(["expert", "task", "span_type", "layer", "module"], as_index=False)
            .agg(
                signed_std=("signed_effect", "std"),
                positive_fraction=("positive_fraction", "mean"),
                prompt_count=("prompt_id", "nunique"),
            )
            .fillna({"signed_std": 0.0})
        )
        unstable = prompt_stats[
            (prompt_stats["prompt_count"] >= 3)
            & (prompt_stats["positive_fraction"] > 0.15)
            & (prompt_stats["positive_fraction"] < 0.85)
        ]
        for item in unstable.sort_values("signed_std", ascending=False).head(top_k).to_dict("records"):
            rows.append(
                {
                    "question_type": "prompt_span_bias",
                    "priority": "P1",
                    "question": "Which prompt or span choice flips the signed utility sign?",
                    "evidence": format_module_evidence(item),
                    "next_action": "Split by response/code/tool-call span and rerun the probe on pass/fail or task-specific spans.",
                }
            )

    if not interference.empty:
        routing = interference[
            (interference["module"].isin(["q", "k"]))
            & ((interference["conflict_score"] >= interference["conflict_score"].quantile(0.80)) | (interference["cosine"] < 0.0))
        ]
        for item in routing.sort_values(["conflict_score", "cosine"], ascending=[False, True]).head(top_k).to_dict("records"):
            rows.append(
                {
                    "question_type": "qk_routing_conflict",
                    "priority": "P0",
                    "question": "Are q/k routing residuals forcing two experts to attend to incompatible evidence?",
                    "evidence": format_module_evidence(item),
                    "next_action": "Test a q/k-only restore or suppression ablation for this layer and pair.",
                }
            )

    if not gate.empty:
        mlp = gate[(gate["module_family"] == "mlp") & (gate["alpha"] < 0.97)]
        if "owner_signal" in mlp.columns:
            mlp = mlp[mlp["owner_signal"] >= mlp["owner_signal"].quantile(0.70)]
        for item in mlp.sort_values(["owner_signal", "alpha"], ascending=[False, True]).head(top_k).to_dict("records"):
            rows.append(
                {
                    "question_type": "mlp_over_suppressed",
                    "priority": "P1",
                    "question": "Is a stable MLP owner-utility channel being over-suppressed by conflict/noise terms?",
                    "evidence": format_module_evidence(item),
                    "next_action": "Restore this MLP module toward alpha=1 and check owner/protected eval deltas.",
                }
            )

    return pd.DataFrame(rows)


def format_module_evidence(item: Mapping[str, Any]) -> str:
    pieces = []
    for key in ("candidate", "expert", "expert_a", "expert_b", "task", "span_type", "layer", "module"):
        if key in item and item[key] not in {None, ""}:
            pieces.append(f"{key}={item[key]}")
    for key in ("signed_effect", "expression", "positive_fraction", "cosine", "conflict_score", "cross_harm", "alpha", "owner_signal"):
        if key in item and item[key] not in {None, ""}:
            pieces.append(f"{key}={safe_float(item[key]):.4g}")
    return ", ".join(pieces)
