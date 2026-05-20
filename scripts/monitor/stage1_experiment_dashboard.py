#!/usr/bin/env python3
"""Read-only dashboard for ExpertGym/TRC experiment decisions."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

EXPERTS = ("tool", "memory", "code")
REPO_ROOT = Path(__file__).resolve().parents[2]
TRC_RUNS_ROOT = Path("/tmp/shared-storage/OnPolicy/runs/trc")
LEDGER_PATH = REPO_ROOT / "docs/harness/20260520_experiment_ledger.md"
ROUND_EVAL_PATHS = [
    REPO_ROOT / "docs/evaluation/20260520_trc_round10_eval.md",
    REPO_ROOT / "docs/evaluation/20260520_trc_round11_eval.md",
    REPO_ROOT / "docs/evaluation/20260520_trc_round12_eval.md",
]
ROUND13_CONFIG_PATH = REPO_ROOT / "docs/config/hiddenstate/20260520_round13_evalleak_code16.md"

EVALS: dict[str, dict[str, Any]] = {
    "anchor_i4": {
        "label": "TRC anchor i4",
        "path": "/tmp/shared-storage/OnPolicy/eval/full_suite/trc_stage1_v3_anchor_i4_20260519/stage1_20260519",
        "checkpoint": "trc_stage1_v3_anchor_i4_20260519",
        "checkpoint_path": "/tmp/shared-storage/OnPolicy/checkpoints/trc_stage1_v3_anchor_i4_20260519",
        "train_run": "/tmp/shared-storage/OnPolicy/runs/trc/trc_layer_init1_v3_directional_anchor_20260519",
        "train_command": "skill/command/run_20260519_trc_layer_init1_v3_directional.sh",
        "setting": {
            "family": "TRC directional anchor",
            "epoch": "4",
            "init": "1.0",
            "objective": "1 - cos(r_merge, r_expert)",
            "parameterization": "layer-band-coefficient",
            "calibration": "trc96 expert trajectories",
            "lr": "0.02",
            "optimizer": "adamw",
            "accumulation_steps": "96",
            "hidden_layers": "8,16,24,28",
            "span": "auto: tool_call / code block / memory response",
            "beta_base": "0.05",
            "gamma_gate": "0.005",
            "coefficient_floor": "0.95",
            "coefficient_floor_weight": "0.1",
            "baked_delta_entries": "588",
        },
    },
    "anchor_i8": {
        "label": "TRC anchor i8 rerun",
        "path": "/tmp/shared-storage/OnPolicy/eval/full_suite/trc_stage1_v3_anchor_i8_20260519/stage1_20260519_rerun",
        "checkpoint": "trc_stage1_v3_anchor_i8_20260519",
        "checkpoint_path": "/tmp/shared-storage/OnPolicy/checkpoints/trc_stage1_v3_anchor_i8_20260519",
        "train_run": "/tmp/shared-storage/OnPolicy/runs/trc/trc_layer_init1_v3_directional_anchor_20260519",
        "train_command": "skill/command/run_20260519_trc_layer_init1_v3_directional.sh",
        "setting": {
            "family": "TRC directional anchor",
            "epoch": "8",
            "init": "1.0",
            "objective": "1 - cos(r_merge, r_expert)",
            "parameterization": "layer-band-coefficient",
            "calibration": "trc96 expert trajectories",
            "lr": "0.02",
            "optimizer": "adamw",
            "accumulation_steps": "96",
            "hidden_layers": "8,16,24,28",
            "span": "auto: tool_call / code block / memory response",
            "beta_base": "0.05",
            "gamma_gate": "0.005",
            "coefficient_floor": "0.95",
            "coefficient_floor_weight": "0.1",
            "baked_delta_entries": "588",
        },
    },
    "dir_i8": {
        "label": "TRC dir i8",
        "path": "/tmp/shared-storage/OnPolicy/eval/full_suite/trc_stage1_v3_dir_i8_20260519/stage1_20260519_dir_i8",
        "checkpoint": "trc_stage1_v3_dir_i8_20260519",
        "checkpoint_path": "/tmp/shared-storage/OnPolicy/checkpoints/trc_stage1_v3_dir_i8_20260519",
        "train_run": "/tmp/shared-storage/OnPolicy/runs/trc/trc_layer_init1_v3_directional_20260519",
        "train_command": "skill/command/run_20260519_trc_layer_init1_v3_directional.sh",
        "setting": {
            "family": "TRC directional",
            "epoch": "8",
            "init": "1.0",
            "objective": "1 - cos(r_merge, r_expert)",
            "parameterization": "layer-band-coefficient",
            "calibration": "trc96 expert trajectories",
            "lr": "0.03",
            "optimizer": "adamw",
            "accumulation_steps": "96",
            "hidden_layers": "8,16,24,28",
            "span": "auto: tool_call / code block / memory response",
            "beta_base": "0.02",
            "gamma_gate": "0.001",
            "directional_projection_floor": "0.8",
            "directional_projection_weight": "0.1",
            "coefficient_floor": "0.9",
            "coefficient_floor_weight": "0.05",
            "baked_delta_entries": "588",
        },
    },
}

RUNS: dict[str, dict[str, str]] = {}

def main() -> None:
    args = parse_args()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_html(INDEX_HTML)
                return
            if parsed.path == "/api/state":
                self._send_json(build_state())
                return
            self.send_error(404)

        def log_message(self, fmt: str, *items: Any) -> None:
            if not args.quiet:
                super().log_message(fmt, *items)

        def _send_html(self, text: str) -> None:
            body = text.encode("utf-8")
            self.send_response(200)
            self.send_header("content-type", "text/html; charset=utf-8")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
            self.send_response(200)
            self.send_header("content-type", "application/json; charset=utf-8")
            self.send_header("cache-control", "no-store")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[stage1-dashboard] url=http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8802)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def build_state() -> dict[str, Any]:
    evals = build_trc_experiments()
    if not evals:
        evals = [build_eval_state(eval_id, spec) for eval_id, spec in EVALS.items()]
    ranking = candidate_ranking(evals)
    guidance = build_guidance(evals, ranking)
    return {
        "format": "trc_experiment_dashboard_v3",
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "evals": evals,
        "runs": [],
        "ranking": ranking,
        "ideas": guidance,
        "guidance": guidance,
        "sources": [
            str(LEDGER_PATH),
            *(str(path) for path in ROUND_EVAL_PATHS),
            str(ROUND13_CONFIG_PATH),
            str(TRC_RUNS_ROOT),
        ],
    }


def build_trc_experiments() -> list[dict[str, Any]]:
    attempts = parse_ledger_attempts(LEDGER_PATH)
    by_artifact = {row["artifact"]: row for row in attempts if row.get("artifact")}
    experiments: dict[str, dict[str, Any]] = {}
    for path in ROUND_EVAL_PATHS:
        for row in parse_round_eval_doc(path):
            exp_id = row["id"]
            checkpoint = row.get("checkpoint") or ""
            run_id = run_id_from_checkpoint(checkpoint) or infer_run_id_from_eval_id(exp_id, by_artifact)
            attempt = find_attempt(run_id, exp_id, attempts)
            item = experiments.setdefault(exp_id, base_doc_experiment(exp_id, run_id, checkpoint, path, attempt))
            if run_id and not item.get("run_id"):
                item["run_id"] = run_id
                item["checkpoint"] = checkpoint or item.get("checkpoint")
                item["checkpoint_path"] = checkpoint or item.get("checkpoint_path")
            item["round_doc"] = str(path)
            if row["kind"] == "tool_memory":
                item["setting"]["key_config"] = row.get("key_config")
                item["tool"] = doc_tool_state(row, path)
                item["memory"] = doc_memory_state(row, path)
                item["quick_gate"] = row.get("decision")
                item["failure_reason"] = failure_reason(row.get("decision"))
            elif row["kind"] == "code":
                item["code"] = doc_code_state(row, path)
                if row.get("status"):
                    item["code_status_text"] = row["status"]
            elif row["kind"] == "training":
                item["training_doc"] = row
                if row.get("run id"):
                    item["train_run_id"] = strip_code(row.get("run id"))
                    if not item.get("run_id"):
                        item["run_id"] = item["train_run_id"]
    for item in experiments.values():
        enrich_doc_experiment(item, attempts)

    for item in build_round13_experiments(attempts):
        experiments[item["id"]] = item
    return sorted(experiments.values(), key=experiment_sort_key)


def base_doc_experiment(
    exp_id: str, run_id: str | None, checkpoint: str, source_path: Path, attempt: dict[str, Any] | None
) -> dict[str, Any]:
    run_id = run_id or ""
    return {
        "id": exp_id,
        "label": exp_id,
        "run_id": run_id,
        "checkpoint": checkpoint,
        "checkpoint_path": checkpoint,
        "train_run": str(TRC_RUNS_ROOT / run_id) if run_id else "",
        "train_command": "",
        "setting": {"family": "TRC", "source": source_path.name},
        "exists": bool(run_id and (TRC_RUNS_ROOT / run_id).exists()),
        "mtime": file_mtime(source_path),
        "tool": pending_status(source_path),
        "memory": pending_status(source_path),
        "code": pending_status(source_path),
        "training": {"status": "pending", "run_root": str(TRC_RUNS_ROOT / run_id) if run_id else ""},
        "quick_gate": None,
        "failure_reason": None,
        "attempt": attempt or {},
        "data_sources": [str(source_path), str(LEDGER_PATH)],
    }


def enrich_doc_experiment(item: dict[str, Any], attempts: list[dict[str, Any]]) -> None:
    run_id = item.get("run_id") or item.get("train_run_id") or ""
    if not run_id:
        run_id = infer_run_id_from_eval_id(str(item.get("id") or ""), {row["artifact"]: row for row in attempts if row.get("artifact")}) or ""
        item["run_id"] = run_id
    run_root = TRC_RUNS_ROOT / run_id if run_id else Path("")
    env = parse_run_env(run_root / "run.env") if run_id else {}
    item["run_env"] = env
    item["train_run"] = str(run_root) if run_id else item.get("train_run", "")
    item["exists"] = run_root.exists() if run_id else False
    item["setting"].update(setting_from_env_and_text(env, str((item.get("setting") or {}).get("key_config") or "")))
    item["setting"]["quick_gate"] = item.get("quick_gate")
    item["setting"]["failure_reason"] = item.get("failure_reason")
    item["setting"]["attempt"] = (item.get("attempt") or {}).get("id")
    item["setting"]["hypothesis"] = (item.get("attempt") or {}).get("hypothesis")
    item["training"] = parse_trc_training(
        {
            "train_run": str(run_root),
            "setting": {"epoch": env.get("EPOCHS") or selected_epoch_from_run(run_root) or infer_epoch_from_run_id(run_id)},
        }
    )
    gates = item["training"].get("gate_means") if isinstance(item.get("training"), dict) else {}
    if isinstance(gates, dict):
        item["setting"]["tool_gate"] = gates.get("tool")
        item["setting"]["memory_gate"] = gates.get("memory")
        item["setting"]["code_gate"] = gates.get("code")
    item["completed_axes"] = sum(1 for axis in (item["tool"], item["memory"], item["code"]) if axis.get("status") == "done")
    item["pending_axes"] = [
        name
        for name, axis in (("Tool", item["tool"]), ("Memory", item["memory"]), ("Code", item["code"]))
        if axis.get("status") != "done"
    ]
    values = []
    if isinstance(item["tool"].get("mean_accuracy"), float):
        values.append(item["tool"]["mean_accuracy"])
    if isinstance(item["memory"].get("mean_f1"), float):
        values.append(item["memory"]["mean_f1"])
    if isinstance(item["code"].get("mean_acc"), float):
        values.append(item["code"]["mean_acc"])
    item["composite_mean"] = statistics.mean(values) if values else None
    item["formal_status"] = formal_status(item["tool"], item["memory"], item["code"])


def parse_round_eval_doc(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    output = []
    for section, kind in (("Tool / Memory", "tool_memory"), ("Code / CURE", "code"), ("Training / Bake", "training")):
        table_rows = parse_markdown_table(section_text(text, section))
        for row in table_rows:
            row = {key.lower(): value for key, value in row.items()}
            if not row.get("id"):
                continue
            output.append(
                {
                    **row,
                    "kind": kind,
                    "id": strip_code(row.get("id")),
                    "checkpoint": strip_code(row.get("checkpoint")),
                    "key_config": strip_code(row.get("key config")),
                    "run id": strip_code(row.get("run id")),
                    "status": strip_code(row.get("status")),
                    "decision": strip_code(row.get("decision")),
                }
            )
    return output


def section_text(text: str, heading: str) -> str:
    match = re.search(rf"^##\s+{re.escape(heading)}\s*$", text, flags=re.MULTILINE)
    if not match:
        return ""
    next_match = re.search(r"^##\s+", text[match.end() :], flags=re.MULTILINE)
    end = match.end() + next_match.start() if next_match else len(text)
    return text[match.end() : end]


def parse_markdown_table(text: str) -> list[dict[str, str]]:
    lines = [line.strip() for line in text.splitlines() if line.strip().startswith("|")]
    if len(lines) < 2:
        return []
    rows = []
    idx = 0
    while idx + 1 < len(lines):
        headers = split_md_row(lines[idx])
        sep = split_md_row(lines[idx + 1])
        if headers and sep and all(set(cell.replace(":", "").replace("-", "")) == set() for cell in sep):
            idx += 2
            while idx < len(lines):
                cells = split_md_row(lines[idx])
                if len(cells) != len(headers):
                    break
                rows.append(dict(zip(headers, cells)))
                idx += 1
            continue
        idx += 1
    return rows


def split_md_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def parse_ledger_attempts(path: Path) -> list[dict[str, Any]]:
    rows = parse_markdown_table(section_text(read_text(path), "Attempt 表"))
    attempts = []
    for row in rows:
        normalized = {key.lower(): strip_code(value) for key, value in row.items()}
        artifact = normalized.get("run_id / artifact") or normalized.get("run_id") or ""
        attempts.append(
            {
                "id": normalized.get("id"),
                "artifact": artifact,
                "layer": normalized.get("layer"),
                "axis": normalized.get("axis"),
                "status": normalized.get("status"),
                "hypothesis": normalized.get("hypothesis"),
                "decision": normalized.get("decision"),
            }
        )
    return attempts


def doc_tool_state(row: dict[str, Any], path: Path) -> dict[str, Any]:
    keys = ("live_parallel", "live_parallel_multiple", "parallel", "parallel_multiple")
    rows = [{"name": key, "accuracy": as_float(row.get(key)), "correct": None, "total": None} for key in keys]
    return {
        "status": "done" if as_float(row.get("tool mean")) is not None else "pending",
        "mean_accuracy": as_float(row.get("tool mean")),
        "live_mean_accuracy": statistics.mean([v for v in (as_float(row.get("live_parallel")), as_float(row.get("live_parallel_multiple"))) if v is not None])
        if any(as_float(row.get(key)) is not None for key in ("live_parallel", "live_parallel_multiple"))
        else None,
        "rows": rows,
        "path": str(path),
        "mtime": file_mtime(path),
    }


def doc_memory_state(row: dict[str, Any], path: Path) -> dict[str, Any]:
    keys = ("eval_50", "eval_100", "qa_32768", "qa_65536")
    rows = [{"name": key, "f1": as_float(row.get(key)), "em": None, "sub_em": None, "total": None} for key in keys]
    return {
        "status": "done" if as_float(row.get("memory mean f1")) is not None else "pending",
        "mean_f1": as_float(row.get("memory mean f1")),
        "mean_em": None,
        "rows": rows,
        "path": str(path),
        "mtime": file_mtime(path),
    }


def doc_code_state(row: dict[str, Any], path: Path) -> dict[str, Any]:
    skipped = str(row.get("run id") or "").lower() == "skipped"
    status_text = str(row.get("status") or "")
    livebench = as_float(row.get("livebench acc"))
    livecode = as_float(row.get("livecodebench acc"))
    rows = [
        {"name": "LiveBench", "status": "done" if livebench is not None else ("skipped" if skipped else "partial"), "acc": livebench, "bon": as_float(row.get("livebench bon")), "tp": None},
        {"name": "LiveCodeBench", "status": "done" if livecode is not None else ("skipped" if skipped else "partial"), "acc": livecode, "bon": as_float(row.get("livecodebench bon")), "tp": None},
    ]
    mean_acc = as_float(row.get("mean acc"))
    return {
        "status": "skipped" if skipped else ("done" if mean_acc is not None else ("partial" if "pending" in status_text.lower() or status_text else "pending")),
        "mean_score": mean_acc,
        "mean_acc": mean_acc,
        "mean_tp": None,
        "mean_bon": as_float(row.get("mean bon")),
        "rows": rows,
        "run_id": strip_code(row.get("run id")),
        "status_text": status_text,
        "path": str(path),
        "mtime": file_mtime(path),
    }


def build_round13_experiments(attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    text = read_text(ROUND13_CONFIG_PATH)
    if not text:
        return []
    variants = {strip_code(row.get("variant")): row for row in parse_markdown_table(section_text(text, "数据"))}
    specs = [
        ("R13A", "rfmem_only", "trc_r13a_evalleak_rfmem_compact600_e8_20260520", "No-R1 task vector, RF/Mem trajectories only"),
        ("R13B", "all_with_r1", "trc_r13b_evalleak_all_r1_compact600_e8_20260520", "all trajectories + scaled R1 task vector"),
    ]
    output = []
    for exp_id, variant, run_id, label in specs:
        attempt = find_attempt(run_id, exp_id, attempts)
        variant_row = variants.get(variant) or {}
        item = base_doc_experiment(exp_id, run_id, "", ROUND13_CONFIG_PATH, attempt)
        item["label"] = exp_id
        item["quick_gate"] = (attempt or {}).get("decision")
        item["failure_reason"] = failure_reason((attempt or {}).get("decision"))
        item["setting"].update(
            {
                "key_config": label,
                "data_bank": variant,
                "eval_leak": "yes",
                "r1": "yes" if variant == "all_with_r1" else "no",
                "bank_rows": strip_code(variant_row.get("rows")),
                "bank_tool_rows": strip_code(variant_row.get("Tool")),
                "bank_memory_rows": strip_code(variant_row.get("Memory")),
                "bank_code_rows": strip_code(variant_row.get("Code")),
                "code_source": strip_code(variant_row.get("Code source")),
                "code_gate_target": strip_code(variant_row.get("Code gate target")),
            }
        )
        enrich_doc_experiment(item, attempts)
        output.append(item)
    return output


def setting_from_env_and_text(env: dict[str, str], key_config: str) -> dict[str, Any]:
    lower = key_config.lower()
    calib = env.get("CALIB") or ""
    return {
        "epoch": env.get("EPOCHS") or infer_epoch_from_run_id(env.get("EXP_ID") or ""),
        "lr": env.get("LR"),
        "data_bank": infer_data_bank(calib, key_config),
        "eval_leak": "yes" if "evalleak" in calib.lower() or "eval-leak" in lower else "no",
        "r1": "yes" if "r1" in (env.get("CONFIG") or "").lower() or "all_r1" in (env.get("EXP_ID") or "") else "no",
        "span": env.get("TASK_RESPONSE_SPAN_MODE") or infer_span_mode(key_config),
        "topk": env.get("TASK_TOPK_TOKENS") or infer_topk(key_config),
        "memory_multiplier": task_assignment_value(env.get("TASK_LOSS_MULTIPLIER"), "memory"),
        "tool_multiplier": task_assignment_value(env.get("TASK_LOSS_MULTIPLIER"), "tool"),
        "code_multiplier": task_assignment_value(env.get("TASK_LOSS_MULTIPLIER"), "code"),
        "loss_multiplier": env.get("TASK_LOSS_MULTIPLIER"),
        "tool_gate": None,
        "memory_gate": None,
        "code_gate": None,
        "max_seq": env.get("MAX_SEQ_LENGTH"),
        "max_response": env.get("MAX_RESPONSE_TOKENS"),
        "calibration": calib,
        "mode": env.get("MODE"),
        "config": env.get("CONFIG"),
    }


def parse_run_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line or line.lstrip().startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip().strip("'\"")
    except OSError:
        pass
    return env


def task_assignment_value(text: str | None, task: str) -> str | None:
    if not text:
        return None
    for chunk in text.split():
        if "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        if key == task:
            return value
    return None


def infer_data_bank(calib: str, key_config: str) -> str:
    text = f"{calib} {key_config}".lower()
    if "round13" in text or "evalleak" in text:
        return "round13 eval-leak code16"
    if "hybrid" in text:
        return "hybrid stable Tool/Memory + Code"
    if "rfonly" in text or "rf-only" in text:
        return "CodeP0 RF-only tag-quota"
    if "tag" in text:
        return "CodeP0 tag-quota"
    if "r8d" in text:
        return "CodeP0 RF-only code-block"
    return Path(calib).parent.name if calib else "from evaluation doc"


def infer_span_mode(key_config: str) -> str:
    lower = key_config.lower()
    if "code-block" in lower or "codeblock" in lower:
        return "code=code-block"
    if "response" in lower:
        return "code=response"
    return ""


def infer_topk(key_config: str) -> str:
    match = re.search(r"topk\s*(\d+)", key_config, flags=re.IGNORECASE)
    return f"code={match.group(1)}" if match else ""


def infer_epoch_from_run_id(run_id: str) -> str:
    match = re.search(r"_e(\d+)_", run_id)
    return match.group(1) if match else ""


def selected_epoch_from_run(run_root: Path) -> str:
    selection = read_json(run_root / "selected.gates.selection.json")
    if isinstance(selection, dict):
        for key in ("selected_epoch", "epoch"):
            if selection.get(key) is not None:
                return str(selection[key])
    return ""


def run_id_from_checkpoint(checkpoint: str | None) -> str | None:
    if not checkpoint:
        return None
    name = Path(strip_code(checkpoint)).name
    return re.sub(r"-selected$", "", name) if name else None


def infer_run_id_from_eval_id(exp_id: str, by_artifact: dict[str, dict[str, Any]]) -> str | None:
    lower_id = exp_id.lower()
    for artifact in by_artifact:
        if lower_id in artifact.lower():
            return artifact
    return None


def find_attempt(run_id: str | None, exp_id: str, attempts: list[dict[str, Any]]) -> dict[str, Any] | None:
    needles = [value.lower() for value in (run_id, exp_id) if value]
    for row in attempts:
        artifact = str(row.get("artifact") or "").lower()
        if artifact and any(needle in artifact for needle in needles):
            return row
    return None


def failure_reason(decision: str | None) -> str | None:
    if not decision:
        return None
    lowered = decision.lower()
    if any(token in lowered for token in ("reject", "below", "failed", "oom", "corrupted", "miss")):
        return decision
    return None


def experiment_sort_key(item: dict[str, Any]) -> tuple[int, str]:
    match = re.match(r"R(\d+)([A-Z]*)", str(item.get("id") or ""))
    if not match:
        return (999, str(item.get("id") or ""))
    return (int(match.group(1)), match.group(2))


def strip_code(value: Any) -> str:
    text = str(value or "").strip()
    if text.startswith("`") and text.endswith("`"):
        return text[1:-1]
    return text.replace("`", "")


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def build_eval_state(eval_id: str, spec: dict[str, Any]) -> dict[str, Any]:
    root = Path(spec["path"])
    logs = root / "logs"
    tool = parse_tool_log(logs / "tool_bfcl.log")
    memory = parse_memory_log(logs / "memory_hotpotqa.log")
    code = parse_code_eval(root, spec.get("checkpoint", ""))
    training = parse_trc_training(spec)
    composite_values = []
    if tool.get("status") == "done" and isinstance(tool.get("mean_accuracy"), (int, float)):
        composite_values.append(tool["mean_accuracy"])
    if memory.get("status") == "done" and isinstance(memory.get("mean_f1"), (int, float)):
        composite_values.append(memory["mean_f1"])
    if code.get("status") == "done" and isinstance(code.get("mean_score"), (int, float)):
        composite_values.append(code["mean_score"])
    return {
        "id": eval_id,
        "label": spec.get("label", eval_id),
        "root": str(root),
        "checkpoint": spec.get("checkpoint"),
        "checkpoint_path": spec.get("checkpoint_path"),
        "train_run": spec.get("train_run"),
        "train_command": spec.get("train_command"),
        "setting": spec.get("setting") or {},
        "exists": root.exists(),
        "mtime": max_mtime([logs / "tool_bfcl.log", logs / "memory_hotpotqa.log", logs / "code_cure.log"]),
        "tool": tool,
        "memory": memory,
        "code": code,
        "training": training,
        "completed_axes": sum(1 for axis in (tool, memory, code) if axis.get("status") == "done"),
        "pending_axes": [name for name, axis in (("Tool", tool), ("Memory", memory), ("Code", code)) if axis.get("status") != "done"],
        "composite_mean": statistics.mean(composite_values) if composite_values else None,
        "formal_status": formal_status(tool, memory, code),
    }


def parse_tool_log(path: Path) -> dict[str, Any]:
    payload = read_json_object_from_log(path)
    if not payload:
        return pending_status(path)
    scores = payload.get("scores") if isinstance(payload.get("scores"), dict) else {}
    rows = []
    for name, item in sorted(scores.items()):
        if isinstance(item, dict):
            rows.append(
                {
                    "name": name,
                    "accuracy": as_float(item.get("accuracy")),
                    "correct": as_int(item.get("correct_count")),
                    "total": as_int(item.get("total_count")),
                }
            )
    accuracies = [row["accuracy"] for row in rows if isinstance(row.get("accuracy"), float)]
    live_accuracies = [
        row["accuracy"]
        for row in rows
        if isinstance(row.get("accuracy"), float) and str(row.get("name", "")).startswith("live_")
    ]
    return {
        "status": "done" if rows else "pending",
        "mean_accuracy": statistics.mean(accuracies) if accuracies else None,
        "live_mean_accuracy": statistics.mean(live_accuracies) if live_accuracies else None,
        "rows": rows,
        "model_name": payload.get("model_name"),
        "path": str(path),
        "mtime": file_mtime(path),
    }


def parse_memory_log(path: Path) -> dict[str, Any]:
    payload = read_json_object_from_log(path)
    if not payload:
        return pending_status(path)
    datasets = payload.get("datasets") if isinstance(payload.get("datasets"), list) else []
    rows = []
    for item in datasets:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "name": item.get("dataset"),
                "f1": as_float(item.get("avg_f1")),
                "em": as_float(item.get("exact_match_rate")),
                "sub_em": as_float(item.get("sub_exact_match_rate")),
                "total": as_int(item.get("total_samples")),
            }
        )
    f1s = [row["f1"] for row in rows if isinstance(row.get("f1"), float)]
    ems = [row["em"] for row in rows if isinstance(row.get("em"), float)]
    return {
        "status": "done" if rows else "pending",
        "mean_f1": statistics.mean(f1s) if f1s else None,
        "mean_em": statistics.mean(ems) if ems else None,
        "rows": rows,
        "model_name": payload.get("model_name"),
        "path": str(path),
        "mtime": file_mtime(path),
    }


def parse_code_eval(root: Path, checkpoint: str) -> dict[str, Any]:
    log_path = root / "logs" / "code_cure.log"
    payload = read_json_object_from_log(log_path)
    progress = parse_process_progress(log_path)
    log_rows = parse_code_metrics_from_log(log_path, progress)
    log_acc = [row.get("acc") for row in log_rows if isinstance(row.get("acc"), float)]
    log_tp = [row.get("tp") for row in log_rows if isinstance(row.get("tp"), float)]
    log_bon = [row.get("bon") for row in log_rows if isinstance(row.get("bon"), float)]
    if payload:
        rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
        return {
            "status": "done",
            "mean_score": as_float(payload.get("mean_score") or payload.get("accuracy")),
            "mean_acc": as_float(payload.get("mean_acc") or payload.get("accuracy")),
            "mean_tp": as_float(payload.get("mean_tp")),
            "mean_bon": as_float(payload.get("mean_bon")),
            "rows": rows or log_rows or code_pending_rows(progress, "done"),
            "progress": progress,
            "path": str(log_path),
            "mtime": file_mtime(log_path),
        }
    if log_rows:
        done = all(row.get("status") == "done" for row in log_rows)
        return {
            "status": "done" if done else "partial",
            "mean_score": statistics.mean(log_acc) if log_acc else None,
            "mean_acc": statistics.mean(log_acc) if log_acc else None,
            "mean_tp": statistics.mean(log_tp) if log_tp else None,
            "mean_bon": statistics.mean(log_bon) if log_bon else None,
            "rows": log_rows,
            "progress": progress,
            "path": str(log_path),
            "mtime": file_mtime(log_path),
        }
    return {
        "status": "partial" if progress.get("current") else "pending",
        "mean_score": None,
        "mean_acc": None,
        "mean_tp": None,
        "mean_bon": None,
        "rows": code_pending_rows(progress, "partial" if progress.get("current") else "pending"),
        "progress": progress,
        "path": str(log_path),
        "mtime": file_mtime(log_path),
    }


def code_pending_rows(progress: dict[str, Any], status: str) -> list[dict[str, Any]]:
    return [
        {
            "name": suite,
            "status": status,
            "items": progress.get("current") or 0,
            "checks": progress.get("total") or 0,
            "score": None,
        }
        for suite in ("LiveBench", "LiveCodeBench")
    ]


def parse_code_metrics_from_log(path: Path, progress: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    starts = list(re.finditer(r"START_DATASET\s+(\S+)", text))
    if not starts:
        return []
    rows = []
    for idx, match in enumerate(starts):
        suite = match.group(1)
        block_end = starts[idx + 1].start() if idx + 1 < len(starts) else len(text)
        block = text[match.start() : block_end]
        acc = regex_float(r"code acc .*?:\s*([0-9.eE+-]+)", block)
        tp = regex_float(r"code accumulate acc .*?:\s*([0-9.eE+-]+)", block)
        bon = regex_float(r"BoN setting\s*\(4,\s*4\):\s*\nacc:\s*([0-9.eE+-]+)", block)
        bon_tp = regex_float(r"BoN setting\s*\(4,\s*4\):\s*\nacc:\s*[0-9.eE+-]+,\s*accumulate acc:\s*([0-9.eE+-]+)", block)
        is_done = re.search(rf"END_DATASET\s+{re.escape(suite)}\b", block) is not None
        row = {
            "name": suite,
            "status": "done" if is_done else "partial",
            "items": progress.get("current") or 0,
            "checks": progress.get("total") or 0,
            "score": acc,
            "acc": acc,
            "tp": tp,
            "bon": bon,
            "bon_tp": bon_tp,
        }
        rows.append(row)
    return rows


def regex_float(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text, flags=re.MULTILINE)
    if not match:
        return None
    return as_float(match.group(1))


def parse_code_artifact(path: Path, suite: str) -> dict[str, Any] | None:
    data = read_json(path)
    if not isinstance(data, list):
        return None
    bools: list[bool] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        for key in ("test_bool_table", "case_bool_table"):
            table = row.get(key)
            if isinstance(table, list):
                collect_bools(table, bools)
    return {
        "name": suite,
        "status": "partial",
        "items": len(data),
        "checks": len(bools),
        "score": (sum(1 for item in bools if item) / len(bools)) if bools else None,
        "path": str(path),
        "mtime": file_mtime(path),
    }


def parse_trc_training(spec: dict[str, Any]) -> dict[str, Any]:
    train_run = str(spec.get("train_run") or "")
    run_root = Path(train_run) if train_run else Path("/nonexistent-trc-run")
    setting = spec.get("setting") if isinstance(spec.get("setting"), dict) else {}
    target_epoch = as_int(setting.get("epoch"))
    if not run_root.exists() or target_epoch <= 0:
        return {"status": "pending", "run_root": str(run_root), "target_epoch": target_epoch}

    epochs = []
    metrics_path = run_root / "trc_metrics.jsonl"
    try:
        with metrics_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if item.get("event") != "epoch":
                    continue
                epoch = as_int(item.get("epoch"))
                if epoch <= target_epoch:
                    epochs.append(compact_trc_epoch(item))
    except OSError:
        pass
    epochs = sorted(epochs, key=lambda item: item.get("epoch") or 0)

    gates_path = run_root / f"epoch_{target_epoch:03d}.gates.json"
    gates_payload = read_json(gates_path)
    if not isinstance(gates_payload, dict):
        gates_payload = read_json(run_root / "trc_gates.json")
    summary = gates_payload.get("epoch_summary") if isinstance(gates_payload, dict) else {}
    if not isinstance(summary, dict):
        summary = epochs[-1].get("raw_summary", {}) if epochs else {}
    gate_values = gates_payload.get("gates") if isinstance(gates_payload, dict) else {}
    if not isinstance(gate_values, dict):
        gate_values = summary.get("gate_values") if isinstance(summary.get("gate_values"), dict) else {}
    gate_means = summary.get("gate_means") if isinstance(summary.get("gate_means"), dict) else gate_means_from_values(gate_values)
    task_loss = summary.get("task_loss") if isinstance(summary.get("task_loss"), dict) else {}
    return {
        "status": "done" if gate_values else "partial",
        "run_root": str(run_root),
        "target_epoch": target_epoch,
        "metrics_path": str(metrics_path),
        "gates_path": str(gates_path),
        "epochs": epochs,
        "gate_means": {expert: as_float(gate_means.get(expert)) for expert in EXPERTS},
        "layer_gate_rows": layer_gate_rows(gate_values),
        "losses": {
            "mean_residual_loss": as_float(summary.get("mean_residual_loss")),
            "mean_total_loss": as_float(summary.get("mean_total_loss")),
            "mean_base_drift_loss": as_float(summary.get("mean_base_drift_loss")),
            "mean_gate_anchor_loss": as_float(summary.get("mean_gate_anchor_loss")),
            "mean_coefficient_floor_loss": as_float(summary.get("mean_coefficient_floor_loss")),
        },
        "task_loss": compact_task_loss(task_loss),
        "mtime": max_mtime([metrics_path, gates_path, run_root / "trc_gates.json"]),
    }


def compact_trc_epoch(item: dict[str, Any]) -> dict[str, Any]:
    task_loss = item.get("task_loss") if isinstance(item.get("task_loss"), dict) else {}
    gate_means = item.get("gate_means") if isinstance(item.get("gate_means"), dict) else {}
    return {
        "epoch": as_int(item.get("epoch")),
        "elapsed_seconds": as_float(item.get("elapsed_seconds")),
        "mean_residual_loss": as_float(item.get("mean_residual_loss")),
        "mean_total_loss": as_float(item.get("mean_total_loss")),
        "mean_base_drift_loss": as_float(item.get("mean_base_drift_loss")),
        "mean_gate_anchor_loss": as_float(item.get("mean_gate_anchor_loss")),
        "mean_coefficient_floor_loss": as_float(item.get("mean_coefficient_floor_loss")),
        "gate_means": {expert: as_float(gate_means.get(expert)) for expert in EXPERTS},
        "task_loss": compact_task_loss(task_loss),
    }


def compact_task_loss(task_loss: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for expert in EXPERTS:
        item = task_loss.get(expert) if isinstance(task_loss.get(expert), dict) else {}
        output[expert] = {
            "rows": as_int(item.get("rows")),
            "span_tokens": as_float(item.get("span_tokens")),
            "residual_loss": as_float(item.get("residual_loss")),
            "total_loss": as_float(item.get("total_loss")),
            "base_drift_loss": as_float(item.get("base_drift_loss")),
        }
    return output


def gate_means_from_values(gate_values: dict[str, Any]) -> dict[str, float | None]:
    rows = layer_gate_rows(gate_values)
    return {
        expert: statistics.mean([row[expert] for row in rows if isinstance(row.get(expert), float)])
        if any(isinstance(row.get(expert), float) for row in rows)
        else None
        for expert in EXPERTS
    }


def layer_gate_rows(gate_values: dict[str, Any]) -> list[dict[str, Any]]:
    layers: dict[int, dict[str, Any]] = {}
    for key, value in gate_values.items():
        match = re.fullmatch(r"layer(\d+)\.(tool|memory|code)", str(key))
        if not match:
            continue
        layer = int(match.group(1))
        expert = match.group(2)
        layers.setdefault(layer, {"layer": layer})[expert] = as_float(value)
    return [layers[layer] for layer in sorted(layers)]


def collect_bools(value: Any, output: list[bool]) -> None:
    if isinstance(value, bool):
        output.append(value)
        return
    if isinstance(value, list):
        for item in value:
            collect_bools(item, output)


def build_run_state(run_id: str, spec: dict[str, str]) -> dict[str, Any]:
    root = Path(spec["path"])
    iterations = []
    for iter_dir in sorted(root.glob("iter_*")):
        if not iter_dir.is_dir():
            continue
        summary_path = iter_dir / "gate_updates.summary.json"
        summary = read_json(summary_path)
        if isinstance(summary, dict):
            iterations.append(compact_iteration(iter_dir.name, summary, summary_path))
    latest = iterations[-1] if iterations else None
    trend = gate_trend(iterations)
    diagnostics = run_diagnostics(latest, trend)
    return {
        "id": run_id,
        "label": spec.get("label", run_id),
        "root": str(root),
        "exists": root.exists(),
        "latest_iteration": latest.get("iteration") if latest else None,
        "last_update": latest.get("created_at") if latest else None,
        "mtime": max((item.get("mtime") or 0.0 for item in iterations), default=0.0),
        "iterations": iterations,
        "latest": latest,
        "trend": trend,
        "diagnostics": diagnostics,
    }


def compact_iteration(name: str, payload: dict[str, Any], path: Path) -> dict[str, Any]:
    epoch = last_epoch(payload)
    final_gates = payload.get("final_gates") if isinstance(payload.get("final_gates"), dict) else {}
    return {
        "iteration": name,
        "created_at": payload.get("created_at"),
        "mtime": file_mtime(path),
        "frontier_rows": as_int(payload.get("kept_frontier_rows")),
        "frontier_task_counts": int_dict(payload.get("frontier_task_counts")),
        "raw_frontier_task_counts": int_dict(payload.get("raw_frontier_task_counts")),
        "opd_rows": as_int(payload.get("opd_distill_rows")),
        "opd_task_counts": int_dict(payload.get("opd_distill_task_counts")),
        "opd_all_success_rows": as_int(payload.get("opd_all_success_rows")),
        "retention_rows": as_int(payload.get("retention_rows")),
        "tool_nullspace_replay_rows": as_int(payload.get("tool_nullspace_replay_rows")),
        "updates": as_int(payload.get("updates") or epoch.get("updates")),
        "gate_grad_nonzero": bool(payload.get("gate_grad_nonzero")),
        "grad_norm_max": as_float(epoch.get("grad_norm_max")),
        "gate_delta_max": as_float(epoch.get("gate_delta_max")),
        "global_gates": {expert: as_float(final_gates.get(f"__global__::{expert}")) for expert in (*EXPERTS, "reasoning")},
        "layer_gate_means": layer_gate_means(final_gates),
        "path": str(path),
    }


def gate_trend(iterations: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {"series": [], "delta_latest": {}, "delta_total": {}}
    for item in iterations:
        row = {"iteration": item["iteration"]}
        row.update(item.get("global_gates") or {})
        output["series"].append(row)
    if len(iterations) >= 2:
        prev = iterations[-2].get("global_gates") or {}
        cur = iterations[-1].get("global_gates") or {}
        output["delta_latest"] = {expert: diff(cur.get(expert), prev.get(expert)) for expert in (*EXPERTS, "reasoning")}
    if len(iterations) >= 1:
        first = iterations[0].get("global_gates") or {}
        cur = iterations[-1].get("global_gates") or {}
        output["delta_total"] = {expert: diff(cur.get(expert), first.get(expert)) for expert in (*EXPERTS, "reasoning")}
    return output


def run_diagnostics(latest: dict[str, Any] | None, trend: dict[str, Any]) -> list[str]:
    if not latest:
        return ["尚未发现 gate_updates.summary.json。"]
    messages = []
    opd_rows = latest.get("opd_rows") or 0
    retention_rows = latest.get("retention_rows") or 0
    grad = latest.get("grad_norm_max") or 0.0
    frontier = latest.get("frontier_task_counts") or {}
    if opd_rows <= 0:
        messages.append("OPD 当前为空，缺少可学习反例信号。")
    elif min((frontier.get(expert, 0) for expert in EXPERTS), default=0) <= 0:
        messages.append("frontier 缺任务，OPD 可能偏单轴。")
    else:
        messages.append(f"OPD 有信号：{opd_rows} rows，frontier 覆盖 tool/memory/code。")
    if retention_rows <= 0:
        messages.append("retention rows 为 0，遗忘约束缺失。")
    elif retention_rows < 8:
        messages.append(f"retention rows 偏少：{retention_rows}。")
    if grad < 1e-3:
        messages.append(f"grad norm 很小：{grad:.2e}，gate 可能基本不动。")
    deltas = trend.get("delta_latest") or {}
    movers = [f"{expert} {deltas[expert]:+.4f}" for expert in EXPERTS if isinstance(deltas.get(expert), float) and abs(deltas[expert]) >= 1e-4]
    messages.append("最新 gate 变化：" + (", ".join(movers) if movers else "三轴近似停滞。"))
    return messages


def candidate_ranking(evals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in evals:
        tool = item["tool"].get("mean_accuracy") if item["tool"].get("status") == "done" else None
        tool_live = item["tool"].get("live_mean_accuracy") if item["tool"].get("status") == "done" else None
        memory_em = item["memory"].get("mean_em") if item["memory"].get("status") == "done" else None
        memory = item["memory"].get("mean_f1") if item["memory"].get("status") == "done" else None
        code = item["code"].get("mean_acc") if item["code"].get("status") == "done" else None
        code_tp = item["code"].get("mean_tp") if item["code"].get("status") == "done" else None
        code_bon = item["code"].get("mean_bon") if item["code"].get("status") == "done" else None
        completed = [value for value in (tool, memory, code) if isinstance(value, (int, float))]
        rows.append(
            {
                "id": item["id"],
                "label": item["label"],
                "tool": tool,
                "tool_live": tool_live,
                "memory_em": memory_em,
                "memory": memory,
                "code": code,
                "code_tp": code_tp,
                "code_bon": code_bon,
                "completed_axes": item["completed_axes"],
                "pending_axes": item["pending_axes"],
                "score": statistics.mean(completed) if completed else None,
                "formal_status": item.get("formal_status"),
            }
        )
    return sorted(rows, key=lambda row: (row["score"] is not None, row["score"] or -1.0, row["completed_axes"]), reverse=True)


def formal_status(tool: dict[str, Any], memory: dict[str, Any], code: dict[str, Any]) -> str:
    pending = [name for name, axis in (("Tool", tool), ("Memory", memory), ("Code", code)) if axis.get("status") != "done"]
    done = [name for name, axis in (("Tool", tool), ("Memory", memory), ("Code", code)) if axis.get("status") == "done"]
    if not pending:
        return "done"
    return f"pending {','.join(pending)}; done {','.join(done) or 'none'}"


def build_guidance(evals: list[dict[str, Any]], ranking: list[dict[str, Any]]) -> list[str]:
    ideas = [
        "本页自动合并 20260520 experiment ledger、Round10-12 evaluation docs、Round13 hidden-state config，以及 /tmp/shared-storage/OnPolicy/runs/trc 下的 run.env / trc_metrics / gates。",
        "排序按已完成轴的 Tool mean、Memory F1、Code Acc 均值给出；quick-gate 未过的 Code 会显示 skipped，不应按三轴完成候选解读。",
        "变量列优先使用 run.env：data bank、eval-leak、R1、span mode、topK、Tool/Memory/Code multiplier 和目标 epoch gate mean。",
        "Round13 是 eval-leak hidden-state diagnostic，不进入论文主结果；当前只展示训练/quick-gate准备状态，正式 Tool/Memory/Code 需后续文档或日志补齐。",
    ]
    if ranking:
        best = ranking[0]
        ideas.append(f"当前已完成指标均值领先：{best['label']}，pending={','.join(best['pending_axes']) or '无'}。")
    for item in evals:
        tool = item["tool"].get("mean_accuracy")
        if isinstance(tool, float) and tool >= 0.80 and item["pending_axes"]:
            ideas.append(f"{item['label']} Tool 强，但 {','.join(item['pending_axes'])} pending，适合优先补齐确认是否可作为候选。")
    if not any(not item.get("pending_axes") for item in evals):
        ideas.append("当前还没有三轴全部完成的候选；不要只凭 Tool 或 LiveBench 单项定最终模型。")
    return ideas[:8]


def read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def read_json_object_from_log(path: Path) -> dict[str, Any] | None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            payload, end = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and not text[match.start() + end :].strip():
            return payload
    return None


def parse_process_progress(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    matches = re.findall(r"process\s+(\d+)\s*/\s*(\d+)", text)
    if not matches:
        return {}
    current, total = matches[-1]
    current_i = int(current)
    total_i = int(total)
    return {"current": current_i, "total": total_i, "ratio": current_i / total_i if total_i else None}


def pending_status(path: Path) -> dict[str, Any]:
    return {"status": "pending", "path": str(path), "mtime": file_mtime(path)}


def last_epoch(payload: dict[str, Any]) -> dict[str, Any]:
    epochs = payload.get("epoch_summaries")
    if isinstance(epochs, list) and epochs and isinstance(epochs[-1], dict):
        return epochs[-1]
    return {}


def layer_gate_means(gates: dict[str, Any]) -> dict[str, float | None]:
    output: dict[str, float | None] = {}
    for expert in (*EXPERTS, "reasoning"):
        values = [
            as_float(value)
            for key, value in gates.items()
            if re.match(r"^layer\d+\." + re.escape(expert) + r"$", str(key))
        ]
        values = [value for value in values if isinstance(value, float)]
        output[expert] = statistics.mean(values) if values else None
    return output


def int_dict(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {str(key): as_int(val) for key, val in sorted(value.items())}


def as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value: Any) -> int:
    try:
        if value is None:
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


def diff(current: Any, previous: Any) -> float | None:
    cur = as_float(current)
    prev = as_float(previous)
    if cur is None or prev is None:
        return None
    return cur - prev


def file_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def max_mtime(paths: list[Path]) -> float:
    return max((file_mtime(path) for path in paths), default=0.0)


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ExpertGym TRC Experiment Dashboard</title>
  <style>
    :root { color-scheme: light; --bg:#f5f7fa; --panel:#ffffff; --ink:#17202a; --muted:#667085; --line:#d8dee9; --accent:#126b63; --warn:#9a5b00; --bad:#a82020; --good:#146c2e; }
    * { box-sizing: border-box; }
    body { margin:0; font:13px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:var(--bg); color:var(--ink); }
    header { padding:14px 20px; background:#112126; color:#f6fbfb; display:flex; justify-content:space-between; gap:16px; align-items:end; }
    h1 { font-size:20px; margin:0 0 2px; font-weight:700; letter-spacing:0; }
    h2 { font-size:15px; margin:0 0 10px; }
    main { padding:14px 20px 26px; display:grid; gap:14px; }
    section { background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:12px; box-shadow:0 1px 2px rgba(0,0,0,.03); }
    .grid { display:grid; grid-template-columns:1.05fr .95fr; gap:14px; align-items:start; }
    table { width:100%; border-collapse:collapse; table-layout:fixed; }
    th, td { border-bottom:1px solid #e7ebf0; padding:6px 7px; text-align:left; vertical-align:top; overflow:hidden; text-overflow:ellipsis; }
    th { color:#475467; font-weight:650; background:#f8fafc; }
    tr:last-child td { border-bottom:0; }
    .num { text-align:right; font-variant-numeric:tabular-nums; }
    .muted { color:var(--muted); }
    .pill { display:inline-block; border:1px solid var(--line); border-radius:999px; padding:1px 7px; font-size:12px; background:#fff; white-space:nowrap; }
    .good { color:var(--good); font-weight:650; }
    .warn { color:var(--warn); font-weight:650; }
    .bad { color:var(--bad); font-weight:650; }
    .cards { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; }
    .run { border:1px solid var(--line); border-radius:6px; padding:10px; background:#fcfdff; }
    .kv { display:grid; grid-template-columns:145px 1fr; gap:2px 8px; margin:6px 0; }
    .spark { height:42px; width:100%; border:1px solid #e2e8f0; border-radius:4px; background:#fff; }
    ul { margin:6px 0 0 18px; padding:0; }
    li { margin:3px 0; }
    .subgrid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px; }
    .axis { border:1px solid var(--line); border-radius:6px; overflow:hidden; }
    .axis h3 { margin:0; padding:7px 8px; background:#f7f9fb; font-size:13px; }
    .small { font-size:12px; }
    @media (max-width: 980px) { .grid, .subgrid, .cards { grid-template-columns:1fr; } header { display:block; } }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>ExpertGym / TRC 实验决策看板</h1>
      <div class="muted" style="color:#c7d7da">关键变量、loss 曲线、quick-gate、正式 Tool/Memory/Code 结果</div>
    </div>
    <div id="updated" class="small"></div>
  </header>
  <main>
    <div class="grid">
      <section>
        <h2>当前 best candidate ranking</h2>
        <div id="ranking"></div>
      </section>
      <section>
        <h2>阅读引导</h2>
        <ul id="ideas"></ul>
      </section>
    </div>
    <section>
        <h2>实验变量 / quick-gate 决策</h2>
      <div id="settings"></div>
    </section>
    <section>
      <h2>TRC gate 系数 / 训练动态</h2>
      <div id="training" class="cards"></div>
    </section>
    <section>
      <h2>评测对照</h2>
      <div id="evals"></div>
    </section>
  </main>
<script>
const fmtPct = v => (typeof v === 'number' && isFinite(v)) ? (v * 100).toFixed(2) + '%' : '<span class="warn">pending</span>';
const fmtNum = v => (typeof v === 'number' && isFinite(v)) ? v.toFixed(4) : '<span class="warn">pending</span>';
const fmtInt = v => (typeof v === 'number') ? String(v) : '0';
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function age(ts) { return ts ? new Date(ts * 1000).toLocaleString() : 'missing'; }
function table(headers, rows) {
  return `<table><thead><tr>${headers.map(h=>`<th${h.cls?` class="${h.cls}"`:''}>${h.t}</th>`).join('')}</tr></thead><tbody>${rows.join('')}</tbody></table>`;
}
function renderRanking(rows) {
  document.getElementById('ranking').innerHTML = table(
    [{t:'#'},{t:'candidate'},{t:'Tool mean',cls:'num'},{t:'Tool live',cls:'num'},{t:'Memory F1',cls:'num'},{t:'Code Acc',cls:'num'},{t:'Code BoN',cls:'num'},{t:'状态'}],
    rows.map((r,i)=>`<tr><td>${i+1}</td><td><b>${esc(r.label)}</b></td><td class="num">${fmtPct(r.tool)}</td><td class="num">${fmtPct(r.tool_live)}</td><td class="num">${fmtPct(r.memory)}</td><td class="num">${fmtPct(r.code)}</td><td class="num">${fmtPct(r.code_bon)}</td><td>${r.completed_axes}/3 <span class="pill">${esc((r.pending_axes||[]).join(', ') || 'done')}</span></td></tr>`)
  );
}
function spark(series) {
  const keys = ['tool','memory','code'];
  const colors = {tool:'#126b63', memory:'#8a4a00', code:'#2358a7'};
  const w=420,h=42,p=5;
  const vals = series.flatMap(r => keys.map(k => r[k]).filter(v => typeof v === 'number'));
  const lo = Math.min(...vals, 0), hi = Math.max(...vals, 1), span = Math.max(hi-lo, 1e-6);
  const lines = keys.map(k => {
    const pts = series.map((r,i) => {
      const x = p + (series.length <= 1 ? 0 : i*(w-2*p)/(series.length-1));
      const y = h-p - ((r[k] ?? lo)-lo)*(h-2*p)/span;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(' ');
    return `<polyline points="${pts}" fill="none" stroke="${colors[k]}" stroke-width="2"/>`;
  }).join('');
  return `<svg class="spark" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">${lines}</svg>`;
}
function renderRuns(runs) {
  document.getElementById('runs').innerHTML = runs.map(run => {
    const l = run.latest || {};
    const g = l.global_gates || {};
    const lg = l.layer_gate_means || {};
    const iterRows = (run.iterations || []).map(it => {
      const ig = it.global_gates || {};
      return `<tr><td>${esc(it.iteration)}</td><td class="num">${fmtInt(it.opd_rows)}</td><td class="num">${fmtInt(it.retention_rows)}</td><td class="num">${fmtInt(it.frontier_rows)}</td><td class="num">${fmtNum(it.grad_norm_max)}</td><td class="num">${fmtNum(ig.tool)}</td><td class="num">${fmtNum(ig.memory)}</td><td class="num">${fmtNum(ig.code)}</td></tr>`;
    }).join('');
    return `<div class="run">
      <h2>${esc(run.id)} <span class="muted">${esc(run.label)}</span></h2>
      ${spark((run.trend||{}).series || [])}
      <div class="kv">
        <div class="muted">latest / update</div><div>${esc(run.latest_iteration)} / ${esc(run.last_update || 'missing')}</div>
        <div class="muted">OPD / retention</div><div><b>${fmtInt(l.opd_rows)}</b> / <b>${fmtInt(l.retention_rows)}</b> rows</div>
        <div class="muted">frontier counts</div><div>${esc(JSON.stringify(l.frontier_task_counts || {}))}</div>
        <div class="muted">OPD counts</div><div>${esc(JSON.stringify(l.opd_task_counts || {}))}</div>
        <div class="muted">grad / delta</div><div>${fmtNum(l.grad_norm_max)} / ${fmtNum(l.gate_delta_max)}</div>
        <div class="muted">global gate</div><div>tool ${fmtNum(g.tool)} · memory ${fmtNum(g.memory)} · code ${fmtNum(g.code)}</div>
        <div class="muted">layer mean</div><div>tool ${fmtNum(lg.tool)} · memory ${fmtNum(lg.memory)} · code ${fmtNum(lg.code)}</div>
      </div>
      <ul>${(run.diagnostics||[]).map(x=>`<li>${esc(x)}</li>`).join('')}</ul>
      <div style="margin-top:8px; max-height:260px; overflow:auto">
        ${table([{t:'iter'},{t:'OPD',cls:'num'},{t:'ret',cls:'num'},{t:'frontier',cls:'num'},{t:'grad',cls:'num'},{t:'tool',cls:'num'},{t:'memory',cls:'num'},{t:'code',cls:'num'}], iterRows ? [iterRows] : ['<tr><td colspan="8" class="warn">pending</td></tr>'])}
      </div>
    </div>`;
  }).join('');
}
function axisBlock(title, headers, rows) {
  return `<div class="axis"><h3>${title}</h3>${table(headers, rows)}</div>`;
}
function codeState(code) {
  const rows = code.rows || [];
  if (code.status === 'done') return 'Code 已完成';
  if (!rows.length || rows.every(r => r.status === 'pending')) return 'Code 未进入评测';
  const parts = rows.map(r => `${r.name}:${r.status || 'pending'}`);
  return `Code ${code.status || 'pending'} · ${parts.join(' · ')}`;
}
function renderSettings(evals) {
  const rows = evals.map(ev => {
    const s = ev.setting || {};
    const identity = [
      `<b>${esc(ev.id || ev.label)}</b>`,
      esc(ev.run_id || ev.checkpoint || ''),
      ev.attempt?.id ? `<span class="pill">${esc(ev.attempt.id)}</span>` : ''
    ].join('<br>');
    const data = [
      `bank=${s.data_bank || s.calibration || 'NA'}`,
      `eval-leak=${s.eval_leak || 'NA'}`,
      `R1=${s.r1 || 'NA'}`,
      s.bank_rows ? `rows=${s.bank_rows} T/M/C=${s.bank_tool_rows || '?'} / ${s.bank_memory_rows || '?'} / ${s.bank_code_rows || '?'}` : '',
      s.code_source ? `code=${s.code_source}` : ''
    ].filter(Boolean).join('<br>');
    const knobs = [
      `span=${s.span || 'NA'}`,
      `topK=${s.topk || 'NA'}`,
      `memory x${s.memory_multiplier || 'NA'}`,
      `tool x${s.tool_multiplier || 'NA'}`,
      `code x${s.code_multiplier || 'NA'}`,
      `epoch=${s.epoch || 'NA'} lr=${s.lr || 'NA'}`
    ].join('<br>');
    const gates = [
      `tool=${fmtNum(s.tool_gate)}`,
      `memory=${fmtNum(s.memory_gate)}`,
      `code=${fmtNum(s.code_gate)}`
    ].join('<br>');
    const decision = [
      ev.quick_gate || s.quick_gate || 'pending',
      s.failure_reason ? `<span class="bad">${esc(s.failure_reason)}</span>` : '',
      s.hypothesis ? `<span class="muted">${esc(s.hypothesis)}</span>` : ''
    ].filter(Boolean).join('<br>');
    return `<tr>
      <td>${identity}</td>
      <td>${data}</td>
      <td>${knobs}</td>
      <td>${gates}</td>
      <td>${decision}<br><span class="pill">${esc(ev.formal_status || '')}</span><br>${esc(codeState(ev.code || {}))}</td>
      <td class="small">${esc(ev.checkpoint_path || '')}<br>${esc(ev.train_run || '')}<br>${esc(ev.root || '')}</td>
    </tr>`;
  });
  document.getElementById('settings').innerHTML = table(
    [{t:'experiment / run_id'},{t:'data bank / leak / R1'},{t:'span / topK / multipliers'},{t:'selected gates'},{t:'quick-gate / failure'},{t:'可追溯路径'}],
    rows
  );
}
function renderTraining(evals) {
  document.getElementById('training').innerHTML = evals.map(ev => {
    const t = ev.training || {};
    const gm = t.gate_means || {};
    const losses = t.losses || {};
    const epochs = t.epochs || [];
    const gateSeries = epochs.map(e => {
      const g = e.gate_means || {};
      return {iteration:`e${e.epoch}`, tool:g.tool, memory:g.memory, code:g.code};
    });
    const epochRows = epochs.map(e => {
      const g = e.gate_means || {};
      const tl = e.task_loss || {};
      return `<tr>
        <td>${fmtInt(e.epoch)}</td>
        <td class="num">${fmtNum(e.mean_residual_loss)}</td>
        <td class="num">${fmtNum(e.mean_total_loss)}</td>
        <td class="num">${fmtNum((tl.tool||{}).residual_loss)}</td>
        <td class="num">${fmtNum((tl.memory||{}).residual_loss)}</td>
        <td class="num">${fmtNum((tl.code||{}).residual_loss)}</td>
        <td class="num">${fmtNum(g.tool)}</td>
        <td class="num">${fmtNum(g.memory)}</td>
        <td class="num">${fmtNum(g.code)}</td>
      </tr>`;
    });
    const layerRows = (t.layer_gate_rows || []).map(r => `<tr>
      <td>${fmtInt(r.layer)}</td>
      <td class="num">${fmtNum(r.tool)}</td>
      <td class="num">${fmtNum(r.memory)}</td>
      <td class="num">${fmtNum(r.code)}</td>
    </tr>`);
    const taskRows = ['tool','memory','code'].map(k => {
      const row = (t.task_loss || {})[k] || {};
      return `<tr><td>${k}</td><td class="num">${fmtInt(row.rows)}</td><td class="num">${fmtNum(row.span_tokens)}</td><td class="num">${fmtNum(row.residual_loss)}</td><td class="num">${fmtNum(row.total_loss)}</td><td class="num">${fmtNum(row.base_drift_loss)}</td></tr>`;
    });
    return `<div class="run">
      <h2>${esc(ev.id || ev.label)} <span class="muted">${esc(ev.run_id || '')}</span> <span class="pill">target epoch ${fmtInt(t.target_epoch)}</span></h2>
      ${gateSeries.length ? spark(gateSeries) : '<div class="warn">training curve pending</div>'}
      <div class="kv">
        <div class="muted">gate mean</div><div>tool <b>${fmtNum(gm.tool)}</b> · memory <b>${fmtNum(gm.memory)}</b> · code <b>${fmtNum(gm.code)}</b></div>
        <div class="muted">loss</div><div>residual ${fmtNum(losses.mean_residual_loss)} · total ${fmtNum(losses.mean_total_loss)} · base ${fmtNum(losses.mean_base_drift_loss)}</div>
        <div class="muted">regularizer</div><div>gate anchor ${fmtNum(losses.mean_gate_anchor_loss)} · coeff floor ${fmtNum(losses.mean_coefficient_floor_loss)}</div>
        <div class="muted">files</div><div class="small">${esc(t.gates_path || '')}<br>${esc(t.metrics_path || '')}</div>
      </div>
      <h3 class="small">目标 epoch task loss</h3>
      ${table([{t:'task'},{t:'rows',cls:'num'},{t:'span tok',cls:'num'},{t:'residual',cls:'num'},{t:'total',cls:'num'},{t:'base drift',cls:'num'}], taskRows)}
      <h3 class="small">epoch 动态</h3>
      <div style="max-height:230px; overflow:auto">
        ${table([{t:'epoch'},{t:'resid',cls:'num'},{t:'total',cls:'num'},{t:'tool loss',cls:'num'},{t:'mem loss',cls:'num'},{t:'code loss',cls:'num'},{t:'tool gate',cls:'num'},{t:'mem gate',cls:'num'},{t:'code gate',cls:'num'}], epochRows.length ? epochRows : ['<tr><td colspan="9" class="warn">pending</td></tr>'])}
      </div>
      <h3 class="small">layer gate coefficients</h3>
      <div style="max-height:300px; overflow:auto">
        ${table([{t:'layer'},{t:'tool',cls:'num'},{t:'memory',cls:'num'},{t:'code',cls:'num'}], layerRows.length ? layerRows : ['<tr><td colspan="4" class="warn">pending</td></tr>'])}
      </div>
    </div>`;
  }).join('');
}
function renderEvals(evals) {
  document.getElementById('evals').innerHTML = evals.map(ev => {
    const toolRows = (ev.tool.rows||[]).map(r=>`<tr><td>${esc(r.name)}</td><td class="num">${fmtPct(r.accuracy)}</td><td class="num">${fmtInt(r.correct)}/${fmtInt(r.total)}</td></tr>`);
    const memRows = (ev.memory.rows||[]).map(r=>`<tr><td>${esc(r.name)}</td><td class="num">${fmtPct(r.f1)}</td><td class="num">${fmtPct(r.em)}</td><td class="num">${fmtPct(r.sub_em)}</td></tr>`);
    const codeRows = (ev.code.rows||[]).map(r=>`<tr><td>${esc(r.name)}</td><td>${esc(r.status || 'pending')}</td><td class="num">${fmtPct(r.acc ?? r.score)}</td><td class="num">${fmtPct(r.tp)}</td><td class="num">${fmtPct(r.bon)}</td></tr>`);
    const p = ev.code.progress || {};
    return `<div style="margin-bottom:12px">
      <h2>${esc(ev.label)} <span class="pill">${esc((ev.pending_axes||[]).join(', ') || 'done')}</span> <span class="muted small">mtime ${age(ev.mtime)}</span></h2>
      <div class="subgrid">
        ${axisBlock('Tool 子类', [{t:'category'},{t:'acc',cls:'num'},{t:'correct',cls:'num'}], toolRows.length ? toolRows : ['<tr><td colspan="3" class="warn">pending</td></tr>'])}
        ${axisBlock('Memory 子集', [{t:'dataset'},{t:'F1',cls:'num'},{t:'EM',cls:'num'},{t:'Sub EM',cls:'num'}], memRows.length ? memRows : ['<tr><td colspan="4" class="warn">pending</td></tr>'])}
        ${axisBlock(`Code CURE <span class="muted">process ${p.current||0}/${p.total||0}</span>`, [{t:'suite'},{t:'status'},{t:'Acc',cls:'num'},{t:'TP',cls:'num'},{t:'BoN',cls:'num'}], codeRows.length ? codeRows : ['<tr><td colspan="5" class="warn">pending</td></tr>'])}
      </div>
    </div>`;
  }).join('');
}
async function refresh() {
  const state = await fetch('/api/state', {cache:'no-store'}).then(r=>r.json());
  document.getElementById('updated').textContent = `更新：${state.updated_at}`;
  renderRanking(state.ranking || []);
  document.getElementById('ideas').innerHTML = (state.ideas||[]).map(x=>`<li>${esc(x)}</li>`).join('');
  renderSettings(state.evals || []);
  renderTraining(state.evals || []);
  renderEvals(state.evals || []);
}
refresh();
setInterval(refresh, 30000);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
