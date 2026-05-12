#!/usr/bin/env python3
"""Read-only OP-VEC run monitor with a small browser UI."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from statistics import mean, pstdev
from typing import Any
from urllib.parse import parse_qs, urlparse

EXPERTS = ("tool", "memory", "code")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise SystemExit(f"run dir not found: {run_dir}")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_html(INDEX_HTML)
                return
            if parsed.path == "/api/state":
                query = parse_qs(parsed.query)
                max_prompt_rows = int(query.get("max_prompt_rows", [args.max_prompt_rows])[0])
                state = build_state(run_dir, init_value=args.init_value, max_prompt_rows=max_prompt_rows)
                self._send_json(state)
                return
            self.send_error(404)

        def log_message(self, fmt: str, *items: Any) -> None:
            if args.quiet:
                return
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
    print(f"[monitor] run_dir={run_dir}", flush=True)
    print(f"[monitor] url=http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def build_state(run_dir: Path, *, init_value: float, max_prompt_rows: int) -> dict[str, Any]:
    iterations = []
    for iter_dir in sorted(run_dir.glob("iter_*")):
        if not iter_dir.is_dir():
            continue
        iterations.append(_iteration_state(iter_dir, init_value=init_value, max_prompt_rows=max_prompt_rows))
    return {
        "format": "opvec_monitor_state_v1",
        "run_dir": str(run_dir),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "iterations": iterations,
        "reward_series": _reward_series(iterations),
        "coefficient_series": _coefficient_series(iterations),
        "alerts": _alerts(iterations),
    }


def _iteration_state(iter_dir: Path, *, init_value: float, max_prompt_rows: int) -> dict[str, Any]:
    rollout_path = iter_dir / "rollouts.jsonl"
    rollout_summary_path = iter_dir / "rollouts.reward_summary.json"
    update_summary_path = iter_dir / "gate_updates.summary.json"
    gates_path = iter_dir / "gate_updates.gates.json"
    rows = _read_jsonl(rollout_path) if rollout_path.exists() else []
    prompt_rows = [_prompt_row(row) for row in rows[:max_prompt_rows]]
    task_stats = _task_stats(rows)
    rollout_summary = _read_json(rollout_summary_path) if rollout_summary_path.exists() else None
    update_summary = _read_json(update_summary_path) if update_summary_path.exists() else None
    gates = _load_gates(gates_path) if gates_path.exists() else {}
    status = _status(rollout_path=rollout_path, update_summary_path=update_summary_path, gates_path=gates_path)
    return {
        "iteration": iter_dir.name,
        "status": status,
        "paths": {
            "rollouts": str(rollout_path) if rollout_path.exists() else None,
            "rollout_summary": str(rollout_summary_path) if rollout_summary_path.exists() else None,
            "update_summary": str(update_summary_path) if update_summary_path.exists() else None,
            "gates": str(gates_path) if gates_path.exists() else None,
        },
        "mtime": _max_mtime([rollout_path, rollout_summary_path, update_summary_path, gates_path]),
        "rollout_rows": len(rows),
        "task_stats": task_stats,
        "prompt_rows": prompt_rows,
        "rollout_summary": rollout_summary,
        "update": _compact_update(update_summary),
        "gate_stats": _gate_stats(gates, init_value=init_value) if gates else None,
    }


def _status(*, rollout_path: Path, update_summary_path: Path, gates_path: Path) -> str:
    if gates_path.exists():
        return "updated"
    if update_summary_path.exists():
        return "update_summary_only"
    if rollout_path.exists():
        return "rollout_or_update_running"
    return "pending"


def _prompt_row(row: dict[str, Any]) -> dict[str, Any]:
    samples = row.get("samples") if isinstance(row.get("samples"), list) else []
    rewards = [_float(sample.get("reward")) for sample in samples if isinstance(sample, dict)]
    successes = [bool(sample.get("success")) or _float(sample.get("reward")) > 0.0 for sample in samples if isinstance(sample, dict)]
    return {
        "prompt_id": row.get("prompt_id"),
        "task": row.get("task"),
        "keep_for_policy_loss": bool(row.get("keep_for_policy_loss")),
        "skip_reason": row.get("skip_reason"),
        "mean_reward": mean(rewards) if rewards else 0.0,
        "std_reward": pstdev(rewards) if len(rewards) > 1 else 0.0,
        "success_count": sum(1 for item in successes if item),
        "samples": [
            {
                "sample_id": sample.get("sample_id"),
                "reward": _float(sample.get("reward")),
                "success": bool(sample.get("success")) or _float(sample.get("reward")) > 0.0,
                "length": _float(sample.get("length")),
                "text_preview": str(sample.get("text") or "")[:180].replace("\n", " "),
            }
            for sample in samples
            if isinstance(sample, dict)
        ],
    }


def _task_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    acc: dict[str, dict[str, Any]] = defaultdict(lambda: {"rows": 0, "samples": 0, "rewards": [], "successes": 0, "kept": 0})
    for row in rows:
        task = str(row.get("task") or "unknown")
        acc[task]["rows"] += 1
        acc[task]["kept"] += int(bool(row.get("keep_for_policy_loss")))
        samples = row.get("samples") if isinstance(row.get("samples"), list) else []
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            reward = _float(sample.get("reward"))
            acc[task]["samples"] += 1
            acc[task]["rewards"].append(reward)
            acc[task]["successes"] += int(bool(sample.get("success")) or reward > 0.0)
    output = {}
    for task, item in sorted(acc.items()):
        rewards = item["rewards"]
        samples = int(item["samples"])
        output[task] = {
            "rows": int(item["rows"]),
            "samples": samples,
            "kept_frontier_rows": int(item["kept"]),
            "mean_reward": mean(rewards) if rewards else 0.0,
            "std_reward": pstdev(rewards) if len(rewards) > 1 else 0.0,
            "success_rate": float(item["successes"]) / samples if samples else 0.0,
            "min_reward": min(rewards) if rewards else 0.0,
            "max_reward": max(rewards) if rewards else 0.0,
        }
    return output


def _compact_update(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not payload:
        return None
    return {
        "kept_frontier_rows": payload.get("kept_frontier_rows"),
        "frontier_task_counts": payload.get("frontier_task_counts"),
        "raw_frontier_task_counts": payload.get("raw_frontier_task_counts"),
        "updates": payload.get("updates"),
        "gate_grad_nonzero": payload.get("gate_grad_nonzero"),
        "parameter_coefficients": payload.get("parameter_coefficients"),
        "stopped_early_at_step": payload.get("stopped_early_at_step"),
        "epoch_summaries": [
            {
                "step": item.get("step"),
                "updates": item.get("updates"),
                "grad_norm_max": item.get("grad_norm_max"),
                "gate_delta_max": item.get("gate_delta_max"),
            }
            for item in payload.get("epoch_summaries", [])
            if isinstance(item, dict)
        ],
    }


def _gate_stats(gates: dict[str, float], *, init_value: float) -> dict[str, Any]:
    coeffs = _effective_coefficients(gates)
    values = [value for _, _, value in coeffs]
    by_expert: dict[str, list[float]] = defaultdict(list)
    for _, expert, value in coeffs:
        by_expert[expert].append(value)
    abs_deltas = [abs(value - init_value) for value in values]
    top_changed = sorted(
        [
            {"name": name, "expert": expert, "value": value, "delta": value - init_value}
            for name, expert, value in coeffs
        ],
        key=lambda item: abs(float(item["delta"])),
        reverse=True,
    )[:24]
    return {
        "num_coefficients": len(values),
        "mean": mean(values) if values else 0.0,
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
        "mean_abs_delta_from_init": mean(abs_deltas) if abs_deltas else 0.0,
        "max_abs_delta_from_init": max(abs_deltas) if abs_deltas else 0.0,
        "expert_means": {expert: mean(items) for expert, items in sorted(by_expert.items()) if items},
        "expert_delta": {expert: mean(items) - init_value for expert, items in sorted(by_expert.items()) if items},
        "top_changed": top_changed,
    }


def _effective_coefficients(gates: dict[str, float]) -> list[tuple[str, str, float]]:
    param_keys = [key for key in gates if "::" in key and not key.startswith("__global__::")]
    if param_keys:
        output = []
        for key in sorted(param_keys):
            name, expert = key.rsplit("::", 1)
            if expert in EXPERTS:
                output.append((name, expert, float(gates[key])))
        return output
    band_names = sorted({key.split(".", 1)[0] for key in gates if "." in key and "::" not in key})
    if band_names:
        output = []
        for band in band_names:
            common = float(gates.get(f"{band}.common", gates.get("common", 0.5)))
            residuals = [float(gates.get(f"{band}.{expert}_residual", gates.get(f"{expert}_residual", 0.0))) for expert in EXPERTS]
            residual_mean = sum(residuals) / len(residuals)
            for expert, residual in zip(EXPERTS, residuals):
                output.append((band, expert, common + residual - residual_mean))
        return output
    common = float(gates.get("common", 0.5))
    residuals = [float(gates.get(f"{expert}_residual", 0.0)) for expert in EXPERTS]
    residual_mean = sum(residuals) / len(residuals)
    return [("global", expert, common + residual - residual_mean) for expert, residual in zip(EXPERTS, residuals)]


def _reward_series(iterations: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    series: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in iterations:
        for task, stats in (item.get("task_stats") or {}).items():
            series[task].append(
                {
                    "iteration": item["iteration"],
                    "mean_reward": stats.get("mean_reward", 0.0),
                    "success_rate": stats.get("success_rate", 0.0),
                    "kept_frontier_rows": stats.get("kept_frontier_rows", 0),
                }
            )
    return dict(sorted(series.items()))


def _coefficient_series(iterations: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    series: dict[str, list[dict[str, Any]]] = {expert: [] for expert in EXPERTS}
    for item in iterations:
        gate_stats = item.get("gate_stats") or {}
        means = gate_stats.get("expert_means") or {}
        deltas = gate_stats.get("expert_delta") or {}
        for expert in EXPERTS:
            if expert in means:
                series[expert].append(
                    {
                        "iteration": item["iteration"],
                        "value": means[expert],
                        "delta": deltas.get(expert, 0.0),
                    }
                )
    return series


def _alerts(iterations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    alerts = []
    for item in iterations:
        update = item.get("update") or {}
        if update and update.get("gate_grad_nonzero") is False:
            alerts.append({"level": "error", "iteration": item["iteration"], "reason": "gate_grad_zero"})
        counts = update.get("frontier_task_counts") or {}
        for task in EXPERTS:
            if counts and int(counts.get(task, 0)) == 0:
                alerts.append({"level": "warn", "iteration": item["iteration"], "task": task, "reason": "no_frontier_rows"})
    return alerts


def _load_gates(path: Path) -> dict[str, float]:
    payload = _read_json(path)
    if isinstance(payload.get("gates"), dict):
        payload = payload["gates"]
    return {str(key): float(value) for key, value in payload.items() if isinstance(value, (int, float))}


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
    except OSError:
        return []
    return rows


def _max_mtime(paths: list[Path]) -> float | None:
    mtimes = [path.stat().st_mtime for path in paths if path.exists()]
    return max(mtimes) if mtimes else None


def _float(value: Any) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(output) or math.isinf(output):
        return 0.0
    return output


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OP-VEC Monitor</title>
  <style>
    :root { color-scheme: dark; --bg:#111318; --panel:#181b22; --line:#2a2f3a; --text:#e8edf6; --muted:#99a2b3; --tool:#48d597; --memory:#6ea8ff; --code:#ffb86c; --bad:#ff6b6b; }
    * { box-sizing: border-box; }
    body { margin:0; font:14px/1.45 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background:var(--bg); color:var(--text); }
    header { padding:16px 20px; border-bottom:1px solid var(--line); display:flex; align-items:flex-start; justify-content:space-between; gap:18px; position:sticky; top:0; background:rgba(17,19,24,.95); z-index:5; }
    h1 { margin:0 0 4px; font-size:20px; font-weight:650; }
    .path { color:var(--muted); font-family:ui-monospace, SFMono-Regular, Menlo, monospace; overflow-wrap:anywhere; }
    .grid { display:grid; grid-template-columns:repeat(12,1fr); gap:14px; padding:14px; }
    .panel { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px; min-width:0; }
    .span4 { grid-column:span 4; } .span6 { grid-column:span 6; } .span8 { grid-column:span 8; } .span12 { grid-column:span 12; }
    h2 { margin:0 0 10px; font-size:14px; font-weight:650; color:#f3f6fb; }
    table { width:100%; border-collapse:collapse; }
    th, td { padding:7px 8px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }
    th { color:var(--muted); font-weight:600; font-size:12px; }
    .pill { display:inline-block; padding:2px 7px; border:1px solid var(--line); border-radius:999px; color:var(--muted); font-size:12px; }
    .num { font-variant-numeric:tabular-nums; font-family:ui-monospace, SFMono-Regular, Menlo, monospace; }
    .task-tool { color:var(--tool); } .task-memory { color:var(--memory); } .task-code { color:var(--code); }
    canvas { width:100%; height:260px; display:block; background:#101218; border:1px solid var(--line); border-radius:6px; }
    .controls { display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin-bottom:10px; }
    select, input { background:#101218; color:var(--text); border:1px solid var(--line); border-radius:6px; padding:6px 8px; }
    .samples { display:flex; gap:4px; align-items:center; min-width:140px; }
    .bar { height:14px; min-width:8px; border-radius:3px; background:#3a4050; }
    .bar.pos { background:#43c878; } .bar.neg { background:#d85d5d; }
    .alert { color:var(--bad); }
    @media (max-width: 980px) { .span4, .span6, .span8 { grid-column:span 12; } header { position:static; } }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>OP-VEC Run Monitor</h1>
      <div id="runDir" class="path"></div>
    </div>
    <div class="num" id="updatedAt">loading</div>
  </header>
  <main class="grid">
    <section class="panel span12"><h2>Alerts</h2><div id="alerts" class="path">loading</div></section>
    <section class="panel span6"><h2>Task Reward</h2><canvas id="rewardChart" width="900" height="320"></canvas></section>
    <section class="panel span6"><h2>Task Vector Coefficients</h2><canvas id="coefChart" width="900" height="320"></canvas></section>
    <section class="panel span12"><h2>Iterations</h2><div id="iterations"></div></section>
    <section class="panel span12">
      <h2>Prompt Rewards</h2>
      <div class="controls">
        <label>iteration <select id="iterFilter"></select></label>
        <label>task <select id="taskFilter"><option value="">all</option><option>tool</option><option>memory</option><option>code</option></select></label>
        <label>max rows <input id="maxRows" type="number" min="20" max="1000" value="300"></label>
      </div>
      <div id="promptRows"></div>
    </section>
  </main>
  <script>
    const colors = {tool:'#48d597', memory:'#6ea8ff', code:'#ffb86c'};
    let state = null;
    async function loadState() {
      const maxRows = document.getElementById('maxRows').value || 300;
      const res = await fetch('/api/state?max_prompt_rows=' + encodeURIComponent(maxRows), {cache:'no-store'});
      state = await res.json();
      render();
    }
    function render() {
      document.getElementById('runDir').textContent = state.run_dir;
      document.getElementById('updatedAt').textContent = state.updated_at;
      renderAlerts();
      renderIterations();
      renderFilters();
      drawSeries('rewardChart', state.reward_series, 'mean_reward');
      drawCoef();
      renderPromptRows();
    }
    function renderAlerts() {
      const box = document.getElementById('alerts');
      if (!state.alerts.length) { box.textContent = 'no alerts'; return; }
      box.innerHTML = state.alerts.map(a => '<span class="alert">' + esc(JSON.stringify(a)) + '</span>').join('<br>');
    }
    function renderIterations() {
      let html = '<table><thead><tr><th>iter</th><th>status</th><th>rows</th><th>task rewards</th><th>frontier</th><th>gate mean / max delta</th></tr></thead><tbody>';
      for (const it of state.iterations) {
        const rewards = Object.entries(it.task_stats || {}).map(([task,s]) => `<span class="task-${task}">${task}</span> <span class="num">${fmt(s.mean_reward)}</span> sr <span class="num">${fmt(s.success_rate)}</span>`).join('<br>');
        const frontier = it.update ? JSON.stringify(it.update.frontier_task_counts || {}) : '';
        const gs = it.gate_stats;
        const gate = gs ? `n=${gs.num_coefficients} mean=${fmt(gs.mean)} max_d=${fmt(gs.max_abs_delta_from_init)}` : '';
        html += `<tr><td class="num">${it.iteration}</td><td><span class="pill">${it.status}</span></td><td class="num">${it.rollout_rows}</td><td>${rewards}</td><td class="num">${esc(frontier)}</td><td class="num">${gate}</td></tr>`;
      }
      html += '</tbody></table>';
      document.getElementById('iterations').innerHTML = html;
    }
    function renderFilters() {
      const select = document.getElementById('iterFilter');
      const old = select.value;
      select.innerHTML = '<option value="">latest</option>' + state.iterations.map(it => `<option>${it.iteration}</option>`).join('');
      if ([...select.options].some(o => o.value === old)) select.value = old;
    }
    function renderPromptRows() {
      const iterValue = document.getElementById('iterFilter').value;
      const taskValue = document.getElementById('taskFilter').value;
      const it = iterValue ? state.iterations.find(x => x.iteration === iterValue) : state.iterations[state.iterations.length - 1];
      if (!it) { document.getElementById('promptRows').textContent = 'no rows'; return; }
      let rows = it.prompt_rows || [];
      if (taskValue) rows = rows.filter(r => r.task === taskValue);
      let html = '<table><thead><tr><th>prompt</th><th>task</th><th>mean</th><th>success</th><th>samples</th><th>status</th></tr></thead><tbody>';
      for (const row of rows) {
        const bars = row.samples.map(s => `<span title="${esc(s.sample_id)} reward=${fmt(s.reward)}" class="bar ${s.reward > 0 ? 'pos' : (s.reward < 0 ? 'neg' : '')}" style="width:${Math.max(8, Math.min(70, 12 + Math.abs(s.reward) * 12))}px"></span>`).join('');
        html += `<tr><td class="path">${esc(row.prompt_id)}</td><td class="task-${row.task}">${row.task}</td><td class="num">${fmt(row.mean_reward)}</td><td class="num">${row.success_count}/${row.samples.length}</td><td><div class="samples">${bars}</div></td><td>${row.keep_for_policy_loss ? '<span class="pill">frontier</span>' : esc(row.skip_reason || '')}</td></tr>`;
      }
      html += '</tbody></table>';
      document.getElementById('promptRows').innerHTML = html;
    }
    function drawCoef() {
      const series = state.coefficient_series || {};
      const normalized = {};
      for (const [task, points] of Object.entries(series)) {
        normalized[task] = points.map(p => ({iteration:p.iteration, mean_reward:p.value}));
      }
      drawSeries('coefChart', normalized, 'mean_reward');
    }
    function drawSeries(id, series, field) {
      const canvas = document.getElementById(id), ctx = canvas.getContext('2d');
      ctx.clearRect(0,0,canvas.width,canvas.height);
      const tasks = Object.keys(series);
      const labels = [...new Set(tasks.flatMap(t => series[t].map(p => p.iteration)))];
      let vals = tasks.flatMap(t => series[t].map(p => Number(p[field] || 0)));
      if (!vals.length) vals = [0,1];
      const min = Math.min(0, ...vals), max = Math.max(1e-6, ...vals);
      const pad = 42, w = canvas.width - pad*2, h = canvas.height - pad*2;
      ctx.strokeStyle = '#2a2f3a'; ctx.lineWidth = 1; ctx.strokeRect(pad, pad, w, h);
      ctx.fillStyle = '#99a2b3'; ctx.font = '22px ui-monospace';
      ctx.fillText(fmt(max), 6, pad+8); ctx.fillText(fmt(min), 6, pad+h);
      tasks.forEach(task => {
        const pts = series[task];
        ctx.strokeStyle = colors[task] || '#e8edf6'; ctx.fillStyle = ctx.strokeStyle; ctx.lineWidth = 4; ctx.beginPath();
        pts.forEach((p, i) => {
          const x = pad + (labels.indexOf(p.iteration) / Math.max(1, labels.length-1)) * w;
          const y = pad + h - ((Number(p[field] || 0) - min) / Math.max(1e-9, max - min)) * h;
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
          ctx.fillRect(x-4, y-4, 8, 8);
        });
        ctx.stroke();
      });
      labels.forEach((label, i) => { const x = pad + (i / Math.max(1, labels.length-1)) * w; ctx.fillStyle = '#99a2b3'; ctx.fillText(label.replace('iter_','i'), x-18, canvas.height-10); });
    }
    function fmt(x) { return Number(x || 0).toFixed(4); }
    function esc(s) { return String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
    document.getElementById('iterFilter').addEventListener('change', renderPromptRows);
    document.getElementById('taskFilter').addEventListener('change', renderPromptRows);
    document.getElementById('maxRows').addEventListener('change', loadState);
    loadState();
    setInterval(loadState, 10000);
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--init-value", type=float, default=1.0 / 3.0)
    parser.add_argument("--max-prompt-rows", type=int, default=300)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
