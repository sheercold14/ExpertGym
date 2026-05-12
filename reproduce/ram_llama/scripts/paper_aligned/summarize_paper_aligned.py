#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def find_lmeval_results(model_dir: Path) -> dict[str, dict[str, float]]:
    out = {}
    for path in sorted((model_dir / "math_lmeval").rglob("results*.json")):
        data = read_json(path).get("results", {})
        for task, metrics in data.items():
            task_metrics = out.setdefault(task, {})
            for key in ("exact_match,flexible-extract", "exact_match,none", "acc,none", "exact_match,strict-match"):
                if key in metrics:
                    acc = metrics[key]
                    if isinstance(acc, (int, float)):
                        task_metrics["exact_match"] = float(acc)
                    break
            verify = metrics.get("math_verify,none")
            if isinstance(verify, (int, float)):
                task_metrics["math_verify"] = float(verify)
    return out


def read_bfcl(model_dir: Path) -> dict[str, str]:
    csv_path = model_dir / "bfcl_project" / "score" / "data_overall.csv"
    if not csv_path.exists():
        return {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else {}


def read_search_summary(model_dir: Path) -> dict[str, float]:
    summary = read_json(model_dir / "summary.json")
    out = {}
    for name, metrics in summary.get("benchmarks", {}).items():
        acc = metrics.get("accuracy")
        if isinstance(acc, (int, float)):
            out[name] = float(acc)
    return out


def avg(values: list[float]) -> str:
    return "" if not values else f"{sum(values) / len(values):.4f}"


def metric_value(metrics: dict[str, dict[str, float]], task: str, metric: str) -> float | None:
    value = metrics.get(task, {}).get(metric)
    return value if isinstance(value, (int, float)) else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize paper-aligned RAM Llama outputs.")
    parser.add_argument("run_dir")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)

    rows = []
    for model_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        math = find_lmeval_results(model_dir)
        bfcl = read_bfcl(model_dir)
        search = read_search_summary(model_dir)
        gsm8k = metric_value(math, "gsm8k", "exact_match")
        math500_exact = metric_value(math, "minerva_math500", "exact_match")
        math500_verify = metric_value(math, "minerva_math500", "math_verify")
        math_avg = avg([v for v in [gsm8k, math500_exact] if v is not None])
        rows.append(
            {
                "model": model_dir.name,
                "math_avg": math_avg,
                "gsm8k_flex": f"{gsm8k:.4f}" if gsm8k is not None else "",
                "math500_exact": f"{math500_exact:.4f}" if math500_exact is not None else "",
                "math500_verify": f"{math500_verify:.4f}" if math500_verify is not None else "",
                "tool_overall": bfcl.get("Overall Acc", ""),
                "tool_live": bfcl.get("Live Acc", ""),
                "tool_non_live": bfcl.get("Non-Live AST Acc", ""),
                "search_avg": avg(list(search.values())),
                "search_nq": f"{search['nq_open']:.4f}" if "nq_open" in search else "",
                "search_2wiki": f"{search['two_wiki']:.4f}" if "two_wiki" in search else "",
            }
        )

    headers = [
        "model",
        "math_avg",
        "gsm8k_flex",
        "math500_exact",
        "math500_verify",
        "tool_overall",
        "tool_live",
        "tool_non_live",
        "search_avg",
        "search_nq",
        "search_2wiki",
    ]
    md = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        md.append("| " + " | ".join(row[h] for h in headers) + " |")

    (run_dir / "paper_aligned_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (run_dir / "paper_aligned_summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(run_dir / "paper_aligned_summary.md")


if __name__ == "__main__":
    main()
