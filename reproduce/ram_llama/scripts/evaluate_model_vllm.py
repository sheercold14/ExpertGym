#!/usr/bin/env python3
import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from evaluate_model import (
    DEFAULT_EVAL_ROOT,
    DEFAULT_OUTPUT_ROOT,
    EvalItem,
    build_chat_prompt,
    build_zerosearch_prompt,
    extract_answer_tag,
    get_search_query,
    json_safe,
    load_items,
    retrieve_for_search,
    score_item,
)


def model_name_from_path(path: str) -> str:
    return Path(path).name.replace("/", "__")


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_prompts(tokenizer: Any, items: list[EvalItem]) -> list[str]:
    return [build_chat_prompt(tokenizer, item.prompt) for item in items]


def generate_static(llm: LLM, tokenizer: Any, items: list[EvalItem], args: argparse.Namespace) -> list[tuple[str, list[dict[str, str]]]]:
    prompts = build_prompts(tokenizer, items)
    params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )
    outputs = llm.generate(prompts, params)
    preds = []
    for out in outputs:
        preds.append((out.outputs[0].text.strip(), []))
    return preds


def generate_zerosearch(
    llm: LLM,
    tokenizer: Any,
    items: list[EvalItem],
    args: argparse.Namespace,
) -> list[tuple[str, list[dict[str, str]]]]:
    prompts = []
    final_parts: list[list[str]] = []
    traces: list[list[dict[str, str]]] = []
    finished = []

    for item in items:
        question = item.meta.get("question")
        prompt = build_chat_prompt(tokenizer, build_zerosearch_prompt(question)) if question else build_chat_prompt(tokenizer, item.prompt)
        prompts.append(prompt)
        final_parts.append([])
        traces.append([])
        finished.append(False)

    stop = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n"]
    params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        stop=stop,
        include_stop_str_in_output=True,
    )

    for turn in range(args.search_max_turns + 1):
        active = [i for i, done in enumerate(finished) if not done]
        if not active:
            break
        active_prompts = [fit_prompt_to_budget(tokenizer, prompts[i], args.max_model_len - args.max_new_tokens - 8) for i in active]
        outputs = llm.generate(active_prompts, params)
        for local_idx, out in enumerate(outputs):
            i = active[local_idx]
            text = out.outputs[0].text
            prompts[i] += text
            final_parts[i].append(text)
            query = get_search_query(text)
            if not query or extract_answer_tag(text) is not None or turn >= args.search_max_turns:
                finished[i] = True
                continue
            info = retrieve_for_search(query, items[i], args)
            traces[i].append({"turn": str(turn), "query": query, "information": info})
            info_text = f"\n\n<information>{info}</information>\n\n"
            prompts[i] += info_text
            final_parts[i].append(info_text)

    return [("".join(parts).strip(), trace) for parts, trace in zip(final_parts, traces)]


def fit_prompt_to_budget(tokenizer: Any, prompt: str, max_tokens: int) -> str:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return prompt
    return tokenizer.decode(ids[-max_tokens:], skip_special_tokens=False)


def evaluate(args: argparse.Namespace) -> Path:
    eval_root = Path(args.eval_root)
    output_dir = Path(args.output_root) / args.run_id / args.model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=args.trust_remote_code,
    )

    summary = {"model": args.model, "model_name": args.model_name, "run_id": args.run_id, "benchmarks": {}}
    for benchmark in args.benchmarks.split(","):
        benchmark = benchmark.strip()
        if not benchmark:
            continue
        items = load_items(benchmark, eval_root, args.max_samples)
        bench_dir = output_dir / benchmark
        bench_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = bench_dir / "metrics.json"
        if metrics_path.exists() and not args.overwrite:
            metrics = json.loads(metrics_path.read_text())
            summary["benchmarks"][benchmark] = metrics
            print(f"[skip] {benchmark}: metrics already exists")
            continue
        t0 = time.time()
        if args.search_mode == "zerosearch" and benchmark in {"nq_open", "two_wiki"}:
            preds = generate_zerosearch(llm, tokenizer, items, args)
        else:
            preds = generate_static(llm, tokenizer, items, args)

        correct = 0
        with (bench_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
            for item, (pred, search_trace) in zip(items, preds):
                is_correct = score_item(item, pred)
                correct += int(is_correct)
                f.write(
                    json.dumps(
                        {
                            "id": item.item_id,
                            "benchmark": benchmark,
                            "correct": is_correct,
                            "prediction": pred,
                            "answer": json_safe(item.answer),
                            "meta": json_safe(item.meta),
                            "search_trace": json_safe(search_trace),
                            "error": None,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        total = len(items)
        metrics = {
            "benchmark": benchmark,
            "total": total,
            "correct": correct,
            "accuracy": correct / total if total else math.nan,
            "seconds": time.time() - t0,
            "max_samples": args.max_samples,
            "engine": "vllm",
            "search_mode": args.search_mode if benchmark in {"nq_open", "two_wiki"} else "static",
            "search_backend": args.search_backend if benchmark in {"nq_open", "two_wiki"} else None,
        }
        write_json(bench_dir / "metrics.json", metrics)
        summary["benchmarks"][benchmark] = metrics
        print(f"[done] {benchmark}: {correct}/{total} = {metrics['accuracy']:.4f}")

    scores = [m["accuracy"] for m in summary["benchmarks"].values() if not math.isnan(m["accuracy"])]
    summary["macro_accuracy"] = sum(scores) / len(scores) if scores else math.nan
    write_json(output_dir / "summary.json", summary)
    print(f"results: {output_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM evaluator for RAM Llama benchmarks.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--benchmarks", default="gsm8k,math500,two_wiki,bfcl_live,bfcl_non_live")
    parser.add_argument("--eval-root", default=str(DEFAULT_EVAL_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-id", default=time.strftime("eval-vllm-%Y%m%d-%H%M%S"))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--search-mode", choices=["static", "zerosearch"], default="static")
    parser.add_argument("--search-backend", choices=["none", "google", "exa", "wiki", "local_context"], default="none")
    parser.add_argument("--search-url", default="http://localhost:6002/retrieve")
    parser.add_argument("--search-topk", type=int, default=5)
    parser.add_argument("--search-max-turns", type=int, default=5)
    parser.add_argument("--search-info-max-chars", type=int, default=3500)
    parser.add_argument("--exa-search-type", choices=["auto", "fast", "instant", "deep-lite", "deep"], default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.model_name = args.model_name or model_name_from_path(args.model)
    evaluate(args)


if __name__ == "__main__":
    main()
