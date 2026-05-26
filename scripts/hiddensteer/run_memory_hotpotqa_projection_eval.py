#!/usr/bin/env python3
"""Small HotpotQA recurrent F1 eval with direct Memory projection.

This runs a local HF version of the MemAgent recurrent-boxed HotpotQA protocol:
chunk context -> update memory -> final boxed answer.  It compares baseline
generation against direct code-component projection on the same examples.
"""

from __future__ import annotations

import argparse
import json
import re
import string
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.hiddensteer import train_memory_residual_immunization as base
from scripts.hiddensteer.run_memory_code_component_projection import (
    DirectCodeComponentProjector,
    build_centered_code_bad_geometry,
)


DEFAULT_INIT1_MODEL = "/tmp/shared-storage/OnPolicy/checkpoints/ta_init1_global_20260517"
DEFAULT_EVAL_DATA = "/tmp/shared-storage/dataset/hotpotqa/eval_50.json"
DEFAULT_OUTPUT_DIR = "/tmp/shared-storage/ExpertGym/hiddensteer/memory_hotpotqa_projection_eval_20260525"

TEMPLATE_UPDATE = """You are presented with a problem, a section of an article that may contain the answer to the problem, and a previous memory. Please read the provided section carefully and update the memory with the new information that helps to answer the problem. Be sure to retain all relevant details from the previous memory while adding any new, useful information.

<problem> 
{prompt}
</problem>

<memory>
{memory}
</memory>

<section>
{chunk}
</section>

Updated memory:
"""

TEMPLATE_FINAL_BOXED = """You are presented with a problem and a previous memory. Please answer the problem based on the previous memory and put the answer in \\boxed{{}}.

<problem> 
{prompt}
</problem>

<memory>
{memory}
</memory>

Your answer:
"""

NO_MEMORY = "No previous memory"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base.seed_everything(int(args.seed))

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[str(args.torch_dtype)]
    layers = base.parse_layers(args.layers)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    geometry_path = Path(args.geometry_path).expanduser().resolve() if args.geometry_path else output_dir / "memory_code_bad_centered_basis.pt"
    if args.geometry_path:
        geometry = torch.load(geometry_path, map_location="cpu")
    else:
        args.threshold_quantiles = str(args.threshold_quantile)
        geometry_examples = load_geometry_examples(args, tokenizer)
        geometry = build_centered_code_bad_geometry(
            args=args,
            encoded_examples=geometry_examples,
            layers=layers,
            dtype=dtype,
            AutoModelForCausalLM=AutoModelForCausalLM,
        )
        torch.save(geometry, geometry_path)

    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype, trust_remote_code=True)
    model.to(args.device)
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    projector = DirectCodeComponentProjector(
        layers=layers,
        geometry=geometry,
        alpha=float(args.alpha),
        threshold_quantile=float(args.threshold_quantile),
        device=str(args.device),
        dtype=dtype,
    )
    hooks = base.install_corrector_hooks(model, projector)

    data = load_eval_items(Path(args.eval_data).expanduser().resolve(), offset=int(args.offset), limit=int(args.num_samples))
    records = []
    try:
        started = time.perf_counter()
        for local_idx, item in enumerate(data):
            print(f"[{local_idx + 1}/{len(data)}] index={item.get('index')} baseline", flush=True)
            baseline = run_recurrent_item(model, tokenizer, projector, item, args, enabled=False)
            print(f"[{local_idx + 1}/{len(data)}] index={item.get('index')} projected", flush=True)
            projector.reset_stats()
            projected = run_recurrent_item(model, tokenizer, projector, item, args, enabled=True)
            gold = str((item.get("answers") or [""])[0])
            baseline_scores = score_prediction(baseline["pred"], gold)
            projected_scores = score_prediction(projected["pred"], gold)
            record = {
                "index": item.get("index"),
                "_local_id": local_idx,
                "question": item.get("input"),
                "gold": gold,
                "baseline": baseline,
                "projected": projected,
                "baseline_scores": baseline_scores,
                "projected_scores": projected_scores,
                "delta_f1": projected_scores["f1"] - baseline_scores["f1"],
                "delta_em": projected_scores["em"] - baseline_scores["em"],
                "projection_stats": dict(projector.stats),
            }
            records.append(record)
            append_jsonl(output_dir / "paired_results.jsonl", record)
            print(
                f"  f1 {baseline_scores['f1']:.3f}->{projected_scores['f1']:.3f} "
                f"em {baseline_scores['em']:.0f}->{projected_scores['em']:.0f}",
                flush=True,
            )
        summary = summarize(records)
        summary.update(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "objective": "HotpotQA recurrent boxed subset F1 with direct Memory projection",
                "model_path": str(args.model_path),
                "memory_model_path": str(args.memory_model_path),
                "code_model_path": str(args.code_model_path),
                "eval_data": str(args.eval_data),
                "offset": int(args.offset),
                "num_samples": len(data),
                "layers": layers,
                "basis_rank": int(args.basis_rank),
                "alpha": float(args.alpha),
                "threshold_quantile": float(args.threshold_quantile),
                "geometry_path": str(geometry_path),
                "wall_time_sec": time.perf_counter() - started,
                "generation": {
                    "do_sample": bool(args.do_sample),
                    "temperature": float(args.temperature),
                    "top_p": float(args.top_p),
                    "chunk_size": int(args.recurrent_chunk_size),
                    "max_context_tokens": int(args.recurrent_max_context_tokens),
                    "max_update_new_tokens": int(args.max_update_new_tokens),
                    "max_final_new_tokens": int(args.max_final_new_tokens),
                    "use_chat_template": not bool(args.no_chat_template),
                    "project_scope": str(args.project_scope),
                },
            }
        )
        base.write_json(output_dir / "summary.json", summary)
        write_markdown(output_dir / "README.md", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        for handle in hooks:
            handle.remove()
        model.cpu()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_geometry_examples(args: argparse.Namespace, tokenizer: Any) -> list[base.EncodedExample]:
    examples = base.load_memory_examples(
        Path(args.rollout_path).expanduser().resolve(),
        turn_kinds=base.split_csv(args.turn_kinds),
        max_examples=int(args.max_geometry_examples),
        max_turns_per_prompt=int(args.max_turns_per_prompt),
        prefer_success=not bool(args.allow_failed_teacher_samples),
    )
    train_examples, _heldout = base.split_examples_by_prompt(
        examples,
        train_count=int(args.geometry_train_examples),
        heldout_count=max(1, int(args.geometry_heldout_examples)),
        seed=int(args.seed),
    )
    return [
        base.encode_example(
            tokenizer,
            item,
            max_seq_length=int(args.geometry_max_seq_length),
            response_tail_tokens=int(args.geometry_response_tail_tokens),
            device=str(args.device),
        )
        for item in train_examples
    ]


def load_eval_items(path: Path, *, offset: int, limit: int) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data[offset : offset + limit]


def run_recurrent_item(
    model: torch.nn.Module,
    tokenizer: Any,
    projector: DirectCodeComponentProjector,
    item: dict[str, Any],
    args: argparse.Namespace,
    *,
    enabled: bool,
) -> dict[str, Any]:
    context = str(item["context"]).strip()
    question = str(item["input"]).strip()
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    max_context = int(args.recurrent_max_context_tokens)
    if len(context_ids) > max_context:
        context_ids = context_ids[: max_context // 2] + context_ids[-max_context // 2 :]
    memory = NO_MEMORY
    update_outputs = []
    started = time.perf_counter()
    old_enabled = projector.enabled
    try:
        for start in range(0, len(context_ids), int(args.recurrent_chunk_size)):
            projector.enabled = enabled and str(args.project_scope) in {"all", "update"}
            chunk_ids = context_ids[start : start + int(args.recurrent_chunk_size)]
            chunk = tokenizer.decode(chunk_ids, skip_special_tokens=True)
            prompt = TEMPLATE_UPDATE.format(prompt=question, memory=memory, chunk=chunk)
            response = generate_text(
                model,
                tokenizer,
                prompt,
                args,
                max_new_tokens=int(args.max_update_new_tokens),
            )
            memory = extract_solution(response) or response.strip()
            update_outputs.append(response)
        projector.enabled = enabled and str(args.project_scope) in {"all", "final"}
        final_prompt = TEMPLATE_FINAL_BOXED.format(prompt=question, memory=memory)
        final_response = generate_text(
            model,
            tokenizer,
            final_prompt,
            args,
            max_new_tokens=int(args.max_final_new_tokens),
        )
    finally:
        projector.enabled = old_enabled
    pred = extract_boxed_answer(final_response) or extract_answer(final_response) or final_response.strip()
    return {
        "pred": pred,
        "final_response": final_response,
        "memory_preview": memory[:500],
        "updates": len(update_outputs),
        "wall_time_sec": time.perf_counter() - started,
    }


def generate_text(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    args: argparse.Namespace,
    *,
    max_new_tokens: int,
) -> str:
    if not bool(args.no_chat_template) and getattr(tokenizer, "chat_template", None):
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
        )
    else:
        input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    input_ids = input_ids.to(args.device)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=bool(args.do_sample),
            temperature=float(args.temperature) if bool(args.do_sample) else None,
            top_p=float(args.top_p) if bool(args.do_sample) else None,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    return tokenizer.decode(generated[0, input_ids.shape[1] :], skip_special_tokens=True).strip()


def extract_solution(text: str) -> str | None:
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text.strip() or None


def extract_answer(text: str) -> str | None:
    raw = text.replace("*", "")
    lower = raw.lower()
    marker = "the answer is"
    if marker not in lower:
        return None
    start = lower.rfind(marker) + len(marker)
    return raw[start:].strip().strip(".").strip()


def extract_boxed_answer(text: str) -> str | None:
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    brace = text.find("{", idx)
    if brace < 0:
        tail = text[idx + len("\\boxed") :].strip()
        return tail.splitlines()[0].strip() if tail else None
    depth = 0
    for pos in range(brace, len(text)):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                return text[brace + 1 : pos].strip()
    return None


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(text.lower())))


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


def score_prediction(prediction: str | None, gold: str) -> dict[str, float]:
    pred = prediction or ""
    f1, precision, recall = f1_score(pred, gold)
    normalized_pred = normalize_answer(pred)
    normalized_gold = normalize_answer(gold)
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "em": float(normalized_pred == normalized_gold),
        "sub_em": float((normalized_gold in normalized_pred) or (normalized_pred in normalized_gold)),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = max(len(records), 1)
    baseline = {key: sum(r["baseline_scores"][key] for r in records) / n for key in ["f1", "em", "sub_em"]}
    projected = {key: sum(r["projected_scores"][key] for r in records) / n for key in ["f1", "em", "sub_em"]}
    return {
        "baseline": baseline,
        "projected": projected,
        "delta": {key: projected[key] - baseline[key] for key in baseline},
        "wins": sum(1 for r in records if r["delta_f1"] > 1.0e-9),
        "losses": sum(1 for r in records if r["delta_f1"] < -1.0e-9),
        "ties": sum(1 for r in records if abs(r["delta_f1"]) <= 1.0e-9),
    }


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Memory HotpotQA Projection Eval",
        "",
        f"- Created: `{summary['created_at']}`",
        f"- Model: `{summary['model_path']}`",
        f"- Data: `{summary['eval_data']}`",
        f"- Offset / samples: `{summary['offset']}` / `{summary['num_samples']}`",
        f"- Layers: `{summary['layers']}`",
        f"- Rank / threshold / alpha: `{summary['basis_rank']}` / `{summary['threshold_quantile']}` / `{summary['alpha']}`",
        f"- Geometry: `{summary['geometry_path']}`",
        "",
        "| metric | baseline | projected | delta |",
        "|---|---:|---:|---:|",
    ]
    for metric in ["f1", "em", "sub_em"]:
        lines.append(
            f"| {metric} | {summary['baseline'][metric]:.4f} | "
            f"{summary['projected'][metric]:.4f} | {summary['delta'][metric]:+.4f} |"
        )
    lines.extend(
        [
            "",
            f"- F1 wins/losses/ties: `{summary['wins']}` / `{summary['losses']}` / `{summary['ties']}`",
            f"- Wall time: `{summary['wall_time_sec']:.2f}s`",
            "",
            "This is a local deterministic HF subset check, not a formal vLLM eval_50 reproduction.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_INIT1_MODEL)
    parser.add_argument("--memory-model-path", default=base.DEFAULT_MEMORY_MODEL_PATH)
    parser.add_argument("--code-model-path", default=base.DEFAULT_CODE_MODEL_PATH)
    parser.add_argument("--rollout-path", default=base.DEFAULT_ROLLOUT_PATH)
    parser.add_argument("--eval-data", default=DEFAULT_EVAL_DATA)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--geometry-path", default="")
    parser.add_argument("--turn-kinds", default="memory_update")
    parser.add_argument("--layers", default="16-27")
    parser.add_argument("--basis-rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--threshold-quantile", type=float, default=0.90)
    parser.add_argument("--max-geometry-examples", type=int, default=32)
    parser.add_argument("--geometry-train-examples", type=int, default=20)
    parser.add_argument("--geometry-heldout-examples", type=int, default=8)
    parser.add_argument("--max-turns-per-prompt", type=int, default=1)
    parser.add_argument("--geometry-max-seq-length", type=int, default=1536)
    parser.add_argument("--geometry-response-tail-tokens", type=int, default=64)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--recurrent-max-context-tokens", type=int, default=120000)
    parser.add_argument("--recurrent-chunk-size", type=int, default=5000)
    parser.add_argument("--max-update-new-tokens", type=int, default=512)
    parser.add_argument("--max-final-new-tokens", type=int, default=128)
    parser.add_argument("--project-scope", choices=["all", "update", "final"], default="all")
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--allow-failed-teacher-samples", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
