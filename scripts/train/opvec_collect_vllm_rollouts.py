#!/usr/bin/env python3
"""Collect vLLM rollouts from a baked OP-VEC policy checkpoint."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from opvec.config import load_config, write_json
from opvec.data.io import read_jsonl, write_jsonl
from opvec.data.prompt_filters import filter_memory_records, parse_memory_kind_filter
from opvec.data.prompt_sampler import task_balanced_sample
from opvec.data.schema import make_gate_id, validate_rollout_row
from opvec.modeling.bake import load_gate_values
from opvec.modeling.logprob import render_chat_prompt
from opvec.rewards.router import RewardRouter
from opvec.train.frontier import should_keep_frontier

from scripts.train.opvec_collect_hf_rollouts import (
    MEMAGENT_FINAL_TEMPLATE,
    MEMAGENT_UPDATE_TEMPLATE,
    _apply_behavior_span_mix,
    _memagent_trajectory,
)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    summary = collect_vllm_rollouts(config, args)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def collect_vllm_rollouts(config: dict, args: argparse.Namespace) -> dict:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    started = time.time()
    seed = int(args.seed if args.seed is not None else config["run"]["seed"])
    random.seed(seed)
    manifest_path = Path(args.seed_manifest or config["data"]["seed_manifest"])
    records = read_jsonl(manifest_path)
    if args.tasks:
        task_filter = {item.strip() for item in args.tasks.split(",") if item.strip()}
        records = [record for record in records if record.get("task") in task_filter]
        if not records:
            raise SystemExit(f"No prompt records matched --tasks={args.tasks}")
    memory_kinds = parse_memory_kind_filter(args.memory_kind)
    if memory_kinds is not None:
        records = filter_memory_records(records, memory_kinds)
        if not records:
            raise SystemExit(f"No prompt records matched --memory-kind={args.memory_kind}")
    if args.prompt_id:
        prompt_ids = {item.strip() for item in args.prompt_id.split(",") if item.strip()}
        prompts = [record for record in records if record.get("prompt_id") in prompt_ids]
        if len(prompts) != len(prompt_ids):
            found = {record.get("prompt_id") for record in prompts}
            missing = sorted(prompt_ids - found)
            raise SystemExit(f"Missing prompt_id records: {missing}")
    else:
        if args.prompt_offset:
            records = records[int(args.prompt_offset) :]
    if args.prompt_id:
        pass
    elif args.use_manifest_order:
        prompts = records[: args.num_prompts]
    else:
        prompts = task_balanced_sample(
            records,
            batch_size=args.num_prompts,
            task_weights={"tool": 1.0, "memory": 1.0, "code": 1.0},
            seed=seed,
        )

    model_path = args.policy_model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    sampling_params = SamplingParams(
        max_tokens=int(args.max_new_tokens),
        temperature=0.0 if args.greedy else float(args.temperature),
        top_p=1.0 if args.greedy else float(args.top_p),
        logprobs=1 if args.store_token_logprobs else None,
        skip_special_tokens=True,
    )
    memory_update_sampling_params = SamplingParams(
        max_tokens=int(args.memory_update_max_new_tokens or args.max_new_tokens),
        temperature=0.0 if args.greedy else float(args.temperature),
        top_p=1.0 if args.greedy else float(args.top_p),
        logprobs=1 if args.store_token_logprobs else None,
        skip_special_tokens=True,
    )
    memory_final_sampling_params = SamplingParams(
        max_tokens=int(args.memory_final_max_new_tokens or args.max_new_tokens),
        temperature=0.0 if args.greedy else float(args.temperature),
        top_p=1.0 if args.greedy else float(args.top_p),
        logprobs=1 if args.store_token_logprobs else None,
        skip_special_tokens=True,
    )
    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        trust_remote_code=True,
        dtype=args.dtype,
        tensor_parallel_size=int(args.tensor_parallel_size),
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        max_model_len=int(args.max_model_len),
        max_num_seqs=int(args.vllm_batch_size),
        seed=seed,
    )

    router = RewardRouter()
    gate_values = _gate_values(config, args)
    rollout_rows = []
    output_path = Path(args.output)
    stream_handle = None
    if args.stream_output:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        stream_handle = output_path.open("w", encoding="utf-8")
    try:
        for step, prompt_record in enumerate(prompts, start=1):
            row = collect_one_prompt_vllm(
                llm=llm,
                tokenizer=tokenizer,
                sampling_params=sampling_params,
                memory_update_sampling_params=memory_update_sampling_params,
                memory_final_sampling_params=memory_final_sampling_params,
                router=router,
                prompt_record=prompt_record,
                step=step,
                args=args,
                config=config,
                gate_values=gate_values,
            )
            validate_rollout_row(row)
            rollout_rows.append(row)
            if stream_handle is not None:
                stream_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream_handle.flush()
            if args.progress_every and (step % args.progress_every == 0 or step == len(prompts)):
                elapsed = time.time() - started
                print(
                    f"[opvec_collect_vllm_rollouts] completed {step}/{len(prompts)} prompts; "
                    f"last_prompt={prompt_record.get('prompt_id')} task={prompt_record.get('task')} "
                    f"elapsed={elapsed:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        if stream_handle is not None:
            stream_handle.close()
    count = len(rollout_rows) if args.stream_output else write_jsonl(args.output, rollout_rows)
    kept = sum(1 for row in rollout_rows if row["keep_for_policy_loss"])
    summary = {
        "format": "opvec_vllm_gated_grpo_rollout_v1",
        "run_id": args.run_id,
        "output": str(args.output),
        "rows": count,
        "kept_frontiers": kept,
        "greedy": bool(args.greedy),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "seed": seed,
        "prompt_offset": int(args.prompt_offset),
        "policy_model": model_path,
        "gate_checkpoint": args.gate_checkpoint,
        "gate_values": gate_values,
        "policy_id": args.policy_id,
        "old_logprob": (
            "stored from vLLM generation token logprobs"
            if args.store_token_logprobs
            else "missing; fill in update with --fill-missing-old-logprob"
        ),
        "store_token_logprobs": bool(args.store_token_logprobs),
        "token_logprob_samples": _token_logprob_sample_count(rollout_rows),
        "tensor_parallel_size": int(args.tensor_parallel_size),
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "max_model_len": int(args.max_model_len),
        "vllm_batch_size": int(args.vllm_batch_size),
        "memory_update_max_new_tokens": int(args.memory_update_max_new_tokens or args.max_new_tokens),
        "memory_final_max_new_tokens": int(args.memory_final_max_new_tokens or args.max_new_tokens),
        "tasks": dict(_task_counts(rollout_rows)),
        "elapsed_seconds": time.time() - started,
        "reward_definition": {
            "main": "task_reward",
            "behavior_span_weight": float(
                args.behavior_span_reward_weight
                if args.behavior_span_reward_weight is not None
                else config.get("reward", {}).get("behavior_span_weight", 0.0)
            ),
        },
    }
    write_json(output_path.with_suffix(".summary.json"), summary)
    return summary


def collect_one_prompt_vllm(
    *,
    llm,
    tokenizer,
    sampling_params,
    memory_update_sampling_params,
    memory_final_sampling_params,
    router: RewardRouter,
    prompt_record: dict,
    step: int,
    args: argparse.Namespace,
    config: dict,
    gate_values: dict[str, float],
) -> dict:
    trajectory = _memagent_trajectory(prompt_record)
    if trajectory is not None:
        return collect_one_memagent_trajectory_vllm(
            llm=llm,
            tokenizer=tokenizer,
            update_sampling_params=memory_update_sampling_params,
            final_sampling_params=memory_final_sampling_params,
            router=router,
            prompt_record=prompt_record,
            step=step,
            args=args,
            config=config,
            gate_values=gate_values,
            trajectory=trajectory,
        )
    prompt_text = render_chat_prompt(tokenizer, prompt_record.get("messages"), prompt_record.get("prompt", ""))
    prompt_ids = _prompt_token_ids(tokenizer, prompt_text, args.max_prompt_tokens)
    generated = _vllm_generate_samples(
        llm,
        [prompt_ids for _ in range(args.samples_per_prompt)],
        sampling_params,
        batch_size=args.vllm_batch_size,
        store_token_logprobs=args.store_token_logprobs,
    )
    texts = [item["text"] for item in generated]
    samples = []
    scored_outputs = router.batch_score(prompt_record, texts)
    for sample_idx, (generated_item, scored) in enumerate(zip(generated, scored_outputs)):
        response_text = str(generated_item["text"])
        scored = _apply_behavior_span_mix(
            scored,
            behavior_span_weight=float(
                args.behavior_span_reward_weight
                if args.behavior_span_reward_weight is not None
                else config.get("reward", {}).get("behavior_span_weight", 0.0)
            ),
        )
        samples.append(
            {
                "sample_id": f"{prompt_record['prompt_id']}__k{sample_idx}",
                "text": response_text,
                "old_logprob": generated_item.get("old_logprob"),
                "old_logprob_max_length": None,
                "length": len(generated_item.get("response_token_ids", [])) or _token_len(tokenizer, response_text),
                **scored,
                **_token_payload_fields(generated_item),
            }
        )
    keep, skip_reason, frontier = should_keep_frontier(
        samples,
        min_frontier_weight=float(config["frontier"]["min_frontier_weight"]),
        min_reward_std=float(config["frontier"]["min_reward_std"]),
    )
    return {
        "run_id": args.run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "step": step,
        "policy_id": args.policy_id,
        "gate_checkpoint": args.gate_checkpoint,
        "gate_values": gate_values,
        "gate_id": make_gate_id(gate_values, args.gate_checkpoint),
        "group_id": prompt_record.get("group_id") or prompt_record["prompt_id"],
        "seed": int(args.seed if args.seed is not None else config["run"]["seed"]),
        "installed_modules": [],
        "prompt_id": prompt_record["prompt_id"],
        "task": prompt_record["task"],
        "prompt": prompt_record.get("prompt", ""),
        "reference": prompt_record.get("reference", {}),
        "rendered_prompt": prompt_text,
        "samples": samples,
        "frontier": frontier,
        "keep_for_policy_loss": keep,
        "skip_reason": skip_reason,
    }


def collect_one_memagent_trajectory_vllm(
    *,
    llm,
    tokenizer,
    update_sampling_params,
    final_sampling_params,
    router: RewardRouter,
    prompt_record: dict,
    step: int,
    args: argparse.Namespace,
    config: dict,
    gate_values: dict[str, float],
    trajectory: dict,
) -> dict:
    question = str(trajectory["prompt"])
    chunks = [str(item) for item in trajectory["chunks"]]
    memory_texts = ["No previous memory" for _ in range(args.samples_per_prompt)]
    turns_by_sample = [[] for _ in range(args.samples_per_prompt)]
    total_lengths = [0 for _ in range(args.samples_per_prompt)]
    first_prompt_text = None
    for chunk_idx, chunk in enumerate(chunks, start=1):
        prompt_texts = []
        for memory_text in memory_texts:
            user_text = MEMAGENT_UPDATE_TEMPLATE.format(prompt=question, memory=memory_text, chunk=chunk)
            prompt_text = render_chat_prompt(tokenizer, [{"role": "user", "content": user_text}], user_text)
            prompt_texts.append(prompt_text)
        first_prompt_text = first_prompt_text or prompt_texts[0]
        generated = _vllm_generate_samples(
            llm,
            [_prompt_token_ids(tokenizer, prompt_text, args.max_prompt_tokens) for prompt_text in prompt_texts],
            update_sampling_params,
            batch_size=args.vllm_batch_size,
            store_token_logprobs=args.store_token_logprobs,
        )
        for sample_idx, generated_item in enumerate(generated):
            response_text = str(generated_item["text"])
            response_len = len(generated_item.get("response_token_ids", [])) or _token_len(tokenizer, response_text)
            turns_by_sample[sample_idx].append(
                {
                    "turn": chunk_idx,
                    "kind": "memory_update",
                    "prompt_text": prompt_texts[sample_idx],
                    "text": response_text,
                    "old_logprob": generated_item.get("old_logprob"),
                    "length": response_len,
                    **_token_payload_fields(generated_item),
                }
            )
            total_lengths[sample_idx] += response_len
            memory_texts[sample_idx] = response_text
    final_prompt_texts = []
    for memory_text in memory_texts:
        user_text = MEMAGENT_FINAL_TEMPLATE.format(prompt=question, memory=memory_text)
        prompt_text = render_chat_prompt(tokenizer, [{"role": "user", "content": user_text}], user_text)
        final_prompt_texts.append(prompt_text)
    first_prompt_text = first_prompt_text or final_prompt_texts[0]
    final_generated = _vllm_generate_samples(
        llm,
        [_prompt_token_ids(tokenizer, prompt_text, args.max_prompt_tokens) for prompt_text in final_prompt_texts],
        final_sampling_params,
        batch_size=args.vllm_batch_size,
        store_token_logprobs=args.store_token_logprobs,
    )
    final_texts = [item["text"] for item in final_generated]
    samples = []
    scored_outputs = router.batch_score(prompt_record, final_texts)
    for sample_idx, (generated_item, scored) in enumerate(zip(final_generated, scored_outputs)):
        final_text = str(generated_item["text"])
        final_len = len(generated_item.get("response_token_ids", [])) or _token_len(tokenizer, final_text)
        total_lengths[sample_idx] += final_len
        turns_by_sample[sample_idx].append(
            {
                "turn": len(chunks) + 1,
                "kind": "final_answer",
                "prompt_text": final_prompt_texts[sample_idx],
                "text": final_text,
                "old_logprob": generated_item.get("old_logprob"),
                "length": final_len,
                **_token_payload_fields(generated_item),
            }
        )
        scored = _apply_behavior_span_mix(
            scored,
            behavior_span_weight=float(
                args.behavior_span_reward_weight
                if args.behavior_span_reward_weight is not None
                else config.get("reward", {}).get("behavior_span_weight", 0.0)
            ),
        )
        sample_payload = _aggregate_trajectory_token_payload(turns_by_sample[sample_idx])
        samples.append(
            {
                "sample_id": f"{prompt_record['prompt_id']}__k{sample_idx}",
                "text": final_text,
                "old_logprob": sample_payload.get("old_logprob"),
                "old_logprob_max_length": None,
                "length": total_lengths[sample_idx],
                "trajectory": turns_by_sample[sample_idx],
                **scored,
                **_token_payload_fields(sample_payload),
            }
        )
    keep, skip_reason, frontier = should_keep_frontier(
        samples,
        min_frontier_weight=float(config["frontier"]["min_frontier_weight"]),
        min_reward_std=float(config["frontier"]["min_reward_std"]),
    )
    return {
        "run_id": args.run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "step": step,
        "policy_id": args.policy_id,
        "gate_checkpoint": args.gate_checkpoint,
        "gate_values": gate_values,
        "gate_id": make_gate_id(gate_values, args.gate_checkpoint),
        "group_id": prompt_record.get("group_id") or prompt_record["prompt_id"],
        "seed": int(args.seed if args.seed is not None else config["run"]["seed"]),
        "installed_modules": [],
        "prompt_id": prompt_record["prompt_id"],
        "task": prompt_record["task"],
        "prompt": question,
        "reference": prompt_record.get("reference", {}),
        "rendered_prompt": first_prompt_text or "",
        "memagent_trajectory": {"num_chunks": len(chunks), "official_templates": True},
        "samples": samples,
        "frontier": frontier,
        "keep_for_policy_loss": keep,
        "skip_reason": skip_reason,
    }


def _vllm_generate_samples(
    llm,
    prompt_token_ids_list: list[list[int]],
    sampling_params,
    *,
    batch_size: int,
    store_token_logprobs: bool,
) -> list[dict]:
    samples = []
    for start in range(0, len(prompt_token_ids_list), batch_size):
        prompts = [{"prompt_token_ids": token_ids} for token_ids in prompt_token_ids_list[start : start + batch_size]]
        outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
        for output in outputs:
            completion = output.outputs[0] if output.outputs else None
            samples.append(_vllm_completion_payload(completion, store_token_logprobs=store_token_logprobs))
    return samples


def _vllm_completion_payload(completion, *, store_token_logprobs: bool) -> dict:
    if completion is None:
        return {"text": "", "old_logprob": None}
    payload = {"text": str(getattr(completion, "text", "") or "")}
    if store_token_logprobs:
        token_payload = _vllm_token_logprob_payload(completion)
        if token_payload is not None:
            payload.update(token_payload)
    return payload


def _vllm_token_logprob_payload(completion) -> dict | None:
    token_ids = [int(token_id) for token_id in list(getattr(completion, "token_ids", []) or [])]
    if not token_ids:
        return None
    logprobs_by_position = getattr(completion, "logprobs", None)
    if logprobs_by_position is None:
        raise ValueError("vLLM did not return token logprobs; set SamplingParams(logprobs=1).")
    if len(logprobs_by_position) != len(token_ids):
        raise ValueError(f"vLLM token/logprob length mismatch: tokens={len(token_ids)} logprobs={len(logprobs_by_position)}")
    old_logprobs = [
        _sampled_token_logprob(position_logprobs, token_id)
        for position_logprobs, token_id in zip(logprobs_by_position, token_ids)
    ]
    return {
        "response_token_ids": token_ids,
        "old_logprobs": old_logprobs,
        "response_mask": [1 for _ in token_ids],
        "old_logprob": float(sum(old_logprobs)),
    }


def _sampled_token_logprob(position_logprobs, token_id: int) -> float:
    if position_logprobs is None:
        raise ValueError(f"Missing vLLM logprob entry for sampled token_id={token_id}")
    if isinstance(position_logprobs, dict):
        entry = position_logprobs.get(int(token_id))
        if entry is None:
            entry = position_logprobs.get(str(token_id))
        if entry is None and len(position_logprobs) == 1:
            entry = next(iter(position_logprobs.values()))
        if entry is None:
            available = list(position_logprobs.keys())[:8]
            raise ValueError(f"Sampled token_id={token_id} not found in vLLM logprobs; available={available}")
        return _logprob_value(entry)
    return _logprob_value(position_logprobs)


def _logprob_value(entry) -> float:
    if hasattr(entry, "logprob"):
        return float(entry.logprob)
    if isinstance(entry, dict) and "logprob" in entry:
        return float(entry["logprob"])
    if isinstance(entry, (tuple, list)) and entry:
        return float(entry[0])
    return float(entry)


def _token_payload_fields(payload: dict) -> dict:
    keys = ("response_token_ids", "old_logprobs", "response_mask")
    return {key: payload[key] for key in keys if key in payload}


def _aggregate_trajectory_token_payload(turns: list[dict]) -> dict:
    token_ids = []
    old_logprobs = []
    response_mask = []
    for turn in turns:
        if not all(key in turn for key in ("response_token_ids", "old_logprobs", "response_mask")):
            continue
        token_ids.extend(int(value) for value in turn["response_token_ids"])
        old_logprobs.extend(float(value) for value in turn["old_logprobs"])
        response_mask.extend(int(value) for value in turn["response_mask"])
    if not token_ids:
        return {"old_logprob": None}
    return {
        "response_token_ids": token_ids,
        "old_logprobs": old_logprobs,
        "response_mask": response_mask,
        "old_logprob": float(sum(old_logprobs)),
    }


def _prompt_token_ids(tokenizer, prompt_text: str, max_prompt_tokens: int) -> list[int]:
    return tokenizer(prompt_text, truncation=True, max_length=max_prompt_tokens).input_ids


def _token_len(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def _gate_values(config: dict, args: argparse.Namespace) -> dict[str, float]:
    if args.no_gate_values:
        return {}
    if args.gate_checkpoint:
        return load_gate_values(args.gate_checkpoint)
    return {str(key): float(value) for key, value in config.get("initial_gates", {}).items()}


def _task_counts(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        task = str(row.get("task"))
        counts[task] = counts.get(task, 0) + 1
    return counts


def _token_logprob_sample_count(rows: list[dict]) -> int:
    count = 0
    for row in rows:
        for sample in row.get("samples", []):
            if all(key in sample for key in ("response_token_ids", "old_logprobs", "response_mask")):
                count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/gated_grpo.yaml")
    parser.add_argument("--seed-manifest", default=None)
    parser.add_argument("--mode-manifest", default=None, help="Accepted for launcher symmetry; vLLM uses the baked policy checkpoint.")
    parser.add_argument("--policy-model", required=True, help="Baked policy checkpoint loaded by vLLM.")
    parser.add_argument("--policy-id", default="gated-grpo-vllm-baked-policy")
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id", default="vllm-rollout")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-prompts", type=int, default=3)
    parser.add_argument("--prompt-offset", type=int, default=0, help="Skip this many records after task/memory filters.")
    parser.add_argument("--tasks", default=None, help="Optional comma-separated task filter: tool,memory,code.")
    parser.add_argument("--memory-kind", default=None, help="Optional Memory phase filter: final_answer,memory_update.")
    parser.add_argument("--prompt-id", default=None, help="Optional comma-separated prompt_id filter.")
    parser.add_argument("--use-manifest-order", action="store_true", help="Use the manifest rows directly instead of task-balanced resampling.")
    parser.add_argument("--samples-per-prompt", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--memory-update-max-new-tokens", type=int, default=None)
    parser.add_argument("--memory-final-max-new-tokens", type=int, default=None)
    parser.add_argument("--max-prompt-tokens", type=int, default=8192)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--vllm-batch-size", type=int, default=32)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--store-token-logprobs", action="store_true", help="Ask vLLM to store sampled response token ids and old token logprobs.")
    parser.add_argument("--stream-output", action="store_true")
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--gate-checkpoint", default=None)
    parser.add_argument("--no-gate-values", action="store_true", help="Use when --policy-model is a static non-gated baseline.")
    parser.add_argument("--behavior-span-reward-weight", type=float, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
