#!/usr/bin/env python3
"""Collect real HF rollouts from a gated OP-VEC policy."""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from opvec.config import load_config, write_json
from opvec.data.io import read_jsonl, write_jsonl
from opvec.data.prompt_filters import filter_memory_records, parse_memory_kind_filter
from opvec.data.prompt_sampler import task_balanced_sample
from opvec.data.schema import make_gate_id, validate_rollout_row
from opvec.modeling.apply_gates import install_gated_linears_from_manifest
from opvec.modeling.bake import load_gate_values
from opvec.modeling.devices import model_input_device, model_load_device_kwargs
from opvec.modeling.gate_parameters import make_torch_gate_manager
from opvec.modeling.logprob import render_chat_prompt, response_logprob_details_from_text, response_logprob_from_text
from opvec.modeling.manifest import manifest_param_names
from opvec.rewards.router import RewardRouter
from opvec.train.frontier import should_keep_frontier


MEMAGENT_UPDATE_TEMPLATE = """You are presented with a problem, a section of an article that may contain the answer to the problem, and a previous memory. Please read the provided section carefully and update the memory with the new information that helps to answer the problem. Be sure to retain all relevant details from the previous memory while adding any new, useful information.

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

MEMAGENT_FINAL_TEMPLATE = """You are presented with a problem and a previous memory. Please answer the problem based on the previous memory and put the answer in \\boxed{{}}.

<problem> 
{prompt}
</problem>

<memory>
{memory}
</memory>

Your answer:
"""


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    summary = collect_hf_rollouts(config, args)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def collect_hf_rollouts(config: dict, args: argparse.Namespace) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    seed = int(args.seed if args.seed is not None else config["run"]["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if args.gate_checkpoint:
        config = {**config, "initial_gates": load_gate_values(args.gate_checkpoint)}
    device = args.device
    dtype = getattr(torch, args.torch_dtype)
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
    elif args.use_manifest_order:
        prompts = records[: args.num_prompts]
    else:
        prompts = task_balanced_sample(
            records,
            batch_size=args.num_prompts,
            task_weights={"tool": 1.0, "memory": 1.0, "code": 1.0},
            seed=seed,
        )
    model_path = args.policy_model or config["models"]["base"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        **model_load_device_kwargs(device_map=args.device_map, max_memory=args.max_memory),
    )
    if not args.device_map:
        model.to(device)
    input_device = model_input_device(model, torch, device)
    model.eval()
    gate_parameterization = _normalize_gate_parameterization(args.gate_parameterization)
    param_names = None
    gate_manager = None
    installed: list[str] = []
    if not args.disable_gates:
        param_names = manifest_param_names(args.mode_manifest) if _uses_parameter_names(gate_parameterization) else None
        gate_manager = make_torch_gate_manager(
            torch,
            config,
            parameterization=gate_parameterization,
            param_names=param_names,
        ).to(input_device)
        installed = install_gated_linears_from_manifest(
            torch,
            model,
            mode_manifest_path=args.mode_manifest,
            gate_manager=gate_manager,
            max_modules=None if args.max_gated_modules in (None, 0) else args.max_gated_modules,
            device=None if args.device_map else device,
        )
    router = RewardRouter()
    rollout_rows = []
    stream_handle = None
    if args.stream_output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        stream_handle = output_path.open("w", encoding="utf-8")
    try:
        for step, prompt_record in enumerate(prompts, start=1):
            row = collect_one_prompt(
                torch=torch,
                model=model,
                tokenizer=tokenizer,
                router=router,
                prompt_record=prompt_record,
                step=step,
                args=args,
                config=config,
                input_device=input_device,
                installed=installed,
                gate_manager=gate_manager,
            )
            validate_rollout_row(row)
            rollout_rows.append(row)
            if stream_handle is not None:
                stream_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream_handle.flush()
            if args.progress_every and (step % args.progress_every == 0 or step == len(prompts)):
                print(
                    f"[opvec_collect_hf_rollouts] completed {step}/{len(prompts)} prompts; "
                    f"last_prompt={prompt_record.get('prompt_id')} task={prompt_record.get('task')}",
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        if stream_handle is not None:
            stream_handle.close()
    count = len(rollout_rows) if args.stream_output else write_jsonl(args.output, rollout_rows)
    kept = sum(1 for row in rollout_rows if row["keep_for_policy_loss"])
    summary = {
        "format": "opvec_gated_grpo_rollout_v1",
        "run_id": args.run_id,
        "output": str(args.output),
        "rows": count,
        "kept_frontiers": kept,
        "greedy": bool(args.greedy),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "tasks": {task: sum(1 for row in rollout_rows if row["task"] == task) for task in sorted({row["task"] for row in rollout_rows})},
        "installed_modules": installed,
        "gate_values": gate_manager.gate_values() if gate_manager is not None else {},
        "gate_parameterization": gate_parameterization,
        "device_map": args.device_map,
        "parameter_coefficients": 0 if param_names is None else len(param_names) * 3,
        "gate_checkpoint": args.gate_checkpoint,
        "policy_model": model_path,
        "disable_gates": bool(args.disable_gates),
        "skip_logprob": bool(args.skip_logprob),
        "store_token_logprobs": bool(args.store_token_logprobs),
        "stream_output": bool(args.stream_output),
        "reward_definition": {
            "main": "task_reward",
            "behavior_span_weight": float(
                args.behavior_span_reward_weight
                if args.behavior_span_reward_weight is not None
                else config.get("reward", {}).get("behavior_span_weight", 0.05)
            ),
        },
        "seed": seed,
    }
    write_json(Path(args.output).with_suffix(".summary.json"), summary)
    return summary


def collect_one_prompt(
    *,
    torch,
    model,
    tokenizer,
    router: RewardRouter,
    prompt_record: dict,
    step: int,
    args: argparse.Namespace,
    config: dict,
    input_device,
    installed: list[str],
    gate_manager,
) -> dict:
    trajectory = _memagent_trajectory(prompt_record)
    if trajectory is not None:
        return collect_one_memagent_trajectory(
            torch=torch,
            model=model,
            tokenizer=tokenizer,
            router=router,
            prompt_record=prompt_record,
            step=step,
            args=args,
            config=config,
            input_device=input_device,
            installed=installed,
            gate_manager=gate_manager,
            trajectory=trajectory,
        )
    prompt_text = render_chat_prompt(tokenizer, prompt_record.get("messages"), prompt_record.get("prompt", ""))
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=args.max_prompt_tokens).to(input_device)
    samples = []
    for sample_idx in range(args.samples_per_prompt):
        generation_kwargs = {
            **inputs,
            "max_new_tokens": args.max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if args.greedy:
            generation_kwargs["do_sample"] = False
        else:
            generation_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                }
            )
        with torch.no_grad():
            output_ids = model.generate(**generation_kwargs)
        response_ids = output_ids[0, inputs["input_ids"].shape[1] :]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
        scored = router.score(prompt_record, response_text)
        scored = _apply_behavior_span_mix(
            scored,
            behavior_span_weight=float(
                args.behavior_span_reward_weight
                if args.behavior_span_reward_weight is not None
                else config.get("reward", {}).get("behavior_span_weight", 0.05)
            ),
        )
        old_logprob = None
        token_payload = {}
        if not args.skip_logprob:
            old_logprob, token_payload = _old_logprob_payload(
                torch,
                model,
                tokenizer,
                prompt_text=prompt_text,
                response_text=response_text,
                args=args,
                input_device=input_device,
            )
        sample = {
            "sample_id": f"{prompt_record['prompt_id']}__k{sample_idx}",
            "text": response_text,
            "old_logprob": old_logprob,
            "old_logprob_max_length": None if args.skip_logprob else args.max_logprob_tokens,
            "length": len(response_ids),
            **scored,
        }
        sample.update(token_payload)
        samples.append(sample)
    keep, skip_reason, frontier = should_keep_frontier(
        samples,
        min_frontier_weight=float(config["frontier"]["min_frontier_weight"]),
        min_reward_std=float(config["frontier"]["min_reward_std"]),
    )
    gate_values = gate_manager.gate_values() if gate_manager is not None else {}
    return {
        "run_id": args.run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "step": step,
        "policy_id": "gated-grpo-current-policy",
        "gate_checkpoint": args.gate_checkpoint,
        "gate_values": gate_values,
        "gate_id": make_gate_id(gate_values, args.gate_checkpoint),
        "group_id": prompt_record.get("group_id") or prompt_record["prompt_id"],
        "seed": int(args.seed if args.seed is not None else config["run"]["seed"]),
        "installed_modules": installed,
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


def collect_one_memagent_trajectory(
    *,
    torch,
    model,
    tokenizer,
    router: RewardRouter,
    prompt_record: dict,
    step: int,
    args: argparse.Namespace,
    config: dict,
    input_device,
    installed: list[str],
    gate_manager,
    trajectory: dict,
) -> dict:
    """Collect MemAgent-style recurrent memory trajectory rollouts.

    Intermediate memory-update turns are generated by the current policy and
    fed into the next turn.  The final answer is scored with the official
    MemAgent HotpotQA boxed-answer verifier.  For policy updates, old_logprob
    is the sum over all update turns plus the final-answer turn.
    """

    question = str(trajectory["prompt"])
    chunks = [str(item) for item in trajectory["chunks"]]
    samples = []
    first_prompt_text = None
    for sample_idx in range(args.samples_per_prompt):
        memory_text = "No previous memory"
        turns = []
        old_logprob_total = 0.0
        has_logprob = not args.skip_logprob
        sample_token_ids = []
        sample_old_logprobs = []
        sample_response_mask = []
        total_length = 0
        for chunk_idx, chunk in enumerate(chunks, start=1):
            user_text = MEMAGENT_UPDATE_TEMPLATE.format(prompt=question, memory=memory_text, chunk=chunk)
            prompt_text = render_chat_prompt(tokenizer, [{"role": "user", "content": user_text}], user_text)
            first_prompt_text = first_prompt_text or prompt_text
            response_text, response_len = _generate_response_text(
                torch,
                model,
                tokenizer,
                prompt_text=prompt_text,
                args=args,
                input_device=input_device,
            )
            turn_logprob = None
            turn_token_payload = {}
            if not args.skip_logprob:
                turn_logprob, turn_token_payload = _old_logprob_payload(
                    torch,
                    model,
                    tokenizer,
                    prompt_text=prompt_text,
                    response_text=response_text,
                    args=args,
                    input_device=input_device,
                )
                if turn_logprob is None:
                    has_logprob = False
                else:
                    old_logprob_total += float(turn_logprob)
                    sample_token_ids.extend(turn_token_payload.get("response_token_ids", []))
                    sample_old_logprobs.extend(turn_token_payload.get("old_logprobs", []))
                    sample_response_mask.extend(turn_token_payload.get("response_mask", []))
            total_length += response_len
            turn = {
                "turn": chunk_idx,
                "kind": "memory_update",
                "prompt_text": prompt_text,
                "text": response_text,
                "old_logprob": turn_logprob,
                "length": response_len,
            }
            turn.update(turn_token_payload)
            turns.append(turn)
            memory_text = response_text
        user_text = MEMAGENT_FINAL_TEMPLATE.format(prompt=question, memory=memory_text)
        prompt_text = render_chat_prompt(tokenizer, [{"role": "user", "content": user_text}], user_text)
        first_prompt_text = first_prompt_text or prompt_text
        final_text, final_len = _generate_response_text(
            torch,
            model,
            tokenizer,
            prompt_text=prompt_text,
            args=args,
            input_device=input_device,
        )
        final_logprob = None
        final_token_payload = {}
        if not args.skip_logprob:
            final_logprob, final_token_payload = _old_logprob_payload(
                torch,
                model,
                tokenizer,
                prompt_text=prompt_text,
                response_text=final_text,
                args=args,
                input_device=input_device,
            )
            if final_logprob is None:
                has_logprob = False
            else:
                old_logprob_total += float(final_logprob)
                sample_token_ids.extend(final_token_payload.get("response_token_ids", []))
                sample_old_logprobs.extend(final_token_payload.get("old_logprobs", []))
                sample_response_mask.extend(final_token_payload.get("response_mask", []))
        total_length += final_len
        final_turn = {
            "turn": len(chunks) + 1,
            "kind": "final_answer",
            "prompt_text": prompt_text,
            "text": final_text,
            "old_logprob": final_logprob,
            "length": final_len,
        }
        final_turn.update(final_token_payload)
        turns.append(final_turn)
        scored = router.score(prompt_record, final_text)
        scored = _apply_behavior_span_mix(
            scored,
            behavior_span_weight=float(
                args.behavior_span_reward_weight
                if args.behavior_span_reward_weight is not None
                else config.get("reward", {}).get("behavior_span_weight", 0.0)
            ),
        )
        sample = {
            "sample_id": f"{prompt_record['prompt_id']}__k{sample_idx}",
            "text": final_text,
            "old_logprob": None if args.skip_logprob or not has_logprob else old_logprob_total,
            "old_logprob_max_length": None if args.skip_logprob else args.max_logprob_tokens,
            "length": total_length,
            "trajectory": turns,
            **scored,
        }
        if args.store_token_logprobs and has_logprob and sample_token_ids:
            sample.update(
                {
                    "response_token_ids": sample_token_ids,
                    "old_logprobs": sample_old_logprobs,
                    "response_mask": sample_response_mask,
                }
            )
        samples.append(sample)
    keep, skip_reason, frontier = should_keep_frontier(
        samples,
        min_frontier_weight=float(config["frontier"]["min_frontier_weight"]),
        min_reward_std=float(config["frontier"]["min_reward_std"]),
    )
    gate_values = gate_manager.gate_values() if gate_manager is not None else {}
    return {
        "run_id": args.run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "step": step,
        "policy_id": "gated-grpo-current-policy",
        "gate_checkpoint": args.gate_checkpoint,
        "gate_values": gate_values,
        "gate_id": make_gate_id(gate_values, args.gate_checkpoint),
        "group_id": prompt_record.get("group_id") or prompt_record["prompt_id"],
        "seed": int(args.seed if args.seed is not None else config["run"]["seed"]),
        "installed_modules": installed,
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


def _generate_response_text(torch, model, tokenizer, *, prompt_text: str, args: argparse.Namespace, input_device) -> tuple[str, int]:
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=args.max_prompt_tokens).to(input_device)
    generation_kwargs = {
        **inputs,
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.greedy:
        generation_kwargs["do_sample"] = False
    else:
        generation_kwargs.update({"do_sample": True, "temperature": args.temperature, "top_p": args.top_p})
    with torch.no_grad():
        output_ids = model.generate(**generation_kwargs)
    response_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(response_ids, skip_special_tokens=True), int(len(response_ids))


def _old_logprob_payload(
    torch,
    model,
    tokenizer,
    *,
    prompt_text: str,
    response_text: str,
    args: argparse.Namespace,
    input_device,
) -> tuple[float | None, dict]:
    if args.store_token_logprobs:
        details = response_logprob_details_from_text(
            torch,
            model,
            tokenizer,
            prompt_text=prompt_text,
            response_text=response_text,
            device=input_device,
            max_length=args.max_logprob_tokens,
        )
        if details is None:
            return None, {}
        return float(details["sum_logprob"]), {
            "response_token_ids": details["response_token_ids"],
            "old_logprobs": details["old_logprobs"],
            "response_mask": details["response_mask"],
        }
    logprob = response_logprob_from_text(
        torch,
        model,
        tokenizer,
        prompt_text=prompt_text,
        response_text=response_text,
        device=input_device,
        max_length=args.max_logprob_tokens,
    )
    return logprob, {}


def _memagent_trajectory(prompt_record: dict) -> dict | None:
    if prompt_record.get("task") != "memory":
        return None
    reference = prompt_record.get("reference", {}) if isinstance(prompt_record.get("reference"), dict) else {}
    metadata = reference.get("metadata", {}) if isinstance(reference.get("metadata"), dict) else {}
    chunks = metadata.get("memagent_chunks") or reference.get("memagent_chunks")
    prompt = metadata.get("memagent_prompt") or reference.get("memagent_prompt") or prompt_record.get("prompt")
    if isinstance(chunks, list) and chunks and prompt:
        return {"prompt": str(prompt), "chunks": [str(item) for item in chunks]}
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/gated_grpo.yaml")
    parser.add_argument("--seed-manifest", default=None)
    parser.add_argument("--mode-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id", default="hf-smoke")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-prompts", type=int, default=3)
    parser.add_argument("--tasks", default=None, help="Optional comma-separated task filter: tool,memory,code.")
    parser.add_argument("--memory-kind", default=None, help="Optional Memory phase filter: final_answer,memory_update.")
    parser.add_argument("--prompt-id", default=None, help="Optional comma-separated prompt_id filter.")
    parser.add_argument("--use-manifest-order", action="store_true", help="Use the manifest rows directly instead of task-balanced resampling.")
    parser.add_argument("--samples-per-prompt", type=int, default=2)
    parser.add_argument("--max-gated-modules", type=int, default=None, help="Maximum gated Linear modules to install. Default None means all mergeable modules; use 1 only for smoke tests.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--max-logprob-tokens", type=int, default=1536)
    parser.add_argument("--skip-logprob", action="store_true", help="Skip old-policy logprob scoring for reward-only evaluation rollouts.")
    parser.add_argument("--store-token-logprobs", action="store_true", help="Store response_token_ids, old_logprobs, and response_mask for token-level PPO/GRPO.")
    parser.add_argument("--stream-output", action="store_true", help="Write rollout rows as each prompt finishes.")
    parser.add_argument("--progress-every", type=int, default=0, help="Print progress every N prompts; 0 disables progress logs.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--greedy", action="store_true", help="Use deterministic greedy decoding instead of sampling.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default=None, help="Optional HF device_map, e.g. auto, for multi-GPU sharding.")
    parser.add_argument("--max-memory", action="append", default=[], help="HF max_memory entry, e.g. 0=70GiB. Repeatable.")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument(
        "--gate-parameterization",
        choices=["global", "layer-band", "layer-band-coefficient", "layer-band-parameter", "parameter", "global-parameter", "global-coefficient"],
        default="global",
    )
    parser.add_argument("--gate-checkpoint", default=None)
    parser.add_argument("--policy-model", default=None, help="Optional policy checkpoint to sample from instead of config models.base.")
    parser.add_argument("--disable-gates", action="store_true", help="Do not install OP-VEC gated modules; useful for baked policy checkpoints.")
    parser.add_argument("--behavior-span-reward-weight", type=float, default=None)
    return parser.parse_args()


def _normalize_gate_parameterization(value: str) -> str:
    aliases = {
        "layer_band": "layer-band",
        "layer_band_coefficient": "layer-band-coefficient",
        "layer-band-coefficients": "layer-band-coefficient",
        "layer_band_coefficients": "layer-band-coefficient",
        "layer-band-direct": "layer-band-coefficient",
        "layer_band_direct": "layer-band-coefficient",
        "layer_band_parameter": "layer-band-parameter",
        "layer-band-param": "layer-band-parameter",
        "layer_band_param": "layer-band-parameter",
        "layer-band-residual": "layer-band-parameter",
        "layer_band_residual": "layer-band-parameter",
        "layer-band-hierarchical": "layer-band-parameter",
        "layer_band_hierarchical": "layer-band-parameter",
        "hierarchical-layer-band": "layer-band-parameter",
        "hierarchical_layer_band": "layer-band-parameter",
        "param": "parameter",
        "param-coefficients": "parameter",
        "parameter-coefficients": "parameter",
        "global_parameter": "global-parameter",
        "global-param": "global-parameter",
        "global_param": "global-parameter",
        "global-residual": "global-parameter",
        "global_residual": "global-parameter",
        "global_coefficient": "global-coefficient",
        "global-coefficients": "global-coefficient",
        "global_coefficients": "global-coefficient",
        "global-direct": "global-coefficient",
        "global_direct": "global-coefficient",
        "expert-coefficient": "global-coefficient",
        "expert_coefficient": "global-coefficient",
    }
    return aliases.get(str(value), str(value))


def _uses_parameter_names(gate_parameterization: str) -> bool:
    return gate_parameterization in {"parameter", "global-parameter"}


def _apply_behavior_span_mix(scored: dict, *, behavior_span_weight: float) -> dict:
    weight = max(0.0, min(float(behavior_span_weight), 0.25))
    train_reward = _clamp01(float(scored.get("reward_train", scored.get("task_reward", scored.get("reward", 0.0)))))
    if weight <= 0.0:
        details = dict(scored.get("details", {}))
        details["reward_definition"] = details.get("reward_definition", "official_task_reward")
        details["behavior_span_weight"] = 0.0
        details["behavior_span_reward"] = _behavior_span_reward(scored)
        return {
            **scored,
            "reward": float(scored.get("task_reward", scored.get("reward", 0.0))),
            "task_reward": float(scored.get("task_reward", scored.get("reward", 0.0))),
            "reward_train": train_reward,
            "behavior_span_reward": details["behavior_span_reward"],
            "details": details,
        }
    span_reward = _behavior_span_reward(scored)
    mixed = (1.0 - weight) * train_reward + weight * span_reward
    details = dict(scored.get("details", {}))
    details["reward_definition"] = "reward_train=(1-w)*normalized_task_reward+w*behavior_span_reward"
    details["behavior_span_weight"] = weight
    details["behavior_span_reward"] = span_reward
    return {
        **scored,
        "reward": float(mixed),
        "task_reward": float(scored.get("task_reward", scored.get("reward", 0.0))),
        "reward_train": float(mixed),
        "behavior_span_reward": span_reward,
        "details": details,
    }


def _behavior_span_reward(scored: dict) -> float:
    details = scored.get("details", {}) if isinstance(scored.get("details"), dict) else {}
    for key in ("name_recall", "token_f1"):
        if key in details:
            return _clamp01(float(details[key]))
    if "syntax_ok" in details:
        return 1.0 if details.get("syntax_ok") else 0.0
    if "parseable" in details:
        return 1.0 if details.get("parseable") else 0.0
    return _clamp01(float(scored.get("contract_reward", 0.0)) * 10.0)


def _clamp01(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


if __name__ == "__main__":
    main()
