"""MemAgent trajectory rollout loop for OP-VEC verl training."""

from __future__ import annotations

import json
import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

from .path_utils import ensure_opvec_on_path

ensure_opvec_on_path()

from opvec.rewards.router import RewardRouter  # noqa: E402

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

MEMAGENT_UPDATE_TEMPLATE = """You are presented with a problem, a section of an article that may contain the answer to the problem, and a previous memory. Please read the provided section carefully and update the memory with the new information that helps to answer the problem. Be sure to retain all relevant details from the previous memory while adding any new, useful information.

Problem:
{prompt}

Previous memory:
<memory>
{memory}
</memory>

Section:
{chunk}

Updated memory:"""

MEMAGENT_FINAL_TEMPLATE = """You are presented with a problem and a previous memory. Please answer the problem based on the previous memory and put the answer in \\boxed{{}}.

Problem:
{prompt}

Previous memory:
<memory>
{memory}
</memory>

Answer:"""


@register("opvec_memagent")
class OpVecMemAgentAgentLoop(AgentLoopBase):
    """Run MemAgent's recurrent memory-update rollout inside verl.

    The training scope follows the official recurrent MemAgent objective:
    every memory-update turn and the final-answer turn is returned as an
    independent prompt-response row, while the final answer reward is shared by
    all rows in the same rollout.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = int(self.rollout_config.prompt_length)
        self.response_length = int(self.rollout_config.response_length)
        self.max_model_len = int(self.rollout_config.max_model_len)
        self.update_max_tokens = _env_int("OPVEC_MEMAGENT_UPDATE_MAX_TOKENS", 1024)
        self.final_max_tokens = _env_int("OPVEC_MEMAGENT_FINAL_MAX_TOKENS", 1024)
        self.truncation_side = os.environ.get("OPVEC_MEMAGENT_TRUNCATION_SIDE", "right")
        self.router = RewardRouter()

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput | list[AgentLoopOutput]:
        trajectory = _extract_memagent_trajectory(kwargs)
        if trajectory is None:
            return await self._run_single_turn_fallback(sampling_params, **kwargs)

        question = trajectory["prompt"]
        chunks = trajectory["chunks"]
        metrics: dict[str, Any] = {}
        request_id = uuid4().hex
        turn_records: list[dict[str, Any]] = []
        update_max_tokens = min(self.update_max_tokens, self.response_length)
        final_max_tokens = min(self.final_max_tokens, self.response_length)

        memory_text = "No previous memory"
        for chunk_idx, chunk in enumerate(chunks, start=1):
            user_text = MEMAGENT_UPDATE_TEMPLATE.format(prompt=question, memory=memory_text, chunk=chunk)
            prompt_ids = await self._prompt_ids(user_text, max_new_tokens=update_max_tokens)
            output = await self._generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=_turn_sampling_params(sampling_params, max_tokens=update_max_tokens),
                metrics=metrics,
            )
            update_ids = output.token_ids[: self.response_length]
            update_text = await self._decode(update_ids)
            turn_records.append(
                {
                    "turn_index": chunk_idx,
                    "kind": "memory_update",
                    "prompt_ids": prompt_ids,
                    "response_ids": update_ids,
                    "response_logprobs": output.log_probs[: len(update_ids)] if output.log_probs else None,
                    "routed_experts": _slice_routed_experts(output.routed_experts, len(prompt_ids), len(update_ids)),
                    "prompt_tokens": len(prompt_ids),
                    "length": len(update_ids),
                    "text": update_text,
                }
            )
            memory_text = update_text

        final_user_text = MEMAGENT_FINAL_TEMPLATE.format(prompt=question, memory=memory_text)
        final_prompt_ids = await self._prompt_ids(final_user_text, max_new_tokens=final_max_tokens)
        final_output = await self._generate(
            request_id=request_id,
            prompt_ids=final_prompt_ids,
            sampling_params=_turn_sampling_params(sampling_params, max_tokens=final_max_tokens),
            metrics=metrics,
        )

        final_ids = final_output.token_ids[: self.response_length]
        final_text = await self._decode(final_ids)
        turn_records.append(
            {
                "turn_index": len(chunks) + 1,
                "kind": "final_answer",
                "prompt_ids": final_prompt_ids,
                "response_ids": final_ids,
                "response_logprobs": final_output.log_probs[: len(final_ids)] if final_output.log_probs else None,
                "routed_experts": _slice_routed_experts(
                    final_output.routed_experts, len(final_prompt_ids), len(final_ids)
                ),
                "prompt_tokens": len(final_prompt_ids),
                "length": len(final_ids),
                "text": final_text,
            }
        )

        prompt_record = _prompt_record_from_kwargs(kwargs)
        scored = self.router.score(prompt_record, final_text)
        reward = float(scored.get("reward", scored.get("task_reward", 0.0)))
        public_turns = [
            {
                "turn": item["turn_index"],
                "kind": item["kind"],
                "prompt_tokens": item["prompt_tokens"],
                "length": item["length"],
                "text": item["text"],
            }
            for item in turn_records
        ]

        outputs: list[AgentLoopOutput] = []
        num_turns = len(turn_records)
        for item in turn_records:
            is_final_turn = item["kind"] == "final_answer"
            extra_fields: dict[str, Any] = {
                "turn_scores": [],
                "tool_rewards": [],
                "memagent_final_text": final_text,
                "memagent_num_chunks": len(chunks),
                "memagent_num_turns": num_turns,
                "memagent_training_scope": "trajectory_all_turns",
                "memagent_reward": reward,
                "memagent_turns": public_turns,
                "memagent_turn_kind": item["kind"],
                "memagent_turn_index": item["turn_index"],
                "memagent_final_mask": is_final_turn,
                "memagent_rollout_key": request_id,
                "memagent_truncated": False,
            }
            if is_final_turn:
                extra_fields.update(final_output.extra_fields or {})
            outputs.append(
                AgentLoopOutput(
                    prompt_ids=item["prompt_ids"],
                    response_ids=item["response_ids"],
                    response_mask=[1] * len(item["response_ids"]),
                    response_logprobs=item["response_logprobs"],
                    routed_experts=item["routed_experts"],
                    multi_modal_data={},
                    reward_score=reward,
                    num_turns=num_turns,
                    metrics=metrics if is_final_turn else _empty_metrics(),
                    extra_fields=extra_fields,
                )
            )

        return outputs

    async def _run_single_turn_fallback(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        prompt_ids = await self.apply_chat_template(messages)
        final_max_tokens = min(self.final_max_tokens, self.response_length)
        prompt_ids = self._truncate_prompt(prompt_ids, max_new_tokens=final_max_tokens)
        metrics: dict[str, Any] = {}
        output = await self._generate(
            request_id=uuid4().hex,
            prompt_ids=prompt_ids,
            sampling_params=_turn_sampling_params(sampling_params, max_tokens=final_max_tokens),
            metrics=metrics,
        )
        response_ids = output.token_ids[: self.response_length]
        response_mask = [1] * len(response_ids)
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=output.log_probs[: len(response_ids)] if output.log_probs else None,
            routed_experts=(
                output.routed_experts[: len(prompt_ids) + self.response_length]
                if output.routed_experts is not None
                else None
            ),
            multi_modal_data={},
            num_turns=2,
            metrics=metrics,
            extra_fields={"turn_scores": [], "tool_rewards": []},
        )

    async def _prompt_ids(self, user_text: str, *, max_new_tokens: int) -> list[int]:
        prompt_ids = await self.apply_chat_template([{"role": "user", "content": user_text}])
        return self._truncate_prompt(prompt_ids, max_new_tokens=max_new_tokens)

    def _truncate_prompt(self, prompt_ids: list[int], *, max_new_tokens: int) -> list[int]:
        max_prompt = min(self.prompt_length, max(1, self.max_model_len - int(max_new_tokens)))
        if len(prompt_ids) <= max_prompt:
            return prompt_ids
        if self.truncation_side == "left":
            return prompt_ids[-max_prompt:]
        return prompt_ids[:max_prompt]

    async def _generate(
        self,
        request_id: str,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        metrics: dict[str, Any],
    ) -> TokenOutput:
        turn_metrics: dict[str, Any] = {}
        with simple_timer("generate_sequences", turn_metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=dict(sampling_params),
            )
        metrics["generate_sequences"] = metrics.get("generate_sequences", 0.0) + turn_metrics.get(
            "generate_sequences", 0.0
        )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        elif output.num_preempted is not None and metrics["num_preempted"] != -1:
            metrics["num_preempted"] += output.num_preempted
        return output

    async def _decode(self, token_ids: list[int]) -> str:
        return await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(token_ids, skip_special_tokens=True),
        )


def _extract_memagent_trajectory(kwargs: dict[str, Any]) -> dict[str, Any] | None:
    prompt_record = _prompt_record_from_kwargs(kwargs)
    if prompt_record.get("task") != "memory":
        return None
    reference = prompt_record.get("reference", {}) or {}
    metadata = reference.get("metadata", {}) if isinstance(reference, dict) else {}
    chunks = metadata.get("memagent_chunks") or reference.get("memagent_chunks")
    question = metadata.get("memagent_prompt") or reference.get("memagent_prompt") or prompt_record.get("prompt")
    if not question or not isinstance(chunks, list) or not chunks:
        return None
    return {"prompt": str(question), "chunks": [str(item) for item in chunks]}


def _prompt_record_from_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    extra_info = kwargs.get("extra_info") or {}
    prompt_record = dict(extra_info)
    reference = prompt_record.get("reference")
    if reference is None and prompt_record.get("reference_json"):
        reference = _loads_json(prompt_record.get("reference_json"), default={})
    prompt_record["reference"] = reference or {}
    prompt_record.setdefault("task", prompt_record.get("ability") or kwargs.get("ability") or "memory")
    return prompt_record


def _turn_sampling_params(sampling_params: dict[str, Any], *, max_tokens: int) -> dict[str, Any]:
    params = dict(sampling_params)
    params["max_tokens"] = max(0, int(max_tokens))
    return params


def _slice_routed_experts(routed_experts: Any, prompt_length: int, response_length: int) -> Any:
    if routed_experts is None:
        return None
    return routed_experts[: prompt_length + response_length]


def _empty_metrics() -> dict[str, Any]:
    return {
        "generate_sequences": 0.0,
        "tool_calls": 0.0,
        "compute_score": 0.0,
        "num_preempted": -1,
    }


def _loads_json(value: Any, *, default: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid integer for %s=%r; using %s", name, value, default)
        return default
