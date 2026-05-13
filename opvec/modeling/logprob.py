"""Sequence log-probability helpers."""

from __future__ import annotations

from typing import Any


def response_logprob_from_text(
    torch: Any,
    model: Any,
    tokenizer: Any,
    *,
    prompt_text: str,
    response_text: str,
    device: str,
    max_length: int = 4096,
) -> float | None:
    """Return summed log p(response | prompt) for one text pair."""

    details = response_logprob_details_from_text(
        torch,
        model,
        tokenizer,
        prompt_text=prompt_text,
        response_text=response_text,
        device=device,
        max_length=max_length,
    )
    if details is None:
        return None
    return float(details["sum_logprob"])


def response_logprob_tensor_from_text(
    torch: Any,
    model: Any,
    tokenizer: Any,
    *,
    prompt_text: str,
    response_text: str,
    device: str,
    max_length: int = 4096,
):
    """Differentiable summed log p(response | prompt)."""

    details = response_logprob_tensor_details_from_text(
        torch,
        model,
        tokenizer,
        prompt_text=prompt_text,
        response_text=response_text,
        device=device,
        max_length=max_length,
    )
    if details is None:
        return None
    return details["logprobs"].sum()


def response_logprob_details_from_text(
    torch: Any,
    model: Any,
    tokenizer: Any,
    *,
    prompt_text: str,
    response_text: str,
    device: str,
    max_length: int = 4096,
) -> dict[str, Any] | None:
    """Return token-level old-policy logprobs for JSON rollout storage."""

    with torch.no_grad():
        details = response_logprob_tensor_details_from_text(
            torch,
            model,
            tokenizer,
            prompt_text=prompt_text,
            response_text=response_text,
            device=device,
            max_length=max_length,
        )
    if details is None:
        return None
    token_logprobs = details["logprobs"].detach().float().cpu().tolist()
    return {
        "response_token_ids": details["response_token_ids"],
        "old_logprobs": [float(value) for value in token_logprobs],
        "response_mask": details["response_mask"],
        "sum_logprob": float(sum(token_logprobs)),
    }


def response_logprob_tensor_details_from_text(
    torch: Any,
    model: Any,
    tokenizer: Any,
    *,
    prompt_text: str,
    response_text: str,
    device: str,
    max_length: int = 4096,
) -> dict[str, Any] | None:
    """Differentiable per-token log p(response | prompt)."""

    window = _response_token_window(tokenizer, prompt_text=prompt_text, response_text=response_text, max_length=max_length)
    if window is None:
        return None
    input_ids = window["input_ids"]
    logit_positions = window["logit_positions"]
    target_ids = window["target_ids"]
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    position_tensor = torch.tensor(logit_positions, dtype=torch.long, device=device)
    target_tensor = torch.tensor(target_ids, dtype=torch.long, device=device)
    masked_logits = _selected_logits(torch, model, ids, position_tensor)
    log_probs = torch.nn.functional.log_softmax(masked_logits, dim=-1)
    token_logprobs = log_probs.gather(-1, target_tensor.to(log_probs.device).unsqueeze(-1)).squeeze(-1)
    return {
        "response_token_ids": target_ids,
        "logprobs": token_logprobs,
        "response_mask": [1 for _ in target_ids],
    }


def response_logprob_tensor_details_from_token_ids(
    torch: Any,
    model: Any,
    tokenizer: Any,
    *,
    prompt_text: str,
    response_token_ids: list[int],
    device: str,
    max_length: int = 4096,
):
    """Differentiable per-token log p(response_token_ids | prompt).

    This is used for vLLM rollouts, where the behavior policy can return the
    exact sampled token ids and old logprobs. Reusing those token ids avoids
    detokenize/re-tokenize drift between vLLM rollout storage and HF update
    scoring.
    """

    window = _response_token_window_from_ids(
        tokenizer,
        prompt_text=prompt_text,
        response_token_ids=response_token_ids,
        max_length=max_length,
    )
    if window is None:
        return None
    input_ids = window["input_ids"]
    logit_positions = window["logit_positions"]
    target_ids = window["target_ids"]
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    position_tensor = torch.tensor(logit_positions, dtype=torch.long, device=device)
    target_tensor = torch.tensor(target_ids, dtype=torch.long, device=device)
    masked_logits = _selected_logits(torch, model, ids, position_tensor)
    log_probs = torch.nn.functional.log_softmax(masked_logits, dim=-1)
    token_logprobs = log_probs.gather(-1, target_tensor.to(log_probs.device).unsqueeze(-1)).squeeze(-1)
    return {
        "response_token_ids": target_ids,
        "logprobs": token_logprobs,
        "response_mask": [1 for _ in target_ids],
    }


def _response_token_window(tokenizer: Any, *, prompt_text: str, response_text: str, max_length: int) -> dict[str, Any] | None:
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    response_ids = tokenizer(response_text, add_special_tokens=False).input_ids
    return _response_token_window_from_ids(
        tokenizer,
        prompt_text=prompt_text,
        response_token_ids=response_ids,
        max_length=max_length,
    )


def _response_token_window_from_ids(
    tokenizer: Any,
    *,
    prompt_text: str,
    response_token_ids: list[int],
    max_length: int,
) -> dict[str, Any] | None:
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    response_ids = [int(token_id) for token_id in response_token_ids]
    if not response_ids:
        return None
    if len(response_ids) >= max_length:
        return None
    input_ids = prompt_ids + response_ids
    if len(input_ids) > max_length:
        overflow = len(input_ids) - max_length
        prompt_ids = prompt_ids[overflow:]
        input_ids = prompt_ids + response_ids
    logit_positions, target_ids = _target_logit_positions(prompt_ids, response_ids)
    if len(input_ids) < 2 or not target_ids:
        return None
    return {
        "input_ids": input_ids,
        "logit_positions": logit_positions,
        "target_ids": target_ids,
    }


def _target_logit_positions(prompt_ids: list[int], response_ids: list[int]) -> tuple[list[int], list[int]]:
    """Return causal-logit positions and token ids matching the old shifted-label scoring."""

    labels = [-100] * len(prompt_ids) + response_ids
    logit_positions = []
    target_ids = []
    for label_position, token_id in enumerate(labels[1:], start=1):
        if token_id == -100:
            continue
        logit_positions.append(label_position - 1)
        target_ids.append(int(token_id))
    return logit_positions, target_ids


def _selected_logits(torch: Any, model: Any, ids: Any, position_tensor: Any):
    try:
        # With HF device_map sharding, Qwen indexes hidden states before the
        # lm_head. CPU indices are accepted by PyTorch regardless of which GPU
        # currently owns the hidden states; moving the index tensor to the
        # output head device can be wrong when the final block lives elsewhere.
        logits_to_keep = [int(item) for item in position_tensor.detach().cpu().tolist()]
        logits = model(input_ids=ids, use_cache=False, logits_to_keep=logits_to_keep).logits
        return logits[0].float()
    except (TypeError, RuntimeError) as error:
        if isinstance(error, RuntimeError) and "indices should" not in str(error):
            raise
        logits = model(input_ids=ids, use_cache=False).logits
        return logits[:, position_tensor.to(logits.device), :][0].float()


def render_chat_prompt(tokenizer: Any, messages: list[dict[str, str]] | None, fallback_prompt: str) -> str:
    """Render messages using a tokenizer chat template when available."""

    if messages:
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            return "\n".join(f"{item.get('role', 'user')}: {item.get('content', '')}" for item in messages)
    return fallback_prompt
