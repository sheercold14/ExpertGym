"""BFCL AST verifier adapter for Tool live/control rollouts."""

from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .base import RewardResult


DEFAULT_BFCL_ROOT = Path("/mnt/cache/wuruixiao/users/lsc/gorilla/berkeley-function-call-leaderboard")


class BFCLToolRewardAdapter:
    name = "bfcl_ast"

    def score(self, prompt_record: dict[str, Any], output_text: str) -> RewardResult:
        bfcl = _bfcl_reference(prompt_record)
        try:
            verifier = _load_bfcl_verifier()
            decoded = verifier["decode"](
                output_text,
                verifier["ReturnFormat"].PYTHON,
                bool(bfcl.get("has_tool_call_tag", False)),
            )
        except Exception as error:  # noqa: BLE001 - verifier errors are reward signal.
            return RewardResult(
                reward=0.05 if output_text.strip() else 0.0,
                task_reward=0.0,
                contract_reward=0.0,
                success=False,
                details={
                    "bfcl_category": bfcl.get("category"),
                    "bfcl_id": bfcl.get("id"),
                    "decode_error": str(error),
                    "parseable": False,
                },
            )

        if not verifier["is_format"](decoded):
            return RewardResult(
                reward=0.10,
                task_reward=0.0,
                contract_reward=0.05,
                success=False,
                details={
                    "bfcl_category": bfcl.get("category"),
                    "bfcl_id": bfcl.get("id"),
                    "decoded_calls": len(decoded) if isinstance(decoded, list) else 0,
                    "parseable": True,
                    "wrong_output_format": True,
                },
            )

        model_name = str(bfcl.get("model_name") or "opvec-bfcl-offline")
        _ensure_bfcl_model_config(verifier["MODEL_CONFIG_MAPPING"], model_name)
        functions = bfcl["function"]
        possible_answer = bfcl["possible_answer"]
        checker = verifier["ast_checker"](
            functions,
            decoded,
            possible_answer,
            verifier["Language"].PYTHON,
            str(bfcl.get("category") or "live_parallel"),
            model_name,
        )
        full_success = bool(checker.get("valid"))
        matched_calls = _count_partial_matches(
            verifier=verifier,
            functions=functions,
            decoded=decoded,
            possible_answer=possible_answer,
            model_name=model_name,
        )
        expected_calls = len(possible_answer)
        partial_score = matched_calls / max(1, expected_calls)
        name_score = _name_recall(decoded, possible_answer)
        count_score = 1.0 if len(decoded) == expected_calls else 0.0
        if full_success:
            reward = 1.0
        else:
            reward = min(0.90, 0.20 + 0.50 * partial_score + 0.15 * name_score + 0.05 * count_score)
        return RewardResult(
            reward=reward,
            task_reward=reward,
            contract_reward=0.10 if decoded else 0.05,
            success=full_success,
            details={
                "bfcl_category": bfcl.get("category"),
                "bfcl_id": bfcl.get("id"),
                "bfcl_error_type": checker.get("error_type"),
                "decoded_calls": len(decoded),
                "expected_calls": expected_calls,
                "matched_calls": matched_calls,
                "name_recall": name_score,
                "parseable": True,
            },
        )


def _bfcl_reference(prompt_record: dict[str, Any]) -> dict[str, Any]:
    reference = prompt_record.get("reference", {})
    bfcl = reference.get("bfcl", {}) if isinstance(reference, dict) else {}
    if not bfcl:
        raise ValueError(f"Prompt record is missing reference.bfcl: {prompt_record.get('prompt_id')}")
    return bfcl


def _load_bfcl_verifier() -> dict[str, Any]:
    root = Path(os.environ.get("BFCL_ROOT", str(DEFAULT_BFCL_ROOT)))
    if not root.exists():
        raise RuntimeError(f"BFCL_ROOT does not exist: {root}")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    from bfcl_eval.constants.enums import Language, ReturnFormat
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
    from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker, find_description, simple_function_checker
    from bfcl_eval.model_handler.utils import default_decode_ast_prompting
    from bfcl_eval.utils import is_function_calling_format_output

    return {
        "Language": Language,
        "MODEL_CONFIG_MAPPING": MODEL_CONFIG_MAPPING,
        "ReturnFormat": ReturnFormat,
        "ast_checker": ast_checker,
        "decode": default_decode_ast_prompting,
        "find_description": find_description,
        "is_format": is_function_calling_format_output,
        "simple_function_checker": simple_function_checker,
    }


def _ensure_bfcl_model_config(mapping: dict[str, Any], model_name: str) -> None:
    key = model_name.replace("_", "/")
    if key not in mapping:
        mapping[key] = SimpleNamespace(underscore_to_dot=False)


def _count_partial_matches(
    *,
    verifier: dict[str, Any],
    functions: list[dict[str, Any]],
    decoded: list[dict[str, Any]],
    possible_answer: list[dict[str, Any]],
    model_name: str,
) -> int:
    matched_output_indices: set[int] = set()
    matched = 0
    for answer in possible_answer:
        func_name = next(iter(answer.keys()))
        func_description = verifier["find_description"](functions, func_name)
        for index, output in enumerate(decoded):
            if index in matched_output_indices:
                continue
            result = verifier["simple_function_checker"](
                func_description,
                output,
                answer,
                verifier["Language"].PYTHON,
                model_name,
            )
            if result.get("valid"):
                matched_output_indices.add(index)
                matched += 1
                break
    return matched


def _name_recall(decoded: list[dict[str, Any]], possible_answer: list[dict[str, Any]]) -> float:
    expected = Counter(next(iter(item.keys())) for item in possible_answer)
    predicted = Counter(next(iter(item.keys())) for item in decoded if isinstance(item, dict) and item)
    if not expected:
        return 0.0
    overlap = sum((expected & predicted).values())
    return overlap / sum(expected.values())
