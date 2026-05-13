"""Official-aligned verifier adapters used by OP-VEC rollouts.

The adapters intentionally mirror the reward definitions used by the source
expert training repos:

* ToolRL: ``verl/utils/reward_score/rlla.py``.
* MemAgent: ``verl/utils/reward_score/hotpotqa.py``.
* CURE: code reward is the ground-truth test pass rate before group
  normalization, matching CURE's code-side reward signal.
"""

from __future__ import annotations

import ast
import os
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

from opvec.data.prompt_filters import memory_prompt_kind

from .base import RewardResult


class ToolRewardAdapter:
    name = "toolrl_source_reward"

    def score(self, prompt_record: dict[str, Any], output_text: str) -> RewardResult:
        reference = str(prompt_record.get("reference", {}).get("response") or "")
        format_score = 1.0 if _toolrl_format_ok(output_text, reference) else 0.0
        ref_calls = _extract_tool_payloads_strict(reference)
        pred_calls = _extract_tool_payloads_strict(output_text)
        parseable = bool(pred_calls)
        has_reference_tool = "<tool_call>" in reference
        if has_reference_tool:
            correctness_raw, correctness_details = _toolrl_correctness_raw(reference, output_text)
            score = format_score + correctness_raw
            success = bool(format_score == 1.0 and correctness_details["exact_tool_match"])
        else:
            correctness_raw = 0.0
            correctness_details = {"exact_tool_match": False}
            score = format_score
            success = bool(format_score == 1.0)
        name_score = _match_score(_tool_names(ref_calls), _tool_names(pred_calls)) if ref_calls else 0.0
        return RewardResult(
            reward=score,
            task_reward=score,
            reward_train=_tool_train_reward(score),
            contract_reward=0.1 if parseable else 0.0,
            success=success,
            details={
                "reward_source": "ToolRL/verl/utils/reward_score/rlla.py",
                "reward_definition": "format_reward + tool_call_correctness_reward",
                "official_repo": "https://github.com/qiancheng0/ToolRL",
                "format_score": format_score,
                "toolrl_correctness_raw": correctness_raw,
                "toolrl_raw_total": score,
                "toolrl_score_range": [-3.0, 4.0],
                "reward_train_definition": "clip((toolrl_raw_total + 3) / 7, 0, 1)",
                "parseable": parseable,
                "prediction_calls": len(pred_calls),
                "reference_calls": len(ref_calls),
                "name_recall": name_score,
                **correctness_details,
            },
        )


class MemoryRewardAdapter:
    name = "memagent_source_reward"

    def score(self, prompt_record: dict[str, Any], output_text: str) -> RewardResult:
        prompt_kind = memory_prompt_kind(prompt_record)
        if prompt_kind == "final_answer":
            return _score_memory_final_answer(prompt_record, output_text)
        return _score_memory_update(prompt_record, output_text, prompt_kind=prompt_kind)


def _score_memory_final_answer(prompt_record: dict[str, Any], output_text: str) -> RewardResult:
    references = _memagent_ground_truths(prompt_record)
    reference = references[0] if references else ""
    answer_region = _truncate_memory_final_answer(output_text)
    prediction = _extract_memory_prediction(answer_region)
    score = _memagent_compute_score(answer_region, references)
    exact = any(_normalize_text(prediction) == _normalize_text(item) for item in references) if references else False
    sub_exact = max((_sub_exact_match(prediction, item) for item in references), default=False)
    f1 = max((_token_f1(prediction, item) for item in references), default=0.0)
    boxed = _extract_boxed_answer(answer_region) is not None
    return RewardResult(
        reward=score,
        task_reward=score,
        contract_reward=(0.05 if output_text.strip() else 0.0) + (0.05 if boxed else 0.0),
        success=bool(score >= 1.0),
        details={
            "reward_source": "MemAgent/verl/utils/reward_score/hotpotqa.py",
            "reward_definition": "compute_score(solution[-300:], ground_truth_list)",
            "official_repo": "https://github.com/BytedTsinghua-SIA/MemAgent",
            "boxed_found": boxed,
            "exact_match": exact,
            "training_exact_match": bool(score >= 1.0),
            "memory_prompt_kind": "final_answer",
            "prediction": prediction,
            "ground_truths": references,
            "sub_exact_match": sub_exact,
            "token_f1": f1,
            "compression_factor": _compression_factor(prediction, reference),
            "truncated_for_scoring": answer_region != (output_text or ""),
        },
    )


def _score_memory_update(prompt_record: dict[str, Any], output_text: str, *, prompt_kind: str) -> RewardResult:
    reference = str(
        prompt_record.get("reference", {}).get("response")
        or prompt_record.get("reference", {}).get("answer")
        or ""
    )
    exact = _normalize_text(output_text) == _normalize_text(reference) if reference else False
    sub_exact = _sub_exact_match(output_text, reference) if reference else False
    f1 = _token_f1(output_text, reference) if reference else 0.0
    score = 0.0
    return RewardResult(
        reward=score,
        task_reward=score,
        contract_reward=f1,
        success=False,
        details={
            "reward_source": "MemAgent recurrent workflow",
            "reward_definition": "memory update turns are intermediate state; final boxed answer receives HotpotQA reward",
            "official_repo": "https://github.com/BytedTsinghua-SIA/MemAgent",
            "exact_match": exact,
            "memory_prompt_kind": prompt_kind,
            "sub_exact_match": sub_exact,
            "token_f1": f1,
        },
    )


class CodeRewardAdapter:
    name = "cure_code_pass_rate"

    def score(self, prompt_record: dict[str, Any], output_text: str) -> RewardResult:
        code = _extract_code(output_text).strip()
        syntax_ok = False
        parse_error = None
        input_used = False
        output_used = False
        try:
            if code:
                tree = ast.parse(code)
                syntax_ok = True
                input_used = _uses_input(tree)
                output_used = _uses_output(tree)
        except SyntaxError as error:
            parse_error = {"lineno": error.lineno, "msg": error.msg}
        score = 0.05 if code else 0.0
        if syntax_ok:
            score += 0.20
        if input_used:
            score += 0.10
        if output_used:
            score += 0.10
        reference = str(prompt_record.get("reference", {}).get("response") or "")
        prompt_text = str(prompt_record.get("prompt") or "")
        examples = _extract_code_examples(prompt_text)
        source_tests = _code_source_tests(prompt_record)
        source_test_result = None
        example_result = None
        if syntax_ok and source_tests:
            source_test_result = _run_source_code_tests(code, source_tests)
            score = source_test_result["pass_rate"]
        elif syntax_ok and examples:
            example_result = _run_public_examples(code, examples)
            pass_rate = example_result["pass_rate"]
            if input_used and output_used:
                score += 0.50 * pass_rate
                if pass_rate >= 1.0:
                    score = max(score, 0.95)
            else:
                score = min(score + 0.15 * pass_rate, 0.55)
        elif syntax_ok and reference:
            score += 0.20 * _reference_call_overlap(code, _extract_code(reference))
        if reference and _normalize_code(code) == _normalize_code(_extract_code(reference)):
            score = 1.0
        success = score >= 0.95
        return RewardResult(
            reward=min(score, 1.0),
            task_reward=min(score, 1.0),
            contract_reward=0.1 if syntax_ok else 0.0,
            success=success,
            details={
                "syntax_ok": syntax_ok,
                "parse_error": parse_error,
                "code_present": bool(code),
                "input_used": input_used,
                "output_used": output_used,
                "reward_source": "CURE/optimization/reward.py",
                "reward_definition": "ground-truth-test pass rate; GRPO normalizes within prompt group like CURE normalize_reward",
                "official_repo": "https://github.com/Gen-Verse/CURE",
                "source_tests": source_test_result,
                "public_examples": example_result,
            },
        )


def adapter_for_task(task: str):
    if task == "tool":
        return ToolRewardAdapter()
    if task == "memory":
        return MemoryRewardAdapter()
    if task == "code":
        return CodeRewardAdapter()
    raise ValueError(f"Unknown task: {task}")


def _tool_train_reward(raw_score: float) -> float:
    return max(0.0, min((float(raw_score) + 3.0) / 7.0, 1.0))


def _extract_tool_payloads_strict(text: str) -> list[dict[str, Any]]:
    payloads = []
    if "<tool_call>" not in (text or "") or "</tool_call>" not in (text or ""):
        return payloads
    candidate = str(text).split("<tool_call>")[1].split("</tool_call>")[0].strip()
    for line in candidate.split("\n"):
        stripped = line.strip()
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            return []
        if not isinstance(payload, dict):
            return []
        payloads.append(payload)
    return payloads


def _toolrl_format_ok(response: str, answer: str) -> bool:
    candidate = (response or "").strip()
    if "<response>" in answer and "<tool_call>" not in answer:
        pattern = r"^<think>.*?</think>\n<response>.*?</response>$"
        return (
            re.search(pattern, candidate, re.DOTALL) is not None
            and candidate.count("<response>") == 1
            and candidate.count("</response>") == 1
            and "<tool_call>" not in candidate
        )
    if "<response>" not in answer and "<tool_call>" in answer:
        pattern = r"^<think>.*?</think>\n<tool_call>\n.*?\n</tool_call>$"
        return (
            re.search(pattern, candidate, re.DOTALL) is not None
            and candidate.count("<tool_call>") == 1
            and candidate.count("</tool_call>") == 1
        )
    if "<response>" in answer and "<tool_call>" in answer:
        pattern = r"^<think>.*?</think>\n<tool_call>\n.*?\n</tool_call>\n<response>.*?</response>$"
        return (
            re.search(pattern, candidate, re.DOTALL) is not None
            and candidate.count("<tool_call>") == 1
            and candidate.count("</tool_call>") == 1
            and candidate.count("<response>") == 1
            and candidate.count("</response>") == 1
        )
    pattern = r"^<think>.*?</think>$"
    return re.search(pattern, candidate, re.DOTALL) is not None


def _toolrl_correctness_raw(reference: str, response: str) -> tuple[float, dict[str, Any]]:
    min_possible = -3.0
    max_possible = 3.0
    ref_calls = _extract_tool_payloads_strict(reference)
    pred_calls = _extract_tool_payloads_strict(response)
    if not ref_calls:
        return 0.0, {"exact_tool_match": False, "tool_call_parse_error": None}
    if not pred_calls:
        return min_possible, {"exact_tool_match": False, "tool_call_parse_error": "missing_or_invalid_tool_call"}
    try:
        reward = _compute_tool_call_reward_source(ref_calls, pred_calls, max_possible, min_possible)
    except (KeyError, TypeError):
        return min_possible, {"exact_tool_match": False, "tool_call_parse_error": "missing_or_invalid_tool_call"}
    return reward, {
        "exact_tool_match": ref_calls == pred_calls,
        "tool_call_parse_error": None,
        "reference_tool_names": _tool_names(ref_calls),
        "prediction_tool_names": _tool_names(pred_calls),
    }


def _compute_tool_call_reward_source(
    gt_tools: list[dict[str, Any]],
    pd_tools: list[dict[str, Any]],
    max_possible_reward: float,
    min_possible_reward: float,
) -> float:
    if gt_tools == pd_tools:
        return max_possible_reward
    gt_names = [tool["name"] for tool in gt_tools]
    pd_names = [tool["name"] for tool in pd_tools]
    score = _match_score(gt_names, pd_names)
    local_max_possible = 1.0
    used_pd_indices: set[int] = set()
    for gt_tool in gt_tools:
        gt_name = gt_tool["name"]
        gt_params = gt_tool["parameters"]
        local_max_possible += 1.0 + len(gt_params)
        best_match_score = 0.0
        best_match_index = -1
        for index, pd_tool in enumerate(pd_tools):
            if index in used_pd_indices or pd_tool["name"] != gt_name:
                continue
            pd_params = pd_tool["parameters"]
            param_score = _match_score(list(gt_params.keys()), list(pd_params.keys()))
            correctness_score = sum(1.0 for key, value in gt_params.items() if key in pd_params and pd_params[key] == value)
            total_score = param_score + correctness_score
            if total_score > best_match_score:
                best_match_score = total_score
                best_match_index = index
        if best_match_index >= 0:
            used_pd_indices.add(best_match_index)
            score += best_match_score
    return (max_possible_reward - min_possible_reward) * score / local_max_possible + min_possible_reward


def _match_score(list1: list[Any], list2: list[Any]) -> float:
    if list1 == list2:
        return 1.0
    if not list1 or not list2:
        return 0.0
    count1 = Counter(list1)
    count2 = Counter(list2)
    intersection = sum(min(count1[key], count2[key]) for key in count1.keys() & count2.keys())
    max_possible = len(list1) + len(list2) - intersection
    return intersection / max_possible if max_possible > 0 else 0.0


def _has_tool_tag(text: str) -> bool:
    return "<tool_call>" in (text or "")


def _tool_names(calls: list[dict[str, Any]]) -> list[str]:
    return [str(call.get("name", "")) for call in calls]


def _tool_args(calls: list[dict[str, Any]]) -> list[Any]:
    return [call.get("parameters", call.get("arguments", {})) for call in calls]


def _normalize_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _sub_exact_match(prediction: str, reference: str) -> bool:
    pred = _normalize_text(prediction)
    ref = _normalize_text(reference)
    return bool(pred and ref and (ref in pred or pred in ref))


def _extract_memory_prediction(text: str) -> str:
    boxed = _extract_boxed_answer(text)
    if boxed is not None:
        return boxed.strip()
    cleaned = (text or "").replace("*", "")
    lowered = cleaned.lower()
    marker = "the answer is"
    if marker in lowered:
        start = lowered.rfind(marker) + len(marker)
        answer = cleaned[start:].strip().strip(".").strip()
        if answer:
            return answer
    return text or ""


def _memagent_compute_score(solution_text: str, ground_truths: list[str]) -> float:
    """Exact MemAgent training reward from ``hotpotqa.py``.

    The official code lowercases the last 300 characters of the solution,
    extracts the final boxed answer, normalizes math-style strings, and returns
    the max exact-match score across ground-truth aliases.
    """

    solution = (solution_text or "")[-300:].lower()
    return max((_memagent_compute_single(solution, item) for item in ground_truths), default=0.0)


def _memagent_compute_single(solution_text: str, ground_truth: str) -> float:
    boxed = _last_boxed_only_string(solution_text)
    if boxed is None or not ground_truth:
        return 0.0
    try:
        answer = _remove_boxed(boxed)
    except (AssertionError, ValueError):
        return 0.0
    return 1.0 if _strip_memagent_string(answer) == _strip_memagent_string(str(ground_truth).lower()) else 0.0


def _memagent_ground_truths(prompt_record: dict[str, Any]) -> list[str]:
    reference = prompt_record.get("reference", {}) if isinstance(prompt_record.get("reference"), dict) else {}
    candidates: list[Any] = []
    if "answer" in reference:
        candidates.append(reference.get("answer"))
    metadata = reference.get("metadata", {}) if isinstance(reference.get("metadata"), dict) else {}
    for key in ("answers", "answer", "ground_truth", "ground_truths"):
        if key in metadata:
            candidates.append(metadata.get(key))
    if "response" in reference:
        candidates.append(_extract_memory_prediction(str(reference.get("response") or "")))
    output: list[str] = []
    for item in candidates:
        if item is None:
            continue
        if isinstance(item, list):
            output.extend(str(value) for value in item if value is not None and str(value).strip())
        elif str(item).strip():
            output.append(str(item))
    deduped = list(dict.fromkeys(value.strip() for value in output if value.strip()))
    return deduped


def _truncate_memory_final_answer(text: str) -> str:
    value = text or ""
    cut_positions = []
    for marker in ("\nHuman:", "\nAssistant:", "<|im_start|>", "<|im_end|>"):
        idx = value.find(marker)
        if idx >= 0:
            cut_positions.append(idx)
    if not cut_positions:
        return value
    return value[: min(cut_positions)]


def _remove_boxed(text: str) -> str:
    if "\\boxed " in text:
        prefix = "\\boxed "
        if not text.startswith(prefix):
            raise AssertionError("invalid boxed prefix")
        return text[len(prefix) :]
    prefix = "\\boxed{"
    if not text.startswith(prefix) or not text.endswith("}"):
        raise AssertionError("invalid boxed braces")
    return text[len(prefix) : -1]


def _strip_memagent_string(text: str) -> str:
    value = text.replace("\n", "")
    value = value.replace("\\!", "")
    value = value.replace("\\\\", "\\")
    value = value.replace("tfrac", "frac")
    value = value.replace("dfrac", "frac")
    value = value.replace("\\left", "")
    value = value.replace("\\right", "")
    value = value.replace("^{\\circ}", "")
    value = value.replace("^\\circ", "")
    value = value.replace("\\$", "")
    value = value.replace("\\%", "")
    value = value.replace("\\%", "")
    value = value.replace(" .", " 0.")
    value = value.replace("{.", "{0.")
    return value.replace(" ", "")


def _extract_boxed_answer(text: str) -> str | None:
    boxed = _last_boxed_only_string(text or "")
    if boxed is None:
        return None
    if "\\boxed " in boxed:
        return boxed[len("\\boxed ") :]
    prefix = "\\boxed{"
    if boxed.startswith(prefix) and boxed.endswith("}"):
        return boxed[len(prefix) : -1]
    return None


def _last_boxed_only_string(text: str) -> str | None:
    idx = text.rfind("\\boxed")
    if "\\boxed " in text:
        return "\\boxed " + text.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None
    depth = 0
    right_idx = None
    for pos in range(idx, len(text)):
        if text[pos] == "{":
            depth += 1
        if text[pos] == "}":
            depth -= 1
            if depth == 0:
                right_idx = pos
                break
    if right_idx is None:
        return None
    return text[idx : right_idx + 1]


def _compression_factor(prediction: str, reference: str) -> float:
    pred_len = max(1, len(_normalize_text(prediction).split()))
    ref_len = max(1, len(_normalize_text(reference).split()))
    return min(1.0, (ref_len + 1.0) / pred_len)


def _token_f1(prediction: str, reference: str) -> float:
    pred_tokens = _normalize_text(prediction).split()
    ref_tokens = _normalize_text(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2.0 * precision * recall / (precision + recall)


def _extract_code(text: str) -> str:
    matches = re.findall(r"```python(.*?)```", text or "", flags=re.DOTALL)
    if matches:
        return matches[-1]
    return "We can not extract the code in the output. "


def _normalize_code(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def _uses_input(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "input":
                return True
            if _attribute_name(node.func).startswith("sys.stdin"):
                return True
        if isinstance(node, ast.Attribute) and _attribute_name(node).startswith("sys.stdin"):
            return True
    return False


def _uses_output(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "print":
                return True
            if _attribute_name(node.func).startswith("sys.stdout"):
                return True
    return False


def _attribute_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _attribute_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _extract_code_examples(prompt_text: str, *, max_examples: int = 2) -> list[tuple[str, str]]:
    text = (prompt_text or "").replace("\r\n", "\n")
    marker = re.search(r"(?im)^\s*(examples?|sample\s+input)\s*$", text)
    if not marker:
        return []
    text = text[marker.start() :]
    pattern = re.compile(
        r"(?:sample\s+)?input\s*\n+(?P<input>.*?)\n+(?:sample\s+)?output\s*\n+(?P<output>.*?)(?=\n\s*(?:sample\s+)?input\s*\n|\n\s*(?:note|explanation|constraints)\b|\Z)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    examples = []
    for match in pattern.finditer(text):
        sample_input = _clean_example_block(match.group("input"))
        sample_output = _clean_example_block(match.group("output"))
        if sample_input and sample_output:
            examples.append((sample_input, sample_output))
        if len(examples) >= max_examples:
            break
    return examples


def _clean_example_block(text: str) -> str:
    lines = []
    for line in (text or "").strip().splitlines():
        stripped = line.rstrip()
        if stripped.startswith("```"):
            continue
        lines.append(stripped)
    return "\n".join(lines).strip() + ("\n" if lines else "")


def _run_public_examples(code: str, examples: list[tuple[str, str]], *, timeout_seconds: float | None = None) -> dict[str, Any]:
    if timeout_seconds is None:
        timeout_seconds = float(os.environ.get("OPVEC_CODE_REWARD_TIMEOUT", "5"))
    if not examples:
        return {"total": 0, "passed": 0, "pass_rate": 0.0}
    passed = 0
    rows = []
    with tempfile.TemporaryDirectory() as temp:
        script_path = Path(temp) / "solution.py"
        script_path.write_text(code, encoding="utf-8")
        for sample_input, expected_output in examples:
            try:
                proc = subprocess.run(
                    [sys.executable, str(script_path)],
                    input=sample_input,
                    text=True,
                    capture_output=True,
                    timeout=timeout_seconds,
                    cwd=temp,
                    env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                    check=False,
                )
                actual = _normalize_program_output(proc.stdout)
                expected = _normalize_program_output(expected_output)
                ok = proc.returncode == 0 and actual == expected
                passed += int(ok)
                rows.append(
                    {
                        "passed": ok,
                        "returncode": proc.returncode,
                        "stdout_head": proc.stdout[:200],
                        "stderr_head": proc.stderr[:200],
                    }
                )
            except subprocess.TimeoutExpired:
                rows.append({"passed": False, "timeout": True})
    return {
        "total": len(examples),
        "passed": passed,
        "pass_rate": passed / len(examples),
        "cases": rows,
    }


def _code_source_tests(prompt_record: dict[str, Any]) -> dict[str, Any] | None:
    reference = prompt_record.get("reference", {}) if isinstance(prompt_record.get("reference"), dict) else {}
    metadata = reference.get("metadata", {}) if isinstance(reference.get("metadata"), dict) else {}
    direct_inputs = reference.get("test_input") or metadata.get("test_input")
    direct_outputs = reference.get("test_output") or metadata.get("test_output")
    if direct_inputs and direct_outputs:
        return {
            "test_input": list(direct_inputs),
            "test_output": list(direct_outputs),
            "test_time_limit": metadata.get("test_time_limit", reference.get("test_time_limit", 1)),
            "source": "prompt_record",
        }
    question_id = metadata.get("question_id")
    if question_id is None:
        return None
    record = _codecontests_train_record(str(question_id))
    if record is None:
        return None
    return {
        "test_input": list(record.get("test_input") or []),
        "test_output": list(record.get("test_output") or []),
        "test_time_limit": record.get("test_time_limit", 1),
        "source": "CodeContests_train",
        "task_id": record.get("task_id"),
    }


@lru_cache(maxsize=1)
def _codecontests_train_by_task_id() -> dict[str, dict[str, Any]]:
    path = Path(
        os.environ.get(
            "OPVEC_CODECONTESTS_TRAIN_JSON",
            "/mnt/cache/wuruixiao/users/lsc/ExpertMerging/dataset/CodeContests_train/train/CodeContests_train.json",
        )
    )
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        return {}
    return {str(item.get("task_id")): item for item in payload if isinstance(item, dict) and item.get("task_id") is not None}


def _codecontests_train_record(task_id: str) -> dict[str, Any] | None:
    return _codecontests_train_by_task_id().get(str(task_id))


def _run_source_code_tests(code: str, source_tests: dict[str, Any]) -> dict[str, Any]:
    inputs = list(source_tests.get("test_input") or [])
    outputs = list(source_tests.get("test_output") or [])
    max_tests = int(os.environ.get("OPVEC_CODE_REWARD_MAX_TESTS", "8"))
    total = min(len(inputs), len(outputs), max_tests)
    if total <= 0:
        return {"total": 0, "passed": 0, "pass_rate": 0.0, "source": source_tests.get("source")}
    timeout_seconds = float(source_tests.get("test_time_limit") or os.environ.get("OPVEC_CODE_REWARD_TIMEOUT", "2"))
    passed = 0
    rows = []
    with tempfile.TemporaryDirectory() as temp:
        script_path = Path(temp) / "solution.py"
        script_path.write_text(code, encoding="utf-8")
        for sample_input, expected_output in zip(inputs[:total], outputs[:total]):
            try:
                proc = subprocess.run(
                    [sys.executable, str(script_path)],
                    input=sample_input,
                    text=True,
                    capture_output=True,
                    timeout=timeout_seconds,
                    cwd=temp,
                    env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                    check=False,
                )
                actual = _normalize_cure_program_output(proc.stdout)
                expected = _normalize_cure_program_output(expected_output)
                ok = proc.returncode == 0 and actual == expected
                passed += int(ok)
                rows.append(
                    {
                        "passed": ok,
                        "returncode": proc.returncode,
                        "stdout_head": proc.stdout[:200],
                        "stderr_head": proc.stderr[:200],
                    }
                )
            except subprocess.TimeoutExpired:
                rows.append({"passed": False, "timeout": True})
    return {
        "total": total,
        "passed": passed,
        "pass_rate": passed / total,
        "source": source_tests.get("source"),
        "task_id": source_tests.get("task_id"),
        "cases": rows,
    }


def _normalize_program_output(text: str) -> str:
    return "\n".join(line.rstrip() for line in (text or "").strip().splitlines())


def _normalize_cure_program_output(text: str) -> str:
    return " ".join((text or "").split())


def _reference_call_overlap(code: str, reference_code: str) -> float:
    predicted = _call_names(code)
    reference = _call_names(reference_code)
    if not predicted or not reference:
        return 0.0
    overlap = predicted & reference
    return len(overlap) / len(reference)


def _call_names(code: str) -> set[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _attribute_name(node.func)
            if name and name not in {"input", "print"}:
                names.add(name)
    return names
