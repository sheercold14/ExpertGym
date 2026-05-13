#!/usr/bin/env python3
import argparse
import ast
import json
import math
import os
import re
import string
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_EVAL_ROOT = Path("/tmp/shared-storage/ExpertGym/LLaMA/eval")
DEFAULT_OUTPUT_ROOT = Path("/tmp/shared-storage/ExpertGym/LLaMA/results")


@dataclass
class EvalItem:
    benchmark: str
    item_id: str
    prompt: str
    answer: Any
    meta: dict[str, Any]


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


class StopOnSequence(transformers.StoppingCriteria):
    def __init__(self, target_sequences: list[str], tokenizer: Any):
        self.target_ids = [tokenizer.encode(seq, add_special_tokens=False) for seq in target_sequences]
        self.target_ids = [ids for ids in self.target_ids if ids]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:
        for ids in self.target_ids:
            if len(ids) <= input_ids.shape[1] and input_ids[0, -len(ids) :].tolist() == ids:
                return True
        return False


def normalize_text(text: Any) -> str:
    text = str(text).lower()
    text = text.replace("\u2019", "'")
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


def normalize_math(text: Any) -> str:
    text = str(text)
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\,", "").replace("\\!", "")
    text = text.replace("$", "")
    text = re.sub(r"\\text\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\s+", "", text)
    return text.strip().lower()


def extract_boxed(text: str) -> str | None:
    for marker in ["\\boxed{", "boxed{"]:
        idx = text.rfind(marker)
        if idx < 0:
            continue
        start = idx + len(marker)
        depth = 1
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i]
    return None


def extract_answer_tag(text: str) -> str | None:
    m = re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    return m[-1].strip() if m else None


def extract_last_number(text: str) -> str | None:
    nums = re.findall(r"[-+]?\d*\.?\d+(?:/\d+)?", text.replace(",", ""))
    return nums[-1] if nums else None


def score_gsm8k(pred: str, gold: str) -> bool:
    gold_num = gold.split("####")[-1].strip().replace(",", "")
    pred_ans = extract_answer_tag(pred) or extract_boxed(pred) or pred
    pred_num = extract_last_number(pred_ans)
    return pred_num is not None and pred_num == gold_num


def score_math500(pred: str, gold: str) -> bool:
    pred_ans = extract_answer_tag(pred) or extract_boxed(pred) or pred.strip().splitlines()[-1]
    return normalize_math(pred_ans) == normalize_math(gold)


def score_qa(pred: str, answers: Any) -> bool:
    pred_ans = extract_answer_tag(pred) or pred
    pred_norm = normalize_text(pred_ans)
    if not isinstance(answers, (list, tuple)):
        if hasattr(answers, "tolist"):
            answers = answers.tolist()
        else:
            answers = [answers]
    for ans in answers:
        ans_norm = normalize_text(ans)
        if ans_norm and (ans_norm in pred_norm or pred_norm in ans_norm):
            return True
    return False


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_items(benchmark: str, eval_root: Path, max_samples: int | None) -> list[EvalItem]:
    if benchmark == "gsm8k":
        df = pd.read_parquet(eval_root / "math/gsm8k/main/test-00000-of-00001.parquet")
        rows = df.to_dict("records")
        items = [
            EvalItem(
                benchmark,
                f"gsm8k_{i}",
                "Solve the problem. Put only the final numeric answer in <answer>...</answer>.\n\n"
                f"Problem: {r['question']}",
                r["answer"],
                {},
            )
            for i, r in enumerate(rows)
        ]
    elif benchmark == "math500":
        rows = load_jsonl(eval_root / "math/MATH-500/test.jsonl")
        items = [
            EvalItem(
                benchmark,
                str(r.get("unique_id", i)),
                "Solve the math problem. Put the final answer in <answer>...</answer>.\n\n"
                f"Problem: {r['problem']}",
                r["answer"],
                {"subject": r.get("subject"), "level": r.get("level")},
            )
            for i, r in enumerate(rows)
        ]
    elif benchmark == "nq_open":
        df = pd.read_parquet(eval_root / "search/nq_open/nq_open/validation-00000-of-00001.parquet")
        rows = df.to_dict("records")
        items = [
            EvalItem(
                benchmark,
                f"nq_open_{i}",
                "Answer the question concisely. Put the final answer in <answer>...</answer>.\n\n"
                f"Question: {r['question']}",
                r["answer"],
                {"question": r["question"]},
            )
            for i, r in enumerate(rows)
        ]
    elif benchmark == "two_wiki":
        with (eval_root / "search/2WikiMultihopQA/dev.json").open("r", encoding="utf-8") as f:
            rows = json.load(f)
        items = []
        for i, r in enumerate(rows):
            context = "\n".join(f"{title}: {' '.join(sents)}" for title, sents in r["context"])
            prompt = (
                "Answer the multi-hop question using the provided context. "
                "Put the final answer in <answer>...</answer>.\n\n"
                f"Context:\n{context}\n\nQuestion: {r['question']}"
            )
            items.append(
                EvalItem(
                    benchmark,
                    str(r.get("_id", i)),
                    prompt,
                    r["answer"],
                    {"type": r.get("type"), "question": r["question"], "context": r["context"]},
                )
            )
    elif benchmark in {"bfcl_live", "bfcl_non_live"}:
        items = load_bfcl_items(eval_root, live=(benchmark == "bfcl_live"))
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    return items[:max_samples] if max_samples else items


def bfcl_files(live: bool) -> list[str]:
    if live:
        return [
            "BFCL_v3_live_multiple.json",
            "BFCL_v3_live_irrelevance.json",
            "BFCL_v3_live_simple.json",
            "BFCL_v3_live_parallel.json",
            "BFCL_v3_live_parallel_multiple.json",
        ]
    return [
        "BFCL_v3_multiple.json",
        "BFCL_v3_irrelevance.json",
        "BFCL_v3_simple.json",
        "BFCL_v3_parallel.json",
        "BFCL_v3_parallel_multiple.json",
    ]


def load_bfcl_ground_truth(root: Path, file_name: str) -> dict[str, Any]:
    gt_path = root / "possible_answer" / file_name
    if not gt_path.exists():
        return {}
    return {r["id"]: r.get("ground_truth") for r in load_jsonl(gt_path)}


def load_bfcl_items(eval_root: Path, live: bool) -> list[EvalItem]:
    root = eval_root / "tool/BFCL"
    items = []
    for file_name in bfcl_files(live):
        gt = load_bfcl_ground_truth(root, file_name)
        for r in load_jsonl(root / file_name):
            user_msg = r["question"][0][-1]["content"]
            functions = json.dumps(r["function"], ensure_ascii=False)
            expected = gt.get(r["id"], [])
            prompt = (
                "You are a function-calling model. Choose the required function calls from the provided tools.\n"
                "Return only a JSON array like "
                '[{"name":"function.name","arguments":{"arg":"value"}}]. '
                "If no function should be called, return [].\n\n"
                f"Tools:\n{functions}\n\nUser request:\n{user_msg}"
            )
            items.append(
                EvalItem(
                    "bfcl_live" if live else "bfcl_non_live",
                    r["id"],
                    prompt,
                    expected,
                    {"source_file": file_name, "has_ground_truth": bool(expected)},
                )
            )
    return items


def coerce_json(text: str) -> Any | None:
    candidates = []
    code_blocks = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates.extend(code_blocks)
    bracket = re.search(r"(\[.*\])", text, flags=re.DOTALL)
    if bracket:
        candidates.append(bracket.group(1))
    brace = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if brace:
        candidates.append(brace.group(1))
    candidates.append(text)

    for cand in candidates:
        cand = cand.strip()
        try:
            return json.loads(cand)
        except Exception:
            try:
                return ast.literal_eval(cand)
            except Exception:
                continue
    return None


def parse_function_calls(text: str) -> list[dict[str, Any]]:
    data = coerce_json(text)
    if data is None:
        calls = []
        for name, arg_text in re.findall(r"([A-Za-z_][\\w.]+)\s*\((.*?)\)", text):
            args = {}
            for key, value in re.findall(r"([A-Za-z_][\\w]*)\s*=\s*([^,]+)", arg_text):
                args[key] = value.strip().strip("'\"")
            calls.append({"name": name, "arguments": args})
        return calls
    if isinstance(data, dict):
        data = [data]
    calls = []
    if not isinstance(data, list):
        return calls
    for item in data:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("function") or item.get("tool_name")
        args = item.get("arguments") or item.get("args") or item.get("parameters") or {}
        if not name and len(item) == 1:
            name, args = next(iter(item.items()))
        if isinstance(args, str):
            parsed = coerce_json(args)
            args = parsed if isinstance(parsed, dict) else {}
        calls.append({"name": str(name), "arguments": args if isinstance(args, dict) else {}})
    return calls


def values_match(pred: Any, gold_values: list[Any]) -> bool:
    pred_norm = normalize_text(pred)
    return any(pred_norm == normalize_text(gold) for gold in gold_values)


def score_bfcl(pred: str, expected: list[dict[str, Any]]) -> bool:
    calls = parse_function_calls(pred)
    if not expected:
        return len(calls) == 0

    used = set()
    for gt_call in expected:
        fn_name, params = next(iter(gt_call.items()))
        matched = False
        for i, call in enumerate(calls):
            if i in used or call["name"] != fn_name:
                continue
            args = call["arguments"]
            ok = True
            for key, gold_values in params.items():
                if key not in args or not values_match(args[key], gold_values):
                    ok = False
                    break
            if ok:
                used.add(i)
                matched = True
                break
        if not matched:
            return False
    return True


def score_item(item: EvalItem, pred: str) -> bool:
    if item.benchmark == "gsm8k":
        return score_gsm8k(pred, item.answer)
    if item.benchmark == "math500":
        return score_math500(pred, item.answer)
    if item.benchmark in {"nq_open", "two_wiki"}:
        return score_qa(pred, item.answer)
    if item.benchmark in {"bfcl_live", "bfcl_non_live"}:
        return score_bfcl(pred, item.answer)
    raise ValueError(item.benchmark)


def build_chat_prompt(tokenizer: Any, content: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
    return content


def build_zerosearch_prompt(question: str) -> str:
    question = question.strip()
    if question and question[-1] != "?":
        question += "?"
    return (
        "Answer the given question. "
        "You must conduct reasoning inside <think> and </think> first every time you get new information. "
        "After reasoning, if you find you lack some knowledge, you can call a search engine by "
        "<search> query </search> and it will return the top searched results between "
        "<information> and </information>. "
        "You can search as many times as your want. "
        "If you find no further external knowledge needed, you can directly provide the answer inside "
        "<answer> and </answer>, without detailed illustrations. "
        f"For example, <answer> Beijing </answer>. Question: {question}\n"
    )


def get_search_query(text: str) -> str | None:
    matches = re.findall(r"<search>(.*?)</search>", text, flags=re.DOTALL | re.IGNORECASE)
    return matches[-1].strip() if matches else None


INVALID_SEARCH_ACTION_FEEDBACK = (
    "\nMy previous action is invalid. "
    "If I want to search, I should put the query between <search> and </search>. "
    "If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n"
)


def retrieve_google(query: str, topk: int) -> str:
    api_key = os.environ.get("SER_API_KEY") or os.environ.get("SERPAPI_API_KEY")
    if not api_key:
        return "No information available"
    params = {"engine": "google", "q": query, "api_key": api_key, "num": topk}
    try:
        resp = requests.get("https://serpapi.com/search.json", params=params, timeout=30)
        resp.raise_for_status()
        results = resp.json().get("organic_results", [])[:topk]
    except Exception:
        return "No information available"
    docs = []
    for i, item in enumerate(results, 1):
        text = " ".join(str(item.get(k, "")) for k in ("title", "snippet") if item.get(k))
        docs.append(f"Doc {i}: {text}")
    return "\n".join(docs) if docs else "No information available"


def load_api_key_from_file(path: str, field: str = "key") -> str | None:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None
    value = data.get(field)
    return str(value).strip() if value else None


def retrieve_serper(query: str, topk: int) -> str:
    api_key = os.environ.get("SERPER_API_KEY")
    key_file = os.environ.get("SERPER_API_KEY_FILE")
    if not api_key and key_file:
        api_key = load_api_key_from_file(key_file)
    if not api_key:
        return "No information available"
    payload = {"q": query, "num": topk}
    try:
        resp = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        results = resp.json().get("organic", [])[:topk]
    except Exception:
        return "No information available"
    docs = []
    for i, item in enumerate(results, 1):
        parts = []
        if item.get("title"):
            parts.append(str(item["title"]))
        if item.get("snippet"):
            parts.append(str(item["snippet"]))
        if item.get("link"):
            parts.append(str(item["link"]))
        docs.append(f"Doc {i}: {' '.join(parts)}")
    return "\n".join(docs) if docs else "No information available"


def retrieve_exa(query: str, topk: int, search_type: str) -> str:
    api_key = os.environ.get("EXA_API_KEY")
    if not api_key:
        return "No information available"
    payload = {
        "query": query,
        "numResults": topk,
        "type": search_type,
        "contents": {"highlights": True},
    }
    try:
        resp = requests.post(
            "https://api.exa.ai/search",
            headers={"x-api-key": api_key, "Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])[:topk]
    except Exception:
        return "No information available"
    docs = []
    for i, item in enumerate(results, 1):
        parts = []
        if item.get("title"):
            parts.append(str(item["title"]))
        if item.get("url"):
            parts.append(str(item["url"]))
        highlights = item.get("highlights") or []
        if highlights:
            parts.append(" ".join(str(x) for x in highlights[:3]))
        elif item.get("text"):
            parts.append(str(item["text"])[:1200])
        docs.append(f"Doc {i}: {' '.join(parts)}")
    return "\n".join(docs) if docs else "No information available"


def retrieve_wiki(query: str, topk: int, url: str) -> str:
    try:
        resp = requests.post(url, json={"query": query, "top_k": topk}, timeout=30)
        resp.raise_for_status()
        results = resp.json()[:topk]
    except Exception:
        return "No information available"
    docs = []
    for i, doc in enumerate(results, 1):
        text = doc.get("text") if isinstance(doc, dict) else str(doc)
        docs.append(f"Doc {i}: {text}")
    return "\n".join(docs) if docs else "No information available"


def retrieve_local_context(query: str, item: EvalItem, topk: int) -> str:
    context = item.meta.get("context") or []
    if not context:
        return "No information available"
    q_words = set(normalize_text(query).split())
    scored = []
    for title, sentences in context:
        text = f"{title}: {' '.join(sentences)}"
        words = set(normalize_text(text).split())
        scored.append((len(q_words & words), text))
    scored.sort(key=lambda x: x[0], reverse=True)
    return "\n".join(f"Doc {i}: {text}" for i, (_, text) in enumerate(scored[:topk], 1))


def normalize_search_query(query: str) -> str:
    return " ".join(query.lower().strip().split())


def clean_search_query(query: str) -> str:
    query = query.strip()
    match = re.fullmatch(r"(?is)query\s*=\s*(.+)", query)
    if match:
        query = match.group(1).strip()
    if len(query) >= 2 and query[0] == query[-1] and query[0] in {"'", '"'}:
        query = query[1:-1].strip()
    return query


def search_cache_key(query: str, args: argparse.Namespace) -> str:
    query = clean_search_query(query)
    payload = {
        "backend": args.search_backend,
        "query": normalize_search_query(query),
        "topk": args.search_topk,
        "search_url": args.search_url if args.search_backend == "wiki" else None,
        "exa_search_type": args.exa_search_type if args.search_backend == "exa" else None,
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def load_search_cache(args: argparse.Namespace) -> dict[str, str]:
    if hasattr(args, "_search_cache"):
        return args._search_cache
    cache: dict[str, str] = {}
    cache_path = getattr(args, "search_cache_path", None)
    if cache_path:
        path = Path(cache_path)
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    key = row.get("key")
                    info = row.get("information")
                    if key and isinstance(info, str):
                        cache[key] = info
    args._search_cache = cache
    return cache


def save_search_cache(args: argparse.Namespace, key: str, info: str) -> None:
    cache_path = getattr(args, "search_cache_path", None)
    if not cache_path:
        return
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "information": info}, ensure_ascii=False) + "\n")


def retrieve_for_search(query: str, item: EvalItem, args: argparse.Namespace) -> str:
    args._last_search_cache_hit = False
    query_for_backend = clean_search_query(query)
    args._last_search_query = query_for_backend
    cache_path = getattr(args, "search_cache_path", None)
    key = search_cache_key(query_for_backend, args) if cache_path else None
    if key:
        cache = load_search_cache(args)
        if key in cache:
            args._last_search_cache_hit = True
            info = cache[key]
            max_chars = getattr(args, "search_info_max_chars", 0)
            if max_chars and len(info) > max_chars:
                return info[:max_chars] + "\n[truncated]"
            return info

    if args.search_backend == "google":
        info = retrieve_google(query_for_backend, args.search_topk)
    elif args.search_backend == "serper":
        info = retrieve_serper(query_for_backend, args.search_topk)
    elif args.search_backend == "exa":
        info = retrieve_exa(query_for_backend, args.search_topk, args.exa_search_type)
    elif args.search_backend == "wiki":
        info = retrieve_wiki(query_for_backend, args.search_topk, args.search_url)
    elif args.search_backend == "local_context":
        info = retrieve_local_context(query_for_backend, item, args.search_topk)
    else:
        info = "No information available"
    if key:
        load_search_cache(args)[key] = info
        save_search_cache(args, key, info)
    max_chars = getattr(args, "search_info_max_chars", 0)
    if max_chars and len(info) > max_chars:
        return info[:max_chars] + "\n[truncated]"
    return info


def load_model(model_path: str, dtype: str, trust_remote_code: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    torch_dtype = {"auto": "auto", "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    return tokenizer, model


def generate(tokenizer: Any, model: Any, prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
    text = build_chat_prompt(tokenizer, prompt)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        gen_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": top_p})
    else:
        gen_kwargs.update({"do_sample": False})
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    return tokenizer.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()


def generate_from_text(
    tokenizer: Any,
    model: Any,
    text: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    stopping_criteria: Any | None = None,
) -> str:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if stopping_criteria is not None:
        gen_kwargs["stopping_criteria"] = stopping_criteria
    if temperature > 0:
        gen_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": top_p})
    else:
        gen_kwargs.update({"do_sample": False})
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    return tokenizer.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)


def generate_zerosearch(tokenizer: Any, model: Any, item: EvalItem, args: argparse.Namespace) -> tuple[str, list[dict[str, str]]]:
    question = item.meta.get("question")
    if not question:
        return generate(tokenizer, model, item.prompt, args.max_new_tokens, args.temperature, args.top_p), []

    prompt = build_chat_prompt(tokenizer, build_zerosearch_prompt(question))
    target_sequences = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n"]
    stopping = transformers.StoppingCriteriaList([StopOnSequence(target_sequences, tokenizer)])
    trace = []
    final_text_parts = []

    for turn in range(args.search_max_turns + 1):
        generated = generate_from_text(tokenizer, model, prompt, args.max_new_tokens, args.temperature, args.top_p, stopping)
        final_text_parts.append(generated)
        prompt += generated
        query = get_search_query(generated)
        if extract_answer_tag(generated) is not None:
            break
        if not query:
            if getattr(args, "zerosearch_invalid_feedback", False) and turn < args.search_max_turns:
                prompt += INVALID_SEARCH_ACTION_FEEDBACK
                final_text_parts.append(INVALID_SEARCH_ACTION_FEEDBACK)
                continue
            break
        info = retrieve_for_search(query, item, args)
        trace.append(
            {
                "turn": str(turn),
                "query": query,
                "backend_query": getattr(args, "_last_search_query", query),
                "information": info,
                "cache_hit": str(bool(getattr(args, "_last_search_cache_hit", False))),
            }
        )
        search_text = f"\n\n<information>{info}</information>\n\n"
        prompt += search_text
        final_text_parts.append(search_text)

    return "".join(final_text_parts).strip(), trace


def evaluate(args: argparse.Namespace) -> Path:
    eval_root = Path(args.eval_root)
    output_dir = Path(args.output_root) / args.run_id / args.model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer, model = load_model(args.model, args.dtype, args.trust_remote_code)

    summary = {"model": args.model, "model_name": args.model_name, "run_id": args.run_id, "benchmarks": {}}
    for benchmark in args.benchmarks.split(","):
        benchmark = benchmark.strip()
        if not benchmark:
            continue
        items = load_items(benchmark, eval_root, args.max_samples)
        bench_dir = output_dir / benchmark
        bench_dir.mkdir(parents=True, exist_ok=True)
        correct = 0
        pred_path = bench_dir / "predictions.jsonl"
        t0 = time.time()
        with pred_path.open("w", encoding="utf-8") as f:
            for idx, item in enumerate(items, 1):
                try:
                    if args.search_mode == "zerosearch" and item.benchmark in {"nq_open", "two_wiki"}:
                        pred, search_trace = generate_zerosearch(tokenizer, model, item, args)
                    else:
                        pred = generate(tokenizer, model, item.prompt, args.max_new_tokens, args.temperature, args.top_p)
                        search_trace = []
                    is_correct = score_item(item, pred)
                    error = None
                except Exception as exc:
                    pred = ""
                    search_trace = []
                    is_correct = False
                    error = repr(exc)
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
                            "error": error,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                if args.log_every and idx % args.log_every == 0:
                    print(f"[{benchmark}] {idx}/{len(items)} acc={correct / idx:.4f}")
        total = len(items)
        metrics = {
            "benchmark": benchmark,
            "total": total,
            "correct": correct,
            "accuracy": correct / total if total else math.nan,
            "seconds": time.time() - t0,
            "max_samples": args.max_samples,
            "search_mode": args.search_mode if benchmark in {"nq_open", "two_wiki"} else "static",
            "search_backend": args.search_backend if benchmark in {"nq_open", "two_wiki"} else None,
        }
        (bench_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        summary["benchmarks"][benchmark] = metrics
        print(f"[done] {benchmark}: {correct}/{total} = {metrics['accuracy']:.4f}")

    scores = [m["accuracy"] for m in summary["benchmarks"].values() if not math.isnan(m["accuracy"])]
    summary["macro_accuracy"] = sum(scores) / len(scores) if scores else math.nan
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate any local/HF causal LM on RAM Llama benchmarks.")
    parser.add_argument("--model", required=True, help="HF repo id or local model path.")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--benchmarks", default="gsm8k,math500,nq_open,two_wiki,bfcl_live,bfcl_non_live")
    parser.add_argument("--eval-root", default=str(DEFAULT_EVAL_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-id", default=time.strftime("eval-%Y%m%d-%H%M%S"))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--search-mode", choices=["static", "zerosearch"], default="static")
    parser.add_argument("--search-backend", choices=["none", "google", "serper", "exa", "wiki", "local_context"], default="none")
    parser.add_argument("--search-url", default="http://localhost:6002/retrieve")
    parser.add_argument("--search-topk", type=int, default=5)
    parser.add_argument("--search-max-turns", type=int, default=5)
    parser.add_argument("--search-info-max-chars", type=int, default=3500)
    parser.add_argument("--search-cache-path", default=None)
    parser.add_argument("--zerosearch-invalid-feedback", action="store_true")
    parser.add_argument("--exa-search-type", choices=["auto", "fast", "instant", "deep-lite", "deep"], default="auto")
    args = parser.parse_args()
    args.model_name = args.model_name or Path(args.model).name.replace("/", "__")
    out = evaluate(args)
    print(f"results: {out}")


if __name__ == "__main__":
    main()
