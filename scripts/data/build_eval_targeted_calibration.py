#!/usr/bin/env python3
"""Build a leakage-safe eval-targeted OP-VEC calibration manifest.

This builder uses eval case studies only as a failure-taxonomy signal.  It does
not copy BFCL/CURE official prompts, answers, tests, model outputs, or hidden
cases into the training manifest.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from opvec.data.schema import stable_hash, validate_seed_record


DEFAULT_CASE_CANDIDATES = "/tmp/shared-storage/OnPolicy/analysis/eval_case_browser/bfcl_live_calibration_candidates.jsonl"
DEFAULT_PAPER96 = "/tmp/shared-storage/OnPolicy/data/calibration/qbank_c033333_paper96_balanced32_seed20260514.prompts.jsonl"
DEFAULT_CODECONTESTS = "/mnt/cache/wuruixiao/users/lsc/ExpertMerging/dataset/CodeContests_train/train/CodeContests_train.json"
DEFAULT_OUTPUT_DIR = "/tmp/shared-storage/OnPolicy/data/calibration/eval_targeted96_20260517"

TASKS = ("tool", "memory", "code")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    created_at = datetime.now(timezone.utc).isoformat()
    candidates = read_jsonl(Path(args.case_candidates).expanduser())
    paper96 = read_jsonl(Path(args.paper96).expanduser())
    code_source = read_json(Path(args.codecontests).expanduser())

    profile = build_case_profile(candidates)
    tool_rows, tool_blueprints = build_tool_rows(
        paper96=paper96,
        source_count=args.tool_source_count,
        synthetic_count=args.tool_synthetic_count,
        seed=args.seed,
        created_at=created_at,
        profile=profile,
    )
    memory_rows = build_memory_rows(paper96, count=args.memory_count, seed=args.seed, created_at=created_at)
    code_rows, code_blueprints = build_code_rows(
        paper96=paper96,
        source_rows=code_source,
        count=args.code_count,
        source_count=args.code_source_count,
        targeted_count=args.code_targeted_count,
        seed=args.seed,
        created_at=created_at,
        profile=profile,
        codecontests_path=Path(args.codecontests).expanduser().resolve(),
    )
    prompts = round_robin_interleave({"tool": tool_rows, "memory": memory_rows, "code": code_rows})

    for row in prompts:
        validate_seed_record(row)

    prompt_path = output_dir / "eval_targeted96.prompts.jsonl"
    tool_blueprint_path = output_dir / "tool_synthetic_blueprints.jsonl"
    code_blueprint_path = output_dir / "code_train_blueprints.jsonl"
    summary_path = output_dir / "summary.json"
    readme_path = output_dir / "README.md"
    write_jsonl(prompt_path, prompts)
    write_jsonl(tool_blueprint_path, tool_blueprints)
    write_jsonl(code_blueprint_path, code_blueprints)
    summary = {
        "format": "opvec_eval_targeted_calibration_v1",
        "created_at": created_at,
        "seed": args.seed,
        "inputs": {
            "case_candidates": str(Path(args.case_candidates).expanduser().resolve()),
            "paper96": str(Path(args.paper96).expanduser().resolve()),
            "codecontests_train": str(Path(args.codecontests).expanduser().resolve()),
        },
        "outputs": {
            "prompts": str(prompt_path),
            "tool_blueprints": str(tool_blueprint_path),
            "code_blueprints": str(code_blueprint_path),
            "summary": str(summary_path),
            "readme": str(readme_path),
        },
        "counts": {
            "prompts": len(prompts),
            "tasks": dict(Counter(row["task"] for row in prompts)),
            "tool_sources": dict(Counter(row.get("eval_targeted_calibration", {}).get("role") for row in tool_rows)),
            "tool_bfcl_categories": dict(
                Counter(
                    row.get("reference", {}).get("bfcl", {}).get("category")
                    for row in tool_rows
                    if row.get("reference", {}).get("bfcl")
                )
            ),
            "code_sources": dict(Counter(row.get("eval_targeted_calibration", {}).get("role") for row in code_rows)),
            "code_tags": top_counter(tag for row in code_rows for tag in row.get("eval_targeted_calibration", {}).get("code_tags", [])),
            "memory_source": dict(Counter(str(row.get("source")) for row in memory_rows)),
        },
        "case_study_profile": profile,
        "leakage_policy": {
            "tool": "Mixed 16 paper96 ToolRL/RLLA source anchors with 16 fresh synthetic BFCL-style prompts; no official BFCL prompt, possible_answer, or model output is copied.",
            "memory": "Reuses existing paper96 HotpotQA-train memory rows, not formal eval rows.",
            "code": "Optionally mixes paper96 CodeContests source anchors with CodeContests-train rows selected by eval-derived CURE tags; no LiveBench/LiveCodeBench prompt, tests, generated code, or output is copied.",
        },
        "training_intent": {
            "tool": "Preserve source-domain ToolRL/RLLA behavior while exposing BFCL-like multi-call, enum/default, canonicalization, and mixed-function failures.",
            "memory": "Keep the previous memory calibration anchor because memory already met the near-term target.",
            "code": "Preserve previously observed code frontier anchors while increasing CURE-like stdin/stdout, math, greedy/array/string/simulation/graph coverage using non-eval executable tests.",
        },
    }
    write_json(summary_path, summary)
    readme_path.write_text(render_readme(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def build_case_profile(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    tool_rows = [row for row in candidates if row.get("source_benchmark") == "BFCL"]
    code_rows = [row for row in candidates if row.get("source_benchmark") == "CURE"]
    return {
        "candidate_rows": len(candidates),
        "by_benchmark": dict(Counter(row.get("source_benchmark") for row in candidates)),
        "by_status": dict(Counter(row.get("status") for row in candidates)),
        "by_priority": dict(Counter(row.get("priority") for row in candidates)),
        "tool": {
            "rows": len(tool_rows),
            "categories": dict(Counter(row.get("source_category") for row in tool_rows)),
            "failure_tags": top_counter(tag for row in tool_rows for tag in row.get("failure_tags", [])),
            "languages": dict(Counter(row.get("source_language") for row in tool_rows)),
        },
        "code": {
            "rows": len(code_rows),
            "categories": dict(Counter(row.get("source_category") for row in code_rows)),
            "failure_tags": top_counter(tag for row in code_rows for tag in row.get("failure_tags", [])),
            "code_tags": top_counter(tag for row in code_rows for tag in row.get("code_tags", [])),
        },
    }


def build_tool_rows(
    *,
    paper96: list[dict[str, Any]],
    source_count: int,
    synthetic_count: int,
    seed: int,
    created_at: str,
    profile: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_rows, source_blueprints = build_source_tool_rows(
        paper96,
        count=source_count,
        seed=seed,
        created_at=created_at,
        profile=profile,
    )
    specs = synthetic_tool_specs()
    rng = random.Random(seed)
    rng.shuffle(specs)
    selected = specs[:synthetic_count]
    synthetic_rows = []
    synthetic_blueprints = []
    for index, spec in enumerate(selected):
        row = tool_spec_to_seed_record(spec, index=index, created_at=created_at, profile=profile)
        synthetic_rows.append(row)
        synthetic_blueprints.append(
            {
                "format": "eval_targeted_tool_blueprint_v1",
                "prompt_id": row["prompt_id"],
                "category": spec["category"],
                "failure_tags_targeted": spec["failure_tags"],
                "language": spec["language"],
                "num_functions": len(spec["functions"]),
                "num_reference_calls": len(spec["answers"]),
                "source": "fresh_synthetic_from_case_study_taxonomy",
                "leakage_policy": row["eval_targeted_calibration"]["leakage_policy"],
            }
        )
    return interleave_lists([source_rows, synthetic_rows]), source_blueprints + synthetic_blueprints


def build_source_tool_rows(
    paper96: list[dict[str, Any]],
    *,
    count: int,
    seed: int,
    created_at: str,
    profile: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tool_rows = [row for row in paper96 if row.get("task") == "tool"]
    if len(tool_rows) < count:
        raise SystemExit(f"Not enough source Tool rows in paper96: need={count}, have={len(tool_rows)}")
    # Prefer older high-information frontier rows over all-success retention-like
    # rows, but keep selection deterministic.
    def sort_key(row: dict[str, Any]) -> tuple[int, str]:
        selection = row.get("question_bank_selection", {}) if isinstance(row.get("question_bank_selection"), dict) else {}
        bucket = str(selection.get("bucket") or "")
        bucket_rank = 0 if bucket in {"all_fail_partial", "mid_frontier", "low_frontier"} else 1
        return bucket_rank, stable_hash({"seed": seed, "prompt_id": row.get("prompt_id")})

    selected = sorted(tool_rows, key=sort_key)[:count]
    rows = []
    blueprints = []
    for index, row in enumerate(selected):
        cloned = copy.deepcopy(row)
        cloned["tags"] = sorted(set(cloned.get("tags") or []) | {"eval_targeted", "source_tool_anchor", "toolrl_rlla"})
        cloned["eval_targeted_calibration"] = {
            "created_at": created_at,
            "role": "source_tool_anchor_from_paper96",
            "reason": "Keep original ToolRL/RLLA distribution so BFCL-style synthetic rows do not replace source-domain tool-call behavior.",
            "case_study_source": "mixed with BFCL-style synthetic after eval case study",
            "profile_snapshot": {
                "top_tool_failure_tags": profile.get("tool", {}).get("failure_tags", [])[:8],
            },
            "leakage_policy": "Existing paper96 Tool source row; not BFCL official eval.",
        }
        rows.append(cloned)
        blueprints.append(
            {
                "format": "eval_targeted_tool_blueprint_v1",
                "prompt_id": cloned["prompt_id"],
                "source": "paper96_tool_source_anchor",
                "source_manifest": cloned.get("source_manifest"),
                "source_row": cloned.get("source_row"),
                "bucket": (cloned.get("question_bank_selection") or {}).get("bucket") if isinstance(cloned.get("question_bank_selection"), dict) else None,
                "leakage_policy": cloned["eval_targeted_calibration"]["leakage_policy"],
            }
        )
    return rows, blueprints


def tool_spec_to_seed_record(spec: dict[str, Any], *, index: int, created_at: str, profile: dict[str, Any]) -> dict[str, Any]:
    prompt = bfcl_prompt(spec["functions"], spec["user"])
    payload = {
        "task": "tool",
        "prompt": prompt,
        "category": spec["category"],
        "answers": spec["answers"],
        "functions": spec["functions"],
    }
    prompt_hash = stable_hash(payload)
    prompt_id = f"tool_evaltarget__{prompt_hash[:16]}"
    row = {
        "prompt_id": prompt_id,
        "task": "tool",
        "source": "synthetic_bfcl_live_style_case_study",
        "source_row": index,
        "split": "eval_targeted_calibration_train",
        "prompt": prompt,
        "messages": [],
        "reference": {
            "bfcl": {
                "id": f"synthetic_{spec['category']}_{prompt_hash[:12]}",
                "category": spec["category"],
                "function": spec["functions"],
                "possible_answer": spec["answers"],
                "model_name": "opvec-bfcl-offline",
            },
            "response": spec["response"],
            "metadata": {
                "source": "eval_targeted_synthetic_bfcl",
                "failure_tags_targeted": spec["failure_tags"],
                "language": spec["language"],
            },
        },
        "verifier": {"name": "bfcl_ast", "config": {"source": "fresh_synthetic"}},
        "tags": sorted(
            {
                "eval_targeted",
                "synthetic",
                "bfcl_style",
                spec["category"],
                *[f"tag:{tag}" for tag in spec["failure_tags"]],
            }
        ),
        "difficulty": None,
        "prompt_hash": prompt_hash,
        "eval_targeted_calibration": {
            "created_at": created_at,
            "role": "on_policy_eval_style_tool_probe",
            "case_study_source": "eval_case_browser calibration candidates",
            "profile_snapshot": {
                "top_tool_failure_tags": profile.get("tool", {}).get("failure_tags", [])[:8],
                "tool_categories": profile.get("tool", {}).get("categories", {}),
            },
            "failure_tags_targeted": spec["failure_tags"],
            "leakage_policy": "Fresh schemas/entities/prompts; no official BFCL prompt, answer, or model output copied.",
        },
    }
    return row


def bfcl_prompt(functions: list[dict[str, Any]], user_text: str) -> str:
    function_json = json.dumps(functions, ensure_ascii=False, indent=4, sort_keys=True)
    system = (
        "You are an expert in composing functions.You are given a question and a set of possible functions. "
        "Based on the question, you will need to make one or more function/tool calls to achieve the purpose. "
        "If none of the functions can be used, point it out. If the given question lacks the parameters required "
        "by the function, also point it out.\n\n"
        "You should only return the function calls in your response.\n\n"
        "If you decide to invoke any of the function(s), you MUST put it in the format of "
        "[func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)].  "
        "You SHOULD NOT include any other text in the response.\n\n"
        "At each turn, you should try your best to complete the tasks requested by the user within the current turn. "
        "Continue to output functions to call until you have fulfilled the user's request to the best of your ability. "
        "Once you have no more functions to call, the system will consider the current turn complete and proceed to "
        "the next turn or task.\n\n"
        "Here is a list of functions in json format that you can invoke.\n"
        f"{function_json}"
    )
    return f"<|im_start|>system\n{system}\n<|im_end|>\n<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"


def synthetic_tool_specs() -> list[dict[str, Any]]:
    weather = {
        "name": "fetch_city_conditions",
        "description": "Retrieves current weather conditions for a city using canonical English location strings.",
        "parameters": {
            "type": "dict",
            "required": ["location"],
            "properties": {
                "location": {"type": "string", "description": "Canonical location, e.g. 'Lisbon, Portugal' or 'Austin, TX'."},
                "unit": {"type": "string", "description": "Temperature unit.", "enum": ["celsius", "fahrenheit"], "default": "fahrenheit"},
            },
        },
    }
    food = {
        "name": "record_meal_item",
        "description": "Logs one consumed food or drink item with a portion size.",
        "parameters": {
            "type": "dict",
            "required": ["item_name", "portion_amount", "portion_unit"],
            "properties": {
                "item_name": {"type": "string", "description": "Food or drink name."},
                "portion_amount": {"type": "float", "description": "Numeric amount consumed."},
                "portion_unit": {"type": "string", "enum": ["cup", "gram", "slice", "piece", "bowl"], "description": "Exact singular unit enum."},
                "meal_type": {"type": "string", "enum": ["breakfast", "lunch", "dinner", "snack"], "default": "snack", "description": "Meal category."},
            },
        },
    }
    device_set = {
        "name": "set_home_device",
        "description": "Sets a smart home device state in a named room.",
        "parameters": {
            "type": "dict",
            "required": ["room", "device", "power"],
            "properties": {
                "room": {"type": "string", "description": "Canonical room name."},
                "device": {"type": "string", "description": "Device name."},
                "power": {"type": "string", "enum": ["on", "off"], "description": "Desired power state."},
                "mode": {"type": "string", "enum": ["eco", "normal", "boost"], "default": "normal", "description": "Optional operating mode."},
            },
        },
    }
    device_get = {
        "name": "get_home_device",
        "description": "Gets the current status of a smart home device without changing it.",
        "parameters": {
            "type": "dict",
            "required": ["room", "device"],
            "properties": {
                "room": {"type": "string"},
                "device": {"type": "string"},
            },
        },
    }
    message = {
        "name": "send_team_message",
        "description": "Sends a short message to a teammate.",
        "parameters": {
            "type": "dict",
            "required": ["recipient", "body"],
            "properties": {
                "recipient": {"type": "string"},
                "body": {"type": "string"},
                "channel": {"type": "string", "enum": ["sms", "email", "slack"], "default": "slack"},
            },
        },
    }
    reminder = {
        "name": "create_task_reminder",
        "description": "Creates a personal task reminder.",
        "parameters": {
            "type": "dict",
            "required": ["title", "date"],
            "properties": {
                "title": {"type": "string"},
                "date": {"type": "string", "description": "Date in YYYY-MM-DD."},
                "priority": {"type": "string", "enum": ["low", "medium", "high"], "default": "medium"},
            },
        },
    }
    rows: list[dict[str, Any]] = []
    weather_cases = [
        ("请查一下里斯本和奥斯陆现在的天气。", [("Lisbon, Portugal", ""), ("Oslo, Norway", "")], "zh"),
        ("¿Qué tiempo hace ahora en Bogotá y en Quito? Usa Celsius.", [("Bogota, Colombia", "celsius"), ("Quito, Ecuador", "celsius")], "es"),
        ("Check current weather for Austin, Texas, Madison, Wisconsin, and Halifax, Canada.", [("Austin, TX", ""), ("Madison, WI", ""), ("Halifax, Canada", "")], "en"),
        ("서울과 부산의 현재 날씨를 알려줘.", [("Seoul, South Korea", ""), ("Busan, South Korea", "")], "ko"),
        ("Could you get the weather for Porto and Valencia in Celsius?", [("Porto, Portugal", "celsius"), ("Valencia, Spain", "celsius")], "en"),
        ("帮我看看墨尔本、珀斯和阿德莱德的天气。", [("Melbourne, Australia", ""), ("Perth, Australia", ""), ("Adelaide, Australia", "")], "zh"),
        ("Weather now for Albany, NY and Albany, Georgia.", [("Albany, NY", ""), ("Albany, GA", "")], "en"),
        ("Consulta el clima actual de Lima, Cusco y Arequipa.", [("Lima, Peru", ""), ("Cusco, Peru", ""), ("Arequipa, Peru", "")], "es"),
    ]
    for user, locations, language in weather_cases:
        answers = []
        calls = []
        for location, unit in locations:
            accepted = {"location": [location], "unit": ["", "fahrenheit"] if not unit else [unit]}
            answers.append({"fetch_city_conditions": accepted})
            args = f"location={location!r}" + (f", unit={unit!r}" if unit else "")
            calls.append(f"fetch_city_conditions({args})")
        rows.append(spec("live_parallel", [weather], user, answers, calls, ["parallel_alignment", "canonicalization", "default_value", "enum_exactness"], language))

    food_cases = [
        ("I had one bowl of tomato soup, two slices of rye toast, and a cup of tea.", [("tomato soup", 1, "bowl"), ("rye toast", 2, "slice"), ("tea", 1, "cup")]),
        ("Log 120 grams of tofu, 1 bowl of rice, and 3 pieces of pineapple for dinner.", [("tofu", 120, "gram", "dinner"), ("rice", 1, "bowl", "dinner"), ("pineapple", 3, "piece", "dinner")]),
        ("For my snack I ate 2 pieces of mochi and drank 1 cup of soy milk.", [("mochi", 2, "piece"), ("soy milk", 1, "cup")]),
        ("Add a lunch entry: 1 bowl lentil salad, 2 slices sourdough, 1 cup sparkling water.", [("lentil salad", 1, "bowl", "lunch"), ("sourdough", 2, "slice", "lunch"), ("sparkling water", 1, "cup", "lunch")]),
        ("Record breakfast: 80 grams oatmeal and a cup of black coffee.", [("oatmeal", 80, "gram", "breakfast"), ("black coffee", 1, "cup", "breakfast")]),
        ("I snacked on 4 pieces of dried apricot and 1 cup kefir.", [("dried apricot", 4, "piece"), ("kefir", 1, "cup")]),
        ("Please log 2 slices of melon and 1 bowl of yogurt.", [("melon", 2, "slice"), ("yogurt", 1, "bowl")]),
        ("For dinner, save 160 grams salmon, 1 bowl noodles, and 1 cup ginger tea.", [("salmon", 160, "gram", "dinner"), ("noodles", 1, "bowl", "dinner"), ("ginger tea", 1, "cup", "dinner")]),
    ]
    for user, items in food_cases:
        answers = []
        calls = []
        for item in items:
            name, amount, unit = item[:3]
            meal = item[3] if len(item) > 3 else ""
            accepted = {"item_name": [name], "portion_amount": [float(amount)], "portion_unit": [unit], "meal_type": [meal] if meal else ["", "snack"]}
            answers.append({"record_meal_item": accepted})
            args = f"item_name={name!r}, portion_amount={float(amount)!r}, portion_unit={unit!r}" + (f", meal_type={meal!r}" if meal else "")
            calls.append(f"record_meal_item({args})")
        rows.append(spec("live_parallel", [food], user, answers, calls, ["parallel_alignment", "default_value", "enum_exactness", "parameter_value_error"], "en"))

    mixed_cases = [
        ("Turn off the bedroom humidifier, turn on the office lamp in eco mode, and remind me to order filters on 2026-05-22.", [("set_home_device", {"room": "bedroom", "device": "humidifier", "power": "off"}), ("set_home_device", {"room": "office", "device": "lamp", "power": "on", "mode": "eco"}), ("create_task_reminder", {"title": "order filters", "date": "2026-05-22"})]),
        ("Message Priya that the demo moved to 3 PM, and create a high priority reminder for slides on 2026-06-01.", [("send_team_message", {"recipient": "Priya", "body": "The demo moved to 3 PM"}), ("create_task_reminder", {"title": "slides", "date": "2026-06-01", "priority": "high"})]),
        ("In the kitchen turn the fan off, then ask Jordan on email to bring the adapter.", [("set_home_device", {"room": "kitchen", "device": "fan", "power": "off"}), ("send_team_message", {"recipient": "Jordan", "body": "Please bring the adapter", "channel": "email"})]),
        ("Create reminders for passport renewal on 2026-07-03 and tax documents on 2026-06-10 with high priority.", [("create_task_reminder", {"title": "passport renewal", "date": "2026-07-03", "priority": "high"}), ("create_task_reminder", {"title": "tax documents", "date": "2026-06-10", "priority": "high"})]),
        ("Switch the nursery heater to boost and ping Mateo on Slack: heater is on.", [("set_home_device", {"room": "nursery", "device": "heater", "power": "on", "mode": "boost"}), ("send_team_message", {"recipient": "Mateo", "body": "heater is on"})]),
        ("请把客厅空气净化器打开，并提醒我 2026-05-30 更换滤芯。", [("set_home_device", {"room": "living room", "device": "air purifier", "power": "on"}), ("create_task_reminder", {"title": "replace filter", "date": "2026-05-30"})]),
        ("Email Nia that the report is uploaded, and turn off the hallway lights.", [("send_team_message", {"recipient": "Nia", "body": "The report is uploaded", "channel": "email"}), ("set_home_device", {"room": "hallway", "device": "lights", "power": "off"})]),
        ("Set the garage charger to normal mode and remind me to unplug the bike on 2026-05-18.", [("set_home_device", {"room": "garage", "device": "charger", "power": "on", "mode": "normal"}), ("create_task_reminder", {"title": "unplug the bike", "date": "2026-05-18"})]),
    ]
    function_map = {fn["name"]: fn for fn in [device_set, device_get, message, reminder]}
    for user, call_specs in mixed_cases:
        answers = []
        calls = []
        for name, params in call_specs:
            answer_params = {}
            for key, value in params.items():
                answer_params[key] = [value]
            if name == "set_home_device" and "mode" not in params:
                answer_params["mode"] = ["", "normal"]
            if name == "send_team_message" and "channel" not in params:
                answer_params["channel"] = ["", "slack"]
            if name == "create_task_reminder" and "priority" not in params:
                answer_params["priority"] = ["", "medium"]
            answers.append({name: answer_params})
            calls.append(f"{name}({', '.join(f'{k}={v!r}' for k, v in params.items())})")
        rows.append(spec("live_parallel_multiple", [device_set, device_get, message, reminder], user, answers, calls, ["wrong_function", "parallel_alignment", "default_value", "enum_exactness"], "zh" if "请" in user else "en"))

    # Non-live controls keep parallel structure but remove live multilingual/default pressure.
    # Use both single-function and mixed-function copies so the 96-prompt bank
    # can probe BFCL parallel and parallel_multiple separately.
    rows.extend(copy.deepcopy(rows[:4] + rows[16:20]))
    for row in rows[-8:]:
        row["category"] = "parallel_multiple" if row["category"].endswith("multiple") else "parallel"
        row["failure_tags"] = [tag for tag in row["failure_tags"] if tag != "canonicalization"] or ["parallel_alignment"]
    return rows


def spec(category: str, functions: list[dict[str, Any]], user: str, answers: list[dict[str, Any]], calls: list[str], failure_tags: list[str], language: str) -> dict[str, Any]:
    return {
        "category": category,
        "functions": copy.deepcopy(functions),
        "user": user,
        "answers": answers,
        "response": "[" + ", ".join(calls) + "]",
        "failure_tags": failure_tags,
        "language": language,
    }


def build_memory_rows(paper96: list[dict[str, Any]], *, count: int, seed: int, created_at: str) -> list[dict[str, Any]]:
    memory_rows = [row for row in paper96 if row.get("task") == "memory"]
    if len(memory_rows) < count:
        raise SystemExit(f"Not enough memory rows in paper96: need={count}, have={len(memory_rows)}")
    selected = sorted(memory_rows, key=lambda row: stable_hash({"seed": seed, "prompt_id": row.get("prompt_id")}))[:count]
    output = []
    for index, row in enumerate(selected):
        cloned = copy.deepcopy(row)
        cloned["source_row"] = cloned.get("source_row", index)
        cloned["tags"] = sorted(set(cloned.get("tags") or []) | {"eval_targeted", "memory_anchor", "paper96_reuse"})
        cloned["eval_targeted_calibration"] = {
            "created_at": created_at,
            "role": "memory_anchor_from_paper96",
            "reason": "Memory is not the current formal-eval bottleneck; keep previous HotpotQA-train calibration anchor.",
            "leakage_policy": "Existing paper96 memory rows from train/source manifests, not formal eval rows.",
        }
        output.append(cloned)
    return output


def build_code_rows(
    *,
    paper96: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    count: int,
    source_count: int,
    targeted_count: int | None,
    seed: int,
    created_at: str,
    profile: dict[str, Any],
    codecontests_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if source_count < 0:
        raise SystemExit("--code-source-count must be >= 0")
    if source_count > count:
        raise SystemExit(f"--code-source-count cannot exceed --code-count: source_count={source_count}, count={count}")
    if targeted_count is None:
        targeted_count = count - source_count
    if targeted_count < 0:
        raise SystemExit("--code-targeted-count must be >= 0")
    if source_count + targeted_count != count:
        raise SystemExit(
            f"Code counts must add up to --code-count: source_count={source_count}, targeted_count={targeted_count}, count={count}"
        )
    source_code_rows, source_blueprints = build_source_code_rows(
        paper96,
        count=source_count,
        seed=seed,
        created_at=created_at,
        profile=profile,
    )
    excluded_task_ids = {
        str(((row.get("reference") or {}).get("metadata") or {}).get("task_id"))
        for row in source_code_rows
        if isinstance(row.get("reference"), dict) and isinstance(row.get("reference", {}).get("metadata"), dict)
    }
    targeted_rows, targeted_blueprints = build_targeted_code_rows(
        source_rows,
        count=targeted_count,
        seed=seed,
        created_at=created_at,
        profile=profile,
        codecontests_path=codecontests_path,
        exclude_task_ids=excluded_task_ids,
    )
    return interleave_lists([source_code_rows, targeted_rows]), source_blueprints + targeted_blueprints


def build_source_code_rows(
    paper96: list[dict[str, Any]],
    *,
    count: int,
    seed: int,
    created_at: str,
    profile: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if count == 0:
        return [], []
    code_rows = [row for row in paper96 if row.get("task") == "code"]
    if len(code_rows) < count:
        raise SystemExit(f"Not enough source Code rows in paper96: need={count}, have={len(code_rows)}")

    def sort_key(row: dict[str, Any]) -> tuple[int, str]:
        selection = row.get("question_bank_selection", {}) if isinstance(row.get("question_bank_selection"), dict) else {}
        bucket = str(selection.get("bucket") or "")
        bucket_rank = {
            "mid_frontier": 0,
            "low_but_solvable": 1,
            "all_fail_partial": 2,
            "high_frontier": 3,
        }.get(bucket, 4)
        return bucket_rank, stable_hash({"seed": seed, "prompt_id": row.get("prompt_id")})

    selected = sorted(code_rows, key=sort_key)[:count]
    rows = []
    blueprints = []
    for index, row in enumerate(selected):
        cloned = copy.deepcopy(row)
        cloned["source_row"] = cloned.get("source_row", index)
        cloned["tags"] = sorted(set(cloned.get("tags") or []) | {"eval_targeted", "source_code_anchor", "paper96_reuse"})
        selection = cloned.get("question_bank_selection") if isinstance(cloned.get("question_bank_selection"), dict) else {}
        metadata = cloned.get("reference", {}).get("metadata", {}) if isinstance(cloned.get("reference"), dict) else {}
        cloned["eval_targeted_calibration"] = {
            "created_at": created_at,
            "role": "source_code_anchor_from_paper96",
            "reason": "Keep previously observed code frontier rows so the eval-targeted CURE-style probes do not replace the original paper96 code distribution.",
            "selection_bucket": selection.get("bucket"),
            "baseline_success_rate": selection.get("success_rate"),
            "baseline_mean_reward": selection.get("mean_reward"),
            "code_tags": ["paper96_source_anchor"],
            "case_study_source": "mixed with CodeContests-train CURE-style targeted rows after eval case study",
            "profile_snapshot": {
                "top_code_failure_tags": profile.get("code", {}).get("failure_tags", [])[:8],
                "top_code_tags": profile.get("code", {}).get("code_tags", [])[:8],
            },
            "leakage_policy": "Existing paper96 CodeContests-train source row; not LiveBench/LiveCodeBench official eval.",
        }
        rows.append(cloned)
        blueprints.append(
            {
                "format": "eval_targeted_code_blueprint_v1",
                "prompt_id": cloned["prompt_id"],
                "source": "paper96_code_source_anchor",
                "source_manifest": cloned.get("source_manifest"),
                "source_row": cloned.get("source_row"),
                "task_id": metadata.get("task_id") or metadata.get("question_id"),
                "selection_bucket": selection.get("bucket"),
                "baseline_success_rate": selection.get("success_rate"),
                "baseline_mean_reward": selection.get("mean_reward"),
                "code_tags": ["paper96_source_anchor"],
                "leakage_policy": cloned["eval_targeted_calibration"]["leakage_policy"],
            }
        )
    return rows, blueprints


def build_targeted_code_rows(
    source_rows: list[dict[str, Any]],
    *,
    count: int,
    seed: int,
    created_at: str,
    profile: dict[str, Any],
    codecontests_path: Path,
    exclude_task_ids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(source_rows, list):
        raise SystemExit("CodeContests source must be a JSON list")
    if count == 0:
        return [], []
    target_tags = ["math", "format_sensitive", "greedy", "array", "string", "simulation", "graph", "dynamic_programming"]
    pools: dict[str, list[tuple[float, int, dict[str, Any], list[str]]]] = defaultdict(list)
    for idx, raw in enumerate(source_rows):
        if not isinstance(raw, dict):
            continue
        task_id = str(raw.get("task_id"))
        if task_id in exclude_task_ids:
            continue
        question = str(raw.get("question") or "")
        tests_in = raw.get("test_input") or []
        tests_out = raw.get("test_output") or []
        if not question.strip() or len(tests_in) < 4 or len(tests_out) < 4:
            continue
        tags = code_tags(question)
        if not tags:
            continue
        score = code_priority_score(question, tags, tests_in, tests_out)
        for tag in tags:
            if tag in target_tags:
                pools[tag].append((score, idx, raw, tags))
    rng = random.Random(seed)
    selected_indices: set[int] = set()
    selected_items: list[tuple[int, dict[str, Any], list[str], str]] = []
    quotas = {
        "math": 7,
        "format_sensitive": 5,
        "greedy": 5,
        "array": 5,
        "string": 4,
        "simulation": 3,
        "graph": 2,
        "dynamic_programming": 1,
    }
    for tag, quota in quotas.items():
        pool = sorted(pools.get(tag, []), key=lambda item: (-item[0], stable_hash({"seed": seed, "idx": item[1], "tag": tag})))
        for _, idx, raw, tags in pool:
            if len([item for item in selected_items if item[3] == tag]) >= quota:
                break
            if idx in selected_indices:
                continue
            selected_indices.add(idx)
            selected_items.append((idx, raw, tags, tag))
    if len(selected_items) < count:
        fallback = []
        for tag_pool in pools.values():
            fallback.extend(tag_pool)
        fallback = sorted(fallback, key=lambda item: (-item[0], stable_hash({"seed": seed, "idx": item[1]})))
        for _, idx, raw, tags in fallback:
            if len(selected_items) >= count:
                break
            if idx in selected_indices:
                continue
            selected_indices.add(idx)
            selected_items.append((idx, raw, tags, "fallback"))
    selected_items = selected_items[:count]
    rng.shuffle(selected_items)

    rows = []
    blueprints = []
    for out_idx, (idx, raw, tags, bucket) in enumerate(selected_items):
        row = code_record(raw, idx=idx, source_path=codecontests_path, tags=tags, bucket=bucket, created_at=created_at, profile=profile)
        rows.append(row)
        blueprints.append(
            {
                "format": "eval_targeted_code_blueprint_v1",
                "prompt_id": row["prompt_id"],
                "source_row": idx,
                "task_id": raw.get("task_id"),
                "selection_bucket": bucket,
                "code_tags": tags,
                "num_tests": min(len(raw.get("test_input") or []), len(raw.get("test_output") or [])),
                "source": str(codecontests_path),
                "leakage_policy": row["eval_targeted_calibration"]["leakage_policy"],
            }
        )
    if len(rows) < count:
        raise SystemExit(f"Not enough targeted code rows: need={count}, have={len(rows)}")
    return rows, blueprints


def code_record(raw: dict[str, Any], *, idx: int, source_path: Path, tags: list[str], bucket: str, created_at: str, profile: dict[str, Any]) -> dict[str, Any]:
    question = str(raw.get("question") or "")
    messages = cure_code_messages(question)
    prompt_hash = stable_hash({"task": "code", "question": question, "source": str(source_path), "task_id": raw.get("task_id")})
    prompt_id = f"code_evaltarget__{prompt_hash[:16]}"
    metadata = {
        "source_dataset": "CodeContests_train",
        "source_path": str(source_path),
        "task_id": raw.get("task_id"),
        "question_id": raw.get("task_id"),
        "exe_method": raw.get("exe_method") or "stdin",
        "test_time_limit": raw.get("test_time_limit", 1),
        "test_input": list(raw.get("test_input") or []),
        "test_output": list(raw.get("test_output") or []),
        "code_tags": tags,
    }
    return {
        "prompt_id": prompt_id,
        "task": "code",
        "source": str(source_path),
        "source_row": idx,
        "split": "eval_targeted_calibration_train",
        "prompt": question,
        "messages": messages,
        "reference": {
            "answer": None,
            "response": "",
            "metadata": metadata,
        },
        "verifier": {"name": "cure_code_pass_rate", "config": {"source": "CodeContests_train"}},
        "tags": sorted({"eval_targeted", "codecontests_train", "cure_style", f"bucket:{bucket}", *[f"code_tag:{tag}" for tag in tags]}),
        "difficulty": None,
        "prompt_hash": prompt_hash,
        "eval_targeted_calibration": {
            "created_at": created_at,
            "role": "on_policy_eval_style_code_probe",
            "selection_bucket": bucket,
            "code_tags": tags,
            "case_study_source": "CURE candidates from eval case browser",
            "profile_snapshot": {
                "top_code_failure_tags": profile.get("code", {}).get("failure_tags", [])[:8],
                "top_code_tags": profile.get("code", {}).get("code_tags", [])[:8],
            },
            "leakage_policy": "Selected from CodeContests train only; no LiveBench/LiveCodeBench prompt, tests, generated code, or output copied.",
        },
    }


def cure_code_messages(question: str) -> list[dict[str, str]]:
    content = (
        "You need to think first then write python script. You should use input() to input and print() to output in your script. "
        "Your code should output the results based on the input read in, rather than generating the given test example.\n"
        "This is the problem:\n"
        f"{question}"
    )
    return [
        {"role": "system", "content": "You are a helpful assistant help user solve problems."},
        {"role": "user", "content": content},
    ]


def code_priority_score(question: str, tags: list[str], inputs: list[Any], outputs: list[Any]) -> float:
    score = 0.0
    tag_weights = {
        "math": 2.0,
        "format_sensitive": 1.6,
        "greedy": 1.4,
        "array": 1.2,
        "string": 1.1,
        "simulation": 1.0,
        "graph": 1.5,
        "dynamic_programming": 1.5,
    }
    score += sum(tag_weights.get(tag, 0.2) for tag in tags)
    score += min(2.0, len(inputs) / 10.0)
    lowered = question.lower()
    if "sample" in lowered or "example" in lowered:
        score += 0.4
    if any(marker in lowered for marker in ["mod", "minimum", "maximum", "shortest", "lexicographic"]):
        score += 0.5
    if len(question) > 1200:
        score += 0.3
    return score


def code_tags(text: str) -> list[str]:
    lowered = text.lower()
    tags = {"stdin_stdout"}
    math_words = ["sum", "product", "gcd", "mod", "integer", "prime", "divisible", "number", "formula", "probability"]
    if any(word in lowered for word in math_words):
        tags.add("math")
    if any(word in lowered for word in ["array", "sequence", "list", "permutation", "subarray", "prefix"]):
        tags.add("array")
    if any(word in lowered for word in ["string", "substring", "character", "lexicographic", "palindrome"]):
        tags.add("string")
    if any(word in lowered for word in ["greedy", "minimum", "maximum", "at most", "at least", "choose", "remove", "operation"]):
        tags.add("greedy")
    if any(word in lowered for word in ["simulate", "simulation", "move", "turn", "round", "step", "game"]):
        tags.add("simulation")
    if any(word in lowered for word in ["graph", "tree", "edge", "vertex", "node", "path", "connected", "shortest"]):
        tags.add("graph")
    if any(word in lowered for word in ["dynamic programming", "dp", "subsequence"]):
        tags.add("dynamic_programming")
    if any(word in lowered for word in ["print", "output", "format", "yes", "no", "case"]):
        tags.add("format_sensitive")
    return sorted(tags)


def round_robin_interleave(groups: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    output = []
    max_len = max((len(rows) for rows in groups.values()), default=0)
    for idx in range(max_len):
        for task in TASKS:
            rows = groups.get(task, [])
            if idx < len(rows):
                output.append(rows[idx])
    return output


def interleave_lists(groups: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    output = []
    max_len = max((len(rows) for rows in groups), default=0)
    for idx in range(max_len):
        for rows in groups:
            if idx < len(rows):
                output.append(rows[idx])
    return output


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def top_counter(values, *, limit: int = 20) -> list[tuple[str, int]]:
    return Counter(str(value) for value in values if value is not None).most_common(limit)


def render_readme(summary: dict[str, Any]) -> str:
    counts = summary["counts"]
    lines = [
        "# Eval-Targeted 96 Calibration",
        "",
        f"生成时间：`{summary['created_at']}`",
        "",
        "## 定位",
        "",
        "这版数据类比 paper96 的 32/32/32 结构，但 Tool 和 Code 改为由 formal-eval case study 反推的能力分布。它不是 official eval 题拷贝。",
        "",
        "## 文件",
        "",
        f"- prompts: `{summary['outputs']['prompts']}`",
        f"- tool blueprints: `{summary['outputs']['tool_blueprints']}`",
        f"- code blueprints: `{summary['outputs']['code_blueprints']}`",
        f"- summary: `{summary['outputs']['summary']}`",
        "",
        "## 计数",
        "",
        "| task | rows |",
        "|---|---:|",
    ]
    for task, value in counts["tasks"].items():
        lines.append(f"| {task} | {value} |")
    lines.extend(
        [
            "",
            "## 泄漏控制",
            "",
        f"- Tool: {summary['leakage_policy']['tool']}",
            f"- Memory: {summary['leakage_policy']['memory']}",
            f"- Code: {summary['leakage_policy']['code']}",
            "",
            "## 使用建议",
            "",
            "先用小样本 smoke 验证 RewardRouter 能给 Tool/Code 产生非饱和信号，再进入正式 20-iter gate 训练。",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-candidates", default=DEFAULT_CASE_CANDIDATES)
    parser.add_argument("--paper96", default=DEFAULT_PAPER96)
    parser.add_argument("--codecontests", default=DEFAULT_CODECONTESTS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tool-source-count", type=int, default=16)
    parser.add_argument("--tool-synthetic-count", type=int, default=16)
    parser.add_argument("--memory-count", type=int, default=32)
    parser.add_argument("--code-count", type=int, default=32)
    parser.add_argument(
        "--code-source-count",
        type=int,
        default=0,
        help="Number of paper96 Code source-anchor rows to mix into the Code portion. Default 0 preserves the original builder behavior.",
    )
    parser.add_argument(
        "--code-targeted-count",
        type=int,
        default=None,
        help="Number of eval-targeted CodeContests-train rows. Default is --code-count minus --code-source-count.",
    )
    parser.add_argument("--seed", type=int, default=20260517)
    return parser.parse_args()


if __name__ == "__main__":
    main()
