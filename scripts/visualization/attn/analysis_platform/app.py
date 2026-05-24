#!/usr/bin/env python3
"""Streamlit RCRF Mechanism Discovery Workbench."""

from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

from rcrf_schema import (  # noqa: E402
    DEFAULT_INPUT_PATHS,
    MODULE_ORDER,
    RCRF_CANDIDATES,
    MechanismData,
    load_mechanism_data,
    module_family,
    summarize_research_questions,
)


st.set_page_config(page_title="RCRF Mechanism Discovery", layout="wide")


METRIC_LABELS = {
    "alpha": "gate alpha / 模块注入系数",
    "signed_effect": "signed_effect = mean[-<g, DeltaW h>] / 一阶收益",
    "expression": "expression = mean||DeltaW h||^2 / 响应表达能量",
    "cross_harm": "cross_harm / 跨任务负收益",
    "conflict_score": "conflict_score / residual 方向冲突",
    "positive_fraction": "positive_fraction / 正向比例",
    "cosine": "residual cosine / induced residual 余弦",
    "shared_positive_effect": "shared positive effect / 共享正收益",
}

LATEST_ANALYSIS_DIR = Path(
    "/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/memory_code_conflict_20260521"
)
LATEST_CONTRAST_GATES_DIR = Path(
    "/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/contrast_gates"
)
PAIRWISE_ZERO_DIR = Path(
    "/tmp/shared-storage/ExpertGym/rcrf/code_hurt_subset_20260521/analysis/pairwise_zero_diagnostics_20260523"
)
LATEST_VARIANT_ORDER = [
    "v1_code_only",
    "v2_spanaware",
    "v3_memory_hard_floor",
    "v4_memory_utility_floor",
    "v5_memory_top_preserve",
]
LATEST_VARIANT_LABELS = {
    "v1_code_only": "v1 final-code contrast",
    "v2_spanaware": "v2 span-aware conservative",
    "v3_memory_hard_floor": "v3 memory hard floor",
    "v4_memory_utility_floor": "v4 memory utility floor",
    "v5_memory_top_preserve": "v5 memory top preserve",
}
LATEST_GATE_NAMES = {
    "rcrf_code_contrast_v1": "v1_code_only",
    "rcrf_code_spanaware_conservative_v2": "v2_spanaware",
    "rcrf_code_spanaware_memory_preserve_v3": "v3_memory_hard_floor",
    "rcrf_code_spanaware_memory_utility_preserve_v4": "v4_memory_utility_floor",
    "rcrf_code_spanaware_memory_top_preserve_v5": "v5_memory_top_preserve",
}


def main() -> None:
    st.title("RCRF Mechanism Discovery Workbench")
    st.caption("只围绕 RCRF：在 attention q/k/v/o 与 MLP gate/up/down 上观察 DeltaW h 的表达、收益、冲突和 gate 抑制。")

    data = sidebar_data_loader()
    if data.used_example:
        st.warning("没有在所选路径中找到完整真实 RCRF 机制文件，当前使用 96 prompt example 数据启动，页面不会崩溃。")

    render_project_brief(data)

    page = st.sidebar.radio(
        "页面",
        [
            "0. Latest Findings / 最新实验发现",
            "1. Module Mechanism Map / 模块机制地图",
            "2. Prompt Sensitivity / Prompt 与 Span 敏感性",
            "3. Interference Explorer / 跨专家干扰",
            "4. Gate Sparsity View / Gate 稀疏与抑制",
            "5. Research Questions / 下一步机制问题",
            "6. Pairwise-Zero / 两两置零诊断",
        ],
    )

    if page.startswith("0."):
        page_latest_findings(data)
    elif page.startswith("1."):
        page_module_map(data)
    elif page.startswith("2."):
        page_prompt_sensitivity(data)
    elif page.startswith("3."):
        page_interference(data)
    elif page.startswith("4."):
        page_gate_sparsity(data)
    elif page.startswith("5."):
        page_research_questions(data)
    else:
        page_pairwise_zero()


@st.cache_data(show_spinner=False)
def load_cached(paths_text: str, include_defaults: bool) -> MechanismData:
    paths = [line.strip() for line in paths_text.splitlines() if line.strip()]
    return load_mechanism_data(paths, include_defaults=include_defaults, example_on_empty=True)


def sidebar_data_loader() -> MechanismData:
    st.sidebar.header("数据")
    include_defaults = st.sidebar.checkbox("读取默认 RCRF 真实路径", value=True)
    default_text = "\n".join(str(path) for path in DEFAULT_INPUT_PATHS)
    paths_text = st.sidebar.text_area("额外数据路径", value="" if include_defaults else default_text, height=140)
    data = load_cached(paths_text, include_defaults)
    with st.sidebar.expander("已加载来源", expanded=False):
        for note in data.source_notes[-40:]:
            st.write(note)
    st.sidebar.metric("Residual 明细", f"{len(data.residual):,}")
    st.sidebar.metric("Gate 明细", f"{len(data.gate):,}")
    st.sidebar.metric("Interference 明细", f"{len(data.interference):,}")
    st.sidebar.metric("Eval 明细", f"{len(data.eval):,}")
    with st.sidebar.expander("怎么读这个站", expanded=False):
        st.markdown(
            """
            1. 先看 **Module Mechanism Map**：找 layer-module 热点。
            2. 再看 **Prompt Sensitivity**：判断热点是不是只由少数 prompt/span 驱动。
            3. 再看 **Interference Explorer**：确认 tool/memory/code 是否在 q/k 或 MLP 上互相冲突。
            4. 再看 **Gate Sparsity View**：检查 RCRF 为什么压低或放大某些模块。
            5. 最后看 **Research Questions**：把异常模式转成下一轮 ablation 或论文图。
            """
        )
    return data


def render_project_brief(data: MechanismData) -> None:
    with st.expander("中文阅读指南：这个 Workbench 要回答什么", expanded=True):
        st.markdown(
            """
            **核心问题**：task vector 合并不应该只看 expert 级别系数，而要看某个 expert 在某个模块上对当前响应实际诱导出的 residual：

            `u_{e,m,t} = Delta W_{e,m} h_{m,t}`

            RCRF 关注两个量：

            - `signed_effect = mean_t[- <g_{m,t}, Delta W_{e,m} h_{m,t}>]`：正值表示这个模块残差一阶上降低 teacher-forced response loss，负值表示它在当前 span 上有害。
            - `expression = mean_t ||Delta W_{e,m} h_{m,t}||^2`：表示这个残差是否真的在当前 prompt/span 上被激活。

            **当前项目理解**：

            - Memory 的 MLP residual 方向最清晰，表达能量也最大；它容易成为强能力通道，但也可能主导合并。
            - Tool 更依赖 tool-call behavior span 和 late attention/MLP 的格式稳定性；只看源分布 ToolRL 不够，BFCL live/parallel 更能暴露格式与泛化。
            - Code 当前不能只靠 positive code-block 的 signed utility 下结论；LiveCodeBench 更像 hidden-test 程序正确性，需要 prompt/code span 与 pass-fail contrast 才能补足。
            - RCRF 的主价值不是训练新能力，而是发现哪些 `DeltaW h` 是能力表达、哪些只是能量大、冲突大或符号不稳定。

            **最新实验更新（2026-05-21）**：

            - Code hurt 子集已经切成 LiveBench 16 条 + LiveCodeBench 16 条，用来专门观察 RCRF 失败而 TA0.75/TA1.0 可修的 case。
            - 当前最强正结果是 `v2_spanaware`：用 prompt / reasoning / final-code 的 pass-fail contrast 做 conservative routing，而不是只推 code expert。
            - 最清楚的机制发现是 source/span 条件化冲突：`LB_prompt` 与 `LCB_prompt` 的 Pearson 约为 -0.995，说明 Code 不是一个平滑统一方向。
            - `v3/v4` 证伪了粗粒度 memory 保护：expert-level hard floor 或小 memory utility floor 都不能替代 residual key 级证据。

            **建议阅读顺序**：先找热点模块，再看 prompt/span 是否稳定，再查 expert pair 干扰，最后看 gate 是否合理压低这些位置。
            """
        )
        cols = st.columns(4)
        cols[0].metric("Residual records", f"{len(data.residual):,}")
        cols[1].metric("Interference records", f"{len(data.interference):,}")
        cols[2].metric("Gate records", f"{len(data.gate):,}")
        cols[3].metric("Eval records", f"{len(data.eval):,}")


@st.cache_data(show_spinner=False)
def load_latest_findings() -> dict[str, Any]:
    def read_csv(name: str) -> pd.DataFrame:
        path = LATEST_ANALYSIS_DIR / name
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)

    summary_path = LATEST_ANALYSIS_DIR / "analysis_summary.json"
    summary: dict[str, Any] = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = {"parse_error": str(summary_path)}

    gate_summaries: dict[str, str] = {}
    for gate_name, variant in LATEST_GATE_NAMES.items():
        summary_file = LATEST_CONTRAST_GATES_DIR / gate_name / "summary.md"
        if summary_file.exists():
            gate_summaries[variant] = summary_file.read_text(encoding="utf-8", errors="ignore")

    return {
        "summary": summary,
        "code_metrics": read_csv("code_hurt_metrics.csv"),
        "source_conflicts": read_csv("source_conflict_pairs.csv"),
        "gate_memory": read_csv("gate_memory_summary.csv"),
        "memory_delta_rows": read_csv("memory_delta_rows.csv"),
        "preserve_rows": read_csv("preserve_rows.csv"),
        "gate_summaries": gate_summaries,
        "analysis_dir_exists": LATEST_ANALYSIS_DIR.exists(),
    }


def page_latest_findings(data: MechanismData) -> None:
    latest = load_latest_findings()
    code_metrics = latest["code_metrics"].copy()
    source_conflicts = latest["source_conflicts"].copy()
    gate_memory = latest["gate_memory"].copy()
    memory_delta_rows = latest["memory_delta_rows"].copy()
    preserve_rows = latest["preserve_rows"].copy()

    st.subheader("Latest Findings / 最新实验发现")
    st.markdown(
        f"""
        这一页跟进当前代码库最新 RCRF 实验，只展示真实落盘产物；如果某个候选还没有评测结果，会明确标成“未验证”。

        当前读取目录：

        `{LATEST_ANALYSIS_DIR}`

        **当前主结论**：Code hurt 的有效修复不是来自单纯增大 code expert，而是来自
        **outcome-aware pass/fail contrast + source/span-aware conservative residual routing**。
        v2 是当前最干净的正结果；v3/v4 的失败说明粗粒度保护 Memory expert 不够，下一步要做 residual key 级的
        Tool/Memory/Code 行为 utility 证据表。
        """
    )
    if not latest["analysis_dir_exists"]:
        st.warning("最新实验目录不存在，下面只会展示空表或 example 数据。")

    render_latest_kpis(code_metrics, source_conflicts, gate_memory)
    render_leader_briefing(code_metrics, source_conflicts, gate_memory)
    render_paper_narrative(data, code_metrics, source_conflicts, gate_memory, memory_delta_rows, preserve_rows)
    render_latest_terms_with_examples(data, code_metrics, source_conflicts, gate_memory, memory_delta_rows, preserve_rows)

    st.markdown("### 1. Code hurt 修复效果")
    st.markdown(
        """
        Code hurt 子集只包含“RCRF 失败、TA 系列可修”的困难样本，因此这里看的是 **能否补回当前 RCRF 的 code 弱点**，
        不是完整 benchmark 分数。`pass_any` 表示每条题多次候选中至少一次通过；`candidate_pass_rate` 表示所有候选的平均通过率；
        `test_point_rate` 更接近 hidden tests 的细粒度通过比例。
        """
    )
    if code_metrics.empty:
        st.info("没有找到 code_hurt_metrics.csv。")
    else:
        code_metrics["variant_label"] = code_metrics["variant"].map(LATEST_VARIANT_LABELS).fillna(code_metrics["variant"])
        metric_choice = st.selectbox(
            "Code hurt 指标",
            ["pass_any", "candidate_pass_rate", "test_point_rate"],
            format_func=lambda item: {
                "pass_any": "pass_any / 每题至少一次通过",
                "candidate_pass_rate": "candidate_pass_rate / 候选平均通过率",
                "test_point_rate": "test_point_rate / hidden test-point 通过率",
            }[item],
        )
        fig_code = px.bar(
            code_metrics,
            x="variant_label",
            y=metric_choice,
            color="dataset",
            barmode="group",
            text=metric_choice,
            category_orders={"variant_label": [LATEST_VARIANT_LABELS[v] for v in LATEST_VARIANT_ORDER]},
            title=f"Code Hurt Repair: {metric_choice}",
            labels={
                "variant_label": "gate variant / 候选 gate",
                metric_choice: f"{metric_choice} / 通过率",
                "dataset": "dataset / 数据集",
            },
            hover_data=["num_cases", "pass_any_count", "pass_count", "source"],
        )
        fig_code.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig_code.update_yaxes(range=[0, min(1.05, max(0.1, float(code_metrics[metric_choice].max()) * 1.22))])
        fig_code.update_layout(margin=dict(l=10, r=10, t=60, b=80), xaxis_tickangle=-20)
        st.plotly_chart(fig_code, use_container_width=True)
        export_controls(fig_code, code_metrics, f"latest_code_hurt_{metric_choice}")
        render_chart_guide(
            "这张 Code hurt 图怎么看",
            [
                "每个柱子是一个 gate variant 在 LiveBench/LiveCodeBench hurt16 上的真实评测结果。",
                "v1 只用 final-code contrast；v2 加入 prompt/reasoning/final-code span-aware conservative aggregation。",
                "v3/v4 在 v2 上加入 memory 保护，用来测试 memory side-effect 是否能靠粗粒度 floor 解决。",
            ],
            [
                "v2 同时提升 LB 和 LCB，是当前最干净的正结果。",
                "v3/v4 在 LCB 明显回退，说明 memory expert hard floor 或小 utility floor 粒度过粗。",
                "如果某个 variant 只提升 pass_any 而 test_point_rate 不升，可能只是偶然 candidate 命中，不是稳定代码能力。",
            ],
            "下一步不是继续 sweep gate 阈值，而是把 source/span contrast 与 Tool/Memory 行为 utility 对齐到同一 residual evidence table。"
        )
        st.dataframe(
            code_metrics.sort_values(["dataset", "variant"], key=lambda col: col.map({v: i for i, v in enumerate(LATEST_VARIANT_ORDER)}).fillna(99) if col.name == "variant" else col),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("### 2. Source / Span 方向冲突")
    st.markdown(
        """
        这里比较的是同一个 residual key 在不同 code source/span 上的 pass-fail contrast 方向。
        如果两个 source/span 的 Pearson 很负或 conflict_rate 很高，说明这个 residual key 对一种 code 分布有用，
        对另一种 code 分布可能相反，不能用单一 code gate 标量解决。
        """
    )
    if source_conflicts.empty:
        st.info("没有找到 source_conflict_pairs.csv。")
    else:
        conflict_metric = st.selectbox(
            "Source/span 冲突指标",
            ["pearson", "conflict_rate", "conflict_count"],
            format_func=lambda item: {
                "pearson": "Pearson / contrast 相关性",
                "conflict_rate": "conflict_rate / 符号冲突比例",
                "conflict_count": "conflict_count / 冲突 key 数",
            }[item],
        )
        matrix = source_conflict_matrix(source_conflicts, conflict_metric)
        fig_conflict = px.imshow(
            matrix,
            aspect="auto",
            color_continuous_scale="RdBu_r" if conflict_metric == "pearson" else "Magma",
            color_continuous_midpoint=0.0 if conflict_metric == "pearson" else None,
            title=f"Code Source/Span Conflict: {conflict_metric}",
            labels={
                "x": "source/span",
                "y": "source/span",
                "color": f"{conflict_metric} / 冲突指标",
            },
            text_auto=".3f" if conflict_metric != "conflict_count" else ".0f",
        )
        fig_conflict.update_layout(margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig_conflict, use_container_width=True)
        export_controls(fig_conflict, matrix.reset_index().rename(columns={"index": "source"}), f"latest_source_conflict_{conflict_metric}")
        render_chart_guide(
            "这张 source/span 冲突图怎么看",
            [
                "横纵轴是 Code 的不同来源和 span，例如 LB_prompt、LCB_code。",
                "Pearson 负值表示同一 residual key 的 contrast 方向整体相反；conflict_rate 表示有多少 key 正负号冲突。",
                "它解释为什么 Code 看起来像异类：Code 能力被 prompt 理解、边界条件、final code 多个 span 分摊，而且不同 benchmark 的方向不一致。",
            ],
            [
                "LB_prompt vs LCB_prompt 接近 -1：prompt residual 高度分布条件化，不能盲目对齐。",
                "LB_code vs LCB_code 冲突超过一半：final-code span 也不是统一方向。",
                "冲突高但 effect 很小的位置要谨慎，不应只按符号做强 gate。",
            ],
            "把冲突 pair 对应的 residual key 与 Memory/Tool 行为 utility 合并，找出哪些 key 应该回到 base，而不是强推或强压。"
        )
        st.dataframe(
            source_conflicts.sort_values(["conflict_rate", "conflict_count"], ascending=False),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("### 3. Memory side-effect 与保护策略")
    st.markdown(
        """
        这部分回答：Code 修复是否靠牺牲 Memory？现有结果显示 v2 对 Tool 基本稳定、Memory 有小幅 side-effect；
        但 v3/v4 说明“把 memory expert 负向 delta 全保护”或“用小 memory utility floor”并不能稳定解决。
        更合理的粒度是具体 residual key 在 Memory 行为 span 上是否有 utility。
        """
    )
    if gate_memory.empty:
        st.info("没有找到 gate_memory_summary.csv。")
    else:
        gate_memory["variant_label"] = gate_memory["variant"].map(LATEST_VARIANT_LABELS).fillna(gate_memory["variant"])
        count_cols = ["memory_positive", "memory_negative", "protected_negative_overlay", "preserve_utility_floor"]
        count_cols = [col for col in count_cols if col in gate_memory.columns]
        memory_counts = gate_memory.melt(
            id_vars=["variant", "variant_label"],
            value_vars=count_cols,
            var_name="memory_delta_type",
            value_name="count",
        )
        fig_memory = px.bar(
            memory_counts,
            x="variant_label",
            y="count",
            color="memory_delta_type",
            barmode="group",
            category_orders={"variant_label": [LATEST_VARIANT_LABELS[v] for v in LATEST_VARIANT_ORDER]},
            title="Memory Gate Delta / Protection Counts",
            labels={
                "variant_label": "gate variant / 候选 gate",
                "count": "residual key count / key 数",
                "memory_delta_type": "delta type / Memory 改写类型",
            },
        )
        fig_memory.update_layout(margin=dict(l=10, r=10, t=60, b=80), xaxis_tickangle=-20)
        st.plotly_chart(fig_memory, use_container_width=True)
        export_controls(fig_memory, memory_counts, "latest_memory_gate_counts")
        render_chart_guide(
            "这张 Memory gate 图怎么看",
            [
                "每个柱子统计一个 variant 对 memory expert residual key 的改写数量。",
                "memory_negative 是被 code contrast 认为应该压低的 memory key；protected/floor 是保护策略保留下来的 key。",
                "v3 memory_negative 变成 0，是 hard floor 的预期效果，但 LCB 指标反而下降。",
            ],
            [
                "Memory delta 少不等于好：v3 保护太多，说明某些 memory residual 对 LCB code 是有害的。",
                "v4 只保护 21 个 floor key 仍然不够，说明小 signature 太噪或覆盖不足。",
                "v2 有 129 个 memory_negative 但 Code 最好，提示需要 key/span 级证据，而不是 expert 级直觉。",
            ],
            "下一步要在 Memory 完整轨迹 update/final span 上重建 utility，再与 Code source/span contrast 做冲突表。"
        )
        st.dataframe(gate_memory, use_container_width=True, hide_index=True)

    st.markdown("### 4. 具体 residual key 证据")
    left, right = st.columns(2)
    with left:
        st.markdown("**Memory delta 最大的 key**")
        if memory_delta_rows.empty:
            st.info("没有 memory_delta_rows.csv。")
        else:
            cols = [col for col in ["variant", "layer", "family", "param_name", "delta", "reason", "score", "preserve_utility"] if col in memory_delta_rows.columns]
            st.dataframe(
                memory_delta_rows.assign(abs_delta=memory_delta_rows["delta"].abs())
                .sort_values("abs_delta", ascending=False)[cols + ["abs_delta"]]
                .head(50),
                use_container_width=True,
                hide_index=True,
            )
    with right:
        st.markdown("**被 Memory utility floor 保护的 key**")
        if preserve_rows.empty:
            st.info("没有 preserve_rows.csv。")
        else:
            cols = [col for col in ["variant", "layer", "family", "param_name", "reason", "preserve_task", "preserve_utility", "positive_fraction"] if col in preserve_rows.columns]
            st.dataframe(
                preserve_rows.sort_values("preserve_utility", ascending=False)[cols].head(50),
                use_container_width=True,
                hide_index=True,
            )

    st.markdown("### 5. 当前可以写成论文 insight 的发现")
    st.markdown(
        """
        **最有规律性的发现**：

        - `task vector` 不是 task-pure direction；一个 expert residual 里同时有能力、冗余和分布特异 harm。
        - Code 的方向最不平滑，尤其 prompt span 在 LiveBench 与 LiveCodeBench 之间强烈翻转。
        - v2 表明 training-free 的 outcome-aware residual routing 已经能修复一部分 Code hurt case。
        - v3/v4 表明 expert-level preserve 不够，机制粒度必须落到 residual key + behavior span。
        - 下一版方法应收束为：稳定正 contrast 保留/增强，稳定负 contrast 压低，source/span 或 task 间冲突回到 base/保守处理。
        """
    )
    gate_summaries = latest["gate_summaries"]
    if gate_summaries:
        with st.expander("Gate summary 原始摘要 / 真实 summary.md", expanded=False):
            for variant in [v for v in LATEST_VARIANT_ORDER if v in gate_summaries]:
                st.markdown(f"#### {LATEST_VARIANT_LABELS.get(variant, variant)}")
                st.code(gate_summaries[variant][:2600], language="markdown")
                if variant == "v5_memory_top_preserve" and code_metrics is not None and not code_metrics.empty:
                    if variant not in set(code_metrics["variant"]):
                        st.caption("v5 已生成 gate summary，但当前 code_hurt_metrics.csv 里还没有 v5 快速验证结果。")


def render_latest_kpis(code_metrics: pd.DataFrame, source_conflicts: pd.DataFrame, gate_memory: pd.DataFrame) -> None:
    cols = st.columns(4)
    num_cases = int(code_metrics["num_cases"].sum()) if not code_metrics.empty and "num_cases" in code_metrics else 0
    unique_cases = int(code_metrics.groupby("dataset")["num_cases"].max().sum()) if not code_metrics.empty else 0
    cols[0].metric("Code hurt cases", f"{unique_cases or num_cases}")

    if not code_metrics.empty and {"variant", "dataset", "pass_any"}.issubset(code_metrics.columns):
        v2_lcb = code_metrics[(code_metrics["variant"] == "v2_spanaware") & (code_metrics["dataset"] == "LiveCodeBench")]
        value = float(v2_lcb["pass_any"].iloc[0]) if not v2_lcb.empty else float(code_metrics["pass_any"].max())
        cols[1].metric("Best current LCB pass_any", f"{value:.4f}")
    else:
        cols[1].metric("Best current LCB pass_any", "N/A")

    if not source_conflicts.empty:
        top = source_conflicts.sort_values("conflict_rate", ascending=False).iloc[0]
        cols[2].metric("Top source conflict", f"{float(top['conflict_rate']):.2%}", f"{top['left']} vs {top['right']}")
    else:
        cols[2].metric("Top source conflict", "N/A")

    if not gate_memory.empty and "memory_negative" in gate_memory:
        v2 = gate_memory[gate_memory["variant"] == "v2_spanaware"]
        neg = int(v2["memory_negative"].iloc[0]) if not v2.empty else int(gate_memory["memory_negative"].max())
        cols[3].metric("v2 memory negative keys", f"{neg}")
    else:
        cols[3].metric("v2 memory negative keys", "N/A")


def render_leader_briefing(code_metrics: pd.DataFrame, source_conflicts: pd.DataFrame, gate_memory: pd.DataFrame) -> None:
    st.markdown("### Leader Briefing / 项目负责人汇报")
    st.markdown(
        """
        **一句话结论**：这轮实验已经把 Code 弱点从“可能是 code gate 太低”推进到更清楚的机制判断：
        Code 不是单一能力方向，必须按 **成功/失败结果** 和 **prompt/reasoning/final-code 片段** 做保守路由。
        当前最值得保留的正结果是 `v2_spanaware`；`v3/v4` 是有价值的反例，说明粗粒度保护 Memory expert 不是正确解法。
        """
    )

    v1_lb = latest_metric_row(code_metrics, "v1_code_only", "LiveBench")
    v2_lb = latest_metric_row(code_metrics, "v2_spanaware", "LiveBench")
    v1_lcb = latest_metric_row(code_metrics, "v1_code_only", "LiveCodeBench")
    v2_lcb = latest_metric_row(code_metrics, "v2_spanaware", "LiveCodeBench")
    v3_lcb = latest_metric_row(code_metrics, "v3_memory_hard_floor", "LiveCodeBench")
    v4_lcb = latest_metric_row(code_metrics, "v4_memory_utility_floor", "LiveCodeBench")

    cols = st.columns(3)
    if v1_lb is not None and v2_lb is not None:
        cols[0].metric(
            "LiveBench hurt: 至少一次通过",
            pass_any_text(v2_lb),
            delta=pass_any_delta_text(v1_lb, v2_lb),
        )
    else:
        cols[0].metric("LiveBench hurt: 至少一次通过", "N/A")
    if v1_lcb is not None and v2_lcb is not None:
        cols[1].metric(
            "LiveCodeBench hurt: 候选通过数",
            pass_count_text(v2_lcb),
            delta=pass_count_delta_text(v1_lcb, v2_lcb),
        )
    else:
        cols[1].metric("LiveCodeBench hurt: 候选通过数", "N/A")
    if not source_conflicts.empty:
        top = source_conflicts.sort_values("conflict_rate", ascending=False).iloc[0]
        cols[2].metric(
            "最强 source/span 冲突",
            f"{int(top['conflict_count'])}/{int(top['overlap_count'])}",
            delta=f"{top['left']} vs {top['right']}",
        )
    else:
        cols[2].metric("最强 source/span 冲突", "N/A")

    progress_rows = [
        {
            "模块": "数据构造",
            "已经完成": "Code hurt 子集 32 条：LiveBench 16 + LiveCodeBench 16。",
            "为什么重要": "只看 RCRF 当前会失败、但 TA 系列有机会做对的 case，直接定位短板。",
        },
        {
            "模块": "正向结果",
            "已经完成": "v2_spanaware 在 LB/LCB 两个 hurt 子集都有提升。",
            "为什么重要": "说明无需引入训练，单靠 pass/fail + span-aware gate 已能恢复一部分 Code 行为。",
        },
        {
            "模块": "反例验证",
            "已经完成": "v3/v4 的 Memory 保护策略已跑完快速验证。",
            "为什么重要": "它们把 LCB 拉回，证伪“直接保护整个 Memory expert”这条粗粒度路线。",
        },
        {
            "模块": "机制证据",
            "已经完成": "对 588 个 residual key 比较 LB/LCB prompt/code/reasoning 的方向冲突。",
            "为什么重要": "证明 Code 方向强烈依赖 source/span，不适合继续调单一 code 系数。",
        },
    ]
    st.dataframe(pd.DataFrame(progress_rows), use_container_width=True, hide_index=True)

    result_rows = []
    if v1_lb is not None and v2_lb is not None:
        result_rows.append(
            {
                "结果": "LiveBench hurt 修复",
                "旧方案": f"v1: {pass_any_text(v1_lb)} 题至少一次通过",
                "新方案": f"v2: {pass_any_text(v2_lb)} 题至少一次通过",
                "项目判断": "有提升，但还不是最终方案；LB 需要继续看 prompt/reasoning span。",
            }
        )
    if v1_lcb is not None and v2_lcb is not None:
        result_rows.append(
            {
                "结果": "LiveCodeBench hurt 修复",
                "旧方案": f"v1: {pass_count_text(v1_lcb)} 候选通过",
                "新方案": f"v2: {pass_count_text(v2_lcb)} 候选通过",
                "项目判断": "最强正信号；test-point rate 也从 "
                f"{float(v1_lcb['test_point_rate']):.3f} 到 {float(v2_lcb['test_point_rate']):.3f}。",
            }
        )
    if v2_lcb is not None and v3_lcb is not None and v4_lcb is not None:
        result_rows.append(
            {
                "结果": "Memory 保护反例",
                "旧方案": f"v2: {pass_any_text(v2_lcb)} LCB 至少一次通过",
                "新方案": f"v3/v4: {pass_any_text(v3_lcb)} / {pass_any_text(v4_lcb)}",
                "项目判断": "不能用 expert-level floor 保护 Memory；需要 residual key + behavior span 粒度。",
            }
        )
    if not source_conflicts.empty:
        top = source_conflicts.sort_values("conflict_rate", ascending=False).iloc[0]
        result_rows.append(
            {
                "结果": "Code 方向不统一",
                "旧方案": "假设存在一个统一 code 方向",
                "新方案": f"{top['left']} vs {top['right']} 有 {int(top['conflict_count'])}/{int(top['overlap_count'])} 个 key 方向相反",
                "项目判断": "应停止把 Code 当单一 scalar gate，转向 outcome-aware residual routing。",
            }
        )
    if result_rows:
        st.markdown("**核心结果表：我会这样向项目组汇报**")
        st.dataframe(pd.DataFrame(result_rows), use_container_width=True, hide_index=True)

    decision_rows = [
        {
            "决策点": "是否继续调 v3/v4 阈值",
            "建议": "不作为主线。",
            "依据": "v3/v4 已说明 Memory 保护粒度太粗，继续调阈值大概率只是局部 trade-off。",
        },
        {
            "决策点": "下一步该补什么证据",
            "建议": "构建统一 residual evidence table。",
            "依据": "同一 key 需要同时看 Code pass/fail、Memory 行为 utility、Tool tool-call utility 和冲突次数。",
        },
        {
            "决策点": "论文 insight 怎么收束",
            "建议": "写成 task vector utility 是 residual-key/span 条件化的。",
            "依据": "Code source/span 冲突和 Memory 保护反例共同支持这个机制叙事。",
        },
    ]
    st.markdown("**下一步决策建议**")
    st.dataframe(pd.DataFrame(decision_rows), use_container_width=True, hide_index=True)


def render_paper_narrative(
    data: MechanismData,
    code_metrics: pd.DataFrame,
    source_conflicts: pd.DataFrame,
    gate_memory: pd.DataFrame,
    memory_delta_rows: pd.DataFrame,
    preserve_rows: pd.DataFrame,
) -> None:
    st.markdown("### Paper Storyline / 论文视角逐步分析")
    st.markdown(
        """
        这部分按论文写法组织：先定义问题，再说明为什么旧假设不够，再给机制测量、关键证据、反例和方法收束。
        读这一段时不要把它当实验流水账，而要看它能否支撑一个清晰 claim。
        """
    )

    v1_lb = latest_metric_row(code_metrics, "v1_code_only", "LiveBench")
    v2_lb = latest_metric_row(code_metrics, "v2_spanaware", "LiveBench")
    v1_lcb = latest_metric_row(code_metrics, "v1_code_only", "LiveCodeBench")
    v2_lcb = latest_metric_row(code_metrics, "v2_spanaware", "LiveCodeBench")
    v3_lcb = latest_metric_row(code_metrics, "v3_memory_hard_floor", "LiveCodeBench")
    top_conflict = top_conflict_row(source_conflicts)
    top_delta = top_abs_delta_row(memory_delta_rows)
    top_preserve = top_preserve_row(preserve_rows)
    top_residual = top_expression_residual_row(data)

    storyline_rows = [
        {
            "论文段落": "1. Problem",
            "要回答的问题": "为什么 RCRF 合并后 Code 仍然掉，尤其是 hidden-test 风格题？",
            "当前证据": "构造了 32 条 Code hurt 子集：LiveBench 16 + LiveCodeBench 16，只看 RCRF 失败但 TA 系列可修的 case。",
            "可以写成论文的话": "We focus on regressions where the merged model loses recoverable code capability.",
        },
        {
            "论文段落": "2. Object",
            "要回答的问题": "task vector 的作用发生在哪里？",
            "当前证据": residual_example_sentence(top_residual),
            "可以写成论文的话": "Task-vector effects are measured as induced residual responses at expert-module-token granularity.",
        },
        {
            "论文段落": "3. Measurement",
            "要回答的问题": "怎么判断一个 residual 是有用、无用还是有害？",
            "当前证据": "使用 signed_effect 看方向，expression 看激活强度；Code 额外用 pass-fail contrast，避免把失败代码的低 NLL 误当能力。",
            "可以写成论文的话": "Positive imitation alone is insufficient; outcome contrast is needed to separate successful and failing trajectories.",
        },
        {
            "论文段落": "4. Main finding",
            "要回答的问题": "Code 是否是一条统一能力方向？",
            "当前证据": conflict_example_sentence(top_conflict),
            "可以写成论文的话": "Code utility is source- and span-conditioned rather than a smooth global direction.",
        },
        {
            "论文段落": "5. Positive method evidence",
            "要回答的问题": "按 source/span 做保守路由是否真的有用？",
            "当前证据": code_repair_sentence(v1_lb, v2_lb, v1_lcb, v2_lcb),
            "可以写成论文的话": "A training-free span-aware residual routing rule recovers part of the lost code capability.",
        },
        {
            "论文段落": "6. Negative control",
            "要回答的问题": "直接保护 Memory expert 能否解决 side-effect？",
            "当前证据": memory_control_sentence(v2_lcb, v3_lcb, gate_memory),
            "可以写成论文的话": "Expert-level preservation is too coarse; useful and harmful directions coexist within the same expert vector.",
        },
        {
            "论文段落": "7. Mechanism implication",
            "要回答的问题": "下一版方法应当怎么收束？",
            "当前证据": residual_key_sentence(top_delta, top_preserve),
            "可以写成论文的话": "Routing decisions should be made at residual-key and behavior-span granularity with conservative handling of conflicts.",
        },
    ]
    st.dataframe(pd.DataFrame(storyline_rows), use_container_width=True, hide_index=True)

    with st.expander("论文图应该怎么对应这些结论", expanded=True):
        figure_rows = [
            {
                "论文图": "Figure 1: Code hurt repair",
                "图在网页哪里": "下方 Code Hurt Repair bar chart",
                "支撑结论": "v2_spanaware 是当前正结果，不是只靠 code-only final span。",
            },
            {
                "论文图": "Figure 2: Source/span conflict heatmap",
                "图在网页哪里": "Source / Span 方向冲突",
                "支撑结论": "Code residual utility 在 LB/LCB prompt/code/reasoning 之间强烈翻转。",
            },
            {
                "论文图": "Figure 3: Memory protection negative control",
                "图在网页哪里": "Memory Gate Delta / Protection Counts",
                "支撑结论": "保护整个 Memory expert 会破坏 LCB 修复，说明粒度必须更细。",
            },
            {
                "论文图": "Figure 4: Residual-key case study",
                "图在网页哪里": "具体 residual key 证据",
                "支撑结论": "同一 expert 内部存在应增强、应压低、应保护的不同 key。",
            },
        ]
        st.dataframe(pd.DataFrame(figure_rows), use_container_width=True, hide_index=True)


def render_latest_terms_with_examples(
    data: MechanismData,
    code_metrics: pd.DataFrame,
    source_conflicts: pd.DataFrame,
    gate_memory: pd.DataFrame,
    memory_delta_rows: pd.DataFrame,
    preserve_rows: pd.DataFrame,
) -> None:
    st.markdown("### Glossary With Examples / 术语逐项解释")
    examples = []
    v2_lcb = latest_metric_row(code_metrics, "v2_spanaware", "LiveCodeBench")
    v1_lcb = latest_metric_row(code_metrics, "v1_code_only", "LiveCodeBench")
    v2_lb = latest_metric_row(code_metrics, "v2_spanaware", "LiveBench")
    v2_gate = gate_memory[gate_memory["variant"] == "v2_spanaware"].iloc[0] if not gate_memory.empty and (gate_memory["variant"] == "v2_spanaware").any() else None
    top_conflict = top_conflict_row(source_conflicts)
    top_delta = top_abs_delta_row(memory_delta_rows)
    top_preserve = top_preserve_row(preserve_rows)
    top_residual = top_expression_residual_row(data)

    examples.extend(
        [
            {
                "术语": "RCRF",
                "白话解释": "当前这条主线的方法：不训练新模型，而是根据每个 expert-module 的 utility/harm/conflict 调整 task vector 注入。",
                "真实例子": "v1-v4 都从同一个 RCRF gate checkpoint 出发，只改 residual routing 规则后评估 code hurt 子集。",
            },
            {
                "术语": "expert",
                "白话解释": "一个能力来源，对应一个任务向量，例如 tool、memory、code。",
                "真实例子": "网页里的 gate_memory_summary 统计的是 memory expert 被改写了多少 residual key。",
            },
            {
                "术语": "task vector / Delta W",
                "白话解释": "expert 权重相对 base 的差值，不直接等于能力；它只有乘上当前 hidden state 后才知道在这个样本上实际做了什么。",
                "真实例子": "RCRF 关心的不是单个 code 系数，而是每个 `Delta W_{expert,module} h` 在当前 prompt/span 上的效果。",
            },
            {
                "术语": "module",
                "白话解释": "模型里可定位的线性模块，包括 attention q/k/v/o 和 MLP gate/up/down。",
                "真实例子": delta_module_example(top_delta),
            },
            {
                "术语": "residual key",
                "白话解释": "一个具体 expert + layer + module 的决策位置，是 RCRF 调 gate 的最小分析单元。",
                "真实例子": residual_key_example(top_delta),
            },
            {
                "术语": "span",
                "白话解释": "只看输出或输入中的某一段行为，而不是把整条轨迹混在一起。",
                "真实例子": "Code 里区分 LB_prompt、LB_reasoning、LCB_code、LCB_prompt；这些 span 的方向会冲突。",
            },
        ]
    )
    if v2_lcb is not None:
        examples.append(
            {
                "术语": "Code hurt case",
                "白话解释": "当前 RCRF 在这些题上掉了，但 TA 系列说明它不是完全无解。",
                "真实例子": f"LiveCodeBench hurt 有 {int(v2_lcb['num_cases'])} 条，v2 至少一次通过 {int(v2_lcb['pass_any_count'])} 条。",
            }
        )
        examples.append(
            {
                "术语": "pass_any",
                "白话解释": "一题采多个候选，只要有一个候选通过，就算这题被救回来。",
                "真实例子": f"v2 LCB pass_any = {float(v2_lcb['pass_any']):.4f}，也就是 {pass_any_text(v2_lcb)}。",
            }
        )
        examples.append(
            {
                "术语": "candidate_pass_rate",
                "白话解释": "所有候选里有多少比例通过，比 pass_any 更能看稳定性。",
                "真实例子": f"v2 LCB candidate_pass_rate = {float(v2_lcb['candidate_pass_rate']):.4f}，即 {pass_count_text(v2_lcb)} 候选通过。",
            }
        )
        examples.append(
            {
                "术语": "test_point_rate",
                "白话解释": "把 hidden tests 或 test points 拆细看通过比例，比单题 pass/fail 更细。",
                "真实例子": f"v2 LCB test_point_rate = {float(v2_lcb['test_point_rate']):.4f}。",
            }
        )
    if v2_lb is not None:
        examples.append(
            {
                "术语": "LiveBench / LiveCodeBench hurt",
                "白话解释": "两个不同 Code 子分布；这轮不是跑全量榜单，而是专门看 RCRF 容易掉的困难切片。",
                "真实例子": f"LiveBench hurt v2 是 {pass_any_text(v2_lb)}；LiveCodeBench hurt v2 是 {pass_any_text(v2_lcb) if v2_lcb is not None else 'N/A'}。",
            }
        )
    if v1_lcb is not None and v2_lcb is not None:
        examples.append(
            {
                "术语": "span-aware",
                "白话解释": "不把整段输出混在一起，而是分开看 prompt 理解、reasoning、final code 哪段在贡献。",
                "真实例子": f"只看 final code 的 v1 是 {pass_count_text(v1_lcb)}；加入 span-aware 的 v2 到 {pass_count_text(v2_lcb)}。",
            }
        )
        examples.append(
            {
                "术语": "outcome-aware / pass-fail contrast",
                "白话解释": "同一类题里比较成功轨迹和失败轨迹，不再只模仿看起来像 expert 的输出。",
                "真实例子": "v1/v2 的 gate 都来自 pass/fail contrast；v2 加入更多 span 后 LCB 候选通过数从 "
                f"{pass_count_text(v1_lcb)} 到 {pass_count_text(v2_lcb)}。",
            }
        )
    if top_residual is not None:
        examples.extend(
            [
                {
                    "术语": "u = Delta W h",
                    "白话解释": "task vector 在当前 token hidden state 上诱导出来的实际 residual 响应。",
                    "真实例子": residual_example_sentence(top_residual),
                },
                {
                    "术语": "signed_effect",
                    "白话解释": "方向指标。正值表示这个 residual 在当前 span 上一阶降低 loss，负值表示可能有害。",
                    "真实例子": f"{residual_short_name(top_residual)} 的 signed_effect = {float(top_residual['signed_effect']):.4g}。",
                },
                {
                    "术语": "expression",
                    "白话解释": "强度指标。只表示 residual 被激活多强，不表示方向一定正确。",
                    "真实例子": f"{residual_short_name(top_residual)} 的 expression = {float(top_residual['expression']):.4g}；它需要和 signed_effect 一起看。",
                },
                {
                    "术语": "positive_fraction",
                    "白话解释": "同类样本里 signed_effect 为正的比例，用来判断方向是否稳定。",
                    "真实例子": f"{residual_short_name(top_residual)} 的 positive_fraction = {float(top_residual['positive_fraction']):.3f}。",
                },
            ]
        )
    if top_conflict is not None:
        examples.append(
            {
                "术语": "residual key conflict",
                "白话解释": "同一个模块位置，在一个数据源上应该增强，在另一个数据源上反而应该压低。",
                "真实例子": f"{top_conflict['left']} vs {top_conflict['right']}：{int(top_conflict['conflict_count'])}/{int(top_conflict['overlap_count'])} 个 key 方向相反。",
            }
        )
        examples.append(
            {
                "术语": "Pearson = -0.995",
                "白话解释": "两个 source/span 的方向几乎整体相反，不是轻微噪声。",
                "真实例子": f"{top_conflict['left']} 和 {top_conflict['right']} 的 Pearson = {float(top_conflict['pearson']):.4f}。",
            }
        )
        examples.append(
            {
                "术语": "conflict_rate",
                "白话解释": "方向相反的 key 占 overlap key 的比例。",
                "真实例子": f"{top_conflict['left']} vs {top_conflict['right']} 的 conflict_rate = {float(top_conflict['conflict_rate']):.2%}。",
            }
        )
    if v2_gate is not None:
        examples.append(
            {
                "术语": "memory_negative",
                "白话解释": "在当前 Code contrast 下，被判定应该压低的 Memory residual key 数量。",
                "真实例子": f"v2 有 {int(v2_gate['memory_negative'])} 个 memory_negative key；v3 全保护后 LCB 反而回退。",
            }
        )
        examples.append(
            {
                "术语": "gate alpha",
                "白话解释": "每个 expert-module 的注入强度旋钮；不是训练模型，而是决定某个 task vector residual 加多少。",
                "真实例子": f"v2 对 memory expert 改写 {int(v2_gate['memory_changed'])} 个 key，其中 {int(v2_gate['memory_positive'])} 个增强、{int(v2_gate['memory_negative'])} 个压低。",
            }
        )
    if top_preserve is not None:
        examples.append(
            {
                "术语": "utility floor / preserve",
                "白话解释": "如果某个 Memory key 在 Memory 任务上很有用，就尝试不让 Code overlay 把它压坏。",
                "真实例子": preserve_example(top_preserve),
            }
        )
    examples.extend(
        [
            {
                "术语": "hard floor",
                "白话解释": "粗粒度保护策略：例如 v3 直接保护 memory expert 的负向 overlay。",
                "真实例子": "v3 把 memory_negative 变成 0，但 LiveCodeBench pass_any 从 v2 的 13/16 回落到 10/16。",
            },
            {
                "术语": "conservative routing",
                "白话解释": "遇到 source/span 或 task 间冲突时不强推，尽量回到 base 或只小幅改动。",
                "真实例子": "v2 用 conservative aggregation + conflict penalty，是当前最干净的正结果。",
            },
            {
                "术语": "residual evidence table",
                "白话解释": "下一步要构造的统一证据表：每个 residual key 同时记录 Code、Memory、Tool 上的 utility 和 conflict。",
                "真实例子": "网页已经展示了 Code source/span conflict、Memory delta、Memory preserve rows；下一步要把 Tool tool-call utility 补进同一表。",
            },
        ]
    )
    st.dataframe(pd.DataFrame(examples), use_container_width=True, hide_index=True)


def top_conflict_row(source_conflicts: pd.DataFrame) -> pd.Series | None:
    if source_conflicts.empty or "conflict_rate" not in source_conflicts:
        return None
    return source_conflicts.sort_values("conflict_rate", ascending=False).iloc[0]


def top_abs_delta_row(memory_delta_rows: pd.DataFrame) -> pd.Series | None:
    if memory_delta_rows.empty or "delta" not in memory_delta_rows:
        return None
    return memory_delta_rows.assign(abs_delta=memory_delta_rows["delta"].abs()).sort_values("abs_delta", ascending=False).iloc[0]


def top_preserve_row(preserve_rows: pd.DataFrame) -> pd.Series | None:
    if preserve_rows.empty or "preserve_utility" not in preserve_rows:
        return None
    return preserve_rows.sort_values("preserve_utility", ascending=False).iloc[0]


def top_expression_residual_row(data: MechanismData) -> pd.Series | None:
    residual = data.residual.copy()
    if residual.empty or "expression" not in residual:
        return None
    return residual.assign(abs_expression=residual["expression"].abs()).sort_values("abs_expression", ascending=False).iloc[0]


def residual_short_name(row: pd.Series) -> str:
    return f"{row.get('expert', 'expert')} L{int(row.get('layer', 0))}.{row.get('module', 'module')} on {row.get('task', 'task')}/{row.get('span_type', 'span')}"


def residual_example_sentence(row: pd.Series | None) -> str:
    if row is None:
        return "当前 residual 明细为空；需要加载 signed_utility_rows 或 signed_utility_summary。"
    return (
        f"当前机制数据中，{residual_short_name(row)} 的 expression={float(row['expression']):.4g}，"
        f"signed_effect={float(row['signed_effect']):.4g}，positive_fraction={float(row['positive_fraction']):.3f}。"
    )


def conflict_example_sentence(row: pd.Series | None) -> str:
    if row is None:
        return "当前 source_conflict_pairs.csv 为空。"
    return (
        f"{row['left']} vs {row['right']} 的 Pearson={float(row['pearson']):.4f}，"
        f"{int(row['conflict_count'])}/{int(row['overlap_count'])} 个 residual key 符号相反。"
    )


def code_repair_sentence(v1_lb: pd.Series | None, v2_lb: pd.Series | None, v1_lcb: pd.Series | None, v2_lcb: pd.Series | None) -> str:
    parts = []
    if v1_lb is not None and v2_lb is not None:
        parts.append(f"LB pass_any 从 {pass_any_text(v1_lb)} 到 {pass_any_text(v2_lb)}")
    if v1_lcb is not None and v2_lcb is not None:
        parts.append(f"LCB 候选通过从 {pass_count_text(v1_lcb)} 到 {pass_count_text(v2_lcb)}")
    return "；".join(parts) + "。" if parts else "当前 code_hurt_metrics.csv 为空。"


def memory_control_sentence(v2_lcb: pd.Series | None, v3_lcb: pd.Series | None, gate_memory: pd.DataFrame) -> str:
    parts = []
    if v2_lcb is not None and v3_lcb is not None:
        parts.append(f"v3 hard floor 后 LCB pass_any 从 v2 的 {pass_any_text(v2_lcb)} 回落到 {pass_any_text(v3_lcb)}")
    if not gate_memory.empty and "memory_negative" in gate_memory:
        v3 = gate_memory[gate_memory["variant"] == "v3_memory_hard_floor"]
        if not v3.empty:
            parts.append(f"v3 memory_negative={int(v3['memory_negative'].iloc[0])}，说明它确实把 Memory 负向 overlay 全保护了")
    return "；".join(parts) + "。" if parts else "当前 Memory gate summary 为空。"


def residual_key_sentence(top_delta: pd.Series | None, top_preserve: pd.Series | None) -> str:
    parts = []
    if top_delta is not None:
        parts.append(
            f"最大 gate delta 例子：{top_delta['variant']} 的 {top_delta['param_name']} delta={float(top_delta['delta']):.4f}，reason={top_delta['reason']}"
        )
    if top_preserve is not None:
        parts.append(
            f"Memory preserve 例子：{top_preserve['param_name']} preserve_utility={float(top_preserve['preserve_utility']):.4g}"
        )
    return "；".join(parts) + "。" if parts else "当前 residual key 明细为空。"


def delta_module_example(row: pd.Series | None) -> str:
    if row is None:
        return "例子会显示为 `layer.module`，例如 L24.attn_q；当前 delta 明细为空。"
    return f"{row['param_name']} 对应 layer {int(row['layer'])} 的 {row['family']} 模块。"


def residual_key_example(row: pd.Series | None) -> str:
    if row is None:
        return "当前没有 memory_delta_rows.csv，无法展示真实 key。"
    return f"`{row['variant']} | {row['param_name']}` 是一个具体 residual key，delta={float(row['delta']):.4f}。"


def preserve_example(row: pd.Series | None) -> str:
    if row is None:
        return "当前没有 preserve_rows.csv。"
    return (
        f"{row['param_name']} 在 {row['preserve_task']} 上 preserve_utility={float(row['preserve_utility']):.4g}，"
        f"positive_fraction={float(row['positive_fraction']):.3f}。"
    )


def latest_metric_row(code_metrics: pd.DataFrame, variant: str, dataset: str) -> pd.Series | None:
    if code_metrics.empty or not {"variant", "dataset"}.issubset(code_metrics.columns):
        return None
    rows = code_metrics[(code_metrics["variant"] == variant) & (code_metrics["dataset"] == dataset)]
    if rows.empty:
        return None
    return rows.iloc[0]


def total_candidates(row: pd.Series) -> int:
    rate = float(row.get("candidate_pass_rate", 0.0))
    count = int(row.get("pass_count", 0))
    if rate <= 0:
        return 0
    return int(round(count / rate))


def pass_any_text(row: pd.Series) -> str:
    return f"{int(row['pass_any_count'])}/{int(row['num_cases'])}"


def pass_count_text(row: pd.Series) -> str:
    total = total_candidates(row)
    if total <= 0:
        return str(int(row.get("pass_count", 0)))
    return f"{int(row['pass_count'])}/{total}"


def pass_any_delta_text(old: pd.Series, new: pd.Series) -> str:
    delta = int(new["pass_any_count"]) - int(old["pass_any_count"])
    return f"{delta:+d} cases vs v1"


def pass_count_delta_text(old: pd.Series, new: pd.Series) -> str:
    delta = int(new["pass_count"]) - int(old["pass_count"])
    return f"{delta:+d} passed candidates vs v1"


def source_conflict_matrix(source_conflicts: pd.DataFrame, metric: str) -> pd.DataFrame:
    sources = sorted(set(source_conflicts["left"].dropna()) | set(source_conflicts["right"].dropna()))
    matrix = pd.DataFrame(0.0, index=sources, columns=sources)
    diag = 1.0 if metric == "pearson" else 0.0
    for source in sources:
        matrix.loc[source, source] = diag
    for _, row in source_conflicts.iterrows():
        left = row["left"]
        right = row["right"]
        value = float(row[metric])
        matrix.loc[left, right] = value
        matrix.loc[right, left] = value
    return matrix


def page_module_map(data: MechanismData) -> None:
    st.subheader("Module Mechanism Map / 模块机制地图")
    st.markdown(
        """
        这一页先回答：**哪一层、哪一类模块真的在表达某个 RL task vector？**

        读图方式：
        - `signed_effect` 高且为正：这个 expert-module 的 residual 在当前 task/span 上方向正确。
        - `expression` 高但 `signed_effect` 低或为负：残差被强激活，但可能是噪声或反方向，是最值得检查的异常。
        - `q/k` 异常：通常对应 attention routing 变化，可能影响全局证据选择。
        - `gate/up/down` 异常：通常对应能力写入或行为模式表达，尤其 Memory/Code 更要看 MLP。
        """
    )

    metric = st.selectbox(
        "指标",
        ["alpha", "signed_effect", "expression", "cross_harm", "conflict_score", "positive_fraction"],
        format_func=lambda item: METRIC_LABELS[item],
    )
    render_metric_note(metric)
    table = metric_source_table(data, metric)
    if table.empty:
        st.info("当前没有这个指标的记录。")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        experts = multiselect_or_all("Expert / 专家", sorted(table["expert"].dropna().unique()) if "expert" in table else [])
    with c2:
        tasks = multiselect_or_all("Task / 任务", sorted(table["task"].dropna().unique()) if "task" in table else [])
    with c3:
        candidates = multiselect_or_all("Candidate / 候选 gate", sorted(table["candidate"].dropna().unique()) if "candidate" in table else [])
    with c4:
        modules = multiselect_or_all("Module / 模块", [m for m in MODULE_ORDER if m in set(table["module"])])

    filtered = filter_table(table, experts=experts, tasks=tasks, candidates=candidates, modules=modules)
    grid = (
        filtered.groupby(["layer", "module"], as_index=False)[metric]
        .mean()
        .sort_values(["layer", "module"])
    )
    if grid.empty:
        st.info("当前筛选条件下没有数据。")
        return

    fig = heatmap_layer_module(
        grid,
        value=metric,
        title=f"Layer x Module 机制地图: {METRIC_LABELS[metric]}",
        colorbar=METRIC_LABELS[metric],
    )
    st.plotly_chart(fig, use_container_width=True)
    export_controls(fig, grid, f"module_map_{metric}")
    render_chart_guide(
        "这张热图怎么看",
        [
            "纵轴是 layer，横轴是模块 q/k/v/o/gate/up/down；每个格子是当前筛选条件下该 layer-module 的平均指标。",
            "颜色越亮不一定越好，要先看你选的指标：signed_effect 看方向，expression 看激活强度，alpha 看最终 gate 决策。",
            "q/k 是 attention routing，异常通常意味着模型在“看哪里”上变了；v/o 更像内容写回；gate/up/down 是 MLP 能力表达主通道。",
        ],
        [
            "expression 高但 signed_effect 低或为负：残差被强烈激活，但方向可能错。",
            "q/k 上 conflict_score 或 cross_harm 高：可能是 Tool/Memory/Code 抢 attention routing。",
            "MLP alpha 被压低，但 signed_effect/positive_fraction 很好：可能误伤 owner utility。",
        ],
        "先用这页找热点 layer-module，再切到 Prompt Sensitivity 看它是不是由少数 prompt/span 驱动。"
    )

    st.markdown("**明细表：用于定位具体 expert/task/candidate 的来源行**")
    st.dataframe(
        filtered.sort_values(metric, ascending=False).head(300),
        use_container_width=True,
        hide_index=True,
    )


def page_prompt_sensitivity(data: MechanismData) -> None:
    st.subheader("Prompt Sensitivity / Prompt 与 Span 敏感性")
    residual = data.residual.copy()
    if residual.empty:
        st.info("当前没有 residual 记录。")
        return
    non_aggregate = residual[residual["prompt_id"] != "aggregate"].copy()
    if non_aggregate.empty:
        st.warning("当前只有 aggregate signed utility。要看逐 prompt 敏感性，需要 signed_utility_rows.jsonl。")
        non_aggregate = residual.copy()

    st.markdown(
        """
        这一页回答：**某个模块是真的稳定有用，还是只被少数 prompt/span 偶然点亮？**

        RCRF 的风险在于：如果 calibration 里某些 prompt/span 偏了，`signed_effect` 可能只学习到局部行为，而不是可泛化能力。
        因此这里重点看两类模块：

        - 高敏感模块：`expression` 很高，说明 `DeltaW h` 确实进入了计算。
        - 不稳定模块：不同 prompt 下 `signed_effect` 正负摇摆，说明它可能依赖 span 或样本构造。

        对 Code 尤其要看这里：如果 code-block span 下信号弱或不稳定，说明需要 prompt+code 或 pass/fail contrast，而不是继续调同一个 gate 公式。
        """
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        tasks = multiselect_or_all("Task / 任务", sorted(non_aggregate["task"].dropna().unique()))
    with c2:
        spans = multiselect_or_all("Span / 响应片段", sorted(non_aggregate["span_type"].dropna().unique()))
    with c3:
        experts = multiselect_or_all("Expert / 专家", sorted(non_aggregate["expert"].dropna().unique()))
    with c4:
        metric = st.selectbox("Prompt 热图指标", ["signed_effect", "expression", "positive_fraction"], format_func=lambda item: METRIC_LABELS[item])
    render_metric_note(metric)

    filtered = filter_table(non_aggregate, experts=experts, tasks=tasks)
    if spans:
        filtered = filtered[filtered["span_type"].isin(spans)]
    if filtered.empty:
        st.info("当前筛选条件下没有 prompt 记录。")
        return

    filtered["expert_module"] = (
        filtered["expert"].astype(str)
        + ":L"
        + filtered["layer"].astype(str)
        + "."
        + filtered["module"].astype(str)
    )
    top_modules = (
        filtered.groupby("expert_module")["expression"].mean().sort_values(ascending=False).head(36).index.tolist()
    )
    heat = filtered[filtered["expert_module"].isin(top_modules)]
    heat = heat.groupby(["expert_module", "prompt_id"], as_index=False)[metric].mean()
    fig = px.imshow(
        heat.pivot(index="expert_module", columns="prompt_id", values=metric).fillna(0.0),
        aspect="auto",
        color_continuous_scale="RdBu_r" if metric == "signed_effect" else "Viridis",
        title=f"Prompt 敏感性: {METRIC_LABELS.get(metric, metric)}",
        labels={"x": "prompt_id / prompt", "y": "expert-module / 专家模块", "color": METRIC_LABELS.get(metric, metric)},
    )
    fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig, use_container_width=True)
    export_controls(fig, heat, f"prompt_sensitivity_{metric}")
    render_chart_guide(
        "这张 prompt 热图怎么看",
        [
            "横轴是 prompt_id，纵轴是 expert-layer-module；颜色是当前选择的指标。",
            "一整行颜色稳定，说明这个模块对不同 prompt 的响应比较一致。",
            "只有少数列很亮，说明该模块可能被个别 prompt 或 span 偶然触发。",
        ],
        [
            "signed_effect 一会正一会负：这个模块方向不稳定，不能直接当稳定能力通道。",
            "code 只在 code-block span 有弱信号：说明当前 code calibration 可能没有覆盖 hidden-test 关键行为。",
            "某个 prompt 让大量模块同时异常：可能是 prompt/span 构造偏置，而不是模型机制。",
        ],
        "点开下方高敏感/不稳定表，找具体 layer-module，再回 Module Map 或 Interference 看它是否也有冲突。"
    )

    stats = (
        filtered.groupby(["expert", "task", "span_type", "layer", "module"], as_index=False)
        .agg(
            prompt_count=("prompt_id", "nunique"),
            mean_expression=("expression", "mean"),
            mean_signed_effect=("signed_effect", "mean"),
            signed_effect_std=("signed_effect", "std"),
            positive_fraction=("positive_fraction", "mean"),
        )
        .fillna({"signed_effect_std": 0.0})
    )
    stats["instability"] = stats["signed_effect_std"] * (1.0 - (stats["positive_fraction"] - 0.5).abs() * 2.0).clip(lower=0.0)

    left, right = st.columns(2)
    with left:
        st.markdown("**高敏感模块：优先看表达能量和正收益是否同时高**")
        st.caption("这不是图，而是候选模块排名。mean_expression 高说明残差确实进入计算；mean_signed_effect 高说明方向也比较对。")
        st.dataframe(
            stats.sort_values(["mean_expression", "mean_signed_effect"], ascending=[False, False]).head(30),
            use_container_width=True,
            hide_index=True,
        )
    with right:
        st.markdown("**不稳定模块：优先查 span 选择或 prompt 偏置**")
        st.caption("instability 高表示 signed_effect 在 prompt 间波动大且正负比例不稳定，适合拿去做 span/pass-fail 对照。")
        st.dataframe(
            stats.sort_values("instability", ascending=False).head(30),
            use_container_width=True,
            hide_index=True,
        )


def page_interference(data: MechanismData) -> None:
    st.subheader("Interference Explorer / 跨专家干扰")
    interference = data.interference.copy()
    if interference.empty:
        st.info("当前没有 interference 记录。")
        return
    interference["pair"] = interference["expert_a"] + " | " + interference["expert_b"]

    st.markdown(
        """
        这一页回答：**tool / memory / code 的 residual 是否在同一模块上互相拉扯？**

        这里的 cosine 不是参数余弦，而是同一个输入上 induced residual `DeltaW h` 的方向关系。
        - `cosine < 0`：两个 expert 在这个位置把 hidden state 推向相反方向。
        - `q/k` 冲突：更像 attention routing 冲突，可能改变模型看哪些证据。
        - `cross_harm` 高：说明这个模块不仅方向冲突，而且对别的任务 trajectory 有负收益。
        - `shared_positive_effect` 高：说明可能是共享能力通道，不应轻易压低。
        """
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        tasks = multiselect_or_all("Task / 任务", sorted(interference["task"].dropna().unique()))
    with c2:
        pairs = multiselect_or_all("Expert pair / 专家对", sorted(interference["pair"].dropna().unique()))
    with c3:
        families = multiselect_or_all("Group / 模块组", sorted(interference["module_family"].dropna().unique()))
    with c4:
        metric = st.selectbox(
            "干扰指标",
            ["cosine", "conflict_score", "cross_harm", "shared_positive_effect"],
            format_func=lambda item: METRIC_LABELS[item],
        )
    render_metric_note(metric)

    filtered = interference.copy()
    if tasks:
        filtered = filtered[filtered["task"].isin(tasks)]
    if pairs:
        filtered = filtered[filtered["pair"].isin(pairs)]
    if families:
        filtered = filtered[filtered["module_family"].isin(families)]
    if filtered.empty:
        st.info("当前筛选条件下没有干扰记录。")
        return

    grid = filtered.groupby(["layer", "module"], as_index=False)[metric].mean()
    fig = heatmap_layer_module(
        grid,
        value=metric,
        title=f"两两 residual 干扰: {METRIC_LABELS[metric]}",
        colorbar=METRIC_LABELS[metric],
    )
    st.plotly_chart(fig, use_container_width=True)
    export_controls(fig, grid, f"interference_{metric}")
    render_chart_guide(
        "这张干扰热图怎么看",
        [
            "纵轴是 layer，横轴是模块；颜色表示当前 expert pair/task/group 下的平均干扰指标。",
            "cosine 看方向是否相反；conflict_score 是把负 cosine 和负相关比例压成一个冲突强度；cross_harm 看是否真的伤到别的 task。",
            "q/k 上的冲突更像路由冲突，MLP 上的冲突更像能力写入冲突。",
        ],
        [
            "late-layer q/k conflict 高：可能是 Tool 格式、Memory evidence routing 或 Code 约束理解互相抢通道。",
            "cosine 不太负但 cross_harm 高：方向不一定完全相反，但对 loss 的一阶影响仍有害。",
            "shared_positive_effect 高却 alpha 被压：可能是共享能力被过度保守处理。",
        ],
        "用右侧散点图确认冲突是否同时带来 cross_harm；只负 cosine 但没有 harm 的位置不一定要压。"
    )

    left, right = st.columns(2)
    pair_summary = (
        filtered.groupby(["task", "pair", "module_family"], as_index=False)
        .agg(
            cosine=("cosine", "mean"),
            conflict_score=("conflict_score", "mean"),
            cross_harm=("cross_harm", "mean"),
            shared_positive_effect=("shared_positive_effect", "mean"),
        )
        .sort_values(["conflict_score", "cross_harm"], ascending=False)
    )
    with left:
        st.markdown("**专家对汇总表**")
        st.caption("按 task、expert pair、模块组聚合。先找 conflict_score/cross_harm 最高的组合，再回热图定位 layer-module。")
        st.dataframe(pair_summary.head(80), use_container_width=True, hide_index=True)
    with right:
        scatter = px.scatter(
            filtered,
            x="cosine",
            y="cross_harm",
            color="pair",
            symbol="module_family",
            hover_data=["task", "layer", "module", "conflict_score", "shared_positive_effect"],
            title="Residual 方向冲突 vs 跨任务 harm",
            labels={
                "cosine": "residual cosine / induced residual 余弦",
                "cross_harm": "cross_harm / 跨任务负收益",
                "pair": "expert pair / 专家对",
                "module_family": "module group / 模块组",
            },
        )
        st.plotly_chart(scatter, use_container_width=True)
        export_controls(scatter, filtered, "interference_scatter")
        render_chart_guide(
            "这张散点图怎么看",
            [
                "每个点是一个 layer-module 上的 expert pair 干扰记录。",
                "横轴越靠左，两个 expert 的 induced residual 越相反；纵轴越高，对其他任务的负收益越大。",
                "右上/左上点比单纯左侧点更危险，因为它既有方向冲突又有实际 harm。",
            ],
            [
                "左上角：优先做 suppress/restore ablation。",
                "靠左但很低：可能只是几何方向相反，不一定影响行为。",
                "靠右但很高：可能是 gradient/loss 层面的 harm，不是简单 cosine 能解释。",
            ],
            "优先挑左上角点，回 Gate Sparsity 看 RCRF 是否真的压低了对应模块。"
        )


def page_gate_sparsity(data: MechanismData) -> None:
    st.subheader("Gate Sparsity View / Gate 稀疏与抑制")
    gate = data.gate.copy()
    if gate.empty:
        st.info("当前没有 gate 记录。")
        return
    gate["action"] = gate["alpha"].map(gate_action)

    st.markdown(
        """
        这一页回答：**RCRF 最终为什么放大、保留、压低或稀疏某些 expert-module？**

        四个候选的机制含义：
        - `owner_only`：只看本任务 response 上是否有用，是最简洁的机制基线。
        - `energy_only`：只看 `DeltaW h` 是否大，不看方向；如果它差，说明“能量大”不等于“能力好”。
        - `no_conflict`：保留 owner utility 和跨任务正贡献，但不压冲突；如果 Memory 掉，说明 conflict 项有必要。
        - `rcrf`：主版本，综合 owner utility、synergy、harm/conflict、noise。

        读图时重点看：MLP 是否被过度压低，q/k 是否被合理保守处理，Code 是否因为不稳定而整体被压。
        """
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        candidates = multiselect_or_all("Candidate / 候选方法", ordered_candidates(gate["candidate"].dropna().unique()))
    with c2:
        experts = multiselect_or_all("Expert / 专家", sorted(gate["expert"].dropna().unique()))
    with c3:
        families = multiselect_or_all("Group / 模块组", sorted(gate["module_family"].dropna().unique()))
    filtered = filter_table(gate, experts=experts, candidates=candidates)
    if families:
        filtered = filtered[filtered["module_family"].isin(families)]
    if filtered.empty:
        st.info("当前筛选条件下没有 gate 记录。")
        return

    stacked = (
        filtered.groupby(["candidate", "expert", "action"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    fig_bar = px.bar(
        stacked,
        x="candidate",
        y="count",
        color="action",
        facet_col="expert",
        title="不同候选方法的 Gate 动作分布",
        labels={"candidate": "candidate / 候选方法", "count": "module count / 模块数", "action": "gate action / 动作"},
        category_orders={"candidate": ordered_candidates(filtered["candidate"].unique())},
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    export_controls(fig_bar, stacked, "gate_sparsity_actions")
    render_chart_guide(
        "这张 Gate 动作柱状图怎么看",
        [
            "横轴是候选方法，分面是 expert，颜色表示 gate 动作：amplify、retain、suppress、sparse。",
            "它回答的是：不同 RCRF ablation 最终对各 expert 的模块处理是否更激进或更保守。",
            "owner_only 通常代表最简机制；energy_only 用来验证只看能量是否足够；no_conflict 用来验证冲突项是否必要；rcrf 是主版本。",
        ],
        [
            "某个 expert 大面积 sparse/suppress：可能能力被整体压低。",
            "energy_only 大量 amplify 但 eval 不好：说明能量大不等于方向好。",
            "no_conflict 比 rcrf 更放大但 Memory/Tool 掉分：说明冲突抑制可能是必要机制。",
        ],
        "看完分布后，下面的 alpha 热图能告诉你这些动作集中在哪些 layer-module。"
    )

    selected_candidate = st.selectbox("热图候选方法", ordered_candidates(filtered["candidate"].unique()))
    heat_rows = filtered[filtered["candidate"] == selected_candidate].groupby(["layer", "module"], as_index=False)["alpha"].mean()
    fig_heat = heatmap_layer_module(
        heat_rows,
        value="alpha",
        title=f"Gate Alpha 热图: {selected_candidate}",
        colorbar=METRIC_LABELS["alpha"],
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    export_controls(fig_heat, heat_rows, f"gate_alpha_{selected_candidate}")
    render_chart_guide(
        "这张 alpha 热图怎么看",
        [
            "纵轴是 layer，横轴是模块，颜色是该候选方法下的平均 alpha。",
            "alpha > 1 表示放大 task vector residual；alpha < 1 表示压低；接近 1 表示基本保留 init=1 的能力预算。",
            "RCRF 不是训练新参数，而是根据 signed utility、harm、conflict、noise 给每个 expert-module 设注入强度。",
        ],
        [
            "q/k alpha 系统性低：RCRF 认为 routing 风险较高。",
            "MLP alpha 系统性低：要小心是否误伤 Memory/Code 能力表达。",
            "Code alpha 低但 eval Code 没提升：说明当前 code signed utility/span 还没找到正确能力方向。",
        ],
        "如果某个位置 alpha 低但你怀疑它有用，回 Module Map 看 signed_effect/expression，再做局部 restore ablation。"
    )

    left, right = st.columns(2)
    with left:
        hist = px.histogram(
            filtered,
            x="alpha",
            color="candidate",
            facet_row="module_family",
            nbins=40,
            title="不同模块组的 Alpha 分布",
            labels={"alpha": "gate alpha / 注入系数", "candidate": "candidate / 候选方法", "count": "module count / 模块数"},
        )
        st.plotly_chart(hist, use_container_width=True)
        export_controls(hist, filtered, "gate_alpha_histogram")
        render_chart_guide(
            "这张 alpha 分布图怎么看",
            [
                "横轴是 alpha，纵向分面是模块组；它显示每个候选方法的 gate 系数分布。",
                "分布集中在 1 附近表示保守；左移表示整体压低；右移表示整体放大。",
                "这个图适合快速比较 ablation 的激进程度，而不是定位具体模块。",
            ],
            [
                "某个候选整体左移但 eval 没变好：可能过滤太粗。",
                "MLP 分布明显左移：要检查 Memory/Code 是否被压。",
                "attention q/k 分布左移：通常符合 RCRF 对 routing 的保守假设，但需要 eval 验证。",
            ],
            "用它判断候选方法整体风格，再用 alpha 热图定位具体层。"
        )
    with right:
        scatter = px.scatter(
            filtered,
            x="owner_signal",
            y="harm_signal",
            color="alpha",
            symbol="module_family",
            hover_data=["candidate", "expert", "layer", "module", "noise_score"],
            color_continuous_scale="RdBu_r",
            title="Owner utility 与 harm suppression 的关系",
            labels={
                "owner_signal": "owner_signal / 本任务正向信号",
                "harm_signal": "harm_signal / max(direct harm, conflict)",
                "alpha": "gate alpha / 注入系数",
                "module_family": "module group / 模块组",
            },
        )
        st.plotly_chart(scatter, use_container_width=True)
        export_controls(scatter, filtered, "gate_owner_vs_harm")
        render_chart_guide(
            "这张 owner-vs-harm 散点图怎么看",
            [
                "横轴是 owner_signal：本 expert 在自己任务上有用的程度。",
                "纵轴是 harm_signal：direct harm 和 conflict_score 中较强的抑制信号。",
                "颜色是 alpha：最终 gate 到底放大还是压低。",
            ],
            [
                "右上角：既有 owner utility 又有 harm/conflict，是最值得做 ablation 的取舍位置。",
                "右下角 alpha 低：可能过度抑制了稳定 owner utility。",
                "左上角 alpha 高：可能没有充分压制有害 residual。",
            ],
            "优先看右上角点：这类位置最容易形成论文里的 trade-off 机制图。"
        )


def page_research_questions(data: MechanismData) -> None:
    st.subheader("Research Questions / 下一步机制问题")
    st.markdown(
        """
        这一页把前面几页的异常模式自动转成研究问题。它不是最终结论，而是下一轮最值得做的机制验证列表。

        推荐用法：
        - P0 问题优先做 targeted ablation：只恢复/压低某个 layer-module，然后看 Tool/Memory/Code eval 是否按预期变化。
        - 对 `high_expression_negative_effect`：优先检查是否 span 选错、trajectory 不代表正式 eval。
        - 对 `qk_routing_conflict`：优先做 q/k-only restore 或 suppress，观察是否影响 tool-call 格式或 memory evidence routing。
        - 对 `mlp_over_suppressed`：优先验证 MLP owner utility 是否被 conflict/noise 项误伤。
        """
    )
    questions = summarize_research_questions(data.residual, data.interference, data.gate)
    if questions.empty:
        st.info("当前数据无法生成自动 research questions。")
    else:
        qtypes = multiselect_or_all("问题类型", sorted(questions["question_type"].unique()))
        view = questions[questions["question_type"].isin(qtypes)] if qtypes else questions
        st.caption("这张表不是评测结果，而是自动生成的机制假设清单。每一行都应该对应一个可复现 ablation 或一张论文图。")
        st.dataframe(view, use_container_width=True, hide_index=True)
        st.download_button(
            "Export questions CSV",
            view.to_csv(index=False).encode("utf-8"),
            file_name="rcrf_research_questions.csv",
            mime="text/csv",
        )

    st.markdown("**Eval 结果上下文：用来判断机制异常是否真的对应行为变化**")
    if data.eval.empty:
        st.info("当前没有 eval 记录。")
        return
    metric_options = sorted(data.eval["metric"].dropna().unique())
    metric = st.selectbox("Eval 指标", metric_options, index=metric_options.index("accuracy") if "accuracy" in metric_options else 0)
    eval_view = data.eval[data.eval["metric"] == metric]
    fig = px.bar(
        eval_view,
        x="subset",
        y="score",
        color="candidate",
        facet_col="task",
        barmode="group",
        title=f"RCRF Eval 上下文: {metric}",
        labels={"subset": "subset / 子集", "score": f"{metric} score / 分数", "candidate": "candidate / 候选方法"},
    )
    fig.update_yaxes(range=[0, max(1.0, float(eval_view["score"].max()) * 1.10 if not eval_view.empty else 1.0)])
    st.plotly_chart(fig, use_container_width=True)
    export_controls(fig, eval_view, f"eval_{metric}")
    render_chart_guide(
        "这张 eval 上下文图怎么看",
        [
            "横轴是 eval subset，颜色是 candidate，纵轴是分数。",
            "它不是机制证据本身，而是帮你判断前面发现的机制是否对应真实行为变化。",
            "例如 energy_only 如果表达能量很高但 eval 弱，说明只看 `||DeltaW h||` 不够。",
        ],
        [
            "Tool live 提升但 ToolRL 源分布略降：可能是泛化格式更稳，但源分布细节有损失。",
            "Memory no_conflict 掉分：说明跨任务冲突抑制可能保护了 Memory 长轨迹稳定性。",
            "Code LiveCodeBench 没提升：说明当前 code span/signed utility 没抓住 hidden-test 正确性。",
        ],
        "把 eval 差异反查到 Gate/Interference/Prompt 页面，形成机制闭环。"
    )


@st.cache_data(show_spinner=False)
def load_pairwise_zero_findings() -> dict[str, Any]:
    def read_csv(path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)

    report_path = PAIRWISE_ZERO_DIR / "pairwise_zero_diagnostic_report.md"
    report = ""
    if report_path.exists():
        report = report_path.read_text(encoding="utf-8", errors="ignore")
    return {
        "exists": PAIRWISE_ZERO_DIR.exists(),
        "report": report,
        "zero_summary": read_csv(PAIRWISE_ZERO_DIR / "zero_expert_summary.csv"),
        "conflict_summary": read_csv(PAIRWISE_ZERO_DIR / "pairwise_conflict_summary.csv"),
        "module_conflict": read_csv(PAIRWISE_ZERO_DIR / "pairwise_module_conflict_summary.csv"),
        "figures": [
            PAIRWISE_ZERO_DIR / "figures" / "pairwise_conflict_rate_heatmap.png",
            PAIRWISE_ZERO_DIR / "figures" / "zero_expert_role_risk_heatmap.png",
            PAIRWISE_ZERO_DIR / "figures" / "pairwise_expression_dominance.png",
        ],
    }


def page_pairwise_zero() -> None:
    latest = load_pairwise_zero_findings()
    st.subheader("Pairwise-Zero Diagnostics / 两两置零诊断")
    st.markdown(
        f"""
        这一页把三专家联合视角拆成三个二元问题：每次保留两个 expert，把第三个 expert 的 residual coefficient 置为 0。

        当前读取目录：

        `{PAIRWISE_ZERO_DIR}`

        **用途**：这里不是最终评测页面，而是论文机制诊断页面。它帮助判断某个 expert 被置零时，到底删掉了能力 residual、行为保护 residual，还是冲突/noise residual。
        """
    )
    if not latest["exists"]:
        st.warning("pairwise-zero 目录还不存在。先运行 `scripts/analysis/build_rcrf_pairwise_zero_diagnostics.py`。")
        return

    zero_summary = latest["zero_summary"]
    conflict_summary = latest["conflict_summary"]
    module_conflict = latest["module_conflict"]

    if not conflict_summary.empty:
        st.markdown("### 二元冲突：哪个 pair 在哪个 task 上方向相反")
        pivot = conflict_summary.pivot_table(
            index="pair_name",
            columns="task",
            values="opposite_sign_rate",
            aggfunc="mean",
            observed=False,
        )
        fig = px.imshow(
            pivot,
            aspect="auto",
            color_continuous_scale="Reds",
            title="Opposite-sign rate by pair and task",
            labels={"x": "task / 任务", "y": "pair / 二元视角", "color": "opposite rate"},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(conflict_summary, use_container_width=True, hide_index=True)

    if not zero_summary.empty:
        st.markdown("### 置零一个 expert 会删掉什么 role")
        role_cols = [col for col in zero_summary.columns if col.startswith("role_")]
        if role_cols:
            role_view = zero_summary[["pair_name", "zero_expert", *role_cols]].melt(
                id_vars=["pair_name", "zero_expert"],
                var_name="role",
                value_name="count",
            )
            role_view["role"] = role_view["role"].str.replace("role_", "", regex=False)
            fig = px.bar(
                role_view,
                x="pair_name",
                y="count",
                color="role",
                title="Removed residual roles after zeroing one expert",
                labels={"pair_name": "pairwise view / 二元视角", "count": "removed rows / 删除行数"},
            )
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(zero_summary, use_container_width=True, hide_index=True)

    if not module_conflict.empty:
        st.markdown("### 层段 / 模块族冲突定位")
        selected_pair = st.selectbox("pair", sorted(module_conflict["pair_name"].unique()))
        selected_task = st.selectbox("task", sorted(module_conflict["task"].unique()))
        view = module_conflict[
            (module_conflict["pair_name"] == selected_pair) & (module_conflict["task"] == selected_task)
        ].copy()
        if not view.empty:
            fig = px.bar(
                view,
                x="layer_band",
                y="opposite_sign_rate",
                color="module_family",
                barmode="group",
                title=f"{selected_pair} on {selected_task}: conflict by layer band and module family",
                labels={"opposite_sign_rate": "opposite-sign rate / 方向相反比例"},
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(view, use_container_width=True, hide_index=True)

    st.markdown("### 论文报告摘要")
    if latest["report"]:
        st.markdown(latest["report"])
    else:
        st.info("没有找到 pairwise_zero_diagnostic_report.md。")

    st.markdown("### 静态图")
    for figure in latest["figures"]:
        if figure.exists():
            st.image(str(figure), caption=figure.name, use_container_width=True)


def metric_source_table(data: MechanismData, metric: str) -> pd.DataFrame:
    if metric == "alpha":
        return data.gate.copy()
    if metric in {"signed_effect", "expression", "positive_fraction"}:
        return data.residual.copy()
    if metric in {"cross_harm", "conflict_score"}:
        table = data.interference.copy()
        if "expert" not in table.columns:
            table["expert"] = table["expert_a"] + "|" + table["expert_b"]
        return table
    return pd.DataFrame()


def render_metric_note(metric: str) -> None:
    notes = {
        "alpha": (
            "`alpha` 是最终写进 gate 的模块注入系数。大于 1 表示放大 expert residual，"
            "小于 1 表示压低。它是结果变量，需要回看 owner_signal、harm_signal 和 noise_score 才能解释。"
        ),
        "signed_effect": (
            "`signed_effect` 是 RCRF 的核心方向指标。正值表示 `DeltaW h` 与负梯度同向，"
            "一阶上会降低当前 teacher-forced response loss；负值说明这个模块残差在当前 span 上可能有害。"
        ),
        "expression": (
            "`expression` 只说明 residual 被激活得多不多，不说明方向对不对。"
            "高 expression + 负 signed_effect 是最重要的异常模式。"
        ),
        "cross_harm": (
            "`cross_harm` 表示某 expert-module 对非 owner task 的负收益。"
            "它高时，说明这个 residual 可能是跨任务干扰源。"
        ),
        "conflict_score": (
            "`conflict_score` 来自不同 expert induced residual 的方向冲突。"
            "q/k 上高 conflict 更像路由冲突，MLP 上高 conflict 更像能力写入冲突。"
        ),
        "positive_fraction": (
            "`positive_fraction` 衡量 signed_effect 在样本上的符号稳定性。"
            "均值为正但 positive_fraction 低，通常说明 prompt/span 选择不稳定。"
        ),
        "cosine": (
            "`cosine` 是同一输入上两个 expert 的 `DeltaW h` 余弦。负值说明它们在该模块推向相反方向。"
        ),
        "shared_positive_effect": (
            "`shared_positive_effect` 表示两个 expert 在同一任务上同时产生正向一阶收益。"
            "这类位置更像共享能力通道，不能只因跨任务出现就压低。"
        ),
    }
    if metric in notes:
        st.info(notes[metric])


def render_chart_guide(title: str, meaning: list[str], red_flags: list[str], next_step: str) -> None:
    with st.expander(title, expanded=True):
        left, middle, right = st.columns(3)
        with left:
            st.markdown("**这张图表达什么**")
            for item in meaning:
                st.markdown(f"- {item}")
        with middle:
            st.markdown("**重点看什么异常**")
            for item in red_flags:
                st.markdown(f"- {item}")
        with right:
            st.markdown("**下一步怎么验证**")
            st.markdown(next_step)


def filter_table(
    table: pd.DataFrame,
    *,
    experts: Iterable[str] | None = None,
    tasks: Iterable[str] | None = None,
    candidates: Iterable[str] | None = None,
    modules: Iterable[str] | None = None,
) -> pd.DataFrame:
    result = table.copy()
    if experts and "expert" in result.columns:
        result = result[result["expert"].isin(experts)]
    if tasks and "task" in result.columns:
        result = result[result["task"].isin(tasks)]
    if candidates and "candidate" in result.columns:
        result = result[result["candidate"].isin(candidates)]
    if modules and "module" in result.columns:
        result = result[result["module"].isin(modules)]
    return result


def multiselect_or_all(label: str, options: Iterable[str]) -> list[str]:
    values = [str(item) for item in options if str(item) not in {"", "None", "nan"}]
    return st.multiselect(label, values, default=values)


def ordered_candidates(values: Iterable[str]) -> list[str]:
    values = [str(v) for v in values]
    known = [candidate for candidate in RCRF_CANDIDATES if candidate in values]
    rest = sorted(v for v in values if v not in known)
    return known + rest


def heatmap_layer_module(df: pd.DataFrame, *, value: str, title: str, colorbar: str) -> go.Figure:
    table = df.copy()
    table["module"] = pd.Categorical(table["module"], categories=MODULE_ORDER, ordered=True)
    pivot = table.pivot_table(index="layer", columns="module", values=value, aggfunc="mean", observed=False).sort_index()
    pivot = pivot.reindex(columns=[m for m in MODULE_ORDER if m in pivot.columns])
    zmid = 0.0 if value in {"signed_effect", "cosine"} else None
    scale = "RdBu_r" if value in {"signed_effect", "cosine"} else "Viridis"
    fig = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale=scale,
        color_continuous_midpoint=zmid,
        title=title,
        labels={"x": "module / 模块", "y": "layer / 层", "color": colorbar},
    )
    fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
    fig.update_xaxes(title_text="module / 模块 (q/k/v/o/gate/up/down)")
    fig.update_yaxes(title_text="layer / 层")
    return fig


def gate_action(alpha: float) -> str:
    alpha = float(alpha)
    if alpha >= 1.03:
        return "amplify"
    if alpha >= 0.97:
        return "retain"
    if alpha >= 0.70:
        return "suppress"
    return "sparse"


def export_controls(fig: go.Figure, table: pd.DataFrame, stem: str) -> None:
    cols = st.columns(4)
    csv_bytes = table.to_csv(index=False).encode("utf-8")
    cols[0].download_button("CSV", csv_bytes, file_name=f"{stem}.csv", mime="text/csv", key=f"{stem}_csv")
    html_bytes = fig.to_html(include_plotlyjs="cdn").encode("utf-8")
    cols[1].download_button("HTML", html_bytes, file_name=f"{stem}.html", mime="text/html", key=f"{stem}_html")
    try:
        png_bytes = fig.to_image(format="png", scale=2)
        pdf_bytes = fig.to_image(format="pdf")
    except Exception:
        cols[2].caption("PNG requires kaleido")
        cols[3].caption("PDF requires kaleido")
        return
    cols[2].download_button("PNG", png_bytes, file_name=f"{stem}.png", mime="image/png", key=f"{stem}_png")
    cols[3].download_button("PDF", pdf_bytes, file_name=f"{stem}.pdf", mime="application/pdf", key=f"{stem}_pdf")


if __name__ == "__main__":
    main()
