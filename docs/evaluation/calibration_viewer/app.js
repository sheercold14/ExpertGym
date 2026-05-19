const embedded = window.CALIBRATION_VIEWER_DATA || { dataset: {}, rows: [], reward_explanations: {} };

const state = {
  data: embedded,
  rows: embedded.rows || [],
  filteredRows: embedded.rows || [],
  selectedId: null,
  task: "all",
  role: "all",
  query: "",
  activeTab: "overview",
};

const els = {
  datasetPath: document.querySelector("#datasetPath"),
  copyPathButton: document.querySelector("#copyPathButton"),
  fileInput: document.querySelector("#fileInput"),
  searchInput: document.querySelector("#searchInput"),
  taskFilters: document.querySelector("#taskFilters"),
  roleFilters: document.querySelector("#roleFilters"),
  summaryStats: document.querySelector("#summaryStats"),
  rowList: document.querySelector("#rowList"),
  tabs: Array.from(document.querySelectorAll(".tab")),
  panels: {
    overview: document.querySelector("#overviewTab"),
    rollouts: document.querySelector("#rolloutsTab"),
    reward: document.querySelector("#rewardTab"),
    livecodebench: document.querySelector("#livecodebenchTab"),
    reference: document.querySelector("#referenceTab"),
    raw: document.querySelector("#rawTab"),
  },
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function jsonBlock(value, extraClass = "") {
  return `<pre class="${extraClass}">${escapeHtml(JSON.stringify(value, null, 2))}</pre>`;
}

function textBlock(value, extraClass = "prompt-block") {
  return `<pre class="${extraClass}">${escapeHtml(value || "")}</pre>`;
}

function countBy(rows, keyFn) {
  return rows.reduce((acc, row) => {
    const key = keyFn(row) || "unknown";
    acc[key] = (acc[key] || 0) + 1;
    return acc;
  }, {});
}

function sortedKeys(object) {
  return Object.keys(object || {}).sort((a, b) => a.localeCompare(b));
}

function taskBadge(task) {
  return `<span class="badge ${escapeHtml(task)}">${escapeHtml(task)}</span>`;
}

function compactPrompt(prompt) {
  return String(prompt || "").replace(/\s+/g, " ").trim();
}

function selectedRow() {
  return state.rows.find((row) => row.prompt_id === state.selectedId) || state.filteredRows[0] || state.rows[0];
}

function matchesQuery(row, query) {
  if (!query) return true;
  const haystack = [
    row.prompt_id,
    row.prompt,
    row.task,
    row.role,
    row.verifier,
    row.source,
    ...(row.tags || []),
  ]
    .join(" ")
    .toLowerCase();
  return haystack.includes(query.toLowerCase());
}

function applyFilters() {
  state.filteredRows = state.rows.filter((row) => {
    const taskOk = state.task === "all" || row.task === state.task;
    const roleOk = state.role === "all" || row.role === state.role;
    return taskOk && roleOk && matchesQuery(row, state.query);
  });
  if (!state.filteredRows.some((row) => row.prompt_id === state.selectedId)) {
    state.selectedId = state.filteredRows[0]?.prompt_id || state.rows[0]?.prompt_id || null;
  }
}

function renderDatasetHeader() {
  const dataset = state.data.dataset || {};
  els.datasetPath.textContent = dataset.path || "Loaded from browser file";
}

function renderTaskFilters() {
  const tasks = ["all", ...sortedKeys(countBy(state.rows, (row) => row.task))];
  els.taskFilters.innerHTML = tasks
    .map(
      (task) =>
        `<button type="button" class="${state.task === task ? "active" : ""}" data-task="${escapeHtml(task)}">${escapeHtml(task)}</button>`,
    )
    .join("");
}

function renderRoleFilters() {
  const roles = ["all", ...sortedKeys(countBy(state.rows, (row) => row.role || "none"))];
  els.roleFilters.innerHTML = roles
    .map((role) => {
      const label = role === "" ? "none" : role;
      return `<button type="button" class="chip ${state.role === role ? "active" : ""}" data-role="${escapeHtml(role)}">${escapeHtml(label)}</button>`;
    })
    .join("");
}

function renderStats() {
  const dataset = state.data.dataset || {};
  const taskCounts = dataset.task_counts || countBy(state.rows, (row) => row.task);
  const visibleCounts = countBy(state.filteredRows, (row) => row.task);
  const stats = [
    ["Rows", state.rows.length],
    ["Visible", state.filteredRows.length],
    ["Tool", `${visibleCounts.tool || 0}/${taskCounts.tool || 0}`],
    ["Memory", `${visibleCounts.memory || 0}/${taskCounts.memory || 0}`],
    ["Code", `${visibleCounts.code || 0}/${taskCounts.code || 0}`],
    ["Roles", sortedKeys(dataset.role_counts || countBy(state.rows, (row) => row.role)).length],
  ];
  els.summaryStats.innerHTML = stats
    .map(
      ([label, value]) => `
        <div class="stat">
          <div class="value">${escapeHtml(value)}</div>
          <div class="label">${escapeHtml(label)}</div>
        </div>`,
    )
    .join("");
}

function renderRowList() {
  if (!state.filteredRows.length) {
    els.rowList.innerHTML = `<div class="empty-state">No prompts match the current filters.</div>`;
    return;
  }
  els.rowList.innerHTML = state.filteredRows
    .map((row) => {
      const active = row.prompt_id === state.selectedId ? "active" : "";
      const prompt = compactPrompt(row.prompt).slice(0, 220);
      const rolloutCount = (row.rollouts || []).reduce((sum, source) => sum + (source.samples || []).length, 0);
      return `
        <button class="row-item ${active}" type="button" data-id="${escapeHtml(row.prompt_id)}">
          <div class="row-item-title">
            <span class="pid">#${row.index} ${escapeHtml(row.prompt_id)}</span>
            ${taskBadge(row.task)}
          </div>
          ${rolloutCount ? `<div class="rollout-count">${escapeHtml(rolloutCount)} rollout samples</div>` : ""}
          <div class="row-prompt">${escapeHtml(prompt)}</div>
        </button>`;
    })
    .join("");
}

function metaItem(label, value) {
  return `
    <div class="meta-item">
      <div class="label">${escapeHtml(label)}</div>
      <div class="value">${escapeHtml(value ?? "")}</div>
    </div>`;
}

function metricText(value) {
  if (value === null || value === undefined || value === "") return "";
  if (typeof value === "number") return value.toFixed(4);
  return String(value);
}

function boolPill(value) {
  const label = value ? "pass" : "fail";
  return `<span class="bool-pill ${value ? "pass" : "fail"}">${label}</span>`;
}

function vectorPills(values) {
  if (!Array.isArray(values) || !values.length) return `<span class="muted">n/a</span>`;
  return `<div class="vector-pills">${values.map((value) => boolPill(Boolean(value))).join("")}</div>`;
}

function renderMetricCards(metrics) {
  const cards = [
    ["Total cases", metrics.total],
    ["Sample code acc", metrics.sample_acc],
    ["Hidden-test acc", metrics.hidden_test_acc],
    ["Any-success acc", metrics.any_success_acc],
    ["BoN(4,4) acc", metrics.bon_acc],
    ["BoN hidden acc", metrics.bon_hidden_test_acc],
  ];
  return `
    <div class="stat-grid livecode-metrics">
      ${cards
        .map(
          ([label, value]) => `
            <div class="stat">
              <div class="value">${escapeHtml(metricText(value))}</div>
              <div class="label">${escapeHtml(label)}</div>
            </div>`,
        )
        .join("")}
    </div>`;
}

function rewardNarrative(row) {
  if (!row) return "";
  if (row.task === "code") {
    return `
      <div class="explain-card">
        <strong>Code prompt 的 reward 判分方式</strong>
        <p>模型先生成一段自然语言和/或代码；判分器会从输出中抽取 Python 代码，检查语法、是否读取 stdin、是否打印输出。只要这条 calibration row 的 <code>reference.metadata.test_input/test_output</code> 存在，就会直接运行这些 CodeContests/CURE-style 测试。</p>
        <p>核心分数是测试通过率：<code>reward = passed_tests / total_tests</code>。本训练里 <code>reward_train</code> 通常等于这个 pass rate；<code>success=true</code> 约等于 <code>reward &gt;= 0.95</code>。如果代码抽取/语法失败，常见只给很低的格式分，比如有代码但无法解析时可能是 <code>0.05</code>。</p>
      </div>`;
  }
  if (row.task === "tool") {
    return `
      <div class="explain-card">
        <strong>Tool prompt 的 reward 判分方式</strong>
        <p>ToolRL source row 按 <code>&lt;tool_call&gt;</code> 格式和工具调用参数匹配打分；BFCL-style row 则用 AST checker 解码函数调用，检查函数名、调用数量、参数值和枚举/default/canonicalization 等。</p>
      </div>`;
  }
  if (row.task === "memory") {
    return `
      <div class="explain-card">
        <strong>Memory prompt 的 reward 判分方式</strong>
        <p>final-answer prompt 会抽取最终答案，和 HotpotQA/MemAgent 的 ground truth 做 exact/sub-exact/F1 相关判定；训练主要看 final boxed answer 是否命中。</p>
      </div>`;
  }
  return "";
}

function rolloutSourceNarrative(source) {
  const frontier = source.frontier || {};
  const samples = source.samples || [];
  const successCount = samples.filter((sample) => sample.success).length;
  const rewards = samples
    .map((sample) => Number(sample.reward_train ?? sample.reward))
    .filter((value) => Number.isFinite(value));
  const mean = rewards.length ? rewards.reduce((sum, value) => sum + value, 0) / rewards.length : null;
  const frontierText = frontier.has_variance
    ? "这组 rollout 内有 reward 方差，因此可作为 GRPO/frontier 信号。"
    : "这组 rollout 内没有明显 reward 方差，通常不会提供强 GRPO frontier 信号。";
  return `
    <div class="explain-card slim">
      <strong>这组 rollout 怎么读</strong>
      <p>这里展示的是同一个 prompt 在 <code>${escapeHtml(source.label || "rollout")}</code> 下的真实采样输出。当前嵌入 ${samples.length}/${source.sample_count || samples.length} 个 sample，其中 ${successCount} 个成功${mean === null ? "" : `，展示样本平均 reward_train 约 ${mean.toFixed(3)}`}。</p>
      <p>${escapeHtml(frontierText)} 原始 frontier 统计来自 rollout 文件，训练时会用同 prompt 内不同 sample 的 reward 差异来更新 gate。</p>
    </div>`;
}

function sampleInterpretation(sample) {
  const details = sample.details || {};
  const sourceTests = details.source_tests || null;
  const publicExamples = details.public_examples || null;
  const reward = sample.reward_train ?? sample.reward;
  if (sourceTests) {
    const total = sourceTests.total ?? 0;
    const passed = sourceTests.passed ?? 0;
    const passRate = sourceTests.pass_rate ?? reward;
    return `
      <div class="judge-note">
        <strong>中文判分解读</strong>
        <p>判分器抽取了模型输出中的代码，并在这条 prompt 自带的 source tests 上执行。通过 ${escapeHtml(passed)}/${escapeHtml(total)} 个测试，所以 pass rate 是 <code>${escapeHtml(passRate)}</code>，该值进入 <code>reward/reward_train</code>。${sample.success ? "因为分数达到成功阈值，所以记为 success。" : "因为没有达到成功阈值，所以记为 fail。"}</p>
      </div>`;
  }
  if (details.syntax_ok === false || details.parse_error) {
    return `
      <div class="judge-note">
        <strong>中文判分解读</strong>
        <p>判分器没有得到可执行的合法 Python 代码：<code>${escapeHtml(details.parse_error?.msg || "syntax/code extraction failed")}</code>。这种情况下不会跑正式测试，通常只保留很低的格式/代码存在分，因此这里 reward 是 <code>${escapeHtml(reward)}</code>。</p>
      </div>`;
  }
  if (publicExamples) {
    return `
      <div class="judge-note">
        <strong>中文判分解读</strong>
        <p>这条 sample 没有进入 source tests，判分器使用 prompt 中的 public examples fallback。通过 ${escapeHtml(publicExamples.passed ?? 0)}/${escapeHtml(publicExamples.total ?? 0)} 个 public examples，得到当前 reward。</p>
      </div>`;
  }
  if (sample.success) {
    return `
      <div class="judge-note">
        <strong>中文判分解读</strong>
        <p>该 sample 被 verifier 判为成功，<code>reward_train=${escapeHtml(reward)}</code>。具体细节见下方 Reward details。</p>
      </div>`;
  }
  return `
    <div class="judge-note">
      <strong>中文判分解读</strong>
      <p>该 sample 未达到 verifier 成功阈值，<code>reward_train=${escapeHtml(reward)}</code>。具体失败原因通常在下方 <code>details</code> 中，例如语法、输入输出、测试通过率或格式解析问题。</p>
    </div>`;
}

function renderOverview(row) {
  if (!row) {
    els.panels.overview.innerHTML = `<div class="empty-state">No row selected.</div>`;
    return;
  }
  const summary = row.reference_summary || {};
  els.panels.overview.innerHTML = `
    <section class="section">
      <div class="meta-grid">
        ${metaItem("prompt_id", row.prompt_id)}
        ${metaItem("task", row.task)}
        ${metaItem("role", row.role || "none")}
        ${metaItem("verifier", row.verifier)}
        ${metaItem("source_row", row.source_row)}
        ${metaItem("split", row.split)}
      </div>
    </section>
    <section class="section">
      <h3>Prompt</h3>
      ${textBlock(row.prompt)}
    </section>
    <section class="section">
      <h3>Reference Summary</h3>
      ${jsonBlock(summary)}
    </section>
    <section class="section">
      <h3>Selection Metadata</h3>
      ${jsonBlock({
        question_bank_selection: row.question_bank_selection,
        eval_targeted_calibration: row.eval_targeted_calibration,
        tags: row.tags,
        source: row.source,
      })}
    </section>`;
}

function renderReward(row) {
  if (!row) return;
  const reward = row.reward || {};
  const fields = reward.fields || [];
  const flow = [
    ["Route", reward.route || "RewardRouter selects adapter by task/verifier"],
    ["Adapter", `${reward.adapter || "adapter"} in ${reward.code || "reward code"}`],
    ["Formula", reward.formula || "See reward adapter source."],
    ["Success", reward.success || "See reward adapter source."],
  ];
  els.panels.reward.innerHTML = `
    <section class="section">
      <div class="meta-grid">
        ${metaItem("Verifier", row.verifier)}
        ${metaItem("Reward adapter", reward.adapter || "")}
        ${metaItem("Code path", reward.code || "")}
      </div>
    </section>
    <section class="section">
      <h3>Reward Flow</h3>
      <div class="reward-flow">
        ${flow
          .map(
            ([title, body]) => `
          <div class="flow-step">
            <strong>${escapeHtml(title)}</strong>
            <span>${escapeHtml(body)}</span>
          </div>`,
          )
          .join("")}
      </div>
    </section>
    <section class="section">
      <h3>中文解读</h3>
      ${rewardNarrative(row)}
    </section>
    <section class="section">
      <h3>Fields Used</h3>
      <ul class="field-list">
        ${fields.map((field) => `<li>${escapeHtml(field)}</li>`).join("")}
      </ul>
    </section>
    <section class="section">
      <h3>Row-Specific Reward Inputs</h3>
      ${jsonBlock({
        reference_summary: row.reference_summary,
        verifier: row.verifier,
        reference: row.reference,
      })}
    </section>`;
}

function renderLiveCodeBench() {
  const live = state.data.livecodebench_eval;
  if (!live) {
    els.panels.livecodebench.innerHTML = `
      <section class="section">
        <h3>No LiveCodeBench audit data embedded</h3>
        <p class="note">Regenerate this viewer with <code>--livecode-app-data</code> to attach real CURE LiveCodeBench rollout and metric examples.</p>
      </section>`;
    return;
  }
  if (live.missing) {
    els.panels.livecodebench.innerHTML = `
      <section class="section">
        <h3>LiveCodeBench audit data missing</h3>
        ${jsonBlock(live)}
      </section>`;
    return;
  }
  const examples = live.examples || [];
  els.panels.livecodebench.innerHTML = `
    <section class="section">
      <h3>真实评测流程</h3>
      <div class="explain-card">
        <strong>CURE / LiveCodeBench 如何测试</strong>
        <p>这部分来自真实的 CURE eval artifact，不是 calibration 训练数据。评测对每道 LiveCodeBench 题采样代码候选和生成测试，执行得到 <code>test_bool_table</code> 与 <code>case_bool_table</code>，再计算 code acc、unit-test acc 和 BoN(4,4)。</p>
        <p>LiveCodeBench 只用于正式 heldout eval；当前 calibration code 行没有复制这些 prompt、hidden tests 或输出。</p>
      </div>
      <div class="reward-flow livecode-flow">
        ${(live.flow || [])
          .map(
            (step) => `
              <div class="flow-step">
                <strong>${escapeHtml(step.title)}</strong>
                <span>${escapeHtml(step.description)}</span>
              </div>`,
          )
          .join("")}
      </div>
    </section>
    <section class="section">
      <h3>本地真实路径</h3>
      <div class="meta-grid">
        ${metaItem("model", `${live.display_name || live.model_id} (${live.model_id})`)}
        ${metaItem("cases", live.total_cases)}
        ${metaItem("app_data", live.app_data_path)}
        ${metaItem("result_txt", live.result_path)}
        ${metaItem("eval.py", live.cure_eval_script)}
        ${metaItem("outputs", (live.output_files || []).join(" | "))}
      </div>
    </section>
    <section class="section">
      <h3>全量指标</h3>
      ${renderMetricCards(live.metrics || {})}
      <details>
        <summary>Raw result text</summary>
        ${textBlock(live.raw_result_text || "", "prompt-block")}
      </details>
      <details>
        <summary>Metric formulas</summary>
        ${jsonBlock(live.formulas || {})}
      </details>
    </section>
    <section class="section">
      <h3>真实 rollout 与指标计算例子</h3>
      ${examples.map((example) => livecodeExampleCard(example)).join("")}
    </section>`;
}

function livecodeExampleCard(example) {
  const metric = example.metric_calculation || {};
  const candidates = example.candidates || [];
  const tests = example.generated_tests || [];
  const metrics = example.metrics || {};
  return `
    <article class="livecode-example">
      <div class="sample-head">
        <div>
          <strong>${escapeHtml(example.label)} · ${escapeHtml(example.case_id)}</strong>
          <span>${escapeHtml((example.code_tags || []).join(", "))}</span>
        </div>
        <div class="sample-score">
          <b>${escapeHtml(metricText(metrics.bon_hidden_test_acc))}</b>
          <span>${escapeHtml(metrics.bon_success ? "BoN pass" : "BoN fail")}</span>
        </div>
      </div>
      <div class="explain-card slim">
        <strong>这条样例怎么读</strong>
        <p>候选代码先在 generated tests 上打分：前 4 个候选的通过数是 <code>${escapeHtml((metric.case_scores_first_4x4 || []).join(", "))}</code>，CURE 选择 index <code>${escapeHtml(metric.bon_selected_index)}</code>。然后只用 hidden tests 评估被选中的代码：结果向量如下，全部通过才给 BoN success。</p>
        ${vectorPills(metric.selected_hidden_test_vector)}
      </div>
      <div class="meta-grid compact">
        ${metaItem("code_success", `${metrics.code_success_count}/${metrics.code_sample_count}`)}
        ${metaItem("hidden_test_acc", metricText(metrics.hidden_test_acc))}
        ${metaItem("generated_tests", metrics.generated_test_count)}
        ${metaItem("hidden_tests", metrics.hidden_test_count)}
      </div>
      <details>
        <summary>Problem prompt</summary>
        ${textBlock(example.question || "", "prompt-block")}
      </details>
      <details open>
        <summary>Candidate code rollout</summary>
        <div class="candidate-grid">
          ${candidates.map((candidate) => livecodeCandidateCard(candidate)).join("")}
        </div>
      </details>
      <details>
        <summary>Generated unit tests</summary>
        <div class="test-grid">
          ${tests.map((test) => livecodeGeneratedTestCard(test)).join("")}
        </div>
      </details>
      <details>
        <summary>Bool tables and hidden tests</summary>
        ${jsonBlock({
          metric_calculation: example.metric_calculation,
          case_bool_table_first_4x4: example.case_bool_table_first_4x4,
          test_bool_table: example.test_bool_table,
          hidden_tests_preview: example.hidden_tests_preview,
          output_file: example.output_file,
          failure_tags: example.failure_tags,
        })}
      </details>
    </article>`;
}

function livecodeCandidateCard(candidate) {
  return `
    <section class="candidate-card ${candidate.selected_by_bon ? "selected" : ""}">
      <div class="candidate-head">
        <strong>candidate ${escapeHtml(candidate.index)}</strong>
        ${candidate.selected_by_bon ? `<span>BoN selected</span>` : ""}
      </div>
      <div class="mini-row">
        <span>generated tests</span>
        ${vectorPills(candidate.generated_test_vector_first_4)}
      </div>
      <div class="mini-row">
        <span>hidden tests</span>
        ${vectorPills(candidate.hidden_test_vector)}
      </div>
      <div class="mini-row">
        <span>passes all hidden</span>
        ${boolPill(candidate.passes_all_hidden_tests)}
      </div>
      ${textBlock(candidate.code_preview || "", "prompt-block candidate-code")}
    </section>`;
}

function livecodeGeneratedTestCard(test) {
  return `
    <section class="candidate-card">
      <div class="candidate-head">
        <strong>generated test ${escapeHtml(test.index)}</strong>
        <span>${escapeHtml(test.wrong_candidate_reject_count)}/${escapeHtml(test.wrong_candidate_count)} wrong rejected</span>
      </div>
      <div class="mini-row">
        <span>candidate pass vector</span>
        ${vectorPills(test.pass_vector_by_candidate)}
      </div>
      <div class="mini-row">
        <span>all correct pass</span>
        ${boolPill(test.passed_by_all_correct_candidates)}
      </div>
      <div class="two-col">
        ${textBlock(test.input || "", "prompt-block generated-test")}
        ${textBlock(test.output || "", "prompt-block generated-test")}
      </div>
    </section>`;
}

function sampleCard(sample, sourceLabel, index) {
  const passed = sample.success ? "success" : "failure";
  const sourceTests = sample.details?.source_tests || null;
  const testSummary = sourceTests
    ? `${sourceTests.passed ?? 0}/${sourceTests.total ?? 0} tests passed`
    : sample.details?.public_examples
      ? `${sample.details.public_examples.passed ?? 0}/${sample.details.public_examples.total ?? 0} public examples passed`
      : "no executable test details";
  return `
    <section class="sample-card ${passed}">
      <div class="sample-head">
        <div>
          <strong>${escapeHtml(sourceLabel)} · sample ${index + 1}</strong>
          <span>${escapeHtml(sample.sample_id || "")}</span>
        </div>
        <div class="sample-score">
          <b>${escapeHtml(sample.reward_train ?? sample.reward ?? "")}</b>
          <span>${escapeHtml(sample.success ? "success" : "fail")}</span>
        </div>
      </div>
      <div class="meta-grid compact">
        ${metaItem("reward", sample.reward)}
        ${metaItem("reward_train", sample.reward_train)}
        ${metaItem("length", sample.length)}
        ${metaItem("tests", testSummary)}
      </div>
      ${sampleInterpretation(sample)}
      <details open>
        <summary>Model output</summary>
        ${textBlock(sample.text || "")}
      </details>
      <details>
        <summary>Reward details</summary>
        ${jsonBlock(sample.details || {})}
      </details>
    </section>`;
}

function renderRollouts(row) {
  if (!row) return;
  const sources = row.rollouts || [];
  const datasetSources = state.data.dataset?.rollout_sources || [];
  if (!sources.length) {
    els.panels.rollouts.innerHTML = `
      <section class="section">
        <h3>No rollout samples embedded for this prompt</h3>
        <p class="note">This viewer currently embeds real rollout samples for code prompts. Use task=code or regenerate data with additional rollout task filters.</p>
      </section>
      <section class="section">
        <h3>Embedded Rollout Sources</h3>
        ${jsonBlock(datasetSources)}
      </section>`;
    return;
  }
  els.panels.rollouts.innerHTML = `
    <section class="section">
      <h3>What This Shows</h3>
      <div class="explain-card">
        <strong>Rollout 信息说明</strong>
        <p>这里不是 calibration 的 reference answer，而是模型/专家在这条 prompt 上真实采样出的输出。每个 rollout source 代表一个 policy checkpoint 或 expert policy；每个 sample 都已经被同一个 RewardRouter 打过分。</p>
        <p>对 code prompt，最重要的是看三件事：模型输出的代码是什么、source tests 通过了多少、<code>reward_train</code> 是否足以让该 sample 被当作 success 或 OPD positive。</p>
      </div>
      <div class="reward-flow">
        <div class="flow-step"><strong>Prompt</strong><span>The selected calibration row is sent to the current policy or expert policy.</span></div>
        <div class="flow-step"><strong>Output</strong><span>Each rollout sample is the actual generated text stored in JSONL.</span></div>
        <div class="flow-step"><strong>Verifier</strong><span>RewardRouter runs the task adapter; for code this is CURE-style pass rate.</span></div>
        <div class="flow-step"><strong>Training</strong><span>reward_train and success drive frontier/OPD selection for gate updates.</span></div>
      </div>
    </section>
    ${sources
      .map((source) => {
        const samples = source.samples || [];
        return `
          <section class="section rollout-source">
            <h3>${escapeHtml(source.label || "rollout")}</h3>
            ${rolloutSourceNarrative(source)}
            <div class="meta-grid">
              ${metaItem("run_id", source.run_id)}
              ${metaItem("policy_id", source.policy_id)}
              ${metaItem("samples embedded", `${samples.length}/${source.sample_count || samples.length}`)}
              ${metaItem("keep_for_policy_loss", source.keep_for_policy_loss)}
              ${metaItem("path", source.path)}
              ${metaItem("frontier", source.frontier ? JSON.stringify(source.frontier) : "")}
            </div>
            ${samples.map((sample, index) => sampleCard(sample, source.label || "rollout", index)).join("")}
          </section>`;
      })
      .join("")}`;
}

function renderReference(row) {
  if (!row) return;
  els.panels.reference.innerHTML = `
    <section class="section">
      <h3>Messages</h3>
      ${jsonBlock(row.messages)}
    </section>
    <section class="section">
      <h3>Reference</h3>
      ${jsonBlock(row.reference)}
    </section>`;
}

function renderRaw(row) {
  if (!row) return;
  els.panels.raw.innerHTML = `
    <section class="section">
      <h3>Raw Preview</h3>
      ${jsonBlock(row.raw_preview)}
    </section>
    <section class="section">
      <h3>Embedded Row</h3>
      ${jsonBlock(row)}
    </section>`;
}

function renderDetails() {
  const row = selectedRow();
  if (row && !state.selectedId) state.selectedId = row.prompt_id;
  renderOverview(row);
  renderRollouts(row);
  renderReward(row);
  renderLiveCodeBench();
  renderReference(row);
  renderRaw(row);
}

function renderAll() {
  applyFilters();
  renderDatasetHeader();
  renderTaskFilters();
  renderRoleFilters();
  renderStats();
  renderRowList();
  renderDetails();
}

function parseJsonl(text, fileName) {
  const rows = text
    .split(/\r?\n/)
    .filter((line) => line.trim())
    .map((line, index) => {
      const raw = JSON.parse(line);
      return {
        index: index + 1,
        prompt_id: raw.prompt_id || `row_${index + 1}`,
        prompt_hash: raw.prompt_hash,
        task: raw.task || "unknown",
        role: raw.eval_targeted_calibration?.role || "",
        verifier: raw.verifier?.name || "",
        reward: embedded.reward_explanations?.[raw.verifier?.name] || {},
        source: raw.source,
        source_row: raw.source_row,
        split: raw.split,
        tags: raw.tags || [],
        prompt: raw.prompt || "",
        messages: raw.messages || [],
        reference: raw.reference || {},
        reference_summary: raw.reference_summary || {},
        eval_targeted_calibration: raw.eval_targeted_calibration || {},
        question_bank_selection: raw.question_bank_selection || {},
        raw_preview: raw,
      };
    });
  return {
    dataset: {
      name: fileName,
      path: fileName,
      row_count: rows.length,
      task_counts: countBy(rows, (row) => row.task),
      role_counts: countBy(rows, (row) => row.role),
      verifier_counts: countBy(rows, (row) => row.verifier),
    },
    reward_explanations: embedded.reward_explanations || {},
    rows,
  };
}

els.searchInput.addEventListener("input", (event) => {
  state.query = event.target.value;
  renderAll();
});

els.taskFilters.addEventListener("click", (event) => {
  const button = event.target.closest("[data-task]");
  if (!button) return;
  state.task = button.dataset.task;
  renderAll();
});

els.roleFilters.addEventListener("click", (event) => {
  const button = event.target.closest("[data-role]");
  if (!button) return;
  state.role = button.dataset.role;
  renderAll();
});

els.rowList.addEventListener("click", (event) => {
  const button = event.target.closest("[data-id]");
  if (!button) return;
  state.selectedId = button.dataset.id;
  renderRowList();
  renderDetails();
});

els.tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    state.activeTab = tab.dataset.tab;
    els.tabs.forEach((item) => item.classList.toggle("active", item.dataset.tab === state.activeTab));
    Object.entries(els.panels).forEach(([name, panel]) => {
      panel.classList.toggle("active", name === state.activeTab);
    });
  });
});

els.copyPathButton.addEventListener("click", async () => {
  const path = state.data.dataset?.path || "";
  if (!path) return;
  try {
    await navigator.clipboard.writeText(path);
    els.copyPathButton.textContent = "Copied";
    setTimeout(() => {
      els.copyPathButton.textContent = "Copy Path";
    }, 1200);
  } catch {
    els.copyPathButton.textContent = "Copy failed";
    setTimeout(() => {
      els.copyPathButton.textContent = "Copy Path";
    }, 1200);
  }
});

els.fileInput.addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  const text = await file.text();
  state.data = parseJsonl(text, file.name);
  state.rows = state.data.rows;
  state.task = "all";
  state.role = "all";
  state.query = "";
  state.selectedId = state.rows[0]?.prompt_id || null;
  els.searchInput.value = "";
  renderAll();
});

renderAll();
