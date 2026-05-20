# 20260520 Stable Strong Model Mainline

## Objective

The main objective is not to maximize a gate coefficient. The objective is to obtain a stable merged model that is jointly strong on:

- Tool: BFCL quick mean, with explicit attention to non-live `parallel` and `parallel_multiple`.
- Memory: HotpotQA mean F1.
- Code: CURE LiveBench + LiveCodeBench mean Acc, with BoN gap used only as diagnosis.

Promotion rule remains evaluation-first:

1. Run Tool/Memory quick gate.
2. Only run expensive Code if Tool mean >= `0.79` and Memory mean F1 >= `0.76`.
3. Use gate/loss curves only to explain dynamics, not to select a paper claim by themselves.

## Current Strong Anchors

| family | status | lesson |
|---|---|---|
| R5A / R11B | historically strongest balanced TRC candidates | Tool and Memory can be kept stable; Code has high BoN but modest single-sample Acc. |
| R16B | passed quick gate; Code running | Non-leak CodeP0 pass/fail contrast with weight 3.0 can keep Tool/Memory above threshold. Code result decides whether this is a useful mainline. |
| R16C | quick gate passed; Code running | ReasonFlux-only Code rows do not break Tool/Memory, but Tool non-live still lags stronger historical runs. Code eval tests whether RF-only trajectories improve Acc. |
| R17A | Tool/Memory passed | No prompt-base drift gives faster hidden-loss descent and Tool mean `0.7969`; despite low memory-gate telemetry, official Memory F1 is `0.7645`, so it is a valid Code candidate. |
| R17B | Tool/Memory passed | Prompt expert-residual loss is the cleaner hypothesis test for prompt understanding. Tool passes (`0.7969`) and official Memory passes (`0.7643`), but `qa_65536=0.7391` remains weak. |

## Method Insight So Far

TRC hidden-residual learning is promising because it directly optimizes the task-vector behavior span instead of relying on sparse reward-only gate gradients. The stable story should be:

- task vectors contain behavior directions;
- expert trajectories identify where those directions matter;
- calibration should expose recoverable ability spans, not just final rewards;
- selection must be by heldout evaluation, because gate magnitude and proxy loss are not reliable standalone metrics.

## Current Failure Modes

### Code

Code is not simply "coefficient too low". Multiple models can raise Code gate and still get formal Acc around `0.32-0.33`. The more useful diagnostic is BoN-to-Acc gap:

- high BoN but low Acc means the model sometimes reaches a correct program but cannot make the correct response the default;
- same-prompt pass/fail contrast is the current best clean direction;
- formal Code eval is expensive, so only quick-gate-passing models should enter it.

### Tool

Tool mean can pass while non-live remains weak. Current recurring pattern:

- live_parallel often `0.8125`;
- live_parallel_multiple often `0.6250`;
- non-live parallel around `0.8800`;
- non-live parallel_multiple around `0.8550`;
- stronger historical models reached roughly `0.905 / 0.875` on non-live.

This suggests a specific Tool TODO: build non-leak BFCL-style non-live anchors, inspect `cannot_find_match`, `wrong_count`, and `decoder_failed`, and protect exact tool-call formatting. Do not tune only against mean.

### Memory

Memory is fragile under prompt-span changes. Removing prompt-base drift may remove an artificial constraint, but it can also let Code/Tool span losses reshape shared prompt representations in a way that hurts long-context memory behavior. R17A/R17B are the clean tests.

## Next Decisions

1. Finish R16B Code eval. If Code improves over R5A/R11B while Tool/Memory pass, R16B becomes a mainline candidate.
2. Finish R16C Memory. If it passes, run Code only if GPU budget is acceptable; RF-only Code rows test expert-vector purity.
3. Run R16D quick gate after a Tool slot opens. It tests whether Code response span is better than code-block span.
4. Finish R17A/R17B, then quick gate. If R17A fails Memory and R17B passes, the prompt story becomes: "do not pull prompt to base; use task-vector prompt residual carefully." If both fail Memory, keep prompt alignment out of the main method.
5. Add a Tool non-live repair branch only after current Code/prompt tests are recorded, so the paper story remains disentangled.

## Next Repair Branches

### Tool Non-Live Repair

The current Tool issue is not global tool collapse. The model usually preserves python-like tool calls, but BFCL non-live still loses points on:

- `cannot_find_match`: generated call exists but does not match a required function/argument tuple;
- `wrong_count`: multi-call count/order/coverage mismatch;
- `decoder_failed`: parser-level formatting failure.

The next Tool calibration should therefore not simply add more generic ToolRL rows. It should add a small non-leak BFCL-style bank:

- keep 32 Tool rows total to avoid becoming a training set;
- retain a stable ToolRL / existing Tool bank subset;
- add BFCL non-live anchors, especially `parallel` and `parallel_multiple`;
- require a real success trajectory from a historically strong checkpoint or the Tool expert;
- if available, store the current failing trajectory for same-prompt positive/negative tool-call span contrast.

Current BFCL Tool augmentation is only 16 rows: 4 `parallel`, 4 `parallel_multiple`, 4 `live_parallel`, and 4 `live_parallel_multiple`. This is too thin for non-live robustness, especially because the recurring errors are multi-call coverage and exact argument matching. The next Tool branch should shift the 16 BFCL slots toward non-live, e.g. 6-8 `parallel`, 8-10 `parallel_multiple`, and keep live coverage mostly through the original Tool bank unless live starts to regress.

Training objective should protect exact tool-call spans, not general answer text. Promotion requires both BFCL quick mean and non-live sub-scores, plus ToolRL all80 sanity so the method does not overfit BFCL formatting.

### Code BoN-To-Acc Repair

The Code bottleneck is still default-sample accuracy, not capability ceiling. Many candidates have reasonable BoN but modest Acc. R16B/R16C/R16D will decide:

- R16B: non-leak CodeP0 pass/fail contrast with stronger contrast weight;
- R16C: RF-only positive/contrast trajectories, testing expert-vector purity;
- R16D: response-span topK256, testing whether reasoning span helps default Acc.

If all three stay near the historical `0.32-0.33` mean Acc band, the conclusion is that hidden residual alone gives a capability direction but insufficient selection pressure. The next Code method should use same-prompt pass/fail contrast more directly on ability spans, and should avoid adding formal eval leak to the paper mainline.

### Prompt Span

R17A/R17B are ablations, not mainline by default. If either fails Memory, prompt-span alignment should be kept out of the primary method. If one passes, the claim should be narrow: prompt hidden states can encode task understanding, but must be aligned by expert residual rather than pulled to base.

## Live Update 2026-05-20 15:48 CST

- R16B Code formal eval is near completion. LiveBench is `Acc=0.3770, BoN=0.4375`; final decision waits for LiveCodeBench and mean Acc.
- R16C Tool/Memory passed (`0.7931` / `0.7632`), and Code formal eval has been launched. It can test whether RF-only Code trajectories are cleaner than mixed RF/DeepSeek trajectories.
- R16D quick gate has started. This is the clean span-control against R16A/B.
- R17A Tool/Memory passed (`0.7969` / `0.7645`) despite Memory telemetry decline; this supports treating prompt-base drift removal as a real candidate, pending Code.
- R17B completed and baked; final Memory telemetry remains low. This suggests prompt residual is not obviously a free improvement; evaluate before making it part of the paper method.
- R16D Tool/Memory passed (`0.7944` / `0.7604`), barely above the Memory threshold. It is a valid Code candidate and directly tests whether Code response-span topK256 improves formal Code over code-block384.
- R17B Tool/Memory passed (`0.7969` / `0.7643`). Because both R17A and R17B passed despite similar low Memory telemetry, prompt-span experiments should be judged by Code outcome and long-context Memory stability, not by gate telemetry alone.
- R17B Tool matches R17A, so prompt-span experiments are specifically about long-context Memory preservation and Code transfer, not Tool robustness.
