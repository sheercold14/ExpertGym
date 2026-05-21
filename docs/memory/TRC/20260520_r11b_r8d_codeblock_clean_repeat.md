# R11B / R8D-e08 Code-Block Early-Stop Memory

## Core Setting

R11B is the early-stop alias of R8D epoch 8:

- Checkpoint: `/tmp/shared-storage/OnPolicy/checkpoints/trc_r11b_r8d_codeblock_e08_20260520-selected`
- Source run: `/tmp/shared-storage/OnPolicy/runs/trc/trc_r8d_codep0_rf_codeblock_e12_20260520`
- Source config: CodeP0 RF-only calibration, Code response span set to `code-block`, Code topK tokens 384, hidden layers `4,8,12,16,20,24,28`.
- Training selected epoch 8 rather than epoch 12 because Code behavior appears early-stop sensitive.

## Clean Evaluation Rule

Do not use gate coefficients to decide promotion. The clean rule is:

1. Tool and Memory are fast enough to run first.
2. Promote to Code only if Tool mean acc >= 0.79 and Memory mean F1 >= 0.76.
3. Gate coefficients are diagnostic only.

## Clean Repeat Results

Tool BFCL repeat on 2026-05-20:

- Mean over four selected BFCL categories: `0.7944`
- live_parallel: `0.8125`
- live_parallel_multiple: `0.6250`
- parallel: `0.8850`
- parallel_multiple: `0.8550`

Memory HotpotQA repeat on 2026-05-20:

- Mean F1: `0.7668`
- eval_50: `0.7817`
- eval_100: `0.7704`
- eval_qa_1_32768: `0.7825`
- eval_qa_1_65536: `0.7325`

This passes the Tool/Memory gate and was sent to Code as `r11b_code_repeat_clean_20260520`.

Code clean repeat:

- LiveBench: Acc `0.3691`, BoN `0.4609`
- LiveCodeBench: Acc `0.2715`, BoN `0.3444`
- Mean Code Acc: `0.3203`
- Mean Code BoN: `0.4027`

## Method Insight

R11B is valuable because it keeps Tool and Memory above the practical promotion threshold while preserving a strong Code BoN signal. The likely useful ingredients are:

- RF-only CodeP0 rows instead of mixed Code teacher sources.
- Code-block span alignment rather than full response alignment.
- Early stopping before over-pushing the code direction.
- Stable Tool/Memory rows shared with the stronger late3 banks.

The clean Code repeat did not reproduce the historical high BoN score: historical mean was `0.3272/0.4310`, clean repeat mean is `0.3203/0.4027`. This keeps R11B as a useful stable reference, but not as proof that the code-block early-stop branch solves Code. The correct takeaway is narrower:

- Tool/Memory stability is reproducible.
- LiveBench single-sample remains in the same band.
- Code BoN is unstable and likely sensitive to sampling / execution / resource details.
- Future Code claims should rely on clean repeated CURE runs, not a single high-BoN run.
