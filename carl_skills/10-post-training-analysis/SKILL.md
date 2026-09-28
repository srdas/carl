---
name: 10-post-training-analysis
description: Analyze final training results and propose improvements for the next RL iteration
metadata:
  sources:
    - rl-training (Final Evaluation, Diagnostics, Hyperparameter Iteration)
    - rl-visualization (Metrics, Plots, TensorBoard)
    - rl-problem-workflow (Post-launch Iteration)
---

# Step 10: Post-Training Analysis

## Audience And Voice (MANDATORY, GOVERNS ALL OUTPUT)

Write for the person who owns the problem, not for the training script. Open at
that density; never open technical and wait to be asked to simplify.

- Use the problem's own units and nouns, not tensor names, config keys, symbols,
  or estimator jargon. Expand a term on first use or drop it (SNR, GAE,
  advantage standardization, exploration support).
- Keep only numbers that carry an argument, and say what each means.
- Full technical detail goes in the `memory.md` postmortem, not the message.
- Close every substantive reply with the decision question you need answered,
  then one line offering the technical detail if the user wants it.

## Pre-Analysis Revisit Turn (MANDATORY, CONTROLLER-LED)

The controller collects optional user instructions and observations before this
analysis begins and before artifacts are inspected. Honor that preflight guidance
as user input; if the user has none, proceed normally. Never list, read, or
describe artifacts under an excluded path -- those exclusions outrank the
artifact-inventory instructions below.

## Honor Prior-Step Directives (MANDATORY, FIRST)

Before interpreting results, read `memory.md`, `step_08_conversation.md`, and
`step_09_conversation.md` -- usually the best summary of outputs, final
evaluations, and known issues -- then scan prior `step_0N_conversation.md` files
for directives affecting evaluation, benchmarks, constraints, or prohibited
methods. Emit a brief "Prior-Decisions Contract" preserving exact user DO and
DO-NOT wording; directives override the defaults below. If a file is unavailable,
say so and continue; do not pause.

Treat directive adherence as a first-class diagnostic. In `memory.md`, look for
`User directive`, `User directives honored`, and `Substitution from skill
defaults`, and verify Step 8/9 artifacts actually honored them before judging
training quality. Preserve any substitution made to respect a directive rather
than "fixing" it back to the default.

Use the full Step 9 run -- evidence unavailable during Steps 1-7 -- to decide
whether the next iteration changes training, reward, state, action, evaluation,
or the MDP itself.

## Result Inventory

Summarize final Step 9 artifacts:

- Training command, config, seeds, environment version, and time budget
- Best, latest, initial, and pretrained checkpoints
- `train.log` files, especially under `code/` or experiment directories
- Output from `long_train.sh` if run after Step 9, typically under
  `code/experiments/*long*/`, `**/runs/*long*/`, or `long_train/experiments/`
- Metrics separating direct objective, training reward/surrogate,
  checkpoint-selection score, and failure/truncation rates
- Any `.tex` files under `results/`, which often hold the written analysis

Compare **best**, **latest**, and **initial/pretrained** separately; latest can
have lower reward but better geometry, sensitivity, or objective. Compare any
long-run artifacts against the in-session Step 9 run before diagnosing plateau,
instability, or undertraining. Report direct objective performance, not only
reward or action MAE; label any benchmark that misses the implemented objective a
heuristic comparator.

## Visual Evidence Review (MANDATORY, BEFORE ANY RECOMMENDATION)

Call `list_revisit_artifacts` and treat its `plots` list as a **worklist, not a
sample**: call `describe_plot` on **every** entry before proposing any change. If
the list is empty or implausibly short, say so and investigate with
`run_command` (`find . -name '*.png'`) first -- a missing plot inventory is
itself a finding, not a reason to skip this section.

Report per plot what it says about the policy in problem terms; put the axes and
exact ranges in the record. Group descriptions; do not narrate call by call.

- **Trajectory reading:** for every rollout or state/action-over-time plot,
  describe the behaviour -- bang-bang, saturating, dithering, dead zones,
  premature termination -- not just "reward goes up".
- **Baseline comparison is required:** name the comparator series present on
  every performance plot (learned best/latest, initial/pretrained, heuristic,
  analytical/oracle) and which are *missing*. With no analytical benchmark
  plotted, call every "lift" a lift over a heuristic and flag adding the true
  benchmark as a next-run change.
- **Coverage check:** compare existing plots against Step 9's
  `### Essential Plots`; missing diagnostic views are a prescription item.
- **Evidence binding:** every entry in the `## MDP Revision Recommendations`
  ranked list must trace to specific plot(s) and/or log line(s) -- cited
  compactly in the message, in full in the record. A recommendation with no
  evidence is dropped or downgraded to "needs instrumentation".

## Training Progression Diagnostics

Inspect evaluation progression across the full run, not just the final number.

- Eval rises then degrades: check learning rate, PPO clip, entropy, ramp speed,
  value loss, and checkpoint selection. Do not simply train longer.
- Eval improves only in the first ~10%: suspect early saturation. Try slower
  learning, staged objectives, or curriculum.
- Eval oscillates widely: suspect noisy updates. Increase batch size, parameter
  diversity, eval episodes, and CRN/block-centering when valid.
- Eval is flat from the start: check reward alignment, state sufficiency, action
  ranges, initialization, and gradient flow.
- Reward improves but direct objective worsens: treat this as an MDP/reward
  issue before tuning hyperparameters.
- Geometry improves while reward is flat: repeat direct simulations; the
  objective basin may be shallow.
- Only catastrophic modes were avoided: use a curriculum -- learn safe
  trajectories first, then refine with gentler updates or tighter ranges.

## Signal, Noise, And Batch Adequacy

Estimate whether simulation noise is comparable to the objective gap between the
best learned policy and the benchmark/oracle/heuristic, using repeated evals on
fixed held-out parameter sets and independent noise paths. If the gap is the same
order as eval noise, increase eval episodes before checkpoint decisions; for the
next run, prefer larger batches, more parameter diversity, CRN/block-centering,
and curriculum stages. Small noisy PPO batches look like poor algorithms --
increase batch size before adding model complexity when returns vary widely
across parameters or noise paths.

## Training Throughput Diagnostics

Before recommending larger batches, more eval episodes, or longer training, check
whether per-update wall time is dominated by rollout collection, using
`train.log` timestamps, progress output, profiler summaries, or timing prints.

Look for independent work that can be parallelized without changing the MDP:

- Episodes within the same update
- Parameter blocks or held-out eval parameter sets
- Independent noise paths under a fixed parameter draw
- Evaluation rollouts separate from training rollouts

Recommend parallel or vectorized rollout collection only when it preserves the
randomization contract, CRN/block structure, seeding discipline, terminal
handling, and reward accounting; never trade reproducibility or which
trajectories enter an update for wall-clock time. If rollout collection is
already fast and optimizer/backprop dominates, focus on model size, device
placement, or batch schedule instead.

## Objective And Reward Alignment

First verify whether training and checkpoint selection used the full-path direct
objective or a positive affine transform. Per-step compression, clipping,
shaping, and shifted-agent averages are not ordinally reliable unless proven at
the trajectory level.

- Compare direct-best vs surrogate-best on identical held-out paths; report
  ranking inversions as reward-selection mismatch.
- Did shaping help learning but hurt the true objective, and was it annealed
  away before final evaluation?
- Did clipping, hard floors, `tanh`, or squashing hide objective differences?
- Did discounting block terminal rewards/penalties from early actions?
- Did terminal/failure rewards reach each responsible agent/buffer?
- Did truncation make early failure attractive by skipping future penalties?

## Policy Behavior Diagnostics

Inspect what the learned policy does, not only what return it achieved.

- Action distributions by dimension: mean, std, min/max, boundary hits
- Action range utilization compared with legal action ranges
- Per-step action noise: early episode vs late episode behavior
- Sensitivity sweeps over each state feature and randomized parameter
- Best/latest/initial/pretrained actions on the same representative states
- Failure cases: states and actions before breach, truncation, or severe loss

If early actions are much noisier than late ones, reward may be too sparse or
`gamma` too low: consider shaping, higher `gamma`, GAE tuning, or curriculum. If
the policy ignores a variable that should matter, diagnose observability and
features before network size. If boundary-pinned, check whether the optimum lies
outside the action range. If the trained policy is not materially better than the
initial/pretrained one, revisit state, action, reward, terminal rules, and
benchmark assumptions before a larger training job.

## Checkpoints And Fresh Initialization

Use old checkpoints and initial/pretrained actors for diagnosis, comparison, and
baselines only -- never to start a revisit training run, including training-only
revisits.

Report whether Step 9, `long_train.sh`, best/latest checkpoints, or
`experiments/pretrain/` are model-contract compatible, as evidence only. Check:

- Observation/state dimension, transforms, normalization, and feature meaning
- Action dimensions, bounds, squashing, distribution families, and role split
- Reward components, clipping/compression, terminal/failure routing, and scale
- Episode horizon, termination/truncation semantics, and parameter sampling
- Actor/critic architecture, hidden sizes, heads, and saved-key names

A changed MDP contract or architecture makes old init/checkpoint files load
incorrectly, fail shape checks, or encode the wrong reward/state/action
semantics. Even with an unchanged contract, revisits start fresh to keep the next
experiment independent of prior runs.

`revisit_train.sh` must launch the revised pipeline from a fresh seed and revised
code; rerun any initialization/pretraining stage inside the new
`code/revisit_N/` workspace first. Do not point `--init-from`, `INIT_FROM`, or
equivalents at the original session's `experiments/pretrain/`, `long_train.sh`
outputs, best/latest checkpoints, or prior revisit init files, and do not add
`--init-from` to `TRAIN_CMD` or script defaults. Record `initialization: fresh
revisit seed` with any local pretraining command or init path used.

## MDP Revision Recommendations

Finish with a ranked next-run prescription. Give each item four parts:

1. **Headline**, bold, one line: the intent in the problem's own terms ("let
   the agent try lower-risk portfolios"), not the mechanism or a config key.
2. **What is wrong**: 1-2 sentences of observed behaviour in the user's units,
   ending in why it blocks learning.
3. **Fix:** one sentence on what changes, still in problem terms.
4. **What stays the same**: the objective, reported score, or user directive
   this change does not touch.

Carry the required evidence as one short parenthetical, then one plain paragraph
on the constraints preserved.

Priority order (your diagnosis; translate before writing):

1. **Objective mismatch:** revise reward, terminal handling, shaping, clipping,
   or evaluation before hyperparameter tuning.
2. **No observability or controllability:** revise state features, transforms,
   action ranges, or action parameterization.
3. **Signal buried in noise:** increase batch/eval size, add CRN/block-centering,
   improve parameter blocking, or narrow curriculum stages.
4. **Training instability:** lower learning rate, clip, entropy, or ramp speed.
5. **Undertraining:** train longer only when eval is still improving and no
   objective/reward mismatch is visible.
6. **Throughput bottleneck:** parallelize/vectorize episode rollouts only when
   it preserves seeding, CRN/blocking, and rollout semantics.

Do not recommend "train longer" as the first fix for a flat, degraded, or
objective-misaligned run.

## Revisit Workspace

Once the user agrees on the next experiment, identify plausible session-local
source directories from the inventory, recommend one, and get explicit
confirmation immediately before calling `prepare_revisit_workspace`. The source
may be `code/`, `code/revisit_N/`, `long_train/`, or another session-local
directory holding the code/config to revise; use a long-run directory only if it
contains source, not merely logs, checkpoints, or outputs.

Then call `prepare_revisit_workspace` with the confirmed `source_dir`; it creates
a fresh self-contained `code/revisit_N/` by copying that source. Do not edit the
original source files or overwrite an older revisit directory.

Make all agreed MDP, hyperparameter, evaluation, and script changes inside that
new directory, and update `revisit_train.sh` so the user can run the revised
experiment later. It must write outputs under its own `experiments/revisit_N/`,
update `results/appendix.tex`, and rebuild `main.pdf` when `pdflatex` exists.

Use the `run_name` `prepare_revisit_workspace` returns as-is: editing `RUN_NAME`
by hand drifts the workspace and output directory apart. If the workspace has
`make_artifact.py`, keep the step where `revisit_train.sh` refreshes `artifact/`
from the revisit's own best checkpoint. Do not execute `revisit_train.sh` here --
this step prepares the next run; the user launches it after the discussion.

## Postmortem And Checklist

Record the Step 10 postmortem in `memory.md` or the step summary. This is where
detail kept out of the message lives: config keys, symbols, per-plot axis ranges,
standard errors, log lines.

- **Evidence reviewed:** logs, checkpoints, evals, plots, simulations
- **Best direct and surrogate-best checkpoints, and any ranking inversion**
- **Long-run result:** absent, still running, failed, or incorporated
- **Initial/pretrained:** improved, matched, or degraded
- **User directives honored:** confirmed, violated, or unclear
- **Skill-default substitutions:** preserved or reconsidered with evidence
- **Revisit workspace:** path, run name, script path, and the confirmed source
  directory copied into `code/revisit_N/`
- **Main diagnosis:** one plain sentence
- **Ranked next-run prescription:** 3-5 concrete changes
- **Do not change next time:** working pieces to preserve

Checklist:

- [ ] Directive adherence audited in `memory.md`; substitutions preserved
- [ ] Every inventoried plot described, trajectories read, comparators named and
      gaps flagged; `long_train.sh` outputs checked
- [ ] Best/latest/initial/pretrained compared separately, with direct objective
      simulations and failure rates
- [ ] Eval progression classified; noise compared with the learned-vs-benchmark
      gap; batch size and parameter diversity judged against variance
- [ ] Per-update rollout time checked for safe parallel/vectorized collection
- [ ] Shaping, clipping/squashing, discounting, terminal routing audited
- [ ] Early-vs-late action noise, observability, controllability, sensitivity
- [ ] Benchmark/oracle assumptions verified or labeled heuristic
- [ ] Old checkpoints for diagnosis only; fresh revisit init specified
- [ ] User explicitly confirmed the source copied into `code/revisit_N/`
- [ ] Fresh `code/revisit_N/` workspace and `revisit_train.sh` prepared
- [ ] Every recommendation plain-first -- headline, what is wrong, fix, what
      stays the same -- with a compact evidence citation
- [ ] Postmortem recorded; technical detail offered, not dumped
