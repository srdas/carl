---
name: 09-launch
description: Write the full RL repository and execute all training phases
metadata:
  sources:
    - rl-repository-setup (Directory Structure, Requirements)
    - rl-agent-implementation (Agent Structure, Networks, Entropy Collapse)
    - rl-training (Training Loop, Vectorized Collection, Intermediate Evaluation, Output Flushing)
    - rl-model-management (Saving, Loading, Versioning)
    - rl-visualization (Metrics, Plots, TensorBoard)
    - rl-problem-workflow (Stages 3-5)
---

# Step 9: Launch

## Honor Prior-Step Directives (MANDATORY, FIRST)

Before any code or training, read `memory.md` and scan each
`step_0N_conversation.md` for user-authored instructions
("**User (edited and sent to CARL):**" blocks). Emit a brief
"Prior-Decisions Contract" preserving exact user DO and DO-NOT wording -- this
is a binding contract, so wherever a directive conflicts with a default recipe
below, follow the directive and record the substitution in `memory.md`
(covering Step 8 artifacts too: do not load or build on them if a directive
would have forbidden their creation). If a prior file is unavailable, say so
and continue; do not pause.

**Time limit:** check the Step 7 decisions for a user-specified time limit and,
if given, plan training duration and evaluation scope to finish within it;
otherwise proceed at your own pace. Then write the entire RL repository and
proceed through the training phases (actor/critic warmup, then RL).

## Directory Structure

```
project_root/
├── environment.py       # Gymnasium environment
├── agent.py             # RL agent(s)
├── models.py            # Network architectures
├── train.py             # Training script
├── evaluate.py          # Evaluation script
├── config.py            # Hyperparameters
├── utils.py             # Helpers
├── requirements.txt
├── experiments/
│   ├── checkpoints/     # best_model.pt, latest_model.pt, epoch_*.pt
│   ├── saved_models/    # v1.0/model.pt + config.json + training_stats.json
│   └── logs/            # TensorBoard data
└── tests/
```

## Agent Structure

Implement four methods: `select_action(state, training)`, `update(batch)`,
`save(path)`, `load(path)`. Use Step 7 architecture; fallback [128, 128].

For supervised pre-training workflows, also implement:
- `freeze_actor()` / `unfreeze_actor()` — toggle `requires_grad` on actor params
- `update(batch, critic_only=True)` — skip policy/entropy loss, train value head only

### Activations
**ReLU** on all hidden layers including the pre-final one. Output layer: **Tanh**
for a bounded policy mean, **Softplus** for scale/variance, none if unbounded.

## Training Phases

### Phase 1: Supervised Pre-training (Actor Only)
- Generate random (obs, target_action) pairs from analytical/heuristic policy
- Train actor mean with MSE loss using separate Adam optimizer (lr=1e-3)
- Do NOT pre-train the critic

### Phase 2: Critic Warmup
- Freeze actor
- Train critic only for 20-50 updates using `critic_only=True`
- Critic learns value estimates from pre-trained policy's returns

When there is no supervised pre-training, still run **20–50 value-only PPO
updates** before enabling the policy gradient (`--critic-warmup N`). Tag these
updates in the log (e.g., a `V` suffix) so the warmup phase is visible.
The policy gradient is flat over a cold critic; skipping warmup produces noisy
advantages that can trap entropy at its initial value.

### Phase 3: Conservative PPO
- Unfreeze actor
- Small clip (0.1), low entropy (0.005), low LR (1e-4)

### Phase 4: Full PPO Training
- Standard hyperparameters from config.py
- Batched collection for on-policy methods (never single-episode updates)
- **Ramp LR, clip, entropy** from Phase 3 values (1e-4 / 0.1 / 0.005) toward
  config targets one knob at a time every 50-100 updates. Checkpoint
  (`pre_ramp_<step>.pt`) before each ramp. If canonical eval score drops >10%, action
  std <0.01, or range utilization <10% within 50 updates, reload the
  checkpoint and re-launch at 0.5x the last-safe value. Two rollbacks at the
  same level = freeze that knob there.

## Training Loop Details

For on-policy (PPO, A2C, REINFORCE): **collect a batch of multiple episodes
before each update** — never update after a single short episode.

### Vectorized Collection

When `steps_per_episode > 25` and `episodes_per_update > 25`, vectorize: run
episodes in parallel per time step. ~3-4x speedup.

### Reward-Buffer Regression Tests

Carry Step 8 reward traces into tests before PPO. Vectorized/split-step
collectors must prove buffer writes for normal next-step reward, terminal
final-step reward, failure/bankruptcy reward, and same-index multiple
components. Accumulate reward components additively; never overwrite shifted
or terminal/failure entries.

### Output Buffer Flushing

Flush stdout/stderr periodically (every 10-100 episodes, after updates, after
benchmark checks) so progress is trackable in real time.

## Progressive Training Stages

| Stage | What | Validates |
|-------|------|-----------|
| 3a. Simple baseline | Value-based or supervised pre-training | Learning works |
| 3b. Single-Agent PPO | Minimal state, full action | Core learning with real inputs |
| 3c. Multi-agent split | Separate agents, same reward | Decomposition is sound |
| 3d. Causal rewards | Shifted rewards for 2nd agent | Credit assignment correct |
| 3e. Variance reduction | CRN block + block-centered advantages + Huber + larger batch | Production configuration |

**At each stage, verify:**
- Agent improves over random policy
- Canonical eval score improves or holds steady
- No action dimension stuck at constant or boundary
**Only proceed when the current stage is confirmed working.**

## Intermediate Policy Evaluation (Degeneracy Detection)

Evaluate on a **fixed, held-out set of parameter sets** (50+) covering the full
training range. Reuse the **exact same set** every evaluation so progression is
comparable run-to-run; never refresh it or overlap it with training seeds. Base
early stopping and canonical `best` checkpoints on the held-out full-path direct
objective or a positive affine transform of it. Shaped/compressed rewards are
surrogate diagnostics unless proven ordinal-equivalent at the trajectory level.

Always evaluate the policy in its **deterministic avatar** (actor mean/mode,
no sampling or exploration noise).

Run every 500-1000 episodes during training. Check:
- **Action variance across states** — near zero means degenerate
- **Benchmark correlation** — track for diagnostics, but declining correlation
  is acceptable if canonical eval score is stable or improving (the agent may find a
  better policy that differs from the heuristic)
- **Action range utilization** — <10% means exploration collapsed
- **Parameter-scaling correlation** — when an analytical scaling is known,
  fit the learned slope in log-log against parameter values. A flat slope
  despite stable canonical eval score indicates the policy has learned the mean
  response but missed the parameter dependence — typically a variance /
  CRN problem, not a capacity problem.

If action std across a diverse state batch drops below ~0.01, the policy
is degenerate; probe per-feature sensitivity by sweeping each feature
min→max with the others held fixed.

## Model Management

### What to Save
**Essential:** model state_dict, optimizer state_dict, episode number, metrics.
**Recommended:** random state, LR scheduler state.

**Required contract (`extra` dict):** every checkpoint must be saved with an
`extra` dict carrying `obs_dim`, `action_dim` (and bounds), a snapshot of the
resolved config, and the selection-metric name + value. `PPOAgent.save` already
accepts `extra: dict | None` -- use it; do not invent a second mechanism. This
is `make_artifact.py`'s source of truth. Never hard-code `obs_dim` (or any
other dimension) as a constructor default: derive it from `config` so it cannot
drift from the environment.

### When to Save
- **Best model:** when the canonical selection score improves. Default
  selection score is the raw/full-path direct objective. Surrogate-selected
  checkpoints, if saved, must be named separately, e.g. `best_surrogate`.
- **Latest model:** every N episodes (for resume)
- **After supervised pre-training:** save before PPO starts (fallback if PPO degrades)
- **Periodic:** every 1000-5000 episodes for long training

## Visualization

### Key Metrics to Track
**Primary:** full-path direct objective and canonical eval curve. If training
uses a reward surrogate, also log it separately. Prefer training rewards that
are a positive affine transform of the direct objective; use nonlinear
compression/clipping only when needed for stability.
**Secondary:** entropy/exploration rate, gradient norms, action distributions,
benchmark MAE and correlation.

### Essential Plots
1. **Direct-objective curve:** raw + 100-episode moving average
2. **Loss curve:** raw + smoothed
3. **Per-step policy plot:** mean +/- 1 std over 1000+ episodes vs analytical
   - Include MAE and correlation in subplot titles
4. **Action heatmap:** action selection across state space
5. **Value function surface:** if using a critic
6. **Pi scatter plot:** learned action mean vs analytical optimal across diverse parameter sets
7. **Sensitivity analysis:** vary one parameter at a time, plot learned vs analytical

## Final Evaluation

Report these in plain language -- what improved, what didn't, why it matters
for shipping -- with a compact citation, not raw MAE/correlation prose. Exact
numbers go in the step summary and `memory.md`.

1. **Benchmark comparison:** per-dimension MAE and correlation on state grid
2. **Oracle comparison (if available):** trained agent + oracle action for one dimension
3. **Distribution analysis:** reward/wealth histograms, trained vs oracle
4. **Best, latest, and any surrogate-best checkpoints:** evaluate all on the
   same held-out direct objective. If surrogate-best differs from direct-best,
   treat surrogate-best as diagnostic.

5. **Direct policy-performance simulation:** report realized objective and
   failure/truncation rates, not only policy-geometry plots.

### Decision: Iterate or Ship

| Observation | Action |
|-------------|--------|
| Canonical eval score still improving | Train longer |
| Canonical eval score plateaued, acceptable | Ship |
| Canonical eval score plateaued, too low | Revisit hyperparams, state, or reward |
| Agent stuck at boundary | Widen action bounds |
| Metrics oscillate wildly | Increase batch size; verify CRN + block-centering are active |

State the call in plain terms ("still improving, keep training" / "not beating
the simple benchmark, reward or state needs another pass") -- the table is
your lookup, not the message.

**Early stopping rule:** track the best canonical selection score and stop after
patience=10-20 evals with no improvement, using the best checkpoint. Use
automated patience, not visual inspection, so compute is not wasted on plateaus.

## Common Issues

| Problem | Root Cause Check | Fix |
|---------|------------------|-----|
| No learning | Reward function; exploration level | Verify state ranges; increase entropy_coeff |
| Entropy constant from first update | `log_std` saturated at `LOG_STD_MIN` (zero gradient) | Replace learnable `log_std` with a linear decay schedule; or raise the floor (e.g. -1.5 vs -3.0) and increase entropy_coeff |
| Degenerate constant-output policy | Feature redundancy; domain randomization speed | Check feature sensitivity; use fixed-parameter episodes |
| Pre-trained actor degrades during PPO | Untrained critic produces noisy advantages | Freeze actor, warmup critic for 20-50 updates |
| Entropy stuck at init value; policy gradient flat for first N updates | Critic not warmed up before policy gradient starts | Add `--critic-warmup N` (20–50 value-only updates); works even without supervised pre-training |
| Value loss in hundreds/thousands | Return outliers; reward scaling | Switch to Huber/smooth_l1 loss |
| Entropy collapses despite raised floor | Entropy coefficient annealing | Tune per-update annealing schedule |
| Reward signal buried in env noise | Stochastic environment dominates gradient | Enable CRN across the block (shared noise path) and use block-centered advantages; fall back to a relative reward only if env noise cannot be shared across twins |
| Policy insensitive to a known parameter | Reward gradient across that parameter is below env-noise floor | Add multi-parameter CRN block (sweep that parameter across twins with shared noise) |
| Eval return plateaus early; geometry metrics still drifting | Shared-path noise dominates gradient | Enable CRN + block-centered advantages before increasing batch size |
| <1 reward-bearing event per episode at most parameters | Episode too short or parameter range too wide | Narrow range via Step 3 SNR check OR extend horizon OR use CRN to recover signal |
| Failure penalties do not affect learning | Reward shift overwrites terminal/failure entries or routes them to only one agent | Add reward-buffer tests; accumulate additively; route terminal/failure terms to all causally responsible agents |
| Policy improves by failing early | Truncation skips future negative rewards or penalty is too small | Score early-failure policies in smoke tests; increase penalty or continue absorbing penalties through original horizon |
| Collection loop slow | Episode parallelization | Vectorize episode collection |
| Policy/value saturated at init | Output activation + default weight init produce degenerate mean or std from step 0 | Run forward pass pre-training; adjust output-layer bias/scale or init scheme |

## Long-Run Script (`long_train.sh`)

In-session training usually stops well before convergence, so leave a
`long_train.sh` beside `train.py` / `evaluate.py` for a longer run after CARL
exits. **Write it, `chmod +x` it, but do NOT execute it.**

Model this on `sessions/NTZ_20260423_direct_RL/long_train/run.sh`:

- `#!/usr/bin/env bash` + `set -euo pipefail`
- `cd` into the script's own directory via `SCRIPT_DIR`
- Env-var overrides with defaults for every knob the user might tune:
  `UPDATES`, `SEED`, `DEVICE`, `RUN_NAME`, `PATIENCE`, `EVAL_EVERY`,
  `EVAL_EPS`, `SAVE_EVERY`, `INIT_FROM` (see parity section below)
- Banner that echoes the resolved config
- `mkdir -p experiments/${RUN_NAME}`, then
  `python train.py <flags> 2>&1 | tee experiments/${RUN_NAME}/train.log`
- Followed by `python evaluate.py 2>&1 | tee -a experiments/${RUN_NAME}/train.log`
- Ending with `python make_artifact.py --run-name "$RUN_NAME"` so the long run
  refreshes `artifact/` from its own (better) checkpoint
- Header comment with one-line usage (`bash long_train.sh`) and the full
  env-var override list

### Defaults

- `UPDATES`: 5-10x in-session; `PATIENCE`: >=2x in-session
- `EVAL_EVERY` / `EVAL_EPS` / `SAVE_EVERY`: sized so eval noise stays low
  across the longer horizon
- `RUN_NAME`: distinct from the in-session run (e.g. `<project>_long`) so the
  long run gets its own `experiments/` dir and never overwrites in-session
  artifacts

Thread through whatever flags your `train.py` actually accepts -- at minimum
`--updates`, `--seed`, `--run-name`, and any early-stopping / eval-frequency
flags.

### Initial-model parity (critical)

The long run MUST start from the same initial model as CARL's in-session PPO
training, or it is not the same experiment on a longer budget and can't compare.

- After any supervised pre-training, save the pretrained actor (and whatever
  else `agent.load()` expects) to `experiments/pretrain/init_actor.pt` -- the
  single source of truth for "the initial model."
- Add `--init-from <path>` to `train.py`: after agent construction and
  before the PPO loop, if the flag is set, call `agent.load(path)`. If the
  flag is set but the file is missing, error out -- do not silently train
  from scratch.
- CARL's own in-session PPO launch must pass
  `--init-from experiments/pretrain/init_actor.pt` whenever that file was
  produced.
- `long_train.sh` exposes an `INIT_FROM` env var defaulting to the same
  path (resolve via `$SCRIPT_DIR` so the script works when invoked from
  elsewhere) and threads it through as `--init-from "$INIT_FROM"`.
- If there was no supervised pre-training, omit `--init-from` in both the
  in-session call and `long_train.sh`; parity is then guaranteed by the
  shared `--seed` plus identical agent construction code.
- Do NOT copy the final post-PPO checkpoint into the long run. The goal is
  to reproduce CARL's starting conditions on a longer budget, not to
  resume mid-training.

## Final Usable Artifact (MANDATORY)

A finished run must ship something a stranger can use to solve a **new problem
instance**, not just replay a fixed eval grid. Build a self-contained
`artifact/` bundle in the code dir: it must still work after being copied
elsewhere, so the checkpoint is **copied** rather than referenced, and nothing
in it may import from the training tree.

`make_artifact.py` (next to `train.py`), invoked as
`python make_artifact.py --run-name <run> [--ckpt PATH] [--out artifact/]`:

- Resolve the best checkpoint from `--ckpt` or the run's selection metric.
  **Never hard-code a run name or checkpoint path** -- a stale default pointing
  at a run that was never written is a bug that has shipped before.
- Copy it to `artifact/best_model.pt`.
- Write `artifact/contract.json`: `obs_dim`, `action_dim` + bounds, observation
  field order and units, action semantics, normalisation constants, resolved
  `config.py` values, a code fingerprint (sha256 of each copied `.py`), the
  selection metric name + value, the seed, one canonical instance, and the
  instance-parameter schema (name -> type, range, default).
- Copy the runtime modules it needs (`agent.py`, `environment.py`, `config.py`,
  model definition) into `artifact/`.

Then, inside `artifact/`:

- **`solve.py`** -- the stranger's entry point. Interface:
  `--instance instance.json`, repeatable `--param k=v`, `--out result.json`,
  `--n-seeds`. Load `contract.json`, validate the instance against the parameter
  schema (fail loudly on unknown or out-of-range keys -- never silently clip),
  build the policy **from the contract** not from module defaults, run the
  deterministic avatar, then print and write the decision trajectory (per-step
  state, action, reward) plus the objective value. Move `evaluate.py`'s existing
  `deterministic_action(dist)` into the artifact; do not reimplement it.
- **`smoke_test.py`** -- load the checkpoint, run the canonical instance from
  `contract.json`, assert the objective reproduces the recorded metric within
  tolerance. Run it in-session with `run_command`; a failure blocks shipping.
- **`MODEL_CARD.md`** -- problem statement, MDP definition, training recipe and
  budget, best-vs-latest-vs-baseline numbers, validity domain (parameter ranges
  actually trained on + an explicit out-of-domain warning), known failure modes,
  and a copy-paste usage example.

## Launch Checklist

- [ ] Repository structure created; environment validated (dry run passed)
- [ ] Agent implemented with all four methods + freeze/unfreeze
- [ ] Training phases completed: supervised pre-training, critic warmup, conservative PPO, full PPO
- [ ] Reward-buffer regression tests and sample-episode reward traces pass before first RL update
- [ ] Intermediate evaluations running every 500-1000 episodes, with failure/truncation rates logged next to return
- [ ] Best model saved on canonical selection-score improvement; TensorBoard active; all essential plots generated
- [ ] Final evaluation reported in plain language; exact metrics recorded in the step summary
- [ ] `long_train.sh` written, `chmod +x`'d, and parity with in-session
      initial model verified (shared `--init-from` path when pre-training
      was used; shared `--seed` otherwise)
- [ ] `artifact/` bundle generated by `make_artifact.py` (checkpoint copied,
      `contract.json` written, runtime modules included)
- [ ] `artifact/smoke_test.py` executed in-session and passing
- [ ] `artifact/MODEL_CARD.md` records the validity domain and the
      best-vs-latest-vs-baseline comparison
- [ ] Decision made: iterate or ship
