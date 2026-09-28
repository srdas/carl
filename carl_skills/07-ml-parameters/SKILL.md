---
name: 07-ml-parameters
description: Present auto-selected ML/RL hyperparameters for user approval (one-turn step)
metadata:
  sources:
    - rl-training (Configuration, Hyperparameters, Parameter Diversity, Parameter Stability, Entropy Annealing, Loss Functions)
---

# Step 7: ML/RL Parameters

This is a **one-turn step** -- no later turn to simplify on, so open plain.
Before the table, give 2-3 sentences on what this configuration means for
their problem: what controls training-signal noise, early exploration, and
whether network size matches the problem's complexity. Show the table as
supporting detail and ask for approval -- mirror the plain phrasing already
used below for the time-limit question, not a config-key recap.

Select all of the following automatically based on the problem structure
established in previous steps. Do not ask the user for input on any of these
unless the user volunteers changes.

## Training Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| episodes_per_update | 250 | ~5000 transitions for 20-step episodes |
| learning_rate | 3e-4 to 5e-4 | |
| ppo_epochs | 4 | Standard PPO |
| mini_batch_size | 512 | Scale with batch: 256-2048. Avoid 64 with large batches (very slow). |
| clip_epsilon | 0.2 | Standard PPO |
| entropy_coeff | 0.01 | Increase if policy collapses |
| gae_lambda | 0.95 | |
| value_loss | smooth_l1 | Huber for finance |

## Batch Size Selection

Set `episodes_per_update` so total transitions >= 10-20x mini-batch size.

| Transitions/update | Quality | Use case |
|-------------------|---------|----------|
| < 500 | Degenerate | Never for PPO |
| 500-1000 | Workable | Simple tasks |
| 1000-5000 | Good | Finance with moderate noise |
| 5000-10000 | Recommended | High-variance environments |

When in doubt, err large — stable gradients beat faster iteration.
`long_train.sh` can cover undertraining after CARL exits, but cannot
rescue a run destabilized by too-small batches.

## Parameter Diversity Per Update

Each PPO update must see **many distinct parameter sets**. Choose sets x
episodes-per-set to match the update batch (e.g., 25 x 10 = 250). Too few sets per update (e.g., 2-3) causes
overfitting to narrow parameter regimes.

## Parameter Stability in Domain Randomization

Fix parameters for 40-200 episodes before sampling new ones.
Randomizing every episode prevents learning parameter dependencies.

Pattern:
- Set EPISODES_PER_PARAMETER_SET = 40 (or problem-dependent)
- Hold parameters fixed for that many episodes
- Only then sample new parameters
- Within fixed-parameter episodes, still randomize initial conditions and noise

## Variance Reduction: CRN Within Blocks

Episodes within a block already share parameters; share the **same exogenous
noise path** across the block as well (Common Random Numbers). Implementation:

- In the vectorized env, draw one noise vector per timestep per block (not
  per episode) and broadcast across the block.
- Compute advantages as usual, then **subtract the per-block mean at each
  timestep** (block-centered advantages). This cancels shared-path noise
  and leaves only the action-induced component.
- Apply block centering to the **advantage only**; keep value-loss targets
  on the raw return so the critic still learns absolute expected return.
- **Evaluation must use independent paths**, not CRN — otherwise the
  reported metric inherits the reduced spread and misrepresents
  out-of-distribution performance.
- **Antithetic variates (optional, within CRN blocks):** split each block in
  half; second half uses −Z of the first half's noise. Halves effective noise
  variance at zero extra environment cost. Requires even `block_size`; implement
  as an `antithetic=True` flag on the noise-sampling helper. Do not use outside
  of CRN blocks — antithetic pairs must share the same parameters and differ
  only in noise sign.
- **Skip global advantage std normalisation** when block-centering is active
  (`--adv-norm none`). Block-centering already removes the dominant noise
  component; dividing again by global std compresses the remaining signal.

CRN + block-centered advantages is the default variance-reduction technique
for this skill.

### Multi-parameter CRN Blocks

When a specific parameter's effect on the optimal policy is the learning
target, sweep **that parameter across the block** while keeping all other
randomized parameters fixed. With shared noise and shared other-parameters,
the block-centered advantage isolates the target parameter's influence on
reward — often the difference between a flat, parameter-insensitive policy
and one that recovers the correct scaling.

Use when an analytical or empirical scaling law across a parameter is known
or desired; it provides a dense gradient signal where marginal randomization
provides almost none.

## Entropy Coefficient Annealing

Start at 0.01-0.05, anneal to 0.001. Never start above 0.1 — if the entropy
bonus rivals the reward magnitude, the policy maximizes entropy instead of
task reward. Apply per update round, not per mini-batch.

**After supervised pre-training:** start much lower (0.005 -> 0.001). The
pre-trained policy is already good — high entropy destroys it.

## Loss Functions

Use **Huber/smooth_l1** for value loss when returns span >10x range (common
in finance). MSE amplifies outliers quadratically. Never trim outlier episodes
from the batch — this biases the policy gradient.

## Log-Std Initialization & Schedule (Continuous Actions)

Initialize `log_std` so that initial std is ~0.1-0.5x the action range:
`log_std_init = log(action_range / 10)`. Default log_std=0 (std=1.0) is
catastrophic when actions are small (e.g., [0, 0.2]). Make `log_std_init`
configurable in config.py.

Prefer a deterministic `log_std` schedule (linear decay from `log_std_init`
over training) over a learnable `log_std` clamped at `LOG_STD_MIN`. A
learnable std that hits the clamp has zero gradient and freezes exploration;
a schedule never pins against an autograd boundary.

Same failure mode for any squashed/positive output (Tanh mean, Softplus
std, bounded value head): verify the init forward pass on representative
states is non-degenerate — a pinned mean or flat std traps the policy.

## Actor Initialization Diagnostics

Before training, print each action head's deterministic action and sampled
quantiles on representative states in actual action units. Compare them with
heuristic/baseline magnitudes. If initial samples hit failure-prone regions,
adjust bounds, output bias, log_std, native support, or residual/scale
parameterization before tuning PPO.

## Network Architecture

| Complexity | Architecture |
|-----------|--------------|
| Simple | [128, 128] |
| Medium (domain randomization) | [256, 256] |
| Complex | [512, 256] or [256, 256, 256] |

**Residual/heuristic-anchored actor:** When a heuristic baseline exists,
consider `output = heuristic(state) + scale * network(state)` with small
initial `scale`. This preserves baseline quality while allowing RL refinement.
Ensure the heuristic base captures the main state-dependent structure — a
constant or oversimplified base limits what RL can discover within the
correction band.

## Configuration

Centralize all hyperparameters in `config.py`. Set all random seeds for
reproducibility.

## Time Limits for Steps 8 and 9

At the end of your summary, ask the user whether they want to set rough time
limits for the autonomous steps:

- **Step 8 (Initial Testing):** environment validation, dry runs, and
  supervised pre-training. Suggest ~15 minutes as a default.
- **Step 9 (Launch):** full RL training and evaluation. Suggest ~45 minutes
  as a default.

Present it as: "Do you want to set rough time limits for the autonomous steps?
I suggest about **15 minutes** for initial testing (Step 8) and **45 minutes**
for full training (Step 9). You can adjust these or leave them open-ended."

Record the user's chosen limits (or "no limit") in the step summary so that
Steps 8 and 9 can respect them.

The summary must also carry the Step 9 budget as a machine-readable line, on its
own line, integer minutes and nothing after the number:

```
TRAINING_BUDGET_MINUTES: 45
```

The controller parses this to size how often it polls the Step 9 training job;
without it, it falls back to a free-text scan and then to a 45-minute default.
Emit the line even when the user accepted the default or declined a limit (use
the default you suggested), and keep it consistent with the prose.

## Checklist

- [ ] Hyperparameter starting points selected from tables above
- [ ] episodes_per_update ensures sufficient transitions (>= 10-20x mini-batch)
- [ ] Parameter diversity per update planned (many parameter sets per gradient step)
- [ ] Parameter stability schedule defined (episodes per parameter set)
- [ ] Entropy annealing schedule defined
- [ ] Loss function chosen (Huber if returns span >10x)
- [ ] Network architecture sized to problem complexity
- [ ] All hyperparameters centralized in config.py
- [ ] Random seeds set for reproducibility
- [ ] log_std_init set from action range; schedule defined (preferred) or floor raised to avoid clamp saturation
- [ ] Initial forward pass recorded in action units; mean near heuristic magnitude; quantiles avoid failure-prone regions
- [ ] CRN within blocks enabled with block-centered advantages; eval pipeline uses independent paths, not CRN
- [ ] Antithetic variates on for even block_size; global adv-norm disabled when block-centering is active (`--adv-norm none`)
- [ ] Target parameter identified for multi-parameter CRN sweep, if a scaling law is sought
- [ ] Time limits for Steps 8 and 9 recorded (or "no limit")
