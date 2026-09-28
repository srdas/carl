---
name: 08-initial-testing
description: Rule-based heuristic, supervised pre-training, and environment validation
metadata:
  sources:
    - rl-testing (Environment Tests, Agent Tests, Policy Quality Tests)
    - rl-environment-design (Dry Run)
    - rl-training (Supervised Pre-training, Progressive Complexity stages 0-2)
    - rl-problem-workflow (Stage 2, Stage 3a)
---

# Step 8: Initial Testing

## Honor Prior-Step Directives (MANDATORY, FIRST)

Before any code or training, read `memory.md` and scan each
`step_0N_conversation.md` for user-authored instructions
("**User (edited and sent to CARL):**" blocks). Emit a brief
"Prior-Decisions Contract" preserving exact user DO and DO-NOT wording.

Treat this as a binding contract: user directives override defaults. For every
conflict between a prior user directive and a default recipe below,
follow the directive and record the substitution in `memory.md`.
If a prior file is unavailable, state that and continue; do not pause.

**Time limit:** Check the Step 7 decisions for a user-specified time limit for
this step. If one was given, plan your work to finish within that budget
(e.g., keep pre-training short, skip non-essential ablations). If no limit was
specified, proceed at your own pace.

Propose a simple rule-based heuristic or analytical solution that computes the
actions using the state variables. Use this heuristic for a short supervised
pretraining phase.

## Dry Run: Validate Before RL (MANDATORY)

**Before implementing any agent**, run these checks:

```python
# 1. Random policy — establishes ranges
rewards, states, actions = [], [], []
for _ in range(500):
    state, _ = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        state, r, term, trunc, _ = env.step(action)
        rewards.append(r); states.append(state); actions.append(action)
        done = term or trunc
# Check: all finite? Magnitudes reasonable (not in thousands)?
# Check: state/action space adequately explored?

# 2. Simple heuristic policy — establishes a better-than-random baseline
# Use a simple rule (e.g. greedy 1-step lookahead, fixed strategy, domain
# common sense). This does NOT need to be optimal — just clearly better
# than random.
```

If rewards span >10x range, plan for Huber loss or tanh bounding.
If states have wildly different scales, add observation normalization.

**IMPORTANT — baseline policy time limits:**
- Baseline and heuristic policies must run in **under 60 seconds**.
- Do NOT attempt brute-force search, exhaustive enumeration, dynamic
  programming over continuous state spaces, or any exact solver.
- The purpose of baselines is to **sanity-check the environment** and
  establish a rough performance range, not to solve the problem optimally.
  Solving the problem optimally is the RL agent's job.
- A simple greedy 1-step lookahead or hand-coded rule is sufficient.
  If it takes more than a minute, simplify it.

### Dry Run Checklist

- [ ] `reset()` and `step()` return correct types and shapes
- [ ] Deterministic with fixed seed
- [ ] **Random policy run (500+ episodes):**
  - [ ] All rewards finite (no NaN/inf)
  - [ ] Reward magnitudes reasonable (not thousands)
  - [ ] State magnitudes reasonable
  - [ ] Action space adequately sampled
  - [ ] Mean/std/min/max rewards recorded
- [ ] **Heuristic policy run (must complete in <60 seconds):**
  - [ ] Beats random policy clearly
  - [ ] Reward distribution is reasonable (no NaN/inf, sensible range)
- [ ] Info dict exposes all reward components
- [ ] Split-step interface works (if multi-agent)

### Reward-Decomposition Smoke Tests (MANDATORY)

Before full training, run short deterministic episodes and inspect a per-step
trace: state, action/sub-action, reward components, terminal/failure terms,
per-agent buffer entries, and scalar objective. Cover normal terminal,
failure/bankruptcy, threshold/floor just above/below cutoff, shaping on/off
when present, and one pathological early-termination/floor/local-reward policy.
Assert hand-computed objective matches env output, terminal/failure rewards
route to intended buffers, shifted rewards are additive, and pathological
returns are worse than the reasonable heuristic unless intended.

## Supervised Pre-training + Critic Warmup

Generate target actions from a reasonable rule-based heuristic. The heuristic
need not be optimal — just clearly better than random. The RL agent refines
the policy afterward. Do NOT spend time computing exact or brute-force
solutions for pre-training targets.

Use a **3-phase** approach:

1. **Supervised pre-training**: generate random (obs, target_action) pairs,
   train actor's mean output with MSE loss. Use a **separate actor-only optimizer**
   (`Adam(actor.parameters(), lr=1e-3)`) — do NOT use the full model optimizer,
   and do NOT pre-train the critic.
2. **Critic warmup**: freeze actor, train critic only for 20-50 updates. The
   critic learns accurate value estimates from the pre-trained policy's returns
   without disturbing the actor. Use `critic_only=True` in the update method.
   **Warning:** If actor and critic share a trunk, critic warmup will perturb the
   actor's outputs. Use fully separate networks when pre-training the actor.
3. **Conservative PPO**: unfreeze actor, use small clip (0.1), low entropy
   (0.005), low LR (1e-4). The critic now provides meaningful advantages.

**Why this order matters:** Without critic warmup, an untrained critic produces
random advantages that push the actor away from its good initialization.

If states, actions, or rewards have magnitudes in the thousands or higher,
normalize them before proceeding.

**Pass criteria:** Actor achieves low per-action MSE on held-out samples from
the chosen target policy (analytical or heuristic).

## Progressive Complexity (Early Stages)

| Stage | What | Validates |
|-------|------|-----------|
| 0. Random policy | Run 500+ episodes | Environment sanity, reward ranges |
| 1. Rule-based heuristic | Feed simple heuristic actions | Clearly better than random |
| 2. Simple value-based | DQN for discrete; simple actor-critic for continuous | Learning works at all |

**At each stage, verify:**
- Agent improves over random policy
- Eval reward improves or holds steady
- No action dimension stuck at constant or boundary
**Only proceed when the current stage is confirmed working.**

## Environment Tests

Write pytest tests for:
- `reset()` returns valid observation in `observation_space` and info dict
- `step()` returns valid 5-tuple with correct types
- Determinism: same seed -> same trajectory
- Action space: sampled actions are valid
- Observation space: observations stay within bounds over many steps
- Episode termination: episodes end within expected step count
- Reward range: all rewards finite, within expected magnitude
- **Random policy baseline:** mean/std/min/max rewards recorded

## Agent Tests

- Agent selects valid actions
- Update runs without error, loss is finite
- Save/load preserves behavior (same state -> same action)
- Gradients flow through network (non-zero after backward pass)
- Short training loop completes without errors

## Policy Quality Tests

Run these during training (every 500-1000 episodes) AND after training:

- **Benchmark comparison:** per-dimension MAE and correlation against analytical solution
- **Not constant:** action std across diverse states > 1e-4
- **Not boundary-stuck:** action range > 1e-3
- **Range utilization:** each action dimension uses >10% of its available range

## Testing Best Practices

- Use pytest with multiple random seeds
- Test boundary/edge cases
- Include regression tests against known-good baselines
- When tests fail: check with simple/known policies first, look for NaN/inf

## Initial Testing Checklist

- [ ] Random policy dry run completed — reward ranges established
- [ ] Heuristic policy validated — beats random, runs in <60 seconds
- [ ] Supervised pre-training completed (if analytical/heuristic targets available)
- [ ] Critic warmup completed (20-50 updates with frozen actor)
- [ ] Pre-trained actor achieves low MSE on held-out target samples
- [ ] Simple value-based agent tried (learning confirmed)
- [ ] Environment pytest tests passing
- [ ] Agent pytest tests passing
- [ ] Reward trace covers normal, failure, floor/threshold, shaping, and gaming paths
- [ ] Scalar objective matches hand computation on normal and failure paths
- [ ] Per-agent buffers route terminal/failure terms and shifted rewards additively
- [ ] Initial policy forward pass reports action means/ranges in actual units
