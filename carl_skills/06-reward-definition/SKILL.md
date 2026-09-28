---
name: 06-reward-definition
description: Design the reward function with normalization and domain randomization
metadata:
  sources:
    - rl-environment-design (Reward Function Design, Domain Randomization)
    - rl-problem-workflow (Stage 1c)
---

# Step 6: Reward Definition

Propose a reward structure, with proper normalization, to be used for training.
Ask the user for any hints or nudges that the model could use in early portions
of the training.

## Reward Function Design

**Keep the reward a single, readable expression.**

- Drop fixed constants that don't affect optimal policy
- Use reference-specified reward functions when available (enables benchmarking)
- Expose each reward component in the info dict
- When magnitudes vary wildly, preserve the direct objective first.
  Maximizing `E[log(reward)]` is not the same as maximizing `E[reward]`.
  For positive trajectory rewards/returns, aggregate in log space with
  log-mean-exp: `logsumexp(log R_i) - log(n) = log(mean R_i)`, which
  preserves the `E[R]` argmax while avoiding overflow. Use nonlinear
  compression (`log1p`, `tanh`), hard floors, or clips only as explicit
  surrogate/objective changes. Never hard clip as a neutral numerical fix.
- Anneal step (shaping) rewards over training so final policy optimizes the true objective
- Be explicit about discounting (in reward vs. RL gamma)

## Reward Credit And Configuration

Before implementation, write two terse tables:
- **Reward config:** component, trigger, scale/floor/clip, discount, shaping/anneal, target agent stream, eval inclusion, terminates/truncates.
- **Credit coverage:** action/agent, immediate effect, delayed/terminal effects, reward stream receives, missing credit.

Each agent's reward stream must include every objective term its action can
causally affect. Earlier actions that change terminal/failure outcomes need
those consequences in their return, or a shared/global reward-to-go. Local
rewards are acceptable only when policy-equivalent to the scalar objective.

For shaping, check the action-gradient direction on representative states.
Do not reward a known failure direction unless a stronger downstream penalty
is in the same reward-to-go. Disable shaping in evaluation unless intentional;
plan a no-shaping smoke run when shaping could reinforce failure.

Hard floors/clips are objective changes, not neutral numerical fixes. Record
thresholds, caps, skipped future rewards after truncation, and whether an
early-failure/floor-seeking policy scores worse than a reasonable baseline.
Benchmark/oracle formulas must match the active reward objective or be labeled
heuristic comparators.

For split-step or vectorized collectors, expose component breakdowns and verify
training buffers receive the same terminal/failure components as the scalar
objective. Multiple reward components landing on one index must be accumulated,
not overwritten.

## Domain Randomization

Randomize **all** parameters that affect the optimal policy (market dynamics,
preferences, initial conditions) at `reset()` time, controlled by an options
flag. Randomized parameters must appear in the observation space.

### Sampling by Derived Optimal Action

When the optimal action is a known function of parameters, sample the optimal
action uniformly and derive the free parameter. This ensures balanced coverage
of the action space during training.

Example: If `a*=f(theta)` is known, sample `a*` uniformly, solve for
`theta`, and reject infeasible draws.

### Domain Randomization Checklist

- [ ] Market dynamics and preferences randomized, not just initial conditions
- [ ] Randomized parameters included in observation space
- [ ] Randomization disabled during evaluation
- [ ] Randomized parameters logged in info dict
- [ ] Reward scaling preserves the direct objective under randomization

## Reward Checklist

- [ ] Reward is a single readable expression
- [ ] Reward configuration table completed before implementation
- [ ] Per-agent credit-coverage table completed if actions are split
- [ ] No unnecessary fixed constants
- [ ] Reward components exposed in info dict
- [ ] Reference reward function used when paper provides one
- [ ] Magnitudes checked with random policy (reasonable range, no NaN/inf)
- [ ] Log-mean-exp/log-sum-exp used for positive wide-scale returns; no mean log reward unless intended
- [ ] No one-sided clips when reward sign can change
- [ ] Floors/clips/truncation checked for reward-gaming local optima
- [ ] Discounting choice documented
- [ ] Early training hints/shaping rewards identified (to be annealed later)
- [ ] Shaping direction checked; shaping disabled in eval unless intended
- [ ] Terminal/failure terms routed to all causally responsible agents
- [ ] Pathological early-failure/floor/local-reward policy scores below heuristic baseline
- [ ] Benchmark/oracle formula matches active objective or is labeled heuristic
