---
name: 03-parameter-exploration
description: Propose problem parameter ranges for domain randomization
metadata:
  sources:
    - rl-environment-design (Domain Randomization)
---

# Step 3: Parameter Exploration

Identify which domain/problem parameters to randomize during training and
their ranges. ML/RL hyperparameters are handled in a later step.

## Problem Parameters (ASK THE USER)

These are the domain-specific parameters that affect the optimal policy.
Propose ranges and ask the user for confirmation. Examples:

- Market dynamics: drift (mu), volatility (sigma), interest rate (r)
- Preferences: risk aversion (gamma), discount rate, utility constants
- Initial conditions: starting wealth, portfolio, time horizon
- Constraints: position limits, transaction cost rates

### What to Ask the User

- Which problem parameters should be varied during training?
- What are realistic ranges for each parameter?
- Are there parameters that should be held fixed vs randomized?
- Are there known relationships between parameters that constrain the ranges?

### Sampling by Derived Optimal Action

When the optimal action is a known function of parameters, sample the optimal
action uniformly and derive the free parameter. This ensures balanced coverage
of the action space during training.

Example: If `a*=f(theta)` is known, sample `a*` uniformly, solve for
`theta`, and reject infeasible draws.

## Signal-to-Noise Feasibility Check

Before fixing ranges, check that reward-bearing events actually occur within
an episode at representative parameter values. Use analytical relations where
available to estimate expected events/episode and signal magnitude vs
environment noise:

- Expected number of signal-carrying events per episode (trades, failures,
  transitions) — should be >= 1 for most sampled parameters, not just the
  easiest ones.
- Ratio of expected reward variation from action-choice to unavoidable
  environment noise over one episode. Multiplied by sqrt(N_episodes *
  N_updates) this gives an effective SNR; aim for >= 1.

If a proposed parameter range puts the SNR below this threshold over most
of its support, narrow the range, extend the episode horizon, or plan a
variance-reduction technique (see Step 7) before training.

Add the SNR estimate to the step summary so Step 7 can react.

**Check the edges of each range, not just the middle** — signal collapses
first at the hardest parameter values; narrow the range if SNR is marginal there.

## Checklist

- [ ] Problem parameters to randomize identified
- [ ] Ranges for each parameter confirmed with user
- [ ] Fixed vs randomized parameters agreed upon
- [ ] SNR / events-per-episode checked at representative parameters
