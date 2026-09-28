---
name: 05-action-definition
description: Define the action space, distributions, and multi-agent decomposition
metadata:
  sources:
    - rl-environment-design (Action Space Design)
    - rl-agent-implementation (Bounded Actions, Algorithm Selection, Multi-Agent)
    - rl-problem-workflow (Stage 1b)
---

# Step 5: Action Definition

Propose a set of actions. Ask for any changes required. If there are multiple
actions, ask if the actions should be output by one RL model or one action
each by separate agents.

## Action Space Design

- Prefer continuous over discrete when actions have natural ordering
- Verify bounds accommodate the optimal solution with headroom (~1.5x expected range)
- Estimate realistic action magnitudes first; avoid [0,1] if the useful range is a tiny corner
- For sequential actions within a step, split into separate sub-step methods
- Normalize heterogeneous action scales to [-1, 1] with an `ActionWrapper`

## Reasonable Heuristic Behavior

Before bounds, initialization, or warmup, ask for a simple safe heuristic
policy. Use it to sanity-check action magnitudes and compare random/init,
heuristic, learned, and analytical/optimal policies. If random initialization
is unsafe, consider heuristic warmup or a residual actor, clearly labeled as
a baseline, not the optimum.

## Bounded Continuous Actions (Finance)

Finance actions are almost always bounded (portfolio weights, consumption rates).
**Do not clip a Gaussian** — use a distribution with native support matching the bounds.

- **Beta** (for [0,1]): native bounded support, no squashing required
- **Tanh + affine transform**: for [a,b], apply Jacobian correction, then scale
- Scaled Beta/tanh actions must compute log-prob with the affine/Jacobian term

**Action bounds checklist:**
- Compute analytical optimum (if known) and verify it's within bounds
- Leave headroom: ~1.5x expected optimal range
- Check initial deterministic and sampled actions in actual action units
- If learned action sits at boundary during eval, bounds are too tight

### The Clip-then-Log-Prob Bug

Never sample from unbounded Gaussian, clip, then compute log-prob from the
unclipped sample. Use tanh-squashed distribution instead.

## Action Reparameterization

When the optimal action has a known scaling factor (e.g., proportional to some
state-derived quantity), factor it out and let the network output the residual
bounded coefficient. This removes power-law or scale dependence from the
learning target, leaving the network with a smoother, O(1) function to learn.

## Algorithm Selection

| Problem Type | Start With | Upgrade To |
|-------------|------------|------------|
| Discrete actions | DQN | Double DQN, Dueling DQN |
| Continuous actions, simple | Simple actor-critic | PPO |
| Continuous actions, complex | PPO | SAC, TD3 |
| Sequential multi-action | Separate simple agents first | PPO per agent |

**Always verify learning works with a simpler algorithm before jumping to PPO.**

## Multi-Agent Decomposition

When actions occur in sequence within a step, use **separate agents** each
with its own network, buffer, and optimizer. Each agent outputs a 1D action.

Sequential sub-step agents are not automatically safe with local rewards:
earlier agents can control later states and terminal outcomes.

**Causal credit assignment:** The second agent should receive **shifted rewards**
— the next step's component that its action causally influenced, not the current
step's full reward. Expose reward components in the info dict. Before
implementation, record whether rewards are shared, local, or local+downstream;
how terminal/failure terms are routed; and whether shifted buffers add to
existing reward terms instead of overwriting them. Every agent whose action
can cause terminal/failure consequences must receive them.

**Per-agent hyperparameters:** Different agents can need different entropy_coeff,
log_std_min, and learning_rate. If one agent converges while the other collapses,
raise entropy_coeff 5-10x for the struggling agent.

## Action Definition Checklist

- [ ] Action bounds defined for each dimension
- [ ] Bounds accommodate optimal solution with ~1.5x headroom
- [ ] Reasonable heuristic policy captured and compared against random/init
- [ ] Bounds checked against heuristic/expected magnitudes
- [ ] Distribution type chosen matching bounds (Beta or tanh-squashed)
- [ ] Initial deterministic and sampled actions checked in actual units
- [ ] Scaled bounded distributions include correct affine/Jacobian log-prob handling
- [ ] No clip-then-log-prob pattern
- [ ] Algorithm selected (simplest viable first)
- [ ] Multi-agent split planned if actions are sequential
- [ ] Causal credit assignment planned for sequential agents
- [ ] Terminal/failure reward routing defined for sequential agents
- [ ] Shifted rewards are additive, not overwriting existing components
- [ ] Per-agent hyperparameters considered if multi-agent
