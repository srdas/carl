---
name: 04-information-proposal
description: Propose and refine state variables for the RL environment
metadata:
  sources:
    - rl-environment-design (State Space Design, Feature Engineering, Normalization)
    - rl-problem-workflow (Stage 1a)
---

# Step 4: Information Proposal (State Variables)

Propose a set of variables to be used as the state; ask for edits, additions,
combinations. Ask the user if they know whether any state variables are better
represented in a different functional form (e.g. inverse, squared, exponential).

## State Space Design

**Prefer the simplest state that satisfies the Markov property.**

Key principles:
- Use normalized, scale-invariant features (ratios, not raw values)
- No linear dependencies between features
- No analytical answers as inputs — provide primitive parameters and derived parameters (such as inverse quantities or powers, depending on physical relationships) and let the network learn
- Raw ratios preferred over logarithms (unless variable spans orders of magnitude)
- Include task feasibility features if the problem has a goal (min/max achievable growth)
- Include preference parameters in observation when they vary via domain randomization

## Feature Engineering: Encode Physical Relationships

If analytical formula uses 1/x, include 1/x as a feature. Don't make the network derive it.

Bad example: Provide only raw parameters even when the policy depends on
reciprocal or multiplicative structure that the network must reconstruct.
Good example: Provide transformed features that align with known structure
(difference terms, reciprocal terms, and justified interaction terms).

Include **product/interaction terms** (e.g., `a * (1-a)`) when the optimal
policy depends on such combinations — MLPs approximate these poorly from
raw inputs alone. Similarly, provide **dimensionless coordinates** that
decouple distinct structural signals (e.g., "where in the zone" vs "how
wide is the zone") when the problem has a natural rescaling.

Anti-pattern: Provide many correlated variants of the same signal
(raw, normalized, ratio, and proxy versions together).
Network sees equivalent paths and often learns unstable or averaged behavior.

Use **physically meaningful values** for formula ingredients.
Avoid naive independent min-max normalization that can distort
multiplicative or reciprocal relationships.

### Feature Engineering Checklist

- [ ] Analytical formula inspected for reciprocals and powers
- [ ] Reciprocals (1/x) included as features, not inferred
- [ ] Redundant feature combinations eliminated
- [ ] Max 2-3 features encode same information

### Benchmark Formula Consistency

Analytical formulas used for validation must list required assumptions and
match the implemented objective: terminal rules, constraints, discounting,
and utility floors/compression. If they differ, label the formula as a
heuristic comparator, not ground truth.

## State Normalization

Use fixed clipping/scaling when ranges are known. Use a `RunningNormalizer`
(Welford online mean/variance) only when ranges are unknown or drifting.

### Avoid Batch Normalization on State Inputs

Use explicit clipping and constant scaling, not batch norm.

Problem with batch norm:
- Statistics depend on batch composition, not true state
- Breaks sensitivity to parameter changes
- Hides inadequate feature engineering (network compensates)
- Training artifact doesn't apply at eval time

Instead: Clip inputs explicitly and scale by fixed constants.

```python
# Bad
obs = torch.nn.BatchNorm1d(9)(obs)  # Data-dependent

# Good
obs = torch.clamp(obs, min_clip, max_clip)
obs = obs * feature_scales  # Fixed per-feature multipliers
```

If online normalization needed, use Welford (running mean/std), not batch stats.

## State Engineering Checklist

- [ ] All decision-relevant information included
- [ ] No redundant or linearly dependent features
- [ ] No analytical answers as state features
- [ ] Features normalized to similar scales
- [ ] Markov property satisfied
- [ ] Ratios used instead of raw values where possible
- [ ] Task feasibility context included (e.g., min/max achievable growth, gap ratio)
- [ ] Preference parameters included when they vary across episodes
- [ ] Observation space properly defined
- [ ] Verified with random policy: all observations finite, reasonable magnitude
- [ ] Validation benchmark assumptions match implemented terminal objective
- [ ] Formula mismatch labeled as heuristic comparator
