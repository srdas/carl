# Model card — Goddard rocket with a hard max-Q constraint

Bundle built by `make_artifact.py` from run **rocket_long**, update
619.
Checkpoint sha256 `03fa09c8ba21c4f7…`

## Problem

Fly a Goddard rocket as high as possible while never exceeding a dynamic-pressure
limit `Q_max`. Thrust is throttleable; propellant is finite; drag and air density
fall off exponentially with altitude. The interesting tension is that the
altitude-optimal ascent wants maximum thrust early, exactly where dynamic
pressure peaks.

## MDP

- **State** (12-D, all dimensionless, clipped to
  [-10.0, 10.0]):
  1. `rho_ratio` — rho(h)/rho0, dimensionless
  2. `v_norm` — v / sqrt(2*Q_max/rho0), dimensionless
  3. `q_ratio` — q / Q_max, dimensionless
  4. `q_headroom` — 1 - q/Q_max, dimensionless
  5. `fuel_frac` — m_fuel / m_fuel_0, dimensionless
  6. `dry_over_m` — m_dry / m, dimensionless
  7. `thrust_weight` — T_max / (m*g), dimensionless
  8. `energy_ratio` — effective-altitude Tsiolkovsky budget / initial budget
  9. `log_q` — ln(Q_max / 30000 Pa)
  10. `log_T` — ln(T_max / 200000 N)
  11. `log_m_fuel` — ln(m_fuel_0 / 1500 kg)
  12. `cd` — drag coefficient, dimensionless
- **Action**: scalar `u` in [0.0, 1.0] — u = throttle fraction of T_max, applied for one DT. The env applies a hard max-Q clamp: u is reduced to env.u_cap_safe() when it would breach Q_max, with a 5% drag penalty on clamped steps.
- **Reward**: `raw_dh` — telescoping altitude
  gain `dh / 1000` per `DT =
  0.1` s. Undiscounted (gamma = 1), so the return is the
  apogee in scaled units.
- **Termination**: apogee (v <= 0 after liftoff), ground contact, or 2000 steps.
- **Constraint handling**: the environment enforces max-Q itself by clamping `u`
  and charging a 5% drag penalty
  on clamped steps, so the constraint cannot be violated by a bad policy — it can
  only be paid for.

## Training recipe

PPO with a Beta-distributed throttle (`beta_mode =
boundary`, so the policy may place mass at u=0 and
u=1), separate [256, 256] tanh MLPs for actor and critic.
Behaviour cloning from a liftoff-then-exponential-decay heuristic, then a
critic-only warmup with the actor frozen, then full PPO. Domain randomisation
over all four instance parameters with rejection sampling on thrust/weight
(>= 1.1).

Training seed 0; evaluation seed 1000 (the eval seed
is what `solve.py` and `smoke_test.py` use, so their numbers are reproducible).
512 episodes per PPO update, CPU-only. This checkpoint is
from update **619** — the update at which the
selection metric last improved, not the end of the run.

## Results — selection metric `eval_apogee_frac_mean`

mean apogee_fraction over the 9-point eval grid (EVAL_M_FUEL_GRID x EVAL_Q_MAX_GRID) x 5 seeds, avatar=mean.

| Policy | eval_apogee_frac_mean |
| --- | --- |
| **This bundle (best)** | **0.0735** |
| Latest checkpoint | 0.0717 |
| Post-BC, pre-PPO | 0.0660 |
| Heuristic baseline | 0.0678 |
| Random init | 0.0630 |

Canonical instance (`{'m_fuel_0': 2000.0, 'T_max': 200000.0, 'cd': 1.5, 'Q_max': 30000.0}`,
seed 1000): apogee
**14479.2 m**, apogee_fraction
0.05884 of the 246065 m vacuum
ballistic ceiling, over 1143 steps.

Read those honestly: the trained policy beats the heuristic on the selection
metric by roughly +0.0057, which is
a small margin, and the random-init baseline is already close behind. The reward
signal here is weak relative to the spread across instances.

## Validity domain

| parameter | type | trained range | canonical |
| --- | --- | --- | --- |
| `m_fuel_0` | float | 300 – 5000 | 2000 |
| `T_max` | float | 100000 – 300000 | 200000 |
| `cd` | float | 1 – 2 | 1.5 |
| `Q_max` | float | 10000 – 50000 | 30000 |

`solve.py` **refuses** parameters outside these ranges rather than clipping them:
the policy has no validity claim there, and silently clipping would return a
confident answer to a question it was never trained on. Pass
`--allow-out-of-domain` to override; the result is then labelled out-of-domain in
both stdout and the result JSON.

Instances must also satisfy `T_max / ((M_DRY + m_fuel_0) * G) >= TW_MIN`
(thrust/weight >= 1.1) — below that the
rocket cannot lift off. Note that within the ranges above the tightest possible
thrust/weight is about
1.70, so this
constraint only bites under `--allow-out-of-domain`.

## Known failure modes

1. **The avatar is not interchangeable.** The selection metric was computed with
   the Beta **mean** (`avatar = "mean"`). The mode-based
   `deterministic_action` in `policy.py` scores about an order of magnitude worse
   on this checkpoint — in boundary mode it collapses to bang-off/bang-on. Use
   the avatar recorded in `contract.json`.
2. **Absolute performance is low.** Best apogee_fraction is ~0.0735
   of the vacuum ballistic ceiling. Gravity and drag losses make ~0.3–0.7
   physically plausible, so this policy is far from the achievable frontier —
   it is a working pipeline, not a solved control problem.
3. **max-Q is never the binding constraint in eval.** `n_clamp_mean` is 0 across
   the eval grid: the policy throttles conservatively enough that the clamp never
   fires, so the constraint-handling machinery is effectively untested by these
   numbers.
4. **Out-of-distribution parameters are unguarded past the range check.** A value
   inside the trained range but far from the eval grid has no accuracy claim.

## Usage

```bash
# canonical instance
python solve.py

# a new instance the run never evaluated
python solve.py --param m_fuel_0=3500 --param Q_max=22000 --out result.json

# from a file, averaged over seeds
echo '{"params": {"m_fuel_0": 1200, "Q_max": 41000}}' > inst.json
python solve.py --instance inst.json --n-seeds 5

# does the bundle still reproduce its recorded number?
python smoke_test.py          # canonical instance
python smoke_test.py --full   # + the full 45-episode selection metric
```

The bundle is self-contained: copy this directory anywhere with `torch` and
`numpy` installed and the commands above work unchanged.
