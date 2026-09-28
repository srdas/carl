#!/usr/bin/env python
"""Build a self-contained artifact/ bundle from a finished training run.

Usage:
    python make_artifact.py --run-name rocket_long
    python make_artifact.py --ckpt experiments/rocket_long/checkpoints/best_model.pt

The bundle must still work after being copied elsewhere, so the checkpoint is
copied (not referenced) and every runtime module the policy needs is copied in.
Nothing in artifact/ imports from the training tree.

There is deliberately no default run name here: a stale default pointing at a
run that was never written is exactly the bug in evaluate.py's __main__
(`experiments/rocket_v3/...`, a run this tree never produced).
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import shutil
import sys

import numpy as np
import torch

import config as C
from agent import PPOAgent
from environment import RocketEnv, h_ref_ballistic
from evaluate import deterministic_action, evaluate_policy

# Field order and units of environment.RocketEnv._obs(). The env builds the
# vector inline and does not name its own components, so this list is
# transcribed by hand and pinned by test_obs_names_match_env below -- if _obs()
# ever changes width, the build fails rather than shipping a wrong contract.
OBS_FIELDS = [
    ("rho_ratio",  "rho(h)/rho0, dimensionless"),
    ("v_norm",     "v / sqrt(2*Q_max/rho0), dimensionless"),
    ("q_ratio",    "q / Q_max, dimensionless"),
    ("q_headroom", "1 - q/Q_max, dimensionless"),
    ("fuel_frac",  "m_fuel / m_fuel_0, dimensionless"),
    ("dry_over_m", "m_dry / m, dimensionless"),
    ("thrust_weight", "T_max / (m*g), dimensionless"),
    ("energy_ratio",  "effective-altitude Tsiolkovsky budget / initial budget"),
    ("log_q",      "ln(Q_max / 30000 Pa)"),
    ("log_T",      "ln(T_max / 200000 N)"),
    ("log_m_fuel", "ln(m_fuel_0 / 1500 kg)"),
    ("cd",         "drag coefficient, dimensionless"),
]

# name -> (type, low, high, default). Ranges are the training distribution
# (config.*_RANGE); the canonical eval point supplies the defaults.
PARAM_SCHEMA = {
    "m_fuel_0": ("float", C.M_FUEL_RANGE[0], C.M_FUEL_RANGE[1], 2000.0),
    "T_max":    ("float", C.T_MAX_RANGE[0],  C.T_MAX_RANGE[1],  C.EVAL_T_MAX),
    "cd":       ("float", C.CD_RANGE[0],     C.CD_RANGE[1],     C.EVAL_CD),
    "Q_max":    ("float", C.Q_MAX_RANGE[0],  C.Q_MAX_RANGE[1],  3.0e4),
}

# Modules the artifact needs at run time. evaluate.py is NOT copied: its
# __main__ carries a stale checkpoint default, and only deterministic_action is
# needed -- that function is moved verbatim into artifact/policy.py below.
RUNTIME_MODULES = ["config.py", "network.py", "agent.py", "environment.py"]

# The stranger's entry points live next to make_artifact.py as templates and are
# copied in under their final names.
ENTRY_POINTS = {
    "artifact_solve_template.py": "solve.py",
    "artifact_smoke_template.py": "smoke_test.py",
}


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_checkpoint(run_name: str | None, ckpt: str | None) -> str:
    """Resolve the best checkpoint from --ckpt or the run's selection metric."""
    if ckpt:
        if not os.path.isfile(ckpt):
            sys.exit(f"ERROR: --ckpt {ckpt} does not exist")
        return ckpt
    if not run_name:
        sys.exit("ERROR: pass --run-name or --ckpt (there is no default run)")
    ckpt_dir = os.path.join("experiments", run_name, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        sys.exit(f"ERROR: no such run: {ckpt_dir}")
    # best_model.pt is written only when the canonical selection score improves
    # (train.py phase_ppo), so it is the selection-metric checkpoint by
    # construction. Fall back to latest only if best was never written.
    for name in ("best_model.pt", "latest_model.pt"):
        p = os.path.join(ckpt_dir, name)
        if os.path.isfile(p):
            if name != "best_model.pt":
                print(f"  [warn] no best_model.pt in {ckpt_dir}; using {name}")
            return p
    sys.exit(f"ERROR: no checkpoint found in {ckpt_dir}")


def obs_dim_from_state_dict(sd) -> int:
    """Derive obs_dim from the first actor layer.

    Pre-contract checkpoints (like this run's) store only model/optimizer/lr/
    update/eval/score -- there is no `extra` block recording the shapes, so they
    have to come from the weights themselves.
    """
    for key in ("actor.0.weight", "critic.0.weight"):
        if key in sd:
            return int(sd[key].shape[1])
    raise SystemExit("ERROR: cannot infer obs_dim from checkpoint state dict")


def rollout(agent, params: dict, seed: int, avatar: str) -> dict:
    """Run one deterministic episode; return trajectory + objective."""
    env = RocketEnv(seed=seed, params=params)
    env.reset()
    traj = []
    t = 0.0
    info = {}
    while not env.done:
        obs = env._obs()
        with torch.no_grad():
            dist = agent.model.dist(torch.from_numpy(obs.astype(np.float32))[None])
            u = (float(dist.mean.item()) if avatar == "mean"
                 else deterministic_action(dist))
        u_cap = env.u_cap_safe()
        u_applied = float(np.clip(min(u, u_cap), 0.0, 1.0))
        h, v, m = env.state
        _, r, _, _, info = env.step(u_applied)
        traj.append({
            "t": round(t, 4), "h": float(h), "v": float(v), "m": float(m),
            "u_policy": float(u), "u_applied": u_applied,
            "q_over_Qmax": info["q_over_Qmax"], "clamped": info["clamp_active"],
            "reward": float(r),
        })
        t += C.DT
    return {
        "objective": float(info["h_apogee"]),
        "objective_name": "h_apogee",
        "objective_units": "m",
        "apogee_fraction": float(info["apogee_fraction"]),
        "h_ref_ballistic": float(info["h_ref"]),
        "n_steps": int(info["n_steps"]),
        "n_clamp_steps": int(info["n_clamp_steps"]),
        "trajectory": traj,
    }


def _baselines(run_name: str) -> dict:
    """Scrape the run log for the heuristic and random-init baselines.

    Best-effort: a missing log means the card says "not recorded" rather than
    inventing a number.
    """
    log = os.path.join("experiments", run_name, "train.log")
    out: dict = {}
    if not os.path.isfile(log):
        return out
    import re
    with open(log, errors="replace") as fh:
        text = fh.read()
    m = re.search(r"Heuristic: apogee_frac_mean=([\d.]+)", text)
    if m:
        out["heuristic"] = float(m.group(1))
    m = re.search(r"Random-init policy baseline ==\s*\n\s*Eval \[mean\]: "
                  r"apogee_frac mean=([\d.]+)", text)
    if m:
        out["random_init"] = float(m.group(1))
    m = re.search(r"Post-BC eval ==\s*\n\s*Eval \[mean\]: "
                  r"apogee_frac mean=([\d.]+)", text)
    if m:
        out["post_bc"] = float(m.group(1))
    return out


def write_model_card(out_dir, contract, canon, raw):
    run = contract["created_from"]["run_name"]
    base = _baselines(run)
    sm = contract["selection_metric"]
    sched = contract["param_schema"]
    latest = os.path.join("experiments", run, "checkpoints", "latest_model.pt")
    latest_score = None
    if os.path.isfile(latest):
        lraw = torch.load(latest, map_location="cpu", weights_only=False)
        latest_score = (lraw.get("eval") or {}).get("apogee_frac_mean")

    def fmt(x):
        return f"{x:.4f}" if isinstance(x, (int, float)) else "not recorded"

    rows = "\n".join(
        f"| `{n}` | {s['type']} | {s['low']:g} – {s['high']:g} | {s['default']:g} |"
        for n, s in sched.items())

    text = f"""# Model card — Goddard rocket with a hard max-Q constraint

Bundle built by `make_artifact.py` from run **{run}**, update
{contract['created_from']['update']}.
Checkpoint sha256 `{contract['created_from']['checkpoint_sha256'][:16]}…`

## Problem

Fly a Goddard rocket as high as possible while never exceeding a dynamic-pressure
limit `Q_max`. Thrust is throttleable; propellant is finite; drag and air density
fall off exponentially with altitude. The interesting tension is that the
altitude-optimal ascent wants maximum thrust early, exactly where dynamic
pressure peaks.

## MDP

- **State** ({contract['obs_dim']}-D, all dimensionless, clipped to
  {contract['normalisation']['obs_clip']}):
{chr(10).join(f"  {i}. `{f['name']}` — {f['units']}" for i, f in enumerate(contract['obs_fields'], 1))}
- **Action**: scalar `u` in {contract['action_bounds']} — {contract['action_semantics']}
- **Reward**: `{contract['normalisation']['reward_mode']}` — telescoping altitude
  gain `dh / {contract['normalisation']['H_SCALE_FIXED']:g}` per `DT =
  {contract['config']['DT']}` s. Undiscounted (gamma = 1), so the return is the
  apogee in scaled units.
- **Termination**: apogee (v <= 0 after liftoff), ground contact, or 2000 steps.
- **Constraint handling**: the environment enforces max-Q itself by clamping `u`
  and charging a {contract['config']['CLAMP_DRAG_PENALTY'] * 100:g}% drag penalty
  on clamped steps, so the constraint cannot be violated by a bad policy — it can
  only be paid for.

## Training recipe

PPO with a Beta-distributed throttle (`beta_mode =
{contract['network']['beta_mode']}`, so the policy may place mass at u=0 and
u=1), separate {contract['network']['hidden']} tanh MLPs for actor and critic.
Behaviour cloning from a liftoff-then-exponential-decay heuristic, then a
critic-only warmup with the actor frozen, then full PPO. Domain randomisation
over all four instance parameters with rejection sampling on thrust/weight
(>= {contract['param_constraint']['TW_MIN']}).

Training seed {C.SEED_TRAIN}; evaluation seed {contract['seed']} (the eval seed
is what `solve.py` and `smoke_test.py` use, so their numbers are reproducible).
{C.EPISODES_PER_UPDATE} episodes per PPO update, CPU-only. This checkpoint is
from update **{contract['created_from']['update']}** — the update at which the
selection metric last improved, not the end of the run.

## Results — selection metric `{sm['name']}`

{sm['how']}.

| Policy | {sm['name']} |
| --- | --- |
| **This bundle (best)** | **{fmt(sm['value'])}** |
| Latest checkpoint | {fmt(latest_score)} |
| Post-BC, pre-PPO | {fmt(base.get('post_bc'))} |
| Heuristic baseline | {fmt(base.get('heuristic'))} |
| Random init | {fmt(base.get('random_init'))} |

Canonical instance (`{ {k: round(v, 3) for k, v in contract['canonical_instance']['params'].items()} }`,
seed {contract['canonical_instance']['seed']}): apogee
**{canon['objective']:.1f} m**, apogee_fraction
{canon['apogee_fraction']:.5f} of the {canon['h_ref_ballistic']:.0f} m vacuum
ballistic ceiling, over {canon['n_steps']} steps.

Read those honestly: the trained policy beats the heuristic on the selection
metric by roughly {(sm['value'] or 0) - base.get('heuristic', 0):+.4f}, which is
a small margin, and the random-init baseline is already close behind. The reward
signal here is weak relative to the spread across instances.

## Validity domain

| parameter | type | trained range | canonical |
| --- | --- | --- | --- |
{rows}

`solve.py` **refuses** parameters outside these ranges rather than clipping them:
the policy has no validity claim there, and silently clipping would return a
confident answer to a question it was never trained on. Pass
`--allow-out-of-domain` to override; the result is then labelled out-of-domain in
both stdout and the result JSON.

Instances must also satisfy `{contract['param_constraint']['expr']}`
(thrust/weight >= {contract['param_constraint']['TW_MIN']}) — below that the
rocket cannot lift off. Note that within the ranges above the tightest possible
thrust/weight is about
{C.T_MAX_RANGE[0] / ((C.M_DRY + C.M_FUEL_RANGE[1]) * C.G):.2f}, so this
constraint only bites under `--allow-out-of-domain`.

## Known failure modes

1. **The avatar is not interchangeable.** The selection metric was computed with
   the Beta **mean** (`avatar = "{contract['avatar']}"`). The mode-based
   `deterministic_action` in `policy.py` scores about an order of magnitude worse
   on this checkpoint — in boundary mode it collapses to bang-off/bang-on. Use
   the avatar recorded in `contract.json`.
2. **Absolute performance is low.** Best apogee_fraction is ~{fmt(sm['value'])}
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
echo '{{"params": {{"m_fuel_0": 1200, "Q_max": 41000}}}}' > inst.json
python solve.py --instance inst.json --n-seeds 5

# does the bundle still reproduce its recorded number?
python smoke_test.py          # canonical instance
python smoke_test.py --full   # + the full 45-episode selection metric
```

The bundle is self-contained: copy this directory anywhere with `torch` and
`numpy` installed and the commands above work unchanged.
"""
    with open(os.path.join(out_dir, "MODEL_CARD.md"), "w") as fh:
        fh.write(text)


def build(run_name, ckpt_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    obs_dim = obs_dim_from_state_dict(sd)
    if obs_dim != RocketEnv.OBS_DIM or obs_dim != len(OBS_FIELDS):
        sys.exit(f"ERROR: checkpoint obs_dim={obs_dim} but env OBS_DIM="
                 f"{RocketEnv.OBS_DIM} and {len(OBS_FIELDS)} named fields")

    shutil.copy2(ckpt_path, os.path.join(out_dir, "best_model.pt"))
    for mod in RUNTIME_MODULES:
        shutil.copy2(mod, os.path.join(out_dir, mod))
    for src, dst in ENTRY_POINTS.items():
        if not os.path.isfile(src):
            sys.exit(f"ERROR: missing entry-point template {src}")
        shutil.copy2(src, os.path.join(out_dir, dst))

    # Move deterministic_action into the bundle rather than reimplementing it,
    # and without dragging in evaluate.py's stale __main__ default.
    with open(os.path.join(out_dir, "policy.py"), "w") as fh:
        fh.write('"""Deterministic avatar, moved verbatim from evaluate.py by '
                 'make_artifact.py."""\n\n\n')
        fh.write(inspect.getsource(deterministic_action))

    # The selection metric was computed with the Beta *mean* avatar
    # (train.py -> evaluate_policy default action_mode='mean'), not with
    # deterministic_action. They are not interchangeable on this policy, so the
    # avatar is part of the contract and solve.py reads it from here.
    avatar = "mean"
    canonical = {k: v[3] for k, v in PARAM_SCHEMA.items()}

    agent = PPOAgent(obs_dim=obs_dim)
    agent.load(ckpt_path, strict=False)
    canon = rollout(agent, canonical, C.SEED_EVAL, avatar)
    print(f"  canonical instance objective: {canon['objective']:.4f} m "
          f"(apogee_fraction {canon['apogee_fraction']:.5f})")

    recorded = raw.get("eval") or {}
    contract = {
        "created_from": {
            "run_name": run_name,
            "checkpoint": os.path.relpath(ckpt_path),
            "checkpoint_sha256": _sha256(ckpt_path),
            "update": raw.get("update"),
        },
        "obs_dim": obs_dim,
        "obs_fields": [{"name": n, "units": u} for n, u in OBS_FIELDS],
        "action_dim": 1,
        "action_bounds": [0.0, 1.0],
        "action_semantics": (
            "u = throttle fraction of T_max, applied for one DT. The env "
            "applies a hard max-Q clamp: u is reduced to env.u_cap_safe() when "
            "it would breach Q_max, with a 5% drag penalty on clamped steps."
        ),
        "avatar": avatar,
        "avatar_note": (
            "'mean' = Beta mean, the avatar train.py used to compute the "
            "selection metric. 'auto' = policy.deterministic_action (Beta "
            "mode); it scores far worse on this checkpoint. Do not swap them."
        ),
        "network": {"hidden": list(C.HIDDEN), "beta_mode": C.BETA_MODE,
                    "beta_eps": C.BETA_EPS},
        "normalisation": {
            "v_ref": "sqrt(2*Q_max/RHO0)", "q_ref": "Q_max",
            "log_q_ref_pa": 30000.0, "log_T_ref_n": 200000.0,
            "log_m_fuel_ref_kg": 1500.0, "obs_clip": [-10.0, 10.0],
            "reward_mode": C.REWARD_MODE, "H_SCALE_FIXED": C.H_SCALE_FIXED,
        },
        "config": {
            "G": C.G, "M_DRY": C.M_DRY, "RHO0": C.RHO0, "H_SCALE": C.H_SCALE,
            "CE_RATIO": C.CE_RATIO, "DT": C.DT,
            "CLAMP_DRAG_PENALTY": C.CLAMP_DRAG_PENALTY, "TW_MIN": C.TW_MIN,
            "HIDDEN": list(C.HIDDEN), "BETA_MODE": C.BETA_MODE,
            "BETA_EPS": C.BETA_EPS, "REWARD_MODE": C.REWARD_MODE,
            "H_SCALE_FIXED": C.H_SCALE_FIXED,
        },
        "param_schema": {
            name: {"type": t, "low": lo, "high": hi, "default": dflt}
            for name, (t, lo, hi, dflt) in PARAM_SCHEMA.items()
        },
        "param_constraint": {
            "expr": "T_max / ((M_DRY + m_fuel_0) * G) >= TW_MIN",
            "TW_MIN": C.TW_MIN,
            "why": "training rejection-sampled anything that cannot lift off",
        },
        "seed": C.SEED_EVAL,
        "selection_metric": {
            "name": "eval_apogee_frac_mean",
            "value": raw.get("score"),
            "how": ("mean apogee_fraction over the 9-point eval grid "
                    "(EVAL_M_FUEL_GRID x EVAL_Q_MAX_GRID) x 5 seeds, "
                    f"avatar={avatar}"),
            "recorded_eval": recorded,
            "note": ("This is a grid aggregate, not a single-instance "
                     "objective. smoke_test.py checks the canonical instance "
                     "objective by default and the grid metric with --full."),
        },
        "canonical_instance": {
            "params": canonical,
            "seed": C.SEED_EVAL,
            "expected_objective": canon["objective"],
            "expected_apogee_fraction": canon["apogee_fraction"],
            "n_steps": canon["n_steps"],
        },
        "validity_domain": {
            name: [lo, hi] for name, (_t, lo, hi, _d) in PARAM_SCHEMA.items()
        },
    }

    write_model_card(out_dir, contract, canon, raw)

    # Fingerprint after all .py files are in place, policy.py included.
    contract["code_fingerprint"] = {
        f: _sha256(os.path.join(out_dir, f))
        for f in sorted(os.listdir(out_dir)) if f.endswith(".py")
    }
    with open(os.path.join(out_dir, "contract.json"), "w") as fh:
        json.dump(contract, fh, indent=2)
    return contract, canon


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-name")
    ap.add_argument("--ckpt")
    ap.add_argument("--out", default="artifact/")
    args = ap.parse_args()

    ckpt_path = resolve_checkpoint(args.run_name, args.ckpt)
    run_name = args.run_name
    if run_name is None:
        # Recover the run name from the resolved path so the contract records
        # provenance even when only --ckpt was given.
        parts = os.path.normpath(ckpt_path).split(os.sep)
        run_name = parts[-3] if len(parts) >= 3 else "unknown"
    print(f"Building artifact from {ckpt_path} (run {run_name})")

    contract, _ = build(run_name, ckpt_path, args.out)
    sm = contract["selection_metric"]
    print(f"  {sm['name']} = {sm['value']}")
    print(f"  wrote {args.out} "
          f"({len(contract['code_fingerprint'])} .py + checkpoint + contract)")
    print("  next: python artifact/smoke_test.py")


if __name__ == "__main__":
    main()
