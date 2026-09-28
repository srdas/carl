#!/usr/bin/env python
"""solve.py -- solve a NEW problem instance with the shipped policy.

Usage:
    python solve.py                                   # canonical instance
    python solve.py --param m_fuel_0=3500 --param Q_max=22000 --out r.json
    python solve.py --instance instance.json --n-seeds 5

Everything comes from contract.json: obs layout, action bounds, the avatar that
produced the selection metric, and the instance-parameter schema. Nothing here
imports from the training tree -- run it from any directory.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from agent import PPOAgent  # noqa: E402
from environment import RocketEnv  # noqa: E402
from policy import deterministic_action  # noqa: E402


def load_contract() -> dict:
    with open(os.path.join(HERE, "contract.json")) as fh:
        return json.load(fh)


def _coerce(name: str, raw, spec: dict):
    try:
        return float(raw) if spec["type"] == "float" else int(raw)
    except (TypeError, ValueError):
        sys.exit(f"ERROR: --param {name}={raw!r} is not a {spec['type']}")


def resolve_instance(contract: dict, instance_file, overrides,
                     allow_ood: bool = False) -> dict:
    """Build a validated instance. Fails loudly; never silently clips."""
    schema = contract["param_schema"]
    params = dict(contract["canonical_instance"]["params"])

    supplied: dict = {}
    if instance_file:
        with open(instance_file) as fh:
            loaded = json.load(fh)
        supplied.update(loaded.get("params", loaded))
    for item in overrides or []:
        if "=" not in item:
            sys.exit(f"ERROR: --param must be k=v, got {item!r}")
        k, v = item.split("=", 1)
        supplied[k.strip()] = v.strip()

    unknown = sorted(set(supplied) - set(schema))
    if unknown:
        sys.exit(f"ERROR: unknown parameter(s) {unknown}; "
                 f"known: {sorted(schema)}")

    for name, raw in supplied.items():
        spec = schema[name]
        val = _coerce(name, raw, spec)
        if not (spec["low"] <= val <= spec["high"]):
            msg = (f"{name}={val:g} is outside the trained range "
                   f"[{spec['low']:g}, {spec['high']:g}]")
            if not allow_ood:
                sys.exit(f"ERROR: {msg}. Refusing to clip -- the policy has no "
                         f"validity claim there. Fix the instance, or pass "
                         f"--allow-out-of-domain to get an answer anyway.")
            print(f"  [OUT OF DOMAIN] {msg} -- result is not covered by any "
                  f"validity claim in MODEL_CARD.md")
        params[name] = val

    # The thrust/weight floor is checked separately because it couples two
    # parameters that are each individually in range. Within the trained ranges
    # the tightest possible T/W is min(T_max)/((M_DRY+max(m_fuel_0))*G) ~= 1.70,
    # so this can only trigger under --allow-out-of-domain -- it is a guard on
    # that escape hatch, not on ordinary input.
    con = contract["param_constraint"]
    cfg = contract["config"]
    tw = params["T_max"] / ((cfg["M_DRY"] + params["m_fuel_0"]) * cfg["G"])
    if tw < con["TW_MIN"]:
        sys.exit(f"ERROR: thrust/weight = {tw:.3f} < {con['TW_MIN']} -- this "
                 f"rocket cannot lift off ({con['why']}).")
    return params


def build_agent(contract: dict) -> PPOAgent:
    """Build the policy from the contract, not from module defaults."""
    import config as C
    net = contract["network"]
    # config.py travels with the bundle, but the contract is authoritative: if
    # someone edits config.py the shapes must still come from the contract.
    C.HIDDEN = tuple(net["hidden"])
    C.BETA_MODE = net["beta_mode"]
    C.BETA_EPS = net["beta_eps"]
    agent = PPOAgent(obs_dim=contract["obs_dim"], hidden=tuple(net["hidden"]))
    agent.model.beta_mode = net["beta_mode"]
    agent.model.beta_eps = net["beta_eps"]
    agent.load(os.path.join(HERE, "best_model.pt"), strict=True)
    agent.model.eval()
    return agent


def act(agent, obs, avatar: str) -> float:
    with torch.no_grad():
        dist = agent.model.dist(torch.from_numpy(obs.astype(np.float32))[None])
        if avatar == "mean":
            return float(dist.mean.item())
        return float(deterministic_action(dist))


def solve_once(agent, contract, params, seed) -> dict:
    lo, hi = contract["action_bounds"]
    env = RocketEnv(seed=seed, params=params)
    env.reset()
    traj, t, info = [], 0.0, {}
    while not env.done:
        obs = env._obs()
        u = act(agent, obs, contract["avatar"])
        u_applied = float(np.clip(min(u, env.u_cap_safe()), lo, hi))
        h, v, m = env.state
        _, r, _, _, info = env.step(u_applied)
        traj.append({
            "t": round(t, 4), "h": float(h), "v": float(v), "m": float(m),
            "u_policy": u, "u_applied": u_applied,
            "q_over_Qmax": info["q_over_Qmax"],
            "clamped": bool(info["clamp_active"]), "reward": float(r),
        })
        t += contract["config"]["DT"]
    return {
        "seed": seed,
        "objective": float(info["h_apogee"]),
        "apogee_fraction": float(info["apogee_fraction"]),
        "h_ref_ballistic": float(info["h_ref"]),
        "n_steps": int(info["n_steps"]),
        "n_clamp_steps": int(info["n_clamp_steps"]),
        "max_q_over_Qmax": max(s["q_over_Qmax"] for s in traj),
        "trajectory": traj,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instance", help="JSON file with a params object")
    ap.add_argument("--param", action="append", metavar="k=v",
                    help="override one parameter; repeatable")
    ap.add_argument("--out", help="write the full result JSON here")
    ap.add_argument("--n-seeds", type=int, default=1)
    ap.add_argument("--allow-out-of-domain", action="store_true",
                    help="run anyway outside the trained ranges; the result "
                         "carries no validity claim")
    ap.add_argument("--max-traj-rows", type=int, default=12,
                    help="printed rows; the JSON always holds the full trajectory")
    args = ap.parse_args()

    contract = load_contract()
    params = resolve_instance(contract, args.instance, args.param,
                              allow_ood=args.allow_out_of_domain)
    agent = build_agent(contract)

    base = contract["seed"]
    runs = [solve_once(agent, contract, params, base + 37 * i)
            for i in range(max(1, args.n_seeds))]
    objs = np.array([r["objective"] for r in runs])

    print("Instance:")
    for k in sorted(params):
        print(f"  {k:10s} = {params[k]:.6g}")
    print(f"\nAvatar: {contract['avatar']}  "
          f"(obs_dim={contract['obs_dim']}, action in "
          f"{contract['action_bounds']})")

    r0 = runs[0]
    traj = r0["trajectory"]
    every = max(1, len(traj) // args.max_traj_rows)
    print(f"\nDecision trajectory (seed {r0['seed']}, every {every} steps of "
          f"{len(traj)}):")
    print(f"  {'t':>7} {'h (m)':>11} {'v (m/s)':>9} {'m (kg)':>9} "
          f"{'u':>6} {'q/Qmax':>7} {'reward':>9}")
    for s in traj[::every] + [traj[-1]]:
        print(f"  {s['t']:7.1f} {s['h']:11.2f} {s['v']:9.2f} {s['m']:9.1f} "
              f"{s['u_applied']:6.3f} {s['q_over_Qmax']:7.3f} "
              f"{s['reward']:9.4f}")

    print(f"\nObjective (h_apogee, m): {objs.mean():.3f}"
          + (f"  +/- {objs.std():.3f} over {len(objs)} seeds"
             if len(objs) > 1 else ""))
    print(f"  apogee_fraction     : {r0['apogee_fraction']:.5f} "
          f"(of ballistic ceiling {r0['h_ref_ballistic']:.1f} m)")
    verdict = ("constraint satisfied" if r0["max_q_over_Qmax"] <= 1.0
               else "*** CONSTRAINT BREACH ***")
    print(f"  max q / Q_max       : {r0['max_q_over_Qmax']:.4f}  ({verdict})")
    print(f"  clamped steps       : {r0['n_clamp_steps']} / {r0['n_steps']}")

    ood = sorted(
        n for n, v in params.items()
        if not (contract["param_schema"][n]["low"] <= v
                <= contract["param_schema"][n]["high"]))
    result = {
        "params": params,
        "out_of_domain_params": ood,
        "avatar": contract["avatar"],
        "objective_mean": float(objs.mean()),
        "objective_std": float(objs.std()),
        "runs": runs,
        "contract": {
            "checkpoint_sha256": contract["created_from"]["checkpoint_sha256"],
            "run_name": contract["created_from"]["run_name"],
            "selection_metric": contract["selection_metric"]["value"],
        },
    }
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
