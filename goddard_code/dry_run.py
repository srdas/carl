# SPDX-License-Identifier: GPL-3.0-only
#
# This file is part of CARL.
#
# CARL is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License, version 3, as published by the
# Free Software Foundation.
#
# CARL is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for details.
#
# You should have received a copy of the GNU General Public License along
# with CARL. If not, see <https://www.gnu.org/licenses/>.

"""Step 8 dry run: random policy vs heuristic. Establishes reward ranges and validates env.
Also does reward-decomposition smoke tests.
"""
import numpy as np
import time
from environment import RocketEnv, dyn_pressure
from heuristic import rollout_heuristic, heuristic_action
import config as C


def random_policy_run(n_eps=500, seed=0):
    rng = np.random.default_rng(seed)
    env = RocketEnv(seed=seed)
    all_rewards = []
    all_states  = []
    ap_fracs    = []
    apogees     = []
    ep_lens     = []
    n_clamp_total = 0
    for i in range(n_eps):
        env.reset()
        while not env.done:
            u = float(rng.uniform(0, 1))
            obs, r, term, trunc, info = env.step(u)
            all_rewards.append(r)
            all_states.append(obs)
        ap_fracs.append(info["apogee_fraction"])
        apogees.append(info["h_apogee"])
        ep_lens.append(info["n_steps"])
        n_clamp_total += info["n_clamp_steps"]
    return {
        "rewards": np.array(all_rewards),
        "states": np.array(all_states),
        "apogee_fraction": np.array(ap_fracs),
        "h_apogee": np.array(apogees),
        "ep_len": np.array(ep_lens),
        "n_clamp": n_clamp_total,
    }


def heuristic_run(n_eps=200, seed=1):
    env = RocketEnv(seed=seed)
    ap_fracs = []
    apogees  = []
    ep_lens  = []
    n_clamp_total = 0
    for i in range(n_eps):
        _, acts, rews, infos = rollout_heuristic(env)
        info = infos[-1]
        ap_fracs.append(info["apogee_fraction"])
        apogees.append(info["h_apogee"])
        ep_lens.append(info["n_steps"])
        n_clamp_total += info["n_clamp_steps"]
    return {
        "apogee_fraction": np.array(ap_fracs),
        "h_apogee": np.array(apogees),
        "ep_len": np.array(ep_lens),
        "n_clamp": n_clamp_total,
    }


def determinism_check():
    env1 = RocketEnv(seed=42); env2 = RocketEnv(seed=42)
    env1.reset(); env2.reset()
    for _ in range(50):
        u = 0.7
        s1, r1, t1, _, _ = env1.step(u)
        s2, r2, t2, _, _ = env2.step(u)
        assert np.allclose(s1, s2), "Determinism failed"
        assert r1 == r2
        if t1 or t2: break
    print("[OK] deterministic")


def reward_decomposition_traces():
    """Short traces: normal, immediate coast (pathological), bang-bang."""
    # Small binding-Q_max scenario
    p = dict(m_fuel_0=2000.0, T_max=2.0e5, cd=1.5, Q_max=15000.0)
    print(f"\n-- Trace: fixed params {p}")
    # Full-throttle bang-bang
    env = RocketEnv(params=p)
    env.reset()
    step = 0
    trace_rows = []
    while not env.done and step < 50:  # sample first 50 steps
        obs, r, term, trunc, info = env.step(1.0)
        trace_rows.append((step, info['h'], info['v'], info['q_over_Qmax'],
                           info['throttle_commanded'], info['throttle_applied'],
                           info['clamp_active'], r))
        step += 1
    for row in trace_rows[:10]:
        print("  step={:3d} h={:8.1f} v={:7.2f} q/Qm={:.3f} u_cmd={:.2f} u_app={:.3f} "
              "clamp={} r={:+.5f}".format(*row))
    # Continue to end
    while not env.done:
        env.step(1.0)
    print(f"  -> bang-bang apogee_fraction = {env.h_apogee / env.h_ref:.3f}, clamps={env.n_clamp}")
    # Pathological: coast (u=0 always)
    env = RocketEnv(params=p); env.reset()
    while not env.done:
        env.step(0.0)
    print(f"  -> zero-throttle apogee_fraction = {env.h_apogee / env.h_ref:.3f}")
    # Heuristic
    env = RocketEnv(params=p)
    _, acts, rews, infos = rollout_heuristic(env)
    print(f"  -> heuristic apogee_fraction = {infos[-1]['apogee_fraction']:.3f}, clamps={infos[-1]['n_clamp_steps']}")
    # Reward sum vs terminal
    r_sum = sum(rews)
    r_term = infos[-1]['apogee_fraction']
    assert abs(r_sum - r_term) < 1e-4, f"Reward telescoping mismatch: {r_sum} vs {r_term}"
    print(f"  [OK] telescoping: sum(r) = {r_sum:.4f} matches apogee_fraction = {r_term:.4f}")


def main():
    print("=" * 60)
    print("STEP 8 DRY RUN")
    print("=" * 60)

    determinism_check()

    print("\n-- Reward decomposition traces --")
    reward_decomposition_traces()

    print("\n-- Random policy (500 eps) --")
    t0 = time.time()
    R = random_policy_run(n_eps=500)
    print(f"time: {time.time()-t0:.1f}s")
    r = R["rewards"]
    print(f"reward: n={len(r)} mean={r.mean():+.5f} std={r.std():.5f} "
          f"min={r.min():+.5f} max={r.max():+.5f} finite={np.isfinite(r).all()}")
    af = R["apogee_fraction"]
    print(f"apogee_fraction: mean={af.mean():.3f} std={af.std():.3f} "
          f"min={af.min():.3f} max={af.max():.3f}")
    print(f"ep_len: mean={R['ep_len'].mean():.0f} min={R['ep_len'].min()} max={R['ep_len'].max()}")
    print(f"n_clamp_total: {R['n_clamp']}")
    print("state ranges per dim (min/max):")
    S = R["states"]
    for i in range(S.shape[1]):
        print(f"  dim{i:2d}: [{S[:,i].min():+.3f}, {S[:,i].max():+.3f}]")

    print("\n-- Heuristic policy (200 eps) --")
    t0 = time.time()
    H = heuristic_run(n_eps=200)
    print(f"time: {time.time()-t0:.1f}s")
    af = H["apogee_fraction"]
    print(f"apogee_fraction: mean={af.mean():.3f} std={af.std():.3f} "
          f"min={af.min():.3f} max={af.max():.3f}")
    print(f"ep_len: mean={H['ep_len'].mean():.0f}")
    print(f"n_clamp_total: {H['n_clamp']}")

    print("\n-- Summary --")
    ratio = H["apogee_fraction"].mean() / max(R["apogee_fraction"].mean(), 1e-6)
    print(f"heuristic mean / random mean = {ratio:.2f}x")
    assert H["apogee_fraction"].mean() > R["apogee_fraction"].mean(), \
        "Heuristic should beat random"
    print("[OK] heuristic beats random")


if __name__ == "__main__":
    main()
