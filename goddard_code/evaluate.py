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

"""Evaluation: held-out grid + deterministic policy.
Deterministic avatar for Beta:
  - unimodal (alpha,beta >= 1): use mean (== mode when both are > 1, well-defined)
  - boundary (alpha or beta < 1): use mode (piecewise: 0, 1, or interior peak).
    If both alpha<1 and beta<1 (bimodal), pick whichever boundary has higher density.
This matches the physical picture: at liftoff the trained boundary-mode Beta has
alpha>1, beta<1 -> mode at u=1 (bang-bang), which is what the physics wants.
"""
from __future__ import annotations
import numpy as np
import torch
from environment import RocketEnv
import config as C
from agent import DEVICE


def deterministic_action(dist) -> float:
    """Robust deterministic action from Beta(alpha, beta)."""
    a = float(dist.concentration1.item())
    b = float(dist.concentration0.item())
    if a > 1 and b > 1:
        return (a - 1) / (a + b - 2)
    if a > 1 and b <= 1:
        return 1.0
    if a <= 1 and b > 1:
        return 0.0
    if a <= 1 and b <= 1:
        # Bimodal: density(0) ~ inf if a<1, density(1) ~ inf if b<1.
        # Pick the boundary with the "heavier" tail: log density_0 = (a-1)*log(x)|x->0
        # dominates when a is smaller. So argmax density is at u=0 if a<b, u=1 if b<a.
        # Fallback: use mean.
        if a < b:
            return 0.0
        if b < a:
            return 1.0
        return 0.5
    return float(dist.mean.item())


def eval_grid_scenarios():
    scenarios = []
    for mf in C.EVAL_M_FUEL_GRID:
        for qm in C.EVAL_Q_MAX_GRID:
            scenarios.append({'m_fuel_0': mf, 'T_max': C.EVAL_T_MAX,
                              'cd': C.EVAL_CD, 'Q_max': qm})
    return scenarios


def evaluate_policy(agent, n_seeds_per_scenario=5, seed_base=C.SEED_EVAL, verbose=False,
                    action_mode='mean'):
    """action_mode: 'auto' -> deterministic_action(); 'mean' -> Beta mean;
    'sample' -> stochastic sample (single-sample avatar)."""
    scenarios = eval_grid_scenarios()
    results = []
    for si, params in enumerate(scenarios):
        for s in range(n_seeds_per_scenario):
            env = RocketEnv(seed=seed_base + si * 100 + s, params=params)
            env.reset()
            traj_actions = []
            while not env.done:
                obs = env._obs()
                with torch.no_grad():
                    ob_t = torch.from_numpy(obs.astype(np.float32))[None].to(DEVICE)
                    dist = agent.model.dist(ob_t)
                    if action_mode == 'mean':
                        u = float(dist.mean.item())
                    elif action_mode == 'sample':
                        u = float(dist.sample().item())
                    else:
                        u = deterministic_action(dist)
                u_capped = min(u, env.u_cap_safe())
                u_capped = max(0.0, min(1.0, u_capped))
                traj_actions.append(u_capped)
                _, _, _, _, info = env.step(u_capped)
            results.append({
                'scenario_idx': si,
                'm_fuel_0': params['m_fuel_0'],
                'Q_max': params['Q_max'],
                'h_apogee': info['h_apogee'],
                'apogee_fraction': info['apogee_fraction'],
                'n_clamp_steps': info['n_clamp_steps'],
                'n_steps': info['n_steps'],
                'action_mean': float(np.mean(traj_actions)),
                'action_std':  float(np.std(traj_actions)),
                'action_min':  float(np.min(traj_actions)),
                'action_max':  float(np.max(traj_actions)),
            })
    apogees = np.array([r['h_apogee'] for r in results])
    apogee_fracs = np.array([r['apogee_fraction'] for r in results])
    action_stds = np.array([r['action_std'] for r in results])
    action_ranges = np.array([r['action_max'] - r['action_min'] for r in results])
    summary = {
        'apogee_mean':     float(apogees.mean()),
        'apogee_frac_mean':float(apogee_fracs.mean()),
        'apogee_frac_med': float(np.median(apogee_fracs)),
        'apogee_frac_min': float(apogee_fracs.min()),
        'apogee_frac_max': float(apogee_fracs.max()),
        'action_std_mean': float(action_stds.mean()),
        'action_range_mean': float(action_ranges.mean()),
        'n_clamp_mean':    float(np.mean([r['n_clamp_steps'] for r in results])),
    }
    if verbose:
        print(f"  Eval [{action_mode}]: apogee_frac mean={summary['apogee_frac_mean']:.3f} "
              f"[{summary['apogee_frac_min']:.3f}, {summary['apogee_frac_max']:.3f}] "
              f"apogee_mean={summary['apogee_mean']:.1f}m "
              f"action_std={summary['action_std_mean']:.3f} "
              f"clamps={summary['n_clamp_mean']:.1f}")
    return summary, results


def evaluate_heuristic(n_seeds_per_scenario=5, seed_base=C.SEED_EVAL):
    from heuristic import heuristic_action
    scenarios = eval_grid_scenarios()
    results = []
    for si, params in enumerate(scenarios):
        for s in range(n_seeds_per_scenario):
            env = RocketEnv(seed=seed_base + si * 100 + s, params=params)
            env.reset()
            t = 0.0
            while not env.done:
                u = heuristic_action(t, env.params)
                u_capped = min(u, env.u_cap_safe())
                _, _, _, _, info = env.step(u_capped)
                t += C.DT
            results.append({
                'scenario_idx': si,
                'm_fuel_0': params['m_fuel_0'],
                'Q_max': params['Q_max'],
                'h_apogee': info['h_apogee'],
                'apogee_fraction': info['apogee_fraction'],
            })
    ap = np.array([r['apogee_fraction'] for r in results])
    hm = np.array([r['h_apogee'] for r in results])
    return {'apogee_frac_mean': float(ap.mean()), 'apogee_mean': float(hm.mean()),
            'apogee_frac_med': float(np.median(ap))}, results


if __name__ == "__main__":
    from agent import PPOAgent
    import os, sys
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "experiments/rocket_v3/checkpoints/best_model.pt"
    print("== Heuristic baseline ==")
    h_sum, _ = evaluate_heuristic(n_seeds_per_scenario=5)
    print(h_sum)

    if os.path.exists(ckpt):
        print(f"\n== Evaluating {ckpt} ==")
        a = PPOAgent()
        a.load(ckpt, strict=False)
        for mode in ['auto', 'mean', 'sample']:
            s_sum, _ = evaluate_policy(a, n_seeds_per_scenario=5, verbose=True, action_mode=mode)
    else:
        print(f"No checkpoint at {ckpt}")
