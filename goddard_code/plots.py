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

"""Plots: training curves, eval-grid heatmap, action heatmap, scatter vs heuristic.
Usage: python plots.py --run-name rocket_v3
"""
import argparse, os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from environment import RocketEnv
from heuristic import heuristic_action
from evaluate import evaluate_policy, evaluate_heuristic, eval_grid_scenarios
from agent import PPOAgent, DEVICE
import config as C


def load_agent(ckpt_path):
    a = PPOAgent()
    a.load(ckpt_path, strict=False)
    return a


def plot_training_curve(log_path, out_path):
    with open(log_path) as f:
        records = json.load(f)
    train = [r for r in records if r.get('phase', '').startswith('Phase')]
    if not train:
        return
    up = np.array([r['update'] for r in train])
    apfrac = np.array([r.get('apogee_frac_mean', np.nan) for r in train])
    pi_l = np.array([r.get('policy_loss', 0) for r in train])
    v_l = np.array([r.get('value_loss', 0) for r in train])
    ent = np.array([r.get('entropy', 0) for r in train])
    kl = np.array([r.get('approx_kl', 0) for r in train])
    evals = [(r['update'], r['eval']['apogee_frac_mean']) for r in train if 'eval' in r]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    ax.plot(up, apfrac, 'b-', alpha=0.4, label='train batch (stochastic)')
    if len(apfrac) >= 5:
        win = min(5, len(apfrac))
        smooth = np.convolve(apfrac, np.ones(win)/win, mode='valid')
        ax.plot(up[win-1:], smooth, 'b-', lw=2, label=f'MA({win})')
    if evals:
        eu, ev = zip(*evals)
        ax.plot(eu, ev, 'r*-', ms=10, label='eval (deterministic)')
    ax.set(xlabel='update', ylabel='apogee_frac_mean', title='Canonical eval score')
    ax.grid(True, alpha=0.3); ax.legend()

    ax = axes[0, 1]
    ax.plot(up, pi_l, label='policy'); ax.plot(up, v_l, label='value')
    ax.set(xlabel='update', ylabel='loss', title='Losses'); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(up, ent); ax.set(xlabel='update', ylabel='entropy', title='Entropy')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(up, kl); ax.set(xlabel='update', ylabel='approx KL', title='PPO approx KL')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"  saved {out_path}")


def plot_action_vs_time(agent, out_path, seed=1234):
    """Overlay action(t) for agent, heuristic on a representative eval scenario."""
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=False)
    scenarios = eval_grid_scenarios()
    for si, params in enumerate(scenarios):
        ax = axes[si // 3, si % 3]
        # Agent
        env = RocketEnv(seed=seed + si, params=params); env.reset()
        ts, us_agent, hs, qs, qcaps = [], [], [], [], []
        t = 0.0
        while not env.done:
            obs = env._obs()
            with torch.no_grad():
                ob_t = torch.from_numpy(obs.astype(np.float32))[None].to(DEVICE)
                d = agent.model.dist(ob_t)
                u = float(d.mean.item())
            u_cap = env.u_cap_safe()
            u_used = min(u, u_cap)
            ts.append(t); us_agent.append(u_used); qcaps.append(u_cap)
            _, _, _, _, info = env.step(u_used)
            hs.append(info['h']); qs.append(info['q_over_Qmax'])
            t += C.DT
        # Heuristic
        env = RocketEnv(seed=seed + si, params=params); env.reset()
        ts_h, us_h = [], []
        t = 0.0
        while not env.done:
            u = heuristic_action(t, env.params)
            u_used = min(u, env.u_cap_safe())
            ts_h.append(t); us_h.append(u_used)
            env.step(u_used); t += C.DT
        ax.plot(ts, us_agent, 'b-', lw=1.5, label='agent')
        ax.plot(ts_h, us_h, 'r--', lw=1, label='heuristic')
        ax.plot(ts, qcaps, 'k:', lw=0.8, alpha=0.5, label='u_cap')
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"m_f={params['m_fuel_0']:.0f} Q={params['Q_max']/1e3:.0f}kPa "
                     f"apogee={hs[-1]:.0f}m", fontsize=9)
        ax.grid(True, alpha=0.3)
        if si == 0:
            ax.legend(fontsize=8, loc='best')
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"  saved {out_path}")


def plot_apogee_scatter(agent, out_path, n_seeds=5):
    """Scatter: agent apogee vs heuristic apogee across eval grid + seeds."""
    _, agent_results = evaluate_policy(agent, n_seeds_per_scenario=n_seeds)
    _, heur_results  = evaluate_heuristic(n_seeds_per_scenario=n_seeds)
    a_ap = np.array([r['h_apogee'] for r in agent_results])
    h_ap = np.array([r['h_apogee'] for r in heur_results])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(h_ap, a_ap, alpha=0.6)
    lim = [0, max(a_ap.max(), h_ap.max()) * 1.05]
    ax.plot(lim, lim, 'k--', alpha=0.5)
    ax.set(xlabel='heuristic apogee (m)', ylabel='agent apogee (m)',
           title=f'Zero-shot eval grid ({len(a_ap)} episodes)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"  saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-name', type=str, required=True)
    ap.add_argument('--ckpt', type=str, default='best_model.pt')
    args = ap.parse_args()

    run_dir = os.path.join("experiments", args.run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    log_path = os.path.join(run_dir, "train_log.json")
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if os.path.exists(log_path):
        plot_training_curve(log_path, os.path.join(plots_dir, 'training.png'))
    else:
        print(f"  no train_log at {log_path}, skipping training-curve plot")

    ckpt = os.path.join(ckpt_dir, args.ckpt)
    if os.path.exists(ckpt):
        print(f"Loading {ckpt}")
        agent = load_agent(ckpt)
        plot_action_vs_time(agent, os.path.join(plots_dir, 'action_vs_time.png'))
        plot_apogee_scatter(agent, os.path.join(plots_dir, 'apogee_scatter.png'))
    else:
        print(f"  no checkpoint at {ckpt}")


if __name__ == "__main__":
    main()
