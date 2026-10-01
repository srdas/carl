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

"""Supervised (BC) pretraining of the actor on heuristic actions, then critic warmup.

Generates (obs, u_target) pairs by rolling out the heuristic under randomized params
(with actor-cap applied so targets are always feasible). Trains actor's Beta mean
`alpha/(alpha+beta)` to match u_target with MSE. Then freezes actor, warms critic on
returns from the pretrained policy.
"""
import numpy as np
import torch
import torch.nn.functional as F
from environment import RocketEnv
from heuristic import heuristic_action
from network import ActorCritic
import config as C


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collect_bc_data(n_episodes=200, seed=0):
    """Roll out heuristic across randomized params, collect (obs, u_target) pairs."""
    env = RocketEnv(seed=seed)
    obss, us = [], []
    for _ in range(n_episodes):
        env.reset()
        t = 0.0
        while not env.done:
            obs = env._obs()
            u_heur = heuristic_action(t, env.params)
            # Apply actor-side cap so target is feasible (avoids driving BC to
            # infeasible clamped-then-penalized region).
            u_target = min(u_heur, env.u_cap_safe())
            u_target = float(np.clip(u_target, 1e-3, 1.0 - 1e-3))
            obss.append(obs)
            us.append(u_target)
            # Step env with the same action (so state distribution matches heuristic trajectory)
            env.step(u_target)
            t += C.DT
    return np.array(obss, dtype=np.float32), np.array(us, dtype=np.float32)


def bc_pretrain(model: ActorCritic, obs_np, u_np,
                steps: int = C.BC_STEPS, lr: float = C.BC_LR,
                batch_size: int = C.BC_BATCH_SIZE, val_frac: float = 0.1):
    """MSE loss on Beta mean vs u_target. Actor-only optimizer."""
    n = obs_np.shape[0]
    idx = np.random.default_rng(0).permutation(n)
    n_val = int(val_frac * n)
    val_i, tr_i = idx[:n_val], idx[n_val:]
    obs_tr = torch.from_numpy(obs_np[tr_i]).to(DEVICE)
    u_tr   = torch.from_numpy(u_np[tr_i]).to(DEVICE)
    obs_va = torch.from_numpy(obs_np[val_i]).to(DEVICE)
    u_va   = torch.from_numpy(u_np[val_i]).to(DEVICE)

    opt = torch.optim.Adam(model.actor.parameters(), lr=lr)
    n_tr = obs_tr.shape[0]
    losses = []
    for step in range(steps):
        # sample minibatch
        i = torch.randint(0, n_tr, (batch_size,), device=DEVICE)
        ob, ut = obs_tr[i], u_tr[i]
        d = model.dist(ob)
        # Beta mean = alpha/(alpha+beta)
        alpha = d.concentration1
        beta  = d.concentration0
        mean  = alpha / (alpha + beta)
        loss = F.mse_loss(mean, ut)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % 100 == 0 or step == steps - 1:
            with torch.no_grad():
                dv = model.dist(obs_va)
                mv = dv.concentration1 / (dv.concentration1 + dv.concentration0)
                val_mse = F.mse_loss(mv, u_va).item()
                ent_v   = dv.entropy().mean().item()
                conc_v  = (dv.concentration1 + dv.concentration0).mean().item()
            print(f"  BC step {step:4d}: train_mse={loss.item():.4f} "
                  f"val_mse={val_mse:.4f} entropy={ent_v:+.3f} alpha+beta={conc_v:.1f}")
    with torch.no_grad():
        dv = model.dist(obs_va)
        mv = dv.concentration1 / (dv.concentration1 + dv.concentration0)
        final_val_mse = F.mse_loss(mv, u_va).item()
    return final_val_mse


def collect_returns_for_critic(model: ActorCritic, n_episodes=200, seed=1):
    """Roll out the current (BC'd) policy, collect (obs, return_to_go) pairs.
    Uses gamma=1 (Monte Carlo return to end of episode)."""
    env = RocketEnv(seed=seed)
    obss_ep, rets_ep = [], []
    for _ in range(n_episodes):
        env.reset()
        ep_obs, ep_r = [], []
        while not env.done:
            ob = env._obs()
            with torch.no_grad():
                d = model.dist(torch.from_numpy(ob).to(DEVICE))
                # Deterministic action = Beta mean, capped
                u = float(d.mean.item())
            u_capped = min(u, env.u_cap_safe())
            ep_obs.append(ob)
            _, r, _, _, _ = env.step(u_capped)
            ep_r.append(r)
        # Monte Carlo returns (gamma=1)
        G = 0.0
        rets = np.zeros_like(ep_r, dtype=np.float32)
        for i in range(len(ep_r) - 1, -1, -1):
            G += ep_r[i]
            rets[i] = G
        obss_ep.append(np.array(ep_obs, dtype=np.float32))
        rets_ep.append(rets)
    return np.concatenate(obss_ep), np.concatenate(rets_ep)


def critic_warmup(model: ActorCritic, obs_np, ret_np,
                  steps: int = C.CRITIC_WARMUP_STEPS, lr: float = 1e-3,
                  batch_size: int = 2048):
    """Freeze actor, train critic on Huber loss vs Monte Carlo returns."""
    obs_t = torch.from_numpy(obs_np).to(DEVICE)
    ret_t = torch.from_numpy(ret_np).to(DEVICE)
    opt = torch.optim.Adam(model.critic.parameters(), lr=lr)
    n = obs_t.shape[0]
    for step in range(steps):
        i = torch.randint(0, n, (min(batch_size, n),), device=DEVICE)
        v = model.value(obs_t[i])
        loss = F.smooth_l1_loss(v, ret_t[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 10 == 0 or step == steps - 1:
            with torch.no_grad():
                v_all = model.value(obs_t)
                r2 = 1.0 - ((v_all - ret_t).var() / max(ret_t.var().item(), 1e-9))
                print(f"  Critic warmup step {step:2d}: loss={loss.item():.5f} "
                      f"R^2={r2.item():+.3f} v_mean={v_all.mean().item():.3f} "
                      f"ret_mean={ret_t.mean().item():.3f}")
    with torch.no_grad():
        v_all = model.value(obs_t)
        r2 = 1.0 - ((v_all - ret_t).var() / max(ret_t.var().item(), 1e-9))
    return float(r2.item())


def actor_diagnostics(model: ActorCritic, label: str):
    """Print action mean/quantiles on 5 representative states."""
    # 5 representative states, all with c_d=1.5, various physical stages
    # State: (h/H, v/vref, q_ratio, q_head, fuel_frac, m_dry/m, tw, E_ratio,
    #         log_q, log_T, log_mf, cd, t_rem_frac, u_hint)
    reps = np.array([
        # Just launched, full fuel, low q (early window -> hint=1)
        [1.00, 0.01, 0.00, 1.00, 1.00, 0.33, 3.00, 1.00,  0.00, 0.00, 0.29, 1.5,  1.00, 1.00],
        # Ascent, q at 80% Q_max, half fuel
        [0.70, 0.90, 0.80, 0.20, 0.50, 0.50, 4.00, 0.60,  0.00, 0.00, 0.29, 1.5,  0.50, 0.60],
        # High altitude, low fuel, q low
        [0.20, 1.20, 0.05, 0.95, 0.10, 0.90, 8.00, 0.30,  0.00, 0.00, 0.29, 1.5,  0.10, 0.30],
        # Near-empty, coasting
        [0.10, 0.30, 0.00, 1.00, 0.02, 0.98, 9.00, 0.10,  0.00, 0.00, 0.29, 1.5, -0.20, 0.10],
        # Constraint saturated
        [0.85, 1.00, 0.99, 0.01, 0.80, 0.42, 3.50, 0.85,  0.00, 0.00, 0.29, 1.5,  0.30, 0.80],
    ], dtype=np.float32)
    with torch.no_grad():
        obs_t = torch.from_numpy(reps).to(DEVICE)
        d = model.dist(obs_t)
        mean = (d.concentration1 / (d.concentration1 + d.concentration0)).cpu().numpy()
        conc = (d.concentration1 + d.concentration0).cpu().numpy()
        samples = d.sample((256,)).cpu().numpy()  # [256, 5]
    print(f"\n-- Actor diagnostics ({label}) --")
    for i in range(5):
        q5, q50, q95 = np.quantile(samples[:, i], [0.05, 0.5, 0.95])
        print(f"  state {i}: mean={mean[i]:.3f} conc(a+b)={conc[i]:5.1f}  "
              f"sampled 5/50/95 = {q5:.3f}/{q50:.3f}/{q95:.3f}")


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    print("=" * 60)
    print("STEP 8 BC PRETRAIN + CRITIC WARMUP")
    print("=" * 60)

    model = ActorCritic(obs_dim=C.OBS_DIM).to(DEVICE)
    print(f"Model on {DEVICE}. Actor params: "
          f"{sum(p.numel() for p in model.actor.parameters())}, "
          f"Critic params: {sum(p.numel() for p in model.critic.parameters())}")

    actor_diagnostics(model, "pre-BC (random init)")

    print("\n-- Collecting BC data (heuristic rollouts, 200 eps) --")
    obs_np, u_np = collect_bc_data(n_episodes=200, seed=0)
    print(f"Collected {obs_np.shape[0]} (obs, u) pairs. u stats: "
          f"mean={u_np.mean():.3f} std={u_np.std():.3f} "
          f"min={u_np.min():.3f} max={u_np.max():.3f}")

    print("\n-- BC pretraining --")
    val_mse = bc_pretrain(model, obs_np, u_np)
    print(f"Final held-out MSE: {val_mse:.4f}")

    actor_diagnostics(model, "post-BC")

    print("\n-- Collecting returns for critic warmup (200 eps of BC policy) --")
    obs_c, ret_c = collect_returns_for_critic(model, n_episodes=200, seed=42)
    print(f"Collected {obs_c.shape[0]} (obs, return) pairs. "
          f"return stats: mean={ret_c.mean():.3f} std={ret_c.std():.3f}")

    print("\n-- Critic warmup --")
    r2 = critic_warmup(model, obs_c, ret_c)
    print(f"Final critic R^2: {r2:+.3f}")

    print("\n-- Checkpoint --")
    torch.save(model.state_dict(), "checkpoint_bc.pt")
    print("Saved checkpoint_bc.pt")


if __name__ == "__main__":
    main()
