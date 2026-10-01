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

"""Vectorized rollout / CRN-block collector for PPO.

Design:
  * CRN block = shared param set + shared env seed for `EPISODES_PER_PARAM_SET`
    twins. Env is deterministic given params + actions, so twins differ only
    in the actor's stochastic action.
  * Block-centered advantages: within each CRN block, subtract the block's
    mean advantage. Raw returns kept for the value target.
"""
from __future__ import annotations
import numpy as np
import torch
from environment import RocketEnv, sample_params
from utils import compute_gae
import config as C


class VecCollector:
    def __init__(self, n_envs=C.N_ENVS, episodes_per_param=C.EPISODES_PER_PARAM_SET,
                 seed=C.SEED_TRAIN):
        self.n_envs = n_envs
        self.ep_per = episodes_per_param
        self.master_rng = np.random.default_rng(seed)
        assert n_envs % episodes_per_param == 0, \
            f"n_envs ({n_envs}) must be divisible by ep_per ({episodes_per_param})"
        self.n_blocks = n_envs // episodes_per_param
        self.envs = [RocketEnv(seed=0) for _ in range(n_envs)]

    def _reset_all(self):
        obs_list = []
        for b in range(self.n_blocks):
            params = sample_params(np.random.default_rng(int(self.master_rng.integers(0, 2**31 - 1))))
            block_seed = int(self.master_rng.integers(0, 2**31 - 1))
            for k in range(self.ep_per):
                idx = b * self.ep_per + k
                self.envs[idx]._fixed_params = dict(params)
                obs, _ = self.envs[idx].reset(seed=block_seed + k)  # slight twin variation
                obs_list.append(obs)
        return np.stack(obs_list)

    def collect(self, agent, max_steps=2000):
        obs_np = self._reset_all()
        N = self.n_envs

        buf_obs = [[] for _ in range(N)]
        buf_act = [[] for _ in range(N)]
        buf_logp = [[] for _ in range(N)]
        buf_val = [[] for _ in range(N)]
        buf_rew = [[] for _ in range(N)]
        buf_done = [[] for _ in range(N)]

        active = np.ones(N, dtype=bool)
        ep_info = [None] * N

        for step in range(max_steps):
            if not active.any():
                break
            act_idx = np.where(active)[0]
            obs_batch = obs_np[act_idx]
            a, logp, v = agent.select_action(obs_batch, training=True)
            # a: shape [B] (scalar action per env)

            u_caps = np.array([self.envs[i].u_cap_safe() for i in act_idx],
                              dtype=np.float32)
            u_apply = np.minimum(a, u_caps)

            for j, i in enumerate(act_idx):
                buf_obs[i].append(obs_batch[j].copy())
                buf_act[i].append(float(a[j]))
                buf_logp[i].append(float(logp[j]))
                buf_val[i].append(float(v[j]))

                obs_next, r, term, trunc, info = self.envs[i].step(float(u_apply[j]))
                buf_rew[i].append(float(r))
                done = term or trunc
                buf_done[i].append(1.0 if done else 0.0)
                if done:
                    active[i] = False
                    ep_info[i] = info
                else:
                    obs_np[i] = obs_next

        # Assemble
        all_obs, all_act, all_logp, all_val, all_ret, all_adv = [], [], [], [], [], []
        block_ids = []
        ep_returns, ep_lens, ep_apogee_frac, ep_n_clamp = [], [], [], []
        for i in range(N):
            T = len(buf_rew[i])
            if T == 0:
                continue
            values = np.asarray(buf_val[i], dtype=np.float32)
            rewards = np.asarray(buf_rew[i], dtype=np.float32)
            dones = np.asarray(buf_done[i], dtype=np.float32)
            adv, ret = compute_gae(rewards, values, dones, last_value=0.0,
                                    gamma=C.GAMMA, lam=C.GAE_LAMBDA)
            all_obs.append(np.asarray(buf_obs[i], dtype=np.float32))
            all_act.append(np.asarray(buf_act[i], dtype=np.float32))
            all_logp.append(np.asarray(buf_logp[i], dtype=np.float32))
            all_val.append(values)
            all_adv.append(adv)
            all_ret.append(ret)
            block_ids.append(np.full(T, i // self.ep_per, dtype=np.int64))
            ep_returns.append(float(rewards.sum()))
            ep_lens.append(int(T))
            if ep_info[i] is not None and 'apogee_fraction' in ep_info[i]:
                ep_apogee_frac.append(float(ep_info[i]['apogee_fraction']))
                ep_n_clamp.append(int(ep_info[i]['n_clamp_steps']))

        obs   = np.concatenate(all_obs, axis=0)
        act   = np.concatenate(all_act, axis=0)
        logp0 = np.concatenate(all_logp, axis=0)
        adv   = np.concatenate(all_adv, axis=0)
        ret   = np.concatenate(all_ret, axis=0)
        blk   = np.concatenate(block_ids, axis=0)

        adv_centered = adv.copy()
        for b in np.unique(blk):
            m = (blk == b)
            adv_centered[m] = adv[m] - adv[m].mean()

        batch = {
            'obs': obs, 'actions': act, 'old_logp': logp0,
            'adv': adv_centered, 'ret': ret,
        }
        stats = {
            'ep_return_mean': float(np.mean(ep_returns)),
            'ep_return_std':  float(np.std(ep_returns)),
            'ep_len_mean':    float(np.mean(ep_lens)),
            'apogee_frac_mean': float(np.mean(ep_apogee_frac)) if ep_apogee_frac else float('nan'),
            'apogee_frac_std':  float(np.std(ep_apogee_frac))  if ep_apogee_frac else float('nan'),
            'n_clamp_mean':   float(np.mean(ep_n_clamp)) if ep_n_clamp else 0.0,
            'n_episodes':     len(ep_returns),
            'n_transitions':  int(obs.shape[0]),
        }
        return batch, stats
