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

"""PPO agent wrapping ActorCritic. Methods: select_action, update, save, load,
freeze_actor, unfreeze_actor. `update(batch, critic_only=True)` for warmup."""
import numpy as np
import torch
import torch.nn.functional as F
from network import ActorCritic
import config as C


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PPOAgent:
    def __init__(self, obs_dim=C.OBS_DIM, hidden=C.HIDDEN, lr=C.LR_INIT):
        self.model = ActorCritic(obs_dim=obs_dim, hidden=hidden).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.lr = lr

    def select_action(self, obs, training=True):
        single = (obs.ndim == 1)
        if single:
            obs = obs[None]
        obs_t = torch.from_numpy(obs.astype(np.float32)).to(DEVICE)
        with torch.no_grad():
            a, logp, v = self.model.act(obs_t, deterministic=not training)
        a_np    = a.cpu().numpy()
        logp_np = logp.cpu().numpy()
        v_np    = v.cpu().numpy()
        if single:
            return a_np[0], float(logp_np[0]), float(v_np[0])
        return a_np, logp_np, v_np

    def value(self, obs):
        single = (obs.ndim == 1)
        if single:
            obs = obs[None]
        obs_t = torch.from_numpy(obs.astype(np.float32)).to(DEVICE)
        with torch.no_grad():
            v = self.model.value(obs_t).cpu().numpy()
        return float(v[0]) if single else v

    def set_lr(self, lr):
        self.lr = lr
        for g in self.optimizer.param_groups:
            g['lr'] = lr

    def freeze_actor(self):
        for p in self.model.actor.parameters():
            p.requires_grad_(False)

    def unfreeze_actor(self):
        for p in self.model.actor.parameters():
            p.requires_grad_(True)

    def update(self, batch, clip_eps=C.CLIP_EPS, ent_coef=C.ENT_INIT,
               ppo_epochs=C.PPO_EPOCHS, mini_batch_size=C.MINI_BATCH_SIZE,
               vf_coef=C.VF_COEF, max_grad_norm=C.MAX_GRAD_NORM,
               critic_only=False, normalize_adv=True,
               target_kl=None):
        """
        normalize_adv: divide by batch std after block-centering (helps when
        block-centering makes the raw scale near-zero).
        target_kl: if not None and approx_kl exceeds this in a mini-batch,
        early-stop the update.
        """
        obs_t   = torch.from_numpy(batch['obs']).to(DEVICE)
        act_t   = torch.from_numpy(batch['actions']).to(DEVICE)
        oldlp_t = torch.from_numpy(batch['old_logp']).to(DEVICE)
        adv_np  = batch['adv']
        if normalize_adv and not critic_only:
            std = adv_np.std()
            if std > 1e-8:
                adv_np = adv_np / std
        adv_t   = torch.from_numpy(adv_np.astype(np.float32)).to(DEVICE)
        ret_t   = torch.from_numpy(batch['ret']).to(DEVICE)

        N = obs_t.shape[0]
        idxs = np.arange(N)

        losses = {'policy': [], 'value': [], 'entropy': [], 'approx_kl': [],
                  'clip_frac': []}
        stopped_early = False
        for epoch in range(ppo_epochs):
            if stopped_early:
                break
            np.random.shuffle(idxs)
            for start in range(0, N, mini_batch_size):
                mb = idxs[start:start + mini_batch_size]
                mb_t = torch.from_numpy(mb).long().to(DEVICE)
                ob   = obs_t[mb_t]
                ac   = act_t[mb_t]
                oldlp= oldlp_t[mb_t]
                adv  = adv_t[mb_t]
                ret  = ret_t[mb_t]

                logp, ent, v = self.model.evaluate(ob, ac)
                if not critic_only:
                    ratio = torch.exp(logp - oldlp)
                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
                    policy_loss = -torch.min(surr1, surr2).mean()
                    ent_loss = -ent.mean()
                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - (logp - oldlp)).mean().item()
                        clip_frac = (torch.abs(ratio - 1) > clip_eps).float().mean().item()
                else:
                    policy_loss = torch.tensor(0.0, device=DEVICE)
                    ent_loss    = torch.tensor(0.0, device=DEVICE)
                    approx_kl   = 0.0
                    clip_frac   = 0.0
                value_loss = F.smooth_l1_loss(v, ret)

                loss = policy_loss + vf_coef * value_loss + ent_coef * ent_loss
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                self.optimizer.step()

                losses['policy'].append(float(policy_loss.detach().cpu()))
                losses['value'].append(float(value_loss.detach().cpu()))
                losses['entropy'].append(float(-ent_loss.detach().cpu()))
                losses['approx_kl'].append(approx_kl)
                losses['clip_frac'].append(clip_frac)

                if target_kl is not None and not critic_only and approx_kl > target_kl:
                    stopped_early = True
                    break

        return {k: float(np.mean(v)) if len(v) else 0.0 for k, v in losses.items()}

    def save(self, path, extra: dict | None = None):
        payload = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr': self.lr,
        }
        if extra is not None:
            payload.update(extra)
        torch.save(payload, path)

    def load(self, path, strict=True):
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        if isinstance(ckpt, dict) and 'model' in ckpt:
            self.model.load_state_dict(ckpt['model'], strict=strict)
            if 'optimizer' in ckpt:
                try:
                    self.optimizer.load_state_dict(ckpt['optimizer'])
                except Exception as e:
                    print(f"  [load] optimizer state skipped: {e}")
            if 'lr' in ckpt:
                self.set_lr(ckpt['lr'])
            return ckpt
        else:
            self.model.load_state_dict(ckpt, strict=strict)
            return {}
