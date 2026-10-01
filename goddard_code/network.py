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

"""Actor-Critic network: separate MLPs for actor and critic (Step 7).
Actor -> Beta(alpha, beta), Critic -> scalar value.

Beta parameterization is controlled by config.BETA_MODE:
  * "unimodal" (default): alpha, beta = softplus(raw) + 1  -- Beta unimodal in (0,1),
    density -> 0 at boundaries. Nicer gradients for a smooth throttle policy.
  * "boundary": alpha, beta = softplus(raw) + eps          -- Beta can put mass
    at u=0 (bang-off) and u=1 (bang-on), which the physical policy may want.
    Set this in config if training plateaus below what a boundary-heavy policy
    would achieve.
"""
import torch
import torch.nn as nn
from torch.distributions import Beta
import config as C


def _mlp(in_dim, hidden, out_dim, final_gain):
    layers = []
    d = in_dim
    for h in hidden:
        lin = nn.Linear(d, h)
        nn.init.orthogonal_(lin.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(lin.bias)
        layers += [lin, nn.Tanh()]
        d = h
    out = nn.Linear(d, out_dim)
    nn.init.orthogonal_(out.weight, gain=final_gain)
    nn.init.zeros_(out.bias)
    layers.append(out)
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim=C.OBS_DIM, hidden=C.HIDDEN):
        super().__init__()
        self.actor = _mlp(obs_dim, hidden, 2, final_gain=0.01)
        self.critic = _mlp(obs_dim, hidden, 1, final_gain=1.0)
        # Beta mode toggle
        self.beta_mode = getattr(C, "BETA_MODE", "unimodal")
        self.beta_eps = getattr(C, "BETA_EPS", 1e-2)

    def _beta_params(self, raw):
        sp = torch.nn.functional.softplus(raw)
        if self.beta_mode == "boundary":
            # Allow alpha, beta < 1 so density can pile up at 0 and 1.
            ab = sp + self.beta_eps
        else:
            # Unimodal: alpha, beta >= 1 => single interior mode.
            ab = sp + 1.0
        return ab

    def dist(self, obs):
        raw = self.actor(obs)
        ab = self._beta_params(raw)
        return Beta(ab[..., 0], ab[..., 1])

    def value(self, obs):
        return self.critic(obs).squeeze(-1)

    def act(self, obs, deterministic=False):
        d = self.dist(obs)
        if deterministic:
            a = d.mean
        else:
            # rsample() only works for concentration > 1; sample() is safe for
            # boundary mode. Grad flow doesn't matter here (no-grad wrapper in agent).
            a = d.sample()
        # Numerical guard: log_prob at exact 0/1 is -inf. Clamp very slightly.
        a = a.clamp(1e-6, 1.0 - 1e-6)
        logp = d.log_prob(a)
        v = self.value(obs)
        return a, logp, v

    def evaluate(self, obs, act):
        d = self.dist(obs)
        act_c = act.clamp(1e-6, 1.0 - 1e-6)
        logp = d.log_prob(act_c)
        ent  = d.entropy()
        v    = self.value(obs)
        return logp, ent, v
