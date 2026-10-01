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

"""Small helpers: seeding, GAE, batching."""
import numpy as np
import torch


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_gae(rewards, values, dones, last_value, gamma=1.0, lam=0.95):
    """rewards, values, dones: shape [T]. last_value: scalar bootstrap.
    Returns (advantages [T], returns [T]).
    dones[t] = 1 if step t is terminal (no bootstrap from t+1).
    """
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    lastgae = 0.0
    for t in reversed(range(T)):
        if t == T - 1:
            nextvalue = last_value
            nonterm = 1.0 - dones[t]
        else:
            nextvalue = values[t + 1]
            nonterm = 1.0 - dones[t]
        delta = rewards[t] + gamma * nextvalue * nonterm - values[t]
        lastgae = delta + gamma * lam * nonterm * lastgae
        adv[t] = lastgae
    ret = adv + np.asarray(values, dtype=np.float32)
    return adv, ret


def linear_anneal(start, end, frac):
    """frac in [0, 1]; returns start when frac=0, end when frac=1."""
    frac = float(np.clip(frac, 0.0, 1.0))
    return start + (end - start) * frac
