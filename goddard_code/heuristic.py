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

"""BC heuristic: liftoff-full-thrust then exponential decay (Step 5)."""
import numpy as np
import config as C


def t_liftoff(params):
    """t_liftoff = (m_dry + m_fuel_0) * v_liftoff / (T_max - (m_dry + m_fuel_0)*g)."""
    m0 = C.M_DRY + params['m_fuel_0']
    denom = params['T_max'] - m0 * C.G
    if denom <= 0:
        return 1e6
    return m0 * C.V_LIFTOFF / denom


def heuristic_action(t_now: float, params: dict) -> float:
    """u(t) = 1 for t < t_liftoff, else 0.2 + 0.8 * 2^{-(t - t_liftoff)/t_half}."""
    tl = t_liftoff(params)
    th = tl  # t_half = t_liftoff (Step 5)
    if t_now < tl:
        return 1.0
    x = (t_now - tl) / max(th, 1e-9)
    return C.DECAY_FLOOR + C.DECAY_INIT * (0.5 ** x)


def rollout_heuristic(env, apply_actor_cap: bool = True):
    """Run one episode with heuristic action, return trajectory dicts."""
    obs, info = env.reset()
    obss, acts, rews = [obs], [], []
    infos = [info]
    t = 0.0
    while not env.done:
        u = heuristic_action(t, env.params)
        if apply_actor_cap:
            u = min(u, env.u_cap_safe())
        obs, r, term, trunc, info = env.step(u)
        obss.append(obs); acts.append(u); rews.append(r); infos.append(info)
        t += C.DT
    return obss, acts, rews, infos
