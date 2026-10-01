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

"""Goddard rocket env with hard max-Q constraint (Revisit 1).

Changes vs Step 9:
  * State grew from 12 to 14 dims: added t_remaining_frac and u_constant_hint.
  * One-sided early-burnout penalty applied ONCE at the fuel-out step:
        deficit = max(0, (t_target - t_burnout) / t_target)
        penalty = -PENALTY_COEF * deficit^2       (fixed constant, raw_dh units)
    where t_target = BURN_TARGET_MULT * m_fuel_0*c_e/T_max.
"""
from __future__ import annotations
import numpy as np
import config as C


# ---------- Physics helpers ----------

def rho(h):
    return C.RHO0 * np.exp(-h / C.H_SCALE)


def dyn_pressure(h, v):
    return 0.5 * rho(h) * v * v


def _deriv(state, thrust_force, cd, m_dry):
    h, v, m = state
    r = rho(h)
    drag = cd * r * v * abs(v)
    a = (thrust_force - drag) / m - C.G
    return np.array([v, a, 0.0])


def rk4_step(state, thrust_force, cd, m_dry, dt):
    k1 = _deriv(state,                     thrust_force, cd, m_dry)
    k2 = _deriv(state + 0.5*dt*k1,         thrust_force, cd, m_dry)
    k3 = _deriv(state + 0.5*dt*k2,         thrust_force, cd, m_dry)
    k4 = _deriv(state +     dt*k3,         thrust_force, cd, m_dry)
    return state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)


def h_ref_ballistic(m_fuel_0, T_max):
    c_e = C.CE_RATIO * T_max
    dv  = c_e * np.log((C.M_DRY + m_fuel_0) / C.M_DRY)
    return dv * dv / (2.0 * C.G)


def full_burn_time(m_fuel_0, T_max):
    """Unconstrained bang-bang burn duration: mdot = T_max/c_e, so t = m_fuel_0/mdot
    = m_fuel_0 * c_e / T_max."""
    c_e = C.CE_RATIO * T_max
    return m_fuel_0 * c_e / T_max


def burn_target_time(m_fuel_0, T_max):
    x = m_fuel_0 / 1000.0
    return 75.0 * x - 8.0 * x * x


# ---------- Parameter sampling ----------

def sample_params(rng: np.random.Generator):
    for _ in range(1000):
        m_fuel_0 = float(np.exp(rng.uniform(np.log(C.M_FUEL_RANGE[0]), np.log(C.M_FUEL_RANGE[1]))))
        T_max    = float(np.exp(rng.uniform(np.log(C.T_MAX_RANGE[0]),  np.log(C.T_MAX_RANGE[1]))))
        cd       = float(rng.uniform(*C.CD_RANGE))
        Q_max    = float(np.exp(rng.uniform(np.log(C.Q_MAX_RANGE[0]),  np.log(C.Q_MAX_RANGE[1]))))
        if T_max / ((C.M_DRY + m_fuel_0) * C.G) >= C.TW_MIN:
            return dict(m_fuel_0=m_fuel_0, T_max=T_max, cd=cd, Q_max=Q_max)
    raise RuntimeError("Parameter rejection sampling failed")


# ---------- Environment ----------

class RocketEnv:
    """
    Obs: 14-D dimensionless state (Revisit 1).
    Action: scalar u in [0,1].
    Reward: (h_{t+1} - h_t) / H_SCALE_FIXED plus one-shot early-burnout penalty
            at the fuel-out step (see module docstring).
    """
    OBS_DIM = C.OBS_DIM   # 14
    ACT_DIM = 1

    def __init__(self, seed: int | None = None, params: dict | None = None):
        self.rng = np.random.default_rng(seed)
        self._fixed_params = params
        self.params = None
        self.state = None
        self.m_fuel = 0.0
        self.launched = False
        self.done = False
        self.h_apogee = 0.0
        self.n_steps = 0
        self.n_clamp = 0
        self.t_now = 0.0
        self.t_target = 0.0
        self.burnout_penalty_applied = False
        self.burnout_penalty = 0.0
        self.burnout_time = None

    # ---------- feature vector ----------
    def _obs(self):
        h, v, m = self.state
        p = self.params
        r_ratio = rho(h) / C.RHO0
        v_ref = np.sqrt(2 * p['Q_max'] / C.RHO0)
        v_n = v / v_ref
        q = dyn_pressure(h, v)
        q_ratio = q / p['Q_max']
        q_head = 1.0 - q_ratio
        fuel_frac = self.m_fuel / p['m_fuel_0']
        dry_over_m = C.M_DRY / m
        tw = p['T_max'] / (m * C.G)
        c_e = C.CE_RATIO * p['T_max']
        E_alt = h + v*v/(2*C.G) + (c_e/C.G) * np.log(max(m/C.M_DRY, 1e-12))
        E0    = (c_e/C.G) * np.log((C.M_DRY + p['m_fuel_0'])/C.M_DRY)
        E_ratio = E_alt / max(E0, 1e-9)
        log_q  = np.log(p['Q_max']  / 30000.0)
        log_T  = np.log(p['T_max']  / 200000.0)
        log_mf = np.log(p['m_fuel_0']/ 1500.0)
        cd     = p['cd']

        # ---- Revisit 1 new features ----
        # (a) time-remaining fraction toward the target burnout time.
        # Positive while behind target, negative once past it. Clipped [-1, 1].
        t_rem_frac = (self.t_target - self.t_now) / max(self.t_target, 1e-9)
        t_rem_frac = float(np.clip(t_rem_frac, -1.0, 1.0))

        # (b) constant-throttle hint: the u that would consume remaining fuel
        # exactly by t_target, given current T_max. mdot_const = m_fuel / dt_rem;
        # u_const = mdot_const / (T_max / c_e) = m_fuel * c_e / (T_max * dt_rem).
        # For the first HINT_EARLY_FRAC of t_target, force hint = 1 (early-boost
        # signal). Once at/past t_target or out of fuel, hint = 0.
        early_window = C.HINT_EARLY_FRAC * self.t_target
        if self.m_fuel <= 0.0:
            u_hint = 0.0
        elif self.t_now < early_window:
            u_hint = 1.0
        else:
            dt_rem = max(self.t_target - self.t_now, 1e-6)
            u_hint = self.m_fuel * c_e / (p['T_max'] * dt_rem)
            u_hint = float(np.clip(u_hint, 0.0, 1.0))

        obs = np.array([r_ratio, v_n, q_ratio, q_head,
                        fuel_frac, dry_over_m, tw,
                        E_ratio,
                        log_q, log_T, log_mf, cd,
                        t_rem_frac, u_hint], dtype=np.float32)
        return np.clip(obs, -10.0, 10.0)

    # ---------- max-Q cap (actor-side helper) ----------
    def u_cap_safe(self):
        p = self.params
        h, v, m = self.state
        Q = p['Q_max']
        r = rho(h)
        v_lim = np.sqrt(2 * Q / max(r, 1e-9))
        dv_allowed = v_lim - v
        if dv_allowed <= 0:
            return 0.0
        drag = p['cd'] * r * v * abs(v)
        T_needed = m * (dv_allowed / C.DT + C.G) + drag
        if T_needed <= 0:
            return 0.0
        u = T_needed / p['T_max']
        return float(np.clip(0.99 * u, 0.0, 1.0))

    # ---------- gym-like API ----------
    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.params = self._fixed_params if self._fixed_params is not None else sample_params(self.rng)
        self.m_fuel = self.params['m_fuel_0']
        m0 = C.M_DRY + self.m_fuel
        self.state = np.array([0.0, 0.0, m0])
        self.launched = False
        self.done = False
        self.h_apogee = 0.0
        self.n_steps = 0
        self.n_clamp = 0
        self.t_now = 0.0
        self.t_target = burn_target_time(self.params['m_fuel_0'], self.params['T_max'])
        self.burnout_penalty_applied = False
        self.burnout_penalty = 0.0
        self.burnout_time = None
        self.h_ref = h_ref_ballistic(self.params['m_fuel_0'], self.params['T_max'])
        return self._obs(), {"params": dict(self.params), "h_ref": self.h_ref,
                             "t_target": self.t_target,
                             "full_burn_time": full_burn_time(self.params['m_fuel_0'], self.params['T_max'])}

    def _env_clamp_thrust(self, u_cmd):
        u_cap = self.u_cap_safe() / 0.99
        clamped = False
        u = u_cmd
        if u > u_cap:
            u = u_cap
            clamped = True
        return float(np.clip(u, 0.0, 1.0)), clamped

    def step(self, u_cmd: float):
        assert not self.done
        u_cmd = float(np.clip(u_cmd, 0.0, 1.0))
        p = self.params
        h, v, m = self.state

        u_applied, clamped = self._env_clamp_thrust(u_cmd)
        if clamped:
            self.n_clamp += 1

        fuel_before = self.m_fuel
        if self.m_fuel <= 0.0:
            u_applied = 0.0

        T_force = u_applied * p['T_max']
        cd_eff = p['cd'] * (1.0 + C.CLAMP_DRAG_PENALTY) if clamped else p['cd']

        new_hv_m = rk4_step(self.state, T_force, cd_eff, C.M_DRY, C.DT)
        c_e = C.CE_RATIO * p['T_max']
        mdot = T_force / c_e
        fuel_burned = mdot * C.DT
        fuel_burned = min(fuel_burned, self.m_fuel)
        self.m_fuel -= fuel_burned
        new_m = C.M_DRY + self.m_fuel

        new_h = max(new_hv_m[0], 0.0)
        new_v = new_hv_m[1]
        if not self.launched and new_v > 0.0:
            self.launched = True
        self.state = np.array([new_h, new_v, new_m])
        self.n_steps += 1
        self.t_now += C.DT

        terminated = False
        if self.launched and new_v <= 0.0:
            terminated = True
        if self.launched and new_h <= 0.0 and self.n_steps > 5:
            terminated = True

        dh = new_h - h
        if C.REWARD_MODE == "raw_dh":
            reward = dh / C.H_SCALE_FIXED
        else:
            reward = dh / max(self.h_ref, 1e-9)

        # -------- Revisit 1: one-shot early-burnout penalty (fixed constant) --------
        penalty = 0.0
        if (not self.burnout_penalty_applied) and (fuel_before > 0.0) and (self.m_fuel <= 0.0):
            deficit = max(0.0, self.t_target - self.t_now) / max(self.t_target, 1e-9)
            deficit = float(np.clip(deficit, 0.0, 1.0))
            penalty = -C.PENALTY_COEF * (deficit * deficit)
            self.burnout_penalty_applied = True
            self.burnout_penalty = penalty
            self.burnout_time = self.t_now
            reward += penalty

        if new_h > self.h_apogee:
            self.h_apogee = new_h

        truncated = self.n_steps >= 2000

        info = {
            "h": float(new_h), "v": float(new_v), "m": float(new_m),
            "q": float(dyn_pressure(new_h, new_v)),
            "q_over_Qmax": float(dyn_pressure(new_h, new_v) / p['Q_max']),
            "throttle_commanded": float(u_cmd),
            "throttle_applied": float(u_applied),
            "clamp_active": bool(clamped),
            "reward_step": float(reward),
            "burnout_penalty_step": float(penalty),
        }
        if terminated or truncated:
            info.update({
                "h_apogee": float(self.h_apogee),
                "h_ref": float(self.h_ref),
                "apogee_fraction": float(self.h_apogee / max(self.h_ref, 1e-9)),
                "n_clamp_steps": int(self.n_clamp),
                "n_steps": int(self.n_steps),
                "t_target": float(self.t_target),
                "burnout_time": float(self.burnout_time) if self.burnout_time is not None else -1.0,
                "burnout_penalty_total": float(self.burnout_penalty),
                **{k: float(v) for k, v in p.items()},
            })

        self.done = terminated or truncated
        return self._obs(), float(reward), bool(terminated), bool(truncated), info
