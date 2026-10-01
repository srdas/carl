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

"""Pytest tests for environment and agent (Step 8)."""
import numpy as np
import torch
import pytest
from environment import RocketEnv, dyn_pressure, rho, sample_params
from network import ActorCritic
from heuristic import heuristic_action, t_liftoff
import config as C


# ---------- Env tests ----------

def test_reset_returns_valid():
    env = RocketEnv(seed=0)
    obs, info = env.reset()
    assert obs.shape == (12,)
    assert np.isfinite(obs).all()
    assert "params" in info and "h_ref" in info
    assert info["h_ref"] > 0


def test_step_5tuple():
    env = RocketEnv(seed=0); env.reset()
    obs, r, term, trunc, info = env.step(0.5)
    assert obs.shape == (12,)
    assert isinstance(r, float)
    assert isinstance(term, bool)
    assert isinstance(trunc, bool)
    assert isinstance(info, dict)
    assert np.isfinite(r)


def test_determinism_same_seed():
    env1 = RocketEnv(seed=123); env2 = RocketEnv(seed=123)
    o1, _ = env1.reset(); o2, _ = env2.reset()
    assert np.allclose(o1, o2)
    for u in [0.3, 0.6, 0.9, 0.1, 0.7]:
        s1, r1, _, _, _ = env1.step(u)
        s2, r2, _, _, _ = env2.step(u)
        assert np.allclose(s1, s2)
        assert r1 == r2


def test_action_clipping():
    env = RocketEnv(seed=0); env.reset()
    # Out-of-bounds actions should not crash
    env.step(-1.0)
    env.step(2.0)
    assert env.state is not None


def test_state_bounds_over_many_steps():
    env = RocketEnv(seed=0)
    for ep in range(20):
        env.reset()
        while not env.done:
            obs, r, _, _, _ = env.step(0.7)
            assert np.isfinite(obs).all()
            assert (obs >= -10.0).all() and (obs <= 10.0).all()
            assert np.isfinite(r)


def test_episode_terminates():
    env = RocketEnv(seed=0)
    for ep in range(20):
        env.reset()
        steps = 0
        while not env.done and steps < 3000:
            env.step(0.5)
            steps += 1
        assert env.done, f"Episode did not terminate in 3000 steps"


def test_reward_range_finite():
    env = RocketEnv(seed=0)
    rewards = []
    for _ in range(20):
        env.reset()
        while not env.done:
            _, r, _, _, _ = env.step(0.5)
            rewards.append(r)
    assert np.isfinite(rewards).all()
    assert max(rewards) < 1.0  # per-step altitude change fraction should be small


def test_max_q_constraint_respected():
    """With env-side clamp, q should never significantly exceed Q_max."""
    env = RocketEnv(seed=0)
    for _ in range(10):
        env.reset()
        qs = []
        while not env.done:
            _, _, _, _, info = env.step(1.0)  # full throttle, most aggressive
            qs.append(info["q_over_Qmax"])
        # small overshoot possible from Euler-approx clamp; tolerate <5%
        assert max(qs) <= 1.05, f"q/Q_max = {max(qs)} exceeds 1.05"


def test_no_fuel_no_thrust():
    """With no fuel, throttle should have no effect."""
    p = dict(m_fuel_0=300.0, T_max=2.0e5, cd=1.5, Q_max=50000.0)
    env = RocketEnv(params=p); env.reset()
    # Burn fuel until empty
    while env.m_fuel > 0 and not env.done:
        env.step(1.0)
    # Now: no fuel. Confirm mass no longer decreases.
    if not env.done:
        m_before = env.state[2]
        env.step(1.0)
        assert env.state[2] == m_before


def test_telescoping_reward_matches_apogee():
    p = dict(m_fuel_0=2000.0, T_max=2.0e5, cd=1.5, Q_max=30000.0)
    env = RocketEnv(params=p); env.reset()
    rews = []
    while not env.done:
        _, r, _, _, info = env.step(0.6)
        rews.append(r)
    import config as C
    if C.REWARD_MODE == "apogee_frac":
        assert abs(sum(rews) - info["apogee_fraction"]) < 1e-4
    else:
        assert abs(sum(rews) - info["h_apogee"] / C.H_SCALE_FIXED) < 1e-3


def test_heuristic_beats_zero():
    """Sanity: heuristic policy achieves nonzero apogee on nominal params."""
    p = dict(m_fuel_0=2000.0, T_max=2.0e5, cd=1.5, Q_max=30000.0)
    env = RocketEnv(params=p); env.reset()
    t = 0.0
    while not env.done:
        u = heuristic_action(t, env.params)
        env.step(min(u, env.u_cap_safe()))
        t += C.DT
    assert env.h_apogee > 100.0  # meters


# ---------- Agent tests ----------

def test_actor_forward_valid_action():
    torch.manual_seed(0)
    model = ActorCritic()
    env = RocketEnv(seed=0); obs, _ = env.reset()
    obs_t = torch.from_numpy(obs).unsqueeze(0)
    a, logp, v = model.act(obs_t)
    assert torch.isfinite(a).all()
    assert (a >= 0).all() and (a <= 1).all()
    assert torch.isfinite(logp).all()
    assert torch.isfinite(v).all()


def test_gradient_flow():
    torch.manual_seed(0)
    model = ActorCritic()
    obs = torch.randn(32, 12)
    d = model.dist(obs)
    loss = -d.entropy().mean()
    loss.backward()
    # Check gradients flowed
    for name, p in model.actor.named_parameters():
        assert p.grad is not None
        assert p.grad.abs().sum() > 0, f"Zero gradient at {name}"


def test_save_load_preserves_actions():
    torch.manual_seed(0)
    m1 = ActorCritic()
    obs = torch.randn(10, 12)
    with torch.no_grad():
        d1 = m1.dist(obs)
        mean1 = d1.mean
    torch.save(m1.state_dict(), "_tmp_ckpt.pt")
    m2 = ActorCritic()
    m2.load_state_dict(torch.load("_tmp_ckpt.pt"))
    with torch.no_grad():
        d2 = m2.dist(obs)
        mean2 = d2.mean
    assert torch.allclose(mean1, mean2)
    import os; os.remove("_tmp_ckpt.pt")


def test_param_sampling_respects_tw():
    rng = np.random.default_rng(0)
    for _ in range(100):
        p = sample_params(rng)
        tw = p['T_max'] / ((C.M_DRY + p['m_fuel_0']) * C.G)
        assert tw >= C.TW_MIN


def test_action_diversity_random_init():
    """Fresh model should produce non-constant, non-boundary-stuck actions."""
    torch.manual_seed(0)
    model = ActorCritic()
    obs = torch.randn(200, 12)
    with torch.no_grad():
        d = model.dist(obs)
        samples = d.sample()
    assert samples.std().item() > 1e-4
    assert (samples.max() - samples.min()).item() > 1e-3
