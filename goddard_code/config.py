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

"""Global config: physical constants, randomization, PPO hyperparameters.

Revisit 1 changes vs Step 9:
  * OBS_DIM 12 -> 14 (added t_remaining_frac, u_constant_hint)
  * New one-sided early-burnout penalty (fixed reward constant, no per-episode
    scaling). See BURN_TARGET_MULT, PENALTY_COEF, HINT_EARLY_FRAC below.
  * Reward normalization unchanged: raw_dh / H_SCALE_FIXED.
"""
import numpy as np

# --- Physics constants (from Steps 1-2) ---
G = 9.81                # m/s^2
M_DRY = 1000.0          # kg
RHO0 = 1.225            # kg/m^3 sea-level density
H_SCALE = 8500.0        # m atmospheric scale height
CE_RATIO = 0.01         # c_e = CE_RATIO * T_max
DT = 0.1                # s integrator step
CLAMP_DRAG_PENALTY = 0.05  # 5% drag increase when clamped

# --- Randomization ranges (Step 3) ---
M_FUEL_RANGE = (300.0, 5000.0)     # log-uniform, kg
T_MAX_RANGE  = (1.0e5, 3.0e5)      # log-uniform, N
CD_RANGE     = (1.0, 2.0)          # uniform
Q_MAX_RANGE  = (1.0e4, 5.0e4)      # log-uniform, Pa
TW_MIN       = 1.1                 # rejection threshold

# --- Eval grid (Step 1) ---
EVAL_T_MAX = 2.0e5
EVAL_CD    = 1.5
EVAL_M_FUEL_GRID = [500.0, 2000.0, 4000.0]
EVAL_Q_MAX_GRID  = [1.5e4, 3.0e4, 4.5e4]

# --- Heuristic (Step 5) ---
V_LIFTOFF = 30.0
DECAY_FLOOR = 0.2
DECAY_INIT  = 0.8

# --- PPO / network (Step 7) ---
HIDDEN = (256, 256)
GAMMA = 1.0
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
PPO_EPOCHS = 2
LR_INIT = 4e-4
LR_FINAL = 2e-4
ENT_INIT = 0.02
ENT_FINAL = 0.005
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
MINI_BATCH_SIZE = 8192

# Parallelism / batch
N_ENVS = 512
EPISODES_PER_PARAM_SET = 8             # CRN block size (64 blocks x 8 twins)
EPISODES_PER_UPDATE = N_ENVS           # 512

# --- Seeds ---
SEED_TRAIN = 0
SEED_EVAL = 1000

# --- Reward normalization ---
REWARD_MODE = "raw_dh"
H_SCALE_FIXED = 1000.0

# --- BC pretraining (Step 8) ---
BC_STEPS = 500
BC_BATCH_SIZE = 4096
BC_LR = 1e-3
CRITIC_WARMUP_STEPS = 200

# --- Training schedule ---
N_UPDATES_INSESSION = 60
N_UPDATES_CONSERVATIVE = 10
EVAL_EVERY_UPDATES = 10
EVAL_EPISODES = 45
SAVE_EVERY_UPDATES = 20
EARLY_STOP_PATIENCE = 12

CONS_LR = 1.0e-4
CONS_CLIP = 0.1
CONS_ENT = 0.01

# --- Beta distribution parameterization ---
BETA_MODE = "boundary"
BETA_EPS  = 0.01

# ==============================================================
# Revisit 1: burnout-time shaping and new state features
# ==============================================================

# OBS_DIM was 12 in Step 9; now 14 (added t_remaining_frac, u_constant_hint).
OBS_DIM = 14

# Target burnout time = BURN_TARGET_MULT * full_burn_time,
# where full_burn_time = m_fuel_0 * c_e / T_max (unconstrained bang-bang duration).
BURN_TARGET_MULT = 4.0

# One-sided penalty applied ONCE at the step where fuel first depletes:
#     deficit = max(0, (t_target - t_burnout) / t_target)   in [0, 1]
#     penalty = - PENALTY_COEF * deficit^2                  (fixed constant, raw_dh units)
#
# Calibration (see burnout_sanity.py):
# heuristic rollouts across the training distribution give raw_dh returns of
# roughly 3 (light) - 8 (nominal) - up to 30-50 (heavy, under a competent
# policy). Choosing PENALTY_COEF = 20 makes the half-target burnout
# (deficit=0.5) cost 0.25 * 20 = 5 in raw_dh units, which is ~half the
# nominal-bin median return (target of "halve the reward at t_target/2")
# and a meaningful but non-destructive fraction of the heavy-bin achievable
# return.
PENALTY_COEF = 20.0

# u_constant_hint: for t < HINT_EARLY_FRAC * t_target the hint is forced to 1.0
# so BC/PPO do not "forget" to boost hard at the start.
HINT_EARLY_FRAC = 0.05
