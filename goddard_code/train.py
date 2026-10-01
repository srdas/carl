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

"""Full training pipeline (Phases 1-4).

Phase 1: BC supervised pretrain (loaded from checkpoint_bc.pt if present, else run).
Phase 2: Critic warmup via value-only PPO updates.
Phase 3: Conservative PPO (small clip, low lr, low entropy) -- N_UPDATES_CONSERVATIVE.
Phase 4: Full PPO with LR + entropy annealed.

Everything writes progress to stdout (flushed each update) and to a JSON log.
Best model chosen by canonical selection score = eval apogee_frac_mean.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import torch
from agent import PPOAgent, DEVICE
from collector import VecCollector
from evaluate import evaluate_policy, evaluate_heuristic
from utils import set_seed, linear_anneal
import config as C


def maybe_flush():
    sys.stdout.flush(); sys.stderr.flush()


def phase_critic_warmup(agent: PPOAgent, collector: VecCollector, n_updates: int, log_prefix: str):
    """Value-only PPO updates: freeze actor, gather rollouts, run update(critic_only=True)."""
    print(f"\n{'='*60}\n{log_prefix} CRITIC WARMUP ({n_updates} updates, actor frozen)\n{'='*60}")
    maybe_flush()
    agent.freeze_actor()
    for u in range(n_updates):
        t0 = time.time()
        batch, stats = collector.collect(agent)
        losses = agent.update(batch, critic_only=True,
                              ppo_epochs=C.PPO_EPOCHS,
                              mini_batch_size=C.MINI_BATCH_SIZE)
        dt = time.time() - t0
        print(f"  [V{u:02d}] ep_ret={stats['ep_return_mean']:+.4f} "
              f"apogee_frac={stats['apogee_frac_mean']:.3f} "
              f"vloss={losses['value']:.5f} "
              f"clamps={stats['n_clamp_mean']:.1f} "
              f"ep_len={stats['ep_len_mean']:.0f} "
              f"({dt:.1f}s)")
        maybe_flush()
    agent.unfreeze_actor()


def phase_ppo(agent: PPOAgent, collector: VecCollector,
              n_updates: int, log_prefix: str,
              lr_start, lr_end, ent_start, ent_end, clip,
              eval_every, save_every,
              best_score, patience_counter, patience_limit,
              ckpt_dir, log_records, heuristic_baseline_score,
              update_offset=0, track_best=True):
    """Runs one PPO phase. Returns (best_score, patience_counter, latest_score, stopped_early).

    When track_best=False, evaluations still run and get logged, but best_model.pt is
    not updated and patience/early-stopping are disabled — used for the conservative
    phase so only the full-PPO phase determines the final checkpoint.
    """
    print(f"\n{'='*60}\n{log_prefix} PPO ({n_updates} updates, "
          f"lr {lr_start:.1e}->{lr_end:.1e}, ent {ent_start:.4f}->{ent_end:.4f}, clip={clip})\n{'='*60}")
    maybe_flush()

    stopped_early = False
    for u in range(n_updates):
        frac = u / max(n_updates - 1, 1)
        lr  = linear_anneal(lr_start, lr_end, frac)
        ent = linear_anneal(ent_start, ent_end, frac)
        agent.set_lr(lr)

        t0 = time.time()
        batch, stats = collector.collect(agent)
        losses = agent.update(batch, clip_eps=clip, ent_coef=ent,
                              ppo_epochs=C.PPO_EPOCHS,
                              mini_batch_size=C.MINI_BATCH_SIZE,
                              target_kl=0.05)
        dt = time.time() - t0
        gu = update_offset + u
        print(f"  [U{gu:03d}] ep_ret={stats['ep_return_mean']:+.4f} "
              f"apogee_frac={stats['apogee_frac_mean']:.3f} "
              f"pi={losses['policy']:+.4f} v={losses['value']:.5f} "
              f"H={losses['entropy']:+.3f} kl={losses['approx_kl']:+.4f} "
              f"clipfr={losses['clip_frac']:.2f} lr={lr:.1e} ent={ent:.4f} "
              f"clamps={stats['n_clamp_mean']:.1f} ({dt:.1f}s)")
        maybe_flush()

        record = {
            'update': int(gu), 'phase': log_prefix,
            'ep_return_mean': stats['ep_return_mean'],
            'apogee_frac_mean': stats['apogee_frac_mean'],
            'n_clamp_mean': stats['n_clamp_mean'],
            'ep_len_mean': stats['ep_len_mean'],
            'policy_loss': losses['policy'], 'value_loss': losses['value'],
            'entropy': losses['entropy'], 'approx_kl': losses['approx_kl'],
            'clip_frac': losses['clip_frac'], 'lr': lr, 'ent_coef': ent,
            'time_s': dt,
        }
        log_records.append(record)

        # ---- Intermediate eval ----
        if (u + 1) % eval_every == 0 or u == n_updates - 1:
            print(f"    [eval @ update {gu}]", end=" "); maybe_flush()
            summary, _ = evaluate_policy(agent, n_seeds_per_scenario=5, verbose=True)
            record['eval'] = summary
            log_records[-1] = record  # replace
            # Save latest
            agent.save(os.path.join(ckpt_dir, 'latest_model.pt'),
                       extra={'update': int(gu), 'eval': summary})
            # Save best (only tracked during the full PPO phase)
            score = summary['apogee_frac_mean']
            if track_best:
                if score > best_score:
                    best_score = score
                    patience_counter = 0
                    agent.save(os.path.join(ckpt_dir, 'best_model.pt'),
                               extra={'update': int(gu), 'eval': summary, 'score': score})
                    print(f"    -> NEW BEST canonical score: {score:.4f} "
                          f"(heuristic baseline: {heuristic_baseline_score:.4f})")
                else:
                    patience_counter += 1
                    print(f"    (best={best_score:.4f}, patience {patience_counter}/{patience_limit})")
            else:
                print(f"    (score={score:.4f}, not tracked for best in this phase)")
            # Degeneracy guard
            if summary['action_std_mean'] < 0.01:
                print("    !! action std < 0.01 across eval grid -- policy degenerate")
            if summary['action_range_mean'] < 0.1:
                print("    !! action range < 0.1 across eval grid -- exploration collapsed")
            maybe_flush()

            if track_best and patience_counter >= patience_limit:
                print(f"    Early stopping: no improvement in {patience_limit} evals.")
                stopped_early = True
                break

        # ---- Periodic checkpoint ----
        if (u + 1) % save_every == 0:
            agent.save(os.path.join(ckpt_dir, f'epoch_{gu:04d}.pt'),
                       extra={'update': int(gu)})

    return best_score, patience_counter, stopped_early


def run_bc_and_warmup_if_needed(agent, run_dir):
    """Run BC + critic warmup and save init_actor.pt for parity with long_train."""
    # We save this fresh so long_train can start from the same init.
    from bc_pretrain import (collect_bc_data, bc_pretrain, collect_returns_for_critic,
                             critic_warmup, actor_diagnostics)
    print("\n== Running BC pretraining fresh (for init_actor.pt parity) ==")
    actor_diagnostics(agent.model, "pre-BC")
    obs_np, u_np = collect_bc_data(n_episodes=200, seed=0)
    print(f"BC data: {obs_np.shape[0]} pairs. u mean={u_np.mean():.3f} std={u_np.std():.3f}")
    val_mse = bc_pretrain(agent.model, obs_np, u_np)
    print(f"BC final val MSE: {val_mse:.4f}")
    actor_diagnostics(agent.model, "post-BC")
    obs_c, ret_c = collect_returns_for_critic(agent.model, n_episodes=200, seed=42)
    r2 = critic_warmup(agent.model, obs_c, ret_c)
    print(f"Critic warmup final R^2: {r2:+.3f}")

    pretrain_path = "experiments/pretrain/init_actor.pt"
    os.makedirs(os.path.dirname(pretrain_path), exist_ok=True)
    agent.save(pretrain_path, extra={'phase': 'bc+critic_warmup',
                                     'bc_val_mse': val_mse, 'critic_r2': r2})
    print(f"Saved initial model: {pretrain_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--updates', type=int, default=C.N_UPDATES_INSESSION)
    ap.add_argument('--cons-updates', type=int, default=C.N_UPDATES_CONSERVATIVE)
    ap.add_argument('--critic-warmup', type=int, default=3,
                    help="value-only PPO warmup updates before policy gradient")
    ap.add_argument('--seed', type=int, default=C.SEED_TRAIN)
    ap.add_argument('--run-name', type=str, default='rocket_insession')
    ap.add_argument('--patience', type=int, default=C.EARLY_STOP_PATIENCE)
    ap.add_argument('--eval-every', type=int, default=C.EVAL_EVERY_UPDATES)
    ap.add_argument('--eval-eps', type=int, default=5,
                    help="seeds per scenario during intermediate eval")
    ap.add_argument('--save-every', type=int, default=C.SAVE_EVERY_UPDATES)
    ap.add_argument('--init-from', type=str, default='',
                    help="path to initial state_dict. Errors if missing when set.")
    ap.add_argument('--skip-bc', action='store_true',
                    help="do not run BC/critic warmup here (assume --init-from provides it)")
    args = ap.parse_args()

    set_seed(args.seed)
    run_dir = os.path.join("experiments", args.run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "train_log.json")

    print(f"== Training run: {args.run_name} ==")
    print(f"Device: {DEVICE}")
    print(f"Config: N_ENVS={C.N_ENVS} EP_PER_PARAM={C.EPISODES_PER_PARAM_SET} "
          f"BATCH_EPS={C.EPISODES_PER_UPDATE} MINI={C.MINI_BATCH_SIZE} "
          f"GAMMA={C.GAMMA} REWARD_MODE={C.REWARD_MODE}")
    print(f"Args: updates={args.updates} cons_updates={args.cons_updates} "
          f"warmup={args.critic_warmup} eval_every={args.eval_every} "
          f"patience={args.patience}")
    maybe_flush()

    agent = PPOAgent(obs_dim=C.OBS_DIM)

    # Heuristic baseline
    print("\n== Heuristic baseline (deterministic, fixed eval grid) ==")
    h_sum, _ = evaluate_heuristic(n_seeds_per_scenario=5)
    print(f"Heuristic: apogee_frac_mean={h_sum['apogee_frac_mean']:.4f} "
          f"med={h_sum['apogee_frac_med']:.4f}")
    heuristic_score = h_sum['apogee_frac_mean']
    maybe_flush()

    # Random-init baseline (before any training)
    print("\n== Random-init policy baseline ==")
    r_sum, _ = evaluate_policy(agent, n_seeds_per_scenario=3, verbose=True)
    maybe_flush()

    # Phase 1: BC + critic warmup (or --init-from)
    if args.init_from:
        if not os.path.exists(args.init_from):
            raise FileNotFoundError(f"--init-from set but path missing: {args.init_from}")
        print(f"\n== Loading initial model from {args.init_from} ==")
        agent.load(args.init_from)
    elif not args.skip_bc:
        # Fresh BC + critic warmup, and save init_actor.pt so long_train can reuse
        run_bc_and_warmup_if_needed(agent, run_dir)

    print("\n== Post-BC eval ==")
    post_bc_sum, _ = evaluate_policy(agent, n_seeds_per_scenario=5, verbose=True)
    maybe_flush()

    log_records = []
    log_records.append({'update': -1, 'phase': 'baseline',
                        'heuristic': h_sum, 'random_init': r_sum,
                        'post_bc': post_bc_sum})

    collector = VecCollector(n_envs=C.N_ENVS, episodes_per_param=C.EPISODES_PER_PARAM_SET,
                             seed=args.seed)

    # best_model tracks only the full-PPO phase. Seed with -inf so the first
    # full-PPO eval always writes the file; post-BC / conservative-phase evals
    # do not count.
    best_score = float('-inf')
    patience_counter = 0

    # Phase 2: Critic warmup with actor frozen (value-only PPO)
    if args.critic_warmup > 0:
        phase_critic_warmup(agent, collector, args.critic_warmup, "Phase 2:")

    # Phase 3: Conservative PPO — not counted toward best_model.
    if args.cons_updates > 0:
        _, _, _ = phase_ppo(
            agent, collector, args.cons_updates, "Phase 3 (conservative)",
            lr_start=C.CONS_LR, lr_end=C.CONS_LR,
            ent_start=C.CONS_ENT, ent_end=C.CONS_ENT,
            clip=C.CONS_CLIP,
            eval_every=args.eval_every, save_every=args.save_every,
            best_score=best_score, patience_counter=patience_counter,
            patience_limit=args.patience,
            ckpt_dir=ckpt_dir, log_records=log_records,
            heuristic_baseline_score=heuristic_score,
            update_offset=0, track_best=False)

    # Phase 4: Full PPO
    if args.updates > 0:
        best_score, patience_counter, _ = phase_ppo(
            agent, collector, args.updates, "Phase 4 (full)",
            lr_start=C.LR_INIT, lr_end=C.LR_FINAL,
            ent_start=C.ENT_INIT, ent_end=C.ENT_FINAL,
            clip=C.CLIP_EPS,
            eval_every=args.eval_every, save_every=args.save_every,
            best_score=best_score, patience_counter=patience_counter,
            patience_limit=args.patience,
            ckpt_dir=ckpt_dir, log_records=log_records,
            heuristic_baseline_score=heuristic_score,
            update_offset=args.cons_updates)

    # Final save
    agent.save(os.path.join(ckpt_dir, 'final_model.pt'),
               extra={'update': args.cons_updates + args.updates})
    with open(log_path, 'w') as f:
        json.dump(log_records, f, indent=1, default=float)
    print(f"\n== Training done. best={best_score:.4f} "
          f"heuristic={heuristic_score:.4f} log={log_path} ==")


if __name__ == "__main__":
    main()
