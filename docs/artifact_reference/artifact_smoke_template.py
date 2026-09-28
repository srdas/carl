#!/usr/bin/env python
"""smoke_test.py -- does the shipped bundle still reproduce its recorded number?

    python smoke_test.py           # canonical instance, ~2 s
    python smoke_test.py --full    # + the 9x5 eval grid, ~2 min

A failure blocks shipping. Exit 0 = pass, 1 = fail.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

from solve import build_agent, load_contract, solve_once, act  # noqa: E402

REL_TOL = 1e-6      # same code, same seed: bitwise-ish, so keep this tight
GRID_TOL = 5e-3     # grid metric is a 45-episode mean; allow a little drift


def _check(label: str, got, want, tol) -> bool:
    ok = abs(got - want) <= tol * max(abs(want), 1e-12)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: got {got:.6f}, "
          f"expected {want:.6f} (rel tol {tol:g})")
    return ok


def grid_metric(agent, contract) -> float:
    """Recompute the selection metric: eval grid x 5 seeds, mean apogee_fraction."""
    import config as C
    fracs = []
    for si, mf in enumerate(C.EVAL_M_FUEL_GRID):
        for qi, qm in enumerate(C.EVAL_Q_MAX_GRID):
            params = {"m_fuel_0": mf, "T_max": C.EVAL_T_MAX,
                      "cd": C.EVAL_CD, "Q_max": qm}
            idx = si * len(C.EVAL_Q_MAX_GRID) + qi
            for s in range(5):
                r = solve_once(agent, contract, params,
                               C.SEED_EVAL + idx * 100 + s)
                fracs.append(r["apogee_fraction"])
    return float(np.mean(fracs))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", action="store_true",
                    help="also recompute the 45-episode selection metric")
    args = ap.parse_args()

    contract = load_contract()
    print(f"Bundle from run {contract['created_from']['run_name']}, "
          f"update {contract['created_from']['update']}, "
          f"avatar {contract['avatar']}")

    # Self-containment: nothing in the bundle may resolve outside it.
    leaked = [m.__name__ for m in
              (sys.modules.get(n) for n in ("config", "agent", "environment",
                                            "network", "policy"))
              if m is not None and getattr(m, "__file__", "")
              and not os.path.abspath(m.__file__).startswith(HERE)]
    if leaked:
        print(f"  [FAIL] imports resolved outside the bundle: {leaked}")
        return 1
    print("  [PASS] every runtime import resolved inside the bundle")

    agent = build_agent(contract)
    canon = contract["canonical_instance"]
    r = solve_once(agent, contract, canon["params"], canon["seed"])

    ok = _check("canonical objective (h_apogee, m)",
                r["objective"], canon["expected_objective"], REL_TOL)
    ok &= _check("canonical apogee_fraction",
                 r["apogee_fraction"], canon["expected_apogee_fraction"], REL_TOL)

    if r["max_q_over_Qmax"] > 1.0 + 1e-9:
        print(f"  [FAIL] max-Q constraint breached: "
              f"q/Q_max = {r['max_q_over_Qmax']:.6f}")
        ok = False
    else:
        print(f"  [PASS] max-Q constraint held "
              f"(peak q/Q_max = {r['max_q_over_Qmax']:.4f})")

    if args.full:
        want = contract["selection_metric"]["value"]
        print("  recomputing the 45-episode selection metric...")
        got = grid_metric(agent, contract)
        if want is None:
            # A bundle built from latest_model.pt has no recorded score --
            # train.py only writes `score` when tracking best. Report the
            # number instead of comparing against nothing.
            print(f"  [SKIP] selection metric: no value recorded in the "
                  f"checkpoint; recomputed {got:.6f} for reference")
        else:
            ok &= _check("selection metric "
                         f"({contract['selection_metric']['name']})",
                         got, want, GRID_TOL)

    print("SMOKE TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
