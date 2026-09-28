# Artifact-bundle reference implementation

A hand-written, working implementation of the `## Final Usable Artifact (MANDATORY)`
section of `carl_skills/09-launch/SKILL.md`, built against
`sessions/goddard_rocket_20260723` (run `rocket_long`) as verification item R2 of
`carl_v2.md`.

**This is a reference, not part of the CARL runtime.** Nothing imports it. It exists
because the skill text only *tells the model* what to build — the only way to know
the instructions are actionable is to follow them, and following them found four
gaps in the wording (recorded in `carl_v2.md` under R2).

| File | Role |
|---|---|
| `make_artifact.py` | Lives next to `train.py`; resolves the best checkpoint, copies runtime modules, writes `contract.json` + `MODEL_CARD.md`, copies the two templates in under their final names |
| `artifact_solve_template.py` | Copied into the bundle as `solve.py` — the stranger's entry point |
| `artifact_smoke_template.py` | Copied into the bundle as `smoke_test.py` — the blocking check |
| `contract.json` | The contract this produced, as a concrete example of the required fields |
| `MODEL_CARD.md` | The card this produced |

To reproduce, from a copy of that session's `code/` directory:

```bash
python make_artifact.py --run-name rocket_long
python artifact/smoke_test.py            # ~2 s, blocks shipping
python artifact/smoke_test.py --full     # ~7 min, recomputes the 45-episode metric
python artifact/solve.py --param m_fuel_0=3350 --param Q_max=22500 --param cd=1.35
```

Verified results: canonical objective reproduces to 1e-6 (14479.236563 m); the full
selection metric reproduces exactly (0.073541); the bundle produces identical numbers
after being copied elsewhere, including with a poisoned `config.py` on both `cwd` and
`PYTHONPATH`.

Two details here are specific to this run and should **not** be copied blindly into a
different problem: the avatar is the Beta **mean** (because that is what `train.py`
used to compute the selection metric — the mode-based `deterministic_action` scores
12× worse on this checkpoint), and `obs_dim` is derived from the actor's first-layer
weight because this checkpoint predates the `extra` contract block.
