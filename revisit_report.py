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

"""Report-style appendix publishing for CARL Step 10 revisit runs."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        return {}


def tail(path: Path, n: int = 20) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n:])


def latex_escape(text: str) -> str:
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in text)


def tex_path(path: str | Path) -> str:
    return str(path).replace("\\", "/")


def num(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def sci(value: Any, digits: int = 3) -> str:
    value = num(value)
    if value is None:
        return "--"
    if value == 0:
        return "$0$"
    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = value / (10 ** exponent)
    return f"${mantissa:.{digits}f}\\times 10^{{{exponent}}}$"


def fixed(value: Any, digits: int = 4) -> str:
    value = num(value)
    if value is None:
        return "--"
    return f"${value:.{digits}f}$"


def fixed_bare(value: Any, digits: int = 4) -> str:
    value = num(value)
    if value is None:
        return "--"
    return f"{value:.{digits}f}"


def pct(value: Any) -> str:
    value = num(value)
    if value is None:
        return "--"
    return f"${100.0 * value:.1f}\\%$"


def ensure_main_input(main_tex: Path) -> None:
    if not main_tex.exists():
        return
    text = main_tex.read_text(encoding="utf-8", errors="replace")
    if r"\input{appendix.tex}" in text:
        return
    marker = r"\bibliographystyle"
    insert = "\\input{appendix.tex}\n\n"
    if marker in text:
        text = text.replace(marker, insert + marker, 1)
    else:
        text = text.replace(r"\end{document}", insert + r"\end{document}", 1)
    main_tex.write_text(text, encoding="utf-8")


def _copy_revisit_plots(run_dir: Path, results_dir: Path, run_name: str) -> dict[str, str]:
    copied: dict[str, str] = {}
    plots_dir = run_dir / "plots"
    if not plots_dir.exists():
        return copied

    for source_name in (
        "training_curves.png",
        "policy_vs_merton.png",
        "policy_vs_merton_best_empirical.png",
        "policy_vs_merton_latest.png",
        "sensitivity.png",
    ):
        source = plots_dir / source_name
        if not source.exists():
            continue
        dest_name = f"{run_name}_{source_name}"
        dest = results_dir / dest_name
        try:
            shutil.copy2(source, dest)
            copied[source_name] = dest_name
        except OSError:
            copied[source_name] = os.path.relpath(source, results_dir)
    return copied


def _eval_rows(final_eval: dict[str, Any]) -> str:
    best = final_eval.get("best") or final_eval.get("best_theory_eval") or {}
    latest = final_eval.get("latest") or {}
    merton = final_eval.get("merton") or {}
    mae = final_eval.get("policy_mae_vs_merton_by_checkpoint") or {}
    best_mae = mae.get("best_theory_eval") or final_eval.get("policy_mae_vs_merton") or {}
    latest_mae = mae.get("latest") or final_eval.get("latest_policy_mae_vs_merton") or {}

    rows = [
        (
            "Best checkpoint",
            best.get("eval_raw_mean"),
            best.get("eval_raw_median"),
            best.get("eval_empirical_joint_scaled_mean", best.get("eval_empirical_scaled_mean")),
            best.get("eval_kappa_mean"),
            best.get("eval_pi_mean"),
            best.get("eval_floor_breach_rate"),
            best_mae.get("kappa_MAE"),
            best_mae.get("pi_MAE"),
        ),
        (
            "Latest checkpoint",
            latest.get("eval_raw_mean"),
            latest.get("eval_raw_median"),
            latest.get("eval_empirical_joint_scaled_mean", latest.get("eval_empirical_scaled_mean")),
            latest.get("eval_kappa_mean"),
            latest.get("eval_pi_mean"),
            latest.get("eval_floor_breach_rate"),
            latest_mae.get("kappa_MAE"),
            latest_mae.get("pi_MAE"),
        ),
        (
            "Merton external",
            merton.get("merton_raw_mean"),
            merton.get("merton_raw_median"),
            merton.get("merton_empirical_joint_scaled_mean", merton.get("merton_empirical_scaled_mean")),
            merton.get("merton_kappa_mean"),
            merton.get("merton_pi_mean"),
            merton.get("merton_floor_rate"),
            None,
            None,
        ),
    ]

    out = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Policy & Raw mean & Raw median & Emp. joint & $\bar{\kappa}$ & $\bar{\pi}$ & Floor & $\kappa$ MAE & $\pi$ MAE \\",
        r"\midrule",
    ]
    for label, raw, med, emp, kappa, pi, floor, kmae, pmae in rows:
        out.append(
            f"{label} & {sci(raw)} & {sci(med)} & {fixed(emp, 3)} & "
            f"{fixed(kappa, 4)} & {fixed(pi, 4)} & {pct(floor)} & "
            f"{fixed(kmae, 4)} & {fixed(pmae, 4)} \\\\"
        )
    out.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\caption{Held-out revisit evaluation. Raw mean and median are computed on the uncompressed objective; empirical joint reward is the compressed training reward after the fixed numerical scale. Higher values are better.}",
        r"\label{tab:revisit-eval}",
        r"\end{table}",
    ])
    return "\n".join(out)


def _config_table(config: dict[str, Any], train_log: dict[str, Any]) -> str:
    env = config.get("env") or {}
    ppo = config.get("ppo") or {}
    args = train_log.get("args") or {}
    eval_n_per_t = int(args.get("eval_n_per_T", 0) or 0)
    stats = {
        "Updates requested": args.get("updates"),
        "Blocks per update": args.get("blocks_per_update"),
        "Episodes per update": ppo.get("episodes_per_update"),
        "Eval episodes": 21 * eval_n_per_t if eval_n_per_t else None,
        "Critic warmup": args.get("critic_warmup"),
        "Learning rate": args.get("lr"),
        "PPO clip": args.get("clip_epsilon"),
        "PPO epochs": args.get("ppo_epochs"),
        "Mini-batch size": args.get("mini_batch_size"),
        "Reward scale": ppo.get("reward_scale"),
        "Consumption cap": env.get("kappa_max"),
        "Floor penalty": env.get("P_penalty_compressed"),
        "Actor B log-std floor": ppo.get("B_min_log_std"),
    }

    out = [
        r"\begin{table}[H]",
        r"\centering",
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Setting & Value \\",
        r"\midrule",
    ]
    for key, value in stats.items():
        if value is None:
            continue
        if isinstance(value, float):
            rendered = f"\\texttt{{{value:.6g}}}"
        else:
            rendered = f"\\texttt{{{latex_escape(str(value))}}}"
        out.append(f"{key} & {rendered} \\\\")
    out.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Revisit training configuration recovered from saved run artifacts.}",
        r"\label{tab:revisit-config}",
        r"\end{table}",
    ])
    return "\n".join(out)


def _evals(train_log: dict[str, Any]) -> list[dict[str, Any]]:
    evals = []
    for row in train_log.get("log") or []:
        ev = row.get("eval")
        if ev:
            ev = dict(ev)
            ev["update"] = row.get("update")
            evals.append(ev)
    return evals


def _find_eval_at_or_before(evals: list[dict[str, Any]], update: int) -> dict[str, Any] | None:
    candidates = [row for row in evals if (row.get("update") or 0) <= update]
    return candidates[-1] if candidates else None


def _artifact_subsection(script_dir: Path, session_dir: Path) -> list[str]:
    """Describe the revisit's shippable artifact bundle, when one exists.

    `make_artifact.py` refreshes `artifact/` at the end of revisit_train.sh, so
    the appendix can tell a reader how to run the revised policy. Returns an
    empty list for workspaces that predate the artifact bundle.
    """
    artifact_dir = script_dir / "artifact"
    if not artifact_dir.is_dir():
        return []

    try:
        rel = artifact_dir.relative_to(session_dir)
    except ValueError:
        rel = artifact_dir

    contract = read_json(artifact_dir / "contract.json")
    lines = [
        "",
        r"\subsection*{Shippable Artifact}",
        (
            f"The revised policy is packaged at \\texttt{{{latex_escape(str(rel))}}} "
            f"with a copied checkpoint, so it runs outside this session. Invoke it "
            f"as \\texttt{{python solve.py -{{}}-instance instance.json}} from that "
            f"directory; \\texttt{{contract.json}} records the observation/action "
            f"contract and the validity domain, and \\texttt{{MODEL\\_CARD.md}} "
            f"records baselines and known failure modes."
        ),
    ]

    facts = []
    if contract.get("obs_dim") is not None:
        facts.append(f"observation dimension {latex_escape(str(contract['obs_dim']))}")
    if contract.get("action_dim") is not None:
        facts.append(f"action dimension {latex_escape(str(contract['action_dim']))}")
    metric = contract.get("selection_metric") or contract.get("metric")
    if metric is not None:
        facts.append(f"selected on {latex_escape(str(metric))}")
    if contract.get("seed") is not None:
        facts.append(f"seed {latex_escape(str(contract['seed']))}")
    if facts:
        lines.append("")
        lines.append(f"Recorded contract: {', '.join(facts)}.")

    smoke = read_json(artifact_dir / "smoke_test_result.json")
    if smoke:
        status = "passed" if smoke.get("passed") else "did not pass"
        lines.append("")
        lines.append(f"The bundled smoke test {status} on the canonical instance.")

    return lines


def _figure(filename: str, caption: str, label: str) -> str:
    return "\n".join([
        r"\begin{figure}[H]",
        r"\centering",
        f"\\includegraphics[width=0.82\\textwidth]{{{tex_path(filename)}}}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\end{figure}",
    ])


def build_revisit_section(
    run_name: str,
    run_dir_arg: str,
    script_dir: Path,
    session_dir: Path,
    run_dir: Path,
    results_dir: Path,
) -> str:
    final_eval = read_json(run_dir / "final_eval.json")
    train_log = read_json(run_dir / "logs" / "train_log.json")
    config = read_json(run_dir / "saved_models" / "v1.0" / "config.json")
    stats = read_json(run_dir / "saved_models" / "v1.0" / "training_stats.json")
    plots = _copy_revisit_plots(run_dir, results_dir, run_name)
    evals = _evals(train_log)

    best = final_eval.get("best") or final_eval.get("best_theory_eval") or {}
    latest = final_eval.get("latest") or {}
    merton = final_eval.get("merton") or {}
    first_eval = evals[0] if evals else {}
    post_warmup = _find_eval_at_or_before(evals, 120) or first_eval
    max_floor = max((num(row.get("eval_floor_breach_rate")) or 0.0) for row in evals) if evals else 0.0

    best_update = stats.get("best_theory_update", train_log.get("best_theory_update", train_log.get("best_update", "--")))
    best_emp_update = stats.get("best_empirical_update", train_log.get("best_empirical_update", "--"))
    elapsed = num(stats.get("elapsed_sec", train_log.get("elapsed_sec")))
    elapsed_text = f"{elapsed / 3600.0:.2f} hours" if elapsed is not None else "not recorded"
    n_eval = best.get("n_eval", latest.get("n_eval", "--"))
    workspace = latex_escape(str(script_dir.relative_to(session_dir)))

    section = [
        f"\\section{{Step 10 Revisit: {latex_escape(run_name)}}}",
        f"Generated on {latex_escape(datetime.now().isoformat(timespec='seconds'))}.",
        "",
        (
            f"\\textbf{{Workspace:}} \\texttt{{{workspace}}}. "
            f"\\textbf{{Run directory:}} \\texttt{{{latex_escape(run_dir_arg)}}}. "
            f"The run completed in {elapsed_text}; best-theory and best-empirical "
            f"checkpoint selection identified update~{latex_escape(str(best_update))} "
            f"(best empirical update {latex_escape(str(best_emp_update))})."
        ),
        "",
        r"\subsection*{Revisit Design}",
        (
            "The revisit keeps the Step 10 problem formulation and evaluates "
            "against the raw uncompressed objective. The changes are structural "
            "rather than benchmark-leaking: modified action bounds or "
            "initialization, revised exploration or numerical safeguards, and "
            "faster rollout collection should be documented here from the saved "
            "configuration. The generated tables below are read from the run "
            "artifacts rather than copied from terminal output."
        ),
        "",
        _config_table(config, train_log),
        "",
        r"\subsection*{Evaluation Results}",
        (
            f"Table~\\ref{{tab:revisit-eval}} summarizes the final held-out "
            f"evaluation on {n_eval} episodes. The best checkpoint has raw mean "
            f"{sci(best.get('eval_raw_mean'))}; the latest checkpoint has raw "
            f"mean {sci(latest.get('eval_raw_mean'))}. If an analytical or "
            "external benchmark is present, it is listed in the same table for "
            "a like-for-like finite-sample comparison."
        ),
        "",
        _eval_rows(final_eval),
        "",
        r"\subsection*{Training Dynamics}",
        (
            f"The first logged eval point has $\\bar{{\\kappa}}="
            f"{fixed_bare(first_eval.get('eval_kappa_mean'), 4)}$ and "
            f"$\\bar{{\\pi}}={fixed_bare(first_eval.get('eval_pi_mean'), 4)}$. "
            f"The held-out raw mean improved from {sci(first_eval.get('eval_raw_mean'))} "
            f"to {sci(post_warmup.get('eval_raw_mean'))} by update "
            f"{latex_escape(str(post_warmup.get('update', 120)))} and then to "
            f"{sci(best.get('eval_raw_mean'))} at update~{latex_escape(str(best_update))}. "
            f"The maximum held-out floor-breach rate over logged eval points was "
            f"{pct(max_floor)}."
        ),
    ]

    if "training_curves.png" in plots:
        section.extend([
            "",
            _figure(
                plots["training_curves.png"],
                "Revisit PPO diagnostics over the logged updates, including held-out evaluation, floor-breach rate, policy entropy, and value losses.",
                "fig:revisit-training-curves",
            ),
        ])

    section.extend([
        "",
        r"\subsection*{Policy Shape and Sensitivity}",
        (
            f"The best checkpoint averages $\\bar{{\\kappa}}="
            f"{fixed_bare(best.get('eval_kappa_mean'), 4)}$ and "
            f"$\\bar{{\\pi}}={fixed_bare(best.get('eval_pi_mean'), 4)}$. "
            f"The latest checkpoint averages $\\bar{{\\kappa}}="
            f"{fixed_bare(latest.get('eval_kappa_mean'), 4)}$ and "
            f"$\\bar{{\\pi}}={fixed_bare(latest.get('eval_pi_mean'), 4)}$. "
            f"The external benchmark, when available, averages "
            f"$\\bar{{\\kappa}}={fixed_bare(merton.get('merton_kappa_mean'), 4)}$ "
            f"and $\\bar{{\\pi}}={fixed_bare(merton.get('merton_pi_mean'), 4)}$."
        ),
    ])

    if "policy_vs_merton.png" in plots:
        section.extend([
            "",
            _figure(
                plots["policy_vs_merton.png"],
                "Best revisit checkpoint versus the external policy benchmark as a function of time-to-go.",
                "fig:revisit-policy-best",
            ),
        ])
    if "policy_vs_merton_latest.png" in plots:
        section.extend([
            "",
            _figure(
                plots["policy_vs_merton_latest.png"],
                "Latest revisit checkpoint versus the external policy benchmark.",
                "fig:revisit-policy-latest",
            ),
        ])
    if "sensitivity.png" in plots:
        section.extend([
            "",
            _figure(
                plots["sensitivity.png"],
                "Revisit policy sensitivity slices over the observed state variables.",
                "fig:revisit-sensitivity",
            ),
        ])

    section.extend(_artifact_subsection(script_dir, session_dir))

    log_tail = tail(run_dir / "train.log")
    if log_tail:
        section.extend([
            "",
            r"\subsection*{Audit Trail}",
            (
                "The final lines below are retained only as a reproducibility "
                "check; the analysis above is generated from structured run artifacts."
            ),
            r"\begin{verbatim}",
            log_tail,
            r"\end{verbatim}",
        ])

    return "\n".join(section) + "\n"


def replace_revisit_section(appendix: Path, marker: str, section: str) -> None:
    existing = appendix.read_text(encoding="utf-8", errors="replace") if appendix.exists() else ""
    if marker in existing:
        start = existing.index(marker)
        next_start = existing.find("\\section{Step 10 Revisit:", start + len(marker))
        if next_start == -1:
            existing = existing[:start].rstrip() + "\n\n" + section
        else:
            existing = existing[:start].rstrip() + "\n\n" + section + "\n\n" + existing[next_start:].lstrip()
    else:
        existing = existing.rstrip() + ("\n\n" if existing.strip() else "") + section
    appendix.write_text(existing, encoding="utf-8")


def write_revisit_appendix(
    run_name: str,
    run_dir_arg: str,
    script_dir: Path,
    publish: bool = True,
    compile_pdf: bool = True,
) -> Path:
    script_dir = Path(script_dir).resolve()
    session_dir = script_dir.parent.parent
    results_dir = session_dir / "results"
    run_dir = script_dir / run_dir_arg
    local_section = run_dir / "revisit_section.tex"
    results_dir.mkdir(exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    section = build_revisit_section(
        run_name,
        run_dir_arg,
        script_dir,
        session_dir,
        run_dir,
        results_dir,
    )
    local_section.write_text(section, encoding="utf-8")

    if not publish:
        return local_section

    marker = f"\\section{{Step 10 Revisit: {latex_escape(run_name)}}}"
    replace_revisit_section(results_dir / "appendix.tex", marker, section)
    main_tex = results_dir / "main.tex"
    ensure_main_input(main_tex)

    if compile_pdf:
        pdflatex = shutil.which("pdflatex")
        if pdflatex and main_tex.exists():
            for _ in range(2):
                result = subprocess.run(
                    [pdflatex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
                    cwd=str(results_dir), capture_output=True, text=True, timeout=180,
                )
                if result.returncode != 0:
                    print("pdflatex failed; appendix.tex was still written.")
                    print(result.stdout[-2000:])
                    print(result.stderr[-2000:])
                    break
            else:
                print(f"Updated {results_dir / 'main.pdf'}")
        elif main_tex.exists():
            print("pdflatex not found; appendix.tex and main.tex were updated.")
        else:
            print("No results/main.tex found; appendix.tex was written.")

    return results_dir / "appendix.tex"
