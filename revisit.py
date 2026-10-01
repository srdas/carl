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

"""Helpers for CARL Step 10 revisit mode."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from bedrock_converse import converse_text
from report import find_session_plots, is_checkpoint_dupe

_TEXT_SUFFIXES = {
    ".md", ".txt", ".log", ".json", ".csv", ".tex", ".py", ".sh", ".yaml", ".yml",
}
_SKIP_DIRS = {
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "experiments", "logs", "checkpoints", "saved_models",
}
_SKIP_SUFFIXES = {".pt", ".pth", ".ckpt", ".pkl", ".pyc", ".log"}
_SOURCE_SUFFIXES = {".py", ".sh", ".toml", ".yaml", ".yml", ".ipynb"}
_SOURCE_MARKER_FILES = {
    "config.py",
    "environment.py",
    "long_train.sh",
    "main.py",
    "pyproject.toml",
    "requirements.txt",
    "revisit_train.sh",
    "setup.py",
    "train.py",
}
_NONE_REVISIT_INPUTS = {
    "",
    "n/a",
    "na",
    "no",
    "none",
    "nope",
    "nothing",
    "nothing else",
    "no instructions",
    "no prior input",
    "no prior inputs",
    "no prior observations",
}
_NEGATIVE_PATH_CUES = (
    "don't",
    "dont",
    "do not",
    "skip",
    "exclude",
    "ignore",
    "avoid",
    "do not inspect",
    "don't inspect",
    "do not look",
    "don't look",
    "do not read",
    "don't read",
)

REVISIT_TOOLS = [
    {
        "name": "list_revisit_artifacts",
        "description": "List logs, plots, result files, checkpoints, and long-run artifacts in the revisited session.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "read_artifact",
        "description": "Read a text artifact from the revisited session directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to the session directory"},
                "max_chars": {"type": "integer", "description": "Maximum characters to return"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "describe_plot",
        "description": "Describe a PNG plot from the revisited session using the vision model.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "PNG path relative to the session directory"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "prepare_revisit_workspace",
        "description": "Create a fresh self-contained code/revisit_N workspace by copying a user-confirmed session-local source directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "source_dir": {
                    "type": "string",
                    "description": "Confirmed source directory to copy, relative to the session directory or absolute within it",
                },
            },
            "required": ["source_dir"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file in the session code/ directory. In Step 10, use only paths inside code/revisit_N/.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path within code/"},
                "content": {"type": "string", "description": "Full file content"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "run_command",
        "description": "Run a short inspection or validation command in code/. Step 10 refuses retraining commands.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute"},
            },
            "required": ["command"],
        },
    },
]


def resolve_revisit_session(carl_dir: str | Path, session_dir: str | Path) -> Path:
    """Resolve and validate an existing session path under CARL/sessions."""
    base = Path(carl_dir).resolve()
    sessions_root = (base / "sessions").resolve()
    path = Path(session_dir)
    if not path.is_absolute():
        path = (base / path).resolve()
    else:
        path = path.resolve()

    try:
        path.relative_to(sessions_root)
    except ValueError as exc:
        raise ValueError(f"Revisit path must be under {sessions_root}: {path}") from exc
    if not path.is_dir():
        raise FileNotFoundError(f"Session directory not found: {path}")
    if not (path / "memory.md").is_file():
        raise FileNotFoundError(f"Not a CARL session: missing {path / 'memory.md'}")
    return path


def _rel(session_dir: Path, path: Path) -> str:
    try:
        return path.relative_to(session_dir).as_posix()
    except ValueError:
        return path.as_posix()


def _read_capped(path: Path, max_chars: int = 12000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    if len(text) > max_chars:
        return text[:max_chars] + "\n...(truncated)"
    return text


def build_revisit_preflight_prompt(session_dir: Path) -> str:
    """Return the controller-level prompt shown before artifact inspection."""
    return (
        f"We're in Step 10 revisit mode for: `{session_dir}`. "
        "Before I inspect the session artifacts, do you have any instructions "
        "or observations I should apply? For example: "
        "`don't inspect code/revisit_1/`, "
        "`focus on the instability after update 120`, or `none`."
    )


def normalize_revisit_guidance(user_input: str | None) -> str:
    """Normalize optional pre-analysis guidance from the user."""
    if user_input is None:
        return ""
    text = user_input.strip()
    lowered = re.sub(r"\s+", " ", text.lower()).strip(" .")
    if lowered in _NONE_REVISIT_INPUTS:
        return ""
    return text


def _looks_path_like(text: str) -> bool:
    return (
        "/" in text
        or "\\" in text
        or bool(re.search(r"\brevisit_\d+\b", text, re.IGNORECASE))
    )


def _clean_exclusion_path(text: str) -> str:
    cleaned = text.strip().strip("`'\".,;:()[]{}")
    cleaned = cleaned.replace("\\", "/")
    cleaned = re.sub(r"/+", "/", cleaned)
    cleaned = cleaned.rstrip("/")
    if cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return cleaned


def _dedupe_ordered(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value and value not in seen:
            out.append(value)
            seen.add(value)
    return out


def extract_revisit_exclusions(guidance: str | None) -> list[str]:
    """Best-effort extraction of paths the user asked us not to inspect."""
    text = normalize_revisit_guidance(guidance)
    if not text:
        return []

    exclusions: list[str] = []
    segments = re.split(r"[\n;]+|(?<=[.!?])\s+", text)
    for segment in segments:
        lowered = segment.lower()
        if not any(cue in lowered for cue in _NEGATIVE_PATH_CUES):
            continue

        quoted = re.findall(r"`([^`]+)`|'([^']+)'|\"([^\"]+)\"", segment)
        for groups in quoted:
            candidate = next((g for g in groups if g), "")
            if _looks_path_like(candidate):
                exclusions.append(_clean_exclusion_path(candidate))

        for raw in segment.split():
            candidate = _clean_exclusion_path(raw)
            if _looks_path_like(candidate):
                exclusions.append(candidate)

    return _dedupe_ordered(exclusions)


def _path_for_exclusion_match(session_dir: Path, path: str | Path) -> str:
    path_obj = Path(path)
    if path_obj.is_absolute():
        try:
            return _clean_exclusion_path(path_obj.resolve().relative_to(session_dir.resolve()).as_posix())
        except ValueError:
            return _clean_exclusion_path(path_obj.as_posix())
    return _clean_exclusion_path(str(path))


def is_revisit_path_excluded(
    session_dir: Path,
    rel_path: str | Path,
    excluded_paths: list[str] | None = None,
) -> bool:
    """Return whether a session-relative artifact path is blocked by preflight."""
    if not excluded_paths:
        return False

    rel = _path_for_exclusion_match(session_dir, rel_path)
    rel_parts = [part for part in rel.lower().split("/") if part]
    for excluded in excluded_paths:
        ex = _path_for_exclusion_match(session_dir, excluded)
        if not ex:
            continue
        ex_lower = ex.lower()
        if rel.lower() == ex_lower or rel.lower().startswith(ex_lower.rstrip("/") + "/"):
            return True
        if "/" not in ex_lower and ex_lower in rel_parts:
            return True
    return False


def read_artifact(
    session_dir: Path,
    rel_path: str,
    max_chars: int = 12000,
    excluded_paths: list[str] | None = None,
) -> str:
    """Read a text artifact from inside the revisited session."""
    path = (session_dir / rel_path).resolve()
    try:
        path.relative_to(session_dir.resolve())
    except ValueError:
        return f"Refusing to read outside session: {rel_path}"
    if is_revisit_path_excluded(session_dir, rel_path, excluded_paths):
        return f"Refusing to read excluded artifact: {rel_path}"
    if not path.is_file():
        return f"Artifact not found: {rel_path}"
    if path.suffix.lower() not in _TEXT_SUFFIXES:
        return f"Refusing to read non-text artifact: {rel_path}"
    return _read_capped(path, max_chars=max_chars)


def _iter_included_files(session_dir: Path, excluded_paths: list[str] | None):
    """Yield files without descending into excluded directories."""
    for root, dirs, names in os.walk(session_dir):
        root_path = Path(root)
        dirs[:] = sorted(
            d for d in dirs
            if not is_revisit_path_excluded(session_dir, root_path / d, excluded_paths)
        )
        for name in sorted(names):
            path = root_path / name
            if not is_revisit_path_excluded(session_dir, path, excluded_paths):
                yield path


def _resolve_source_dir(session_dir: Path, source_dir: str | Path) -> tuple[Path, str]:
    if source_dir is None or not str(source_dir).strip():
        raise ValueError(
            "prepare_revisit_workspace requires source_dir. Ask the user to "
            "confirm a session-local source directory before creating revisit_N."
        )

    session_dir = Path(session_dir).resolve()
    raw_source = Path(str(source_dir).strip())
    source_path = raw_source if raw_source.is_absolute() else session_dir / raw_source
    source_path = source_path.resolve()

    try:
        rel_source = source_path.relative_to(session_dir)
    except ValueError as exc:
        raise ValueError(
            f"Revisit source_dir must resolve inside the active session: {source_dir}"
        ) from exc

    if not source_path.is_dir():
        raise FileNotFoundError(
            f"Revisit source_dir is not a directory: {rel_source.as_posix()}"
        )

    return source_path, rel_source.as_posix()


def _has_immediate_source_files(path: Path) -> bool:
    try:
        entries = list(path.iterdir())
    except OSError:
        return False

    for entry in entries:
        if not entry.is_file():
            continue
        lower = entry.name.lower()
        if lower in _SOURCE_MARKER_FILES or entry.suffix.lower() in _SOURCE_SUFFIXES:
            return True
    return False


def discover_revisit_source_candidates(
    session_dir: Path,
    excluded_paths: list[str] | None = None,
) -> list[str]:
    """Return plausible session-local directories the next revisit may copy."""
    session_dir = Path(session_dir)
    candidates: list[str] = []

    for root, dirs, names in os.walk(session_dir):
        root_path = Path(root)
        rel_root = Path(_rel(session_dir, root_path))
        if rel_root.as_posix() == ".":
            pass
        elif is_revisit_path_excluded(session_dir, root_path, excluded_paths):
            dirs[:] = []
            continue
        dirs[:] = sorted(
            d for d in dirs
            if (
                d.lower() not in {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
                                  "logs", "checkpoints", "saved_models"}
                and not is_revisit_path_excluded(session_dir, root_path / d, excluded_paths)
            )
        )

        if root_path == session_dir:
            continue
        if _has_immediate_source_files(root_path):
            candidates.append(_rel(session_dir, root_path))

    def _candidate_sort_key(path: str) -> tuple[int, str]:
        lower = path.lower()
        if lower == "code":
            return (0, lower)
        if re.search(r"(^|/)revisit_\d+($|/)", lower):
            return (1, lower)
        if lower in {"long_train", "long-run", "longrun"}:
            return (2, lower)
        if "long" in lower:
            return (3, lower)
        return (4, lower)

    return sorted(_dedupe_ordered(candidates), key=_candidate_sort_key)


def discover_artifacts(session_dir: Path, excluded_paths: list[str] | None = None) -> dict:
    """Discover Step 10-relevant artifacts in an existing session."""
    session_dir = Path(session_dir)
    files = {
        "core": [],
        "train_logs": [],
        "plots": [],
        "results_tex": [],
        "json_results": [],
        "long_run": [],
        "source_candidates": [],
    }
    files["source_candidates"] = discover_revisit_source_candidates(
        session_dir, excluded_paths=excluded_paths
    )

    for name in ("memory.md", "step_08_conversation.md", "step_09_conversation.md"):
        p = session_dir / name
        if p.exists() and not is_revisit_path_excluded(session_dir, name, excluded_paths):
            files["core"].append(_rel(session_dir, p))
    for p in sorted(session_dir.glob("step_10*_conversation.md")):
        rel = _rel(session_dir, p)
        if not is_revisit_path_excluded(session_dir, rel, excluded_paths):
            files["core"].append(rel)

    # PNGs get their own pass: _iter_included_files walks the whole session but
    # plots frequently live under code/experiments/**, which the source-file
    # consumers below treat as noise. find_session_plots is the same discovery
    # the report uses, so Step 10 and the PDF see one identical plot set.
    for p in find_session_plots(session_dir):
        rel = _rel(session_dir, p)
        if not is_revisit_path_excluded(session_dir, rel, excluded_paths):
            files["plots"].append(rel)

    for p in _iter_included_files(session_dir, excluded_paths):
        rel = _rel(session_dir, p)
        parts = {part.lower() for part in p.relative_to(session_dir).parts}
        lower_name = p.name.lower()
        lower_rel = rel.lower()
        if lower_name == "train.log" or lower_name.endswith("_train.log"):
            files["train_logs"].append(rel)
        if is_checkpoint_dupe(p):
            continue
        if "results" in parts and p.suffix.lower() == ".tex":
            files["results_tex"].append(rel)
        if p.suffix.lower() == ".json" and (
            "results" in parts or "experiments" in parts or "runs" in parts
        ):
            files["json_results"].append(rel)
        if (
            "long_train" in parts
            or "long" in lower_rel
            or re.search(r"revisit_\d+", lower_rel)
        ):
            if p.suffix.lower() in {".log", ".json", ".png", ".pt", ".tex", ".csv"}:
                files["long_run"].append(rel)

    return files


def build_revisit_context(
    session_dir: Path,
    guidance: str | None = None,
    excluded_paths: list[str] | None = None,
) -> str:
    """Build a compact context block for the Step 10 system prompt."""
    artifacts = discover_artifacts(session_dir, excluded_paths=excluded_paths)
    guidance_text = normalize_revisit_guidance(guidance)
    memory = (
        ""
        if is_revisit_path_excluded(session_dir, "memory.md", excluded_paths)
        else _read_capped(session_dir / "memory.md", 24000)
    )
    step8 = (
        ""
        if is_revisit_path_excluded(session_dir, "step_08_conversation.md", excluded_paths)
        else _read_capped(session_dir / "step_08_conversation.md", 10000)
    )
    step9 = (
        ""
        if is_revisit_path_excluded(session_dir, "step_09_conversation.md", excluded_paths)
        else _read_capped(session_dir / "step_09_conversation.md", 14000)
    )

    parts = [
        f"# Revisit Session\n{session_dir}",
        "# User Preflight Guidance\n" + (guidance_text or "No prior inputs."),
        "# Preflight Excluded Paths\n" + (
            "\n".join(f"- {p}" for p in excluded_paths)
            if excluded_paths else "None."
        ),
        "# Artifact Inventory\n" + json.dumps(artifacts, indent=2),
        "# memory.md\n" + memory,
    ]
    if step8:
        parts.append("# step_08_conversation.md\n" + step8)
    if step9:
        parts.append("# step_09_conversation.md\n" + step9)
    return "\n\n".join(parts)


def describe_plot(
    client,
    model: str,
    session_dir: Path,
    rel_path: str,
    excluded_paths: list[str] | None = None,
) -> str:
    """Describe a PNG plot from inside the revisited session."""
    path = (session_dir / rel_path).resolve()
    try:
        path.relative_to(session_dir.resolve())
    except ValueError:
        return f"Refusing to read outside session: {rel_path}"
    if is_revisit_path_excluded(session_dir, rel_path, excluded_paths):
        return f"Refusing to describe excluded plot: {rel_path}"
    if not path.is_file() or path.suffix.lower() != ".png":
        return f"Plot not found or not PNG: {rel_path}"

    image_bytes = path.read_bytes()
    memory = (
        ""
        if is_revisit_path_excluded(session_dir, "memory.md", excluded_paths)
        else _read_capped(session_dir / "memory.md", 2500)
    )
    system = (
        "You are analysing a plot from a CARL reinforcement-learning run. "
        "Describe axes, trends, visible numerical values, and what the plot "
        "suggests for Step 10 post-training diagnosis. Be factual and concise."
    )
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": f"Session context:\n{memory}\n\nPlot: {rel_path}"},
            {"type": "image_raw", "format": "png", "data": image_bytes},
        ],
    }]
    return converse_text(
        client, model, system, messages,
        max_tokens=900, temperature=0.2,
    ).strip()


def _ignore_code_copy(dir_path: str, names: list[str]) -> set[str]:
    ignored = set()
    for name in names:
        path = Path(dir_path) / name
        lower = name.lower()
        if path.is_dir() and (
            lower in _SKIP_DIRS or re.fullmatch(r"revisit_\d+", lower)
        ):
            ignored.add(name)
        elif path.is_file() and (
            path.suffix.lower() in _SKIP_SUFFIXES
            or lower == ".bg_pids.json"
            or lower.startswith("events.out.tfevents")
        ):
            ignored.add(name)
    return ignored


def next_revisit_name(session_dir: Path) -> str:
    code_dir = session_dir / "code"
    max_seen = 0
    if code_dir.exists():
        for p in code_dir.iterdir():
            m = re.fullmatch(r"revisit_(\d+)", p.name)
            if p.is_dir() and m:
                max_seen = max(max_seen, int(m.group(1)))
    return f"revisit_{max_seen + 1}"


def prepare_revisit_workspace(session_dir: Path, source_dir: str | Path) -> dict:
    """Create a fresh self-contained code/revisit_N workspace."""
    session_dir = Path(session_dir)
    source_path, source_rel = _resolve_source_dir(session_dir, source_dir)
    code_dir = session_dir / "code"
    code_dir.mkdir(exist_ok=True)
    name = next_revisit_name(session_dir)
    dest = code_dir / name
    if dest.exists():
        raise FileExistsError(f"Refusing to overwrite existing revisit workspace: {dest}")

    shutil.copytree(source_path, dest, ignore=_ignore_code_copy)
    _write_revisit_appendix_helper(dest)
    _write_revisit_train_script(dest, name)

    try:
        os.chmod(dest / "revisit_train.sh", 0o755)
    except OSError:
        pass

    return {
        "source_dir": source_rel,
        "workspace": _rel(session_dir, dest),
        "run_name": name,
        "script": _rel(session_dir, dest / "revisit_train.sh"),
        "appendix_helper": _rel(session_dir, dest / "revisit_appendix.py"),
        "output_dir": _rel(session_dir, dest / "experiments" / name),
    }


def prepare_revisit_workspace_tool_response(session_dir: Path, input_data: dict) -> str:
    """Return the Step 10 tool response for workspace creation."""
    source_dir = input_data.get("source_dir") if isinstance(input_data, dict) else None
    if not source_dir:
        return (
            "prepare_revisit_workspace requires source_dir. Ask the user "
            "to confirm a valid session-local source directory before "
            "creating code/revisit_N/."
        )
    try:
        result = prepare_revisit_workspace(session_dir, source_dir)
    except (FileNotFoundError, ValueError) as exc:
        return (
            f"Could not prepare revisit workspace: {exc}. Ask the user "
            "to confirm a valid session-local source directory."
        )
    return json.dumps(result, indent=2)


def _write_revisit_train_script(dest: Path, run_name: str) -> None:
    script = f"""#!/usr/bin/env bash
set -euo pipefail

# Step 10 revisit training script.
# Revisit runs intentionally start from fresh initialization. Old checkpoints,
# original session pretrain files, long-run init files, and prior revisit init
# files are diagnostic baselines only; they are not used to start this run.
# If the revised pipeline has a pretraining/init stage, run it locally in this
# workspace with the revised code.
#
# Override TRAIN_CMD/EVAL_CMD if needed, but keep TRAIN_CMD fresh-start.
# Example: TRAIN_CMD="python train.py --updates 500" bash revisit_train.sh

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_NAME="${{RUN_NAME:-{run_name}}}"
EVAL_CMD="${{EVAL_CMD:-}}"
RUN_DIR="experiments/${{RUN_NAME}}"

mkdir -p "$RUN_DIR"
echo "Step 10 revisit run: $RUN_NAME"
echo "Workspace: $SCRIPT_DIR"
echo "Initialization: fresh revisit seed; no --init-from is used by default."

if [[ "${{TRAIN_CMD:-}}" == *"--init-from"* ]]; then
  echo "Refusing TRAIN_CMD with --init-from; Step 10 revisits start fresh." >&2
  exit 2
fi

if [[ -n "${{TRAIN_CMD:-}}" ]]; then
  echo "Train command override: $TRAIN_CMD"
  eval "$TRAIN_CMD" 2>&1 | tee "$RUN_DIR/train.log"
else
  echo "Train command: python train.py"
  python train.py 2>&1 | tee "$RUN_DIR/train.log"
fi

if [[ -n "$EVAL_CMD" ]]; then
  echo "Eval command: $EVAL_CMD" | tee -a "$RUN_DIR/train.log"
  eval "$EVAL_CMD" 2>&1 | tee -a "$RUN_DIR/train.log"
elif [[ -f evaluate.py ]]; then
  echo "Eval command: python evaluate.py" | tee -a "$RUN_DIR/train.log"
  python evaluate.py 2>&1 | tee -a "$RUN_DIR/train.log" || true
fi

python revisit_appendix.py --run-name "$RUN_NAME" --run-dir "$RUN_DIR"

# Refresh the shippable artifact from THIS run's best checkpoint, if the
# workspace carries the builder. Conditional so a workspace copied from a
# session that predates make_artifact.py still runs.
if [[ -f make_artifact.py ]]; then
  echo "Refreshing artifact/ from $RUN_NAME" | tee -a "$RUN_DIR/train.log"
  python make_artifact.py --run-name "$RUN_NAME" 2>&1 | tee -a "$RUN_DIR/train.log"
fi
"""
    (dest / "revisit_train.sh").write_text(script, encoding="utf-8")


def _write_revisit_appendix_helper(dest: Path) -> None:
    helper = r'''#!/usr/bin/env python3
"""Publish a report-style Step 10 revisit appendix."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Write experiments/<run>/revisit_section.tex but do not publish to results/.",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Update LaTeX only; skip pdflatex even if available.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    carl_root = next(
        (p for p in [script_dir, *script_dir.parents] if (p / "revisit_report.py").is_file()),
        script_dir.parents[2],
    )
    if str(carl_root) not in sys.path:
        sys.path.insert(0, str(carl_root))

    from revisit_report import write_revisit_appendix

    written = write_revisit_appendix(
        args.run_name,
        args.run_dir,
        script_dir,
        publish=not args.local_only,
        compile_pdf=not args.no_pdf,
    )
    print(f"Wrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
    path = dest / "revisit_appendix.py"
    path.write_text(helper, encoding="utf-8")
    try:
        os.chmod(path, 0o755)
    except OSError:
        pass
