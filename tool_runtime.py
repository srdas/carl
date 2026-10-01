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

"""Shared tool-execution logic for CARL's two engines.

`web.py` and `chat.py` are near-duplicate controllers around the same tool
loop. Anything here is the single definition both import, so a behavioural fix
lands once instead of twice.
"""

from __future__ import annotations

import re
import shlex
import sys
import time
from pathlib import PurePosixPath

# ── Tool-loop limits ─────────────────────────────────────────────────────

MAX_TOOL_ROUNDS = 80
STEP8_DEFAULT_BUDGET_SECONDS = 20 * 60  # used when Step 7 named no budget
STEP9_DEFAULT_BUDGET_SECONDS = 45 * 60  # used when Step 7 named no budget
CHECK_PROCESS_MIN_WAIT = 60  # never poll more often than once per minute

# Seconds a step gets *after* its budget to land a summary. A hard cut at the
# budget itself ends the step with nothing recorded, which is worse than a
# small overrun: one observed Step 8 spent 50 minutes diagnosing a stuck
# policy head and all of that would have been lost. Tools stay available in
# the grace window so the model can read one last log and report.
WRAP_UP_GRACE_SECONDS = 3 * 60

# Floor for a clamped run_command timeout. Below this even `tail` on a log can
# fail, so a nearly-exhausted budget would block the wrap-up it is asking for.
MIN_COMMAND_TIMEOUT = 30

# Literature searches allowed per step. A step that reads an empty result as
# "the index is thin" will otherwise keep reformulating the query -- one
# session ran 16 searches against a rate-limited backend before giving up.
MAX_LITERATURE_SEARCHES = 8
LITERATURE_LIMIT_MESSAGE = (
    f"Search limit reached: {MAX_LITERATURE_SEARCHES} searches already run in "
    "this step, so no further search will be executed. Do not call "
    "search_literature again. Summarize what the completed searches did and "
    "did not turn up, state plainly if coverage is thin, and continue the step."
)


# ── Step 10 training refusal ─────────────────────────────────────────────
#
# Step 10 analyses a finished run and prepares the next one; the user launches
# it. The first version of this check matched any substring of the command line,
# which refused read-only work that merely mentioned a script name (`grep
# train.py train.log`, `chmod +x revisit_train.sh`, `ls long_train.sh`) -- 19
# such refusals across three observed sessions. We now match the *executed
# program* of each pipeline segment instead.

# Scripts whose whole purpose is to start a training run.
_TRAINING_SCRIPTS = frozenset({
    "revisit_train.sh", "long_train.sh", "run.sh", "train.sh",
})
# Python entry points that train when executed.
_TRAINING_ENTRY_FILES = frozenset({"train.py", "long_train.py"})
_TRAINING_MODULES = frozenset({"train", "long_train"})

_SHELLS = frozenset({"bash", "sh", "zsh", "dash", "ksh", "shell"})
_PYTHONS = frozenset({"python", "python3", "python2", "py", "ipython"})
# Wrappers that execute their argument list; unwrap and re-check.
_WRAPPERS = frozenset({
    "nohup", "time", "env", "timeout", "stdbuf", "setsid", "sudo", "nice",
    "xargs", "unbuffer", "screen", "tmux",
})
# Verbs that only read or set metadata. Listed so the intent is explicit in
# code review; anything not recognised as a trainer is allowed anyway.
READ_ONLY_VERBS = frozenset({
    "cat", "head", "tail", "grep", "egrep", "fgrep", "rg", "ls", "find", "wc",
    "chmod", "diff", "stat", "file", "du", "df", "sort", "uniq", "cut", "awk",
    "sed", "tr", "echo", "printf", "pwd", "cd", "which", "nl", "realpath",
    "basename", "dirname", "md5sum", "sha256sum", "tree", "jq", "git", "ln",
})

# Characters that shlex(punctuation_chars=True) emits as standalone operator
# tokens; each one ends a pipeline segment.
_OPERATOR_CHARS = set("&|;()<>")

_LEGACY_FORBIDDEN_SUBSTRINGS = (
    "revisit_train.sh", "long_train.sh", "run.sh",
    "python train.py", "python3 train.py", " train.py",
)


def _split_segments(command: str) -> list[list[str]] | None:
    """Split a command line into argv lists, one per pipeline/list segment.

    Returns None when the line cannot be tokenised, so the caller can fall
    back to the conservative substring test.
    """
    try:
        lex = shlex.shlex(command, posix=True, punctuation_chars=True)
        lex.whitespace_split = True
        tokens = list(lex)
    except ValueError:
        return None

    segments: list[list[str]] = []
    current: list[str] = []
    for token in tokens:
        if token and set(token) <= _OPERATOR_CHARS:
            if current:
                segments.append(current)
                current = []
        else:
            current.append(token)
    if current:
        segments.append(current)
    return segments


def _basename(arg: str) -> str:
    return PurePosixPath(arg).name.lower()


def _code_runs_trainer(code: str) -> bool:
    """Whether a `python -c` payload starts training rather than inspecting.

    Naming a training file is not enough -- reading `train.py` to inspect it is
    allowed. The payload must import or execute the trainer.
    """
    lowered = code.lower()
    if re.search(r"\b(import|from)\s+(long_)?train\b", lowered):
        return True
    if re.search(r"\btrain\s*\.\s*(main|train|run)\s*\(", lowered):
        return True
    # Indirect execution: exec/runpy/subprocess/os.system pointed at a trainer.
    executors = r"(exec|eval|runpy\.run_path|os\.system|os\.exec\w*|subprocess\.\w+|popen)"
    names = "|".join(
        re.escape(n) for n in sorted(_TRAINING_ENTRY_FILES | _TRAINING_SCRIPTS)
    )
    return bool(re.search(rf"{executors}\b[^\n]*({names})", lowered))


def _strip_leading_env(argv: list[str]) -> list[str]:
    """Drop leading `VAR=value` assignments, which precede the real program.

    `RUN_NAME=x bash revisit_train.sh` and
    `TRAIN_CMD='python train.py' bash revisit_train.sh` both start training.
    """
    i = 0
    while i < len(argv) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", argv[i], re.DOTALL):
        i += 1
    return argv[i:]


def _python_runs_trainer(args: list[str]) -> bool:
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("-c", "-m"):
            payload = args[i + 1] if i + 1 < len(args) else ""
            if arg == "-m":
                return payload.split(".")[-1].lower() in _TRAINING_MODULES
            return _code_runs_trainer(payload)
        if arg.startswith("-"):
            i += 1
            continue
        # First non-flag argument is the script being run.
        return _basename(arg) in _TRAINING_ENTRY_FILES | _TRAINING_SCRIPTS
    return False  # bare `python` REPL


def _segment_starts_training(argv: list[str], depth: int = 0) -> bool:
    argv = _strip_leading_env(argv)
    if not argv or depth > 3:
        return False
    prog = _basename(argv[0])
    rest = argv[1:]

    if prog in _TRAINING_SCRIPTS:  # ./revisit_train.sh
        return True
    if prog in _SHELLS:
        if "-n" in rest:
            return False  # `bash -n script.sh` is a syntax check, not a run
        if "-c" in rest:
            payload = rest[rest.index("-c") + 1] if rest.index("-c") + 1 < len(rest) else ""
            return is_revisit_training_command(payload)
        return any(
            _basename(a) in _TRAINING_SCRIPTS for a in rest if not a.startswith("-")
        )
    if prog in _PYTHONS:
        return _python_runs_trainer(rest)
    if prog in _WRAPPERS:
        # Drop the wrapper's own flags, and the bare duration/priority argument
        # that `timeout` and `nice` take, to reach the wrapped program.
        inner = [
            a for a in rest
            if not a.startswith("-")
            and not re.fullmatch(r"\d+(\.\d+)?[smhd]?", a)
        ]
        return _segment_starts_training(inner, depth + 1)
    return False


def is_revisit_training_command(command: str) -> bool:
    """Whether `command` would start a training run (refused in Step 10).

    Read-only inspection and metadata commands that merely name a training
    script -- `grep train.py train.log`, `chmod +x revisit_train.sh`,
    `cat long_train.sh` -- are not training commands.
    """
    if not command or not command.strip():
        return False
    segments = _split_segments(command)
    if segments is None:
        # Unparseable (unbalanced quotes). Stay conservative.
        lower = command.lower()
        return any(token in lower for token in _LEGACY_FORBIDDEN_SUBSTRINGS)
    return any(_segment_starts_training(seg) for seg in segments)


def revisit_training_refusal(command: str) -> str:
    """Refusal text that names the allowed alternatives."""
    return (
        f"Refusing to run retraining in Step 10: `{command.strip()[:200]}` would "
        "start a training run. Step 10 prepares the next experiment; the user "
        "launches it afterwards.\n"
        "Allowed here: read-only inspection (cat, head, tail, grep, ls, find, "
        "wc, diff, python -c that only reads), `chmod +x revisit_train.sh`, the "
        "artifact tools (list_revisit_artifacts, read_artifact, describe_plot), "
        "and write_file to edit the revisit workspace. Write revisit_train.sh "
        "and leave execution to the user."
    )


# ── Step 7 time budgets ──────────────────────────────────────────────────
#
# Step 7 names a wall-clock budget for Step 8 (initial testing) and Step 9
# (full training). Both are enforced as real deadlines -- see StepDeadline.
#
# These budgets must be read from Step 7's *raw turn text*, not from its saved
# summary. The summarizer is asked for "concise bullets" and has no reason to
# keep a machine-readable line: one observed session emitted
# TRAINING_BUDGET_MINUTES: 60, had it dropped from the summary, and silently
# fell back to the 45-minute default. The engines persist what they parse from
# the raw text on the Session, and the resolution path is always logged.

TRAINING_BUDGET_MARKER = "TRAINING_BUDGET_MINUTES:"
TESTING_BUDGET_MARKER = "TESTING_BUDGET_MINUTES:"

_MARKER_RE = re.compile(
    r"TRAINING_BUDGET_MINUTES\s*:\s*(\d+(?:\.\d+)?)", re.IGNORECASE
)
_TESTING_MARKER_RE = re.compile(
    r"TESTING_BUDGET_MINUTES\s*:\s*(\d+(?:\.\d+)?)", re.IGNORECASE
)
_STEP9_TIME_RE = re.compile(
    r"[Ss]tep\s*9[^.\n]*?(\d+(?:\.\d+)?)\s*(hour|hr|minute|min)", re.IGNORECASE
)
_STEP8_TIME_RE = re.compile(
    r"[Ss]tep\s*8[^.\n]*?(\d+(?:\.\d+)?)\s*(hour|hr|minute|min)", re.IGNORECASE
)
_ANY_TIME_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(hour|hr|minute|min)", re.IGNORECASE)

# Per-step budget resolution: (marker regex, step-specific free-text regex,
# default seconds, marker name for the log line).
_BUDGET_SPECS = {
    8: (_TESTING_MARKER_RE, _STEP8_TIME_RE, STEP8_DEFAULT_BUDGET_SECONDS,
        TESTING_BUDGET_MARKER),
    9: (_MARKER_RE, _STEP9_TIME_RE, STEP9_DEFAULT_BUDGET_SECONDS,
        TRAINING_BUDGET_MARKER),
}


def _minutes_to_seconds(value: str, unit: str) -> int:
    seconds = float(value) * (3600 if unit.lower().startswith("h") else 60)
    return int(seconds)


def parse_step_budget_seconds(
    step_num: float, step7_text: str | None
) -> tuple[int, str]:
    """Resolve a step's wall-clock budget in seconds from Step 7's text.

    Returns (seconds, source) where source is "marker", "regex", or "default".
    Steps without a budget spec fall back to the Step 9 numbers so a caller
    passing an unexpected step still gets a usable deadline.
    """
    marker_re, step_re, default_seconds, _ = _BUDGET_SPECS.get(
        step_num, _BUDGET_SPECS[9]
    )
    if step7_text:
        m = marker_re.search(step7_text)
        if m:
            minutes = float(m.group(1))
            if minutes > 0:
                return int(minutes * 60), "marker"

        # Only the step's *own* free-text pattern is trusted here. Falling back
        # to _ANY_TIME_RE for Step 8 would happily match the Step 9 number (or
        # an unrelated "20 minutes" elsewhere in the text), which is worse than
        # taking a known default.
        m = step_re.search(step7_text)
        if m:
            seconds = _minutes_to_seconds(m.group(1), m.group(2))
            if seconds > 0:
                return seconds, "regex"

    return default_seconds, "default"


def parse_training_budget_seconds(step7_summary: str | None) -> tuple[int, str]:
    """Resolve the Step 9 training budget in seconds.

    Kept as the Step 9 entry point; unlike `parse_step_budget_seconds` it also
    accepts a bare time mention anywhere in the text, which predates the
    marker and is still the best guess for older sessions.
    """
    seconds, source = parse_step_budget_seconds(9, step7_summary)
    if source == "default" and step7_summary:
        m = _ANY_TIME_RE.search(step7_summary)
        if m:
            fallback = _minutes_to_seconds(m.group(1), m.group(2))
            if fallback > 0:
                return fallback, "regex"
    return seconds, source


def resolve_step_budget(
    step_num: float, step7_text: str | None, log=None
) -> tuple[int, str]:
    """Resolve a step budget and report which path was taken.

    `log` is an optional callable taking one string. The resolution path is
    always reported so a silent fall back to the default is visible.
    """
    if step_num == 9:
        budget_seconds, source = parse_training_budget_seconds(step7_text)
    else:
        budget_seconds, source = parse_step_budget_seconds(step_num, step7_text)
    marker = _BUDGET_SPECS.get(step_num, _BUDGET_SPECS[9])[3]
    message = (
        f"[step {step_num:g} budget] {budget_seconds // 60} min from {source}"
        + (
            f" -- no {marker} line in the Step 7 record"
            if source == "default"
            else ""
        )
    )
    if log is not None:
        log(message)
    else:
        sys.stderr.write(f"{message}\n")
    return budget_seconds, source


def check_interval_seconds(
    budget_seconds: int,
    max_rounds: int = MAX_TOOL_ROUNDS,
    min_wait: int = CHECK_PROCESS_MIN_WAIT,
) -> int:
    """Seconds check_process should wait, spreading polls over the budget.

    Takes an already-resolved budget: the caller owns resolution (and its
    logging) so the interval for Step 8 is sized from Step 8's budget rather
    than from Step 9's, which is what an earlier version did.
    """
    return max(int(budget_seconds) // (max_rounds // 2), min_wait)


# ── Step wall-clock deadlines ────────────────────────────────────────────


class StepDeadline:
    """Wall-clock budget for one step, with a grace window for wrapping up.

    Two thresholds. At `soft_expired` the step is over budget: the engines stop
    starting new work and ask for a final summary. At `hard_expired` the grace
    window is gone: background jobs are killed and the tool loop breaks, which
    still leaves the normal end-of-step summarize path to run.

    Uses a monotonic clock, so a step resumed after a transient API error gets
    a fresh budget rather than inheriting a dead one.
    """

    def __init__(
        self,
        budget_seconds: int,
        source: str = "unknown",
        grace_seconds: int = WRAP_UP_GRACE_SECONDS,
        clock=time.monotonic,
    ):
        self.budget_seconds = max(0, int(budget_seconds))
        self.source = source
        self.grace_seconds = max(0, int(grace_seconds))
        self._clock = clock
        self._start = clock()

    def elapsed(self) -> float:
        return max(0.0, self._clock() - self._start)

    def remaining(self) -> float:
        """Seconds left before the budget is spent (never negative)."""
        return max(0.0, self.budget_seconds - self.elapsed())

    def grace_remaining(self) -> float:
        """Seconds left before the hard cut (never negative)."""
        return max(0.0, self.budget_seconds + self.grace_seconds - self.elapsed())

    def soft_expired(self) -> bool:
        return self.remaining() <= 0

    def hard_expired(self) -> bool:
        return self.grace_remaining() <= 0

    def clamp_sleep(self, seconds: float) -> float:
        """Cap a poll interval so sleeping cannot overrun the budget.

        Returns 0 once the budget is spent: check_process sleeping through the
        grace window is what turned a 15-minute Step 8 into a 50-minute one.
        """
        return max(0.0, min(float(seconds), self.remaining()))

    def clamp_timeout(
        self, seconds: float, minimum: int = MIN_COMMAND_TIMEOUT
    ) -> int:
        """Cap a command timeout at the hard cut, with a usable floor.

        Measured against the grace window rather than the budget: killing a
        pytest run that has 20 seconds left to go accomplishes nothing, and the
        wrap-up turn still needs to be able to read a log.
        """
        return max(minimum, min(int(seconds), int(self.grace_remaining())))

    def describe_remaining(self) -> str:
        """Human-readable time left, e.g. "4m12s" or "over budget by 1m30s"."""
        if self.soft_expired():
            over = int(self.elapsed() - self.budget_seconds)
            return f"over budget by {over // 60}m{over % 60:02d}s"
        left = int(self.remaining())
        return f"{left // 60}m{left % 60:02d}s remaining"

    def status_line(self) -> str:
        return (
            f"[budget] {self.budget_seconds // 60} min for this step; "
            f"{self.describe_remaining()}"
        )


def wrap_up_message(step_num: float, deadline: StepDeadline) -> str:
    """The turn injected once a step goes over budget."""
    elapsed_min = int(deadline.elapsed()) // 60
    grace_min = max(1, deadline.grace_seconds // 60)
    return (
        f"[budget exhausted -- {elapsed_min} min elapsed of the "
        f"{deadline.budget_seconds // 60} min wall-clock budget for Step "
        f"{step_num:g}. You have about {grace_min} more minute(s) before this "
        "step is cut off. Stop starting new work: no new training runs, "
        "sweeps, or files unless a file is needed to record results. Any "
        "background jobs still running will be stopped, so read what their "
        "logs already show and report from the latest checkpoint. Write your "
        "final summary now -- state what was established, what is still "
        "unresolved, and what the next step should do.]"
    )


def append_user_request(messages: list[dict], request: str) -> list[dict]:
    """Return `messages` plus `request`, without breaking role alternation.

    Converse requires alternating roles, so a new user message cannot follow a
    user message. That happens whenever the tool loop exits with a pending
    tool_result turn -- previously only on the tool-round cap, now routinely on
    a wall-clock hard stop. Folding the request into the trailing user turn
    keeps the tool_use/tool_result pairing intact.
    """
    out = list(messages)
    if not out or out[-1].get("role") != "user":
        return out + [{"role": "user", "content": request}]

    last = dict(out[-1])
    content = last.get("content")
    if isinstance(content, str):
        last["content"] = f"{content}\n\n{request}"
    elif isinstance(content, list):
        last["content"] = list(content) + [{"type": "text", "text": request}]
    else:
        return out + [{"role": "user", "content": request}]
    out[-1] = last
    return out


# ── Tool description fragments ───────────────────────────────────────────
#
# Both engines run commands with cwd set to the session's code/ directory.
# Some generated commands redundantly change into or list `code/` from inside
# that directory, so the contract is stated in the tool schema itself.

def normalize_code_relpath(rel_path: str) -> str:
    """Strip a redundant leading `code/` from a path meant to be inside code/.

    The model regularly writes `code/train.py` when cwd is already code/, which
    would create `code/code/train.py`. `command_compat._resolve_code_path` does
    the same for the shell shims; this keeps write_file consistent with it.
    """
    path = PurePosixPath(str(rel_path).replace("\\", "/"))
    parts = path.parts
    if parts and parts[0].lower() == "code":
        parts = parts[1:]
    return str(PurePosixPath(*parts)) if parts else "."


CODE_DIR_CONTRACT = (
    "The working directory is ALWAYS the session's code/ directory: all paths "
    "are relative to it, so use 'train.py' and 'experiments/', never "
    "'code/train.py' and never a 'cd' into the session directory. A leading "
    "'code/' is stripped when present."
)


# ── Shared tool schemas ──────────────────────────────────────────────────

TOOLS = [
    {
        "name": "write_file",
        "description": (
            "Write content to a file in the session's code/ directory. "
            "Creates parent directories automatically. Use for Python files, "
            "config files, requirements.txt, etc. This is the ONLY approved "
            "way to create or edit a file: rewrite the whole file here rather "
            "than patching it in the shell."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Relative path within the code/ directory "
                        "(e.g. 'environment.py', 'tests/test_env.py')"
                    ),
                },
                "content": {
                    "type": "string",
                    "description": "Full file content to write",
                },
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "run_command",
        "description": (
            "Execute a shell command in the session's code/ directory. "
            "Returns stdout and stderr. Use for short commands: pip install, "
            "pytest, dry runs, quick scripts. Timeout: 10 minutes. "
            "For training or long-running jobs, use run_background instead. "
            f"{CODE_DIR_CONTRACT} "
            "Do not edit files here (no sed -i, no heredocs, no 'cat >'); "
            "use write_file so the edit is recorded in the transcript."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute",
                },
            },
            "required": ["command"],
        },
    },
    {
        "name": "run_background",
        "description": (
            "Start a long-running command in the background (e.g. training). "
            "Output streams to a log file in code/. Returns immediately with "
            "a PID. Use check_process to monitor progress. "
            f"{CODE_DIR_CONTRACT}"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to run in the background",
                },
                "log_file": {
                    "type": "string",
                    "description": (
                        "Log file path relative to code/ "
                        "(e.g. 'train.log', 'logs/run1.log')"
                    ),
                },
            },
            "required": ["command", "log_file"],
        },
    },
    {
        "name": "check_process",
        "description": (
            "Check if a background process is still running and read the "
            "tail of its log file. Use after run_background to monitor progress."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "log_file": {
                    "type": "string",
                    "description": "Log file path (same as passed to run_background)",
                },
                "tail_lines": {
                    "type": "integer",
                    "description": "Number of lines to read from the end (default 50)",
                },
            },
            "required": ["log_file"],
        },
    },
]

LITERATURE_TOOL = {
    "name": "search_literature",
    "description": (
        "Search academic literature (arXiv, OpenAlex, Semantic Scholar) for "
        "papers related to the given query. Returns titles, authors, years, "
        "citation counts, and abstracts. Queries are matched as an AND of the "
        "distinctive terms, so 4-6 keywords work better than a long phrase. "
        "If the result reports backend failures, that is a transport problem — "
        "rephrasing will not help. "
        f"At most {MAX_LITERATURE_SEARCHES} searches are permitted per step, so "
        "spend them on distinct topics rather than rewordings of one query."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query (e.g., 'reinforcement learning portfolio optimization')",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results (default: 10, max: 20)",
            },
        },
        "required": ["query"],
    },
}


# ── /suggest prompt templates ────────────────────────────────────────────

SUGGEST_SYNTHESIS_SYSTEM = """\
You are CARL (Chat Assisted RL), an expert in reinforcement learning problem design.
Synthesize 3 to 5 distinct, concrete RL problem formulations inspired by the papers below.

For EACH formulation output EXACTLY this format:

**[N]. <Short Title>**
- Domain framing: one sentence describing the setup
- Core objective: what the agent optimises (one sentence)
- Key feature: what makes this formulation distinct
- Primary reference: paper title and year

Blank line between formulations. No preamble, no conclusion.\
"""

SUGGEST_SYNTHESIS_USER = """\
User's problem description:
{problem_description}

Papers found (title, year, abstract excerpt):
{papers_formatted}

Generate 3-5 candidate RL problem formulations.\
"""

SUGGEST_DOCUMENT_SYSTEM = """\
You are CARL (Chat Assisted RL). Write a detailed RL problem specification in Markdown.
Include these sections in order, using these exact headings:

# <Problem Title>
## Overview
## Objective
## State Space
## Action Space
## Reward Function
## Episode Definition
## Constraints
## Implementation Notes

Be technically precise. Use LaTeX inline math ($...$) for formulas.
State Space: bullet list with name, description, range/units.
Action Space: type, dimensionality, bounds.
Reward Function: per-step formula, terminal reward, discount factor γ with rationale.
Implementation Notes: 3-5 practical coding considerations.\
"""

SUGGEST_DOCUMENT_USER = """\
Chosen RL problem formulation:
{chosen_formulation}

User's original description:
{problem_description}

Supporting papers:
{papers_formatted}

Write the full specification document.\
"""
