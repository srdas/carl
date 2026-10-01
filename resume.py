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

"""Shared resume helpers for CARL step-boundary recovery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys

from steps import STEPS


MARKER_PREFIX = "CARL_RESUME"


class ResumeError(RuntimeError):
    """Raised when a session cannot be resumed safely."""


@dataclass
class ResumeInfo:
    session_dir: Path
    step_idx: int
    step_number: float | int
    step_name: str
    step_decisions: dict[float | int, str]
    suggest_used: bool = False
    user_skip_25: bool = False
    complete: bool = False
    warnings: list[str] | None = None


def _step_num(value: str) -> float | int:
    num = float(value)
    return int(num) if num.is_integer() else num


def _workflow_numbers(skip_25: bool = False) -> list[float | int]:
    return [
        step["number"]
        for step in STEPS
        if not (skip_25 and step["number"] == 2.5)
    ]


def _step_index(number: float | int) -> int:
    for i, step in enumerate(STEPS):
        if step["number"] == number:
            return i
    raise ResumeError(f"Unknown CARL step in resume log: {number}")


def _first_missing_step(
    saved_steps: set[float | int],
    skip_25: bool,
) -> float | int | None:
    for number in _workflow_numbers(skip_25):
        if number not in saved_steps:
            return number
    return None


def format_resume_marker(event: str, step_number: float | int | None = None) -> str:
    if step_number is None:
        return f"{MARKER_PREFIX} {event}"
    return f"{MARKER_PREFIX} {event} number={step_number}"


def emit_resume_marker(event: str, step_number: float | int | None = None) -> None:
    print(format_resume_marker(event, step_number), flush=True)


class _TeeWriter:
    def __init__(self, original, log_file):
        self._original = original
        self._log_file = log_file
        self._ansi_re = re.compile(r"\033\[[0-9;]*m")

    def write(self, text):
        self._original.write(text)
        try:
            self._log_file.write(self._ansi_re.sub("", text))
            self._log_file.flush()
        except Exception:
            pass

    def flush(self):
        self._original.flush()
        try:
            self._log_file.flush()
        except Exception:
            pass

    def fileno(self):
        return self._original.fileno()

    def isatty(self):
        return self._original.isatty()


class SessionLogTee:
    """Mirror stdout/stderr into a session-local web.log."""

    def __init__(self, session_dir: str | Path):
        self.session_dir = Path(session_dir)
        self._stdout = None
        self._stderr = None
        self._handle = None

    def start(self) -> None:
        if self._handle is not None:
            return
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        self._handle = open(self.session_dir / "web.log", "a", encoding="utf-8")
        sys.stdout = _TeeWriter(self._stdout, self._handle)
        sys.stderr = _TeeWriter(self._stderr, self._handle)

    def stop(self) -> None:
        if self._handle is None:
            return
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        try:
            self._handle.close()
        finally:
            self._handle = None
            self._stdout = None
            self._stderr = None

    def restart(self, session_dir: str | Path) -> None:
        was_started = self._handle is not None
        self.stop()
        self.session_dir = Path(session_dir)
        if was_started:
            self.start()


def resolve_resume_session(carl_dir: str | Path, session_dir: str | Path) -> Path:
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
        raise ValueError(f"Resume path must be under {sessions_root}: {path}") from exc
    if not path.is_dir():
        raise FileNotFoundError(f"Session directory not found: {path}")
    if not (path / "memory.md").is_file():
        raise FileNotFoundError(f"Not a CARL session: missing {path / 'memory.md'}")
    if not (path / "web.log").is_file():
        raise FileNotFoundError(f"Cannot resume session without {path / 'web.log'}")
    return path


def parse_memory_decisions(memory_text: str) -> dict[float | int, str]:
    """Parse completed step summaries from memory.md.

    A step can appear more than once: Session.save_step appends its own
    `## Step N:` heading, and CARL has additionally appended notes with
    `cat >> ../memory.md` during Steps 8-9. Bodies for a repeated step are
    concatenated in document order so no prior decision is dropped -- keeping
    only the last block silently lost the controller's own summary.
    """
    heading_re = re.compile(r"^## Step\s+(\d+(?:\.\d+)?):[^\n]*\n", re.MULTILINE)
    matches = list(heading_re.finditer(memory_text))
    decisions: dict[float | int, str] = {}
    for i, match in enumerate(matches):
        number = _step_num(match.group(1))
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(memory_text)
        summary = memory_text[start:end].strip()
        if summary.endswith("---"):
            summary = summary[:-3].rstrip()
        existing = decisions.get(number)
        if existing:
            decisions[number] = f"{existing}\n\n{summary}" if summary else existing
        else:
            decisions[number] = summary
    return decisions


def _looks_like_skip_25(decisions: dict[float | int, str], log_text: str) -> bool:
    step2 = decisions.get(2, "")
    haystack = f"{step2}\n{log_text}".lower()
    return (
        ("step 2.5" in haystack or "literature search" in haystack)
        and "skip" in haystack
    )


def _looks_like_suggest_used(log_text: str) -> bool:
    return "[/suggest]" in log_text.lower()


def _parse_structured_markers(
    log_text: str,
    skip_25: bool,
) -> tuple[float | int | None, bool, list[str]] | None:
    marker_re = re.compile(
        rf"^{MARKER_PREFIX}\s+(\w+)(?:\s+number=(\d+(?:\.\d+)?))?",
        re.MULTILINE,
    )
    matches = list(marker_re.finditer(log_text))
    if not matches:
        return None

    saved: set[float | int] = set()
    last_started: float | int | None = None
    complete = False
    warnings: list[str] = []

    for match in matches:
        event = match.group(1)
        number = _step_num(match.group(2)) if match.group(2) else None
        if event == "step_start" and number is not None and number in _workflow_numbers(False):
            last_started = number
        elif event == "step_saved" and number is not None and number in _workflow_numbers(False):
            saved.add(number)
        elif event == "session_done":
            complete = True

    if complete:
        return None, True, warnings
    if last_started is not None and last_started not in saved:
        return last_started, False, warnings

    target = _first_missing_step(saved, skip_25)
    if target is None:
        return None, True, warnings
    return target, False, warnings


def _legacy_step_starts(log_text: str) -> list[tuple[int, float | int]]:
    starts: list[tuple[int, float | int]] = []
    patterns = [
        re.compile(r"^#{1,2}\s+Step\s+(\d+(?:\.\d+)?):", re.MULTILINE),
        re.compile(
            r"Ready to proceed to\s+(?:\*\*)?Step\s+(\d+(?:\.\d+)?)",
            re.IGNORECASE,
        ),
    ]
    for pattern in patterns:
        for match in pattern.finditer(log_text):
            number = _step_num(match.group(1))
            if number in _workflow_numbers(False):
                starts.append((match.start(), number))
    starts.sort(key=lambda item: item[0])
    return starts


def _parse_legacy_log(
    log_text: str,
    decisions: dict[float | int, str],
    skip_25: bool,
) -> tuple[float | int | None, bool, list[str]]:
    warnings: list[str] = []
    if (
        "[DONE]" in log_text
        or "All steps complete!" in log_text
        or "CARL workflow complete" in log_text
    ):
        return None, True, warnings

    starts = _legacy_step_starts(log_text)
    last_started = starts[-1][1] if starts else None
    last_started_pos = starts[-1][0] if starts else -1
    tail = log_text[last_started_pos:] if last_started_pos >= 0 else log_text
    interrupted = any(
        marker in tail
        for marker in (
            "[Interrupted",
            "Interrupted - cleaning up",
            "API error",
            "LLM call failed",
            "Server error",
            "Traceback",
        )
    )

    if last_started is not None and interrupted:
        return last_started, False, warnings

    saved = set(decisions)
    target = _first_missing_step(saved, skip_25)
    if target is None:
        return None, True, warnings
    if last_started is not None and last_started not in saved:
        target = last_started
    return target, False, warnings


def build_resume_info(carl_dir: str | Path, session_dir: str | Path) -> ResumeInfo:
    session_path = resolve_resume_session(carl_dir, session_dir)
    memory_text = (session_path / "memory.md").read_text(
        encoding="utf-8", errors="replace"
    )
    log_text = (session_path / "web.log").read_text(
        encoding="utf-8", errors="replace"
    )
    all_decisions = parse_memory_decisions(memory_text)
    user_skip_25 = _looks_like_skip_25(all_decisions, log_text)
    suggest_used = _looks_like_suggest_used(log_text)
    skip_25 = user_skip_25 or suggest_used

    structured = _parse_structured_markers(log_text, skip_25)
    if structured is None:
        target, complete, warnings = _parse_legacy_log(log_text, all_decisions, skip_25)
    else:
        target, complete, warnings = structured

    if complete or target is None:
        raise ResumeError(
            "Session appears complete; use --revisit for post-training analysis."
        )

    step_idx = _step_index(target)
    prior = {
        number: summary
        for number, summary in all_decisions.items()
        if _step_index(number) < step_idx
    }
    future_saved = [
        number for number in all_decisions
        if number in _workflow_numbers(False) and _step_index(number) >= step_idx
    ]
    if future_saved:
        warnings.append(
            "memory.md contains summaries for resumed/current or later steps; "
            "web.log is being used as the resume authority."
        )

    step = STEPS[step_idx]
    return ResumeInfo(
        session_dir=session_path,
        step_idx=step_idx,
        step_number=step["number"],
        step_name=step["name"],
        step_decisions=prior,
        suggest_used=suggest_used,
        user_skip_25=user_skip_25,
        complete=False,
        warnings=warnings,
    )
