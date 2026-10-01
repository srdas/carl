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

"""Small cross-platform command shims for CARL tool execution."""

from __future__ import annotations

import shlex
from pathlib import Path


def _resolve_code_path(cwd: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    parts = path.parts
    if parts and parts[0].lower() == "code":
        path = Path(*parts[1:]) if len(parts) > 1 else Path(".")
    return cwd / path


def _tail_file(path: Path, lines: int) -> str:
    if not path.exists():
        return f"tail: cannot open '{path}' for reading: No such file or directory"
    if path.is_dir():
        return f"tail: error reading '{path}': Is a directory"
    text = path.read_text(errors="replace")
    return "\n".join(text.splitlines()[-lines:])


def run_compat_command(command: str, cwd: Path) -> str | None:
    """Return output for supported Unix-style commands, or None to run normally."""
    try:
        tokens = shlex.split(command, posix=False)
    except ValueError:
        return None
    if not tokens or tokens[0].lower() != "tail":
        return None

    lines = 10
    paths: list[str] = []
    i = 1
    while i < len(tokens):
        token = tokens[i]
        if token == "-n" and i + 1 < len(tokens):
            i += 1
            try:
                lines = max(0, int(tokens[i]))
            except ValueError:
                return f"tail: invalid number of lines: '{tokens[i]}'"
        elif token.startswith("-n") and len(token) > 2:
            try:
                lines = max(0, int(token[2:]))
            except ValueError:
                return f"tail: invalid option '{token}'"
        elif token.startswith("-") and token[1:].isdigit():
            lines = int(token[1:])
        elif token == "-f":
            return "tail -f is not supported by run_command; use check_process for live logs."
        elif token.startswith("-"):
            return f"tail: unsupported option '{token}'"
        else:
            paths.append(token.strip("\"'"))
        i += 1

    if not paths:
        return "tail: missing file operand"

    outputs: list[str] = []
    multi = len(paths) > 1
    for raw in paths:
        path = _resolve_code_path(cwd, raw)
        if multi:
            outputs.append(f"==> {raw} <==")
        outputs.append(_tail_file(path, lines))
    return "\n".join(outputs)
