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

"""Session management for CARL -- directory creation and memory persistence."""

import json
import shutil
from datetime import datetime
from pathlib import Path


class Session:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        self.session_dir = self.base_dir / "sessions" / self.timestamp

        self.session_dir.mkdir(parents=True, exist_ok=True)
        (self.session_dir / "references").mkdir(exist_ok=True)
        (self.session_dir / "code").mkdir(exist_ok=True)
        (self.session_dir / "results").mkdir(exist_ok=True)

        self.memory_path = self.session_dir / "memory.md"
        self.memory_path.write_text(
            f"# CARL Session -- {self.timestamp}\n\n",
            encoding="utf-8",
        )

        self.step_decisions: dict[float, str] = {}
        self.step_budgets: dict[float, int] = {}

    @classmethod
    def open_existing(cls, base_dir: str, session_dir: str | Path) -> "Session":
        """Open an existing CARL session without creating or overwriting files."""
        obj = cls.__new__(cls)
        obj.base_dir = Path(base_dir)
        path = Path(session_dir)
        if not path.is_absolute():
            path = obj.base_dir / path
        obj.session_dir = path.resolve()
        obj.timestamp = obj.session_dir.name
        obj.memory_path = obj.session_dir / "memory.md"
        if not obj.session_dir.is_dir():
            raise FileNotFoundError(f"Session directory not found: {obj.session_dir}")
        if not obj.memory_path.is_file():
            raise FileNotFoundError(f"Session memory not found: {obj.memory_path}")
        obj.step_decisions = {}
        obj.step_budgets = {}
        obj.load_step_budgets()
        return obj

    # ── Step time budgets ─────────────────────────────────────────────
    #
    # Step 7 names wall-clock budgets for Steps 8 and 9. They are stored here,
    # parsed from Step 7's raw turn text, because the step *summary* is not a
    # reliable carrier: the summarizer is asked for concise bullets and drops
    # machine-readable lines, which silently reverted one session to the
    # default budget. A small JSON file also survives --resume, where
    # step_decisions are reconstructed from memory.md.

    @property
    def budget_path(self) -> Path:
        return self.session_dir / "step_budgets.json"

    def set_step_budget(self, step_number, minutes: int):
        """Record a step's wall-clock budget in minutes and persist it."""
        self.step_budgets[step_number] = int(minutes)
        payload = {str(k): v for k, v in sorted(self.step_budgets.items())}
        self.budget_path.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )

    def load_step_budgets(self) -> dict[float, int]:
        """Load persisted step budgets; missing or unreadable file means none."""
        try:
            raw = json.loads(self.budget_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return self.step_budgets
        for key, value in raw.items():
            try:
                self.step_budgets[float(key)] = int(value)
            except (TypeError, ValueError):
                continue
        return self.step_budgets

    def get_step_budget_minutes(self, step_number) -> int | None:
        """Persisted budget for a step, or None if Step 7 never named one."""
        value = self.step_budgets.get(step_number)
        return int(value) if value else None

    # ── Persistence ───────────────────────────────────────────────────

    @staticmethod
    def _step_label(step_number) -> str:
        """Format a step number for filenames (handles floats like 2.5)."""
        if isinstance(step_number, float) and not step_number.is_integer():
            return f"{step_number:04.1f}"
        return f"{int(step_number):02d}"

    def save_step(
        self,
        step_number,
        step_name: str,
        summary: str,
        conversation: list[dict],
        msg_origins: "dict[int, dict] | None" = None,
    ):
        """Append a step's summary to session memory and save the full conversation."""
        with open(self.memory_path, "a", encoding="utf-8") as f:
            f.write(f"## Step {step_number}: {step_name}\n\n")
            f.write(f"{summary}\n\n---\n\n")

        self.step_decisions[step_number] = summary

        conv_path = self.session_dir / f"step_{self._step_label(step_number)}_conversation.md"
        with open(conv_path, "w", encoding="utf-8") as f:
            f.write(f"# Step {step_number}: {step_name} -- Full Conversation\n\n")
            for i, msg in enumerate(conversation):
                origin = (msg_origins or {}).get(i)
                f.write(self._format_message(msg, origin))

    @staticmethod
    def _format_content(content) -> str:
        """Format message content for saving — handles plain strings and block lists."""
        if isinstance(content, str):
            return content
        parts = []
        for block in content:
            btype = block.get("type", "")
            if btype == "text":
                parts.append(block["text"])
            elif btype == "tool_use":
                inp = json.dumps(block.get("input", {}), indent=2)
                if len(inp) > 300:
                    inp = inp[:300] + "...(truncated)"
                parts.append(f"[Tool: {block['name']}]\n```\n{inp}\n```")
            elif btype == "tool_result":
                result = str(block.get("content", ""))
                if len(result) > 500:
                    result = result[:500] + "...(truncated)"
                parts.append(f"[Tool Result]\n```\n{result}\n```")
        return "\n".join(parts)

    def _format_message(self, msg: dict, origin: "dict | None") -> str:
        """Format a single message for the conversation log.

        For user messages that involved Trillian, expands into separate
        sections for the suggestion, raw user input, and merged result.
        """
        if msg["role"] == "assistant":
            content = self._format_content(msg["content"])
            return f"**CARL:**\n{content}\n\n"

        # User message — check for Trillian origin
        if origin is None:
            content = self._format_content(msg["content"])
            return f"**User:**\n{content}\n\n"

        otype = origin.get("type", "")

        if otype == "trillian_accepted":
            trillian = origin["trillian"]
            return (
                f"**Trillian suggestion (accepted by user):**\n{trillian}\n\n"
            )

        if otype == "trillian_edited":
            trillian = origin["trillian"]
            final = origin["final"]
            return (
                f"**Trillian suggestion:**\n{trillian}\n\n"
                f"**User (edited and sent to CARL):**\n{final}\n\n"
            )

        # Fallback — unknown origin type
        content = self._format_content(msg["content"])
        return f"**User:**\n{content}\n\n"

    def save_conversation(
        self,
        step_number,
        step_name: str,
        conversation: list[dict],
        msg_origins: "dict[int, dict] | None" = None,
    ):
        """Save a step's conversation to disk without a summary.

        Used for auto-saving during long steps and crash recovery.
        Overwrites the file each call so the latest state is always on disk.
        """
        conv_path = self.session_dir / f"step_{self._step_label(step_number)}_conversation.md"
        with open(conv_path, "w", encoding="utf-8") as f:
            f.write(f"# Step {step_number}: {step_name} -- Full Conversation (partial)\n\n")
            for i, msg in enumerate(conversation):
                origin = (msg_origins or {}).get(i)
                f.write(self._format_message(msg, origin))

    def get_previous_decisions(self) -> str:
        """Format all completed step decisions for the system prompt."""
        if not self.step_decisions:
            return "No previous steps completed yet."

        parts = []
        for num in sorted(self.step_decisions):
            parts.append(f"### Step {num}\n{self.step_decisions[num]}")
        return "\n\n".join(parts)

    # ── Rename ─────────────────────────────────────────────────────────

    def rename_with_topic(self, slug: str):
        """Rename the session directory to include a topic slug.

        Transforms e.g. sessions/2026-03-06_10-21-25
                   into sessions/portfolio_optimization_2026-03-06_10-21-25
        """
        new_dir = self.session_dir.parent / f"{slug}_{self.timestamp}"
        if new_dir.exists():
            return  # avoid collision
        self.session_dir.rename(new_dir)
        self.session_dir = new_dir
        self.memory_path = new_dir / "memory.md"

    # ── File handling ─────────────────────────────────────────────────

    def copy_reference(self, source_path: str) -> Path | None:
        """Copy a reference file into the session's references/ directory."""
        src = Path(source_path)
        if not src.exists():
            return None
        dest = self.session_dir / "references" / src.name
        shutil.copy2(src, dest)
        return dest

    @staticmethod
    def read_file(file_path: str) -> str | None:
        """Read a text file, returning None on failure."""
        try:
            return Path(file_path).read_text(encoding="utf-8", errors="replace")
        except (OSError, UnicodeDecodeError):
            return None
