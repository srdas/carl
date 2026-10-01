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

"""Chat engine for CARL -- LLM interaction, tool use, and step progression."""

import json
import os
import re
import select
import subprocess
import sys
from datetime import date
from pathlib import Path

try:
    import readline
except ImportError:  # Windows fallback: no prefill support, plain input still works.
    class _ReadlineFallback:
        @staticmethod
        def set_startup_hook(*args, **kwargs):
            return None

    readline = _ReadlineFallback()

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from bedrock_converse import (
    converse_stream,
    converse_text,
    thinking_config_for_step,
    thinking_max_tokens_for_step,
)
from command_compat import run_compat_command
from conversation_guards import (
    MISSING_QUESTION_CONTINUE_PROMPT,
    asks_user_question,
    should_continue_for_missing_question,
)
from bedrock_models import (
    DEFAULT_MODEL as _DEFAULT_MODEL,
    DEFAULT_TRILLIAN_MODEL as _DEFAULT_TRILLIAN_MODEL,
    DEFAULT_PERSONA_UPDATE_MODEL as _DEFAULT_PERSONA_UPDATE_MODEL,
    DEFAULT_REPORT_MODEL as _DEFAULT_REPORT_MODEL,
    DEFAULT_REGION as _DEFAULT_REGION,
    resolve_model_id,
)
from session import Session
from steps import (
    STEPS, REVISIT_STEP, load_skill, build_system_prompt,
    load_trillian_persona, trillian_persona_path, build_trillian_prompt,
    _TRILLIAN_LAST_ACTIVE_STEP,
)
from revisit import (
    REVISIT_TOOLS,
    build_revisit_preflight_prompt,
    build_revisit_context,
    describe_plot,
    discover_artifacts,
    extract_revisit_exclusions,
    normalize_revisit_guidance,
    prepare_revisit_workspace_tool_response,
    read_artifact,
    resolve_revisit_session,
)
from resume import (
    SessionLogTee,
    build_resume_info,
    emit_resume_marker,
)
from tool_runtime import (
    CHECK_PROCESS_MIN_WAIT,
    LITERATURE_LIMIT_MESSAGE,
    LITERATURE_TOOL,
    MAX_LITERATURE_SEARCHES,
    MAX_TOOL_ROUNDS,
    STEP9_DEFAULT_BUDGET_SECONDS,
    SUGGEST_DOCUMENT_SYSTEM,
    SUGGEST_DOCUMENT_USER,
    SUGGEST_SYNTHESIS_SYSTEM,
    SUGGEST_SYNTHESIS_USER,
    TESTING_BUDGET_MARKER,
    TOOLS,
    TRAINING_BUDGET_MARKER,
    StepDeadline,
    append_user_request,
    check_interval_seconds,
    is_revisit_training_command,
    normalize_code_relpath,
    resolve_step_budget,
    revisit_training_refusal,
    wrap_up_message,
)

# Steps where the model can write files and run commands
_TOOL_STEPS = {8, 9}
_ONE_TURN_STEPS = {7}
_LITERATURE_STEPS = {2.5}
_MAX_TOOL_ROUND_TAIL_CONTINUES = 5
_MAX_AUTO_ROUNDS = 20
# Retries allowed when a turn streams back no text at all (all of maxTokens
# spent on reasoning); each retry doubles max_tokens up to _MAX_OUTPUT_TOKENS.
_MAX_BLANK_TURNS = 2
_MAX_OUTPUT_TOKENS = 32000
# Highest thinking tier a tool-less (prose-only) turn may use. Measured on a
# Step 3 prompt: high spends ~15.9k output tokens for 9.5k characters, medium
# ~7.8k for 8.0k characters. Tool turns keep the step's own tier.
_TEXT_TURN_TIER_CEILING = "medium"

# Shared with web.py via tool_runtime; aliased so existing call sites read
# the same as before.
_MAX_TOOL_ROUNDS = MAX_TOOL_ROUNDS
_STEP9_DEFAULT_BUDGET_SECONDS = STEP9_DEFAULT_BUDGET_SECONDS
_CHECK_PROCESS_MIN_WAIT = CHECK_PROCESS_MIN_WAIT
_SUGGEST_SYNTHESIS_SYSTEM = SUGGEST_SYNTHESIS_SYSTEM
_SUGGEST_SYNTHESIS_USER = SUGGEST_SYNTHESIS_USER
_SUGGEST_DOCUMENT_SYSTEM = SUGGEST_DOCUMENT_SYSTEM
_SUGGEST_DOCUMENT_USER = SUGGEST_DOCUMENT_USER


def _load_dotenv(carl_dir: str):
    """Load .env file into os.environ (for API keys)."""
    env_path = Path(carl_dir) / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip("'\""))


def _build_bedrock_client(region: str):
    """Create a boto3 bedrock-runtime client using the default AWS credential chain."""
    config = Config(
        read_timeout=900,
        connect_timeout=60,
        retries={"max_attempts": 3},
    )
    session = boto3.Session(region_name=region)
    return session.client(service_name="bedrock-runtime", config=config)


def _resolve_model_id(client, preferred: str) -> str:
    """Try the preferred model; fall back to 4.6 if unavailable."""
    return resolve_model_id(client, preferred, log=lambda msg: print(msg, flush=True))


class CARLChat:
    def __init__(
        self,
        carl_dir: str,
        write_report: bool = False,
        no_persona_update: bool = False,
        revisit_dir: str | None = None,
        resume_dir: str | None = None,
    ):
        if revisit_dir and resume_dir:
            raise ValueError("--resume and --revisit are mutually exclusive")
        self.carl_dir = carl_dir
        self.write_report = write_report
        self.no_persona_update = no_persona_update
        self.resume_info = build_resume_info(carl_dir, resume_dir) if resume_dir else None
        self.resume_mode = self.resume_info is not None
        self.revisit_dir = resolve_revisit_session(carl_dir, revisit_dir) if revisit_dir else None
        self.revisit_mode = self.revisit_dir is not None
        self._revisit_completed = False
        _load_dotenv(carl_dir)
        self.region = os.environ.get("AWS_REGION", _DEFAULT_REGION)
        self.trillian_model = os.environ.get("CARL_TRILLIAN_MODEL", _DEFAULT_TRILLIAN_MODEL)
        self.persona_update_model = os.environ.get("CARL_PERSONA_UPDATE_MODEL", _DEFAULT_PERSONA_UPDATE_MODEL)
        self.report_model = os.environ.get("CARL_REPORT_MODEL", _DEFAULT_REPORT_MODEL)
        self.client = _build_bedrock_client(self.region)
        self.model = self._resolve_model()
        self.session = (
            Session.open_existing(carl_dir, self.revisit_dir)
            if self.revisit_mode
            else Session.open_existing(carl_dir, self.resume_info.session_dir)
            if self.resume_mode
            else Session(carl_dir)
        )
        if self.resume_mode:
            self.session.step_decisions = dict(self.resume_info.step_decisions)
        self.current_step_idx = self.resume_info.step_idx if self.resume_mode else 0
        self._current_step: dict | None = None
        self._current_messages: list[dict] = []
        self._current_system: str = ""
        # Wall-clock deadline for the running step (Steps 8/9 only).
        self.deadline: StepDeadline | None = None
        self.wrap_up_sent = False
        self.hard_stop_done = False
        self._trillian_persona = load_trillian_persona(carl_dir)
        self._trillian_suggestion: str | None = None
        self._current_msg_origins: dict[int, dict] = {}
        self._literature_search_counts: dict[float | int | None, int] = {}
        self._revisit_guidance = ""
        self._revisit_exclusions: list[str] = []
        self._suggest_used = self.resume_info.suggest_used if self.resume_mode else False
        self._user_skip_25 = self.resume_info.user_skip_25 if self.resume_mode else False
        self._session_log = None if self.revisit_mode else SessionLogTee(self.session.session_dir)
        if self._session_log is not None:
            self._session_log.start()

    # ── Model resolution ──────────────────────────────────────────────

    def _resolve_model(self) -> str:
        return _resolve_model_id(self.client, os.environ.get("CARL_MODEL", _DEFAULT_MODEL))

    # ── Main loop ─────────────────────────────────────────────────────

    def run(self):
        """Run the full CARL workflow, step by step."""
        self._print_banner()

        if self.revisit_mode:
            self._print_revisit_banner()
            try:
                result = self._run_step(REVISIT_STEP)
                self._revisit_completed = result == "next"
            except KeyboardInterrupt:
                print("\n\n  \033[31m[Interrupted]\033[0m")
            except Exception as e:
                print(f"\n\n  \033[31m[Error: {e}]\033[0m")
            finally:
                self._save_on_exit()
            print(f"\n  Step 10 revisit complete. Session saved: {self.session.session_dir}")
            self._close_session_log()
            return

        if self.resume_mode:
            self._print_resume_banner()

        try:
            while self.current_step_idx < len(STEPS):
                step = STEPS[self.current_step_idx]
                result = self._run_step(step)
                if result == "next":
                    if step["number"] == 1:
                        self._rename_session_dir()
                    self.current_step_idx += 1
                    # Skip Step 2.5 if /suggest already did the literature search
                    next_step = STEPS[self.current_step_idx] if self.current_step_idx < len(STEPS) else None
                    if next_step and next_step["number"] == 2.5 and (self._suggest_used or self._user_skip_25):
                        reason = "already covered by /suggest" if self._suggest_used else "skipped per user request"
                        print(f"\n  \033[36m[Skipping Step 2.5: Literature Search — {reason}]\033[0m")
                        self.current_step_idx += 1
                    next_step = STEPS[self.current_step_idx] if self.current_step_idx < len(STEPS) else None
                    if next_step and next_step["number"] in _TOOL_STEPS:
                        print(f"\n  \033[36m[Step {next_step['number']} involves code generation and execution. Responses may take longer than usual.]\033[0m\n")
                elif result == "quit":
                    break
        except KeyboardInterrupt:
            print("\n\n  \033[31m[Interrupted]\033[0m")
        except Exception as e:
            print(f"\n\n  \033[31m[Error: {e}]\033[0m")
        finally:
            self._save_on_exit()

        if self.current_step_idx >= len(STEPS):
            if not self.no_persona_update:
                self._update_trillian_persona()
            if self.write_report:
                print("\n  Generating academic report...")
                from report import generate_report
                report_path = generate_report(
                    self.session, self.client, self.report_model,
                )
                if report_path:
                    print(f"  Report saved: {report_path}")
                else:
                    print("  \033[33m[Report generation failed. Session data is still saved.]\033[0m")
            emit_resume_marker("session_done")
            self._print_done()
        self._close_session_log()

    # ── Single step ───────────────────────────────────────────────────

    def _run_step(self, step: dict) -> str:
        """Run one workflow step. Returns 'next' or 'quit'."""
        step_num = step["number"]
        is_revisit_step = self.revisit_mode and step_num == REVISIT_STEP["number"]
        if step_num in _LITERATURE_STEPS:
            tools = [LITERATURE_TOOL]
        elif is_revisit_step:
            tools = REVISIT_TOOLS
        elif step_num in _TOOL_STEPS:
            tools = TOOLS
        else:
            tools = None

        self._print_step_header(step_num, step["name"])
        if not is_revisit_step:
            emit_resume_marker("step_start", step_num)

        preflight_prompt: str | None = None
        preflight_reply: str | None = None
        if is_revisit_step:
            preflight_prompt = build_revisit_preflight_prompt(self.session.session_dir)
            print(f"\n\033[36mCARL:\033[0m {preflight_prompt}")
            preflight_reply = self._prompt_user()
            if preflight_reply is None:
                return "quit"
            self._revisit_guidance = normalize_revisit_guidance(preflight_reply)
            self._revisit_exclusions = extract_revisit_exclusions(preflight_reply)

        # Load ONLY this step's skill file
        skill_content = load_skill(self.carl_dir, step["skill"])
        previous = (
            build_revisit_context(
                self.session.session_dir,
                guidance=self._revisit_guidance,
                excluded_paths=self._revisit_exclusions,
            )
            if is_revisit_step
            else self.session.get_previous_decisions()
        )
        skip = {2.5} if self._suggest_used else None
        if is_revisit_step:
            mode_note = (
                f"Revisit mode is inspecting existing session: {self.session.session_dir}\n"
                "The controller already collected optional pre-analysis "
                "guidance; honor it before inspecting artifacts."
            )
        elif (
            self.resume_mode
            and self.resume_info
            and step_num == self.resume_info.step_number
        ):
            mode_note = (
                f"Resume mode is restarting existing session: {self.session.session_dir}\n"
                f"Restart from the top of Step {step_num}. Prior completed "
                "decisions were restored from memory.md. Partial artifacts "
                "from the unfinished step may already exist in code/, "
                "results/, or references/; inspect and reuse or repair them "
                "as appropriate instead of assuming a blank workspace."
            )
        else:
            mode_note = None
        workflow = [REVISIT_STEP] if is_revisit_step else None
        # Start the clock before the first turn, and tell the model what it is.
        deadline = self._start_step_deadline(step_num)
        system = build_system_prompt(
            step, skill_content, previous, skip_steps=skip,
            workflow_steps=workflow, mode_note=mode_note,
            budget_minutes=(deadline.budget_seconds // 60) if deadline else None,
        )

        # Track state at instance level for crash recovery
        messages: list[dict] = [{"role": "user", "content": "[Begin step]"}]
        if preflight_prompt is not None:
            messages.append({"role": "assistant", "content": preflight_prompt})
            messages.append({
                "role": "user",
                "content": preflight_reply.strip() if preflight_reply and preflight_reply.strip() else "none",
            })
        msg_origins: dict[int, dict] = {}
        self._current_step = step
        self._current_messages = messages
        self._current_msg_origins = msg_origins
        self._current_system = system

        # CARL speaks first
        opening, _ = self._stream_response(
            system,
            messages,
            tools,
            require_user_question=step_num not in _TOOL_STEPS,
        )
        awaiting_confirmation = self._asks_to_proceed(opening)

        # Ask Trillian for a suggestion after CARL's opening
        self._trillian_suggestion = self._get_trillian_suggestion(
            step_num, step["name"], messages
        )
        # Trillian suggestion will be pre-filled in the input buffer by _prompt_user

        while True:
            user_input = self._prompt_user()

            if user_input is None:
                return "quit"

            # Trillian's suggestion was pre-filled in the input buffer.
            # Whatever the user sent is the final text — no merge needed.
            origin: dict | None = None
            if self._trillian_suggestion:
                if user_input == self._trillian_suggestion:
                    origin = {"type": "trillian_accepted", "trillian": self._trillian_suggestion}
                    print(f"  \033[33m[Using Trillian's suggestion]\033[0m")
                else:
                    origin = {
                        "type": "trillian_edited",
                        "trillian": self._trillian_suggestion,
                        "final": user_input,
                    }
            self._trillian_suggestion = None

            cmd = user_input.strip().lower()
            if step_num == 2 and "/skip" in cmd:
                self._user_skip_25 = True
            if (
                self.revisit_mode
                and step_num == REVISIT_STEP["number"]
                and self._is_finish_command(user_input)
            ):
                if origin:
                    msg_origins[len(messages)] = origin
                if user_input:
                    messages.append({"role": "user", "content": user_input})
                summary = self._summarize(system, messages, step)
                self._save_completed_step(
                    step_num, step["name"], summary, messages[1:],
                    {k - 1: v for k, v in msg_origins.items()},
                )
                return "next"
            if cmd in ("/next", "/skip"):
                summary = self._summarize(system, messages, step)
                self._save_completed_step(
                    step_num, step["name"], summary, messages[1:],
                    {k - 1: v for k, v in msg_origins.items()},
                )
                return "next"
            if cmd == "/status":
                if self.revisit_mode and step_num == REVISIT_STEP["number"]:
                    print(f"\n  Step {step_num}: {step['name']} (revisit mode)")
                else:
                    print(f"\n  Step {step_num}/9: {step['name']}")
                continue

            # /suggest keyword — Step 1 only
            if step_num == 1 and "/suggest" in user_input.lower():
                suggest_result = self._handle_suggest(user_input)
                if suggest_result is None:
                    continue  # user aborted or search failed; re-prompt
                user_input = suggest_result

            # Auto-read file paths from user input in early steps
            if step_num in (1, 2):
                user_input = self._inject_file_content(user_input)

            # If CARL asked to proceed and user gives a short affirmative
            if awaiting_confirmation and self._is_affirmative(user_input):
                if origin:
                    msg_origins[len(messages)] = origin
                messages.append({"role": "user", "content": user_input})
                summary = self._summarize(system, messages, step)
                self._save_completed_step(
                    step_num, step["name"], summary, messages[1:],
                    {k - 1: v for k, v in msg_origins.items()},
                )
                return "next"

            if origin:
                msg_origins[len(messages)] = origin
            messages.append({"role": "user", "content": user_input})

            try:
                response_text, used_tools = self._stream_response(
                    system,
                    messages,
                    tools,
                    require_user_question=(
                        step_num not in _TOOL_STEPS
                        and step_num not in _ONE_TURN_STEPS
                    ),
                )
            except Exception as e:
                print(f"\n  \033[31m[API error: {e}]\033[0m")
                print("  Saving progress... You can continue or /quit.")
                self._auto_save_conversation()
                continue

            # ── One-turn step: auto-advance after first user turn ──
            if step_num in _ONE_TURN_STEPS:
                if step_num == 7:
                    # Read the budget markers off the raw conversation, before
                    # the summarizer gets a chance to drop them.
                    self._capture_step7_budgets(
                        "\n".join(
                            self._extract_text(m)
                            for m in messages
                            if m.get("role") == "assistant"
                        )
                    )
                summary = self._summarize(system, messages, step)
                self._save_completed_step(
                    step_num, step["name"], summary, messages[1:],
                    {k - 1: v for k, v in msg_origins.items()},
                )
                return "next"

            # ── Autonomous execution for code-gen steps only ─────
            # Once tools start firing in Steps 8-9, keep going without
            # user input. Literature search (2.5) returns to the user.
            if step_num in _TOOL_STEPS and used_tools:
                self._auto_save_conversation()  # save before long loop
                for _ in range(_MAX_AUTO_ROUNDS):
                    if self._asks_to_proceed(response_text):
                        break
                    if "[DONE]" in response_text:
                        break
                    # Out of budget and out of grace: stop driving the step.
                    # The summarize call still runs, so work is recorded.
                    if self._enforce_budget_hard_stop():
                        break
                    # Check for user interjection via stdin
                    interjection = ""
                    if select.select([sys.stdin], [], [], 0)[0]:
                        interjection = sys.stdin.readline().strip()
                    if interjection:
                        messages.append(
                            {"role": "user", "content": interjection}
                        )
                        print(f"\n  \033[33m[Interjection sent]\033[0m")
                    else:
                        # Over budget: ask for the landing instead of more work.
                        # Only consulted when nothing else is being sent -- the
                        # call marks the directive delivered, so it must not be
                        # discarded here.
                        wrap_up = self._budget_wrap_up_text()
                        messages.append(
                            {"role": "user", "content": wrap_up or "[continue]"}
                        )
                    try:
                        response_text, used_tools = self._stream_response(
                            system,
                            messages,
                            tools,
                            require_user_question=False,
                        )
                    except Exception as e:
                        print(f"\n  \033[31m[API error: {e}]\033[0m")
                        print("  Saving progress... You can continue or /quit.")
                        self._auto_save_conversation()
                        break
                    self._auto_save_conversation()  # save after each round
                    if not used_tools:
                        break  # back to conversation mode

                # If the step went over budget it was told its jobs would be
                # stopped; make that true even when it landed its summary
                # before the hard cut.
                if self.deadline is not None and self.deadline.soft_expired():
                    self._stop_background_jobs_for_budget(
                        "over its wall-clock budget"
                    )

            # Detect model outputting tool calls as text in a non-tool step.
            # This means the model thinks the step is done and wants to code,
            # but the framework hasn't transitioned yet.
            if not tools and self._contains_tool_call_text(response_text):
                next_step = STEPS[self.current_step_idx + 1] if self.current_step_idx + 1 < len(STEPS) else None
                if next_step and next_step["number"] in _TOOL_STEPS:
                    print(
                        f"\n  \033[33m[CARL is trying to write code, but we're still on Step {step_num}. "
                        f"Proceed to Step {next_step['number']}: {next_step['name']}? (yes/no)]\033[0m"
                    )
                    confirm = self._prompt_user()
                    if confirm is not None and self._is_affirmative(confirm):
                        summary = self._summarize(system, messages, step)
                        self._save_completed_step(
                            step_num, step["name"], summary, messages[1:],
                            {k - 1: v for k, v in msg_origins.items()},
                        )
                        return "next"
                    # User said no — continue the conversation in this step
                    awaiting_confirmation = False
                    continue

            awaiting_confirmation = self._asks_to_proceed(response_text)

            # Ask Trillian for a suggestion after CARL responds
            self._trillian_suggestion = self._get_trillian_suggestion(
                step_num, step["name"], messages
            )
            # Trillian suggestion will be pre-filled in the input buffer by _prompt_user

    # ── LLM interaction with tool-use loop ────────────────────────────

    def _stream_response(
        self,
        system: str,
        messages: list[dict],
        tools: list | None = None,
        require_user_question: bool = False,
    ) -> tuple[str, bool]:
        """Stream a response, executing tool calls in a loop.

        Appends all assistant/tool messages to *messages* in place.
        Returns (accumulated_text, whether_tools_were_used).

        Auto-continue: if the model stops with short narration text after
        a tool-use round (e.g. "Now let me write...") instead of calling
        a tool, we automatically inject [continue] so it keeps going
        without requiring user input.
        """
        max_tokens = 16384 if tools else 1024
        step_num = self._current_step["number"] if self._current_step else None
        # Prose-only turns are capped at the medium tier; tool turns keep the
        # step's own tier. See _TEXT_TURN_TIER_CEILING.
        ceiling = None if tools else _TEXT_TURN_TIER_CEILING
        thinking_cfg = thinking_config_for_step(self.model, step_num, ceiling)
        if thinking_cfg:
            max_tokens = thinking_max_tokens_for_step(step_num, max_tokens, ceiling)
        all_text = ""
        had_tool_use = False
        missing_question_continues = 0
        rounds = 0
        tail_continues = 0
        cap_logged = False
        blank_turns = 0

        print("\n\033[36mCARL:\033[0m ", end="", flush=True)

        # self.deadline is None outside Steps 8/9; those steps carry an
        # enforced wall-clock budget checked on every round.
        while (
            not self._budget_hard_expired()
            and (
                rounds < _MAX_TOOL_ROUNDS
                or (
                    self._has_pending_auto_turn(messages)
                    and tail_continues < _MAX_TOOL_ROUND_TAIL_CONTINUES
                )
            )
        ):
            if rounds >= _MAX_TOOL_ROUNDS:
                tail_continues += 1
                if not cap_logged:
                    print(
                        "\n  \033[36m[tool-round cap reached; finishing pending tool/continue turn...]\033[0m",
                        flush=True,
                    )
                    cap_logged = True
            rounds += 1
            self._sanitize_messages(messages)
            content_blocks, stop_reason, usage = self._stream_single_turn(
                system, messages, tools, max_tokens, thinking_config=thinking_cfg
            )
            if usage:
                print(
                    f"\n  \033[90m[tokens: in={usage.get('input_tokens', 0)} "
                    f"(cached={usage.get('cache_creation_input_tokens', 0)}, "
                    f"read={usage.get('cache_read_input_tokens', 0)}) "
                    f"out={usage.get('output_tokens', 0)}]\033[0m",
                    flush=True,
                )

            turn_text = "".join(
                b["text"] for b in content_blocks if b["type"] == "text"
            )
            tool_uses = [b for b in content_blocks if b["type"] == "tool_use"]
            all_text += turn_text

            if not tool_uses:
                # No visible output — usually the whole token allowance went to
                # adaptive reasoning. A blank assistant turn would make the next
                # Converse call fail validation, so retry with more headroom.
                if not turn_text.strip():
                    blank_turns += 1
                    if blank_turns > _MAX_BLANK_TURNS:
                        print(
                            "\n  \033[36m[model returned no text; stopping turn]\033[0m",
                            flush=True,
                        )
                        break
                    max_tokens = min(max_tokens * 2, _MAX_OUTPUT_TOKENS)
                    print(
                        f"\n  \033[36m[empty response; retrying with "
                        f"max_tokens={max_tokens}...]\033[0m",
                        flush=True,
                    )
                    print("\n\033[36mCARL:\033[0m ", end="", flush=True)
                    continue

                # No tool calls this turn
                messages.append({"role": "assistant", "content": turn_text})

                # Hit max_tokens — model was cut off, always continue
                if stop_reason == "max_tokens":
                    messages.append(
                        {"role": "user", "content": "[continue — you were cut off]"}
                    )
                    print("\n\033[36mCARL:\033[0m ", end="", flush=True)
                    continue

                # Auto-continue: if we just finished tool calls and the model
                # paused with short narration, nudge it to keep going.
                if had_tool_use and self._should_auto_continue(turn_text):
                    messages.append(
                        {"role": "user", "content": "[continue]"}
                    )
                    had_tool_use = False
                    print("\n\033[36mCARL:\033[0m ", end="", flush=True)
                    continue

                if (
                    require_user_question
                    and missing_question_continues < 2
                    and should_continue_for_missing_question(turn_text)
                ):
                    messages.append({
                        "role": "user",
                        "content": MISSING_QUESTION_CONTINUE_PROMPT,
                    })
                    missing_question_continues += 1
                    print("\n\033[36mCARL:\033[0m ", end="", flush=True)
                    continue
                break

            had_tool_use = True

            # Tool-use turn — append full content blocks
            messages.append({"role": "assistant", "content": content_blocks})

            # Execute each tool
            tool_results = []
            for tu in tool_uses:
                try:
                    result = self._execute_tool(tu["name"], tu["input"])
                except Exception as e:
                    result = f"Error executing {tu['name']}: {e}"
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu["id"],
                    "content": result,
                })
            # Over budget mid-tool-chain: ride along on the tool-result turn
            # rather than appending a second consecutive user message, which
            # Converse rejects.
            wrap_up = self._budget_wrap_up_text()
            if wrap_up:
                tool_results.append({"type": "text", "text": wrap_up})

            messages.append({"role": "user", "content": tool_results})
            self._auto_save_conversation()

            # Model will continue — print label for next chunk
            print("\n\033[36mCARL:\033[0m ", end="", flush=True)

        # Enforce the hard cut before reporting the cap, so a budget stop is
        # not misattributed to the tool-round cap.
        hard_stopped = self._enforce_budget_hard_stop()

        if self._has_pending_auto_turn(messages) and not hard_stopped:
            print(
                "\n  \033[33m[tool-round cap reached with pending continuation; returning to user]\033[0m",
                flush=True,
            )

        print()
        return all_text, had_tool_use

    @staticmethod
    def _sanitize_messages(messages: list[dict]):
        """Repair message list so every tool_use/tool_result pair is consistent.

        If a pair is broken, truncate from that point onward.
        """
        i = 0
        while i < len(messages):
            msg = messages[i]
            if msg.get("role") != "assistant":
                i += 1
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                i += 1
                continue
            tool_use_ids = {
                b["id"] for b in content
                if isinstance(b, dict) and b.get("type") == "tool_use"
            }
            if not tool_use_ids:
                i += 1
                continue
            next_msg = messages[i + 1] if i + 1 < len(messages) else None
            ok = False
            if next_msg and next_msg.get("role") == "user":
                next_content = next_msg.get("content")
                if isinstance(next_content, list):
                    result_ids = {
                        b["tool_use_id"] for b in next_content
                        if isinstance(b, dict) and b.get("type") == "tool_result"
                    }
                    ok = tool_use_ids == result_ids
            if not ok:
                del messages[i:]
                break
            i += 1

    @staticmethod
    def _should_auto_continue(text: str) -> bool:
        """True if a text-only response after tool use looks like the model
        wants to keep working but stopped short of calling a tool."""
        stripped = text.strip()
        if not stripped:
            return True
        if asks_user_question(stripped):
            return False
        lower = stripped.lower()
        tail = lower[-300:]
        continuations = [
            "let me", "now i'll", "next,", "next i'll", "now let",
            "i'll write", "i'll create", "i'll run", "i'll update",
            "let's run", "let's write", "let's create",
            "writing", "running", "creating", "now running",
            "now writing", "now creating", "i need to",
            "first, let me", "first, i'll",
            "i'll fix", "i'll check", "i'll implement", "i'll add",
            "i'll test", "let me check", "let me test", "let me fix",
            "let me implement", "let me add", "let me update",
        ]
        if any(c in tail for c in continuations):
            return True
        if len(lower) <= 300:
            return True
        return False

    @staticmethod
    def _has_pending_auto_turn(messages: list[dict]) -> bool:
        """True when the last user message was generated by the controller."""
        if not messages:
            return False
        msg = messages[-1]
        if msg.get("role") != "user":
            return False
        content = msg.get("content")
        if isinstance(content, list):
            return any(
                isinstance(block, dict) and block.get("type") == "tool_result"
                for block in content
            )
        if isinstance(content, str):
            stripped = content.strip()
            return (
                stripped.startswith("[continue")
                or stripped == MISSING_QUESTION_CONTINUE_PROMPT
            )
        return False

    # ── Trillian suggestion ─────────────────────────────────────────────

    def _get_trillian_suggestion(self, step_num: int, step_name: str,
                               messages: list[dict]) -> str | None:
        """Ask Trillian to suggest a user response. Returns None if skipped."""
        if step_num > _TRILLIAN_LAST_ACTIVE_STEP:
            return None

        # Build text-only view of conversation for Trillian
        trillian_msgs = []
        for msg in messages:
            text = self._extract_text(msg)
            if not text or text == "[Begin step]":
                continue
            trillian_msgs.append({"role": msg["role"], "content": text})

        if not trillian_msgs:
            return None

        # Ensure alternation: append a nudge if last message is from user
        if trillian_msgs[-1]["role"] == "user":
            return None  # Trillian only responds after CARL speaks

        # Add a final user message to prompt Trillian's reply
        trillian_msgs.append({
            "role": "user",
            "content": "Generate your suggested response to CARL's latest message.",
        })

        system = build_trillian_prompt(
            self._trillian_persona, step_num, step_name,
        )
        try:
            text = converse_text(
                self.client, self.trillian_model, system, trillian_msgs,
                max_tokens=256, temperature=0.5,
            ).strip()
        except Exception as e:
            print(f"\n  \033[33m[Trillian unavailable: {e}]\033[0m")
            return None

        if text == "[SKIP]" or not text:
            return None
        return text

    # ── /suggest feature ─────────────────────────────────────────────

    def _call_llm_non_streaming(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> str | None:
        """Non-streaming Converse call against the primary CARL model."""
        try:
            return converse_text(
                self.client, self.model, system, messages,
                max_tokens=max_tokens, temperature=temperature,
            ).strip()
        except Exception as e:
            print(f"\n  \033[31m[LLM call failed: {e}]\033[0m")
            return None

    def _call_model_non_streaming(
        self,
        model_id: str,
        system: str,
        messages: list[dict],
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> str | None:
        """Non-streaming Converse call with explicit model ID."""
        try:
            return converse_text(
                self.client, model_id, system, messages,
                max_tokens=max_tokens, temperature=temperature,
            ).strip()
        except Exception as e:
            print(f"\n  \033[31m[LLM call failed: {e}]\033[0m")
            return None

    @staticmethod
    def _extract_formulation(suggestions_text: str, index: int) -> str:
        """Extract the Nth formulation block from the suggestions text.

        Formulations are delimited by **[N]. lines.  Returns just the chosen
        block, or the full text as fallback.
        """
        parts = re.split(r"(?=\*\*\[\d+\])", suggestions_text.strip())
        parts = [p.strip() for p in parts if p.strip()]
        for part in parts:
            m = re.match(r"\*\*\[(\d+)\]", part)
            if m and int(m.group(1)) == index:
                return part
        return f"Formulation #{index} from the list below:\n{suggestions_text}"

    def _build_suggest_context(self, latest_request: str) -> str:
        """Build search context from the full current-step conversation."""
        lines: list[str] = []
        for msg in self._current_messages or []:
            text = self._extract_text(msg).strip()
            if not text or text == "[Begin step]":
                continue
            speaker = "CARL" if msg.get("role") == "assistant" else "User"
            lines.append(f"{speaker}: {text}")

        latest = latest_request.strip() or "(No additional text after /suggest.)"
        if not lines:
            return latest
        return (
            "Conversation so far:\n"
            + "\n\n".join(lines)
            + "\n\nLatest /suggest request:\n"
            + latest
        )

    @staticmethod
    def _parse_index_selection(choice: str, max_index: int) -> list[int] | None:
        """Parse comma-separated 1-based indices. Empty -> all indices."""
        if not choice:
            return list(range(1, max_index + 1))

        selected: list[int] = []
        seen: set[int] = set()
        for token in choice.split(","):
            token = token.strip()
            if not token:
                continue
            if not token.isdigit():
                return None
            idx = int(token)
            if idx < 1 or idx > max_index:
                return None
            if idx not in seen:
                seen.add(idx)
                selected.append(idx)

        return selected or list(range(1, max_index + 1))

    @staticmethod
    def _format_literature_shortlist_section(papers: list[dict]) -> str:
        """Build canonical references to append at end of /suggest summary."""
        lines = [
            "## References (from literature_search.md)",
            "",
        ]
        for i, p in enumerate(papers, 1):
            authors = ", ".join((p.get("authors") or [])[:3]) or "Unknown"
            year = p.get("year") or "?"
            title = p.get("title", "Untitled")
            lines.append(f"{i}. {authors} ({year}). *{title}*.")
        return "\n".join(lines)

    def _handle_suggest(self, user_input: str) -> str | None:
        """Handle /suggest keyword: search literature, synthesize RL problems,
        let user pick one, generate a full problem spec, save to refs/."""
        from literature import search_papers, save_literature_results

        # 1. Extract search query — use LLM to distil short keyword queries
        clean_input = re.sub(r"/suggest", "", user_input, flags=re.IGNORECASE).strip().strip(",").strip()
        if not clean_input:
            clean_input = "reinforcement learning problem formulation"

        print(f"\n  \033[36m[/suggest] Searching literature and generating candidates. This may take 1-2 minutes...\033[0m")
        print(f"  \033[36m[/suggest] Generating search queries...\033[0m")
        suggest_context = self._build_suggest_context(clean_input)
        queries_text = self._call_llm_non_streaming(
            system=(
                "You are given a full conversation context plus the user's latest "
                "/suggest request. Infer the RL problem being discussed and output "
                "2-3 short academic search queries (3-6 keywords each) suitable for "
                "academic search (arXiv/OpenAlex). Output one query per line. No numbering, no explanation."
            ),
            messages=[{"role": "user", "content": suggest_context}],
            max_tokens=150,
            temperature=0.3,
        )
        if queries_text:
            queries = [q.strip().strip("-•").strip() for q in queries_text.splitlines() if q.strip()]
        else:
            queries = [clean_input]

        # 2. Search the literature backends with each query, merge results
        all_papers: list[dict] = []
        seen_ids: set[str] = set()
        for q in queries[:3]:
            print(f"  \033[36m[/suggest] Searching: \"{q}\"...\033[0m")
            try:
                results = search_papers(q, limit=8)
            except Exception as e:
                print(f"  \033[31m[Search failed for \"{q}\": {e}]\033[0m")
                continue
            for p in results:
                pid = p.get("paper_id") or p.get("title", "")
                if pid not in seen_ids:
                    seen_ids.add(pid)
                    all_papers.append(p)
        papers = all_papers

        if not papers:
            print("  \033[33m[No papers found. Try a more specific description.]\033[0m")
            return None
        print(f"  \033[32m[Found {len(papers)} papers across {len(queries)} queries]\033[0m")
        shortlist_papers = papers[:10]
        save_literature_results(
            self.session.session_dir, shortlist_papers,
            bedrock_client=self.client, summary_model=self.trillian_model,
            overwrite=True,
        )

        # 3. Ask user which papers from the fixed shortlist to use for synthesis
        print(f"\n\033[1m[Paper Shortlist — literature_search.md]\033[0m\n")
        for i, p in enumerate(shortlist_papers, 1):
            print(f"[{i}] {p['title']} ({p.get('year', '?')})")

        selected_paper_indices: list[int] | None = None
        while selected_paper_indices is None:
            print("\nSelect paper numbers for synthesis (e.g., 1,3,5), Enter for all, or 'c' to cancel: ", end="", flush=True)
            try:
                selection = input().strip().lower()
            except (EOFError, KeyboardInterrupt):
                return None
            if selection in ("c", "cancel"):
                return None
            selected_paper_indices = self._parse_index_selection(selection, len(shortlist_papers))
            if selected_paper_indices is None:
                print("  [Invalid selection. Use comma-separated numbers within the shortlist.]")

        selected_papers = [shortlist_papers[i - 1] for i in selected_paper_indices]

        # 4. Format selected shortlist papers for LLM
        papers_formatted = "\n".join(
            f"[{i+1}]. {p['title']} ({p.get('year', '?')})\n   {(p.get('abstract') or '')[:300]}"
            for i, p in enumerate(selected_papers)
        )

        # 5. Phase 1: synthesize 3-5 candidate RL formulations
        print(f"  \033[36m[/suggest] Synthesizing RL problem candidates...\033[0m")
        suggestions_text = self._call_llm_non_streaming(
            system=_SUGGEST_SYNTHESIS_SYSTEM,
            messages=[{"role": "user", "content":
                _SUGGEST_SYNTHESIS_USER.format(
                    problem_description=clean_input,
                    papers_formatted=papers_formatted,
                )}],
            max_tokens=1200,
            temperature=0.7,
        )
        if not suggestions_text:
            print("  \033[31m[Suggestion generation failed. Continuing without /suggest.]\033[0m")
            return None

        # 5. Present + pick (up to 3 refine/re-search loops)
        chosen_index = None
        for attempt in range(3):
            print(f"\n\033[1m[Suggested RL Problems]\033[0m\n")
            print(suggestions_text)
            print("\nEnter a number to choose, 'r' to refine, 's' to re-search, or 'c' to cancel: ", end="", flush=True)
            try:
                choice = input().strip().lower()
            except (EOFError, KeyboardInterrupt):
                return None
            if choice in ("c", "cancel", ""):
                return None
            if choice in ("s", "search"):
                if attempt == 2:
                    print("  [Max attempts reached. Continuing without /suggest.]")
                    return None
                print("New search description: ", end="", flush=True)
                try:
                    new_desc = input().strip()
                except (EOFError, KeyboardInterrupt):
                    return None
                if new_desc:
                    # Re-run the full search with new terms
                    new_result = self._handle_suggest(new_desc + " /suggest")
                    return new_result
                continue
            if choice in ("r", "refine"):
                if attempt == 2:
                    print("  [Max attempts reached. Continuing without /suggest.]")
                    return None
                print("Refinement note: ", end="", flush=True)
                try:
                    note = input().strip()
                except (EOFError, KeyboardInterrupt):
                    return None
                print(f"  \033[36m[/suggest] Re-synthesizing with refinement...\033[0m")
                suggestions_text = self._call_llm_non_streaming(
                    system=_SUGGEST_SYNTHESIS_SYSTEM,
                    messages=[{"role": "user", "content":
                        _SUGGEST_SYNTHESIS_USER.format(
                            problem_description=clean_input + (f"\n\nRefinement: {note}" if note else ""),
                            papers_formatted=papers_formatted,
                        )}],
                    max_tokens=1200,
                    temperature=0.7,
                )
                if not suggestions_text:
                    print("  \033[31m[Refinement failed. Continuing without /suggest.]\033[0m")
                    return None
                continue
            if choice.isdigit() and 1 <= int(choice) <= 9:
                chosen_index = int(choice)
                break
            print("  [Invalid choice. Enter a number, 'r', 's', or 'c'.]")

        if chosen_index is None:
            return None

        # 6. Phase 2: generate full problem specification document
        print(f"\n  \033[36m[/suggest] Generating detailed problem specification...\033[0m")
        doc_text = self._call_llm_non_streaming(
            system=_SUGGEST_DOCUMENT_SYSTEM,
            messages=[{"role": "user", "content":
                _SUGGEST_DOCUMENT_USER.format(
                    chosen_formulation=self._extract_formulation(suggestions_text, chosen_index),
                    problem_description=clean_input,
                    papers_formatted=papers_formatted,
                )}],
            max_tokens=8192,
            temperature=0.5,
        )
        if not doc_text:
            print("  \033[31m[Document generation failed. Continuing without /suggest.]\033[0m")
            return None

        # Keep the problem summary anchored to the same 10-paper shortlist file.
        doc_text = (
            doc_text.rstrip()
            + "\n\n"
            + self._format_literature_shortlist_section(shortlist_papers)
            + "\n"
        )

        # 7. Slugify the title (first non-empty heading line)
        title_line = next(
            (line.lstrip("#").strip() for line in doc_text.splitlines() if line.strip()),
            "rl_problem",
        )
        slug = re.sub(r"[^a-z0-9_]", "_", title_line.lower())
        slug = re.sub(r"_+", "_", slug).strip("_")[:40]
        if not slug:
            slug = "suggested_rl_problem"

        # 8. Save to refs/<slug>.md with YAML front-matter
        refs_dir = Path(self.carl_dir) / "refs"
        refs_dir.mkdir(exist_ok=True)
        refs_path = refs_dir / f"{slug}.md"
        front_matter = (
            f"---\n"
            f"source: carl-suggest\n"
            f"generated: {date.today().isoformat()}\n"
            f"query: {'; '.join(queries)}\n"
            f"---\n\n"
        )
        try:
            refs_path.write_text(front_matter + doc_text, encoding="utf-8")
        except Exception as e:
            print(f"  \033[31m[Failed to save file: {e}]\033[0m")
            return None

        # 9. Copy to session references
        self.session.copy_reference(str(refs_path))
        print(f"  \033[32m[/suggest] Saved: refs/{slug}.md — injecting as context\033[0m")

        # Mark /suggest as used so Step 2.5 can be skipped
        self._suggest_used = True

        # Return modified input so _inject_file_content picks up the new file
        return clean_input + f"\n\nrefs/{slug}.md"

    @staticmethod
    def _extract_text(msg: dict) -> str:
        """Extract plain text from a message (handles tool-use content blocks)."""
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_result":
                        parts.append(block.get("content", ""))
            return " ".join(parts)
        return ""

    def _stream_single_turn(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict] | None,
        max_tokens: int,
        thinking_config: dict | None = None,
    ) -> tuple[list[dict], str, dict]:
        """Stream one Converse call. Returns (content_blocks, stop_reason, usage).

        System prompt and tool schemas are cached via Converse cachePoint
        markers with a 1h TTL, so every turn after the first in a step
        reuses the cached prefix.
        """
        content_blocks: list[dict] = []
        stop_reason = "end_turn"
        usage: dict = {}

        try:
            for kind, payload in converse_stream(
                self.client, self.model, system, messages,
                tools=tools, max_tokens=max_tokens, temperature=0.7,
                cache_system=True, cache_tools=bool(tools),
                thinking_config=thinking_config,
            ):
                if kind == "text":
                    print(payload, end="", flush=True)
                elif kind == "tool_start":
                    print(
                        f"\n  \033[33m[calling {payload}...]\033[0m",
                        end="", flush=True,
                    )
                elif kind == "usage":
                    usage = payload
                elif kind == "result":
                    content_blocks, stop_reason = payload
        except ClientError as exc:
            if thinking_config and self._is_thinking_error(exc):
                sys.stderr.write(
                    f"[thinking fallback] {exc.response.get('Error', {}).get('Message', str(exc))[:200]}\n"
                )
                base_max_tokens = 16384 if tools else 1024
                for kind, payload in converse_stream(
                    self.client, self.model, system, messages,
                    tools=tools, max_tokens=base_max_tokens, temperature=0.7,
                    cache_system=True, cache_tools=bool(tools),
                ):
                    if kind == "text":
                        print(payload, end="", flush=True)
                    elif kind == "tool_start":
                        print(
                            f"\n  \033[33m[calling {payload}...]\033[0m",
                            end="", flush=True,
                        )
                    elif kind == "usage":
                        usage = payload
                    elif kind == "result":
                        content_blocks, stop_reason = payload
            else:
                raise

        return content_blocks, stop_reason, usage

    @staticmethod
    def _is_thinking_error(exc: ClientError) -> bool:
        err = exc.response.get("Error", {})
        msg = (err.get("Message") or str(exc)).lower()
        return any(kw in msg for kw in (
            "thinking", "budget_tokens", "additionalmodelrequestfields",
            "output_config", "effort",
        ))

    # ── Tool execution ────────────────────────────────────────────────

    def _execute_tool(self, name: str, input_data: dict) -> str:
        """Dispatch a tool call and return the result string."""
        if name == "write_file":
            return self._tool_write_file(
                input_data["path"], input_data["content"]
            )
        if name == "run_command":
            return self._tool_run_command(input_data["command"])
        if name == "run_background":
            return self._tool_run_background(
                input_data["command"], input_data["log_file"]
            )
        if name == "check_process":
            return self._tool_check_process(
                input_data["log_file"],
                input_data.get("tail_lines", 50),
            )
        if name == "list_revisit_artifacts":
            return json.dumps(
                discover_artifacts(
                    self.session.session_dir,
                    excluded_paths=self._revisit_exclusions,
                ),
                indent=2,
            )
        if name == "read_artifact":
            return read_artifact(
                self.session.session_dir,
                input_data["path"],
                input_data.get("max_chars", 12000),
                excluded_paths=self._revisit_exclusions,
            )
        if name == "describe_plot":
            return describe_plot(
                self.client, self.model, self.session.session_dir,
                input_data["path"],
                excluded_paths=self._revisit_exclusions,
            )
        if name == "prepare_revisit_workspace":
            return prepare_revisit_workspace_tool_response(
                self.session.session_dir, input_data
            )
        if name == "search_literature":
            return self._tool_search_literature(
                input_data["query"],
                input_data.get("limit", 10),
            )
        return f"Unknown tool: {name}"

    def _tool_search_literature(self, query: str, limit: int) -> str:
        """Execute a literature search across the configured backends."""
        from literature import (
            search_papers_with_status, format_results_for_tool,
            save_literature_results,
        )
        # Per-step search budget, keyed by step so it resets on step change.
        step_key = self._current_step["number"] if self._current_step else None
        done = self._literature_search_counts.get(step_key, 0)
        if done >= MAX_LITERATURE_SEARCHES:
            print(
                f"\n  \033[31m[Search limit reached: {done}/"
                f"{MAX_LITERATURE_SEARCHES} this step — refusing further "
                f"searches]\033[0m",
                flush=True,
            )
            return LITERATURE_LIMIT_MESSAGE
        self._literature_search_counts[step_key] = done + 1
        limit = min(limit, 20)
        print(
            f"\n  \033[33m[Searching ({done + 1}/{MAX_LITERATURE_SEARCHES}): "
            f"\"{query}\" (limit={limit})]\033[0m",
            flush=True,
        )
        papers, errors = search_papers_with_status(query, limit)
        if papers:
            save_literature_results(
                self.session.session_dir, papers,
                bedrock_client=self.client, summary_model=self.trillian_model,
            )
            print(f"  \033[32m[Found {len(papers)} papers]\033[0m", flush=True)
        else:
            print(f"  \033[31m[No results]\033[0m", flush=True)
        if errors:
            print(f"  \033[31m[Backend failures: {'; '.join(errors)}]\033[0m", flush=True)
        return format_results_for_tool(papers, errors)

    def _tool_write_file(self, rel_path: str, content: str) -> str:
        """Write a file into the session's code/ directory."""
        rel_path = normalize_code_relpath(rel_path)
        if self.revisit_mode and not re.match(r"^revisit_\d+/", rel_path.replace("\\", "/")):
            return "Refusing Step 10 write outside code/revisit_N/."
        code_dir = self.session.session_dir / "code"
        full_path = code_dir / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")
        print(
            f"\n  \033[32m[wrote code/{rel_path} -- "
            f"{len(content)} chars]\033[0m",
            flush=True,
        )
        return f"File written: code/{rel_path} ({len(content)} chars)"

    def _tool_run_command(self, command: str) -> str:
        """Run a shell command in the session's code/ directory."""
        if self.revisit_mode and is_revisit_training_command(command):
            return revisit_training_refusal(command)
        code_dir = self.session.session_dir / "code"
        print(f"\n  \033[33m$ {command}\033[0m", flush=True)
        try:
            compat_output = run_compat_command(command, code_dir)
            if compat_output is not None:
                lines = compat_output.split("\n")
                preview = "\n".join(lines[:30])
                if len(lines) > 30:
                    preview += f"\n... ({len(lines) - 30} more lines)"
                print(f"  \033[33m{preview}\033[0m", flush=True)
                return compat_output[:15000]
            # Clamp the timeout to what's left of the step's budget, so a
            # 10-minute command cannot run past a 2-minute remainder.
            timeout = 600
            if self.deadline is not None:
                timeout = self.deadline.clamp_timeout(timeout)
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=str(code_dir),
                timeout=timeout,
            )
            output = (result.stdout + result.stderr).strip()
            # Show a preview to the user
            lines = output.split("\n")
            preview = "\n".join(lines[:30])
            if len(lines) > 30:
                preview += f"\n... ({len(lines) - 30} more lines)"
            print(f"  \033[33m{preview}\033[0m", flush=True)
            # Cap what goes back to the model
            return output[:15000]
        except subprocess.TimeoutExpired as e:
            limit = int(e.timeout or 600)
            msg = (
                f"Command timed out after {limit}s"
                + (
                    " (capped by this step's remaining wall-clock budget)"
                    if limit < 600
                    else ""
                )
                + ". Use run_background for long jobs."
            )
            print(f"  \033[31m{msg}\033[0m", flush=True)
            return msg
        except Exception as e:
            msg = f"Error: {e}"
            print(f"  \033[31m{msg}\033[0m", flush=True)
            return msg

    _is_revisit_training_command = staticmethod(is_revisit_training_command)

    def _tool_run_background(self, command: str, log_file: str) -> str:
        """Start a command in the background, streaming output to a log file."""
        code_dir = self.session.session_dir / "code"
        log_path = code_dir / log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)
        pid_file = code_dir / ".bg_pids.json"

        print(f"\n  \033[33m$ (background) {command}\033[0m", flush=True)
        print(f"  \033[33m  log: code/{log_file}\033[0m", flush=True)

        try:
            with open(log_path, "w") as lf:
                proc = subprocess.Popen(
                    command, shell=True, stdout=lf, stderr=subprocess.STDOUT,
                    cwd=str(code_dir),
                    start_new_session=True,
                )
            # Track PID — use suffixed keys to avoid overwriting
            pids = {}
            if pid_file.exists():
                pids = json.loads(pid_file.read_text())
            key = log_file
            n = 2
            while key in pids:
                key = f"{log_file}_{n}"
                n += 1
            pids[key] = proc.pid
            pid_file.write_text(json.dumps(pids))

            msg = f"Started (PID {proc.pid}). Output streaming to code/{log_file}. Use check_process to monitor."
            print(f"  \033[32m[{msg}]\033[0m", flush=True)
            return msg
        except Exception as e:
            msg = f"Error starting background process: {e}"
            print(f"  \033[31m{msg}\033[0m", flush=True)
            return msg

    # ── Step wall-clock budgets ───────────────────────────────────────
    #
    # Steps 8 and 9 run under an enforced deadline built from the Step 7
    # budgets. It is checked in four places, because a limit that misses any
    # one of them leaks: the tool-round loop, the auto-continue loop, the
    # check_process sleep, and the run_command timeout.

    def _budget_log(self, msg: str):
        print(f"  \033[36m{msg}\033[0m", flush=True)

    def _capture_step7_budgets(self, step7_text: str):
        """Persist the budget markers from Step 7's raw turn text.

        Read from the raw text, not the summary: the summarizer is asked for
        concise bullets and drops machine-readable lines, which silently
        reverted an observed session to the default budget.
        """
        if not step7_text:
            return
        for step_num, marker in (
            (8, TESTING_BUDGET_MARKER), (9, TRAINING_BUDGET_MARKER),
        ):
            m = re.search(
                re.escape(marker.rstrip(":")) + r"\s*:\s*(\d+(?:\.\d+)?)",
                step7_text,
                re.IGNORECASE,
            )
            if not m:
                continue
            minutes = int(float(m.group(1)))
            if minutes > 0:
                self.session.set_step_budget(step_num, minutes)
                self._budget_log(
                    f"[step {step_num} budget] {minutes} min recorded from Step 7"
                )

    def _start_step_deadline(self, step_num):
        """Register the wall-clock deadline for a step; None for other steps."""
        if step_num not in _TOOL_STEPS:
            self.deadline = None
            return None
        minutes = self.session.get_step_budget_minutes(step_num)
        if minutes:
            self._budget_log(
                f"[step {step_num:g} budget] {minutes} min from persisted marker"
            )
            budget_seconds, source = minutes * 60, "persisted"
        else:
            budget_seconds, source = resolve_step_budget(
                step_num,
                self.session.step_decisions.get(7, ""),
                log=self._budget_log,
            )
        self.deadline = StepDeadline(budget_seconds, source=source)
        self.wrap_up_sent = False
        self.hard_stop_done = False
        self._budget_log(
            f"[step {step_num:g}] wall-clock budget {budget_seconds // 60} min "
            f"({source})"
        )
        return self.deadline

    def _ensure_step_deadline(self, step_num):
        """Deadline for a step, starting the clock only if not already running."""
        if step_num in _TOOL_STEPS and getattr(self, "deadline", None) is not None:
            return self.deadline
        return self._start_step_deadline(step_num)

    def _budget_hard_expired(self) -> bool:
        deadline = getattr(self, "deadline", None)
        return deadline is not None and deadline.hard_expired()

    def _budget_wrap_up_text(self) -> str | None:
        """The wrap-up directive, once, the first time a step goes over budget."""
        deadline = getattr(self, "deadline", None)
        if deadline is None or not deadline.soft_expired():
            return None
        if getattr(self, "wrap_up_sent", False):
            return None
        self.wrap_up_sent = True
        step_num = self._current_step["number"] if self._current_step else 0
        self._budget_log(
            f"[budget exhausted for step {step_num:g}; asking for a final summary]"
        )
        return wrap_up_message(step_num, deadline)

    def _stop_background_jobs_for_budget(self, reason: str) -> bool:
        """Kill this step's background jobs once, because it ran out of budget."""
        if getattr(self, "hard_stop_done", False):
            return True
        self.hard_stop_done = True
        step_num = self._current_step["number"] if self._current_step else 0
        self._budget_log(f"[step {step_num:g} {reason}; stopping background jobs]")
        self._kill_background_processes()
        return True

    def _enforce_budget_hard_stop(self) -> bool:
        """Kill background jobs at the hard cut. True if the cut was enforced."""
        if not self._budget_hard_expired():
            return False
        return self._stop_background_jobs_for_budget("wall-clock limit reached")

    def _get_check_interval(self) -> int:
        """Seconds check_process waits, sized from the *current* step's budget.

        An earlier version always used the Step 9 budget, so Step 8 inherited a
        90-second poll interval and spent a third of one session asleep.
        """
        deadline = getattr(self, "deadline", None)
        if deadline is not None:
            return check_interval_seconds(deadline.budget_seconds)
        budget_seconds, _ = resolve_step_budget(
            9, self.session.step_decisions.get(7, ""), log=self._budget_log,
        )
        return check_interval_seconds(budget_seconds)

    def _tool_check_process(self, log_file: str, tail_lines: int = 50) -> str:
        """Check if a background process is running and return log tail."""
        code_dir = self.session.session_dir / "code"
        log_path = code_dir / log_file
        pid_file = code_dir / ".bg_pids.json"

        # Find PID
        pid = None
        if pid_file.exists():
            pids = json.loads(pid_file.read_text())
            pid = pids.get(log_file)

        # Check if alive
        status = "unknown"
        if pid is not None:
            try:
                os.kill(pid, 0)
                status = "running"
            except ProcessLookupError:
                status = "finished"
            except PermissionError:
                status = "running"  # exists but we can't signal it

        # If still running, wait before returning to avoid burning tool rounds.
        # The wait is clamped to the step's remaining budget: this sleep is the
        # single biggest way a step overruns, since each poll costs a minute or
        # more of wall clock.
        if status == "running":
            import time
            wait = self._get_check_interval()
            if self.deadline is not None:
                wait = self.deadline.clamp_sleep(wait)
            if wait > 0:
                print(
                    f"\n  \033[36m[waiting {wait:.0f}s before checking...]\033[0m",
                    flush=True,
                )
                time.sleep(wait)
            else:
                print(
                    "\n  \033[36m[budget spent; not sleeping before check]\033[0m",
                    flush=True,
                )
            try:
                os.kill(pid, 0)
                status = "running"
            except ProcessLookupError:
                status = "finished"
            except PermissionError:
                status = "running"

        # Read log tail
        log_tail = ""
        if log_path.exists():
            try:
                lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                tail = lines[-tail_lines:] if len(lines) > tail_lines else lines
                log_tail = "\n".join(tail)
            except Exception:
                log_tail = "(could not read log)"

        result = f"Status: {status}"
        if pid is not None:
            result += f" (PID {pid})"
        result += f"\nLog ({log_file}) last {tail_lines} lines:\n{log_tail}"
        if self.deadline is not None:
            result += f"\n{self.deadline.status_line()}"

        # Print preview
        preview_lines = log_tail.split("\n")[-10:]
        print(f"\n  \033[33m[{status}] {log_file} (last lines):\033[0m", flush=True)
        for line in preview_lines:
            print(f"  \033[33m  {line}\033[0m", flush=True)

        return result[:15000]

    # ── Summarization (no tools) ──────────────────────────────────────

    def _summarize(
        self, system: str, messages: list[dict], step: dict
    ) -> str:
        """Generate a bullet-point summary of this step's decisions."""
        summary_msgs = append_user_request(
            messages,
            "Provide a concise bullet-point summary of the key "
            "decisions and information gathered in this step. "
            "Just the bullets, no preamble.\n\n"
            "If the user gave any directive that constrains later "
            "steps (e.g. 'skip X', 'mandatory: Y'), include it "
            "verbatim as a dedicated bullet prefixed with "
            "'**User directive:**'.",
        )

        # Re-use the step's cached system prompt: the summary call runs at
        # step end while the cachePoint from the conversation is still warm.
        summary = converse_text(
            self.client, self.model, system, summary_msgs,
            max_tokens=512, temperature=0.7,
            cache_system=True,
        )

        print(f"\n\033[33m{'~'*44}")
        print(f"  Step {step['number']} Summary")
        print(f"{'~'*44}\033[0m")
        print(summary)
        print(f"\033[33m{'~'*44}\033[0m\n")
        return summary

    def _save_completed_step(
        self,
        step_num,
        step_name: str,
        summary: str,
        conversation: list[dict],
        msg_origins: "dict[int, dict] | None" = None,
    ):
        self.session.save_step(
            step_num, step_name, summary, conversation, msg_origins,
        )
        if step_num in {s["number"] for s in STEPS}:
            emit_resume_marker("step_saved", step_num)

    # ── Confirmation logic ────────────────────────────────────────────

    @staticmethod
    def _asks_to_proceed(text: str) -> bool:
        """True if CARL's message asks the user to proceed to the next step."""
        lower = text.lower()
        return any(
            p in lower
            for p in [
                "ready to proceed",
                "proceed to step",
                "move on to step",
                "shall we proceed",
                "shall we move on",
                "move to step",
                "move to the next",
                "go ahead and proceed",
                "ready to finish",
                "ready to end",
                "finish step",
                "finish this step",
                "end step",
                "end this step",
                "shall we close out",
                "close out step",
                "close out this step",
                "close out step 10",
                "close the session",
                "end the session",
                "are we done",
                "done with the session",
            ]
        )

    @staticmethod
    def _is_finish_command(text: str) -> bool:
        """True if the user explicitly asks to end the current final/revisit step."""
        lower = text.strip().lower()
        if len(lower) > 120:
            return False
        if any(
            w in lower
            for w in [
                "not done", "not finished", "not yet", "don't",
                "do not", "wait", "hold on",
            ]
        ):
            return False
        exact = {
            "/done", "/finish", "/end", "/complete",
            "done", "finish", "finished", "end", "complete",
            "we're done", "we are done", "all done", "looks done",
            "that is all", "that's all", "you can stop", "end step 10",
            "finish step 10", "complete step 10",
        }
        return lower in exact or lower.startswith(
            ("done ", "finish ", "finished ", "end step", "complete step")
        )

    def _is_affirmative(text: str) -> bool:
        """True if user input is a short, clear confirmation."""
        lower = text.strip().lower()
        if len(lower) > 60:
            return False
        if any(
            w in lower
            for w in [
                "but", "however", "wait", "actually", "change",
                "what about", "hold on", "not yet", "no",
            ]
        ):
            return False
        positives = [
            "yes", "y", "proceed", "continue", "next", "go", "go ahead",
            "looks good", "lgtm", "ok", "okay", "sure", "confirmed",
            "ready", "let's go", "lets go", "move on", "sounds good",
            "perfect", "great", "yep", "yeah", "absolutely",
        ]
        return any(lower.startswith(p) for p in positives)

    @staticmethod
    def _contains_tool_call_text(text: str) -> bool:
        """True if the model output contains tool-call-like text patterns.

        This happens when the model tries to call tools but none are defined
        in the API request — it falls back to outputting tool calls as text.
        """
        indicators = [
            "<tool_call>",
            '{"name": "create_file"',
            '{"name": "write_file"',
            '{"name": "read_file"',
            "```python\n# file:",
        ]
        lower = text.lower()
        return any(ind.lower() in lower for ind in indicators)

    # ── File handling (references step) ───────────────────────────────

    def _inject_file_content(self, user_input: str) -> str:
        """Detect file paths in user input, read them, append content."""
        tokens = user_input.split()
        appended: list[str] = []

        for token in tokens:
            clean = token.strip("\"'`,;:().!")
            if not clean:
                continue
            path = Path(clean)
            if not path.is_absolute():
                path = Path(self.carl_dir) / clean
            if path.is_file() and path.suffix in (".md", ".txt", ".text", ".rst"):
                content = self.session.read_file(str(path))
                if content:
                    self.session.copy_reference(str(path))
                    print(f"  \033[32m[Read: {clean} -- {len(content)} chars]\033[0m")
                    appended.append(
                        f"\n\n--- Content of {clean} ---\n"
                        f"{content}\n"
                        f"--- End of {clean} ---"
                    )

        if appended:
            return user_input + "".join(appended)
        return user_input

    # ── Auto-save and crash recovery ──────────────────────────────────

    def _auto_save_conversation(self):
        """Save the current step's conversation to disk (no summary)."""
        if self._current_step and len(self._current_messages) > 1:
            self.session.save_conversation(
                self._current_step["number"],
                self._current_step["name"],
                self._current_messages[1:],  # skip the [Begin step] message
                {k - 1: v for k, v in self._current_msg_origins.items()},
            )

    def _save_on_exit(self):
        """Save whatever we can when the session ends (clean or crash)."""
        self._kill_background_processes()

        if self.revisit_mode:
            if self._revisit_completed:
                return
            step = self._current_step or REVISIT_STEP
            step_num = step["number"]
            if self._current_messages and len(self._current_messages) > 1:
                conv_origins = {k - 1: v for k, v in self._current_msg_origins.items()}
                self.session.save_conversation(
                    step_num, step["name"], self._current_messages[1:], conv_origins
                )
            with open(self.session.memory_path, "a", encoding="utf-8") as f:
                f.write(
                    f"\n**Revisit ended during Step {step_num}: "
                    f"{step['name']}**\n"
                )
            print(f"\n  Session saved: {self.session.session_dir}")
            return

        if self.current_step_idx >= len(STEPS):
            return  # completed normally, everything already saved

        step = STEPS[self.current_step_idx]
        step_num = step["number"]

        # Save the conversation if we have one
        if self._current_messages and len(self._current_messages) > 1:
            conv_origins = {k - 1: v for k, v in self._current_msg_origins.items()}
            self.session.save_conversation(
                step_num, step["name"], self._current_messages[1:], conv_origins
            )
            # Try to generate a summary
            try:
                summary = self._summarize(
                    self._current_system, self._current_messages, step
                )
                self.session.save_step(
                    step_num, step["name"], summary, self._current_messages[1:], conv_origins
                )
            except Exception:
                pass  # API may be down — conversation is already saved

        with open(self.session.memory_path, "a", encoding="utf-8") as f:
            f.write(
                f"\n**Session ended during Step {step_num}: "
                f"{step['name']}**\n"
            )
        print(f"\n  Session saved: {self.session.session_dir}")

    def _kill_background_processes(self):
        """Terminate all background processes started by this session.

        Processes are started with start_new_session=True, so each gets its
        own process group.  We kill the entire group to ensure both the shell
        wrapper and the child python process are terminated.
        """
        import signal as _signal

        if self.revisit_mode:
            return
        pid_file = self.session.session_dir / "code" / ".bg_pids.json"
        if not pid_file.exists():
            return
        try:
            pids = json.loads(pid_file.read_text())
        except Exception:
            return
        for log_file, pid in pids.items():
            try:
                os.killpg(pid, _signal.SIGTERM)
                print(f"  \033[33m[Stopped background process PID {pid} ({log_file})]\033[0m")
            except (ProcessLookupError, PermissionError, OSError):
                try:
                    os.kill(pid, _signal.SIGTERM)
                    print(f"  \033[33m[Stopped background process PID {pid} ({log_file})]\033[0m")
                except (ProcessLookupError, PermissionError):
                    pass

    # ── Trillian persona evolution ────────────────────────────────────

    _TRILLIAN_STEPS = [
        (1, "Problem Definition"), (2, "References"), (2.5, "Literature Search"),
        (3, "Parameter Exploration"), (4, "Information Proposal"),
        (5, "Action Definition"), (6, "Reward Definition"),
    ]

    def _gather_trillian_conversations(self) -> str:
        """Read step 1-6 conversation files and return concatenated text."""
        from session import Session
        parts = []
        for step_num, step_name in self._TRILLIAN_STEPS:
            label = Session._step_label(step_num)
            conv_path = self.session.session_dir / f"step_{label}_conversation.md"
            if not conv_path.exists():
                continue
            text = conv_path.read_text(encoding="utf-8", errors="replace")
            if len(text) > 12_000:
                text = text[:12_000] + "\n[... truncated ...]"
            parts.append(f"=== Step {step_num}: {step_name} ===\n{text}")
        return "\n\n".join(parts)

    def _persona_triage(self, persona: str, conversations: str) -> tuple[bool, str]:
        """Step A: decide whether the persona needs updating."""
        system = (
            "You are analyzing a CARL session to determine if the Trillian persona "
            "specification needs updating. Trillian is a bot that simulates a domain-expert "
            "user in CARL's reinforcement learning design workflow (Steps 1-6).\n\n"
            "Compare the CURRENT persona specification against the ACTUAL user behavior "
            "observed in the session conversations. Look for:\n"
            "1. Substantial deviations from the persona's described behavior patterns "
            "(NOT minor variations -- only systematic differences)\n"
            "2. Entirely new types of interactions or response patterns not covered by "
            "the persona at all\n"
            "3. Consistent quantitative feedback (specific values, thresholds, durations) "
            "that the persona should encode but currently does not\n\n"
            "Do NOT flag:\n"
            "- Minor wording differences that still match the persona's spirit\n"
            "- One-off responses that are within the expected range of variation\n"
            "- Domain-specific content (the persona describes behavior patterns, "
            "not domain knowledge)\n\n"
            "Respond with EXACTLY one of:\n"
            "- YES: [1-2 sentence explanation of what needs updating]\n"
            "- NO"
        )
        user_msg = (
            f"=== CURRENT PERSONA SPECIFICATION ===\n{persona}\n"
            f"=== END PERSONA ===\n\n"
            f"=== SESSION CONVERSATIONS (Steps 1-6) ===\n{conversations}\n"
            f"=== END CONVERSATIONS ===\n\n"
            f"Does the persona need updating based on this session?"
        )
        result = self._call_model_non_streaming(
            self.persona_update_model, system,
            [{"role": "user", "content": user_msg}],
            max_tokens=150, temperature=0.2,
        )
        if result is None:
            return False, ""
        if result.upper().startswith("YES"):
            explanation = result[3:].strip().lstrip(":").strip()
            return True, explanation
        return False, ""

    def _persona_focused_update(
        self, persona: str, conversations: str, triage_explanation: str,
    ) -> str | None:
        """Step B: generate the updated persona with surgical modifications."""
        line_count = persona.count("\n") + 1
        char_count = len(persona)
        system = (
            "You are updating the Trillian persona specification based on observed "
            "session data. Trillian is a bot that simulates a domain-expert user in "
            "CARL's RL design workflow.\n\n"
            "CRITICAL CONSTRAINTS:\n"
            "1. SURGICAL MODIFICATIONS ONLY. Change the minimum necessary. Most "
            "sections should remain untouched.\n"
            f"2. DO NOT GROW THE FILE. The output must have at most 2% more lines "
            f"than the input. The current file has {line_count} lines -- your output "
            f"must not exceed {int(line_count * 1.02) + 1} lines.\n"
            "3. PRESERVE STRUCTURE. Keep all existing section numbers, headings, and "
            "formatting. Do not reorganize, rename sections, or change the markdown "
            "structure.\n"
            "4. SLOW EVOLUTION. This persona evolves over many sessions. Make "
            "incremental adjustments, not wholesale rewrites. Only codify patterns "
            "that are clearly consistent.\n"
            "5. INCORPORATE QUANTITATIVE FEEDBACK. If the session reveals consistent "
            "numerical values, parameter preferences, or duration patterns, encode "
            "them as examples.\n"
            "6. REPLACE RATHER THAN APPEND. If adding a new example, remove an older "
            "one. If adding a new rule, see if an existing rule can be refined to "
            "subsume it.\n"
            "7. OUTPUT THE COMPLETE FILE. Return the entire updated persona file. "
            "Start with the first line of the file and end with the last line."
        )
        user_msg = (
            f"=== TRIAGE FINDING ===\n{triage_explanation}\n\n"
            f"=== CURRENT PERSONA ({line_count} lines, {char_count} characters) ===\n"
            f"{persona}\n=== END PERSONA ===\n\n"
            f"=== SESSION CONVERSATIONS (Steps 1-6) ===\n{conversations}\n"
            f"=== END CONVERSATIONS ===\n\n"
            f"Output the updated persona file. Remember: at most "
            f"{int(line_count * 1.02) + 1} lines, surgical modifications only."
        )
        return self._call_model_non_streaming(
            self.persona_update_model, system,
            [{"role": "user", "content": user_msg}],
            max_tokens=16384, temperature=0.3,
        )

    def _persona_consistency_check(self, updated_persona: str) -> str | None:
        """Step C: verify the updated persona is internally consistent."""
        line_count = updated_persona.count("\n") + 1
        system = (
            "You are reviewing an updated Trillian persona specification for internal "
            "consistency. Check for:\n\n"
            "1. Contradictions between sections (e.g., a rule in 'Never Do' that "
            "conflicts with an example in 'Answering Technical Questions')\n"
            "2. Contradictions within sections (e.g., two examples that demonstrate "
            "opposite behaviors)\n"
            "3. Instructions that have become ambiguous or unclear after editing\n\n"
            "If you find inconsistencies, resolve them by making minimal edits. "
            "Preserve the overall structure and line count.\n\n"
            "If the file is already consistent, return it unchanged.\n\n"
            "OUTPUT THE COMPLETE FILE in either case."
        )
        user_msg = (
            f"=== PERSONA TO REVIEW ({line_count} lines) ===\n"
            f"{updated_persona}\n=== END PERSONA ===\n\n"
            f"Review for internal consistency. Output the final version of the file."
        )
        return self._call_model_non_streaming(
            self.persona_update_model, system,
            [{"role": "user", "content": user_msg}],
            max_tokens=16384, temperature=0.2,
        )

    def _update_trillian_persona(self) -> None:
        """Post-session: evaluate and optionally update the Trillian persona."""
        try:
            persona_path = trillian_persona_path(self.carl_dir)
            if not persona_path.exists():
                return

            print("\n  \033[36m[Persona update: analyzing session interactions...]\033[0m")

            persona = persona_path.read_text(encoding="utf-8", errors="replace")
            conversations = self._gather_trillian_conversations()
            if not conversations.strip():
                print("  \033[33m[Persona update: no conversation data found]\033[0m")
                return

            original_lines = persona.count("\n") + 1
            original_chars = len(persona)
            max_lines = int(original_lines * 1.02) + 1
            max_chars = int(original_chars * 1.05)

            # Step A: triage
            needs_update, explanation = self._persona_triage(persona, conversations)
            if not needs_update:
                print("  \033[36m[Persona update: no update needed]\033[0m")
                return

            print(f"  \033[36m[Persona update: update warranted -- {explanation}]\033[0m")
            print("  \033[36m[Persona update: generating targeted edits...]\033[0m")

            # Step B: focused update
            updated = self._persona_focused_update(persona, conversations, explanation)
            if updated is None:
                print("  \033[33m[Persona update: update generation failed]\033[0m")
                return

            # Size gate after Step B
            updated_lines = updated.count("\n") + 1
            updated_chars = len(updated)
            if updated_lines > max_lines:
                print(
                    f"  \033[33m[Persona update rejected: grew from "
                    f"{original_lines} to {updated_lines} lines]\033[0m"
                )
                return
            if updated_chars > max_chars:
                print(
                    f"  \033[33m[Persona update rejected: grew from "
                    f"{original_chars} to {updated_chars} chars]\033[0m"
                )
                return

            # Validation: must start with expected header
            if not updated.lstrip().startswith("# Trillian:"):
                print("  \033[33m[Persona update rejected: malformed output]\033[0m")
                return

            # Step C: consistency check
            print("  \033[36m[Persona update: checking consistency...]\033[0m")
            checked = self._persona_consistency_check(updated)
            if checked is not None:
                # Size gate again on consistency-checked output
                checked_lines = checked.count("\n") + 1
                checked_chars = len(checked)
                if checked_lines <= max_lines and checked_chars <= max_chars and checked.lstrip().startswith("# Trillian:"):
                    updated = checked
                    updated_lines = checked_lines

            # Backup and write
            backup_path = persona_path.with_suffix(".md.bak")
            backup_path.write_text(persona, encoding="utf-8")
            persona_path.write_text(updated, encoding="utf-8")

            print(
                f"  \033[36m[Persona update: saved "
                f"({original_lines} -> {updated_lines} lines)]\033[0m"
            )

            # Log to session memory
            with open(self.session.memory_path, "a", encoding="utf-8") as f:
                f.write(
                    f"\n**Persona updated:** {explanation} "
                    f"({original_lines} -> {updated_lines} lines)\n"
                )
        except Exception as e:
            print(f"  \033[33m[Persona update failed: {e}]\033[0m")

    # ── Session naming ─────────────────────────────────────────────────

    def _rename_session_dir(self):
        """After Step 1, ask the LLM for a short topic slug and rename the session dir."""
        summary = self.session.step_decisions.get(1, "")
        if not summary:
            return
        log_stopped = False
        try:
            raw = converse_text(
                self.client, self.model,
                "Output ONLY a short lowercase slug (2-3 words max, "
                "underscores, no spaces, no punctuation, no explanation). "
                "Example: portfolio_opt",
                [{"role": "user", "content": summary}],
                max_tokens=30, temperature=0.0,
            ).strip()
            # Sanitize: keep only lowercase alphanumeric and underscores
            slug = re.sub(r"[^a-z0-9_]", "_", raw.lower())
            slug = re.sub(r"_+", "_", slug).strip("_")[:30]
            if slug:
                if self._session_log is not None:
                    self._session_log.stop()
                    log_stopped = True
                self.session.rename_with_topic(slug)
                if self._session_log is not None:
                    self._session_log.session_dir = self.session.session_dir
                    self._session_log.start()
                    log_stopped = False
                print(f"  \033[32m[Session renamed: {self.session.session_dir.name}]\033[0m")
        except Exception:
            pass  # non-critical — keep the timestamp name

    # ── Display helpers ───────────────────────────────────────────────

    def _print_banner(self):
        print()
        print("\033[1m" + "=" * 56)
        print("   CARL -- Chat Assisted RL")
        print("=" * 56 + "\033[0m")
        print(f"  Session  : {self.session.timestamp}")
        print(f"  Model    : {self.model}")
        print(f"  Region   : {self.region}")
        print(f"  Directory: {self.session.session_dir}")
        print(f"  Commands : /next  /skip  /status  /suggest  /quit")
        print()

    def _print_revisit_banner(self):
        print("  Mode     : Step 10 revisit")
        print(f"  Revisits : {self.session.session_dir}")
        print("  Note     : normal Step 1-9 workflow is not running")
        print()

    def _print_resume_banner(self):
        print("  Mode     : resume")
        print(f"  Resuming : Step {self.resume_info.step_number}: {self.resume_info.step_name}")
        print("  Note     : restarting from the top of the unfinished step")
        for warning in self.resume_info.warnings or []:
            print(f"  Warning  : {warning}")
        print()

    def _print_step_header(self, step_num: int, name: str):
        print(f"\n\033[1;35m{'='*50}")
        if self.revisit_mode and step_num == REVISIT_STEP["number"]:
            print(f"  Step {step_num}: {name}")
        else:
            print(f"  Step {step_num}/9: {name}")
        print(f"{'='*50}\033[0m")

    def _print_done(self):
        print("\n\033[1;32m" + "=" * 56)
        print("   CARL workflow complete!")
        print(f"   Session: {self.session.session_dir}")
        print("=" * 56 + "\033[0m\n")

    def _close_session_log(self):
        if self._session_log is not None:
            self._session_log.stop()

    def _prompt_user(self) -> str | None:
        """Read user input. Returns None on quit/EOF.

        If Trillian has a suggestion, pre-fills the input buffer so the user
        can edit it in-place before pressing Enter.
        """
        try:
            prefill = self._trillian_suggestion or ""
            if prefill:
                readline.set_startup_hook(lambda: readline.insert_text(prefill))
            try:
                text = input(f"\n\033[32mYou:\033[0m ")
            finally:
                readline.set_startup_hook()
            if text.strip().lower() in ("quit", "exit", "/quit", "/exit"):
                return None
            return text
        except (EOFError, KeyboardInterrupt):
            print()
            return None
