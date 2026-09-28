#!/usr/bin/env python3
"""CARL Web UI -- lightweight Flask frontend with polling-based UI."""

import io
import json
import os
import queue
import re
import socket
import subprocess
import sys
from datetime import date
from pathlib import Path

# Backstop for any socket op that doesn't set its own timeout (e.g. the
# container-credentials refresh on 111.222.333.4:80, which has hung in
# CLOSE_WAIT and blocked the whole process in poll()). Bedrock calls set
# their own timeouts via botocore Config and are unaffected.
socket.setdefaulttimeout(60)

import boto3
from botocore.config import Config
from flask import Flask, render_template, request, jsonify

from bedrock_converse import (
    converse_once, converse_stream, converse_text, extract_text_from_converse,
    thinking_config_for_step, thinking_max_tokens_for_step,
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
    ResumeError,
    build_resume_info,
    emit_resume_marker,
    resolve_resume_session,
)

# ── Constants ──────────────────────────────────────────────────────────
_TOOL_STEPS = {8, 9}
_ONE_TURN_STEPS = {7}
_LITERATURE_STEPS = {2.5}
_REVISIT_DIR: Path | None = None
_RESUME_DIR: Path | None = None
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

# Shared with chat.py via tool_runtime; aliased so existing call sites read
# the same as before.
_MAX_TOOL_ROUNDS = MAX_TOOL_ROUNDS
_STEP9_DEFAULT_BUDGET_SECONDS = STEP9_DEFAULT_BUDGET_SECONDS
_CHECK_PROCESS_MIN_WAIT = CHECK_PROCESS_MIN_WAIT
_SUGGEST_SYNTHESIS_SYSTEM = SUGGEST_SYNTHESIS_SYSTEM
_SUGGEST_SYNTHESIS_USER = SUGGEST_SYNTHESIS_USER
_SUGGEST_DOCUMENT_SYSTEM = SUGGEST_DOCUMENT_SYSTEM
_SUGGEST_DOCUMENT_USER = SUGGEST_DOCUMENT_USER


# ── Bedrock client ─────────────────────────────────────────────────────

def _build_bedrock_client(region: str):
    # Tighten timeouts on the container-credentials fetcher so a half-closed
    # socket can't wedge the process (observed: 169.254.170.2:80 stuck in
    # CLOSE_WAIT while main thread slept in do_poll forever). TIMEOUT_SECONDS
    # is passed to urllib3 Timeout() and covers both connect and read.
    from botocore.utils import ContainerMetadataFetcher
    ContainerMetadataFetcher.TIMEOUT_SECONDS = 5

    # read_timeout must comfortably exceed the time Opus 5 takes to generate
    # max_tokens=16384 on autonomous tool turns (Steps 8–9); 120s is too tight
    # and fails mid-generation. 900s matches chat.py.
    config = Config(read_timeout=900, connect_timeout=60, retries={"max_attempts": 3})
    session = boto3.Session(region_name=region)
    return session.client(service_name="bedrock-runtime", config=config)


# ── Load .env ──────────────────────────────────────────────────────────

def _load_dotenv(carl_dir: str):
    env_path = Path(carl_dir) / ".env"
    if not env_path.is_file():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


# ── App setup ──────────────────────────────────────────────────────────

CARL_DIR = str(Path(__file__).resolve().parent)
_load_dotenv(CARL_DIR)

app = Flask(__name__)

REGION = os.environ.get("AWS_REGION", _DEFAULT_REGION)
TRILLIAN_MODEL = os.environ.get("CARL_TRILLIAN_MODEL", _DEFAULT_TRILLIAN_MODEL)
PERSONA_UPDATE_MODEL = os.environ.get("CARL_PERSONA_UPDATE_MODEL", _DEFAULT_PERSONA_UPDATE_MODEL)
REPORT_MODEL = os.environ.get("CARL_REPORT_MODEL", _DEFAULT_REPORT_MODEL)
_HELP_REQUESTED = any(arg in {"-h", "--help"} for arg in sys.argv[1:])


def _early_arg_value(name: str) -> str | None:
    try:
        idx = sys.argv.index(name)
    except ValueError:
        return None
    if idx + 1 >= len(sys.argv):
        return None
    return sys.argv[idx + 1]


_EARLY_RESUME_ARG = None if _HELP_REQUESTED else _early_arg_value("--resume")
if _EARLY_RESUME_ARG is not None:
    try:
        build_resume_info(CARL_DIR, _EARLY_RESUME_ARG)
    except ResumeError as e:
        print(f"Cannot resume: {e}")
        sys.exit(2)

bedrock = None if _HELP_REQUESTED else _build_bedrock_client(REGION)


def _resolve_model(client) -> str:
    """Try the preferred model; fall back to 4.6 if unavailable."""
    preferred = os.environ.get("CARL_MODEL", _DEFAULT_MODEL)
    return resolve_model_id(client, preferred, log=lambda msg: print(msg, flush=True))


MODEL = _DEFAULT_MODEL if _HELP_REQUESTED else _resolve_model(bedrock)
_trillian_persona = load_trillian_persona(CARL_DIR)

# In-memory state (single-user). For multi-user, key by session id.
state: dict = {}

# Event and log buffers for polling (thread-safe)
import threading

_lock = threading.Lock()
_log_entries: list[dict] = []
_log_counter = 0
_chat_events: list[dict] = []
_chat_counter = 0
_busy = False  # True while background generation is running
_interjection_queue: queue.Queue = queue.Queue()


def _log(text: str, level: str = "text"):
    """Append a log entry for the terminal pane."""
    global _log_counter
    with _lock:
        _log_counter += 1
        _log_entries.append({"id": _log_counter, "text": text, "level": level})
        if len(_log_entries) > 2000:
            _log_entries[:] = _log_entries[-1000:]


def _emit(event: str, data: dict):
    """Append a chat event for the frontend."""
    global _chat_counter
    with _lock:
        _chat_counter += 1
        _chat_events.append({"id": _chat_counter, "event": event, "data": data})
        if len(_chat_events) > 500:
            _chat_events[:] = _chat_events[-250:]


class _TeeWriter:
    """Write to both a stream and a log file."""
    def __init__(self, original, log_file):
        self._original = original
        self._log_file = log_file
        # Strip ANSI codes from log file output
        self._ansi_re = re.compile(r'\033\[[0-9;]*m')

    def write(self, text):
        self._original.write(text)
        try:
            clean = self._ansi_re.sub('', text)
            self._log_file.write(clean)
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


_original_stdout = sys.stdout
_original_stderr = sys.stderr
_log_file_handle = None


def _start_session_log(session_dir: Path):
    """Redirect stdout/stderr to also write to a log file in the session dir."""
    global _log_file_handle
    _stop_session_log()
    log_path = session_dir / "web.log"
    _log_file_handle = open(log_path, "a", encoding="utf-8")
    sys.stdout = _TeeWriter(_original_stdout, _log_file_handle)
    sys.stderr = _TeeWriter(_original_stderr, _log_file_handle)


def _stop_session_log():
    """Restore original stdout/stderr and close the log file."""
    global _log_file_handle
    sys.stdout = _original_stdout
    sys.stderr = _original_stderr
    if _log_file_handle:
        try:
            _log_file_handle.close()
        except Exception:
            pass
        _log_file_handle = None


def _init_state():
    """Initialize or reset the conversation state."""
    global _log_counter, _chat_counter
    write_report = bool(state.get("write_report", False))
    no_persona_update = bool(state.get("no_persona_update", False))
    _kill_background_processes()
    _stop_session_log()
    state.clear()
    state["write_report"] = write_report
    state["no_persona_update"] = no_persona_update
    resume_info = build_resume_info(CARL_DIR, _RESUME_DIR) if _RESUME_DIR is not None else None
    state["revisit_mode"] = _REVISIT_DIR is not None
    state["resume_mode"] = resume_info is not None
    state["session"] = (
        Session.open_existing(CARL_DIR, _REVISIT_DIR)
        if _REVISIT_DIR is not None
        else Session.open_existing(CARL_DIR, resume_info.session_dir)
        if resume_info is not None
        else Session(CARL_DIR)
    )
    if resume_info is not None:
        state["session"].step_decisions = dict(resume_info.step_decisions)
    state["resume_info"] = resume_info
    state["resume_warnings"] = resume_info.warnings if resume_info is not None else []
    state["step_idx"] = resume_info.step_idx if resume_info is not None else 0
    state["messages"] = []
    state["awaiting_confirmation"] = False
    state["started"] = False
    state["trillian_suggestion"] = None
    state["suggest_pending"] = None
    state["suggest_used"] = resume_info.suggest_used if resume_info is not None else False
    state["user_skip_25"] = resume_info.user_skip_25 if resume_info is not None else False
    state["msg_origins"] = {}
    state["revisit_completed"] = False
    state["revisit_preflight_done"] = False
    state["revisit_guidance"] = ""
    state["revisit_exclusions"] = []
    with _lock:
        _log_entries.clear()
        _log_counter = 0
        _chat_events.clear()
        _chat_counter = 0
    _start_session_log(state["session"].session_dir)



def _sanitize_messages(messages: list[dict]):
    """Repair message list so every tool_use/tool_result pair is consistent.

    Rules enforced:
    1. Every assistant message with tool_use blocks must be immediately
       followed by a user message whose tool_result ids match exactly.
    2. If the pair is broken, BOTH messages are removed (truncated).
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

        # Check the next message for matching tool_results
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
            # Truncate: remove this message and everything after it
            del messages[i:]
            break
        i += 1


# ── Tool execution ─────────────────────────────────────────────────────

def _execute_tool(name: str, input_data: dict) -> str:
    sess: Session = state["session"]
    if name == "write_file":
        code_dir = sess.session_dir / "code"
        rel_path = normalize_code_relpath(input_data["path"])
        if state.get("revisit_mode") and not re.match(r"^revisit_\d+/", rel_path.replace("\\", "/")):
            return "Refusing Step 10 write outside code/revisit_N/."
        content = input_data["content"]
        full_path = code_dir / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")
        return f"File written: code/{rel_path} ({len(content)} chars)"
    if name == "run_command":
        code_dir = sess.session_dir / "code"
        command = input_data["command"]
        if state.get("revisit_mode") and is_revisit_training_command(command):
            return revisit_training_refusal(command)
        try:
            compat_output = run_compat_command(command, code_dir)
            if compat_output is not None:
                return compat_output[:15000]
            # Clamp the timeout to what's left of the step's budget, so a
            # 10-minute command cannot run past a 2-minute remainder.
            timeout = 600
            deadline = state.get("deadline")
            if deadline is not None:
                timeout = deadline.clamp_timeout(timeout)
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True,
                cwd=str(code_dir), timeout=timeout,
            )
            output = (result.stdout + result.stderr).strip()
            return output[:15000]
        except subprocess.TimeoutExpired as e:
            limit = e.timeout or 600
            return (
                f"Command timed out after {int(limit)}s"
                + (
                    " (capped by this step's remaining wall-clock budget)"
                    if limit < 600
                    else ""
                )
                + ". Use run_background for long jobs."
            )
        except Exception as e:
            return f"Error: {e}"
    if name == "run_background":
        code_dir = sess.session_dir / "code"
        command = input_data["command"]
        log_file = input_data["log_file"]
        log_path = code_dir / log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)
        pid_file = code_dir / ".bg_pids.json"
        try:
            with open(log_path, "w") as lf:
                proc = subprocess.Popen(
                    command, shell=True, stdout=lf, stderr=subprocess.STDOUT,
                    cwd=str(code_dir),
                    start_new_session=True,
                )
            pids = {}
            if pid_file.exists():
                pids = json.loads(pid_file.read_text())
            # Use log_file:pid as key — append _N suffix to avoid overwriting
            key = log_file
            n = 2
            while key in pids:
                key = f"{log_file}_{n}"
                n += 1
            pids[key] = proc.pid
            pid_file.write_text(json.dumps(pids))
            return f"Started (PID {proc.pid}). Output streaming to code/{log_file}. Use check_process to monitor."
        except Exception as e:
            return f"Error starting background process: {e}"
    if name == "check_process":
        import time
        code_dir = sess.session_dir / "code"
        log_file = input_data["log_file"]
        tail_lines = input_data.get("tail_lines", 50)
        log_path = code_dir / log_file
        pid_file = code_dir / ".bg_pids.json"
        pid = None
        if pid_file.exists():
            pids = json.loads(pid_file.read_text())
            pid = pids.get(log_file)
        status = "unknown"
        if pid is not None:
            try:
                os.kill(pid, 0)
                status = "running"
            except ProcessLookupError:
                status = "finished"
            except PermissionError:
                status = "running"
        # If still running, wait before returning so the model doesn't burn
        # tool rounds polling rapidly. The wait is clamped to the step's
        # remaining budget: this sleep is the single biggest way a step
        # overruns, since each poll costs a minute or more of wall clock.
        deadline = state.get("deadline")
        if status == "running":
            wait = _get_check_interval()
            if deadline is not None:
                wait = deadline.clamp_sleep(wait)
            if wait > 0:
                _log(f"[waiting {wait:.0f}s before checking...]", "info")
                time.sleep(wait)
            else:
                _log("[budget spent; not sleeping before check]", "info")
            # Re-check status after sleep
            try:
                os.kill(pid, 0)
                status = "running"
            except ProcessLookupError:
                status = "finished"
            except PermissionError:
                status = "running"
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
        if deadline is not None:
            result += f"\n{deadline.status_line()}"
        return result[:15000]
    if name == "list_revisit_artifacts":
        return json.dumps(
            discover_artifacts(
                sess.session_dir,
                excluded_paths=state.get("revisit_exclusions", []),
            ),
            indent=2,
        )
    if name == "read_artifact":
        return read_artifact(
            sess.session_dir,
            input_data["path"],
            input_data.get("max_chars", 12000),
            excluded_paths=state.get("revisit_exclusions", []),
        )
    if name == "describe_plot":
        return describe_plot(
            bedrock, MODEL, sess.session_dir,
            input_data["path"],
            excluded_paths=state.get("revisit_exclusions", []),
        )
    if name == "prepare_revisit_workspace":
        return prepare_revisit_workspace_tool_response(sess.session_dir, input_data)
    if name == "search_literature":
        from literature import (
            search_papers_with_status, format_results_for_tool,
            save_literature_results,
        )
        # Per-step search budget, keyed by step so it resets on step change.
        counts = state.setdefault("literature_search_counts", {})
        step_key = _current_step_number()
        done = counts.get(step_key, 0)
        if done >= MAX_LITERATURE_SEARCHES:
            _log(
                f"[Search limit reached: {done}/{MAX_LITERATURE_SEARCHES} "
                "this step — refusing further searches]", "result",
            )
            _emit("info", {
                "text": f"Literature search limit reached "
                        f"({MAX_LITERATURE_SEARCHES} per step)."
            })
            return LITERATURE_LIMIT_MESSAGE
        counts[step_key] = done + 1
        query = input_data["query"]
        limit = min(input_data.get("limit", 10), 20)
        _log(
            f"[Searching ({done + 1}/{MAX_LITERATURE_SEARCHES}): "
            f"\"{query}\" (limit={limit})]", "tool",
        )
        papers, search_errors = search_papers_with_status(query, limit)
        if papers:
            save_literature_results(
                sess.session_dir, papers,
                bedrock_client=bedrock, summary_model=TRILLIAN_MODEL,
            )
            _log(f"[Found {len(papers)} papers]", "result")
        else:
            _log("[No results]", "result")
        if search_errors:
            _log(f"[Backend failures: {'; '.join(search_errors)}]", "result")
        return format_results_for_tool(papers, search_errors)
    return f"Unknown tool: {name}"


# ── Streaming single turn ─────────────────────────────────────────────

def _stream_single_turn(system, messages, tools, max_tokens, thinking_config=None):
    """Stream one Converse call. Yields ("text", str), ("tool_start", name),
    ("usage", dict), then ("result", (blocks, stop_reason)).

    System prompt and tool schemas are cached via Converse cachePoint markers
    with a 1h TTL, so every turn after the first in a step reuses the cached
    prefix (see bedrock_converse._cache_point)."""
    yield from converse_stream(
        bedrock, MODEL, system, messages,
        tools=tools, max_tokens=max_tokens, temperature=0.7,
        cache_system=True, cache_tools=bool(tools),
        thinking_config=thinking_config,
    )


def _should_auto_continue(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if asks_user_question(stripped):
        return False
    lower = stripped.lower()
    # Check the tail of the text (last 300 chars) for continuation signals.
    # The model often writes long analysis between tool calls but ends with
    # "Let me fix..." or "Now I'll create..." — we should continue.
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
    # Short text that doesn't match — likely a brief status update, continue
    if len(lower) <= 300:
        return True
    return False


def _asks_to_proceed(text: str) -> bool:
    lower = text.lower()
    return any(p in lower for p in [
        "ready to proceed", "proceed to step", "move on to step",
        "shall we proceed", "shall we move on", "move to step",
        "move to the next", "go ahead and proceed",
        "ready to finish", "ready to end", "finish step",
        "finish this step", "end step", "end this step",
        "shall we close out", "close out step", "close out this step",
        "close out step 10", "close the session", "end the session",
        "are we done", "done with the session",
    ])


def _is_finish_command(text: str) -> bool:
    lower = text.strip().lower()
    if len(lower) > 120:
        return False
    if any(w in lower for w in ["not done", "not finished", "not yet", "don't", "do not", "wait", "hold on"]):
        return False
    exact = {
        "/done", "/finish", "/end", "/complete",
        "done", "finish", "finished", "end", "complete",
        "we're done", "we are done", "all done", "looks done",
        "that is all", "that's all", "you can stop", "end step 10",
        "finish step 10", "complete step 10",
    }
    return lower in exact or lower.startswith(("done ", "finish ", "finished ", "end step", "complete step"))


def _is_affirmative(text: str) -> bool:
    lower = text.strip().lower()
    if len(lower) > 60:
        return False
    if any(w in lower for w in [
        "but", "however", "wait", "actually", "change",
        "what about", "hold on", "not yet", "no",
    ]):
        return False
    positives = [
        "yes", "y", "proceed", "continue", "next", "go", "go ahead",
        "looks good", "lgtm", "ok", "okay", "sure", "confirmed",
        "ready", "let's go", "lets go", "move on", "sounds good",
        "perfect", "great", "yep", "yeah", "absolutely",
    ]
    return any(lower.startswith(p) for p in positives)


# ── Step wall-clock budgets ─────────────────────────────────────────────
#
# Steps 8 and 9 get an enforced deadline built from the Step 7 budgets. The
# active deadline lives in state["deadline"] and is consulted at four places,
# because a limit that misses any one of them leaks: the tool-round loop, the
# auto-continue loop, the check_process sleep, and the run_command timeout.


def _capture_step7_budgets(step7_text: str):
    """Persist the budget markers from Step 7's raw turn text.

    Deliberately reads the raw text rather than the summary: the summarizer is
    asked for concise bullets and drops machine-readable lines, which silently
    reverted an observed session to the default budget.
    """
    sess = state.get("session")
    if not sess or not step7_text:
        return
    for step_num, marker in ((8, TESTING_BUDGET_MARKER), (9, TRAINING_BUDGET_MARKER)):
        minutes = _parse_budget_marker(step7_text, marker)
        if minutes:
            sess.set_step_budget(step_num, minutes)
            _log(f"[step {step_num} budget] {minutes} min recorded from Step 7", "info")


def _parse_budget_marker(text: str, marker: str) -> int | None:
    m = re.search(
        re.escape(marker.rstrip(":")) + r"\s*:\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE
    )
    if not m:
        return None
    minutes = int(float(m.group(1)))
    return minutes if minutes > 0 else None


def _resolve_step_budget_seconds(step_num) -> tuple[int, str]:
    """Budget for a step: persisted marker first, then Step 7 text, then default."""
    sess = state.get("session")
    minutes = sess.get_step_budget_minutes(step_num) if sess else None
    if minutes:
        _log(f"[step {step_num:g} budget] {minutes} min from persisted marker", "info")
        return minutes * 60, "persisted"
    step7_text = sess.step_decisions.get(7, "") if sess else ""
    return resolve_step_budget(
        step_num, step7_text or "", log=lambda msg: _log(msg, "info")
    )


def _start_step_deadline(step_num) -> "StepDeadline | None":
    """Register the wall-clock deadline for a step; None for non-tool steps."""
    if step_num not in _TOOL_STEPS:
        state["deadline"] = None
        return None
    budget_seconds, source = _resolve_step_budget_seconds(step_num)
    deadline = StepDeadline(budget_seconds, source=source)
    state["deadline"] = deadline
    state["wrap_up_sent"] = False
    state["hard_stop_done"] = False
    minutes = budget_seconds // 60
    _log(f"[step {step_num:g}] wall-clock budget {minutes} min ({source})", "info")
    _emit("budget", {
        "step": step_num, "minutes": minutes, "source": source,
        "text": f"Step {step_num:g} has a {minutes}-minute wall-clock budget.",
    })
    return deadline


def _ensure_step_deadline(step_num) -> "StepDeadline | None":
    """Deadline for a step, starting the clock only if it isn't already running.

    Steps 8 and 9 normally start their clock in the autonomous driver; this
    covers the interactive and resume entry points, without restarting a clock
    that is already ticking. Any other step clears the deadline.
    """
    if step_num in _TOOL_STEPS and state.get("deadline") is not None:
        return state["deadline"]
    return _start_step_deadline(step_num)


def _budget_hard_expired() -> bool:
    """True once the step is past its budget *and* its wrap-up grace window."""
    deadline = state.get("deadline")
    return deadline is not None and deadline.hard_expired()


def _budget_wrap_up_text() -> str | None:
    """The wrap-up directive, once, the first time a step goes over budget.

    Returns None when there is no deadline, the budget still has time, or the
    directive was already delivered for this step.
    """
    deadline = state.get("deadline")
    if deadline is None or not deadline.soft_expired():
        return None
    if state.get("wrap_up_sent"):
        return None
    state["wrap_up_sent"] = True
    step_num = _current_step_number() or 0
    msg = f"[budget exhausted for step {step_num:g}; asking for a final summary]"
    _log(msg, "info")
    _emit("budget_exhausted", {
        "step": step_num,
        "text": (
            f"Step {step_num:g} reached its {deadline.budget_seconds // 60}-minute "
            "wall-clock budget. Wrapping up."
        ),
    })
    return wrap_up_message(step_num, deadline)


def _stop_background_jobs_for_budget(reason: str) -> bool:
    """Kill this step's background jobs once, because it ran out of budget."""
    if state.get("hard_stop_done"):
        return True
    state["hard_stop_done"] = True
    step_num = _current_step_number() or 0
    msg = f"[step {step_num:g} {reason}; stopping background jobs]"
    _log(msg, "info")
    print(f"\n  \033[36m{msg}\033[0m", flush=True)
    _emit("budget_hard_stop", {
        "step": step_num,
        "text": (
            f"Step {step_num:g} reached its wall-clock limit. Background jobs "
            "stopped; summarizing what was completed."
        ),
    })
    _kill_background_processes()
    return True


def _enforce_budget_hard_stop() -> bool:
    """Kill background jobs at the hard cut. True if the cut was enforced."""
    if not _budget_hard_expired():
        return False
    return _stop_background_jobs_for_budget("wall-clock limit reached")


def _get_check_interval() -> int:
    """Seconds check_process waits, sized from the *current* step's budget.

    An earlier version always used the Step 9 budget, so Step 8 inherited a
    90-second poll interval and spent a third of one session asleep.
    """
    deadline = state.get("deadline")
    if deadline is not None:
        return check_interval_seconds(deadline.budget_seconds)
    budget_seconds, _ = _resolve_step_budget_seconds(9)
    return check_interval_seconds(budget_seconds)


# ── Trillian suggestion ─────────────────────────────────────────────────

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


def _has_pending_auto_turn(messages: list[dict]) -> bool:
    """True when the last user message was generated by the controller.

    Tool results and internal continuation nudges require another assistant turn
    before the UI should be handed back to the user.
    """
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


def _build_suggest_context(latest_request: str) -> str:
    """Build search context from the full current-step conversation."""
    lines: list[str] = []
    for msg in state.get("messages", []):
        text = _extract_text(msg).strip()
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


def _get_trillian_suggestion(step_num: int, step_name: str,
                           messages: list[dict]) -> str | None:
    """Ask Trillian to suggest a user response. Returns None if skipped."""
    if step_num > _TRILLIAN_LAST_ACTIVE_STEP:
        return None

    trillian_msgs = []
    for msg in messages:
        text = _extract_text(msg)
        if not text or text == "[Begin step]":
            continue
        trillian_msgs.append({"role": msg["role"], "content": text})

    if not trillian_msgs:
        return None

    if trillian_msgs[-1]["role"] == "user":
        return None  # Trillian only responds after CARL speaks

    trillian_msgs.append({
        "role": "user",
        "content": "Generate your suggested response to CARL's latest message.",
    })

    system = build_trillian_prompt(_trillian_persona, step_num, step_name)
    try:
        text = converse_text(
            bedrock, TRILLIAN_MODEL, system, trillian_msgs,
            max_tokens=256, temperature=0.5,
        ).strip()
    except Exception as e:
        _log(f"[Trillian unavailable: {e}]", "info")
        return None

    if text == "[SKIP]" or not text:
        return None
    return text


# ── Rename session after Step 1 ───────────────────────────────────────

def _rename_session():
    sess: Session = state["session"]
    summary = sess.step_decisions.get(1, "")
    if not summary:
        return
    try:
        raw = converse_text(
            bedrock, MODEL,
            "Output ONLY a short lowercase slug (2-3 words max, "
            "underscores, no spaces, no punctuation, no explanation). "
            "Example: portfolio_opt",
            [{"role": "user", "content": summary}],
            max_tokens=30, temperature=0.0,
        ).strip()
        slug = re.sub(r"[^a-z0-9_]", "_", raw.lower())
        slug = re.sub(r"_+", "_", slug).strip("_")[:30]
        if slug:
            sess.rename_with_topic(slug)
    except Exception:
        pass


# ── Summarize step ─────────────────────────────────────────────────────

def _summarize(system, messages, step, tools=None):
    summary_msgs = append_user_request(
        messages,
        "Provide a concise bullet-point summary of the key "
        "decisions and information gathered in this step. "
        "Just the bullets, no preamble.\n\n"
        "If the user gave any directive that constrains later steps "
        "(e.g. 'skip X', 'mandatory: Y'), include it verbatim as a "
        "dedicated bullet prefixed with '**User directive:**'.",
    )
    # Re-use the step's cached system prompt: the summary call runs at step
    # end while the cachePoint from the conversation is still warm.
    # `tools` must be forwarded when `messages` contains tool_use/tool_result
    # blocks (autonomous steps) — otherwise Bedrock rejects the call for
    # missing toolConfig.
    return converse_text(
        bedrock, MODEL, system, summary_msgs,
        tools=tools,
        max_tokens=1536, temperature=0.7,
        cache_system=True, cache_tools=bool(tools),
    )


# ── File injection (references step) ──────────────────────────────────

def _inject_file_content(user_input: str) -> str:
    tokens = user_input.split()
    appended = []
    for token in tokens:
        clean = token.strip("\"'`,;:().!")
        if not clean:
            continue
        path = Path(clean)
        if not path.is_absolute():
            path = Path(CARL_DIR) / clean
        if path.is_file() and path.suffix in (".md", ".txt", ".text", ".rst"):
            content = state["session"].read_file(str(path))
            if content:
                state["session"].copy_reference(str(path))
                appended.append(
                    f"\n\n--- Content of {clean} ---\n{content}\n--- End of {clean} ---"
                )
    return user_input + "".join(appended) if appended else user_input


# ── /suggest feature ──────────────────────────────────────────────────

def _call_llm_non_streaming(system: str, messages: list[dict],
                            max_tokens: int = 2048,
                            temperature: float = 0.7) -> str | None:
    """Non-streaming Converse call against the primary CARL model."""
    try:
        return converse_text(
            bedrock, MODEL, system, messages,
            max_tokens=max_tokens, temperature=temperature,
        ).strip()
    except Exception as e:
        _log(f"[LLM call failed: {e}]", "error")
        return None


def _call_model(model_id: str, system: str, messages: list[dict],
                max_tokens: int = 2048, temperature: float = 0.3) -> str | None:
    """Non-streaming Converse call with explicit model ID."""
    try:
        return converse_text(
            bedrock, model_id, system, messages,
            max_tokens=max_tokens, temperature=temperature,
        ).strip()
    except Exception as e:
        _log(f"[LLM call failed: {e}]", "error")
        return None


# ── Trillian persona evolution ────────────────────────────────────────

_TRILLIAN_STEPS = [
    (1, "Problem Definition"), (2, "References"), (2.5, "Literature Search"),
    (3, "Parameter Exploration"), (4, "Information Proposal"),
    (5, "Action Definition"), (6, "Reward Definition"),
]


def _gather_trillian_conversations(sess) -> str:
    """Read step 1-6 conversation files and return concatenated text."""
    from session import Session
    parts = []
    for step_num, step_name in _TRILLIAN_STEPS:
        label = Session._step_label(step_num)
        conv_path = sess.session_dir / f"step_{label}_conversation.md"
        if not conv_path.exists():
            continue
        text = conv_path.read_text(encoding="utf-8", errors="replace")
        if len(text) > 12_000:
            text = text[:12_000] + "\n[... truncated ...]"
        parts.append(f"=== Step {step_num}: {step_name} ===\n{text}")
    return "\n\n".join(parts)


def _clip_feedback_text(text: str, limit: int = 700) -> str:
    """Keep persona-triage examples compact."""
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n[... clipped ...]"


def _extract_persona_feedback(conversations: str) -> str:
    """Extract accepted/edited Trillian suggestions into a compact digest."""
    step_re = re.compile(
        r"^=== Step ([\d.]+): ([^=]+) ===\n",
        re.MULTILINE,
    )
    matches = list(step_re.finditer(conversations))
    if not matches:
        step_chunks = [("?", "Unknown Step", conversations)]
    else:
        step_chunks = []
        for i, match in enumerate(matches):
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(conversations)
            step_chunks.append((match.group(1), match.group(2).strip(), conversations[start:end]))

    edited_pat = re.compile(
        r"^\*\*Trillian suggestion:\*\*\n"
        r"(.*?)\n"
        r"^\*\*User \(edited and sent to CARL\):\*\*\n"
        r"(.*?)(?=\n^\*\*(?:CARL|Trillian suggestion|User)\b|\Z)",
        re.DOTALL | re.MULTILINE,
    )
    accepted_pat = re.compile(
        r"^\*\*Trillian suggestion \(accepted by user\):\*\*\n"
        r"(.*?)(?=\n^\*\*(?:CARL|Trillian suggestion|User)\b|\Z)",
        re.DOTALL | re.MULTILINE,
    )

    by_step: list[tuple[str, str, int, int]] = []
    edited: list[tuple[str, str, str, str]] = []
    accepted: list[tuple[str, str, str]] = []

    for step_num, step_name, chunk in step_chunks:
        step_edited = [
            (
                _clip_feedback_text(m.group(1)),
                _clip_feedback_text(m.group(2)),
            )
            for m in edited_pat.finditer(chunk)
        ]
        step_accepted = [
            _clip_feedback_text(m.group(1), limit=350)
            for m in accepted_pat.finditer(chunk)
        ]
        by_step.append((step_num, step_name, len(step_edited), len(step_accepted)))
        edited.extend((step_num, step_name, t, u) for t, u in step_edited)
        accepted.extend((step_num, step_name, t) for t in step_accepted)

    lines = [
        "=== TRILLIAN FEEDBACK DIGEST ===",
        f"Totals: edited={len(edited)}, accepted={len(accepted)}",
        "By step:",
    ]
    for step_num, step_name, edited_count, accepted_count in by_step:
        lines.append(
            f"- Step {step_num}: {step_name}: edited={edited_count}, accepted={accepted_count}"
        )

    if edited:
        lines.append("\nEdited suggestions (supervised correction signal):")
        for i, (step_num, step_name, trillian, final) in enumerate(edited[:12], 1):
            lines.extend([
                f"\n[{i}] Step {step_num}: {step_name}",
                f"Trillian suggested: {trillian}",
                f"User sent: {final}",
            ])
        if len(edited) > 12:
            lines.append(f"\n[{len(edited) - 12} more edited suggestions omitted]")
    else:
        lines.append("\nEdited suggestions: none")

    if accepted:
        lines.append("\nAccepted suggestions (positive examples):")
        for i, (step_num, step_name, trillian) in enumerate(accepted[:6], 1):
            lines.extend([
                f"\n[{i}] Step {step_num}: {step_name}",
                f"Accepted: {trillian}",
            ])
        if len(accepted) > 6:
            lines.append(f"\n[{len(accepted) - 6} more accepted suggestions omitted]")
    else:
        lines.append("\nAccepted suggestions: none")

    return "\n".join(lines)


def _log_persona_update(message: str, level: str = "info") -> None:
    """Log persona-update details to both the UI buffer and web.log."""
    formatted = f"[{message}]"
    _log(formatted, level)
    print(formatted, flush=True)


def _persona_triage(persona: str, feedback_digest: str) -> tuple[bool, str]:
    """Step A: decide whether the persona needs updating."""
    system = (
        "You are analyzing a CARL session to determine if the Trillian persona "
        "specification needs updating. Trillian is a bot that simulates a domain-expert "
        "user in CARL's reinforcement learning design workflow (Steps 1-6).\n\n"
        "You will receive a compact feedback digest showing Trillian suggestions "
        "that the user either accepted or edited before sending to CARL. Treat "
        "edited suggestion/final user pairs as supervised correction signal. "
        "Treat accepted suggestions as positive evidence that the persona is "
        "already working.\n\n"
        "Compare the CURRENT persona specification against this feedback. Look for:\n"
        "1. Repeated edits that reveal a behavioral mismatch in Trillian's response "
        "style, level of detail, delegation pattern, or approval/correction pattern\n"
        "2. Recurring user preferences about how to answer quantitative/range "
        "questions that the persona should encode as response behavior\n"
        "3. New response patterns not covered by the persona that appear more than "
        "once, or appear once but are clearly a high-impact persona correction\n\n"
        "Do NOT flag:\n"
        "- Purely domain-specific facts that should not live in the persona\n"
        "- One isolated edit that does not imply a reusable behavior pattern\n"
        "- Edits that merely choose different technical values without changing "
        "how Trillian should behave\n"
        "- Minor wording differences that still match the persona's spirit\n\n"
        "Your first output token must be exactly YES or NO.\n"
        "Put the one-sentence reason after a colon.\n"
        "Do not begin with analysis, headings, or any other text.\n"
        "If uncertain, respond NO.\n"
        "- YES: [one sentence describing the update target]\n"
        "- NO: [one sentence reason]"
    )
    user_msg = (
        f"=== CURRENT PERSONA SPECIFICATION ===\n{persona}\n"
        f"=== END PERSONA ===\n\n"
        f"{feedback_digest}\n\n"
        f"Does the persona need updating based on this session?"
    )
    result = _call_model(
        PERSONA_UPDATE_MODEL, system,
        [{"role": "user", "content": user_msg}],
        max_tokens=150, temperature=0.2,
    )
    if result is None:
        return False, "triage model call failed"
    if result.upper().startswith("YES"):
        explanation = result[3:].strip().lstrip(":").strip()
        return True, explanation
    if result.upper().startswith("NO"):
        explanation = result[2:].strip().lstrip(":").strip()
        return False, explanation or "model found no reusable persona update"
    return False, f"unexpected triage response: {result[:120]}"


def _persona_focused_update(
    persona: str, conversations: str, triage_explanation: str,
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
    return _call_model(
        PERSONA_UPDATE_MODEL, system,
        [{"role": "user", "content": user_msg}],
        max_tokens=16384, temperature=0.3,
    )


def _persona_consistency_check(updated_persona: str) -> str | None:
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
    return _call_model(
        PERSONA_UPDATE_MODEL, system,
        [{"role": "user", "content": user_msg}],
        max_tokens=16384, temperature=0.2,
    )


def _update_trillian_persona() -> None:
    """Post-session: evaluate and optionally update the Trillian persona."""
    try:
        sess = state["session"]
        persona_path = trillian_persona_path(CARL_DIR)
        if not persona_path.exists():
            return

        _log_persona_update("Persona update: analyzing session interactions...")
        _emit("info", {"text": "Persona update: analyzing session interactions..."})

        persona = persona_path.read_text(encoding="utf-8", errors="replace")
        conversations = _gather_trillian_conversations(sess)
        if not conversations.strip():
            _log_persona_update("Persona update: no conversation data found")
            return
        feedback_digest = _extract_persona_feedback(conversations)

        original_lines = persona.count("\n") + 1
        original_chars = len(persona)
        max_lines = int(original_lines * 1.02) + 1
        max_chars = int(original_chars * 1.05)

        # Step A: triage
        needs_update, explanation = _persona_triage(persona, feedback_digest)
        if not needs_update:
            _log_persona_update(f"Persona update triage: NO -- {explanation}")
            _emit("info", {"text": f"Persona update: no update needed -- {explanation}"})
            return

        _log_persona_update(f"Persona update triage: YES -- {explanation}")
        _emit("info", {"text": f"Persona update: update warranted -- {explanation}"})
        _emit("info", {"text": "Persona update: generating targeted edits..."})

        before_path = sess.session_dir / "refs" / "Trillian-Persona-before-update.md"
        before_path.parent.mkdir(parents=True, exist_ok=True)
        before_path.write_text(persona)
        _log_persona_update(f"Persona update: saved original persona to {before_path}")

        # Step B: focused update
        updated = _persona_focused_update(persona, conversations, explanation)
        if updated is None:
            _log_persona_update("Persona update: update generation failed")
            _emit("error", {"text": "Persona update: generation failed"})
            return

        # Size gate after Step B
        updated_lines = updated.count("\n") + 1
        updated_chars = len(updated)
        if updated_lines > max_lines:
            _log_persona_update(f"Persona update rejected: grew from {original_lines} to {updated_lines} lines")
            _emit("error", {"text": f"Persona update rejected: grew from {original_lines} to {updated_lines} lines"})
            return
        if updated_chars > max_chars:
            _log_persona_update(f"Persona update rejected: grew from {original_chars} to {updated_chars} chars")
            _emit("error", {"text": f"Persona update rejected: grew from {original_chars} to {updated_chars} chars"})
            return

        # Validation
        if not updated.lstrip().startswith("# Trillian:"):
            _log_persona_update("Persona update rejected: malformed output")
            _emit("error", {"text": "Persona update rejected: malformed output"})
            return

        # Step C: consistency check
        _emit("info", {"text": "Persona update: checking consistency..."})
        checked = _persona_consistency_check(updated)
        if checked is not None:
            checked_lines = checked.count("\n") + 1
            checked_chars = len(checked)
            if checked_lines <= max_lines and checked_chars <= max_chars and checked.lstrip().startswith("# Trillian:"):
                updated = checked
                updated_lines = checked_lines

        # Backup and write
        backup_path = persona_path.with_suffix(".md.bak")
        backup_path.write_text(persona, encoding="utf-8")
        persona_path.write_text(updated, encoding="utf-8")

        msg = f"Persona update: saved ({original_lines} -> {updated_lines} lines)"
        _log_persona_update(msg)
        _emit("info", {"text": msg})

        # Log to session memory
        with open(sess.memory_path, "a", encoding="utf-8") as f:
            f.write(
                f"\n**Persona updated:** {explanation} "
                f"({original_lines} -> {updated_lines} lines)\n"
            )
    except Exception as e:
        _log_persona_update(f"Persona update failed: {e}", "error")
        _emit("error", {"text": f"Persona update failed: {e}"})


def _extract_formulation(suggestions_text: str, index: int) -> str:
    """Extract the Nth formulation block from the suggestions text.

    Formulations are delimited by **[N]. lines.  Returns just the chosen
    block, or the full text as fallback.
    """
    # Split on the **[N]. pattern, keeping the delimiter
    parts = re.split(r"(?=\*\*\[\d+\])", suggestions_text.strip())
    parts = [p.strip() for p in parts if p.strip()]
    for part in parts:
        m = re.match(r"\*\*\[(\d+)\]", part)
        if m and int(m.group(1)) == index:
            return part
    # Fallback: return everything with an explicit instruction
    return f"Formulation #{index} from the list below:\n{suggestions_text}"


def _parse_index_selection(choice: str, max_index: int) -> list[int] | None:
    """Parse comma-separated 1-based indices. Empty -> all indices."""
    choice = (choice or "").strip()
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


def _format_shortlist_prompt(papers: list[dict]) -> str:
    """Render numbered shortlist prompt text for the web UI."""
    lines = [
        "I found the following shortlist in `literature_search.md`:",
        "",
    ]
    for i, p in enumerate(papers, 1):
        lines.append(f"{i}. {p.get('title', 'Untitled')} ({p.get('year', '?')})")
    lines.extend([
        "",
        "Reply with paper numbers to use (e.g., `1,3,5`), press Enter for all, or type `c` to cancel.",
    ])
    return "\n".join(lines)


def _handle_suggest(user_input: str) -> str | None:
    """Phase 1 of /suggest: build the canonical 10-paper shortlist and
    ask the user which shortlist papers should drive synthesis.

    Stores shortlist + context in state["suggest_pending"].
    Returns None because selection is handled in subsequent web turns.
    """
    from literature import search_papers, save_literature_results

    sess: Session = state["session"]

    # 1. Extract search query — use LLM to distil short keyword queries
    clean_input = re.sub(r"/suggest", "", user_input, flags=re.IGNORECASE).strip().strip(",").strip()
    if not clean_input:
        clean_input = "reinforcement learning problem formulation"

    _log("[/suggest] Searching literature and generating candidates. This may take 1-2 minutes...", "info")
    _emit("info", {"text": "[/suggest] Searching literature and generating candidates. This may take 1-2 minutes..."})
    _emit("log", {"text": "[/suggest] Generating search queries..."})
    suggest_context = _build_suggest_context(clean_input)
    queries_text = _call_llm_non_streaming(
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
        queries = [q.strip().lstrip("-* ").strip() for q in queries_text.splitlines() if q.strip()]
    else:
        queries = [clean_input]

    # 2. Search the literature backends with each query, merge results
    all_papers: list[dict] = []
    seen_ids: set[str] = set()
    for q in queries[:3]:
        _log(f"[/suggest] Searching: \"{q}\"", "info")
        _emit("log", {"text": f"[/suggest] Searching: \"{q}\"..."})
        try:
            results = search_papers(q, limit=8)
        except Exception as e:
            _log(f"[/suggest] Search failed for \"{q}\": {e}", "error")
            continue
        for p in results:
            pid = p.get("paper_id") or p.get("title", "")
            if pid not in seen_ids:
                seen_ids.add(pid)
                all_papers.append(p)
    papers = all_papers

    if not papers:
        _log("[/suggest] No papers found", "warn")
        _emit("log", {"text": "[/suggest] No papers found. Try a more specific description."})
        return None
    _log(f"[/suggest] Found {len(papers)} papers across {len(queries)} queries", "info")
    save_literature_results(
        sess.session_dir, papers,
        bedrock_client=bedrock, summary_model=TRILLIAN_MODEL,
    )

    # 3. Format papers for LLM
    papers_formatted = "\n".join(
        f"[{i+1}]. {p['title']} ({p.get('year', '?')})\n   {(p.get('abstract') or '')[:300]}"
        for i, p in enumerate(papers[:10])
    )

    # 4. Synthesize candidate formulations
    _emit("log", {"text": "[/suggest] Synthesizing RL problem candidates..."})
    suggestions_text = _call_llm_non_streaming(
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
        _emit("log", {"text": "[/suggest] Suggestion generation failed."})
        return None

    # 5. Store pending state for Phase 2
    state["suggest_pending"] = {
        "clean_input": clean_input,
        "suggestions_text": suggestions_text,
        "papers_formatted": papers_formatted,
        "queries": queries,
    }

    # 6. Return the candidates as user context so CARL presents them
    _log("[/suggest] Candidates ready - waiting for user selection", "info")
    return (
        f"{clean_input}\n\n"
        f"I used /suggest and the system found {len(papers)} relevant papers "
        f"and synthesized these candidate RL problem formulations:\n\n"
        f"{suggestions_text}\n\n"
        f"Please present these options to me clearly (with a brief explanation "
        f"of each and the key references) and ask which one I'd like to explore."
    )


def _handle_suggest_v2(user_input: str) -> str | None:
    """Web /suggest flow anchored to a canonical 10-paper shortlist.

    After literature is found, this returns a normal user-context string so
    CARL can continue the conversation in its regular style.
    """
    from literature import search_papers, save_literature_results

    sess: Session = state["session"]

    clean_input = re.sub(r"/suggest", "", user_input, flags=re.IGNORECASE).strip().strip(",").strip()
    if not clean_input:
        clean_input = "reinforcement learning problem formulation"

    _log("[/suggest] Searching literature and generating candidates. This may take 1-2 minutes...", "info")
    _emit("info", {"text": "[/suggest] Searching literature and generating candidates. This may take 1-2 minutes..."})
    _emit("log", {"text": "[/suggest] Generating search queries..."})
    suggest_context = _build_suggest_context(clean_input)
    queries_text = _call_llm_non_streaming(
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
        queries = [q.strip().lstrip("-* ").strip() for q in queries_text.splitlines() if q.strip()]
    else:
        queries = [clean_input]
    queries = [q.strip().lstrip("-* ").strip() for q in queries if q.strip()]

    all_papers: list[dict] = []
    seen_ids: set[str] = set()
    for q in queries[:3]:
        _log(f"[/suggest] Searching: \"{q}\"", "info")
        _emit("log", {"text": f"[/suggest] Searching: \"{q}\"..."})
        try:
            results = search_papers(q, limit=8)
        except Exception as e:
            _log(f"[/suggest] Search failed for \"{q}\": {e}", "error")
            continue
        for p in results:
            pid = p.get("paper_id") or p.get("title", "")
            if pid not in seen_ids:
                seen_ids.add(pid)
                all_papers.append(p)

    if not all_papers:
        _log("[/suggest] No papers found", "warn")
        _emit("log", {"text": "[/suggest] No papers found. Try a more specific description."})
        return None

    _log(f"[/suggest] Found {len(all_papers)} papers across {len(queries)} queries", "info")
    shortlist_papers = all_papers[:10]
    save_literature_results(
        sess.session_dir, shortlist_papers,
        bedrock_client=bedrock, summary_model=TRILLIAN_MODEL,
        overwrite=True,
    )

    papers_formatted = "\n".join(
        f"[{i+1}]. {p['title']} ({p.get('year', '?')})\n   {(p.get('abstract') or '')[:300]}"
        for i, p in enumerate(shortlist_papers)
    )
    _emit("log", {"text": "[/suggest] Synthesizing RL problem candidates..."})
    suggestions_text = _call_llm_non_streaming(
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
        _emit("log", {"text": "[/suggest] Suggestion generation failed."})
        return None

    state["suggest_pending"] = {
        "stage": "formulation_selection",
        "clean_input": clean_input,
        "suggestions_text": suggestions_text,
        "papers_formatted": papers_formatted,
        "shortlist_papers": shortlist_papers,
        "queries": queries,
    }
    _log("[/suggest] Candidates ready - waiting for formulation selection", "info")
    return (
        f"{clean_input}\n\n"
        f"I used /suggest and the system found {len(shortlist_papers)} relevant papers "
        f"(saved to literature_search.md) and synthesized these candidate RL problem formulations:\n\n"
        f"{suggestions_text}\n\n"
        f"Please present these options clearly (with concise rationale and references) and ask which one I'd like to explore."
    )


def _handle_suggest_selection_v2(user_input: str) -> str | None:
    """Web /suggest follow-up: user selects a synthesized formulation."""
    pending = state.get("suggest_pending")
    if not pending:
        return None

    msg = (user_input or "").strip()
    msg_lower = msg.lower()

    if msg_lower in ("c", "cancel"):
        state["suggest_pending"] = None
        _emit("info", {"text": "Suggestion flow cancelled."})
        return None

    # formulation_selection stage
    choice_match = re.search(r"\b([1-9])\b", msg)
    if choice_match:
        chosen_index = int(choice_match.group(1))
    else:
        alias_map = {
            "one": 1, "first": 1,
            "two": 2, "second": 2,
            "three": 3, "third": 3,
            "four": 4, "fourth": 4,
            "five": 5, "fifth": 5,
        }
        chosen_index = None
        lower = msg.lower()
        for word, idx in alias_map.items():
            if re.search(rf"\b{word}\b", lower):
                chosen_index = idx
                break

    if chosen_index is None:
        _emit("info", {"text": "Please reply with a formulation number (1-5), or `c` to cancel."})
        return None
    clean_input = pending["clean_input"]
    suggestions_text = pending["suggestions_text"]
    papers_formatted = pending["papers_formatted"]
    queries = pending["queries"]
    shortlist_papers = pending.get("shortlist_papers", [])

    _emit("info", {"text": f"Generating detailed problem specification for option #{chosen_index}... (this may take 1-2 minutes)"})
    doc_text = _call_llm_non_streaming(
        system=_SUGGEST_DOCUMENT_SYSTEM,
        messages=[{"role": "user", "content":
            _SUGGEST_DOCUMENT_USER.format(
                chosen_formulation=_extract_formulation(suggestions_text, chosen_index),
                problem_description=clean_input,
                papers_formatted=papers_formatted,
            )}],
        max_tokens=8192,
        temperature=0.5,
    )
    if not doc_text:
        _emit("info", {"text": "Problem specification generation failed. You can try selecting a formulation again."})
        return None

    doc_text = (
        doc_text.rstrip()
        + "\n\n"
        + _format_literature_shortlist_section(shortlist_papers)
        + "\n"
    )

    title_line = next(
        (line.lstrip("#").strip() for line in doc_text.splitlines() if line.strip()),
        "rl_problem",
    )
    slug = re.sub(r"[^a-z0-9_]", "_", title_line.lower())
    slug = re.sub(r"_+", "_", slug).strip("_")[:40]
    if not slug:
        slug = "suggested_rl_problem"

    refs_dir = Path(CARL_DIR) / "refs"
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
        _log(f"[/suggest] Failed to save file: {e}", "error")
        return None

    sess: Session = state["session"]
    sess.copy_reference(str(refs_path))
    _log(f"[/suggest] Saved: refs/{slug}.md", "info")
    _emit("info", {"text": f"Problem specification written to **refs/{slug}.md** - you can read or edit this file before proceeding."})

    state["suggest_pending"] = None
    state["suggest_used"] = True

    return clean_input + f"\n\nrefs/{slug}.md"


def _handle_suggest_selection(user_input: str) -> str | None:
    """Phase 2 of /suggest: the user has chosen a formulation.
    Generate the full problem spec document and save to refs/.

    Returns a modified user_input with the refs/ path appended,
    or None on failure.
    """
    pending = state.get("suggest_pending")
    if not pending:
        return None

    # Extract the chosen index from user input (e.g. "1", "option 2", "#3")
    choice_match = re.search(r"\b([1-5])\b", user_input)
    if not choice_match:
        # No clear selection — clear pending and let CARL handle normally
        state["suggest_pending"] = None
        return None

    chosen_index = int(choice_match.group(1))
    state["suggest_pending"] = None  # consumed

    clean_input = pending["clean_input"]
    suggestions_text = pending["suggestions_text"]
    papers_formatted = pending["papers_formatted"]
    queries = pending["queries"]

    _emit("info", {"text": f"Generating detailed problem specification for option #{chosen_index}... (this may take 1-2 minutes)"})
    doc_text = _call_llm_non_streaming(
        system=_SUGGEST_DOCUMENT_SYSTEM,
        messages=[{"role": "user", "content":
            _SUGGEST_DOCUMENT_USER.format(
                chosen_formulation=_extract_formulation(suggestions_text, chosen_index),
                problem_description=clean_input,
                papers_formatted=papers_formatted,
            )}],
        max_tokens=8192,
        temperature=0.5,
    )
    if not doc_text:
        _emit("info", {"text": "Problem specification generation failed. You can try selecting a formulation again."})
        return None

    # Slugify the title
    title_line = next(
        (line.lstrip("#").strip() for line in doc_text.splitlines() if line.strip()),
        "rl_problem",
    )
    slug = re.sub(r"[^a-z0-9_]", "_", title_line.lower())
    slug = re.sub(r"_+", "_", slug).strip("_")[:40]
    if not slug:
        slug = "suggested_rl_problem"

    # Save to refs/<slug>.md
    refs_dir = Path(CARL_DIR) / "refs"
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
        _log(f"[/suggest] Failed to save file: {e}", "error")
        return None

    sess: Session = state["session"]
    sess.copy_reference(str(refs_path))
    _log(f"[/suggest] Saved: refs/{slug}.md", "info")
    _emit("info", {"text": f"Problem specification written to **refs/{slug}.md** - you can read or edit this file before proceeding."})

    # Mark /suggest as used so Step 2.5 can be skipped
    state["suggest_used"] = True

    return user_input + f"\n\nrefs/{slug}.md"


def _current_step_number():
    """Number of the step currently executing, or None outside the workflow."""
    workflow = _workflow_steps()
    idx = state.get("step_idx", 0)
    if 0 <= idx < len(workflow):
        return workflow[idx]["number"]
    return None


def _auto_save_conversation():
    """Save current step's conversation to disk for crash recovery."""
    if "session" not in state or "messages" not in state:
        return
    idx = state.get("step_idx", 0)
    workflow = _workflow_steps()
    if idx >= len(workflow):
        return
    messages = state["messages"]
    if len(messages) <= 1:
        return
    step = workflow[idx]
    conv_origins = {k - 1: v for k, v in state.get("msg_origins", {}).items()}
    state["session"].save_conversation(
        step["number"], step["name"], messages[1:], conv_origins
    )


def _kill_background_processes():
    """Terminate all background processes started by this session.

    Processes are started with start_new_session=True, so each gets its
    own process group.  We kill the entire group (os.killpg) to ensure
    both the shell wrapper and the child python process are terminated.
    """
    import signal as _signal

    if "session" not in state:
        return
    if state.get("revisit_mode"):
        return
    pid_file = state["session"].session_dir / "code" / ".bg_pids.json"
    if not pid_file.exists():
        return
    try:
        pids = json.loads(pid_file.read_text())
    except Exception:
        return
    for log_file, pid in pids.items():
        try:
            os.killpg(pid, _signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            # Fallback: try killing just the PID if group kill fails
            try:
                os.kill(pid, _signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass


def _workflow_steps() -> list[dict]:
    return [REVISIT_STEP] if state.get("revisit_mode") else STEPS


def _resume_mode_note(step_num) -> str | None:
    info = state.get("resume_info")
    if not info or step_num != info.step_number:
        return None
    return (
        f"Resume mode is restarting existing session: {state['session'].session_dir}\n"
        f"Restart from the top of Step {step_num}. Prior completed decisions "
        "were restored from memory.md. Partial artifacts from the unfinished "
        "step may already exist in code/, results/, or references/; inspect "
        "and reuse or repair them as appropriate instead of assuming a blank "
        "workspace."
    )


# ── Routes ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    if _REVISIT_DIR is not None:
        return render_template("step10.html")
    return render_template("index.html")


@app.route("/api/start", methods=["POST"])
def start():
    try:
        _init_state()
    except ResumeError as e:
        return jsonify({"error": str(e)}), 400
    sess = state["session"]
    workflow = _workflow_steps()
    return jsonify({
        "session": sess.timestamp,
        "directory": str(sess.session_dir),
        "model": MODEL,
        "steps": [{"number": s["number"], "name": s["name"]} for s in workflow],
        "revisit": state.get("revisit_mode", False),
        "resume": state.get("resume_mode", False),
        "resume_step": (
            {
                "number": state["resume_info"].step_number,
                "name": state["resume_info"].step_name,
            }
            if state.get("resume_info")
            else None
        ),
        "resume_warnings": state.get("resume_warnings", []),
    })


@app.route("/api/status")
def status():
    if not state:
        return jsonify({"started": False})
    idx = state.get("step_idx", 0)
    workflow = _workflow_steps()
    step = workflow[idx] if idx < len(workflow) else None
    return jsonify({
        "started": state.get("started", False),
        "step_idx": idx,
        "step": {"number": step["number"], "name": step["name"]} if step else None,
        "done": idx >= len(workflow),
        "session_dir": str(state["session"].session_dir) if "session" in state else None,
        "revisit": state.get("revisit_mode", False),
    })


@app.route("/api/logs")
def logs():
    """Return new log entries since the given ID (polling endpoint)."""
    since = int(request.args.get("since", 0))
    with _lock:
        if since == 0:
            entries = list(_log_entries)
        else:
            entries = [e for e in _log_entries if e["id"] > since]
    return jsonify({"entries": entries})


@app.route("/api/events")
def events():
    """Return new chat events since the given ID (polling endpoint)."""
    since = int(request.args.get("since", 0))
    with _lock:
        if since == 0:
            entries = list(_chat_events)
        else:
            entries = [e for e in _chat_events if e["id"] > since]
        busy = _busy
    return jsonify({"entries": entries, "busy": busy})


@app.route("/api/message", methods=["POST"])
def message():
    """Accept a user message and run CARL's response in a background thread."""
    global _busy
    data = request.get_json()
    user_input = data.get("message", "").strip()
    if not state or "session" not in state:
        _init_state()

    with _lock:
        if _busy:
            return jsonify({"status": "busy"}), 409
        _busy = True

    def run():
        global _busy
        try:
            _run_turn(user_input)
        except Exception as e:
            import traceback
            traceback.print_exc()
            _auto_save_conversation()
            _emit("error", {"text": f"Server error: {e}"})
        finally:
            _emit("turn_end", {})
            with _lock:
                _busy = False

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"status": "ok"})


@app.route("/api/interject", methods=["POST"])
def interject():
    """Accept a user message during autonomous execution."""
    data = request.get_json()
    msg = data.get("message", "").strip()
    if not msg:
        return jsonify({"status": "empty"}), 400
    _interjection_queue.put(msg)
    return jsonify({"status": "ok"})


_LAST_INTERACTIVE_STEP = 7  # After this step, execution is fully autonomous


def _finish_revisit():
    state["revisit_completed"] = True
    _emit("done", {"text": "Step 10 revisit complete."})
    _schedule_shutdown()


def _advance_step(step_num, system, messages, step, tools=None):
    """Save summary, advance state, emit events. Returns the new step_num."""
    sess: Session = state["session"]
    if step_num == 7:
        # Read the budget markers off the raw conversation, before the
        # summarizer gets a chance to drop them.
        _capture_step7_budgets(
            "\n".join(_extract_text(m) for m in messages if m.get("role") == "assistant")
        )
    summary = _summarize(system, messages, step, tools=tools)
    conv_origins = {k - 1: v for k, v in state.get("msg_origins", {}).items()}
    sess.save_step(step_num, step["name"], summary, messages[1:], conv_origins)
    if step_num in {s["number"] for s in STEPS}:
        emit_resume_marker("step_saved", step_num)
    if step_num == 1:
        _rename_session()
    state["step_idx"] += 1
    workflow = _workflow_steps()
    # Skip Step 2.5 if /suggest already did the literature search
    next_idx = state["step_idx"]
    if (
        not state.get("revisit_mode")
        and next_idx < len(workflow)
        and workflow[next_idx]["number"] == 2.5
        and (state.get("suggest_used") or state.get("user_skip_25"))
    ):
        reason = "already covered by /suggest" if state.get("suggest_used") else "skipped per user request"
        _emit("info", {"text": f"Skipping Step 2.5: Literature Search - {reason}"})
        state["step_idx"] += 1
    state["messages"] = []
    state["started"] = False
    state["awaiting_confirmation"] = False
    state["trillian_suggestion"] = None
    state["msg_origins"] = {}
    _emit("summary", {"text": summary, "step": step_num})
    next_idx = state["step_idx"]
    return workflow[next_idx]["number"] if next_idx < len(workflow) else None


def _run_turn(user_input):
    """Execute one turn of the conversation (runs in background thread)."""
    idx = state["step_idx"]
    workflow = _workflow_steps()
    if idx >= len(workflow):
        text = "Step 10 revisit complete." if state.get("revisit_mode") else "All steps complete."
        _emit("done", {"text": text})
        return

    step = workflow[idx]
    step_num = step["number"]
    is_revisit_step = state.get("revisit_mode") and step_num == REVISIT_STEP["number"]
    if step_num in _LITERATURE_STEPS:
        tools = [LITERATURE_TOOL]
    elif is_revisit_step:
        tools = REVISIT_TOOLS
    elif step_num in _TOOL_STEPS:
        tools = TOOLS
    else:
        tools = None
    sess: Session = state["session"]

    messages = state["messages"]
    if (
        state.get("resume_mode")
        and step_num in _TOOL_STEPS
        and (not state["started"] or not messages)
    ):
        _run_autonomous_steps()
        return

    run_revisit_opening = False
    if is_revisit_step and not state.get("revisit_preflight_done"):
        if not state["started"] or not messages:
            state["started"] = True
            state["messages"] = [{"role": "user", "content": "[Begin step]"}]
            messages = state["messages"]

            _emit("step", {"number": step_num, "name": step["name"]})
            _log(f"-- Step {step_num}: {step['name']} --", "step")

            prompt = build_revisit_preflight_prompt(sess.session_dir)
            messages.append({"role": "assistant", "content": prompt})
            _emit("text", {"text": prompt})
            state["awaiting_confirmation"] = False
            return

        state["revisit_guidance"] = normalize_revisit_guidance(user_input)
        state["revisit_exclusions"] = extract_revisit_exclusions(user_input)
        state["revisit_preflight_done"] = True
        messages.append({
            "role": "user",
            "content": user_input.strip() if user_input.strip() else "none",
        })
        run_revisit_opening = True

    skill_content = load_skill(CARL_DIR, step["skill"])
    previous = (
        build_revisit_context(
            sess.session_dir,
            guidance=state.get("revisit_guidance", ""),
            excluded_paths=state.get("revisit_exclusions", []),
        )
        if is_revisit_step
        else sess.get_previous_decisions()
    )
    skip = {2.5} if state.get("suggest_used") else None
    if is_revisit_step:
        mode_note = (
            f"Revisit mode is inspecting existing session: {sess.session_dir}\n"
            "The controller already collected optional pre-analysis guidance; "
            "honor it before inspecting artifacts."
        )
    else:
        mode_note = _resume_mode_note(step_num)
    revisit_workflow = [REVISIT_STEP] if state.get("revisit_mode") else None
    turn_deadline = _ensure_step_deadline(step_num)
    system = build_system_prompt(
        step, skill_content, previous, skip_steps=skip,
        workflow_steps=revisit_workflow, mode_note=mode_note,
        budget_minutes=(
            turn_deadline.budget_seconds // 60 if turn_deadline else None
        ),
    )

    messages = state["messages"]

    # First message of a step — CARL opens
    if not state["started"] or not messages:
        state["started"] = True
        state["messages"] = [{"role": "user", "content": "[Begin step]"}]
        messages = state["messages"]

        _emit("step", {"number": step_num, "name": step["name"]})
        _log(f"-- Step {step_num}: {step['name']} --", "step")
        if not is_revisit_step:
            emit_resume_marker("step_start", step_num)

        all_text, _ = _stream_full(
            system,
            messages,
            tools,
            require_user_question=step_num not in _TOOL_STEPS,
        )
        state["awaiting_confirmation"] = _asks_to_proceed(all_text)

        # Ask Trillian for a suggestion after CARL's opening
        suggestion = _get_trillian_suggestion(step_num, step["name"], messages)
        if suggestion:
            state["trillian_suggestion"] = suggestion
            _emit("trillian_suggestion", {"text": suggestion})
        return

    if run_revisit_opening:
        all_text, _ = _stream_full(
            system,
            messages,
            tools,
            require_user_question=True,
        )
        state["awaiting_confirmation"] = _asks_to_proceed(all_text)
        return

    # Trillian's suggestion was pre-filled in the textbox.
    # Whatever the user sent is the final text — no merge needed.
    origin = None
    if state.get("trillian_suggestion"):
        if user_input == state["trillian_suggestion"]:
            origin = {"type": "trillian_accepted", "trillian": state["trillian_suggestion"]}
        else:
            origin = {
                "type": "trillian_edited",
                "trillian": state["trillian_suggestion"],
                "final": user_input,
            }
    state["trillian_suggestion"] = None

    # Handle commands
    cmd = user_input.lower()
    if step_num == 2 and "/skip" in cmd:
        state["user_skip_25"] = True
    if state.get("revisit_mode") and _is_finish_command(user_input):
        if origin:
            state["msg_origins"][len(messages)] = origin
        if user_input:
            messages.append({"role": "user", "content": user_input})
        _advance_step(step_num, system, messages, step, tools=tools)
        _finish_revisit()
        return
    if cmd in ("/next", "/skip"):
        next_num = _advance_step(step_num, system, messages, step, tools=tools)
        if state.get("revisit_mode"):
            _finish_revisit()
        elif step_num >= _LAST_INTERACTIVE_STEP:
            _run_autonomous_steps()
        else:
            _emit("step_advance", {"next": state["step_idx"]})
        return

    # /suggest Phase 2: user is selecting from pending candidates
    if state.get("suggest_pending"):
        if "/suggest" in user_input.lower():
            # User wants a new search — clear pending and fall through to Phase 1
            state["suggest_pending"] = None
        else:
            suggest_result = _handle_suggest_selection_v2(user_input)
            if suggest_result is not None:
                user_input = suggest_result
            else:
                _emit("turn_end", {})
                return

    # /suggest Phase 1: keyword triggers search + candidate generation
    if step_num == 1 and "/suggest" in user_input.lower():
        suggest_result = _handle_suggest_v2(user_input)
        if suggest_result is None:
            _emit("turn_end", {})
            return
        user_input = suggest_result

    # Auto-read file paths from user input in early steps
    if step_num in (1, 2):
        user_input = _inject_file_content(user_input)

    # Check for step confirmation
    if state["awaiting_confirmation"] and _is_affirmative(user_input):
        if origin:
            state["msg_origins"][len(messages)] = origin
        messages.append({"role": "user", "content": user_input})
        next_num = _advance_step(step_num, system, messages, step, tools=tools)
        if state.get("revisit_mode"):
            _finish_revisit()
        elif step_num >= _LAST_INTERACTIVE_STEP:
            _run_autonomous_steps()
        else:
            _emit("step_advance", {"next": state["step_idx"]})
        return

    # Normal message — reject empty input (unless Trillian already filled it)
    if not user_input:
        _emit("turn_end", {})
        return
    if origin:
        state["msg_origins"][len(messages)] = origin
    messages.append({"role": "user", "content": user_input})
    try:
        all_text, _ = _stream_full(
            system,
            messages,
            tools,
            require_user_question=(
                step_num not in _TOOL_STEPS
                and step_num not in _ONE_TURN_STEPS
            ),
        )
    except Exception as e:
        _auto_save_conversation()
        _emit("error", {"text": f"API error: {e}. Progress saved."})
        return

    # ── One-turn step: auto-advance after first user turn ──
    if step_num in _ONE_TURN_STEPS:
        _advance_step(step_num, system, messages, step, tools=tools)
        if state.get("revisit_mode"):
            _finish_revisit()
        elif step_num >= _LAST_INTERACTIVE_STEP:
            _run_autonomous_steps()
        else:
            _emit("step_advance", {"next": state["step_idx"]})
        return

    # Detect model outputting tool calls as text in a non-tool step.
    # This means the model thinks the step is done and wants to code,
    # but the framework hasn't transitioned yet. Auto-advance.
    if not tools and _contains_tool_call_text(all_text):
        next_idx = state["step_idx"] + 1
        if next_idx < len(workflow) and workflow[next_idx]["number"] in _TOOL_STEPS:
            _log(
                f"[CARL tried to write code in Step {step_num} - auto-advancing to Step {workflow[next_idx]['number']}]",
                "info",
            )
            _emit("info", {
                "text": f"CARL is ready to start coding. Advancing to Step {workflow[next_idx]['number']}: {workflow[next_idx]['name']}..."
            })
            next_num = _advance_step(step_num, system, messages, step, tools=tools)
            if state.get("revisit_mode"):
                _finish_revisit()
            elif step_num >= _LAST_INTERACTIVE_STEP:
                _run_autonomous_steps()
            else:
                _emit("step_advance", {"next": state["step_idx"]})
            return

    state["awaiting_confirmation"] = _asks_to_proceed(all_text)

    # Ask Trillian for a suggestion after CARL responds
    suggestion = _get_trillian_suggestion(step_num, step["name"], messages)
    if suggestion:
        state["trillian_suggestion"] = suggestion
        _emit("trillian_suggestion", {"text": suggestion})


def _contains_tool_call_text(text: str) -> bool:
    """True if the model output contains tool-call-like text patterns."""
    indicators = [
        "<tool_call>",
        "<tool_calls>",
        '{"name": "create_file"',
        '{"name": "write_file"',
        '{"name": "read_file"',
        "<tool_name>",
        "```python\n# file:",
    ]
    lower = text.lower()
    return any(ind.lower() in lower for ind in indicators)


def _run_autonomous_steps():
    """Run remaining tool steps (8, 9) autonomously, with optional user interjection."""
    _emit("autonomous_start", {
        "text": "Starting autonomous execution. You can send a message at any time to guide CARL."
    })

    while state["step_idx"] < len(STEPS):
        idx = state["step_idx"]
        step = STEPS[idx]
        step_num = step["number"]
        sess: Session = state["session"]

        skill_content = load_skill(CARL_DIR, step["skill"])
        previous = sess.get_previous_decisions()
        skip = {2.5} if state.get("suggest_used") else None

        _emit("step", {"number": step_num, "name": step["name"]})
        _log(f"-- Step {step_num}: {step['name']} --", "step")
        emit_resume_marker("step_start", step_num)

        # Start the clock before the first turn, and tell the model what it is.
        deadline = _start_step_deadline(step_num)
        system = build_system_prompt(
            step, skill_content, previous, skip_steps=skip,
            mode_note=_resume_mode_note(step_num),
            budget_minutes=(deadline.budget_seconds // 60) if deadline else None,
        )

        # Resume with existing messages if recovering from an error,
        # otherwise start fresh.
        if state.get("started") and state.get("messages") and len(state["messages"]) > 1:
            messages = state["messages"]
            _log("[Resuming step with existing conversation]", "info")
        else:
            state["started"] = True
            state["messages"] = [{"role": "user", "content": "[Begin step]"}]
            messages = state["messages"]

        # Run the step's initial turn (or resume turn)
        # If resuming, inject a continue message so CARL picks up where it left off.
        if len(messages) > 1 and messages[-1]["role"] == "assistant":
            messages.append({"role": "user", "content": "[continue — resuming after transient API error]"})

        try:
            all_text, _ = _stream_full(system, messages, TOOLS)
        except Exception as e:
            _auto_save_conversation()
            _emit("error", {"text": f"API error in step {step_num}: {e}. Progress saved."})
            return

        # Auto-continue loop
        _auto_save_conversation()
        for _ in range(_MAX_AUTO_ROUNDS):
            if _asks_to_proceed(all_text):
                break
            if "[DONE]" in all_text:
                break
            # Out of budget and out of grace: stop driving the step. The
            # summarize call below still runs, so the work is recorded.
            if _enforce_budget_hard_stop():
                _log(f"[step {step_num} stopped at its wall-clock limit]", "info")
                break
            # Check for user interjection before continuing
            try:
                interjection = _interjection_queue.get_nowait()
            except queue.Empty:
                interjection = None

            if interjection:
                messages.append({"role": "user", "content": interjection})
                _emit("interjection", {"text": interjection})
                _log(f"[User interjection]: {interjection}", "info")
            else:
                # Over budget: ask for the landing instead of more work. Only
                # consulted when nothing else is being sent -- the call marks
                # the directive delivered, so it must not be discarded here.
                wrap_up = _budget_wrap_up_text()
                messages.append(
                    {"role": "user", "content": wrap_up or "[continue]"}
                )

            try:
                all_text, _ = _stream_full(system, messages, TOOLS)
            except Exception as e:
                _auto_save_conversation()
                _emit("error", {"text": f"API error in step {step_num}: {e}. Progress saved."})
                return
            _auto_save_conversation()

        # If the step went over budget it was told its jobs would be stopped;
        # make that true even when it landed its summary before the hard cut.
        if deadline is not None and deadline.soft_expired():
            _stop_background_jobs_for_budget("over its wall-clock budget")

        # Summarize and advance. Forward TOOLS so toolConfig is present on the
        # Bedrock request — messages will contain tool_use/tool_result blocks.
        summary = _summarize(system, messages, step, tools=TOOLS)
        sess.save_step(step_num, step["name"], summary, messages[1:])
        emit_resume_marker("step_saved", step_num)
        _emit("summary", {"text": summary, "step": step_num})
        state["step_idx"] += 1
        state["messages"] = []
        state["started"] = False
        state["msg_origins"] = {}

    if not state.get("no_persona_update"):
        _update_trillian_persona()

    if state.get("write_report"):
        _log("Generating academic report...", "info")
        _emit("info", {"text": "Generating academic report..."})
        from report import generate_report
        report_path = generate_report(state["session"], bedrock, REPORT_MODEL)
        if report_path:
            _log(f"Report saved: {report_path}", "result")
            _emit("info", {"text": f"Report saved: {report_path}"})
        else:
            _emit("error", {"text": "Report generation failed. Session data is still saved."})

    _emit("done", {"text": "All steps complete! Check the code/ directory for generated files."})
    emit_resume_marker("session_done")
    _schedule_shutdown()


def _schedule_shutdown():
    """Shut down the Flask server after all post-step work has completed.

    Delays briefly so the final events can be polled by the client, then
    sends SIGTERM to this process, which triggers _shutdown_handler and
    exits back to the command line.
    """
    import signal as _signal
    import time as _time

    def _do_shutdown():
        _time.sleep(3)
        _log("[Session complete — exiting back to command line]", "info")
        _emit("info", {"text": "Session complete — exiting back to command line."})
        _time.sleep(2)
        os.kill(os.getpid(), _signal.SIGTERM)

    threading.Thread(target=_do_shutdown, daemon=True).start()


def _stream_full(system, messages, tools=None, require_user_question=False):
    """Full response with tool-use loop. Emits chat events and logs.
    Returns (all_text, had_tool_use)."""
    max_tokens = 16384 if tools else 4096
    # Claude 5 reasons adaptively and those tokens come out of maxTokens, so a
    # step whose tier calls for thinking needs the larger budget — otherwise a
    # turn can spend the whole allowance reasoning and emit no text at all.
    # A prose-only turn is capped at the medium tier: measured on a Step 3
    # prompt, medium answers as fully as high for roughly half the tokens.
    step_num = _current_step_number()
    ceiling = None if tools else _TEXT_TURN_TIER_CEILING
    thinking_cfg = thinking_config_for_step(MODEL, step_num, ceiling)
    if thinking_cfg:
        max_tokens = thinking_max_tokens_for_step(step_num, max_tokens, ceiling)
    all_text = ""
    had_tool_use = False
    missing_question_continues = 0
    rounds = 0
    tail_continues = 0
    cap_logged = False
    blank_turns = 0

    # state["deadline"] is None outside Steps 8/9; those steps carry an
    # enforced wall-clock budget checked on every round.
    while (
        not _budget_hard_expired()
        and (
            rounds < _MAX_TOOL_ROUNDS
            or (
                _has_pending_auto_turn(messages)
                and tail_continues < _MAX_TOOL_ROUND_TAIL_CONTINUES
            )
        )
    ):
        if rounds >= _MAX_TOOL_ROUNDS:
            tail_continues += 1
            if not cap_logged:
                msg = "[tool-round cap reached; finishing pending tool/continue turn...]"
                print(f"\n  \033[36m{msg}\033[0m", flush=True)
                _log(msg, "info")
                _emit("info", {
                    "text": "Tool-round cap reached; finishing the pending internal continuation."
                })
                cap_logged = True
        rounds += 1
        _sanitize_messages(messages)

        content_blocks = []
        stop_reason = "end_turn"
        turn_text = ""

        for kind, payload in _stream_single_turn(
            system, messages, tools, max_tokens, thinking_config=thinking_cfg,
        ):
            if kind == "text":
                print(payload, end="", flush=True)
                _emit("text", {"text": payload})
                _log(payload, "text")
                turn_text += payload
            elif kind == "tool_start":
                msg = f"[calling {payload}...]"
                print(f"\n  \033[33m{msg}\033[0m", end="", flush=True)
                _log(msg, "tool")
            elif kind == "usage":
                u = payload
                msg = (
                    f"[tokens: in={u.get('input_tokens', 0)} "
                    f"(cached={u.get('cache_creation_input_tokens', 0)}, "
                    f"read={u.get('cache_read_input_tokens', 0)}) "
                    f"out={u.get('output_tokens', 0)}]"
                )
                print(f"\n  \033[90m{msg}\033[0m", flush=True)
                _log(msg, "info")
            elif kind == "result":
                content_blocks, stop_reason = payload

        tool_uses = [b for b in content_blocks if b["type"] == "tool_use"]
        all_text += turn_text

        if not tool_uses:
            if not turn_text.strip():
                # No visible output — typically the whole token allowance went
                # to adaptive reasoning. Recording a blank assistant turn would
                # make the *next* Converse call fail validation ("text content
                # blocks must contain non-whitespace text"), so retry the same
                # turn with more headroom instead of poisoning the history.
                blank_turns += 1
                if blank_turns > _MAX_BLANK_TURNS:
                    msg = "[model returned no text; stopping this step's turn]"
                    print(f"\n  \033[36m{msg}\033[0m", flush=True)
                    _log(msg, "info")
                    _emit("info", {
                        "text": "The model returned no text for this turn. "
                                "Send a message to retry."
                    })
                    break
                max_tokens = min(max_tokens * 2, _MAX_OUTPUT_TOKENS)
                msg = f"[empty response; retrying with max_tokens={max_tokens}...]"
                print(f"\n  \033[36m{msg}\033[0m", flush=True)
                _log(msg, "info")
                continue

            messages.append({"role": "assistant", "content": turn_text})

            if stop_reason == "max_tokens":
                msg = "[max_tokens — continuing...]"
                print(f"\n  \033[36m{msg}\033[0m", flush=True)
                _log(msg, "info")
                messages.append(
                    {"role": "user", "content": "[continue — you were cut off]"}
                )
                continue

            if had_tool_use and _should_auto_continue(turn_text):
                msg = "[auto-continuing...]"
                print(f"\n  \033[36m{msg}\033[0m", flush=True)
                _log(msg, "info")
                messages.append({"role": "user", "content": "[continue]"})
                had_tool_use = False
                continue
            if (
                require_user_question
                and missing_question_continues < 2
                and should_continue_for_missing_question(turn_text)
            ):
                msg = "[auto-continuing: no user question detected...]"
                print(f"\n  \033[36m{msg}\033[0m", flush=True)
                _log(msg, "info")
                messages.append({
                    "role": "user",
                    "content": MISSING_QUESTION_CONTINUE_PROMPT,
                })
                missing_question_continues += 1
                continue
            break

        had_tool_use = True
        messages.append({"role": "assistant", "content": content_blocks})

        tool_results = []
        for tu in tool_uses:
            inp = tu["input"]
            if tu["name"] == "write_file":
                label = f"write_file: {inp.get('path', '?')}"
            elif tu["name"] == "run_command":
                label = f"run_command: {inp.get('command', '?')[:80]}"
            elif tu["name"] == "run_background":
                label = f"run_background: {inp.get('command', '?')[:80]}"
            elif tu["name"] == "check_process":
                label = f"check_process: {inp.get('log_file', '?')}"
            elif tu["name"] == "search_literature":
                label = f"search_literature: {inp.get('query', '?')[:80]}"
            else:
                label = tu["name"]
            print(f"\n  \033[33m[{label}]\033[0m", flush=True)
            _log(f"[{label}]", "tool")
            try:
                result = _execute_tool(tu["name"], inp)
            except Exception as e:
                result = f"Error executing {tu['name']}: {e}"
            short = result[:300].replace('\n', ' ')
            print(f"  \033[32m-> {short}\033[0m", flush=True)
            _log(f"-> {short}", "result")
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu["id"],
                "content": result,
            })
        # Check for user interjection before sending tool results
        try:
            interjection = _interjection_queue.get_nowait()
        except queue.Empty:
            interjection = None

        if interjection:
            tool_results.append({"type": "text", "text": f"[User message]: {interjection}"})
            _emit("interjection", {"text": interjection})
            _log(f"[User interjection]: {interjection}", "info")

        # Over budget mid-tool-chain: ride along on the tool-result turn rather
        # than appending a second consecutive user message, which Converse
        # rejects. Same trick the interjection above uses.
        wrap_up = _budget_wrap_up_text()
        if wrap_up:
            tool_results.append({"type": "text", "text": wrap_up})

        messages.append({"role": "user", "content": tool_results})
        _auto_save_conversation()

    # Enforce the hard cut before reporting the cap, so a budget stop is not
    # misattributed to the tool-round cap.
    hard_stopped = _enforce_budget_hard_stop()

    if _has_pending_auto_turn(messages) and not hard_stopped:
        msg = "[tool-round cap reached with pending continuation; returning to user]"
        print(f"\n  \033[33m{msg}\033[0m", flush=True)
        _log(msg, "info")
        _emit("info", {
            "text": (
                "Tool-round cap reached before CARL could finish the pending "
                "tool result. Send a short continue message to resume."
            )
        })

    return all_text, had_tool_use


# ── Entry point ────────────────────────────────────────────────────────

def _shutdown_handler(signum, frame):
    """Handle Ctrl+C: kill background processes, save progress, exit."""
    print("\n\n  \033[31m[Interrupted - cleaning up...]\033[0m", flush=True)
    _kill_background_processes()
    _auto_save_conversation()
    _stop_session_log()
    sys.exit(0)


if __name__ == "__main__":
    import argparse
    import logging
    import signal
    parser = argparse.ArgumentParser(description="CARL Web UI")
    parser.add_argument(
        "--write-report", action="store_true",
        help="Generate a LaTeX+PDF academic report after all steps complete",
    )
    parser.add_argument(
        "--no-persona-update", action="store_true",
        help="Skip the Trillian persona update at session end (use when running multiple instances)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--revisit",
        help="Open an existing CARL session directory directly in Step 10 revisit mode",
    )
    mode.add_argument(
        "--resume",
        help="Resume an existing CARL session from its last unfinished step",
    )
    args = parser.parse_args()
    state["write_report"] = args.write_report
    state["no_persona_update"] = args.no_persona_update
    if args.revisit:
        _REVISIT_DIR = resolve_revisit_session(CARL_DIR, args.revisit)
    if args.resume:
        _RESUME_DIR = resolve_resume_session(CARL_DIR, args.resume)

    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)

    # Suppress repetitive polling endpoint log lines
    class _PollFilter(logging.Filter):
        def filter(self, record):
            msg = record.getMessage()
            return "/api/events" not in msg and "/api/logs" not in msg
    logging.getLogger("werkzeug").addFilter(_PollFilter())

    port = int(os.environ.get("CARL_PORT", 5001))
    if _REVISIT_DIR is not None:
        print(f"\n  CARL Web UI starting on http://localhost:{port}")
        print(f"  Revisit mode: {_REVISIT_DIR}\n")
    elif _RESUME_DIR is not None:
        try:
            info = build_resume_info(CARL_DIR, _RESUME_DIR)
        except ResumeError as e:
            print(f"Cannot resume: {e}")
            sys.exit(2)
        print(f"\n  CARL Web UI starting on http://localhost:{port}")
        print(f"  Resume mode: {_RESUME_DIR}")
        print(f"  Resuming Step {info.step_number}: {info.step_name}\n")
    else:
        print(f"\n  CARL Web UI starting on http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
