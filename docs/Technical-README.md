# CARL — Technical Reference

This document is a dense code-structure reference. It covers every file, class,
method, constant, data format, and integration point. Its purpose is to give a
complete working model of the codebase without requiring file reads.

> **Symbols over line numbers.** This reference locates code by file, class,
> and symbol name rather than line number, since line numbers drift as the code
> changes. The `~N lines` figures in the Repository Map are approximate size
> hints, not exact counts.

---

## Repository Map

```
carl.py                 CLI entry point (~90 lines)
chat.py                 Terminal chat engine — CARLChat class (~2,040 lines)
web.py                  Flask web UI — module-level functions, polling API (~2,480 lines)
session.py              Session directory + memory persistence — Session class (~195 lines)
steps.py                STEPS list, REVISIT_STEP, prompt builders, skill/persona loaders (~395 lines)
tool_runtime.py         Shared tool schemas, tool-loop limits, Step 10 refusal filter,
                        Step 7 budget parsing, /suggest templates (~490 lines)
command_compat.py       `tail` shim for run_command + code/ path resolution (~75 lines)
conversation_guards.py  "did the turn end with a question?" guards (~95 lines)
revisit.py              Step 10 revisit helpers, tools, workspace + appendix scripts (~745 lines)
revisit_report.py       Step 10 revisit LaTeX appendix publisher (~550 lines)
resume.py               Session resume: log markers, resume detection, web.log tee (~350 lines)
bedrock_converse.py     Shared Bedrock Converse API helper with prompt caching + thinking (~447 lines)
bedrock_models.py       Model-ID defaults + startup model resolution (~97 lines)
report.py               Post-session LaTeX/PDF report generator (~1,125 lines)
literature.py           Multi-backend literature search: arXiv, OpenAlex,
                        Semantic Scholar
carl_skills/            11 skill directories, each `<skill-name>/SKILL.md`
                        (normal workflow + Step 10 revisit)
jit_skills/             8 reference RL skill directories, each `<skill-name>/SKILL.md`
                        (source material the carl_skills cite in `metadata.sources`)
trillian_skills/        trillian-persona/SKILL.md (user-simulator spec, 276 lines)
tests/                  pytest suite (`python -m pytest tests/ -q`)
templates/index.html    Single-page web frontend (copy of index-light.html — served default)
templates/index-light.html  Light theme variant
templates/index-dark.html   Dark theme variant (Tokyo-Night palette)
templates/step10.html   Step 10 revisit wrapper around index.html
static/                 carl.png only — all CSS/JS is inline in index.html
sessions/               Generated session output directories
refs/                   Saved /suggest problem specs (markdown)
docs/                   This reference, Design-README.md, launch_commands.md,
                        rl-skills-index.md (index of jit_skills/)
```

---

## bedrock_converse.py

Shared Bedrock Converse API helper. All LLM calls in `web.py`, `chat.py`,
`literature.py`, and `report.py` route through this module. It centralizes
`client.converse` / `client.converse_stream` request construction, including
`cachePoint` placement, tool schema conversion, message conversion, usage
normalization, per-step extended-thinking configuration, and model-specific
inference options.

### Exports

| Name | Kind | Purpose |
|------|------|---------|
| `converse_once` | function | Non-streaming call; returns raw Converse response dict |
| `converse_text` | function | Convenience wrapper around `converse_once`; returns extracted text |
| `extract_text_from_converse` | function | Pull concatenated text from a Converse response dict |
| `converse_stream` | function | Streaming call; yields normalized events (see below) |
| `anthropic_tool_to_converse` | function | Convert one native tool spec to a Converse `toolSpec` entry |
| `anthropic_tools_to_converse` | function | Convert a list of native tool specs |
| `messages_to_converse` | function | Convert CARL's in-memory native-shape messages to Converse shape |

### `converse_once` / `converse_text`

```python
converse_once(client, model, system, messages, *, tools=None, max_tokens,
              temperature=None, cache_system=False, cache_tools=False,
              cache_ttl="1h") -> dict

converse_text(client, model, system, messages, *, ...) -> str
```

`converse_text` is `converse_once` + `extract_text_from_converse` in one call.
Most one-off non-streaming helpers use the default `cache_system=False`.
Step summaries in `chat.py` and `web.py` set `cache_system=True` to reuse the
still-warm step prompt cache; `web.py` also forwards the current tool list when
summarizing steps that contain tool-use blocks.

### `converse_stream`

```python
converse_stream(client, model, system, messages, *, tools=None, max_tokens,
                temperature=None, cache_system=True, cache_tools=True,
                cache_ttl="1h") -> Iterator[tuple[str, Any]]
```

Yields normalized events consumed by `_stream_single_turn` callers in
`web.py` and `chat.py`:

| Yield | Meaning |
|-------|---------|
| `("text", str)` | One text delta |
| `("tool_start", name)` | A tool-use block just started |
| `("usage", dict)` | Token usage with **Anthropic-style keys** (see below) |
| `("result", (blocks, stop_reason))` | Final content blocks in CARL's native shape + stop reason |

Usage keys are normalized from Converse names to Anthropic names so existing
token-log formatting in callers is unchanged:

| Converse key | Normalized to |
|--------------|---------------|
| `inputTokens` | `input_tokens` |
| `outputTokens` | `output_tokens` |
| `cacheWriteInputTokens` | `cache_creation_input_tokens` |
| `cacheReadInputTokens` | `cache_read_input_tokens` |

### CachePoint placement

`cachePoint` markers are inserted at the API boundary only; the in-memory
message list stays in CARL's native shape.

**System block** (when `cache_system=True`):
```python
[{"text": system_prompt_string}, {"cachePoint": {"type": "default", "ttl": "1h"}}]
```
Stable within a step because `build_system_prompt(step, skill_content,
previous, skip_steps)` is called once per step and is byte-identical across
all turns of that step.

**Tool config** (when `cache_tools=True`):
```python
{"tools": [*toolSpec_entries, {"cachePoint": {"type": "default", "ttl": "1h"}}]}
```
Stable because the tool list (`TOOLS` for steps 8–9, `[LITERATURE_TOOL]` for
2.5, `REVISIT_TOOLS` for Step 10, absent otherwise) never changes within a step.

TTL is `"1h"` for all steps. Bedrock supports only `"5m"` or `"1h"`;
`"1h"` is the minimum floor that covers both the ~30-min interactive window
(steps 1–7) and the ~90-min training runs in step 9.

**Minimum token threshold.** No cache entry is written unless the cacheable
prefix exceeds a per-model minimum. The minimum is **not** monotonic across
generations, and it applies on every platform the model is available on —
there is no Bedrock exception:

| Model | Role in CARL | Minimum cacheable prefix |
|---|---|---|
| Opus 5 | `DEFAULT_MODEL` | **512** tokens |
| Sonnet 5 | reports, persona, fallback | **1024** tokens |
| Haiku 4.5 | Trillian | **4096** tokens |
| *(Opus 4.7)* | *former default* | *2048 tokens* |

Two consequences. Opus 5's 512-token floor is a quarter of Opus 4.7's, so early
steps that previously fell below the threshold should now produce cache writes.
Conversely the Trillian path needs a 4096-token prefix (persona plus
`_TRILLIAN_SYSTEM_TEMPLATE`) before its `cache_ttl="1h"` does anything at all.

A below-minimum prefix **fails silently** — the request succeeds and reports
`cacheWriteInputTokens: 0` rather than erroring. So `cached=0, read=0` in a log
line means "prefix was under the floor", which is expected on short prompts and
is the only symptom you will get.

### Message shape conversion

`messages_to_converse` converts the in-memory list at the API boundary:

| Native shape | Converse shape |
|---|---|
| `{"role": "user", "content": "hello"}` | `{"role": "user", "content": [{"text": "hello"}]}` |
| `{"type": "tool_use", "id": X, "name": Y, "input": Z}` | `{"toolUse": {"toolUseId": X, "name": Y, "input": Z}}` |
| `{"type": "tool_result", "tool_use_id": X, "content": str}` | `{"toolResult": {"toolUseId": X, "content": [{"text": str}]}}` |
| `{"type": "image", "source": {"type": "base64", ...}}` | `{"image": {"format": "png", "source": {"bytes": raw_bytes}}}` (decoded on the fly) |
| `{"type": "image_raw", "format": "png", "data": bytes}` | `{"image": {"format": "png", "source": {"bytes": bytes}}}` |

The native shape is never mutated; `_sanitize_messages`, `_extract_text`,
auto-save, `msg_origins`, and conversation-file format all remain unchanged.

### Model quirk: temperature deprecation

`_accepts_temperature(model) -> bool` returns False for the Claude 5 family
(`_is_claude_5`, matching `claude-{opus,sonnet,fable,mythos}-5`) and for
Opus 4.7, both of which reject an explicit `temperature`.
`_build_inference_config` omits `temperature` from `inferenceConfig` when this
returns False, and also whenever extended thinking is enabled for the call.
Temperature handling for `chat.py`, `web.py`, `report.py`, and `literature.py`
is centralized here — the ~40 call sites still pass `temperature=`, so pinning
an older model via `CARL_MODEL` keeps working.

`DEFAULT_TRILLIAN_MODEL` (Haiku 4.5) is the only shipped default that still
accepts `temperature`, so this branch is not dead code.

### Provider portability

The core request and response conversions work with models from any provider
that supports the Bedrock Converse API. `_is_anthropic_model(model)` gates
CARL's provider-specific extensions: prompt-cache points and extended-thinking
fields are sent only to Anthropic model IDs. Other Converse providers receive
the portable system, messages, tools, and inference configuration instead.
Their support for tool use and image input remains model-specific.

### Model quirk: adaptive-thinking token floor

`_ADAPTIVE_MIN_MAX_TOKENS = 256`. Claude 5 thinks adaptively even with no
thinking config attached, and those reasoning tokens are drawn from
`maxTokens` — so a very small budget is spent reasoning before any text is
emitted and the call returns an empty string rather than an error. Measured at
`maxTokens=30` (the session-slug helpers, `web.py:739` / `chat.py:1957`), which
came back empty 3 times in 5. `_build_inference_config` raises `maxTokens` to
the floor for `_uses_adaptive_thinking` models and never lowers it. Fixed here
rather than at the call sites, for the same reason as the temperature gate.

### Extended thinking

Per-step extended thinking is configured here and consumed by the
`_stream_single_turn` callers in `chat.py` / `web.py`.

| Function | Purpose |
|----------|---------|
| `thinking_tier_for_step(step_number, ceiling=None)` | Tier for a step, clamped to at most `ceiling` (`_TIER_ORDER` = medium < high < extra_high). Never promotes a lower tier |
| `thinking_config_for_step(model, step_number, ceiling=None)` | Returns `additionalModelRequestFields` for the (possibly clamped) tier, or `None` when the step has no tier |
| `thinking_max_tokens_for_step(step_number, base, ceiling=None)` | Raises `max_tokens` to fit the thinking budget plus output. Pass the same `ceiling` used for the config so the allowance matches the requested effort |

`_STEP_THINKING_TIER` maps step → tier: step 1 = `medium`; steps 2, 2.5, 3–7,
and 10 = `high`; steps 8–9 = `extra_high`.

Both engines pass `ceiling=_TEXT_TURN_TIER_CEILING` (`"medium"`) on turns with
no tools, because reasoning tokens are drawn from `maxTokens`. Measured on a
Step 3 prompt: `high` @ 16384 spends ~15.9k output tokens for a 9.5k-character
reply; `medium` @ 8192 spends ~6.2k for 7.3k characters, both finishing on
`end_turn`. Tool turns keep the step's own tier, where the extra reasoning is
planning real work.

- **Claude 5, Opus 4.7, Sonnet 4.6** (`_uses_adaptive_thinking`):
  `{"thinking": {"type": "adaptive"}, "output_config": {"effort": <effort>}}`
  where `_EFFORT_MAP` maps `medium`/`high`/`extra_high` →
  `medium`/`high`/`max`. These models reject `thinking.type=enabled` outright.
- **Older models** — in practice only Haiku 4.5:
  `{"thinking": {"type": "enabled", "budget_tokens": <n>}}` where
  `_THINKING_BUDGET` = `{medium: 4000, high: 8000, extra_high: 16000}`.
  Haiku 4.5 errors on `effort`, so it must stay on this branch; getting the
  predicate wrong 400s the Trillian suggestion path while the main workflow
  looks healthy, and a failed suggestion just means no suggestion appears.

`_THINKING_MAX_TOKENS` = `{medium: 8192, high: 16384, extra_high: 32000}`.
If a model rejects the thinking config, `chat.py._stream_single_turn` detects
the error via `_is_thinking_error` and retries the turn without thinking.

---

## bedrock_models.py

Model-ID defaults and startup model resolution, shared by `chat.py` and
`web.py`. The four model constants moved here out of `chat.py`; `chat.py` now
imports them (aliased with `_` prefixes), and `_FALLBACK_MODEL` is no longer a
`chat.py` constant.

### Constants

| Name | Value |
|------|-------|
| `DEFAULT_MODEL` | `"us.anthropic.claude-opus-5"` |
| `FALLBACK_MODEL` | `"us.anthropic.claude-sonnet-5"` |
| `DEFAULT_REPORT_MODEL` | `"us.anthropic.claude-sonnet-5"` |
| `DEFAULT_TRILLIAN_MODEL` | `"us.anthropic.claude-haiku-4-5-20251001-v1:0"` |
| `DEFAULT_PERSONA_UPDATE_MODEL` | `"us.anthropic.claude-sonnet-5"` |
| `DEFAULT_REGION` | `"us-west-2"` |

Three models, four roles: **Opus 5** main (Sonnet 5 fallback), **Sonnet 5**
reports and persona update, **Haiku 4.5** Trillian. Each is independently
repinnable via `CARL_MODEL` / `CARL_REPORT_MODEL` /
`CARL_PERSONA_UPDATE_MODEL` / `CARL_TRILLIAN_MODEL`.

**ID shapes differ by family and must not be cross-applied.** Claude 5 IDs are
unsuffixed — `us.anthropic.claude-opus-5-v1:0` is a hard
`ValidationException`. The 4.x IDs are the reverse: only the suffixed
`us.anthropic.claude-haiku-4-5-20251001-v1:0` resolves, while
`us.anthropic.claude-haiku-4-5` and `anthropic.claude-haiku-4-5` both fail.
Haiku 4.5 is also the newest Haiku there is (the Claude 5 family is
Fable / Mythos / Opus / Sonnet only), so `DEFAULT_TRILLIAN_MODEL` needs no
version bump.

### `resolve_model_id(client, preferred=None, *, log=None) -> str`

`preferred` defaults to `CARL_MODEL` env or `DEFAULT_MODEL`. If `preferred !=
DEFAULT_MODEL` (user explicitly set a model), returns it immediately with no
probe. Otherwise probes `DEFAULT_MODEL` with a tiny 5-token `converse` call
(`_probe_model`), then:

- **Credential errors** (`NoCredentialsError` / `PartialCredentialsError` /
  `ProfileNotFound`) → raise `RuntimeError` with a credentials-help message.
- **`BotoCoreError`** (connection) → raise `RuntimeError`.
- **`ClientError`** (model unavailable) → log a notice and try the fallback.

Unlike the previous behavior, the fallback is **probed too**: `FALLBACK_MODEL`
gets its own 5-token probe, and if that also fails, `resolve_model_id` raises
`RuntimeError` rather than blindly returning an unusable model. `chat.py`'s
`_resolve_model_id` and `web.py`'s `_resolve_model` are thin wrappers that pass
a logging callback.

---

## resume.py

Session resume helpers, imported by `carl.py`, `chat.py`, and `web.py`. Resumes
an existing normal session from its last unfinished step; mutually exclusive
with `--revisit`.

### Exports

| Name | Kind | Purpose |
|------|------|---------|
| `ResumeError` | exception | Raised when a session cannot be resumed safely |
| `ResumeInfo` | dataclass | `session_dir`, `step_idx`, `step_number`, `step_name`, `step_decisions`, `suggest_used`, `user_skip_25`, `complete`, `warnings` |
| `build_resume_info(carl_dir, session_dir) -> ResumeInfo` | function | Entry point — resolves the session, parses the log + memory, returns resume target |
| `resolve_resume_session(carl_dir, session_dir) -> Path` | function | Validate path is under `sessions/` and requires both `memory.md` and `web.log` |
| `parse_memory_decisions(memory_text) -> dict` | function | Re-parse completed step summaries from `memory.md`. A repeated `## Step N:` heading has its bodies **concatenated in document order** (blank-line separated), not overwritten — `Session.save_step` writes its own heading and CARL has additionally appended notes with `cat >> ../memory.md` during Steps 8–9, so keeping only the last block silently dropped the controller's own summary |
| `emit_resume_marker(event, step_number=None)` | function | Print a `CARL_RESUME` marker line (flushed) |
| `format_resume_marker(event, step_number=None) -> str` | function | Build the marker string |
| `SessionLogTee` / `_TeeWriter` | classes | Mirror stdout/stderr into a session-local `web.log`, ANSI-stripped |

### Resume markers

`MARKER_PREFIX = "CARL_RESUME"`. Markers written to `web.log` as CARL runs:

- `CARL_RESUME step_start number=N`
- `CARL_RESUME step_saved number=N`
- `CARL_RESUME session_done`

### Detection logic

`build_resume_info` reads `memory.md` + `web.log`, then:

1. Infers skip-2.5 state — `_looks_like_skip_25` (memory/log mention Step 2.5 +
   "skip") and `_looks_like_suggest_used` (`[/suggest]` in log).
2. `_parse_structured_markers` — if `CARL_RESUME` markers exist, tracks the last
   `step_start`, the set of `step_saved` numbers, and `session_done`; returns
   the last started-but-unsaved step, else the first missing workflow step.
3. `_parse_legacy_log` (fallback when no markers) — scans for `## Step N:`
   headings and "Ready to proceed to Step N", plus interruption markers
   (`[Interrupted`, `API error`, `LLM call failed`, `Server error`,
   `Traceback`) and completion markers (`[DONE]`, `All steps complete!`).
4. Raises `ResumeError` if the session appears complete ("use --revisit").
5. Restores `step_decisions` for steps *before* the resume target; if `memory.md`
   holds summaries for the resumed/later steps, appends a warning that `web.log`
   is the resume authority.

`SessionLogTee.start/stop/restart` manage the tee (used by `chat.py` for all
normal CLI runs, restarted around the session rename).

---

## steps.py

### STEPS list

The dict key is `skill` and its value is a **skill directory name**, resolved to
`carl_skills/<skill>/SKILL.md`:

```python
STEPS = [
    {"number": 1,   "name": "Problem Definition",    "skill": "01-problem-definition"},
    {"number": 2,   "name": "References",             "skill": "02-references"},
    {"number": 2.5, "name": "Literature Search",      "skill": "025-literature-search"},
    {"number": 3,   "name": "Parameter Exploration",  "skill": "03-parameter-exploration"},
    {"number": 4,   "name": "Information Proposal",   "skill": "04-information-proposal"},
    {"number": 5,   "name": "Action Definition",      "skill": "05-action-definition"},
    {"number": 6,   "name": "Reward Definition",      "skill": "06-reward-definition"},
    {"number": 7,   "name": "ML/RL Parameters",       "skill": "07-ml-parameters"},
    {"number": 8,   "name": "Initial Testing",        "skill": "08-initial-testing"},
    {"number": 9,   "name": "Launch",                 "skill": "09-launch"},
]

REVISIT_STEP = {
    "number": 10,
    "name": "Post-Training Analysis",
    "skill": "10-post-training-analysis",
}
```

`REVISIT_STEP` is separate from `STEPS`. Normal CARL still ends after Step 9;
Step 10 is launched only through `--revisit`.

### Constants

| Name | Value |
|------|-------|
| `_TRILLIAN_LAST_ACTIVE_STEP` | `6` |
| `_FRONTMATTER_RE` | `re.compile(r"\A---\r?\n.*?\r?\n---[ \t]*\r?\n?", re.DOTALL)` — matches one leading YAML frontmatter block |
| `_LEGACY_SKILL_FILES` | skill dir → pre-standard flat `NN_name.md` filename, for all 11 steps |
| `_SKILL_DIR_BY_LEGACY_FILE` | the inverse mapping, so either form may be passed to `load_skill` |
| `TRILLIAN_PERSONA_PATH` | `Path("trillian_skills") / "trillian-persona" / "SKILL.md"` |
| `_LEGACY_TRILLIAN_PERSONA_PATH` | `Path("trillian_skills") / "Trillian-Persona.md"` |

### Functions

| Signature |
|-----------|
| `strip_frontmatter(text: str) -> str` — drop one leading `---`-delimited YAML block and the blank lines after it |
| `load_skill(carl_dir: str, skill: str) -> str` — read `carl_skills/<skill>/SKILL.md` (utf-8-sig) and return `strip_frontmatter(...)` of it |
| `trillian_persona_path(carl_dir: str \| Path) -> Path` — resolve `TRILLIAN_PERSONA_PATH`, falling back to the flat legacy path |
| `load_trillian_persona(carl_dir: str) -> str` — reads `trillian_persona_path(carl_dir)`, i.e. `trillian_skills/trillian-persona/SKILL.md` |
| `build_trillian_prompt(persona: str, step_num: int, step_name: str) -> str` |
| `build_system_prompt(step: dict, skill_content: str, previous_decisions: str, skip_steps: set \| None = None, workflow_steps: list[dict] \| None = None, mode_note: str \| None = None) -> str` |

### Skill resolution and frontmatter

Only the skill **body** reaches the system prompt. `load_skill` calls
`strip_frontmatter`, so the `name` / `description` / `metadata` keys (tooling and
human metadata) never consume prompt tokens — the body already opens with its own
`# Step N:` heading. The body is re-sent on every tool round, so its size is a
real per-round cost.

Resolution order for `load_skill(carl_dir, skill)`:

1. `skill` is normalized through `_SKILL_DIR_BY_LEGACY_FILE` (and a trailing
   `.md` is stripped), so callers may pass either `"09-launch"` or the old
   `"09_launch.md"`.
2. `carl_skills/<key>/SKILL.md` — the canonical path.
3. `carl_skills/<skill>` or `carl_skills/<legacy NN_name.md>` — a pre-standard
   flat file. Used only as a fallback, and a `[steps] skill ...: falling back to
   pre-standard flat file ...` line is written to stderr so a half-migrated
   checkout still runs but is visible.
4. Otherwise `FileNotFoundError: Skill not found: <canonical>`.

`trillian_persona_path` applies the same two-step rule to the persona file and
logs `[steps] persona: ... falling back to ...` when it uses the legacy path. It
returns the canonical path (even if missing) when neither exists, so the caller
raises on the standard path. Because the persona backup is written as
`persona_path.with_suffix(".md.bak")`, the backup is now
`trillian_skills/trillian-persona/SKILL.md.bak`.

### `build_system_prompt` sections injected

1. **Always**: CARL role, current step, skill content (this step only, frontmatter
   already stripped), previous decisions, guidelines (no sycophancy, ≤3 questions,
   always end with question).
2. **Step 2**: file-reading instructions, extract analytical solutions.
3. **Step 2.5**: `search_literature(query, limit)` tool instructions.
4. **Step 7**: one-turn format — present hyperparameter table, no follow-up, and
   emit the machine-readable budget line `TRAINING_BUDGET_MINUTES: 45` on its own
   line (integer minutes, keep under 90, written even when the user accepted the
   default). `tool_runtime.parse_training_budget_seconds` parses it.
5. **Step 10**: revisit-mode instructions. Interactive; inspect artifacts, prepare
   `code/revisit_N/`, do not run retraining, do not use `run_background`, call
   `describe_plot` on **every** plot returned by `list_revisit_artifacts` (the
   plots list is a worklist, not a sample; a short or empty inventory is itself a
   finding), update `results/appendix.tex`, and ask exactly "Ready to finish
   Step 10?" rather than outputting `[DONE]`.
6. **Steps 8–9**: four tools available, chain tool calls, baselines <60 s. The
   `CRITICAL -- TOOL CALL DISCIPLINE` block additionally requires: edit files
   **only** by rewriting them with `write_file` (no `sed -i`, no `python -c`
   heredocs, no `cat > file << EOF`, no `>>` appends to source, because the saved
   conversation record truncates tool input); never write to `memory.md` (the
   controller owns it and a second copy produces duplicate `## Step N` blocks);
   and `run_command` / `run_background` always start in the session's `code/`
   directory, so paths are relative to it and never prefixed with `code/` or
   preceded by a `cd`.
7. **Step 9 only**: write all code first, use `run_background` for training,
   flush/checkpoint, report metrics per phase; after training, write
   `make_artifact.py`, run it to build the self-contained `artifact/` bundle
   (copied checkpoint, `contract.json`, `solve.py`, `smoke_test.py`,
   `MODEL_CARD.md`), then run `artifact/smoke_test.py` with `run_command` — a
   failing smoke test blocks shipping. The FINAL SUMMARY must include the
   `artifact/` path and the smoke-test result, then `[DONE]` on its own line,
   no questions.

### `_TRILLIAN_SYSTEM_TEMPLATE`

Substitution fields: `{persona}`, `{step_num}`, `{step_name}`. Instructs Trillian to output
only the suggested reply, to respond `[SKIP]` on the very first Step 1 greeting, and `ok` on
"Ready to proceed?" confirmations.

---

## tool_runtime.py

Shared tool-execution logic for CARL's two engines. `web.py` and `chat.py` are
near-duplicate controllers around the same tool loop; anything here is the single
definition both import, so a behavioural fix lands once instead of twice. It sits
alongside the other two shared helpers, `command_compat.py` (the `tail` shim and
`_resolve_code_path`) and `conversation_guards.py` (end-of-turn question guards).

Both engines import from it and keep `_`-prefixed aliases so existing call sites
read as before: `_MAX_TOOL_ROUNDS`, `_STEP9_DEFAULT_BUDGET_SECONDS`,
`_CHECK_PROCESS_MIN_WAIT`, `_SUGGEST_SYNTHESIS_SYSTEM`, `_SUGGEST_SYNTHESIS_USER`,
`_SUGGEST_DOCUMENT_SYSTEM`, `_SUGGEST_DOCUMENT_USER`. `chat.py` also keeps
`CARLChat._is_revisit_training_command = staticmethod(is_revisit_training_command)`.

### Constants

| Name | Value | Purpose |
|------|-------|---------|
| `MAX_TOOL_ROUNDS` | `80` | Max iterations of the tool-use loop per response |
| `STEP9_DEFAULT_BUDGET_SECONDS` | `45 * 60` | Training budget when Step 7 named none |
| `CHECK_PROCESS_MIN_WAIT` | `60` | Never poll `check_process` more than once a minute |
| `TRAINING_BUDGET_MARKER` | `"TRAINING_BUDGET_MINUTES:"` | The machine-readable Step 7 line |
| `READ_ONLY_VERBS` | `cat`, `head`, `tail`, `grep`, `ls`, `find`, `chmod`, `git`, … | Documentary: verbs that only read or set metadata |
| `CODE_DIR_CONTRACT` | prose fragment | Spliced into the `run_command` / `run_background` tool descriptions |

### Exports

| Name | Kind | Purpose |
|------|------|---------|
| `TOOLS` | list | The four Step 8–9 tool schemas (`write_file`, `run_command`, `run_background`, `check_process`) |
| `LITERATURE_TOOL` | dict | The single Step 2.5 `search_literature` schema |
| `SUGGEST_SYNTHESIS_SYSTEM` / `SUGGEST_SYNTHESIS_USER` | str | `/suggest` phase-1 prompt templates (3–5 candidate formulations) |
| `SUGGEST_DOCUMENT_SYSTEM` / `SUGGEST_DOCUMENT_USER` | str | `/suggest` phase-2 prompt templates (full problem spec) |
| `is_revisit_training_command(command) -> bool` | function | Whether a command would start a training run (refused in Step 10) |
| `revisit_training_refusal(command) -> str` | function | Refusal text naming the allowed alternatives |
| `parse_training_budget_seconds(step7_summary) -> tuple[int, str]` | function | `(seconds, source)` where source is `"marker"`, `"regex"`, or `"default"` |
| `check_interval_seconds(step7_summary, log=None, max_rounds=MAX_TOOL_ROUNDS, min_wait=CHECK_PROCESS_MIN_WAIT) -> int` | function | Seconds `check_process` waits, spreading polls over the budget |
| `normalize_code_relpath(rel_path) -> str` | function | Strip a redundant leading `code/` from a path already relative to `code/` |

### Step 10 training refusal

The refusal is **argv[0]-based**, not substring-based. The first version matched
any substring of the command line, which refused read-only work that merely
mentioned a script name — `grep train.py train.log`, `chmod +x revisit_train.sh`,
`ls long_train.sh` (19 such refusals across three observed sessions). The current
check tokenises with `shlex.shlex(command, posix=True, punctuation_chars=True)`,
splits the token stream into pipeline/list segments on the operator characters
`&|;()<>` (`_split_segments`), and asks whether the *executed program* of any
segment is a trainer (`_segment_starts_training`).

Per segment, after `_strip_leading_env` drops leading `VAR=value` assignments:

| `argv[0]` basename | Decision |
|--------------------|----------|
| `_TRAINING_SCRIPTS` = `revisit_train.sh`, `long_train.sh`, `run.sh`, `train.sh` | training |
| `_SHELLS` = `bash`, `sh`, `zsh`, `dash`, `ksh`, `shell` | `-n` (syntax check) → allowed; `-c` → recurse on the payload; else training iff a non-flag argument is a training script |
| `_PYTHONS` = `python`, `python3`, `python2`, `py`, `ipython` | `_python_runs_trainer`: `-m` module in `_TRAINING_MODULES` (`train`, `long_train`); `-c` payload judged by `_code_runs_trainer`; else the first non-flag argument must be in `_TRAINING_ENTRY_FILES` (`train.py`, `long_train.py`) ∪ `_TRAINING_SCRIPTS`; bare REPL → allowed |
| `_WRAPPERS` = `nohup`, `time`, `env`, `timeout`, `stdbuf`, `setsid`, `sudo`, `nice`, `xargs`, `unbuffer`, `screen`, `tmux` | unwrap (dropping flags and bare duration/priority arguments) and recurse, `depth > 3` → allowed |
| anything else | allowed |

`_code_runs_trainer` requires a `python -c` payload to actually *run* the trainer:
an `import`/`from` of `train` / `long_train`, a `train.main|train|run(` call, or an
`exec` / `eval` / `runpy.run_path` / `os.system` / `os.exec*` / `subprocess.*` /
`popen` pointed at a training file. Merely naming `train.py` in a payload that
only reads it is allowed.

If the line cannot be tokenised (unbalanced quotes, `shlex` raises `ValueError`),
`_split_segments` returns `None` and the check falls back to the conservative
`_LEGACY_FORBIDDEN_SUBSTRINGS` scan.

### Step 7 training budget

`parse_training_budget_seconds` resolves the Step 9 budget in three tiers and
reports which one fired:

1. `"marker"` — `_MARKER_RE` (`TRAINING_BUDGET_MINUTES\s*:\s*(\d+(?:\.\d+)?)`,
   case-insensitive) found a positive value in the Step 7 summary.
2. `"regex"` — free-text fallback: `_STEP9_TIME_RE`
   (`[Ss]tep\s*9...(\d+)\s*(hour|hr|minute|min)`) then `_ANY_TIME_RE`.
3. `"default"` — `STEP9_DEFAULT_BUDGET_SECONDS` (45 min).

`check_interval_seconds` returns
`max(budget_seconds // (max_rounds // 2), min_wait)` and **always** reports the
resolution path — through the `log` callable when given, otherwise to stderr — as
`[check_process interval] training budget N min from <source>`, with
`-- no TRAINING_BUDGET_MINUTES: line in the Step 7 summary` appended when the
default fired, so a silent fall back to the default is visible.
`chat.py._get_step9_check_interval` passes a cyan-print callback,
`web.py._get_step9_check_interval` passes `lambda msg: _log(msg, "info")`; both
pass `session.step_decisions.get(7, "")`.

### Path and tool-description contract

Both engines run commands with cwd set to the session's `code/` directory.
Some generated commands redundantly change into `code/` or list `code/` from
*inside* that directory, so `CODE_DIR_CONTRACT` states the rule inside the `run_command` and
`run_background` schemas themselves, and `normalize_code_relpath` strips a leading
`code/` component from `write_file` paths (which would otherwise create
`code/code/train.py`). This mirrors `command_compat._resolve_code_path`, which
does the same for the shell shims. `write_file`'s description also states that it
is the only approved way to create or edit a file, and `run_command`'s adds
"Do not edit files here (no sed -i, no heredocs, no 'cat >')".

---

## session.py

### Session class

#### `__init__(self, base_dir: str)`

Creates `sessions/<YYYYMMDD-HHMM>/` with subdirs `references/`, `code/`, `results/`.
Writes initial `memory.md` header. Sets `step_decisions: dict[float, str] = {}`.

Key attributes: `base_dir`, `timestamp`, `session_dir`, `memory_path`, `step_decisions`.

#### Methods

| Signature | Notes |
|-----------|-------|
| `open_existing(base_dir, session_dir) -> Session` | **classmethod**. Open completed session for Step 10 without creating a new directory or overwriting `memory.md` |
| `_step_label(step_number) -> str` | **static**. `2.5` → `"02.5"`, `1` → `"01"` |
| `save_step(step_number, step_name, summary, conversation, msg_origins=None)` | Appends to `memory.md`; writes `step_NN_conversation.md`; stores in `step_decisions` |
| `_format_content(content) -> str` | **static**. str → passthrough; list → format text/tool_use/tool_result blocks; truncate at 300/500 chars |
| `_format_message(msg, origin) -> str` | Format one message for conversation log; expand Trillian origins into sub-sections |
| `save_conversation(step_number, step_name, conversation, msg_origins=None)` | Write conversation without summary (auto-save, overwrites) |
| `get_previous_decisions() -> str` | Format `step_decisions` into markdown for system prompt |
| `rename_with_topic(slug: str)` | Rename `sessions/TIMESTAMP` → `sessions/slug_TIMESTAMP` |
| `copy_reference(source_path: str) -> Path \| None` | Copy file to `references/` |
| `read_file(file_path: str) -> str \| None` | **static**. Read text file |

#### `memory.md` format

```markdown
# CARL Session -- 20260406-1030

## Step 1: Problem Definition

- bullet 1
- bullet 2

---

## Step 2: References
...

**Persona updated:** <explanation> (271 -> 269 lines)   ← appended on persona change
**Session ended during Step N: Name**                    ← appended on incomplete exit
```

#### `step_NN_conversation.md` format

```markdown
# Step N: Name -- Full Conversation

**CARL:**
<text>

**User:**
<text>

[Tool: write_file]
```
{input json}
```

[Tool Result]
```
result text (truncated at 500 chars)
```

**Trillian suggestion (accepted by user):**   ← when origin = trillian_accepted
<suggestion text>

**Trillian suggestion:**                       ← when origin = trillian_edited
<suggestion text>
**User (edited and sent to CARL):**
<final text>
```

---

## revisit.py

Step 10 revisit helper module. It is imported by both `chat.py` and `web.py`;
its tools and context builders are used when an existing session is opened with
`--revisit`.

### Tool schema

`REVISIT_TOOLS` defines the Step 10 tool set:

| Tool | Purpose |
|------|---------|
| `list_revisit_artifacts` | Return categorized artifacts from the existing session |
| `read_artifact(path, max_chars)` | Read capped text artifacts from inside the session |
| `describe_plot(path)` | Send a PNG plot to the vision model for concise diagnostic description |
| `prepare_revisit_workspace(source_dir)` | Create the next self-contained `code/revisit_N/` workspace from a user-confirmed session-local source directory |
| `write_file(path, content)` | Shared tool name; Step 10 callers restrict it to `code/revisit_N/` |
| `run_command(command)` | Shared tool name; Step 10 callers refuse retraining commands |

Step 10 has no `run_background` tool. Revisit training is prepared but not run
during Step 10.

### Session validation and context

| Function | Purpose |
|----------|---------|
| `resolve_revisit_session(carl_dir, session_dir)` | Resolve a user-supplied path under `CARL/sessions/` and require `memory.md` |
| `discover_artifacts(session_dir, excluded_paths=None)` | Find `memory.md`, Step 8/9 conversations, prior Step 10 conversations, `train.log` files, plots, `results/*.tex`, JSON metrics, long-run outputs, revisit artifacts, and plausible source directories for the next revisit |
| `build_revisit_context(session_dir)` | Build the Step 10 context block from the artifact inventory, capped `memory.md`, Step 8 conversation, and Step 9 conversation |
| `read_artifact(session_dir, rel_path, max_chars)` | Read only text-like suffixes from inside the session |
| `describe_plot(client, model, session_dir, rel_path)` | Pass raw PNG bytes through Converse vision with memory context |

`discover_artifacts` returns the keys `core`, `train_logs`, `plots`,
`results_tex`, `json_results`, `long_run`, `source_candidates`.

### Plot inventory (shared with report.py)

`revisit.py` imports `from report import find_session_plots, is_checkpoint_dupe`
and fills `files["plots"]` from **a dedicated PNG pass** rather than from
`_iter_included_files`. `_SKIP_DIRS` prunes `experiments/`, `logs/`,
`checkpoints/`, and `saved_models/` (correct for source-file discovery), but plots
are usually written under `code/experiments/<run>/` — so walking only the
source-file iterator produced an inventory that was missing most plots. Since
Step 10's `## Visual Evidence Review` treats the `plots` list as a *worklist*
(every entry must get a `describe_plot` call), a plot missing here is a plot CARL
never looks at.

Routing plots through `report.find_session_plots` means the Step 10 inventory and
the PDF report see one identical set. `is_checkpoint_dupe` (the `-checkpoint`
stem test) filters `.ipynb_checkpoints/` autosave duplicates, and is applied to
`results_tex` and `json_results` as well as to plots.

### Workspace creation

`prepare_revisit_workspace(session_dir, source_dir)` scans existing
`code/revisit_N/` directories, picks the next number, validates that
`source_dir` resolves inside the active session, copies that user-confirmed
source into the new directory, and refuses to overwrite an existing revisit.
The confirmed source may be an earlier `code/revisit_M/` workspace. The copy
excludes generated outputs and heavyweight/transient files: `experiments/`,
`logs/`, `checkpoints/`, `saved_models/`, cache directories, nested
`revisit_N/` directories, `.pt/.pth/.ckpt/.pkl/.pyc/.log` files,
`.bg_pids.json`, and TensorBoard event files.

The new workspace receives:

- `revisit_train.sh` -- a fresh-start script. It does not pass `--init-from`,
  refuses `TRAIN_CMD` overrides containing `--init-from`, writes logs under
  `experiments/revisit_N/`, runs the appendix helper after training, and finally
  refreshes the shippable bundle with
  `python make_artifact.py --run-name "$RUN_NAME"` — guarded by
  `if [[ -f make_artifact.py ]]` so a workspace copied from a session predating
  the artifact bundle still runs.
- `revisit_appendix.py` -- writes/updates `results/appendix.tex`, inserts
  `\input{appendix.tex}` into `results/main.tex` once when present, and runs
  `pdflatex` twice if available. On LaTeX failure it keeps the `.tex` files
  and prints the failure. It locates and calls
  `revisit_report.write_revisit_appendix`.

---

## revisit_report.py

Report-style appendix publishing for Step 10 revisit runs, called by the
`revisit_appendix.py` helper that `prepare_revisit_workspace` drops into each
`code/revisit_N/`. Its analysis is generated from structured run artifacts
(`*.json` under the run directory) rather than scraped from terminal output;
`read_json(path)` and `tail(path, n=20)` are the file readers, and
`write_revisit_appendix(...)` is the entry point.

### `_artifact_subsection(script_dir, session_dir) -> list[str]`

Adds a `\subsection*{Shippable Artifact}` block to the appendix when
`script_dir / "artifact"` exists, and returns `[]` otherwise, so workspaces that
predate the bundle publish unchanged. `revisit_train.sh` refreshes `artifact/`
via `make_artifact.py` at the end of the run, so the appendix can tell a reader
how to run the revised policy. The block states the bundle's session-relative
path, that the checkpoint is copied so it runs outside the session, and the
`python solve.py --instance instance.json` invocation. From `contract.json` it
appends a `Recorded contract:` sentence naming whichever of `obs_dim`,
`action_dim`, `selection_metric` (or `metric`), and `seed` are present; from
`smoke_test_result.json` it reports whether the bundled smoke test `passed`. It
is inserted after the revisit figures and before the `Audit Trail` verbatim log
tail.

---

## chat.py

### Module-level constants

The four model-ID constants and `_DEFAULT_REGION` are **imported from
`bedrock_models.py`** (aliased with `_` prefixes) rather than defined here:
`_DEFAULT_MODEL`, `_DEFAULT_TRILLIAN_MODEL`, `_DEFAULT_PERSONA_UPDATE_MODEL`,
`_DEFAULT_REGION`. `_FALLBACK_MODEL` is no longer a `chat.py` constant
(`FALLBACK_MODEL` lives only in `bedrock_models.py`). Values are unchanged — see
the [bedrock_models.py](#bedrock_modelspy) table.

Workflow constants defined in `chat.py`:

| Name | Value |
|------|-------|
| `_TOOL_STEPS` | `{8, 9}` |
| `_ONE_TURN_STEPS` | `{7}` |
| `_LITERATURE_STEPS` | `{2.5}` — Step 2.5 for normal literature search (`/suggest` in Step 1 uses direct calls) |
| `_MAX_TOOL_ROUND_TAIL_CONTINUES` | `5` — extra turns allowed past the tool-round cap to finish a pending tool/continue turn |
| `_MAX_AUTO_ROUNDS` | `20` — max `[continue]` injections in autonomous steps 8–9 |

Constants **imported from `tool_runtime.py`** and aliased (identical in `web.py`;
see the [tool_runtime.py](#tool_runtimepy) table for the definitions):

| Alias | Source | Value |
|-------|--------|-------|
| `_MAX_TOOL_ROUNDS` | `MAX_TOOL_ROUNDS` | `80` — max iterations in `_stream_response` tool-use loop |
| `_STEP9_DEFAULT_BUDGET_SECONDS` | `STEP9_DEFAULT_BUDGET_SECONDS` | `2700` (45 min) — default training budget for `check_process` interval |
| `_CHECK_PROCESS_MIN_WAIT` | `CHECK_PROCESS_MIN_WAIT` | `60` — minimum seconds between `check_process` polls |

### Tool definitions

`TOOLS` list (four tools for Steps 8–9) and `LITERATURE_TOOL` dict (one
literature tool for Step 2.5) are **imported from `tool_runtime.py`**, shared
byte-for-byte with `web.py`.
`REVISIT_TOOLS` is imported from `revisit.py` and used only for Step 10
revisit mode.

| Tool | Key inputs | Timeout / limit |
|------|-----------|-----------------|
| `write_file` | `path: str`, `content: str` | — creates parents automatically; path passed through `normalize_code_relpath` |
| `run_command` | `command: str` | 600 s; output capped at 15,000 chars |
| `run_background` | `command: str`, `log_file: str` | returns PID immediately; `start_new_session=True` |
| `check_process` | `log_file: str`, `tail_lines: int=50` | waits `_get_step9_check_interval()` seconds if process still running; minimum 60 s |
| `search_literature` | `query: str`, `limit: int=10` (max 20) | calls `literature.search_papers()`; saves results via `save_literature_results(bedrock_client, summary_model)` |
| Step 10 revisit tools | see `revisit.py` section | artifact inspection, plot description, workspace creation, restricted file writes, short validation commands |

PIDs tracked in `code/.bg_pids.json` as `{log_file: pid}`.
Step 10 does not expose `run_background`; `run_command` refuses training commands
in revisit mode via `tool_runtime.is_revisit_training_command` /
`revisit_training_refusal` (argv[0]-based, so read-only commands that merely name
a training script are allowed).

### /suggest prompt templates

Imported from `tool_runtime.py` and aliased; `web.py` uses the same four.

| Alias | Used for |
|----------|---------|
| `_SUGGEST_SYNTHESIS_SYSTEM` | Synthesize 3–5 RL formulations from papers |
| `_SUGGEST_SYNTHESIS_USER` | Placeholders: `{problem_description}`, `{papers_formatted}` |
| `_SUGGEST_DOCUMENT_SYSTEM` | Generate full problem spec document |
| `_SUGGEST_DOCUMENT_USER` | Placeholders: `{chosen_formulation}`, `{problem_description}`, `{papers_formatted}` |

### `_resolve_model_id(client, preferred: str) -> str`

Thin wrapper that delegates to `bedrock_models.resolve_model_id(client,
preferred, log=...)`, passing a logging callback. The probe/fallback logic
(including probing the fallback and raising on credential/connection errors)
lives in [bedrock_models.py](#bedrock_modelspy).

### `_build_bedrock_client(region: str)`

Removes `AWS_PROFILE`/`AWS_DEFAULT_PROFILE` from env before creating client (avoid profile
conflicts with instance credentials), restores them after. Config: `read_timeout=900`,
`connect_timeout=60`, `max_attempts=3`.

> **Note:** `web.py` has its own copy of `_build_bedrock_client`. Timeouts
> match `chat.py` — `read_timeout=900`, `connect_timeout=60` — because autonomous
> Steps 8–9 use `max_tokens=16384`, and Opus 5 regularly takes several minutes to
> finish a tool turn at that length. The shorter 120 s read timeout used previously
> fired mid-generation on the first tool turn of Step 8.

### CARLChat class

#### `__init__(self, carl_dir, write_report=False, no_persona_update=False, revisit_dir=None, resume_dir=None)`

Raises `ValueError` if both `revisit_dir` and `resume_dir` are given.

| Attribute | Source |
|-----------|--------|
| `carl_dir` | arg |
| `write_report` | arg |
| `no_persona_update` | arg — skip post-session persona evolution |
| `revisit_dir` | arg — existing session path for Step 10 revisit mode |
| `revisit_mode` | `True` when `revisit_dir` is set |
| `_revisit_completed` | `False` until Step 10 is saved and closed |
| `resume_dir` | arg — existing session path for resume mode |
| `resume_info` | `build_resume_info(carl_dir, resume_dir)` when resuming, else `None` |
| `resume_mode` | `True` when `resume_dir` is set |
| `region` | `AWS_REGION` env / default |
| `model` | `_resolve_model()` — probes Bedrock; falls back to 4.6 |
| `trillian_model` | `CARL_TRILLIAN_MODEL` env / default |
| `persona_update_model` | `CARL_PERSONA_UPDATE_MODEL` env / default |
| `client` | `_build_bedrock_client(region)` |
| `session` | `Session.open_existing(...)` in revisit **or** resume mode, otherwise `Session(carl_dir)` |
| `_session_log` | `SessionLogTee(session_dir)` for normal/resume runs; `None` in revisit mode |
| `current_step_idx` | `0`, or `resume_info.step_idx` when resuming |
| `_current_messages` | `[]` — kept in sync for crash recovery |
| `_current_system` | `""` — kept in sync for crash recovery |
| `_trillian_persona` | `load_trillian_persona(carl_dir)` |
| `_trillian_suggestion` | `None` — pre-fills readline buffer |
| `_current_msg_origins` | `{}` — `{msg_idx: origin_dict}` |
| `_suggest_used` | `False` — set True when /suggest completes |
| `_user_skip_25` | `False` — set True when `/skip` issued in Step 2 |

#### `run(self)`

Main event loop. Steps through STEPS. After Step 1: calls `_rename_session_dir()`. Skips Step
2.5 if `_suggest_used` or `_user_skip_25`. Warns on entry to tool steps. On clean completion:
calls `_update_trillian_persona()` (unless `no_persona_update`), then optionally
`generate_report()`, then `_print_done()`.
Wrapped in `try/except/finally` with `_save_on_exit()` in `finally`.

If `revisit_mode` is true, `run()` prints a revisit banner and runs only
`REVISIT_STEP`. It opens the existing session, saves Step 10 on exit, skips the
Trillian persona update and normal report generation, and returns after the
revisit is complete.

If `resume_mode` is true, `__init__` restores `session.step_decisions` from
`resume_info.step_decisions` and reconstructs `_suggest_used` / `_user_skip_25`,
and `run()` prints a resume banner (`_print_resume_banner`) and continues the
normal `STEPS` workflow from `current_step_idx = resume_info.step_idx`. The
resumed step's system prompt carries a resume `mode_note` (built in `_run_step`)
instructing the model to restart the unfinished step from the top and reuse or
repair partial artifacts on disk.

#### `_run_step(self, step: dict) -> str`

Returns `"next"` or `"quit"`. Inner `while True` loop:

1. On first iteration: build system prompt, emit opening CARL message, get Trillian suggestion.
2. Call `_prompt_user()` — pre-fills suggestion via readline.
3. Handle commands: `/next`, `/skip` → summarize, save, return `"next"`; `/quit`, `/exit` →
   return `"quit"`; `/suggest <text>` → call `_handle_suggest()`. In Step 10,
   `/done`, `/finish`, `/end`, `/complete`, and closeout phrases summarize,
   save `step_10_conversation.md`, append Step 10 memory, and finish revisit mode.
4. For Steps 1–2: call `_inject_file_content()` on input.
5. If `awaiting_confirmation` and `_is_affirmative()`: summarize, save, return `"next"`.
6. Append user message, call `_stream_response()`.
7. **One-turn step (Step 7)**: after first user response → summarize, save, return `"next"`.
8. **Tool steps (8–9) with tools used**: enter autonomous loop (max `_MAX_AUTO_ROUNDS`):
   - Break if `_asks_to_proceed()` or `"[DONE]" in response_text`.
   - Check stdin for interjections (non-blocking `select`).
   - Inject `[continue]` or interjection, call `_stream_response()`, auto-save.
   - Break if `used_tools` becomes False.
9. Set `awaiting_confirmation = _asks_to_proceed(response_text)`.
10. Get new Trillian suggestion.

For Step 10, tool selection is `REVISIT_TOOLS`, previous-decision context is
`build_revisit_context(session_dir)`, workflow length is one step, and Trillian
suggestions remain disabled because Step 10 is above `_TRILLIAN_LAST_ACTIVE_STEP`.

#### `_stream_response(self, system, messages, tools=None) -> tuple[str, bool]`

Appends messages in-place. Returns `(accumulated_text, had_tool_use)`.

Loop up to `_MAX_TOOL_ROUNDS`:
1. `_sanitize_messages(messages)`.
2. `_stream_single_turn(system, messages, tools, max_tokens)` → `(content_blocks, stop_reason)`.
3. Extract text and tool_use blocks.
4. **No tool_use, no visible text:** the whole allowance went to adaptive
   reasoning. Appending the blank turn would make the *next* Converse call fail
   validation (`text content blocks must contain non-whitespace text`), so the
   turn is retried instead — up to `_MAX_BLANK_TURNS` times, doubling
   `max_tokens` toward `_MAX_OUTPUT_TOKENS`. History is left untouched.
5. **No tool_use:** append assistant text message.
   - If `stop_reason == "max_tokens"`: inject `[continue — you were cut off]`, continue.
   - If `had_tool_use` and `_should_auto_continue(text)`: inject `[continue]`, continue.
   - Else: break.
6. **Has tool_use:** append assistant message with full content_blocks. Execute each tool.
   Append user message with list of `tool_result` blocks. Auto-save. Continue loop.

##### `_stream_single_turn(self, system, messages, tools, max_tokens, thinking_config=None) -> tuple[list, str, dict]`

Thin wrapper: `yield from converse_stream(self.client, self.model, system, messages, tools=tools, max_tokens=max_tokens, temperature=0.7, cache_system=True, cache_tools=bool(tools), thinking_config=thinking_config)`. `thinking_config` comes from `thinking_config_for_step(MODEL, step_number)`; when it is set the caller has already raised `max_tokens` with `thinking_max_tokens_for_step()`, because adaptive reasoning tokens are drawn from the same `maxTokens` allowance as the visible reply. Collects `("usage", ...)` and `("result", ...)` events and returns `(content_blocks, stop_reason, usage)`. Content blocks are in CARL's native shape. All event-parsing logic lives in `bedrock_converse.converse_stream`.

#### `_sanitize_messages(messages)` **static**

Walks message list. For every assistant message containing tool_use blocks, checks that the
next message has matching tool_result entries for all tool_use IDs. If not: truncates the list
at that assistant message (removes it and everything after).

#### `_should_auto_continue(text) -> bool` **static**

Returns True if the model output looks like narration between tool calls rather than a final
response: empty output, a short response (≤ 300 chars), or text whose tail (last 300 chars)
contains continuation keywords (`"let me"`, `"now i'll"`, `"i'll write"`, `"let me fix"`,
etc.). Returns False if the text ends with `?` (model is asking the user something).

#### `_get_trillian_suggestion(self, step_num, step_name, messages) -> str | None`

Returns None if `step_num > _TRILLIAN_LAST_ACTIVE_STEP` (6) or last message is from user.
Builds conversation as text-only messages (strips tool blocks). Calls `converse_text` with
`self.trillian_model`, Trillian system prompt + `"Generate your suggested response..."` user message. Max tokens: 256, temperature: 0.5. Returns None if response is `[SKIP]`.

#### `_call_llm_non_streaming(self, system, messages, max_tokens=2048, temperature=0.7) -> str | None`

Calls `converse_text(self.client, self.model, ...)` without caching. Used by `/suggest` and other one-off helper calls. Temperature suppression for Claude 5 / Opus 4.7 is handled inside `bedrock_converse._accepts_temperature`.

#### `_call_model_non_streaming(self, model_id, system, messages, max_tokens=2048, temperature=0.3) -> str | None`

Identical but takes explicit `model_id`. Used for the persona evolution pipeline (Sonnet, which accepts temperature).

> **Note:** Temperature handling is centralized in `bedrock_converse._accepts_temperature` / `_build_inference_config`.

#### `_handle_suggest(self, user_input: str) -> str | None`

Full `/suggest` workflow (terminal version). Runs in foreground with user prompts.

1. Strip `/suggest` keyword; build search context from full Step 1 conversation via `_build_suggest_context()`.
2. Use LLM to generate 2–3 search queries from context (max_tokens=150).
3. Call `search_papers()` for each query (limit=8). Merge + deduplicate. Save shortlist (10 papers) via `save_literature_results(..., overwrite=True)`.
4. **Interactive shortlist**: print numbered paper list; user selects subset by comma-separated numbers (Enter = all; `c` = cancel). Uses `_parse_index_selection()`.
5. Format selected papers with 300-char abstract excerpts.
6. Call LLM with `_SUGGEST_SYNTHESIS_SYSTEM` to get 3–5 candidate formulations.
7. Present candidates; interactive loop (max 3 iterations): user picks number, types `r` to refine, or `s` to re-search.
8. Call LLM with `_SUGGEST_DOCUMENT_SYSTEM` to generate detailed problem spec. Appends canonical references block via `_format_literature_shortlist_section()`.
9. Slugify title, save to `refs/<slug>.md` with YAML front-matter.
10. Copy to session `references/`. Set `_suggest_used = True`.
11. Return modified user_input with `\n\nrefs/<slug>.md` appended.

#### `_build_suggest_context(self, latest_request: str) -> str`

Builds an LLM prompt context string from the current-step conversation messages plus the
user's latest `/suggest` input. Used to generate focused search queries.

#### `_parse_index_selection(choice: str, max_index: int) -> list[int] | None` **static**

Parses comma-separated 1-based integer selections. Empty input returns all indices. Returns
None on invalid input (non-numeric or out-of-range).

#### `_format_literature_shortlist_section(papers: list[dict]) -> str` **static**

Builds a `## References` markdown block from a paper list, appended to `/suggest` specs.

#### `_summarize(self, system, messages, step) -> str`

Appends a bullet-point summary request (including instruction to preserve user directives
prefixed with `**User directive:**`) to a copy of messages. Calls `converse_text` with
`cache_system=True` so it reuses the still-warm step cache. Prints summary bordered with `~`. Returns text.

#### `_asks_to_proceed(text) -> bool` **static**

True if text contains phrases: `"ready to proceed"`, `"proceed to step"`, `"move on to step"`,
`"shall we proceed"`, `"shall we move on"`, `"move to step"`, `"move to the next"`,
`"go ahead and proceed"`, plus Step 10 closeout phrases such as `"ready to finish"`,
`"shall we close out"`, `"close out step 10"`, or `"done with the session"`.

#### `_is_finish_command(text) -> bool` **static**

Step 10-specific explicit finish detector. Accepts short inputs such as
`/done`, `/finish`, `/end`, `/complete`, `we are done`, `all done`, `end step 10`,
and `finish step 10`, while rejecting negated phrases like `not done`, `not yet`,
`wait`, or `hold on`.

#### `_is_affirmative(text) -> bool` **static**

True if: `len(text) <= 60` AND no qualifier words (`but`, `however`, `wait`, `actually`,
`change`, `what about`, `hold on`, `not yet`, `no`) AND text starts with an affirmative token
(`yes`, `y`, `proceed`, `continue`, `next`, `go`, `go ahead`, `looks good`, `lgtm`, `ok`,
`okay`, `sure`, `confirmed`, `ready`, `let's go`, `lets go`, `move on`, `sounds good`,
`perfect`, `great`, `yep`, `yeah`, `absolutely`).

#### `_contains_tool_call_text(text) -> bool` **static**

True if text contains `<tool_call>`, `{"name": "write_file"`, `{"name": "run_command"`, or
similar patterns. Used to detect model jumping ahead to coding in non-tool steps.

#### `_inject_file_content(self, user_input: str) -> str`

Tokenize input on whitespace. For each token: strip punctuation, resolve relative paths against
`carl_dir`. If token is an existing `.md`/`.txt`/`.text`/`.rst` file: read it, copy to session
`references/`, append to result with `\n\n--- File: <path> ---\n...\n---` delimiters.

#### `_auto_save_conversation(self)`

If `_current_step` set and `len(_current_messages) > 1`: calls
`session.save_conversation(step_num, step_name, messages[1:], origins)`. Overwrites file.

#### `_save_on_exit(self)`

Called in `finally` of `run()`. Returns early if `current_step_idx >= len(STEPS)` (all done).
Otherwise: kill background processes, save conversation, attempt summary, write
`"**Session ended during Step N: Name**"` to memory.md, print session dir.

#### `_kill_background_processes(self)`

Read `code/.bg_pids.json`. For each PID: `os.killpg(pid, SIGTERM)`. Fall back to
`os.kill(pid, SIGTERM)` if group kill fails.

#### Trillian persona evolution

| Method | Purpose |
|--------|---------|
| `_TRILLIAN_STEPS` class var | List of `(step_num, step_name)` tuples for steps 1–6 (including 2.5) |
| `_gather_trillian_conversations(self) -> str` | Read step 1–6 conversation files (12 KB cap each), concat with `=== Step N: Name ===` headers |
| `_persona_triage(self, persona, conversations) -> tuple[bool, str]` | Step A: Sonnet yes/no. max_tokens=150, temp=0.2. Parses `YES: reason` or `NO`. |
| `_persona_focused_update(self, persona, conversations, triage_explanation) -> str \| None` | Step B: full file output. max_tokens=16384, temp=0.3. Enforces ≤ `int(line_count*1.02)+1` lines. |
| `_persona_consistency_check(self, updated_persona) -> str \| None` | Step C: check contradictions. max_tokens=16384, temp=0.2. Returns complete file. |
| `_update_trillian_persona(self) -> None` | Orchestrator. Size gates after B and C (2% lines, 5% chars). Validates `# Trillian:` header. Backs up to `.md.bak`. Logs to `memory.md`. Full try/except. |

#### `_rename_session_dir(self)`

After Step 1: takes step summary, calls LLM for 2–3 word slug (max_tokens=30, temp=0.0).
Calls `session.rename_with_topic(slug)`.

#### `_prompt_user(self) -> str | None`

If `_trillian_suggestion` set: uses `readline` pre-fill (`readline.set_startup_hook`) to
populate input buffer. Returns None on EOF or if user types `/quit`/`/exit`.

---

## web.py

### Module-level globals

```python
CARL_DIR: str                         # set at module import
REGION, TRILLIAN_MODEL,               # from env
PERSONA_UPDATE_MODEL: str
bedrock: boto3 client
MODEL: str                            # resolved at startup via _resolve_model(bedrock)
_trillian_persona: str

state: dict                           # mutable session state (see below)
_lock: threading.Lock
_log_entries: list[dict]              # polling log buffer (max 2000 → prune to 1000)
_log_counter: int
_chat_events: list[dict]              # polling event buffer (max 500 → prune to 250)
_chat_counter: int
_busy: bool                           # True while background thread running
_interjection_queue: queue.Queue      # messages injected during autonomous steps
_REVISIT_DIR: Path | None             # set by --revisit; selects Step 10 mode
_RESUME_DIR: Path | None              # set by --resume; selects resume mode
_EARLY_RESUME_ARG: str | None         # raw --resume value parsed early at import
```

`MODEL` is set by `_resolve_model(bedrock)` which probes Opus 5 via a 5-token `client.converse(...)` call and falls back to Sonnet 5, identical logic to `chat.py`. `REPORT_MODEL`, `TRILLIAN_MODEL` and `PERSONA_UPDATE_MODEL` are read straight from their env overrides with no probe. Temperature suppression is handled inside `bedrock_converse`.

### `state` dict keys

| Key | Type | Purpose |
|-----|------|---------|
| `"session"` | `Session` | Current session |
| `"step_idx"` | `int` | Index into STEPS |
| `"messages"` | `list[dict]` | Conversation for current step |
| `"awaiting_confirmation"` | `bool` | CARL asked to proceed |
| `"started"` | `bool` | Step opening message sent |
| `"trillian_suggestion"` | `str \| None` | Pre-fill for next user turn |
| `"suggest_pending"` | `dict \| None` | Phase 1 /suggest output awaiting Phase 2 |
| `"suggest_used"` | `bool` | Skip Step 2.5 flag (set by /suggest) |
| `"user_skip_25"` | `bool` | Skip Step 2.5 flag (set by `/skip` in Step 2) |
| `"msg_origins"` | `dict[int, dict]` | Trillian origin metadata |
| `"write_report"` | `bool` | From CLI `--write-report` flag |
| `"no_persona_update"` | `bool` | From CLI `--no-persona-update` flag |
| `"revisit_mode"` | `bool` | True when web server was launched with `--revisit` |
| `"revisit_completed"` | `bool` | Set after Step 10 is summarized and closed |
| `"revisit_exclusions"` | — | Revisit-mode workspace copy exclusions |
| `"revisit_guidance"` | — | Revisit-mode guidance state |
| `"revisit_preflight_done"` | `bool` | Revisit-mode preflight completed |
| `"resume_mode"` | `bool` | True when launched with `--resume` |
| `"resume_info"` | `ResumeInfo \| None` | Resume target + restored decisions |
| `"resume_warnings"` | `list` | Warnings surfaced to the UI on resume |

### Flask routes

| Method | Path | Handler | Notes |
|--------|------|---------|-------|
| GET | `/` | `index()` | Serve `templates/index.html`, or `templates/step10.html` in revisit mode |
| POST | `/api/start` | `start()` | Init state, return session metadata + STEPS/REVISIT_STEP list, `revisit` flag, and resume fields (`resume`, `resume_step`, `resume_warnings`); returns HTTP 400 on `ResumeError` |
| GET | `/api/status` | `status()` | Return step_idx, current step dict, done flag, session_dir, and `revisit` flag |
| GET | `/api/logs?since=N` | `logs()` | Return `{"entries": [{id, text, level},...]}` |
| GET | `/api/events?since=N` | `events()` | Return `{"entries": [{id, event, data},...], "busy": bool}` |
| POST | `/api/message` | `message()` | Fire `_run_turn()` in background thread, return `{"status": "ok"}` |
| POST | `/api/interject` | `interject()` | Put message into `_interjection_queue` |

When launched with `python web.py --revisit sessions/<session_dir>`,
`resolve_revisit_session()` sets `_REVISIT_DIR`, `/` serves
`templates/step10.html`, and `/api/start` returns a one-step workflow containing
only Step 10.

When launched with `python web.py --resume sessions/<session_dir>`, the raw
value is captured early (`_EARLY_RESUME_ARG`), `build_resume_info()` validates
it and sets `_RESUME_DIR`, and `_init_state()` opens the existing session with
`Session.open_existing()`, restores `step_decisions`, and seeds
`state["step_idx"]` from the resume target. `/api/start` then returns `resume:
true`, `resume_step`, and any `resume_warnings`; a `ResumeError` is caught and
returned as HTTP 400. `_resume_mode_note(step_num)` injects a resume note into
the resumed step's prompt, and `emit_resume_marker()` is called at step
start/save and session completion so the `web.log` markers stay current.
`--resume` is mutually exclusive with `--revisit`.

### Key functions

| Name | Purpose |
|------|---------|
| `_log(text, level="text")` | Append to `_log_entries` with auto-id. Common levels: `"text"`, `"tool"`, `"result"`, `"info"`, `"error"`, `"warn"`, `"step"` |
| `_emit(event, data)` | Append to `_chat_events` with auto-id |
| `_start_session_log(session_dir)` | Redirect stdout/stderr to TeeWriter → session `web.log` |
| `_init_state()` | Reset all state; create new `Session` or `Session.open_existing()` in revisit mode; preserves `write_report` and `no_persona_update` flags |
| `_sanitize_messages()` | Identical to chat.py |
| `_execute_tool(name, input_data)` | Dispatch normal, literature, and revisit tools; Step 10 restricts writes to `code/revisit_N/` and refuses retraining commands |
| `_stream_single_turn(system, messages, tools, max_tokens)` | Generator: `yield from converse_stream(bedrock, MODEL, ...)` with `cache_system=True`, `cache_tools=bool(tools)`. Yields `("text", str)`, `("tool_start", name)`, `("usage", dict)`, `("result", (blocks, stop_reason))` |
| `_should_auto_continue(text)` | Identical to chat.py |
| `_asks_to_proceed(text)` | Identical to chat.py |
| `_is_affirmative(text)` | Identical to chat.py |
| `_get_step9_check_interval()` | Thin wrapper over `tool_runtime.check_interval_seconds(step7_summary, log=lambda msg: _log(msg, "info"))`; min 60 s |
| `_get_trillian_suggestion(step_num, step_name, messages)` | Identical to chat.py; emits `trillian_suggestion` event |
| `_rename_session()` | Identical to chat.py |
| `_summarize(system, messages, step, tools=None)` | Emits `summary` event; calls `converse_text` with `cache_system=True` to reuse the still-warm step cache. Accepts the step's `tools` and forwards them (along with `cache_tools=bool(tools)`) so `toolConfig` is present when `messages` contains `tool_use` / `tool_result` blocks — required by Bedrock whenever the step used tools. `max_tokens=1536` (was 512; raised to avoid truncating `**User directive:**` bullets off the end of long step summaries). |
| `_call_llm_non_streaming(system, messages, max_tokens, temperature)` | Uses module-level `MODEL`; calls `converse_text` without caching |
| `_call_model(model_id, system, messages, max_tokens, temperature)` | Explicit model ID; calls `converse_text` without caching |
| `_gather_trillian_conversations(sess)` | Takes `sess` arg instead of `self.session` |
| `_extract_persona_feedback(conversations)` | Builds compact accepted/edited suggestion digest for web persona triage |
| `_persona_triage(persona, feedback_digest)` | Standalone; uses `PERSONA_UPDATE_MODEL` global |
| `_persona_focused_update(persona, conversations, triage_explanation)` | Standalone |
| `_persona_consistency_check(updated_persona)` | Standalone |
| `_update_trillian_persona()` | Uses `state["session"]` and `CARL_DIR`; emits events; skipped if `no_persona_update` |
| `_extract_formulation(suggestions_text, index)` | Parse Nth formulation from synthesis output |
| `_handle_suggest_v2(user_input)` | Active web `/suggest` phase 1: canonical 10-paper shortlist + candidate formulations; stores `state["suggest_pending"]` |
| `_handle_suggest_selection_v2(user_input)` | Active web `/suggest` phase 2: selected formulation -> spec in `refs/<slug>.md` |
| `_auto_save_conversation()` | Uses state dict |
| `_kill_background_processes()` | Identical to chat.py |
| `_workflow_steps()` | Return `[REVISIT_STEP]` in revisit mode, otherwise `STEPS` |
| `_advance_step(step_num, system, messages, step, tools=None)` | Save summary, increment `state["step_idx"]`, skip 2.5 if `suggest_used` or `user_skip_25`, emit events, reset messages. Callers pass the step's `tools` so the summary call has `toolConfig` when needed. The autonomous-loop summary at `_run_autonomous_steps` passes `tools=TOOLS` directly to `_summarize`. |
| `_finish_revisit()` | Mark revisit complete, emit `done`, and schedule shutdown |
| `_run_turn(user_input)` | One conversation turn; runs in background thread spawned by `/api/message` |
| `_run_autonomous_steps()` | Loop steps 8–9 autonomously; handles interjection queue; calls `_update_trillian_persona()` at end |
| `_stream_full(system, messages, tools=None)` | Tool-use loop (identical logic to chat.py `_stream_response`); emits events. Resolves the step's thinking tier via `_current_step_number()` → `thinking_config_for_step()` and raises `max_tokens` to match; retries blank turns rather than recording them |
| `_current_step_number()` | Number of the step currently executing (`state["step_idx"]` into `_workflow_steps()`), or `None` outside the workflow. Used for the thinking tier and the per-step literature-search counter |

In revisit mode, `_run_turn()` builds context with `build_revisit_context()`,
uses `REVISIT_TOOLS`, saves Step 10 through `_advance_step()`, and calls
`_finish_revisit()` after an explicit finish command or affirmative response to
"Ready to finish Step 10?".

### Chat event types emitted by `_emit()`

| Event | `data` keys | When |
|-------|-------------|------|
| `"text"` | `{text}` | CARL streaming text chunk |
| `"step"` | `{number, name}` | New step starts |
| `"summary"` | `{text, step}` | Step summary ready |
| `"trillian_suggestion"` | `{text}` | Suggestion ready for pre-fill |
| `"autonomous_start"` | `{}` | Entering Steps 8-9 |
| `"step_advance"` | `{next}` | Interactive step saved and advanced |
| `"interjection"` | `{text}` | User interjection accepted during autonomous execution |
| `"turn_end"` | `{}` | Background web turn finished without advancing |
| `"log"` | `{text}` | Supplemental status text for the web flow |
| `"info"` | `{text}` | Status message |
| `"error"` | `{text}` | Error message |
| `"done"` | `{text}` | Workflow complete |

---

## report.py

(~1,125 lines.)

### Constants

| Name | Value | Cap applied to |
|------|-------|---------------|
| `_MAX_MEMORY` | 4000 | `memory.md` content |
| `_MAX_LITERATURE` | 6000 | Literature search results |
| `_MAX_CODE_SNIPPET` | 1500 | Each code file |
| `_MAX_LOG_LINES` | 50 | Training log lines |
| `_MAX_STEP_FILE` | 12000 | Each step conversation included in report prompts |
| `_MAX_MODEL_CARD` | 6000 | `artifact/MODEL_CARD.md` |
| `_MAX_CONTRACT` | 4000 | `artifact/contract.json` |
| `_KEY_CODE_FILES` | `["environment.py", "agent.py", "config.py", "models.py", "train.py"]` | Files to include |

### Plot discovery (shared with revisit.py)

| Signature | Purpose |
|-----------|---------|
| `is_checkpoint_dupe(path: Path) -> bool` | `"-checkpoint" in path.stem` — a Jupyter autosave duplicate |
| `find_session_plots(session_dir: Path) -> list[Path]` | All `.png` under the session's `code/` and `results/` dirs, recursively (`rglob`), sorted per subdir, with checkpoint dupes filtered |

`find_session_plots` recurses, so plots written under `code/experiments/<run>/`
are found. `revisit.discover_artifacts` imports both functions, so the report and
the Step 10 inventory see one identical plot set. `_find_plots =
find_session_plots` is kept as a backwards-compatible alias for the existing
`report.py` call sites.

### Artifact bundle discovery

| Signature | Purpose |
|-----------|---------|
| `find_artifact_dirs(session_dir: Path) -> list[Path]` | Every `artifact` directory under the session's `code/` tree (`code/artifact/` after Step 9, `code/revisit_N/artifact/` after a revisit), sorted so the newest is last, checkpoint dupes excluded |
| `gather_artifact_context(session_dir: Path) -> dict` | Metadata for the last (most recent) bundle: `path` (session-relative), capped `model_card` from `MODEL_CARD.md`, capped `contract` from `contract.json`, and `entry_points` (bundle files with a `.py`/`.pt`/`.json`/`.md` suffix). Returns `{}` when no bundle exists, so callers treat the artifact as optional |

`_gather_context` sets `ctx["artifact"] = gather_artifact_context(session_dir)`.

### Sections generated

Seven sections: `abstract`, `introduction`, `literature_review`, `problem_description`,
`methodology`, `results`, `discussion`. `methodology` is generated from two
subsection calls. Each section specifies a LaTeX heading, LLM system prompt,
context keys, and, where useful, which step conversation files to include.

The `artifact` context key is wired into two of them:

- **results** — `context_keys` = `memory`, `training_logs`, `plot_descriptions`,
  `benchmark_context`, `artifact`, `step_conversations`.
- **discussion** — `context_keys` = `memory`, `artifact`, `step_conversations`;
  its system prompt instructs the model to state the artifact's path, the command
  that runs it, and the validity domain it is trusted within.

`_format_section_context` renders the key as a `# Shippable Inference Artifact`
block naming the bundle path, the `python solve.py --instance instance.json`
invocation, and the bundle file list, followed by the `MODEL_CARD.md` text and a
fenced `contract.json`. It instructs the section not to claim performance outside
the recorded validity domain.

### `_describe_plot(client, model, plot_path, memory, ...) -> dict`

Reads `plot_path.read_bytes()` directly and passes raw PNG bytes in a
`{"type": "image_raw", "format": "png", "data": bytes}` block, which
`messages_to_converse` converts to the Converse `{"image": {"format": "png",
"source": {"bytes": ...}}}` shape. Returns `{filename, label, description}`.

### `generate_report(session, client, model) -> Path | None`

1. `_gather_context()` — read memory.md, literature JSON, key code files, logs, plot paths (via `find_session_plots`), the `artifact` bundle context, and capped step conversations.
2. Generate `references.bib` (one entry per paper with auto `cite_key`).
3. Extract benchmark context from memory/code so plot descriptions can name analytical reference formulas.
4. Describe plots in parallel with multimodal Converse calls.
5. `ThreadPoolExecutor(max_workers=len(SECTIONS))` — generate all sections in parallel.
6. `_cleanup_generated_sections()` — normalize citations/references and warn on unresolved labels.
7. `_generate_title()` — LLM call for academic title.
8. `_render_main_tex()` — Jinja2 + `pdflatex`, `bibtex`, and two final `pdflatex` passes.
9. Return PDF path or None.

---

## literature.py

### Constants

`TIMEOUT=20`, `MAX_RETRIES=2`, `RETRY_DELAY=3`, `_API_KEY` from `SEMANTIC_SCHOLAR_API_KEY` env.

Quality-filtering constants:

| Name | Value | Purpose |
|------|-------|---------|
| `MAX_REQUESTS_PER_SEARCH` | `8` | Hard ceiling on HTTP requests for one `search_papers()` call, counting every backend, relaxation round and retry. Without it the worst case is ~27 requests. |
| `_MAX_PAGES` | `2` | Max result pages fetched per backend query |
| `_PAGE_SIZE` | `20` | Results fetched per page (over-fetches to compensate for filtering) |
| `_MAX_TERMS` | `6` | Distinctive query terms kept (longest-first) before ANDing |
| `_MAX_RELAX_ROUNDS` | `4` | Times the arXiv term set is relaxed when a query is over-specified |
| `_MIN_REQUEST_INTERVAL` | `{export.arxiv.org: 3.0}` | Per-host request spacing arXiv asks for |
| `_API_KEY` | env `SEMANTIC_SCHOLAR_API_KEY` | When set, Semantic Scholar moves to the **front** of the backend order; keyless it goes last, because the shared pool answers 429 for most requests |
| `_CONTACT` | env `OPENALEX_CONTACT_EMAIL` | Routes OpenAlex calls to the faster "polite pool" |

Persistence constant: `_MAX_MD_PAPERS = 10` — cap on entries written to `literature_search.md`.

### Functions

| Signature | Returns / Notes |
|-----------|---------|
| `search_papers(query: str, limit: int = 10) -> list[dict]` | List of `{title, authors, year, abstract, citation_count, paper_id, url, pdf_url, doi, source}`. Requires a non-empty abstract (it feeds the summariser); `pdf_url` is preferred and sorts first. `[]` if every backend is empty or failing. |
| `search_papers_with_status(query, limit) -> (list[dict], list[str])` | As above plus one error string per failing backend, so callers can distinguish an unreachable API from an empty index. Both engines call this one. |
| `_backends() -> list[(name, fn)]` | arXiv → OpenAlex → Semantic Scholar; S2 moves first when `SEMANTIC_SCHOLAR_API_KEY` is set. Each backend gets a quota of `ceil(2·limit/3)` slots so one cannot fill the whole list, then leftovers fill any empty slots. |
| `_search_arxiv(query, limit, budget)` | ANDs the distinctive terms; on too few hits drops the shortest (least distinctive) term and retries, up to `_MAX_RELAX_ROUNDS`, accumulating across rounds. Abstract + PDF on every hit; no citation counts. |
| `_search_openalex(query, limit, budget)` | Tries `title_and_abstract.search` on all terms, then on the 3 longest, then a fuzzy `search=`. Rebuilds abstracts from `abstract_inverted_index`. Supplies the citation counts arXiv lacks. |
| `_search_semantic_scholar(query, limit, budget)` | Paginates up to `_MAX_PAGES`. Best relevance ranking, but keyless it is effectively always 429. |
| `_RequestBudget(allowance)` | Shared HTTP allowance for one search. Backends check `.exhausted` and stop cleanly, returning partial results; running out is a cap doing its job, never reported as an error. |
| `_dedupe_keys(paper) -> list[str]` | Title **and** DOI keys. arXiv exposes no DOI, so keying on DOI alone would double-count a preprint and its published record. On a merge the richer record wins: preprint PDF plus published citation count. |
| `_throttle(url)` | Enforces `_MIN_REQUEST_INTERVAL` per host. |
| `_request(url, params, budget, headers=None, parse="json")` | Parsed JSON, or raw text when `parse="text"`. Charges *every* attempt (retries included) against `budget`; returns `None` once spent. Retries on 429 honouring `Retry-After`, then raises `LiteratureBackendError` — a transport failure is never reported as "no papers found". |
| `format_results_for_tool(papers, errors=None) -> str` | Compact text with 200-char abstract excerpts, ≤3 authors, plus the source backend. When `errors` is non-empty the text states explicitly that this is a transport problem and that rephrasing will not help. |
| `_fetch_pdf_text(pdf_url, max_pages=3) -> str` | Download PDF and extract text from first `max_pages` pages via `pypdf`. Returns `""` on any failure or if `pypdf` not installed. |
| `_summarize_paper(title, text, bedrock_client, model_id) -> str` | LLM call producing a 40–50 word summary covering RL algorithm, state/action/reward design, and main result. Input capped at 6000 chars. Returns `""` on failure. |
| `save_literature_results(session_dir, papers, bedrock_client=None, summary_model="", overwrite=False)` | Append to `references/literature_search.json` (no cap) and `.md` (capped at `_MAX_MD_PAPERS` total entries), or replace both when `overwrite=True`. If `bedrock_client` and `summary_model` provided: fetches PDF text (falling back to abstract) and writes an LLM-generated summary below each entry in the markdown file. |

---

## carl.py

Thin entry point. Parses `--write-report`, `--no-persona-update`, and a
mutually-exclusive group of `--revisit` / `--resume`. Calls
`_load_dotenv(carl_dir)`, checks boto3 availability, and instantiates:

```python
CARLChat(
    carl_dir,
    write_report=args.write_report,
    no_persona_update=args.no_persona_update,
    revisit_dir=args.revisit,
    resume_dir=args.resume,
).run()
```

The construction is wrapped in a try/except: a `ResumeError` (imported from
`resume`) prints `Cannot resume: <reason>` and exits with code 2.

`--revisit sessions/<session_dir>` opens an existing session directly in Step
10. `--resume sessions/<session_dir>` reopens an existing session and continues
the normal workflow from its last unfinished step. Normal runs (neither flag)
still create a new session and execute Steps 1-9.

---

## tests/

```bash
python -m pytest tests/ -q
```

Pure-Python unit tests with no AWS calls; the current suite passes. `conftest.py`
puts the repo root on `sys.path` so the tests import the modules CARL actually
runs, not a copy vendored under `paper/`. `tests/test_resume.py` replaces the
stale whole-source-tree copy that used to live at `paper/Code/test_resume.py`.

| File | Covers |
|------|--------|
| `conftest.py` | `sys.path` bootstrap for the repo root |
| `test_resume.py` | `resume.build_resume_info` — marker parsing, legacy log fallback, skip-2.5 reconstruction, `ResumeError` on completed sessions |
| `test_tool_runtime.py` | `is_revisit_training_command` (trainers refused, read-only commands allowed), `parse_training_budget_seconds` / `check_interval_seconds`, `normalize_code_relpath` |
| `test_plot_discovery.py` | `report.find_session_plots` and `revisit.discover_artifacts` agree; `code/experiments/**` plots found; `-checkpoint` dupes filtered |
| `test_memory_parsing.py` | `resume.parse_memory_decisions` concatenates duplicate `## Step N` bodies |
| `test_skills.py` | Every `STEPS` / `REVISIT_STEP` entry resolves to a `carl_skills/<name>/SKILL.md` whose frontmatter `name` matches its directory; `load_skill` strips frontmatter |
| `test_artifact.py` | `report.find_artifact_dirs` / `gather_artifact_context`, the `artifact` report context key, `revisit_report._artifact_subsection`, and quiet degradation when no bundle exists |
| `test_model_config.py` | `bedrock_converse` thinking/temperature shape per model generation |

A root `.gitignore` covers `__pycache__/`, `.pytest_cache/`,
`.ipynb_checkpoints/` (whose autosave duplicates were being counted twice by the
report and the Step 10 inventory), `.env`, `node_modules/`, regenerable session
binaries (`sessions/*/code/**/*.pt`, TensorBoard event files, `.bg_pids.json`),
and LaTeX build products. The LaTeX rules are deliberately scoped to `results/`
rather than a bare `*.log`, because training logs and `web.log` are primary
evidence and must stay tracked.

---

## Message Formats (Bedrock Converse)

All LLM calls go through the Bedrock **Converse API** (`client.converse` /
`client.converse_stream`). Request construction is centralized in
`bedrock_converse._build_kwargs`.

### Converse request shape

```python
{
    "modelId": str,
    "system": [
        {"text": system_prompt_string},
        {"cachePoint": {"type": "default", "ttl": "1h"}},  # when cache_system=True
    ],
    "messages": [                          # Converse shape — see conversion table above
        {"role": "user",      "content": [{"text": "..."}]},
        {"role": "assistant", "content": [{"text": "..."}, {"toolUse": {...}}]},
        {"role": "user",      "content": [{"toolResult": {...}}]},
    ],
    "inferenceConfig": {
        "maxTokens": int,        # 1024 terminal text, 4096 web text, 16384 tool steps, 256 Trillian
        # "temperature": float   # omitted for Claude 5 / Opus 4.7 (see _accepts_temperature)
                                 # 0.7 main, 0.5 Trillian, 0.2-0.3 persona ops
    },
    "toolConfig": {              # omitted for non-tool steps
        "tools": [
            *toolSpec_entries,
            {"cachePoint": {"type": "default", "ttl": "1h"}},  # when cache_tools=True
        ]
    },
}
```

### In-memory message variants (native shape — unchanged)

The in-memory `messages` list still uses CARL's native shape throughout the
codebase. `messages_to_converse` translates it at the API boundary only.

```python
# Simple user/assistant
{"role": "user", "content": "text"}
{"role": "assistant", "content": "text"}

# Assistant with tool calls
{"role": "assistant", "content": [
    {"type": "text", "text": "narration"},
    {"type": "tool_use", "id": "toolu_01...", "name": "write_file",
     "input": {"path": "env.py", "content": "..."}},
]}

# User with tool results (must follow immediately after the above)
{"role": "user", "content": [
    {"type": "tool_result", "tool_use_id": "toolu_01...", "content": "File written."},
]}
```

### `stop_reason` values (Converse)

| Value | Meaning |
|-------|---------|
| `"end_turn"` | Model finished normally |
| `"max_tokens"` | Cut off — inject `[continue — you were cut off]` |
| `"tool_use"` | Model emitted tool call — execute and continue |

### `msg_origins` structure

```python
# Index is position in messages list (0-based after removing the initial [Begin step] message)
{
    3: {"type": "trillian_accepted", "trillian": "ok"},
    7: {"type": "trillian_edited", "trillian": "Use 20-year horizon.", "final": "20 years, fixed."},
}
```

---

## carl_skills/ files

Anthropic standard skill layout: one directory per step, each holding a single
`SKILL.md`.

| File | Step | Size |
|------|------|------|
| `01-problem-definition/SKILL.md` | 1 | 3.5 KB |
| `02-references/SKILL.md` | 2 | 1.4 KB |
| `025-literature-search/SKILL.md` | 2.5 | 2.3 KB |
| `03-parameter-exploration/SKILL.md` | 3 | 2.7 KB |
| `04-information-proposal/SKILL.md` | 4 | 4.4 KB |
| `05-action-definition/SKILL.md` | 5 | 4.8 KB |
| `06-reward-definition/SKILL.md` | 6 | 4.8 KB |
| `07-ml-parameters/SKILL.md` | 7 | 9.5 KB |
| `08-initial-testing/SKILL.md` | 8 | 8.4 KB |
| `09-launch/SKILL.md` | 9 | 19.9 KB |
| `10-post-training-analysis/SKILL.md` | 10 | 15.9 KB |

Frontmatter is YAML with `name` (equal to the containing directory name),
`description`, and — where the skill was distilled from the reference RL skills —
`metadata.sources` listing the `jit_skills/` sections it draws on. `sources` lives
under `metadata:` because it is not a standard skill key.

```yaml
---
name: 09-launch
description: Write the full RL repository and execute all training phases
metadata:
  sources:
    - rl-repository-setup (Directory Structure, Requirements)
    ...
---
```

Only the **current step's** skill is loaded into the system prompt, and only its
body — `load_skill` applies `strip_frontmatter`, so the frontmatter above never
reaches the model. Never all at once. Step 10's skill is loaded only in revisit
mode.

`09-launch/SKILL.md` ends with `## Final Usable Artifact (MANDATORY)` and a
`## Launch Checklist`. The artifact section requires a self-contained `artifact/`
bundle built by a per-session `make_artifact.py` (invoked
`python make_artifact.py --run-name <run> [--ckpt PATH] [--out artifact/]`) that
**copies** the best checkpoint to `artifact/best_model.pt`, writes
`contract.json`, and copies the runtime modules in — nothing in the bundle may
import from the training tree. Inside the bundle: `solve.py` (the stranger's
entry point, `--instance` / `--param k=v` / `--out` / `--n-seeds`, validating the
instance against the contract's parameter schema and failing loudly rather than
clipping), `smoke_test.py` (reproduce the recorded metric on the canonical
instance; a failure blocks shipping), and `MODEL_CARD.md` (validity domain,
best-vs-latest-vs-baseline, known failure modes, copy-paste usage). It also
requires every checkpoint to be saved with an `extra` dict carrying `obs_dim`,
`action_dim` (and bounds), a config snapshot, and the selection metric — that
dict is `make_artifact.py`'s source of truth. Both `long_train.sh` and (via
`revisit.py`) `revisit_train.sh` end by re-running `make_artifact.py` so a longer
or revised run refreshes the bundle from its own better checkpoint. The same
requirements are mirrored in `jit_skills/rl-model-management/SKILL.md` under
`## Inference Artifact`.

`07-ml-parameters/SKILL.md` requires the step summary to carry the budget as a
machine-readable `TRAINING_BUDGET_MINUTES: N` line on its own line (integer
minutes, nothing after the number), emitted even when the user accepted the
default; `tool_runtime.parse_training_budget_seconds` reads it.

`10-post-training-analysis/SKILL.md` is content-only: it instructs CARL to review
prior directives, inventory Step 8/9 artifacts, diagnose eval progression,
signal/noise, reward alignment, policy behavior, checkpoint compatibility,
fresh revisit initialization, and safe rollout parallelization opportunities
before preparing the next-run prescription. Its
`## Visual Evidence Review (MANDATORY, BEFORE ANY RECOMMENDATION)` section
requires a `describe_plot` call on **every** entry of the
`list_revisit_artifacts` `plots` list — a worklist, not a sample — reported with
axes, trend, and visible numerical range; an empty or implausibly short
inventory is itself a finding to investigate (`find . -name '*.png'`), not a
reason to skip the section. It further requires trajectory reading in problem
terms, naming the comparator series present *and missing* on every performance
plot, a coverage check against Step 9's `### Essential Plots`, and evidence
binding: every `## MDP Revision Recommendations` entry must cite the plots and/or
log lines that motivate it.

---

## jit_skills/

Eight reference RL skills in the same standard layout — the source material the
`carl_skills` distil and cite in their `metadata.sources`. Each is
`<skill-name>/SKILL.md` with `name` / `description` frontmatter; the body was
dedented out of the non-standard `instructions: |` frontmatter key it previously
lived in, so it is now ordinary markdown.

| Skill | Size |
|-------|------|
| `rl-repository-setup/SKILL.md` | 0.9 KB |
| `rl-environment-design/SKILL.md` | 9.1 KB |
| `rl-agent-implementation/SKILL.md` | 6.4 KB |
| `rl-training/SKILL.md` | 11.4 KB |
| `rl-model-management/SKILL.md` | 6.3 KB |
| `rl-visualization/SKILL.md` | 2.6 KB |
| `rl-testing/SKILL.md` | 3.5 KB |
| `rl-problem-workflow/SKILL.md` | 6.4 KB |

The tree's former `README.md` now lives at `docs/rl-skills-index.md`.

---

## trillian_skills/

`trillian-persona/SKILL.md` (276 lines, ~16 KB). After the `name` /
`description` frontmatter, ten sections:

1. Overall Persona — domain expert, weak on RL/ML, delegates implementation
2. Tone and Style — terse (2–15 words), no greetings/emoji/praise
3. Approval Patterns — `"ok"`, `"yes"`, `"yes go ahead"`, `"let's proceed"`, `"Ok. Go!"`
4. Corrections — surgical, additive (`"ok, but also..."`), no softening
5. Handling Options — picks immediately, biased toward simplicity
6. Specification Behavior — front-loads problem spec, leaves ML/RL open
7. Distinctive Quirks — self-corrects concisely, additive afterthoughts
8. Step-by-Step Behavior — Steps 1–2 highest engagement; Steps 7–9 deactivated
9. What Trillian Should Never Do — 10 explicit prohibitions
10. Answering Technical Questions — 8 categories (A–H) with examples

Both engines resolve the file through `steps.trillian_persona_path(carl_dir)`, so
the pre-standard `trillian_skills/Trillian-Persona.md` still loads as a logged
fallback. The backup is `persona_path.with_suffix(".md.bak")`, i.e.
`trillian_skills/trillian-persona/SKILL.md.bak`, written before any update. The
web flow additionally saves a pre-update copy at
`<session>/refs/Trillian-Persona-before-update.md`.

---

## Key Constraints and Limits

| Parameter | Value | Location |
|-----------|-------|----------|
| Tool-use loop max rounds | 80 | `tool_runtime.MAX_TOOL_ROUNDS` |
| Autonomous loop max rounds | 20 | `_MAX_AUTO_ROUNDS` |
| Max tokens — text steps | 1024 terminal, 4096 web, then raised to the step's thinking tier **capped at `medium` (8192)** | `chat.py:_stream_response`, `web.py:_stream_full`, `_TEXT_TURN_TIER_CEILING` |
| Max tokens — tool steps | 16384, then raised to the step's thinking tier (no ceiling) | `chat.py:_stream_response`, `web.py:_stream_full` |
| Blank-turn retries | 2, doubling max_tokens up to 32000 | `_MAX_BLANK_TURNS`, `_MAX_OUTPUT_TOKENS` (both engines) |
| Literature searches per step | 8 | `tool_runtime.MAX_LITERATURE_SEARCHES` |
| HTTP requests per literature search | 8 | `literature.MAX_REQUESTS_PER_SEARCH` |
| Max tokens — Trillian | 256 | `chat.py` `_get_trillian_suggestion` |
| Max tokens — summaries | 512 (`chat.py`), 1536 (`web.py`) | `_summarize` |
| Max tokens — persona triage | 150 | persona pipeline |
| Max tokens — persona update/check | 16384 | persona pipeline |
| run_command timeout | 600 s | tool definition |
| Step 10 retraining guard | no `run_background`; refuse commands whose executed program is `train.py`, `long_train.py`, `long_train.sh`, `revisit_train.sh`, `run.sh`, or `train.sh` | `tool_runtime.is_revisit_training_command` |
| Step 7 training budget | `TRAINING_BUDGET_MINUTES: N`, keep under 90; 45 min default | `tool_runtime.parse_training_budget_seconds` |
| check_process min wait | 60 s | `tool_runtime.CHECK_PROCESS_MIN_WAIT` |
| Baselines enforced limit | 60 s | Step 8–9 system prompt |
| Tool result output cap | 15,000 chars | tool execution |
| Conversation file per-step cap | 12 KB (persona gather) | `_gather_trillian_conversations` |
| Persona line growth gate | 2% | `_update_trillian_persona` |
| Persona char growth gate | 5% | `_update_trillian_persona` |
| Log buffer | 2000 → prune 1000 | `web.py` module globals |
| Event buffer | 500 → prune 250 | `web.py` module globals |

---

## Environment Variables

| Variable | Default | Used in |
|----------|---------|---------|
| `AWS_REGION` | `us-west-2` | Both |
| `CARL_MODEL` | `us.anthropic.claude-opus-5` (auto-probed) | Both |
| `CARL_REPORT_MODEL` | `us.anthropic.claude-sonnet-5` | Both + `report.py` CLI |
| `CARL_TRILLIAN_MODEL` | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | Both |
| `CARL_PERSONA_UPDATE_MODEL` | `us.anthropic.claude-sonnet-5` | Both |
| `CARL_PORT` | `5001` | web.py |
| `SEMANTIC_SCHOLAR_API_KEY` | `None` (optional) | literature.py |

---

## Session Directory Layout

```
sessions/<slug_YYYYMMDD-HHMM>/
    memory.md                    # Accumulated step summaries + persona update log
    web.log                      # Tee of all terminal output (CLI and web); resume authority
    references/
        <user_file>.md           # Copies of reference files from Step 2
        literature_search.md     # Formatted paper list (Step 2.5)
        literature_search.json   # Raw paper data (Step 2.5)
        <suggest_slug>.md        # Problem spec from /suggest
    code/
        environment.py           # Generated RL environment
        agent.py                 # Generated agent
        train.py                 # Generated training script
        models.py                # Generated neural networks
        config.py                # Generated config/hyperparameters
        make_artifact.py         # Generated artifact-bundle builder (Step 9)
        long_train.sh            # Generated long-run script; refreshes artifact/ at the end
        requirements.txt         # Generated dependencies
        .bg_pids.json            # {log_file: pid} for active background processes
        *.log                    # Training phase logs
        experiments/<run>/       # Per-run outputs, including *.png plots
        artifact/                # Self-contained shippable bundle (Step 9)
            best_model.pt         # Copied, not referenced
            contract.json         # obs/action contract, config, metric, seed, schema
            solve.py              # Entry point: --instance / --param k=v / --out
            smoke_test.py         # Reproduces the recorded metric; must pass
            MODEL_CARD.md         # Validity domain, baselines, failure modes
        revisit_1/               # Step 10 self-contained copied code workspace
            revisit_train.sh      # Fresh-start next-run script; no --init-from
            revisit_appendix.py   # Updates report appendix after user runs script
            experiments/revisit_1/
            artifact/             # Refreshed by revisit_train.sh when make_artifact.py copied in
        revisit_2/               # Later revisits never overwrite previous ones
    results/
        *.png                    # Generated plots
        *.csv                    # Metrics outputs
        appendix.tex             # Step 10 revisit appendix
    step_01_conversation.md
    step_02_conversation.md
    step_02.5_conversation.md
    step_03_conversation.md
    ...
    step_09_conversation.md
    step_10_conversation.md      # Present after --revisit
    results/                     # Also holds LaTeX report files if --write-report
        main.tex
        main.pdf
        abstract.tex, introduction.tex, ...
        references.bib
```

---

## Data Flow Diagram

```
User input
    │
    ▼ (chat.py) _prompt_user() / (web.py) POST /api/message
    │
    ├─ /suggest → _handle_suggest() / _handle_suggest_v2() → literature backends + LLM → refs/<slug>.md
    │                                  sets _suggest_used = True
    ├─ /next, /skip → summarize → session.save_step() → return "next"
    ├─ file path tokens → _inject_file_content() → append file contents
    │
    ▼ append to messages
    │
    ▼ _stream_response() / _stream_full()
    │   │
    │   ├─ _sanitize_messages()
    │   │
    │   ▼ _stream_single_turn() → bedrock_converse.converse_stream()
    │       │                       → client.converse_stream() [Converse API]
    │       │                       (cachePoint on system + tools, TTL=1h)
    │       │
    │       ├─ text blocks → print to terminal / _emit("text")
    │       └─ tool_use blocks → _execute_tool() → tool_result
    │           │
    │           ├─ write_file → session/code/<path>
    │           ├─ run_command → subprocess.run() → stdout/stderr
    │           ├─ run_background → Popen → .bg_pids.json
    │           ├─ check_process → read .bg_pids.json + log tail
    │           └─ search_literature → arXiv / OpenAlex / Semantic Scholar
    │                                  (merged; max 8 searches per step)
    │
    ▼ response_text
    │
    ├─ _asks_to_proceed() → True → summarize → session.save_step() → next step
    ├─ [DONE] in text → break autonomous loop → summarize → next step
    ├─ step 7 → one-turn auto-advance
    │
    ▼ _get_trillian_suggestion() → bedrock_converse.converse_text() [Haiku, no cache] → pre-fill readline
    │
    ▼ (all steps done)
    │
    ▼ _update_trillian_persona()
        │
        ├─ _gather_trillian_conversations() → step 1-6 conversation files
        ├─ _persona_triage() → bedrock_converse.converse_text() [Sonnet, no cache] → YES/NO
        ├─ _persona_focused_update() → bedrock_converse.converse_text() [Sonnet, no cache] → updated file
        ├─ size gate (2% lines, 5% chars)
        ├─ _persona_consistency_check() → bedrock_converse.converse_text() [Sonnet, no cache] → final file
        ├─ size gate again
        └─ write trillian-persona/SKILL.md (backup → SKILL.md.bak first)
```

Step 10 uses a shorter alternate flow:

```
--revisit sessions/<session>
    -> resolve_revisit_session()
    -> Session.open_existing()
    -> build_revisit_context()
    -> run REVISIT_STEP with REVISIT_TOOLS
    -> prepare_revisit_workspace(source_dir) after user confirms the copy source
    -> save Step 10 summary + step_10_conversation.md
    -> finish after user confirms "Ready to finish Step 10?"
```
