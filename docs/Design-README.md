# CARL -- Chat Assisted Reinforcement Learning: Design Overview

CARL is a conversational assistant that guides users through designing and
building reinforcement learning systems. It walks through a normal 9-step
workflow (plus a literature search sub-step), asking focused questions at each
stage, accumulating decisions, and ultimately generating runnable code. A
separate opt-in Step 10 revisit mode can reopen a completed session for
post-training analysis and next-run preparation. CARL offers two interfaces:
a terminal CLI and a lightweight web UI with a split-pane layout.

## Technology Stack

### Frontend

| Category | Technology | Notes |
|---|---|---|
| Language | Vanilla JavaScript | No React, Vue, or Angular; CSS/JS inline in `index.html` |
| Markup/Style | HTML5, CSS3 | Light theme (default); dark variant via `index-dark.html`. Flexbox, CSS variables |
| Markdown rendering | marked.js v12.0.1 | CDN-hosted |
| Syntax highlighting | highlight.js v11.9.0 | CDN-hosted, Python support |
| Math rendering | MathJax 3 | CDN-hosted (jsDelivr), TeX/MML → CHTML |
| Update mechanism | Polling (500ms) | Hits `/api/logs` and `/api/events` |
| Build tooling | None | No npm, Node.js, Webpack, or Vite |

### Backend

| Category | Technology | Notes |
|---|---|---|
| Language | Python 3 | Primary backend language |
| Web framework | Flask v3.0+ | Lightweight, single-user |
| LLM service | AWS Bedrock Converse (boto3) | Claude Opus 5 (fallback: Sonnet 5), Sonnet 5 (reports, persona), Haiku 4.5 (Trillian) |
| External APIs | arXiv, OpenAlex, Semantic Scholar | Academic paper search (tried in order, results merged) |
| Templating | Jinja2 | LaTeX report generation |
| Concurrency | threading + queue.Queue | Thread-safe event/log buffers |
| Process management | subprocess | Background training job execution |

**ML/RL libraries** (used in generated session code): PyTorch, Gymnasium, NumPy, Pandas, Matplotlib, TensorBoard

### Infrastructure

| Aspect | Detail |
|---|---|
| Deployment | AWS SageMaker |
| LLM service | AWS Bedrock |
| Storage | File system only — Markdown + JSON, no database |
| Default port | 5001 (override via `CARL_PORT`) |
| Config | `.env` file for API keys and model selection |
| CI/CD | None |

---

## Quickstart

```bash
pip install -r requirements.txt
```

**Terminal CLI:**
```bash
python carl.py
python carl.py --revisit sessions/<session_dir>
python carl.py --resume sessions/<session_dir>
```

**Web UI:**
```bash
python web.py                    # http://localhost:5001
CARL_PORT=8080 python web.py     # custom port
python web.py --revisit sessions/<session_dir>
python web.py --resume sessions/<session_dir>
```

Both interfaces require AWS Bedrock credentials (instance role or env vars).
Optionally create a `.env` file with `AWS_REGION`, `CARL_MODEL`,
`CARL_TRILLIAN_MODEL`, and `CARL_PERSONA_UPDATE_MODEL` overrides.

Both `carl.py` and `web.py` accept `--no-persona-update` to skip the
post-session Trillian evolution pipeline.
Both also accept `--write-report` to generate the optional LaTeX/PDF report
after a completed run.

`--revisit` opens an existing session directly in Step 10. It does not create a
new session directory and it skips normal post-session Trillian persona updates
and report generation. Step 10 writes its own conversation and memory summary
back into the existing session.

`--resume` reopens an existing session and continues the normal workflow from
its last unfinished step, restoring prior step decisions. It requires the
session's `memory.md` and `web.log`, and refuses sessions that already completed
(use `--revisit` for those). `--resume` and `--revisit` are mutually exclusive.
See [Session resume](#session-resume) for details.

**Tests:**
```bash
python -m pytest tests/ -q
```

Pure-Python unit tests over the controller-side logic — skill loading and format,
the shared tool runtime (Step 10 refusal filter, training-budget parsing, path
normalization), plot discovery, memory parsing, resume detection, and the artifact
plumbing. No AWS calls, so the suite runs in well under a second.

## Architecture

```
carl.py          Terminal entry point -- loads .env, launches CARLChat
chat.py          Terminal chat engine -- LLM calls, streaming, tool execution
web.py           Web entry point -- Flask app with polling-based UI
bedrock_converse.py  Shared Bedrock Converse request/message/tool helper
bedrock_models.py    Model-ID defaults + startup model resolution (probe/fallback)
tool_runtime.py  Shared tool schemas, tool-loop limits, Step 10 refusal filter,
                 Step 7 budget parsing, /suggest templates
command_compat.py    Shared `tail` shim and code/ path resolution for run_command
conversation_guards.py  Shared "did the turn end with a question?" guards
resume.py        Session resume: log markers, resume detection, web.log tee
literature.py    Multi-backend paper search + literature result persistence
report.py        Optional LaTeX/PDF report generator
revisit.py       Step 10 artifact discovery, revisit workspace, appendix helper
revisit_report.py    Step 10 revisit LaTeX appendix publisher
templates/
  index.html     Single-page frontend -- split pane, marked.js + highlight.js
                 (currently a copy of index-light.html -- the served default)
  index-light.html  Light theme variant
  index-dark.html   Dark theme variant (Tokyo-Night palette)
  step10.html    Revisit-mode wrapper around the normal frontend
steps.py         Step definitions, skill loading, system prompt construction
session.py       Session directory management and memory persistence
carl_skills/     One skill directory per step (<name>/SKILL.md) -- the domain
                 knowledge CARL draws on
jit_skills/      Reference RL skills (<name>/SKILL.md) the carl_skills distil
trillian_skills/ trillian-persona/SKILL.md -- Trillian persona (user-simulator)
tests/           pytest suite -- `python -m pytest tests/ -q`
```

`chat.py` and `web.py` are near-duplicate controllers around the same tool loop,
so the pieces that must not drift between them live in three shared modules:
`tool_runtime.py` (tool schemas, loop limits, the Step 10 retraining refusal, the
Step 7 training-budget parse, `/suggest` prompt templates), `command_compat.py`,
and `conversation_guards.py`. Both engines import from `tool_runtime.py` and keep
`_`-prefixed aliases so existing call sites read unchanged; a behavioural fix
lands once instead of twice.

### Runtime flow

```
carl.py  -->  CARLChat.run()
                |
                +-- for each step in STEPS:
                      |
                      +-- load_skill()           # read carl_skills/<skill>/SKILL.md
                      +-- build_system_prompt()  # inject skill body + prior decisions
                      +-- _run_step()            # conversation loop for this step
                            |
                            +-- _stream_response()   # LLM call with tool-use loop
                            +-- _get_trillian_suggestion()  # user-simulator
                            +-- _summarize()         # bullet-point summary on exit
                            +-- session.save_step()  # persist to memory.md

Revisit mode:

carl.py --revisit sessions/<session>
                |
                +-- Session.open_existing()
                +-- run REVISIT_STEP only
                      |
                      +-- build_revisit_context()    # memory, step 8/9 logs, artifacts
                      +-- REVISIT_TOOLS              # inspect, read, describe, prepare
                      +-- session.save_step(10, ...) # step_10_conversation.md + memory

Resume mode:

carl.py --resume sessions/<session>
                |
                +-- build_resume_info()          # parse web.log markers + memory.md
                +-- Session.open_existing()       # restore prior step_decisions
                +-- run STEPS from resume step_idx forward (normal workflow)
```

## The Steps

| # | Name | Skill | What happens |
|---|------|-------|--------------|
| 1 | Problem Definition | `01-problem-definition` | Understand the user's domain, objective, dynamics, constraints. File paths in user input are auto-read. |
| 2 | References | `02-references` | Ingest markdown/text reference files from user; extract key info. File paths in user input are auto-read. |
| 2.5 | Literature Search | `025-literature-search` | Search arXiv/OpenAlex/Semantic Scholar via `search_literature` tool (max 8 searches per step); extract RL-formulation details. |
| 3 | Parameter Exploration | `03-parameter-exploration` | User confirms problem parameter ranges for domain randomization |
| 4 | Information Proposal | `04-information-proposal` | Design state variables, features, normalization |
| 5 | Action Definition | `05-action-definition` | Define action space, bounded distributions, algorithm selection |
| 6 | Reward Definition | `06-reward-definition` | Design reward function, shaping, domain randomization strategy |
| 7 | ML/RL Parameters | `07-ml-parameters` | CARL presents auto-selected ML/RL hyperparameters; user approves or adjusts (one turn), and records the training budget as a `TRAINING_BUDGET_MINUTES: N` line |
| 8 | Initial Testing | `08-initial-testing` | Write and execute code: dry runs, supervised pre-training, tests |
| 9 | Launch | `09-launch` | Full training pipeline, phased training, visualization, and the shippable `artifact/` bundle |
| 10 | Post-Training Analysis | `10-post-training-analysis` | Opt-in revisit mode only: inspect final artifacts, describe every plot, diagnose training, prepare a fresh next-run workspace |

Each name is a directory under `carl_skills/` holding one `SKILL.md`.

Steps 1-7 are **conversational** -- CARL asks questions and records decisions.
Step 7 is a **one-turn step**: CARL presents ML/RL hyperparameters, the user
approves or suggests changes in a single response, then CARL proceeds.
Step 7 is the **last interactive step**. After step 7, CARL enters
**autonomous mode**: steps 8 and 9 run sequentially without further user input.
The web UI disables the input bar and shows progress only in the Thinking pane.

Steps 8 and 9 have **tool use** -- CARL writes files and runs commands autonomously.
Step 2.5 also has tool use (literature search only).

Step 10 is **not** appended to the normal workflow. It is launched only with
`--revisit sessions/<session_dir>`, stays interactive, and closes only after
the user explicitly confirms the Step 10 closeout.

## Key Design Decisions

### One skill per step

Each step loads only its own `carl_skills/<skill-name>/SKILL.md` into the system
prompt. Skills are never loaded all at once. This keeps the context window
focused and prevents the model from jumping ahead.

All three skill trees (`carl_skills/`, `jit_skills/`, `trillian_skills/`) follow
the Anthropic standard skill layout: one directory per
skill, each holding a single `SKILL.md` whose YAML frontmatter carries `name`
(equal to the directory name) and `description`. Non-standard keys are nested
under `metadata:` — the `carl_skills` record which `jit_skills` sections they were
distilled from as `metadata.sources`. Only the skill **body** is injected: the
loader strips the frontmatter, since the name/description/metadata keys are for
tooling and humans and the body is re-sent on every tool round.

`carl_skills/` holds the 11 step skills. `jit_skills/` holds the eight reference
RL skills they draw on (`rl-repository-setup`, `rl-environment-design`,
`rl-agent-implementation`, `rl-training`, `rl-model-management`,
`rl-visualization`, `rl-testing`, `rl-problem-workflow`) — indexed in
[`docs/rl-skills-index.md`](rl-skills-index.md). Pre-standard flat filenames
(`carl_skills/09_launch.md`, `trillian_skills/Trillian-Persona.md`) still resolve
as a logged fallback so a half-migrated checkout runs.

### Compact memory via decision summaries

When a step completes, CARL generates a bullet-point summary of the decisions
made. Only these summaries are carried forward into subsequent steps' system
prompts (via `DECISIONS FROM PREVIOUS STEPS`). Full conversation history is
saved to disk but not re-injected, keeping token usage bounded.

### Trillian: user simulator

For steps 1-6, after every CARL response, a second model call (Trillian, using
Claude Haiku) generates a suggested user reply based on a domain-expert persona
loaded from `trillian_skills/trillian-persona/SKILL.md`. The suggestion is shown
as an editable prefill:

- **Accept** it by sending the prefilled text unchanged; the suggestion is sent
  to CARL as-is and logged as accepted.
- **Edit or replace** it by changing the prefilled text; the final text is sent
  directly to CARL and logged as edited.
- **Ignore** it by replacing it with unrelated text or by continuing when no
  suggestion was available.

Trillian is deactivated from step 7 onwards, since those steps don't require
the same kind of iterative domain knowledge input.

### Trillian persona evolution

When a normal session completes all 9 steps, CARL automatically runs a three-step
pipeline (using Claude Sonnet) to evaluate whether the Trillian persona file
should be updated based on observed interactions:

- **Step A — Triage**: Sonnet reviews the step 1-6 conversation logs against
  the current persona spec and decides yes/no whether a substantial update is
  warranted. It looks for systematic behavioral deviations, entirely new
  interaction types, or consistent quantitative feedback the persona doesn't yet
  encode. Minor variations and one-off responses are ignored. If no update is
  needed, the pipeline stops here.
- **Step B — Focused update**: If Step A says yes, Sonnet produces an updated
  version of the persona file. The prompt enforces strict constraints: surgical
  modifications only, replace-don't-append (new examples replace old ones),
  slow evolution (only codify clearly consistent patterns), and a hard line-count
  limit (at most 2% growth over the original). The triage explanation is passed
  in to focus the edit.
- **Step C — Consistency check**: The updated file is reviewed for internal
  contradictions (between sections, within sections, or ambiguous instructions
  after editing). Any issues are resolved with minimal edits.

A size gate is applied after both Step B and Step C: if the output exceeds the
original file by more than 2% in lines or 5% in characters, the update is
rejected and the original is preserved. The output must also begin with the
expected `# Trillian:` header or it is rejected as malformed.

Before writing, the original file is backed up alongside itself as
`trillian-persona/SKILL.md.bak`.
The web flow also saves a pre-update copy in the session's `refs/` directory.
The outcome (no update / lines changed) is logged to the session's `memory.md`.
This entire post-session pass is non-critical: any failure prints a warning and
exits silently without touching the persona file or session data.

### Conversation log format

Each step's full conversation is saved to `step_NN_conversation.md`. CARL
turns are labeled `**CARL:**`. User turns are labeled based on how the
response was produced:

- **`**User:**`** -- the user typed this directly (no Trillian suggestion)
- **`**Trillian suggestion (accepted by user):**`** -- user accepted Trillian's
  suggested response without modification
- **`**Trillian suggestion:**` / `**User (edited and sent to CARL):**`**
  -- user edited or replaced the suggestion before sending

### Step confirmation gates

CARL ends each step by asking "Ready to proceed to Step N?" The system detects
this phrase and, if the user responds with a short affirmative ("yes", "ok",
"go", "proceed", etc.), automatically advances to the next step after saving.

Step 10 uses a separate closeout gate: CARL must ask exactly
"Ready to finish Step 10?" and the controller closes the revisit only after a
short affirmative or explicit finish command such as `/done`, `/finish`, or
`/end`. Step 10 does not use the autonomous `[DONE]` marker.

As a safety net, if the model starts outputting tool-call text (e.g. `<tool_call>`
or `create_file` JSON) in a non-tool step, the system intercepts it and prompts
the user to confirm advancing to the next step rather than letting file contents
spill into the chat. This handles cases where the model believes the step is
complete and jumps ahead to coding before the framework has transitioned.

### Streaming with tool-use loop

LLM responses are streamed token-by-token. When the model emits tool-use blocks
(`write_file`, `run_command`, etc.), the system:

1. Executes the tool immediately
2. Sends the result back as a `tool_result` message
3. Lets the model continue (it may call more tools or produce text)
4. Repeats for up to `MAX_TOOL_ROUNDS` (80, defined in `tool_runtime.py`) rounds
   per response

### Autonomous execution (Steps 8, 9)

After step 7 (ML/RL Parameters) is confirmed, CARL enters autonomous mode.
Steps 8 and 9 run sequentially without any user input. The web frontend disables
the input bar and replaces it with an interjection box ("Send to guide CARL").
Two mechanisms keep the model working:

- **Inner auto-continue**: If the model outputs narration text between tool
  calls (e.g. "Let me write...", "Now I'll run..."), the system checks for
  continuation keywords in the tail of the text (or empty/short output) and
  injects `[continue]` to nudge it back into tool calls.
- **max_tokens recovery**: If the model is cut off mid-response (stop reason
  `max_tokens`), the system automatically injects `[continue — you were cut off]`
  so the model resumes without losing context.
- **Outer auto-loop**: After `_stream_full` returns, if CARL hasn't asked to
  proceed, the system injects another `[continue]` and calls `_stream_full`
  again. This repeats for up to `_MAX_AUTO_ROUNDS` (20) iterations.

Users can still send messages during autonomous execution via the interjection
box (web UI) or stdin (terminal). These are injected into the message stream
and CARL responds before continuing.

### Tool call discipline (Steps 8, 9)

Three rules in the autonomous steps' system prompt keep the run auditable and the
session state single-writer:

- **Edits go through `write_file` only.** No `sed -i`, `python -c` heredocs,
  `cat > file << EOF`, or `>>` appends to source. The saved conversation record
  truncates tool input, so a shell patch is unrecoverable from the transcript and
  the next session cannot see what changed; rewriting the whole file is the
  cheaper option anyway. The `write_file` and `run_command` tool descriptions
  state this too.
- **`memory.md` is the controller's file.** CARL never writes to it — a second
  copy produces duplicate `## Step N` blocks. Run notes and tables go in files
  under `code/` or `results/`; decisions go in the step summary, which the
  controller records. (Resume parsing was also hardened to concatenate rather
  than overwrite duplicate step headings, so historic duplicates lose nothing.)
- **`code/` is always the working directory.** `run_command` and `run_background`
  start there, so paths are relative to it (`train.py`, `experiments/run1/`),
  never prefixed with `code/` and never preceded by a `cd`. A redundant leading
  `code/` is stripped rather than creating `code/code/`.

### Step 10 revisit mode

Step 10 reopens an existing completed session for post-training analysis:

```bash
python carl.py --revisit sessions/<session_dir>
python web.py --revisit sessions/<session_dir>
```

Normal CARL remains unchanged and still ends after Step 9, the optional
Trillian update, and optional report generation. Revisit mode uses
`Session.open_existing()` and does not create, rename, or overwrite the session
directory. It saves `step_10_conversation.md` and appends a Step 10 summary to
`memory.md`.

The Step 10 prompt is built from the Step 10 skill plus a generated revisit
context containing `memory.md`, Step 8 and Step 9 conversations, prior Step 10
conversations, artifact inventory, training logs, plots, result `.tex` files,
JSON metrics, long-run outputs, and existing revisit artifacts.

The skill guides CARL through result inventory, eval progression, signal/noise
and batch adequacy, reward alignment, policy behavior, checkpoint diagnostics,
fresh initialization, and training-throughput checks. When larger batches or
longer runs are recommended, Step 10 should also check whether parallel or
vectorized episode rollouts can reduce per-update wall time without changing
seeding, CRN/blocking, terminal handling, or reward accounting.

**Every plot must be looked at.** A mandatory `Visual Evidence Review` section
requires `describe_plot` on *every* entry the artifact inventory returns, before
any recommendation is proposed: the plots list is a worklist, not a sample. An
empty or implausibly short inventory is itself a finding to investigate, not a
reason to skip the section. Each recommendation in the ranked MDP-revision list
must cite the specific plots and/or log lines that motivate it, or be downgraded
to "needs instrumentation".

For that to be sound the inventory has to be complete, so plot discovery is
shared: both the Step 10 inventory and the PDF report go through one function, and
plots get their own recursive pass over `code/` and `results/`. (The
source-file walk deliberately prunes `experiments/`, `logs/`, and `checkpoints/`
as noise — but plots are usually written under `code/experiments/<run>/`, so
reusing that walk hid most of them.) Jupyter `.ipynb_checkpoints` autosave
duplicates are filtered by name from the plots, result `.tex` files, and JSON
metrics alike.

Step 10 exposes a smaller revisit tool set:

- inspect artifacts and read text files in the existing session
- describe PNG plots with the vision model
- create a fresh self-contained `code/revisit_N/` workspace
- write only inside that new revisit workspace
- run short inspection or validation commands

Step 10 deliberately does **not** expose `run_background`, and both CLI and web
refuse training commands such as `python train.py`, `long_train.sh`, and
`revisit_train.sh`. The step prepares the next run; the user launches it later.

The refusal filter (in `tool_runtime.py`) inspects the *executed program* of each
pipeline segment rather than matching substrings of the command line. The original
substring test refused read-only work that merely mentioned a script name —
`grep train.py train.log`, `chmod +x revisit_train.sh`, `ls long_train.sh` — 19
such refusals across three observed sessions. The current filter tokenizes the
command, splits it on shell operators, and unwraps shells, interpreters, and
wrappers (`nohup`, `timeout`, `env`, …) to reach the real program, so inspection
and metadata commands are allowed while `bash revisit_train.sh`,
`nohup python train.py`, `python -m train`, and `RUN=x bash long_train.sh` are
still refused. Unparseable command lines fall back to the conservative substring
check.

Each revisit workspace is independent. Before `prepare_revisit_workspace` is
called, the agent recommends a session-local source directory and gets explicit
user confirmation. The helper copies that confirmed source into
`code/revisit_N/`; this source may itself be an earlier `code/revisit_M/`
workspace. The copy excludes generated outputs, checkpoints, logs, caches, and
nested revisit directories. It writes a `revisit_train.sh` and
`revisit_appendix.py` into that same workspace. The training script starts from
fresh initialization by default and refuses `TRAIN_CMD` overrides containing
`--init-from`, so Step 9 pretrain files, long-run init files, best/latest
checkpoints, and prior revisit init files are diagnostic baselines only.

`revisit_train.sh` writes outputs under its own `experiments/revisit_N/`
directory, then runs `revisit_appendix.py`, then refreshes the `artifact/` bundle
from the new run's best checkpoint when the workspace carries `make_artifact.py`.
The appendix helper writes or updates `results/appendix.tex`, inserts
`\input{appendix.tex}` into `results/main.tex` once when present, and runs
`pdflatex` if available. If LaTeX compilation fails, the `.tex` files are kept and
the failure is printed. When a refreshed bundle exists, the appendix gains a
`Shippable Artifact` subsection recording its path, how to invoke `solve.py`, the
contract facts (observation/action dimensions, selection metric, seed), and
whether the bundled smoke test passed.

### Session resume

`--resume sessions/<session_dir>` reopens an existing normal session and
continues the standard workflow from its last unfinished step. It is
implemented in `resume.py` and is mutually exclusive with `--revisit`.

Resume authority is the session's `web.log`. As CARL runs, it emits structured
markers via `emit_resume_marker()`:

- `CARL_RESUME step_start number=N` — a step began
- `CARL_RESUME step_saved number=N` — a step's summary was saved
- `CARL_RESUME session_done` — the full workflow completed

`build_resume_info()` parses these markers to find the first unfinished step. If
no markers are present (older sessions), it falls back to a legacy log scan
(`## Step N:` headings, "Ready to proceed to Step N", `[DONE]`, and
interruption markers such as `Traceback` / `API error`). Prior step decisions
are re-parsed from `memory.md` (`parse_memory_decisions()`) so completed steps
feed the resumed step's system prompt as usual. A step heading appearing more than
once — `Session.save_step` writes its own, and CARL has additionally appended notes
during Steps 8-9 — has its bodies concatenated in document order rather than
overwritten, since keeping only the last block silently discarded the controller's
own summary. Skip-2.5 state (`/suggest` used, or `/skip` in Step 2) is
reconstructed from the log and restored.

Resume requires both `memory.md` and `web.log` under `sessions/`, and refuses a
session that already completed (it directs the user to `--revisit` instead). On
resume, the resumed step's system prompt carries a mode note telling the model
to restart the unfinished step from the top and reuse or repair any partial
artifacts already on disk. In the web UI, `--resume` surfaces a
`Resume mode: <dir>. Resuming Step N: name` banner and shows any resume warnings
as system messages; there is no session-picker UI — resume is triggered only by
the CLI flag.

### Theming (light / dark)

The web UI now ships a **light theme by default**. `templates/index.html` is a
copy of `templates/index-light.html` (the served default), and
`templates/index-dark.html` holds a dark variant using a Tokyo-Night palette and
`github-dark.min.css` (the light variant uses `github.min.css`). The themes
differ only in their CSS-variable values and the highlight.js stylesheet.

Theme selection is a **manual file swap** — copy `index-dark.html` over
`index.html` to serve dark. There is no runtime toggle, no `localStorage`
persistence, no `prefers-color-scheme` handling, and no theme env var or route.

### Extended thinking

LLM calls use Bedrock extended thinking, configured per step in
`bedrock_converse.py`. `thinking_config_for_step(model, step_num)` returns the
`additionalModelRequestFields` for a step's thinking tier from
`_STEP_THINKING_TIER`: step 1 is `medium`, steps 2–7 and 10 are `high`, and the
autonomous steps 8–9 are `extra_high`.

- **Claude 5 and Opus 4.7** (`_uses_adaptive_thinking`) use adaptive thinking:
  `{"thinking": {"type": "adaptive"}, "output_config": {"effort": ...}}` with
  effort mapped `medium`/`high`/`max`. These models reject
  `thinking.type=enabled`.
- **Haiku 4.5** uses an explicit budget:
  `{"thinking": {"type": "enabled", "budget_tokens": ...}}`
  (4000 / 8000 / 16000), and errors on `effort`. It is the only shipped default
  on this branch, which is why the branch stays.

`thinking_max_tokens_for_step` raises `max_tokens` to accommodate the thinking
budget plus output. Temperature is suppressed whenever thinking is enabled, and
always for Claude 5 and Opus 4.7, which deprecate it. Because Claude 5 also
thinks adaptively with *no* thinking config attached — spending `maxTokens` on
reasoning before emitting text — `_build_inference_config` additionally raises
any `maxTokens` below 256 for those models, so small helper calls cannot come
back empty. If a model rejects the thinking config, `_stream_single_turn`
detects the thinking error and retries the turn without it.

### Message sanitization

Before every API call, `_sanitize_messages` scans the conversation for broken
tool_use/tool_result pairs (caused by interrupted tool execution or stream
errors). If an assistant message with tool_use blocks isn't followed by matching
tool_results, the conversation is truncated at that point. This prevents Bedrock
validation errors while preserving the most conversation history possible.

### Response discipline

The system prompt instructs CARL to always end its turn with a question. After
summarizing information, it must follow up with a clarifying question or ask the
user to confirm/proceed. This ensures the user always knows what to do next.

### Crash recovery and graceful exit

The main loop is wrapped in `try/except/finally`. On any exit (clean quit,
Ctrl+C, or unexpected error):

1. All background processes are terminated (`SIGTERM` via PID tracking)
2. The current step's conversation is saved to disk
3. A summary is attempted (silently skipped if the API is unreachable)
4. A "session ended during Step N" marker is written to memory

During tool steps, conversations are auto-saved after each tool execution round
so mid-execution crashes don't lose progress.

`web.py` registers `SIGINT`/`SIGTERM` handlers that kill tracked background
processes (via `.bg_pids.json`) before exiting. The terminal CLI performs the
same cleanup from its `try/except/finally` shutdown path.

### Background training (Step 9)

Step 9 uses `run_background` for all training phases. Training scripts must
flush output frequently and save checkpoints. CARL monitors progress via
`check_process` and reports key metrics after each phase completes.

`check_process` has an adaptive wait when the process is still running:
it derives a polling interval from the training budget stated in the Step 7
summary, spreading polls across the available tool rounds. The minimum interval
is 60 seconds. This prevents the model from burning through tool rounds by
polling too frequently.

The budget is resolved in three tiers, and which tier fired is always logged so a
silent fall back is visible: Step 7 is required to emit a machine-readable
`TRAINING_BUDGET_MINUTES: N` line (integer minutes, on its own line, under 90,
written even when the user accepted the default); failing that, a free-text time
pattern is scanned out of the Step 7 summary; failing that, a 45-minute default
applies.

### The final artifact bundle (Step 9)

A checkpoint is not a deliverable. A finished run must ship a self-contained
`code/artifact/` bundle that solves a **new problem instance** — not just replay a
fixed eval grid — and that still works after being copied out of the session, so
the checkpoint is *copied* into it and nothing in it imports from the training
tree.

CARL writes a per-session `make_artifact.py` next to `train.py`, invoked as
`python make_artifact.py --run-name <run> [--ckpt PATH] [--out artifact/]`. It
resolves the best checkpoint (never a hard-coded run name or path), copies it to
`artifact/best_model.pt`, writes `artifact/contract.json`, and copies the minimal
runtime modules in. The bundle then contains:

| File | Role |
|------|------|
| `best_model.pt` | The copied checkpoint |
| `contract.json` | `obs_dim`, `action_dim` + bounds, observation field order and units, action semantics, normalization constants, resolved config, code fingerprint, selection metric + value, seed, one canonical instance, and the instance-parameter schema |
| `solve.py` | The entry point for someone who did not train the model: `--instance instance.json`, repeatable `--param k=v`, `--out result.json`, `--n-seeds`. Builds the policy from the contract, validates the instance against the schema and fails loudly rather than silently clipping, writes the decision trajectory and objective value |
| `smoke_test.py` | Loads the checkpoint, runs the canonical instance, asserts the objective reproduces the recorded metric within tolerance |
| `MODEL_CARD.md` | Problem statement, MDP, training recipe and budget, best-vs-latest-vs-baseline numbers, validity domain with an explicit out-of-domain warning, known failure modes, copy-paste usage |

Supporting this, every checkpoint must be saved with an `extra` dict carrying
`obs_dim`, `action_dim` (and bounds), a config snapshot, and the selection metric
— that dict is `make_artifact.py`'s source of truth, so no dimension is ever
hard-coded. Step 9 runs `artifact/smoke_test.py` in-session before the final
summary; a failure blocks shipping and must be fixed rather than reported around.
Both `long_train.sh` and `revisit_train.sh` end by re-running `make_artifact.py`,
so a longer or revised run refreshes the bundle from its own better checkpoint.

The bundle flows into the written output: the optional report's Results and
Discussion sections receive the model card and contract and must state the
bundle's path, the command that runs it, and the validity domain it is trusted
within; the Step 10 revisit appendix gains a `Shippable Artifact` subsection with
the same information plus the smoke-test outcome. Both degrade quietly for
sessions that predate the bundle.

### Server logging

All terminal output (streaming text, tool calls, results) is tee'd to `web.log`
in the session directory with ANSI color codes stripped, providing a persistent
record of CARL's activity. This is no longer web-only: the terminal CLI also
tees to `web.log` (via `SessionLogTee` in `resume.py`) for every normal run,
restarting the tee around the post-Step-1 session rename. `web.log` is the
authority the resume feature reads, so it is required by `--resume`. Tee-ing is
disabled only in Step 10 revisit mode.

## Session Directory Structure

Each run creates a named, timestamped directory:

```
sessions/
  merton_portfolio_20260306-1526/
    memory.md                      # Accumulated step summaries
    web.log                        # Tee'd terminal output (CLI and web); resume authority
    references/                    # Copies of user-provided reference files
    code/                          # All generated Python files
      experiments/<run>/           # Per-run training outputs and plots
      artifact/                    # Shippable bundle: best_model.pt, contract.json,
                                   # solve.py, smoke_test.py, MODEL_CARD.md
      revisit_1/                   # Optional Step 10 self-contained revisit
        revisit_train.sh           # Fresh-start next-run script
        revisit_appendix.py        # Updates results/appendix.tex after run
        experiments/revisit_1/     # Revisit outputs when user runs the script
        artifact/                  # Refreshed at the end of revisit_train.sh
      revisit_2/                   # Later revisits never overwrite earlier ones
    results/                       # Training outputs, plots, optional report files
      appendix.tex                 # Step 10 report appendix when revisits run
    step_01_conversation.md        # Full conversation log for step 1
    step_02_conversation.md        # ...
    step_02.5_conversation.md      # Literature search step
    ...
    step_10_conversation.md        # Revisit conversation when --revisit is used
```

The session name is an LLM-generated 2-3 word slug based on the problem
discussed in Step 1, followed by a compressed timestamp (`YYYYMMDD-HHMM`).

## LLM Backend

CARL uses **AWS Bedrock Converse** through the shared `bedrock_converse.py`
helper. Streaming turns use `client.converse_stream`; non-streaming helper calls
use `client.converse`. The default model is `us.anthropic.claude-opus-5`, with
automatic fallback to `us.anthropic.claude-sonnet-5` if Opus 5 is unavailable in
the region — a drop in capability tier, not a minor-version step, which the
fallback log says explicitly. Model-ID defaults and startup resolution live in
`bedrock_models.py` (`resolve_model_id`). At startup, a probe call determines
which model to use; `CARL_MODEL` overrides this entirely (no probe performed
when explicitly set). The resolver no longer falls back on *any* exception: it
raises a `RuntimeError` on credential or connection errors, falls back to
Sonnet 5 only on a `ClientError`, and **probes the fallback as well** before
returning it (raising if the fallback probe also fails).

Report writing and the post-session persona update run on Sonnet 5, and
Trillian on Haiku 4.5 — three models across four independently repinnable
roles.

`bedrock_converse.py` also centralizes cachePoint placement, message conversion,
tool schema conversion, token usage normalization, per-step extended-thinking
configuration, and the two model-generation gates (temperature suppression and
the adaptive-thinking token floor).

Configuration via environment variables (or `.env` file):

| Variable | Default | Purpose |
|----------|---------|---------|
| `AWS_REGION` | `us-west-2` | Bedrock region |
| `CARL_MODEL` | `us.anthropic.claude-opus-5` | Main model ID (auto-probed) |
| `CARL_REPORT_MODEL` | `us.anthropic.claude-sonnet-5` | Report writing model |
| `CARL_TRILLIAN_MODEL` | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | Trillian (user-simulator) model |
| `CARL_PERSONA_UPDATE_MODEL` | `us.anthropic.claude-sonnet-5` | Persona evolution model (post-session) |

## Tools (Steps 8, 9)

Both engines share one definition of these schemas, in `tool_runtime.py`.

| Tool | Description |
|------|-------------|
| `write_file(path, content)` | Write a file to `sessions/<slug_ts>/code/<path>`; the only approved way to create or edit a file. A redundant leading `code/` is stripped. |
| `run_command(command)` | Execute a shell command in `code/`, 10-min timeout. For short tasks. Not for editing files. |
| `run_background(command, log_file)` | Start a long-running command (training). Output streams to `code/<log_file>`. Returns immediately with PID. |
| `check_process(log_file, tail_lines)` | Check if a background process is still running and read the tail of its log. |

`run_background` + `check_process` handle training runs that can last minutes to
hours. CARL kicks off training, returns immediately, then polls the log file to
monitor progress. PIDs are tracked in `code/.bg_pids.json`.

## Revisit Tools (Step 10)

| Tool | Description |
|------|-------------|
| `list_revisit_artifacts()` | Return a categorized inventory of memory, conversations, train logs, plots, result `.tex` files, JSON metrics, long-run outputs, and revisit artifacts. |
| `read_artifact(path, max_chars)` | Read capped text artifacts from the existing session directory. |
| `describe_plot(path)` | Send a PNG plot to the vision model for factual trend/axis interpretation. Required on **every** plot in the inventory. |
| `prepare_revisit_workspace(source_dir)` | Create the next `code/revisit_N/` directory from the user-confirmed session-local source directory, and write `revisit_train.sh` plus `revisit_appendix.py`. |
| `write_file(path, content)` | In Step 10, writes are allowed only under `code/revisit_N/`. |
| `run_command(command)` | Short inspection/validation only. Step 10 refuses retraining commands. |

Step 10 intentionally has no `run_background` tool. The generated
`revisit_train.sh` starts from fresh initialization and is left for the user to
run after the Step 10 discussion.

## Literature Tool (Step 2.5)

| Tool | Description |
|------|-------------|
| `search_literature(query, limit)` | Search arXiv, OpenAlex and Semantic Scholar, merging results. Returns papers with titles, authors, years, citation counts, abstracts, URLs, and PDF URLs. Backend failures are reported separately from empty results, and the step is capped at `MAX_LITERATURE_SEARCHES` (8) calls. |

## User Commands

| Command | Effect |
|---------|--------|
| `/next`, `/skip` | Force-advance to the next step; `/skip` in Step 2 also marks Step 2.5 to be skipped |
| `/status` | Show current step number in the terminal CLI |
| `/done`, `/finish`, `/end`, `/complete` | Finish Step 10 revisit mode after the closeout is ready |
| `/quit`, `/exit` | Save partial progress and exit |

The web UI handles step status through `/api/status` rather than a typed
`/status` command.

## Web UI

The web UI is the primary frontend, available at `http://localhost:5001`.

### Split-pane layout

- **Left pane (Chat)**: User messages, rendered assistant messages (markdown
  with syntax highlighting), step banners, and summaries. Messages appear as
  complete rendered blocks when the model's turn finishes.
- **Right pane (Thinking)**: Real-time activity log showing streaming text
  tokens, tool calls with filenames/commands, tool results, and status messages
  (auto-continuing, max_tokens recovery). Color-coded: yellow for tools, green
  for results, cyan for status, red for errors.

### Polling architecture

The web UI uses **polling** instead of SSE/WebSocket to ensure reliable
operation behind SageMaker's proxy (which buffers streaming responses).

- `POST /api/message` — fires the generation in a background thread, returns
  immediately with `{"status": "ok"}`
- `GET /api/events?since=N` — returns chat events (`step`, `text`, `summary`,
  `trillian_suggestion`, `step_advance`, `autonomous_start`, `interjection`,
  `turn_end`, `info`, `error`, `done`, plus supplemental `log` events) and a
  `busy` flag indicating whether generation is still running
- `GET /api/logs?since=N` — returns terminal log entries for the Thinking pane
- `POST /api/interject` — injects a user message during autonomous execution
  (steps 8-9); queued and consumed by the autonomous loop between tool rounds

The frontend polls both endpoints every 500ms. When `busy` transitions from
`true` to `false`, the accumulated text is rendered as a markdown message in
the chat pane. The green dot in the Thinking pane header indicates activity.

### Features

- **Markdown rendering** via marked.js (headings, lists, tables, code blocks)
- **Syntax highlighting** via highlight.js (Python, shell, etc.)
- **Step banners and summaries** — visual separation between workflow steps
- **Trillian suggestions** — shown in the chat pane; user can accept, edit, or ignore
- **Auto-advance** — after confirming a step, the next step begins automatically
- **Autonomous mode** — After step 7, input switches to an interjection box;
  steps 8-9 run without required user input but accept optional guidance
- **Revisit mode** -- `web.py --revisit sessions/<session_dir>` serves
  `templates/step10.html`, labels the UI as Step 10, and shows
  `Revisit mode: <session_dir>` at session start
- **Resume mode** -- `web.py --resume sessions/<session_dir>` continues the
  normal workflow from the last unfinished step; the UI shows a
  `Resume mode: <dir>. Resuming Step N: name` banner and surfaces any resume
  warnings as system messages
- **SageMaker proxy support** — all API paths are relative to `window.location`
- **No timeout risk** — long-running tool steps (code generation, training)
  work reliably since no HTTP connection is held open during execution
