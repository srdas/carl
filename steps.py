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

"""Step definitions and system prompt construction for CARL.

Each step maps to exactly one skill in carl_skills/, stored in standard
skill format as <skill-name>/SKILL.md. Only the current step's skill is
loaded -- never all at once.
"""

import re
import sys
from pathlib import Path

STEPS = [
    {"number": 1,   "name": "Problem Definition",    "skill": "01-problem-definition"},
    {"number": 2,   "name": "References",             "skill": "02-references"},
    {"number": 2.5, "name": "Literature Search",      "skill": "025-literature-search"},
    {"number": 3,   "name": "Parameter Exploration",  "skill": "03-parameter-exploration"},
    {"number": 4, "name": "Information Proposal",   "skill": "04-information-proposal"},
    {"number": 5, "name": "Action Definition",      "skill": "05-action-definition"},
    {"number": 6, "name": "Reward Definition",      "skill": "06-reward-definition"},
    {"number": 7, "name": "ML/RL Parameters",       "skill": "07-ml-parameters"},
    {"number": 8, "name": "Initial Testing",        "skill": "08-initial-testing"},
    {"number": 9, "name": "Launch",                 "skill": "09-launch"},
]

REVISIT_STEP = {
    "number": 10,
    "name": "Post-Training Analysis",
    "skill": "10-post-training-analysis",
}

_FRONTMATTER_RE = re.compile(r"\A---\r?\n.*?\r?\n---[ \t]*\r?\n?", re.DOTALL)

# Flat pre-standard filenames, kept for one release so a half-migrated
# checkout or an older sessions/ replay does not hard-fail. Mapped in both
# directions: a caller may pass either the new directory name or the old
# `NN_name.md` filename.
_LEGACY_SKILL_FILES = {
    "01-problem-definition": "01_problem_definition.md",
    "02-references": "02_references.md",
    "025-literature-search": "025_literature_search.md",
    "03-parameter-exploration": "03_parameter_exploration.md",
    "04-information-proposal": "04_information_proposal.md",
    "05-action-definition": "05_action_definition.md",
    "06-reward-definition": "06_reward_definition.md",
    "07-ml-parameters": "07_ml_parameters.md",
    "08-initial-testing": "08_initial_testing.md",
    "09-launch": "09_launch.md",
    "10-post-training-analysis": "10_post_training_analysis.md",
}
_SKILL_DIR_BY_LEGACY_FILE = {v: k for k, v in _LEGACY_SKILL_FILES.items()}


def strip_frontmatter(text: str) -> str:
    """Drop a leading YAML frontmatter block.

    The name/description/metadata keys are for tooling and humans; the model
    only needs the body, which already opens with its own `# Step N:` heading.
    """
    return _FRONTMATTER_RE.sub("", text, count=1).lstrip("\n")


def load_skill(carl_dir: str, skill: str) -> str:
    """Load one skill body from carl_skills/<skill>/SKILL.md.

    `skill` is a skill directory name. A flat pre-standard filename (either
    the legacy `NN_name.md` form or the new directory name) is accepted as a
    fallback and logged, so a half-migrated tree still runs.
    """
    base = Path(carl_dir) / "carl_skills"
    key = _SKILL_DIR_BY_LEGACY_FILE.get(
        skill, skill[:-3] if skill.endswith(".md") else skill
    )
    canonical = base / key / "SKILL.md"
    if canonical.is_file():
        return strip_frontmatter(canonical.read_text(encoding="utf-8-sig"))

    for candidate in (base / skill, base / _LEGACY_SKILL_FILES.get(key, f"{key}.md")):
        if candidate.is_file():
            sys.stderr.write(
                f"[steps] skill {key!r}: {canonical} missing, falling back to "
                f"pre-standard flat file {candidate}\n"
            )
            return strip_frontmatter(candidate.read_text(encoding="utf-8-sig"))

    raise FileNotFoundError(f"Skill not found: {canonical}")


# ── Trillian (user-simulator) ───────────────────────────────────────────

_TRILLIAN_LAST_ACTIVE_STEP = 6  # Trillian is deactivated from step 7 onwards

_TRILLIAN_SYSTEM_TEMPLATE = """\
You are Trillian, a bot that mimics a domain-expert user interacting with an \
AI assistant called CARL. You are generating a suggested response that the \
real user can accept, edit, or replace.

=== PERSONA ===
{persona}
=== END PERSONA ===

CURRENT CONTEXT:
- CARL is on Step {step_num}: {step_name}.
- The conversation so far is shown in the messages.
- CARL's latest message is the last assistant message.
- You must generate ONLY the suggested user reply -- no preamble, no \
  explanation, no surrounding quotes.
- If CARL is greeting the user or asking them to describe their problem \
  (i.e. this is the very first exchange in Step 1), respond with exactly: \
  [SKIP]
- If CARL asks "Ready to proceed?" or presents a summary/checklist for \
  confirmation, respond with: ok
"""



TRILLIAN_PERSONA_PATH = Path("trillian_skills") / "trillian-persona" / "SKILL.md"
_LEGACY_TRILLIAN_PERSONA_PATH = Path("trillian_skills") / "Trillian-Persona.md"


def trillian_persona_path(carl_dir: str | Path) -> Path:
    """Resolve the Trillian persona file, tolerating the pre-standard layout."""
    path = Path(carl_dir) / TRILLIAN_PERSONA_PATH
    if path.is_file():
        return path
    legacy = Path(carl_dir) / _LEGACY_TRILLIAN_PERSONA_PATH
    if legacy.is_file():
        sys.stderr.write(
            f"[steps] persona: {path} missing, falling back to {legacy}\n"
        )
        return legacy
    return path


def load_trillian_persona(carl_dir: str) -> str:
    """Load the Trillian persona file."""
    return trillian_persona_path(carl_dir).read_text(encoding="utf-8-sig")


def build_trillian_prompt(persona: str, step_num: int, step_name: str) -> str:
    """Build the system prompt for a Trillian suggestion call."""
    return _TRILLIAN_SYSTEM_TEMPLATE.format(
        persona=persona, step_num=step_num, step_name=step_name,
    )


def build_system_prompt(
    step: dict,
    skill_content: str,
    previous_decisions: str,
    skip_steps: set | None = None,
    workflow_steps: list[dict] | None = None,
    mode_note: str | None = None,
    budget_minutes: int | None = None,
) -> str:
    """Build the system prompt for the current step."""
    step_num = step["number"]
    workflow = workflow_steps or STEPS
    total = len(workflow)
    step_idx = next(
        (i for i, s in enumerate(workflow) if s["number"] == step_num),
        0,
    )

    # Find the next step that isn't being skipped
    nxt = None
    for j in range(step_idx + 1, total):
        if skip_steps and workflow[j]["number"] in skip_steps:
            continue
        nxt = workflow[j]
        break

    if step_num == 10:
        proceed_line = (
            'When the Step 10 postmortem and revisit workspace are complete, '
            'provide a concise final summary and ask exactly:\n'
            '"Ready to finish Step 10?"'
        )
    elif nxt is not None:
        proceed_line = (
            f'When you have enough information, provide a clear summary of '
            f'the decisions made in this step, then ask:\n'
            f'"Ready to proceed to Step {nxt["number"]}: {nxt["name"]}?"'
        )
    else:
        proceed_line = (
            "This is the final step. When the design is complete, "
            "summarize the full system and confirm with the user."
        )

    if workflow_steps is None:
        step_line = f"You are on Step {step_num} of 9: {step['name']}."
    else:
        step_line = f"You are on Step {step_num}: {step['name']}."

    prompt = f"""\
You are CARL (Chat Assisted RL), an expert assistant guiding a user
through designing and building a reinforcement learning system.

{step_line}
{mode_note or ""}

=== SKILL REFERENCE (this step only) ===
{skill_content}
=== END SKILL REFERENCE ===

=== DECISIONS FROM PREVIOUS STEPS ===
{previous_decisions}
=== END PREVIOUS DECISIONS ===

GUIDELINES:
- No sycophancy. Do not open with "Great", "Perfect", "Excellent choice",
  "This is really helpful", or similar filler praise. Acknowledge the user's
  input by acting on it, not by complimenting it.
- Ask at most 2-3 focused questions. Fewer if the user's input is clear.
- Do not re-ask things already covered in previous decisions.
- Stay on this step's topic. Do not discuss future steps.
- {proceed_line}
- Keep responses concise. Use bullet points for lists and summaries.
- ALWAYS end your response with a question. Never end with a summary alone.
  After summarizing information, follow up with a clarifying question or
  ask the user to confirm / proceed. The user should always know what to do next.
"""

    if step_num == 2:
        prompt += f"""
REFERENCES STEP -- SPECIAL INSTRUCTIONS:
- First check PREVIOUS DECISIONS for any reference file paths already named
  by the user (e.g. paths like refs/*.md). If found, ask the user to share
  the file content by typing the path, rather than asking if they have one.
- Otherwise, ask the user if they have reference files (markdown or text)
  to share. The user can type a relative or absolute file path. The system
  reads it automatically and appends the content to their message.
- After file content appears, extract: analytical solutions, parameter
  ranges, reward formulations, physical relationships, and constraints.
- Ask if there are more files after processing each one.
- If the user has no references, note that. Do NOT conduct a literature
  search yourself -- that is Step {nxt["number"]}: {nxt["name"]} and will
  happen in the next step. Use the exact proceed question from the
  GUIDELINES above.
"""

    if step_num == 2.5:
        prompt += """
LITERATURE SEARCH -- TOOL INSTRUCTIONS:
You have one tool available: search_literature(query, limit).
- Use it to search academic literature (arXiv, OpenAlex, Semantic Scholar).
- Construct queries from the problem keywords and domain terms. Terms are
  ANDed, so 4-6 distinctive keywords beat a long descriptive phrase.
- You have a HARD BUDGET of 8 searches for this step, after which the tool
  refuses to run. Plan them: each query should cover a distinct sub-topic
  (e.g. the algorithm, the domain, the benchmark), not reword the last one.
  Three or four well-aimed searches are usually enough.
- If a result reports backend failures, the APIs were unreachable -- that is
  not evidence the literature is thin, and rephrasing will not help. Say so
  and either retry the same query or move on.
- Present results to the user and ask which papers are relevant.
- Extract ONLY RL-formulation details: state representations, action spaces,
  reward functions, parameter ranges, and analytical baselines.
- Do NOT produce general paper summaries.
- The user can skip this step with /next if they prefer.
"""

    if step_num == 7:
        prompt += """
ML/RL PARAMETERS -- ONE-TURN STEP:
- Present ALL auto-selected ML/RL hyperparameters in a clear table.
- Ask the user to approve or suggest changes.
- This is a ONE-TURN step: after the user responds, incorporate any
  requested changes and proceed immediately to the next step.
- Do NOT ask follow-up questions. Summarize decisions and move on.
- Your reply MUST contain two machine-readable budget lines, each on its own
  line, exactly in this form (integer minutes, no range, no units after the
  number):
      TESTING_BUDGET_MINUTES: 15
      TRAINING_BUDGET_MINUTES: 45
  These are ENFORCED WALL-CLOCK LIMITS, not estimates. Step 8 and Step 9 are
  each cut off shortly after their budget expires, background jobs included.
  Keep testing under 30 and training under 90. Write both lines even when the
  user accepted the defaults, and restate the same numbers in prose for the
  reader. Discuss them with the user if they seem wrong for this problem --
  once the step ends, the numbers are fixed.
"""

    if step_num == 10:
        prompt += """
STEP 10 REVISIT MODE -- TOOL INSTRUCTIONS:
You are revisiting an existing completed CARL session. Use tools to inspect
artifacts, read logs/results, describe plots, and prepare a fresh revisit
workspace. This step is interactive: discuss evidence and proposed changes with
the user before writing code changes.

Important:
- The controller has already asked the user for optional pre-analysis
  instructions before artifact context was built. Treat the "User Preflight
  Guidance" block as authoritative user input, and do not inspect paths listed
  in "Preflight Excluded Paths".
- Do NOT run retraining in Step 10.
- Do NOT use run_background.
- Call describe_plot on EVERY plot returned by list_revisit_artifacts before
  proposing any change. The plots list is a worklist, not a sample -- do not
  sample it, and report a short or empty inventory as a finding.
- Before calling prepare_revisit_workspace, identify plausible session-local
  source directories from the artifact inventory, recommend one, and get the
  user's explicit confirmation. The confirmed source may be code/,
  code/revisit_N/, long_train/, or another session-local directory containing
  the actual code/config to revise; long-run outputs/checkpoints are evidence
  only unless their directory contains source code to copy.
- Use prepare_revisit_workspace with the confirmed source_dir before editing
  code. It creates a fresh self-contained code/revisit_N/ directory by copying
  that source and writes revisit_train.sh there.
- Edit only inside the new revisit_N directory after it is created.
- revisit_train.sh should run the agreed updated MDP/hyperparameters and write
  outputs under its own experiments/revisit_N/ directory.
- The script should update results/appendix.tex and rebuild main.pdf when
  pdflatex is available.
- Step 10 is interactive. Do NOT output [DONE] in Step 10. When the work is
  complete, provide the concise closeout and ask exactly "Ready to finish
  Step 10?" The controller will close the revisit only after the user confirms.
"""

    if step_num in (8, 9) and budget_minutes:
        prompt += f"""
WALL-CLOCK BUDGET -- {budget_minutes} MINUTES:
This step has an enforced wall-clock budget of {budget_minutes} minutes, set in
Step 7. It is a real limit, not a target:
- When the budget expires you get one short grace window to summarize, then the
  step is cut off and any background jobs are killed.
- Time spent waiting inside check_process counts against the budget. Each call
  sleeps before returning, so polling a job repeatedly is not free.
- There is no restriction on WHAT you do with the time -- redesign the
  architecture, run ablations, change hyperparameters, whatever the evidence
  calls for. The only constraint is the clock.
- Plan backwards from it. Long jobs should fit with room to spare, and
  checkpoint often enough that being stopped mid-run still leaves something
  usable.
- A partial result with a clear write-up of what you established beats being
  cut off mid-investigation with nothing recorded. If time runs short, stop and
  report.
"""

    if step_num in (8, 9):
        prompt += """
CODE EXECUTION -- TOOL INSTRUCTIONS:
You have four tools available. USE THEM — do not just show code in chat.

- write_file(path, content): Create/overwrite a file in the session's code/
  directory. Example: write_file("environment.py", "import gymnasium...")
- run_command(command): Execute a shell command in code/ and see output.
  Timeout: 10 minutes. Use for short tasks: pip install, pytest, dry runs.
- run_background(command, log_file): Start a long-running command (training,
  hyperparameter sweeps) in the background. Output streams to code/<log_file>.
  Returns immediately with a PID.
- check_process(log_file, tail_lines=50): Check if a background process is
  still running and read the last N lines of its log file.

Workflow:
1. Use write_file to create each Python file.
2. Use run_command to install dependencies (pip install -r requirements.txt).
3. Use run_command for quick tests and dry runs.
4. For training or any job that may take more than a few minutes, use
   run_background. Then poll with check_process until finished.
5. Read command/log output. If there are errors, fix the file and re-run.
6. Save results and plots to the code/ directory.

CRITICAL -- TOOL CALL DISCIPLINE:
- Call tools DIRECTLY. Do not output text saying "Let me write..." and stop.
  Instead, call write_file or run_command immediately in the same response.
- Chain multiple tool calls in sequence without pausing between them.
  Write all files, then run commands, then check results — all in one turn.
- Every piece of code you propose must be written to a file and executed.
  Do not describe code without creating it.
- Edit files ONLY by rewriting them with write_file. Do not patch in the shell:
  no `sed -i`, no `python -c` heredocs, no `cat > file << EOF`, no `>>`
  appends to source. The saved conversation record truncates tool input, so a
  shell patch is unrecoverable from the transcript and the next session cannot
  see what changed. Rewriting the whole file is always the cheaper option.
- Do NOT write to memory.md. The controller owns that file and appends each
  step's summary itself; a second copy produces duplicate `## Step N` blocks.
  Put run notes, findings, and tables in files under code/ or results/, and
  put decisions in your step summary -- the controller records that.
- run_command and run_background always start in the session's code/ directory.
  Use paths relative to it (`train.py`, `experiments/run1/`); never prefix with
  `code/` and never `cd` to the session directory.

CRITICAL -- BASELINE AND SANITY CHECK TIME LIMITS:
- All baseline/heuristic policies must complete in UNDER 60 SECONDS.
- Do NOT write brute-force solvers, exhaustive search, dynamic programming
  over continuous states, or any exact/optimal solver. These waste time and
  often get killed by the timeout.
- The purpose of baselines is to SANITY-CHECK the environment and establish
  a rough performance range — not to solve the problem exactly.
- Use simple strategies: greedy 1-step lookahead, fixed rules, or domain
  common sense. If a baseline takes >60s, simplify it.
- Finding the optimal solution is the RL agent's job, not the baseline's.
"""

    if step_num == 9:
        prompt += """
LAUNCH STEP -- TRAINING INSTRUCTIONS:
- Write all code files first, then run quick validation (dry run, tests).
- For ALL training phases (supervised pre-training, critic warmup, PPO),
  use run_background to start training and check_process to monitor.
- Training scripts MUST flush output frequently (every update) so
  check_process can track progress in real time.
- Training scripts MUST save checkpoints periodically and on completion.
  Periodically means often enough that the wall-clock budget expiring mid-run
  still leaves a usable checkpoint -- the controller kills background jobs at
  the limit, so an end-of-run-only save loses the entire run.
- After each training phase completes, review the logs and report:
  key metrics (loss, reward, MAE vs analytical), any issues, and next steps.
- After training completes, generate the final usable artifact: write
  make_artifact.py, run it with run_command to build the self-contained
  artifact/ bundle (copied checkpoint, contract.json, solve.py, smoke_test.py,
  MODEL_CARD.md), then run artifact/smoke_test.py with run_command. A failing
  smoke test blocks shipping -- fix it, do not report success around it.
- When all training is complete, provide a FINAL SUMMARY of:
  - Training phases completed and key metrics from each
  - Best model performance vs analytical benchmark
  - Generated files (code, checkpoints, plots, logs)
  - The artifact/ bundle path and the smoke-test result (pass/fail with the
    reproduced objective value)
  - Recommendation: iterate or ship, with specific reasons
- After the final summary, output the marker [DONE] on its own line.
  Do NOT ask follow-up questions after the final summary -- the session
  is running autonomously and there is no one to answer.
- The user may Ctrl+C at any time. Background processes will be stopped
  and the session will auto-save all progress up to that point.
"""

    return prompt
