# CARL (Chat Assisted Reinforcement Learning)

## Introduction

Reinforcement learning (RL) is a powerful tool for sequential decision-making, but most applied RL projects fail not because of weak algorithms, but because the problem itself is poorly specified. Domain experts typically hand off a high-level goal while leaving out critical details — constraints, reward shaping, state representations — and these hidden assumptions surface only after code has been written, triggering costly re-engineering. This "RL specification gap" is the real bottleneck in applied RL, producing well-known failure modes such as reward hacking (the agent satisfies the reward function while violating the task's intent) and state aliasing (the agent cannot distinguish between distinct states that have been collapsed into a single observation).

CARL (Chat Assisted Reinforcement Learning) addresses this gap with a human-AI co-design architecture that enforces interactive problem specification *before* any code is generated. CARL walks a domain expert through a structured ten-step workflow, using step-specific skill files, domain-aware suggestions from an assistant persona called Trillian, and compact decision summaries that preserve context across iterations without overloading it. Once the expert confirms the resulting Markov decision process (MDP), CARL's autonomous tool-use loop generates, tests, executes, and debugs a complete RL training pipeline, then supports an interactive post-training diagnosis and revisit step so the expert can verify results and adjust parameters before launching another iteration.

To showcase the system in this repository, CARL has been applied to two very different path-dependent optimization problems: goals-based wealth management in finance, and dynamic-pressure-constrained Goddard rocket control in aerospace engineering. In both cases, interactive specification engineering with sustained human domain expertise resolves failure modes that unguided, fully autonomous code generation could not, consistently producing viable policies where "vibe coding" alone fails.

This is a minimal CARL controller and support set, copied without prior runs or research-history material.

## Run

1. Create and activate a Python virtual environment (Python 3.10+ recommended). This is best done using `uv`. To install `uv` follow the instructions here: https://docs.astral.sh/uv/getting-started/installation/. Then create the virtual environment and enable it as follows: (Feel free to use any higher version of Python as well. 
)
```
uv venv --python 3.13
source .venv/bin/activate
```

2. Install the controller dependencies: `uv python -m pip install -r requirements.txt`.
3. Configure Bedrock access through the standard AWS credential chain. If helpful, copy `.env.example` to `.env` and fill in only the applicable values.
4. Start the web interface with `python web.py`, or the terminal interface with `python carl.py`.

New runs will create `sessions/` automatically. The optional `/suggest` command will create `refs/` automatically.

## Included

- Root Python modules: terminal and web controllers, Bedrock client, session/resume/revisit/reporting, literature search, step orchestration, and tool runtime.
- `carl_skills/`: all CARL workflow step skills.
- `trillian_skills/`: the required CARL Trillian persona skill.
- `templates/` and `static/`: all served HTML templates, report description template, and the CARL image asset.
- `docs/`: CARL design, technical, launch, overview, and artifact-reference documentation/templates.
- `requirements.txt`, `goddard_code/requirements.txt`, `report_template.tex`, and `install-tex.sh`.

## Deliberately omitted

- `sessions/`, audit sessions/results, screenshots, checkpoints, logs, generated reports, and other prior-run output.
- JITRL/JIT skill material, NoSkill comparisons, legacy GitLab copies, backups, paper artifacts, archives, and the existing reference corpus.
- The source `.env`; it may contain local account configuration. This copy includes a credential-free `.env.example` instead.

## Runtime scope

CARL can be hosted on a local workstation; it does not require SageMaker.
Its LLM integration currently uses the Amazon Bedrock Converse API, however,
so a complete workflow still requires AWS credentials and Bedrock model
access. A fully AWS-free local-model provider is not included. See
`docs/launch_commands.md` for local-host and model-configuration guidance.

## License

Copyright &copy; 2026 Franklin Templeton. All rights reserved.

CARL (Chat Assisted Reinforcement Learning) is developed by Franklin Templeton and released as open-source software under the **GNU General Public License, Version 3.0 (GPL-3.0)**.

You may use, modify, and redistribute this software in accordance with the terms of the GPL-3.0 license. See the [LICENSE](LICENSE) file for full license text.

This software is provided "AS IS", without warranty of any kind, to the extent permitted by applicable law. Franklin Templeton does not provide any guarantee of performance, fitness for a particular purpose, or suitability for investment or other consequential decision-making. 