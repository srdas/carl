# CARL (clean working copy)

This is the minimal CARL controller and support set, copied without prior runs or research-history material.

To begin, install `uv` if not already installed
```
https://docs.astral.sh/uv/getting-started/installation/
```

Then create and activate the `uv` environment:
```
uv venv --python 3.13
source .venv/bin/activate
```

## Run

1. Install the controller dependencies:
`uv python -m pip install -r requirements.txt`.

3. Configure AWS credentials through your usual AWS SSO/profile flow. If helpful, copy `.env.example` to `.env` and fill in the applicable values.

4. Start the web interface with `python web.py`, or the terminal interface with `python carl.py`.

New runs will create `sessions/` automatically. The optional `/suggest` command will create `refs/` automatically.

## Included

- Root Python modules: terminal and web controllers, Bedrock client, session/resume/revisit/reporting, literature search, step orchestration, and tool runtime.
- `carl_skills/`: all CARL workflow step skills.
- `trillian_skills/`: the required CARL Trillian persona skill.
- `templates/` and `static/`: all served HTML templates, report description template, and the CARL image asset.
- `docs/`: CARL design, technical, launch, overview, and artifact-reference documentation/templates.
- `requirements.txt`, `report_template.tex`, and `install-tex.sh`.

## Deliberately omitted

- `sessions/`, audit sessions/results, screenshots, checkpoints, logs, generated reports, and other prior-run output.
- JITRL/JIT skill material, NoSkill comparisons, legacy GitLab copies, backups, paper artifacts, archives, and the existing reference corpus.
- The source `.env`; it may contain local account configuration. This copy includes a credential-free `.env.example` instead.
