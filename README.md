# CARL (clean working copy)

This is the minimal CARL controller and support set, copied without prior runs or research-history material.

## Run

1. Create and activate a Python virtual environment (Python 3.10+ recommended).
2. Install the controller dependencies: `python -m pip install -r requirements.txt`.
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

This project is licensed under the GNU General Public License, version 3. See
[LICENSE](LICENSE).
