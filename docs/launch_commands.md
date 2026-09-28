# CARL Web Launch Commands

## AWS SSO Login

```powershell
aws sso login --profile <your-aws-profile>
aws sso login --profile <your-org>-nonprod-<your-account-id>
```

Run this before launching CARL locally. Replace the placeholders with your own AWS SSO profile and region.

```powershell
$env:AWS_PROFILE="<your-aws-profile>"
$env:AWS_DEFAULT_PROFILE="<your-aws-profile>"
$env:AWS_REGION="<your-aws-region>"
```

## Normal Web Run

```powershell
python web.py
```

Starts the normal 9-step CARL web workflow on `http://localhost:5001`.

## Custom Port

```powershell
$env:CARL_PORT="8080"; python web.py
```

Starts the web UI on a different port.

## Write Report

```powershell
python web.py --write-report
```

Generates the optional LaTeX/PDF report after the normal workflow completes.

## Skip Trillian Update

```powershell
python web.py --no-persona-update
```

Runs the normal workflow but skips the post-session Trillian persona update.

## Report Without Trillian Update

```powershell
python web.py --write-report --no-persona-update
```

Writes the report after completion while leaving the Trillian persona unchanged.

## Step 10 Revisit

```powershell
python web.py --revisit sessions/<session_dir>
```

Opens an existing session directly in Step 10 post-training analysis mode.

## Step 10 Revisit On Custom Port

```powershell
$env:CARL_PORT="8080"; python web.py --revisit sessions/<session_dir>
```

Runs revisit mode on a different local port.

## Resume An Unfinished Session

```powershell
python web.py --resume sessions/<session_dir>
```

Reopens an existing session and continues the normal workflow from its last
unfinished step, restoring prior step decisions. Requires the session's
`memory.md` and `web.log`; refuses sessions that already completed (use
`--revisit` for those). `--resume` and `--revisit` cannot be combined.

## Resume On Custom Port

```powershell
$env:CARL_PORT="8080"; python web.py --resume sessions/<session_dir>
```

Runs resume mode on a different local port.

## Run The Tests

```powershell
python -m pytest tests/ -q
```

Runs the controller-side unit tests (skill loading and format, shared tool
runtime, plot discovery, memory parsing, resume detection, artifact plumbing).
No AWS credentials required.

## Run A Finished Model

```powershell
cd sessions/<session_dir>/code/artifact
python smoke_test.py
python solve.py --instance instance.json --out result.json
```

A completed Step 9 run leaves a self-contained `artifact/` bundle. `smoke_test.py`
reproduces the recorded metric on the canonical instance; `solve.py` solves a new
instance (`--param k=v` may be repeated instead of an instance file). Check
`MODEL_CARD.md` in the same directory for the validity domain before trusting a
result. A revisit's bundle is under `code/revisit_N/artifact/`.
