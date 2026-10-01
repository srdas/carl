# CARL launch commands

Run all commands from the repository root. Replace angle-bracketed values
with values appropriate for your own environment; no account names, account
numbers, or credentials belong in commands or configuration files.

## Local workstation setup

CARL can run on a local workstation and does not require SageMaker. Create a
virtual environment, activate it, and install the controller dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The generated Step 8/9 artifact in `goddard_code/` has an independent
environment. Install it only when you want to run its tests or scripts:

```powershell
python -m pip install -r goddard_code/requirements.txt
Push-Location goddard_code
python -m pytest test_step8.py -q
Pop-Location
```

## Local running without AWS or SageMaker

SageMaker is not required: the web and terminal interfaces run on the local
machine. The shipped LLM provider is Amazon Bedrock Converse, so an actual
CARL workflow still requires AWS credentials and permission to invoke a
Bedrock model. A fully AWS-free local-model backend is not included in this
repository. You can inspect command-line help without credentials:

```powershell
python web.py --help
python carl.py --help
```

## Bedrock credentials and configuration

Use the standard AWS credential chain, such as an AWS SSO profile, environment
credentials, or a workload role. For an SSO profile:

```powershell
aws sso login --profile <aws-profile>
$env:AWS_PROFILE="<aws-profile>"
$env:AWS_DEFAULT_PROFILE="<aws-profile>"
$env:AWS_REGION="<aws-region>"
```

Alternatively, copy `.env.example` to `.env` and set only the variables that
apply. Keep `.env` out of version control and never put real access keys,
tokens, or contact details in it.

## Start CARL locally

```powershell
python web.py
```

The web interface listens on `http://localhost:5001` by default and is bound
to the local machine. To choose another port:

```powershell
$env:CARL_PORT="8080"
python web.py
```

For the terminal interface:

```powershell
python carl.py
```

## Optional workflow modes

```powershell
python web.py --write-report
python web.py --no-persona-update
python web.py --revisit sessions/<session_dir>
python web.py --resume sessions/<session_dir>
```

`--write-report` generates the optional LaTeX/PDF report after completion.
`--no-persona-update` skips the post-session persona update. `--revisit`
opens a completed session in Step 10; `--resume` continues an unfinished
session. `--revisit` and `--resume` are mutually exclusive.

## Non-Anthropic models through Converse

The Bedrock Converse integration is provider-neutral for its core message,
streaming, and tool schemas. To use another model that your account can invoke
through Converse, set its Bedrock model ID before launching:

```powershell
$env:CARL_MODEL="<converse-model-id>"
python web.py
```

Use the role-specific overrides when those tasks should use a different
Converse model:

```powershell
$env:CARL_REPORT_MODEL="<converse-model-id>"
$env:CARL_TRILLIAN_MODEL="<converse-model-id>"
$env:CARL_PERSONA_UPDATE_MODEL="<converse-model-id>"
```

For non-Anthropic models, CARL sends the portable Converse request shape and
does not send Anthropic-only prompt-caching or extended-thinking fields. The
selected model must support the capability used by each workflow stage:
text generation for all stages, tool use for autonomous coding stages, and
image input for report plot descriptions. Model availability and supported
features vary by region and account, so validate a model with a short run
before using it for a full workflow.

## Run the included artifact tests

```powershell
Push-Location goddard_code
python -m pytest test_step8.py -q
Pop-Location
```

The checked-in test assertions must be synchronized with the current generated
artifact before this suite will pass: the tests currently expect a 12-element
observation while the environment and network use 14 elements, and one reward
expectation uses an older formulation.
