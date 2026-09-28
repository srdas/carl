"""Shared Bedrock model defaults and startup resolution for CARL."""

from __future__ import annotations

import os
from collections.abc import Callable

from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    NoCredentialsError,
    PartialCredentialsError,
    ProfileNotFound,
)


# Claude 5 IDs are unsuffixed on the cross-region converse path: adding a
# `-v1` / `-v1:0` suffix is a hard ValidationException. The opposite is true of
# the 4.x IDs below, where the suffixed form is the only one that resolves --
# do not generalize either shape onto the other family.
DEFAULT_MODEL = "us.anthropic.claude-opus-5"
FALLBACK_MODEL = "us.anthropic.claude-sonnet-5"
DEFAULT_REPORT_MODEL = "us.anthropic.claude-sonnet-5"
# Haiku 4.5 is the newest Haiku there is; this needs no version bump, and its
# ID must stay byte-for-byte as written.
DEFAULT_TRILLIAN_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
DEFAULT_PERSONA_UPDATE_MODEL = "us.anthropic.claude-sonnet-5"
DEFAULT_REGION = "us-west-2"

_CREDENTIAL_ERRORS = (NoCredentialsError, PartialCredentialsError, ProfileNotFound)


def _probe_model(client, model_id: str) -> None:
    """Make a tiny request carrying no model-generation-specific fields.

    Deliberately sends neither `temperature` nor `thinking`: those are the two
    fields whose accepted shape differs by model generation, so including
    either would make the probe test the caller's assumptions rather than the
    model's availability.
    """
    client.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": "hi"}]}],
        inferenceConfig={"maxTokens": 5},
    )


def _client_error_summary(exc: ClientError) -> str:
    err = exc.response.get("Error", {})
    code = err.get("Code") or exc.__class__.__name__
    message = err.get("Message") or str(exc)
    return f"{code}: {message}"


def _error_summary(exc: Exception) -> str:
    if isinstance(exc, ClientError):
        return _client_error_summary(exc)
    return f"{exc.__class__.__name__}: {exc}"


def _credential_message(exc: Exception) -> str:
    return (
        "AWS credentials were not available while probing Bedrock models. "
        "Set AWS_PROFILE/AWS_DEFAULT_PROFILE, run aws sso login for that profile, "
        "provide AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY, or set "
        "AWS_BEARER_TOKEN_BEDROCK if using a Bedrock API key. "
        f"Underlying error: {_error_summary(exc)}"
    )


def resolve_model_id(
    client,
    preferred: str | None = None,
    *,
    log: Callable[[str], None] | None = None,
) -> str:
    """Resolve the CARL model, probing the fallback instead of blindly selecting it."""
    preferred = preferred or os.environ.get("CARL_MODEL", DEFAULT_MODEL)
    if preferred != DEFAULT_MODEL:
        return preferred  # user explicitly set a model; trust it

    try:
        _probe_model(client, DEFAULT_MODEL)
        return DEFAULT_MODEL
    except _CREDENTIAL_ERRORS as exc:
        raise RuntimeError(_credential_message(exc)) from exc
    except ClientError as exc:
        default_error = _client_error_summary(exc)
        if log:
            log(
                f"  [Model {DEFAULT_MODEL} unavailable ({default_error}) - "
                f"falling back to {FALLBACK_MODEL}, a lower-capability tier]"
            )
    except BotoCoreError as exc:
        raise RuntimeError(
            "Could not contact Bedrock while probing the default model. "
            f"Underlying error: {_error_summary(exc)}"
        ) from exc

    try:
        _probe_model(client, FALLBACK_MODEL)
        return FALLBACK_MODEL
    except _CREDENTIAL_ERRORS as exc:
        raise RuntimeError(_credential_message(exc)) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Default model {DEFAULT_MODEL} was unavailable, and fallback model "
            f"{FALLBACK_MODEL} also failed its startup probe. "
            f"Fallback error: {_error_summary(exc)}"
        ) from exc
