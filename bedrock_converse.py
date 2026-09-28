"""Shared Bedrock Converse API helper with prompt caching.

This module centralizes CARL's Bedrock Converse request building, including
cachePoint placement, tool schema conversion, message conversion, usage
normalization, and model-specific inference options.

The in-memory messages list stays in the CARL-native shape callers already
use (strings, tool_use blocks, tool_result blocks); conversion to Converse
shape happens only at the API boundary inside this module.
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any, Iterator


# ── Model quirks ──────────────────────────────────────────────────────
#
# Two request fields are model-generation-specific, and getting either wrong is
# a hard ValidationException rather than a degraded answer:
#
#   thinking     Claude 5 and Opus 4.7 require {"type": "adaptive"} plus
#                output_config.effort, and reject {"type": "enabled",
#                "budget_tokens": N}. Haiku 4.5 is the inverse: it needs
#                budget_tokens and errors on effort.
#   temperature  Deprecated on Claude 5 and Opus 4.7. Haiku 4.5 accepts it.
#
# Haiku 4.5 (DEFAULT_TRILLIAN_MODEL) is the only remaining consumer of the
# older profile, and it is the newest Haiku there is -- so neither the
# budget_tokens branch nor the accepts-temperature branch can be deleted as
# dead code. A mistake there silently kills the Trillian suggestion path while
# the main workflow looks healthy.

# Matches the Claude 5 family (Fable / Mythos / Opus / Sonnet), with or without
# a region prefix or a dated snapshot suffix.
_CLAUDE_5_RE = re.compile(r"claude-(opus|sonnet|fable|mythos)-5\b")


def _is_claude_5(model: str) -> bool:
    return bool(_CLAUDE_5_RE.search(model))


def _accepts_temperature(model: str) -> bool:
    """Claude 5 and Opus 4.7 reject explicit temperature; older models accept it."""
    return not (_is_claude_5(model) or _is_opus_47(model))


# ── Extended thinking configuration ─────────────────────────────────

_THINKING_BUDGET = {
    "medium": 4000,
    "high": 8000,
    "extra_high": 16000,
}

_STEP_THINKING_TIER: dict[float | int, str] = {
    1:   "medium",
    2:   "high",
    2.5: "high",
    3:   "high",
    4:   "high",
    5:   "high",
    6:   "high",
    7:   "high",
    8:   "extra_high",
    9:   "extra_high",
    10:  "high",
}

_THINKING_MAX_TOKENS: dict[str, int] = {
    "medium": 8192,
    "high": 16384,
    "extra_high": 32000,
}

_EFFORT_MAP: dict[str, str] = {
    "medium": "medium",
    "high": "high",
    "extra_high": "max",
}


def _is_opus_47(model: str) -> bool:
    return "claude-opus-4-7" in model


def _uses_adaptive_thinking(model: str) -> bool:
    """True for models that require thinking.type=adaptive + output_config.effort.

    Claude 5 and Opus 4.7 reject thinking.type=enabled outright; Sonnet 4.6
    also supports adaptive, so it is named here even though no default targets
    it any more. Haiku 4.5 stays on the budget_tokens branch below -- it errors
    on `effort`, and it is still a live default (Trillian), so that branch is
    not dead code.
    """
    return (
        _is_claude_5(model)
        or _is_opus_47(model)
        or "claude-sonnet-4-6" in model
    )


_TIER_ORDER = ("medium", "high", "extra_high")


def thinking_tier_for_step(
    step_number: float | int | None, ceiling: str | None = None,
) -> str | None:
    """Tier for a step, clamped to at most *ceiling* when one is given.

    The ceiling exists because reasoning tokens come out of `maxTokens`: on a
    measured Step 3 prompt the `high` tier spends ~15.9k output tokens on a
    9.5k-character reply, where `medium` spends ~7.8k on 8.0k characters. Turns
    that only write prose are capped at `medium`; turns with tools keep the
    step's own tier, where the extra reasoning is planning real work.
    """
    tier = _STEP_THINKING_TIER.get(step_number) if step_number is not None else None
    if tier is None or ceiling is None:
        return tier
    return min(tier, ceiling, key=_TIER_ORDER.index)


def thinking_config_for_step(
    model: str, step_number: float | int | None, ceiling: str | None = None,
) -> dict | None:
    """Return additionalModelRequestFields for extended thinking, or None."""
    tier = thinking_tier_for_step(step_number, ceiling)
    if tier is None:
        return None
    if _uses_adaptive_thinking(model):
        return {
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": _EFFORT_MAP[tier]},
        }
    return {
        "thinking": {"type": "enabled", "budget_tokens": _THINKING_BUDGET[tier]},
    }


def thinking_max_tokens_for_step(
    step_number: float | int | None, base: int, ceiling: str | None = None,
) -> int:
    """Return max_tokens that accommodates thinking budget + output.

    Pass the same *ceiling* used for :func:`thinking_config_for_step`, so the
    allowance matches the effort actually requested.
    """
    tier = thinking_tier_for_step(step_number, ceiling)
    if tier is None:
        return base
    return max(base, _THINKING_MAX_TOKENS[tier])


# ── Tool + message shape translation ──────────────────────────────────

def anthropic_tool_to_converse(tool: dict) -> dict:
    """Convert one native tool spec to a Converse toolSpec entry."""
    return {
        "toolSpec": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "inputSchema": {"json": tool["input_schema"]},
        }
    }


def anthropic_tools_to_converse(tools: list[dict]) -> list[dict]:
    return [anthropic_tool_to_converse(t) for t in tools]


# Converse rejects text blocks that are empty *or* whitespace-only
# ("text content blocks must contain non-whitespace text"), so a blank block
# has to be replaced with real characters rather than a space.
_EMPTY_TEXT_PLACEHOLDER = "(no content)"


def _user_content_blocks(blocks: list) -> list[dict]:
    out: list[dict] = []
    for b in blocks:
        if not isinstance(b, dict):
            continue
        btype = b.get("type")
        if btype == "text":
            text = b.get("text", "")
            if text.strip():
                out.append({"text": text})
        elif btype == "tool_result":
            content = b.get("content", "")
            if isinstance(content, str):
                result_blocks = [{
                    "text": content if content.strip()
                    else _EMPTY_TEXT_PLACEHOLDER
                }]
            elif isinstance(content, list):
                result_blocks = []
                for rb in content:
                    if isinstance(rb, dict) and rb.get("type") == "text":
                        text = rb.get("text", "")
                        result_blocks.append({
                            "text": text if text.strip()
                            else _EMPTY_TEXT_PLACEHOLDER
                        })
                if not result_blocks:
                    result_blocks = [{"text": _EMPTY_TEXT_PLACEHOLDER}]
            else:
                result_blocks = [{"text": str(content)}]
            out.append({
                "toolResult": {
                    "toolUseId": b["tool_use_id"],
                    "content": result_blocks,
                }
            })
        elif btype == "image":
            # Native Anthropic image block: {"source": {"type": "base64",
            # "media_type": "image/png", "data": <b64>}}. Converse wants raw
            # bytes. We decode on the fly so callers can still pass the
            # base64 shape if they want, but new code should pass
            # {"type": "image_raw", "format": "png", "data": <bytes>} which
            # we also accept below.
            src = b.get("source") or {}
            if src.get("type") == "base64":
                import base64 as _b64
                raw = _b64.b64decode(src.get("data", ""))
                fmt = (src.get("media_type", "image/png")
                       .split("/", 1)[-1])
                out.append({"image": {"format": fmt, "source": {"bytes": raw}}})
        elif btype == "image_raw":
            out.append({
                "image": {
                    "format": b.get("format", "png"),
                    "source": {"bytes": b["data"]},
                }
            })
    return out


def _assistant_content_blocks(blocks: list) -> list[dict]:
    out: list[dict] = []
    for b in blocks:
        if not isinstance(b, dict):
            continue
        btype = b.get("type")
        if btype == "text":
            text = b.get("text", "")
            if text.strip():
                out.append({"text": text})
        elif btype == "tool_use":
            out.append({
                "toolUse": {
                    "toolUseId": b["id"],
                    "name": b["name"],
                    "input": b.get("input") or {},
                }
            })
    return out


def messages_to_converse(messages: list[dict]) -> list[dict]:
    """Convert CARL's native-shape in-memory messages to Converse shape.

    Preserves role ordering. Blank (empty or whitespace-only) strings are
    replaced with ``_EMPTY_TEXT_PLACEHOLDER`` because Converse rejects text
    blocks that carry no non-whitespace characters. Unknown block types are
    dropped silently.
    """
    out: list[dict] = []
    for msg in messages:
        role = msg.get("role")
        if role not in ("user", "assistant"):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            out.append({
                "role": role,
                "content": [{
                    "text": content if content.strip()
                    else _EMPTY_TEXT_PLACEHOLDER
                }],
            })
            continue
        if isinstance(content, list):
            blocks = (_user_content_blocks(content) if role == "user"
                      else _assistant_content_blocks(content))
            if not blocks:
                blocks = [{"text": _EMPTY_TEXT_PLACEHOLDER}]
            out.append({"role": role, "content": blocks})
    return out


# ── Request-body assembly ─────────────────────────────────────────────

def _cache_point(ttl: str | None) -> dict:
    cp: dict = {"type": "default"}
    if ttl:
        cp["ttl"] = ttl
    return {"cachePoint": cp}


def _build_system_blocks(system: str, cache_system: bool, cache_ttl: str | None) -> list[dict]:
    blocks: list[dict] = [{"text": system}]
    if cache_system:
        blocks.append(_cache_point(cache_ttl))
    return blocks


def _build_tool_config(
    tools: list[dict] | None, cache_tools: bool, cache_ttl: str | None,
) -> dict | None:
    if not tools:
        return None
    entries: list[dict] = anthropic_tools_to_converse(tools)
    if cache_tools:
        entries.append(_cache_point(cache_ttl))
    return {"tools": entries}


# Claude 5 thinks adaptively even with no thinking config attached, and those
# reasoning tokens are drawn from maxTokens. A tiny budget is therefore spent
# reasoning before any text is emitted, and the call returns an empty string
# rather than an error -- measured at maxTokens=30 (the session-slug calls),
# which came back empty 3 times in 5. Callers that ask for less than this get
# raised to it; nothing is ever lowered. Same rationale as the temperature gate
# above: fix it once here rather than at ~40 call sites.
_ADAPTIVE_MIN_MAX_TOKENS = 256


def _build_inference_config(
    model: str, max_tokens: int, temperature: float | None,
    thinking_enabled: bool = False,
) -> dict:
    if _uses_adaptive_thinking(model):
        max_tokens = max(max_tokens, _ADAPTIVE_MIN_MAX_TOKENS)
    cfg: dict = {"maxTokens": max_tokens}
    if (temperature is not None
            and _accepts_temperature(model)
            and not thinking_enabled):
        cfg["temperature"] = temperature
    return cfg


def _build_kwargs(
    model: str,
    system: str,
    messages: list[dict],
    tools: list[dict] | None,
    max_tokens: int,
    temperature: float | None,
    cache_system: bool,
    cache_tools: bool,
    cache_ttl: str | None,
    thinking_config: dict | None = None,
) -> dict:
    kwargs: dict = {
        "modelId": model,
        "system": _build_system_blocks(system, cache_system, cache_ttl),
        "messages": messages_to_converse(messages),
        "inferenceConfig": _build_inference_config(
            model, max_tokens, temperature, thinking_enabled=bool(thinking_config),
        ),
    }
    tool_config = _build_tool_config(tools, cache_tools, cache_ttl)
    if tool_config is not None:
        kwargs["toolConfig"] = tool_config
    if thinking_config is not None:
        kwargs["additionalModelRequestFields"] = thinking_config
    return kwargs


# ── Public: non-streaming ─────────────────────────────────────────────

def converse_once(
    client,
    model: str,
    system: str,
    messages: list[dict],
    *,
    tools: list[dict] | None = None,
    max_tokens: int,
    temperature: float | None = None,
    cache_system: bool = False,
    cache_tools: bool = False,
    cache_ttl: str | None = "1h",
    thinking_config: dict | None = None,
) -> dict:
    """Non-streaming Converse call. Returns the raw response dict."""
    kwargs = _build_kwargs(
        model, system, messages, tools, max_tokens, temperature,
        cache_system, cache_tools, cache_ttl, thinking_config,
    )
    return client.converse(**kwargs)


def extract_text_from_converse(resp: dict) -> str:
    """Return the concatenated text from a Converse response."""
    try:
        blocks = resp["output"]["message"]["content"]
    except (KeyError, TypeError):
        return ""
    return "".join(b.get("text", "") for b in blocks if "text" in b)


def converse_text(
    client,
    model: str,
    system: str,
    messages: list[dict],
    *,
    tools: list[dict] | None = None,
    max_tokens: int,
    temperature: float | None = None,
    cache_system: bool = False,
    cache_tools: bool = False,
    cache_ttl: str | None = "1h",
    thinking_config: dict | None = None,
) -> str:
    """Convenience: non-streaming Converse call that returns just the text."""
    resp = converse_once(
        client, model, system, messages,
        tools=tools, max_tokens=max_tokens, temperature=temperature,
        cache_system=cache_system, cache_tools=cache_tools, cache_ttl=cache_ttl,
        thinking_config=thinking_config,
    )
    return extract_text_from_converse(resp)


# ── Public: streaming ─────────────────────────────────────────────────

def _normalize_usage(u: dict) -> dict:
    """Map Converse usage keys to native Anthropic names callers already log."""
    return {
        "input_tokens": u.get("inputTokens", 0),
        "output_tokens": u.get("outputTokens", 0),
        "cache_creation_input_tokens": u.get("cacheWriteInputTokens", 0),
        "cache_read_input_tokens": u.get("cacheReadInputTokens", 0),
    }


def converse_stream(
    client,
    model: str,
    system: str,
    messages: list[dict],
    *,
    tools: list[dict] | None = None,
    max_tokens: int,
    temperature: float | None = None,
    cache_system: bool = True,
    cache_tools: bool = True,
    cache_ttl: str | None = "1h",
    thinking_config: dict | None = None,
) -> Iterator[tuple[str, Any]]:
    """Streaming Converse call.

    Yields normalized stream events:
        ("text", str)                 — one text delta
        ("tool_start", name)          — a tool-use block just started
        ("usage", dict)               — token usage (Anthropic-style keys)
        ("result", (blocks, reason))  — final content blocks + stop_reason,
                                         blocks in CARL's native shape so
                                         callers can append to messages lists
                                         unchanged.
    """
    kwargs = _build_kwargs(
        model, system, messages, tools, max_tokens, temperature,
        cache_system, cache_tools, cache_ttl, thinking_config,
    )
    resp = client.converse_stream(**kwargs)

    content_blocks: list[dict] = []
    current_block: dict | None = None
    current_tool_json = ""
    stop_reason = "end_turn"
    in_reasoning_block = False

    for event in resp["stream"]:
        if "messageStart" in event:
            continue
        if "contentBlockStart" in event:
            start = event["contentBlockStart"].get("start") or {}
            if "toolUse" in start:
                tu = start["toolUse"]
                current_block = {
                    "type": "tool_use",
                    "id": tu["toolUseId"],
                    "name": tu["name"],
                    "input": {},
                }
                current_tool_json = ""
                yield "tool_start", tu["name"]
            else:
                current_block = {"type": "text", "text": ""}
            continue
        if "contentBlockDelta" in event:
            delta = event["contentBlockDelta"].get("delta") or {}
            if "reasoningContent" in delta:
                in_reasoning_block = True
                continue
            if "text" in delta:
                if current_block is None:
                    current_block = {"type": "text", "text": ""}
                current_block["text"] += delta["text"]
                yield "text", delta["text"]
            elif "toolUse" in delta:
                current_tool_json += delta["toolUse"].get("input", "")
            continue
        if "contentBlockStop" in event:
            if in_reasoning_block:
                in_reasoning_block = False
                continue
            if current_block is not None:
                if current_block.get("type") == "tool_use":
                    try:
                        current_block["input"] = (
                            json.loads(current_tool_json) if current_tool_json else {}
                        )
                        content_blocks.append(current_block)
                    except json.JSONDecodeError as e:
                        sys.stderr.write(
                            f"[bedrock_converse] dropping malformed tool_use "
                            f"block (name={current_block.get('name')!r}, "
                            f"id={current_block.get('id')!r}): {e}; "
                            f"raw_input={current_tool_json!r}\n"
                        )
                else:
                    content_blocks.append(current_block)
            current_block = None
            current_tool_json = ""
            continue
        if "messageStop" in event:
            stop_reason = event["messageStop"].get("stopReason", "end_turn")
            continue
        if "metadata" in event:
            u = event["metadata"].get("usage") or {}
            if u:
                yield "usage", _normalize_usage(u)
            continue

    yield "result", (content_blocks, stop_reason)
