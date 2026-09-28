"""Shared conversation completion guards for CARL controllers."""

from __future__ import annotations

import re


MISSING_QUESTION_CONTINUE_PROMPT = (
    "Your previous turn ended without asking the user a question. "
    "Continue exactly from where you left off. If work remains, keep using "
    "tools as needed. When you are ready for user input, end with one clear "
    "question whose final visible character is a question mark."
)


_TRAILING_QUESTION_RE = re.compile(r"\?\s*(?:[`*_>\)\]\}\"']|\s)*$")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_DIRECT_USER_REQUEST_RE = re.compile(
    r"^(?:please\s+)?(?:"
    r"tell me|give me|share|describe|explain|let me know|send me|provide|"
    r"specify|confirm|choose|pick|select|answer|reply|say|upload|point me to|"
    r"walk me through|fill me in"
    r")\b",
    re.IGNORECASE,
)


def _strip_leading_markup(sentence: str) -> str:
    """Remove common Markdown/list markers before checking the sentence verb."""
    return re.sub(r"^\s*(?:[-*+]\s+|\d+[.)]\s+|>+\s+|[*_`]+)", "", sentence).strip()


def _tail_sentences(text: str, limit: int = 2) -> list[str]:
    """Return the final user-visible sentences, ignoring blank whitespace."""
    normalized = re.sub(r"\s+", " ", text.strip())
    sentences = [
        part.strip()
        for part in _SENTENCE_SPLIT_RE.split(normalized)
        if part.strip()
    ]
    return sentences[-limit:]


def asks_user_question(text: str) -> bool:
    """Return True when the assistant appears to end by asking the user."""
    stripped = text.strip()
    if not stripped:
        return False
    lower = stripped.lower()
    proceed_phrases = [
        "ready to proceed",
        "proceed to step",
        "move on to step",
        "shall we proceed",
        "shall we move on",
        "move to step",
        "move to the next",
        "go ahead and proceed",
        "ready to finish",
        "ready to end",
        "finish step",
        "finish this step",
        "end step",
        "end this step",
        "shall we close out",
        "close out step",
        "close out this step",
        "close out step 10",
        "close the session",
        "end the session",
        "are we done",
        "done with the session",
    ]
    if any(phrase in lower for phrase in proceed_phrases):
        return True
    if _TRAILING_QUESTION_RE.search(stripped):
        return True
    tail_sentences = _tail_sentences(stripped)
    if any("?" in sentence for sentence in tail_sentences):
        return True
    return any(
        _DIRECT_USER_REQUEST_RE.match(_strip_leading_markup(sentence))
        for sentence in tail_sentences
    )


def should_continue_for_missing_question(text: str) -> bool:
    """Return True when an interactive turn should be continued."""
    stripped = text.strip()
    if not stripped:
        return True
    lower = stripped.lower()
    if "[done]" in lower:
        return False
    return not asks_user_question(stripped)
