from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypeGuard

from litellm.types.utils import Choices, StreamingChoices


class _SupportsContent(Protocol):
    content: object


class _SupportsText(Protocol):
    text: object


def _has_text(value: object) -> TypeGuard[_SupportsText]:
    return hasattr(value, "text")


def _has_content(value: object) -> TypeGuard[_SupportsContent]:
    return hasattr(value, "content")


def _collect_text(value: object) -> str:
    match value:
        case str():
            return value
        case Sequence() as seq if not isinstance(seq, (str, bytes, bytearray)):
            return "".join(_collect_text(item) for item in seq)
        case _ if _has_text(value):
            return _collect_text(value.text)
        case _:
            return ""


def extract_choice_text(choice: Choices | StreamingChoices) -> tuple[str, bool]:
    """Return textual content for a LiteLLM choice and flag if it was streaming."""
    match choice:
        case StreamingChoices(delta=delta) if _has_content(delta):
            return _collect_text(delta.content), True
        case StreamingChoices():
            return "", True
        case _:
            return _collect_text(choice.message.content), False
