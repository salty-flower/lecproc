from __future__ import annotations

from collections.abc import Sequence
from typing import TypeGuard

from litellm.types.utils import Choices, StreamingChoices


def _is_sequence(value: object | None) -> TypeGuard[Sequence[object]]:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _collect_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if _is_sequence(value):
        return "".join(_collect_text(item) for item in value)
    text_attr: object | None = getattr(value, "text", None)
    if isinstance(text_attr, str):
        return text_attr
    if _is_sequence(text_attr):
        return "".join(_collect_text(item) for item in text_attr)
    return ""


def extract_choice_text(choice: Choices | StreamingChoices) -> tuple[str, bool]:
    """Return textual content for a LiteLLM choice and flag if it was streaming."""
    if isinstance(choice, StreamingChoices):
        delta = getattr(choice, "delta", None)
        content = getattr(delta, "content", None)
        if _is_sequence(content):
            return _collect_text(content), True
        return "", True

    message_content = choice.message.content
    if isinstance(message_content, str):
        return message_content, False
    if _is_sequence(message_content):
        return _collect_text(message_content), False
    return "", False
