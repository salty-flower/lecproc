"""Shared LiteLLM response protocols used for typing."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class CompletionMessage(Protocol):
    """Subset of LiteLLM chat message content we rely on."""

    content: str | None


@runtime_checkable
class CompletionChoice(Protocol):
    """LiteLLM choice with a chat message."""

    message: CompletionMessage


@runtime_checkable
class CompletionResponse(Protocol):
    """Protocol for LiteLLM responses that expose ``choices``."""

    choices: list[CompletionChoice]


__all__ = [
    "CompletionChoice",
    "CompletionMessage",
    "CompletionResponse",
]
