from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from time import time
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from logging import Logger

import anyio
from litellm.exceptions import RateLimitError

_DEFAULT_WAIT_SECONDS = 15.0
_MAX_DESCRIPTION_LENGTH = 80
_EPOCH_MS_THRESHOLD = 1_000_000_000_000
_EPOCH_SECONDS_THRESHOLD = 1_000_000_000


@dataclass(slots=True)
class RateLimitWaitInfo:
    """Parsed information about a rate-limit response."""

    wait_seconds: float
    reset_time: datetime | None = None
    reason: str | None = None


async def execute_with_rate_limit_retry[T](
    call: Callable[[], Awaitable[T]],
    *,
    logger: Logger,
    max_attempts: int,
    wait_callback: Callable[[float, str | None], Awaitable[None]] | None = None,
    context: str | None = None,
) -> T:
    """Execute ``call`` retrying on :class:`RateLimitError` with bounded waits."""

    attempts = 0
    while True:
        try:
            return await call()
        except RateLimitError as error:
            attempts += 1
            info = extract_rate_limit_wait_info(error)
            reason = _shorten(info.reason if info else None) or _shorten(context)
            reset_note = info.reset_time.isoformat() if info and info.reset_time else None

            if attempts >= max_attempts:
                logger.error(  # noqa: TRY400
                    "Rate limit persisted after %d attempt(s)%s.",
                    attempts,
                    f" ({reason})" if reason else "",
                )
                raise

            wait_seconds = info.wait_seconds if info else _DEFAULT_WAIT_SECONDS
            wait_seconds = max(wait_seconds, 0.0)
            if wait_seconds == 0:
                wait_seconds = _DEFAULT_WAIT_SECONDS

            parts: list[str] = []
            if reason:
                parts.append(reason)
            if reset_note:
                parts.append(f"reset at {reset_note}")
            wait_message = "; ".join(parts) if parts else None
            joined = f" ({wait_message})" if wait_message else ""

            next_attempt = attempts + 1
            if wait_callback is None:
                logger.warning(
                    "Rate limited%s; waiting %.1fs before retrying attempt %d/%d.",
                    joined,
                    wait_seconds,
                    next_attempt,
                    max_attempts,
                )
            else:
                logger.debug(
                    "Rate limited%s; waiting %.1fs before retrying attempt %d/%d.",
                    joined,
                    wait_seconds,
                    next_attempt,
                    max_attempts,
                )

            message_for_wait = wait_message or reason or context

            if wait_callback is None:
                await anyio.sleep(wait_seconds)
            else:
                await wait_callback(wait_seconds, message_for_wait)

            continue


def extract_rate_limit_wait_info(error: RateLimitError) -> RateLimitWaitInfo | None:
    """Return structured wait information if the error encodes rate-limit metadata."""

    body = _decode_body(getattr(error, "body", None))
    headers = _collect_headers(error, body)
    reason = _extract_reason(error, body)

    retry_after = _get_retry_after(headers)
    if retry_after is not None:
        info = _parse_retry_after(retry_after)
        if info is not None:
            info.reason = reason
            return info

    for key, value in headers.items():
        lowered = key.lower()
        if "reset" not in lowered and "retry" not in lowered:
            continue
        info = _parse_reset_value(value, header_key=lowered)
        if info is not None:
            info.reason = reason
            return info

    if reason is not None:
        return RateLimitWaitInfo(wait_seconds=_DEFAULT_WAIT_SECONDS, reason=reason)

    return None


def _shorten(text: str | None) -> str | None:
    if text is None:
        return None
    stripped = text.strip()
    if not stripped:
        return None
    if len(stripped) <= _MAX_DESCRIPTION_LENGTH:
        return stripped
    return stripped[: _MAX_DESCRIPTION_LENGTH - 1] + "\u2026"


def _safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_datetime(value: str) -> datetime | None:
    iso_value = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    try:
        dt = datetime.fromisoformat(iso_value)
    except ValueError:
        try:
            dt = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _parse_retry_after(value: str) -> RateLimitWaitInfo | None:
    numeric = _safe_float(value)
    if numeric is not None:
        wait_seconds = max(0.0, numeric)
        reset_time = datetime.now(UTC) + timedelta(seconds=wait_seconds)
        return RateLimitWaitInfo(wait_seconds=wait_seconds, reset_time=reset_time)
    dt = _parse_datetime(value)
    if dt is None:
        return None
    wait_seconds = max(0.0, (dt - datetime.now(UTC)).total_seconds())
    return RateLimitWaitInfo(wait_seconds=wait_seconds, reset_time=dt)


def _parse_reset_value(value: str, *, header_key: str) -> RateLimitWaitInfo | None:
    numeric = _safe_float(value)
    if numeric is not None:
        if header_key.endswith(("-ms", "_ms")):
            wait_seconds = max(0.0, numeric / 1000.0)
            reset_time = datetime.now(UTC) + timedelta(seconds=wait_seconds)
            return RateLimitWaitInfo(wait_seconds=wait_seconds, reset_time=reset_time)

        epoch_seconds: float | None = None
        if numeric > _EPOCH_MS_THRESHOLD:
            epoch_seconds = numeric / 1000.0
        elif numeric > _EPOCH_SECONDS_THRESHOLD:
            epoch_seconds = numeric
        if epoch_seconds is not None:
            wait_seconds = max(0.0, epoch_seconds - time())
            reset_time = datetime.fromtimestamp(epoch_seconds, tz=UTC)
            return RateLimitWaitInfo(wait_seconds=wait_seconds, reset_time=reset_time)

        wait_seconds = max(0.0, numeric)
        reset_time = datetime.now(UTC) + timedelta(seconds=wait_seconds)
        return RateLimitWaitInfo(wait_seconds=wait_seconds, reset_time=reset_time)

    dt = _parse_datetime(value)
    if dt is None:
        return None
    wait_seconds = max(0.0, (dt - datetime.now(UTC)).total_seconds())
    return RateLimitWaitInfo(wait_seconds=wait_seconds, reset_time=dt)


JsonMapping = Mapping[str, object]


def _decode_body(body: object) -> JsonMapping | None:
    if body is None:
        return None
    if isinstance(body, Mapping):
        return cast("JsonMapping", body)
    if isinstance(body, (bytes, bytearray)):
        try:
            body = body.decode("utf-8", errors="ignore")
        except UnicodeDecodeError:  # pragma: no cover - extremely unlikely
            return None
    if isinstance(body, str):
        try:
            decoded = json.loads(body)
        except json.JSONDecodeError:
            return None
        if isinstance(decoded, Mapping):
            return cast("JsonMapping", decoded)
    return None


def _collect_headers(error: RateLimitError, body: JsonMapping | None) -> dict[str, str]:
    headers: dict[str, str] = {}

    response = getattr(error, "response", None)
    if response is not None:
        response_headers = getattr(response, "headers", None)
        if response_headers is not None:
            for key, value in response_headers.items():
                if value is None:
                    continue
                headers[str(key).lower()] = str(value)

    if body is not None:
        for candidate in _iter_possible_header_blocks(body):
            for key, value in candidate.items():
                if value is None:
                    continue
                headers.setdefault(str(key).lower(), str(value))

    return headers


def _as_mapping(value: object) -> JsonMapping | None:
    if isinstance(value, Mapping):
        return cast("JsonMapping", value)
    return None


def _iter_possible_header_blocks(body: JsonMapping) -> list[JsonMapping]:
    blocks: list[JsonMapping] = []

    direct_headers = _as_mapping(body.get("headers"))
    if direct_headers is not None:
        blocks.append(direct_headers)

    metadata = _as_mapping(body.get("metadata"))
    if metadata is not None:
        headers = _as_mapping(metadata.get("headers"))
        if headers is not None:
            blocks.append(headers)

    error_block = _as_mapping(body.get("error"))
    if error_block is not None:
        error_headers = _as_mapping(error_block.get("headers"))
        if error_headers is not None:
            blocks.append(error_headers)
        error_metadata = _as_mapping(error_block.get("metadata"))
        if error_metadata is not None:
            nested_headers = _as_mapping(error_metadata.get("headers"))
            if nested_headers is not None:
                blocks.append(nested_headers)

    return blocks


def _extract_reason(error: RateLimitError, body: JsonMapping | None) -> str | None:
    if body is not None:
        error_block = body.get("error")
        if isinstance(error_block, Mapping):
            message = error_block.get("message")
            if isinstance(message, str) and message.strip():
                return message.strip()
        message = body.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()

    message_attr = getattr(error, "message", None)
    if isinstance(message_attr, str) and message_attr.strip():
        return message_attr.strip()
    return None


def _get_retry_after(headers: Mapping[str, str]) -> str | None:
    if "retry-after" in headers:
        return headers["retry-after"]

    for key, value in headers.items():
        if "retry-after" in key:
            return value
    return None


__all__ = [
    "RateLimitWaitInfo",
    "execute_with_rate_limit_retry",
    "extract_rate_limit_wait_info",
]
