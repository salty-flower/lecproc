"""LLM-based Typst error fixing using LiteLLM."""

import hashlib
import tempfile
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import cast

import anyio
import litellm
import regex as re
from litellm.types.utils import ModelResponse
from rich.console import Console
from rich.table import Table

from logs import get_logger

from .fix_progress import TypstFixEntry, TypstFixProgress
from .prompt_loader import get_rendered_agent
from .settings import settings
from .typst_parser import TypstBlock, extract_typst_blocks, reconstruct_markdown_with_fixes
from .typst_validator import TypstValidationResult, get_invalid_blocks, validate_all_typst_blocks, validate_typst_block

ReMatch = str | tuple[str, str]

logger = get_logger(__name__)

# Constants for table display
_MAX_PREVIEW_LENGTH = 40
_PREVIEW_TRUNCATE_AT = 37


@lru_cache(maxsize=1)
def _progress_cache_dir() -> Path:
    """Return the directory used to persist Typst fix progress."""

    base_dir = Path(tempfile.gettempdir())
    progress_dir = base_dir / "lecproc" / "typst_fix"
    try:
        progress_dir.mkdir(parents=True, exist_ok=True)
    except OSError:  # pragma: no cover - fall back to cwd if tmpdir unavailable
        fallback_dir = Path.cwd() / ".typst_fix_cache"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("Falling back to current working directory for progress cache: %s", fallback_dir)
        return fallback_dir
    return progress_dir


def _get_progress_file_path(markdown_content: str) -> Path:
    """Generate a unique progress file path based on content hash."""

    content_hash = hashlib.sha256(markdown_content.encode()).hexdigest()[:16]
    return _progress_cache_dir() / f"typst_fix_progress_{content_hash}.json"


async def _llm_fix_typst_error(block: "TypstBlock", error_message: str, model: str) -> str:
    """Call LLM to fix a single Typst error (internal helper - does not validate)."""
    try:
        # Use the new agent system to generate the prompt
        prompts_dir = Path(__file__).parent / "prompts"
        messages = await get_rendered_agent(
            "fixer",
            prompts_dir,
            buggy_code=block.content,  # Pure content without delimiters
            block_type=block.type,  # "inline", "block", or "codeblock"
            compiler_error_message=error_message,
            location=block.location,
            surrounding_context=block.get_context_for_llm(),  # Context without AST paths
        )

        response = await litellm.acompletion(  # pyright: ignore[reportUnknownMemberType]
            model=model,
            messages=messages,
        )

        response = response if isinstance(response, ModelResponse) else ModelResponse.model_validate(response)

        # Parse the response to extract fixed content
        response_text = cast(
            "list[litellm.Choices]",
            response.choices,
        )[0].message.content  # type: ignore[reportUnknownMemberType]
    except (TimeoutError, litellm.exceptions.InternalServerError) as e:
        # Only catch the same humble set as main module
        logger.warning("LLM error in _llm_fix_typst_error: %s", e)
        return block.content
    else:
        cleaned = _sanitize_llm_fix(response_text or "", block.type)
        return cleaned if cleaned else block.content


def _strip_fences(text: str) -> str:
    """Remove surrounding code fences (triple/single backticks) if the whole text is fenced."""
    s = text.strip()
    # Triple backticks with optional language line
    m = re.match(r"^```[\t ]*([a-zA-Z0-9_-]+)?\s*\n([\s\S]*?)\n```\s*$", s)
    if m:
        return m.group(2).strip()
    # Single-line inline code
    m = re.match(r"^`([^`]+)`$", s)
    if m:
        return m.group(1).strip()
    return s


def _strip_math_wrappers(text: str) -> str:
    """Remove math delimiters ($ or $$) and backticks."""
    return text.strip("$` \n")


def _sanitize_llm_fix(response_text: str, block_type: str) -> str:
    """Normalize LLM output to pure Typst content (no markdown backticks or $-delimiters)."""
    s = response_text.strip()
    s = _strip_fences(s)
    # Some models add extra fences inside too; try once more
    s = _strip_fences(s)

    # Remove math delimiters depending on block type
    if block_type in ("inline", "block"):
        s = _strip_math_wrappers(s)
        # After removing $$, there might be remaining single $; strip again
        s = _strip_math_wrappers(s)

    # Ensure no trailing/leading stray backticks remain
    return s.strip("`\n ")


async def fix_single_with_validation(
    block: TypstBlock,
    initial_error_message: str,
    model: str,
    logger_name: str,
    max_attempts: int = 3,
) -> tuple[str | None, bool, str | None]:
    """
    Fix a single Typst block with immediate validation after each LLM attempt.

    Returns:
        (fixed_content, is_valid, last_invalid_attempt):
            - fixed_content: the fixed content if successful, None if all attempts failed
            - is_valid: whether the final result is valid
            - last_invalid_attempt: the last attempted fix that failed validation (for error reporting)
    """
    logger = get_logger(logger_name)
    current_error = initial_error_message
    last_attempt = None

    for attempt in range(max_attempts):
        # Get LLM fix
        try:
            fixed_content = await _llm_fix_typst_error(block, current_error, model)
        except Exception:
            logger.exception(
                "Exception during LLM fix attempt %d/%d for: %s", attempt + 1, max_attempts, block.content[:50]
            )
            continue

        # If LLM returned the same content, no point validating
        if fixed_content == block.content:
            logger.warning(
                "LLM returned unchanged content on attempt %d/%d for: %s",
                attempt + 1,
                max_attempts,
                block.content[:50] + "...",
            )
            continue

        # Track this attempt for error reporting
        last_attempt = fixed_content

        # Validate the fixed content immediately
        fixed_block = block.model_copy(update={"content": fixed_content})
        validation = await validate_typst_block(fixed_block, logger_name)

        if validation.is_valid:
            logger.info(
                "Successfully fixed and validated block on attempt %d/%d: %s",
                attempt + 1,
                max_attempts,
                block.content[:50] + "...",
            )
            return fixed_content, True, None

        # Use the new error message for the next retry
        logger.info(
            "Fix attempt %d/%d failed validation for: %s", attempt + 1, max_attempts, block.content[:50] + "..."
        )
        current_error = validation.error_message

    logger.warning("All %d fix attempts failed for: %s", max_attempts, block.content[:50] + "...")
    return None, False, last_attempt


async def _fix_blocks_batch(
    markdown_content: str,
    validation_results: list[TypstValidationResult],
    logger_name: str,
    max_attempts: int = 3,
) -> tuple[str, bool, Path | None, list[tuple[TypstBlock, str, str | None]]]:
    """
    Fix a batch of Typst blocks with parallel processing and immediate validation.

    Each fix attempt is immediately validated. Only validated fixes are accepted.

    Returns:
        (fixed_content, all_fixed, progress_file, failed_fixes):
            - fixed_content: the fixed markdown content string
            - all_fixed: whether we produced validated fixes for all invalid contents we attempted
            - progress_file: path to the progress file if used, None otherwise
            - failed_fixes: list of (block, original_content, last_attempt) for blocks that failed to fix
    """
    logger = get_logger(logger_name)
    invalid_results = get_invalid_blocks(validation_results)

    if not invalid_results:
        return markdown_content, True, None, []

    logger.info("Attempting to fix %d Typst error(s) using LLM", len(invalid_results))

    current_content = markdown_content
    progress_file = _get_progress_file_path(markdown_content)
    progress_file_used = progress_file.exists()
    content_hash = hashlib.sha256(markdown_content.encode()).hexdigest()[:16]

    # Load existing progress
    progress = await TypstFixProgress.load_from_file(progress_file)
    if progress is None:
        progress = TypstFixProgress(content_hash=content_hash, fixes={})
    elif progress.content_hash != content_hash:
        logger.info(
            "Ignoring mismatched progress cache at %s (expected %s, found %s)",
            progress_file,
            content_hash,
            progress.content_hash,
        )
        progress = TypstFixProgress(content_hash=content_hash, fixes={})
    else:
        logger.info(
            "Loaded %d existing fix(es) from progress file %s",
            len(progress.fixes),
            progress_file,
        )
        progress_file_used = True

    # Group fixes by content to avoid duplicate work, but keep block reference for type info
    unique_fixes: dict[str, tuple[TypstBlock, str]] = {}  # content -> (block, error_message)
    for result in invalid_results:
        content_key = result.block.content
        if content_key not in unique_fixes and content_key not in progress.fixes:
            unique_fixes[content_key] = (result.block, result.error_message)

    remaining_fixes = {k: v for k, v in unique_fixes.items() if k not in progress.fixes}

    # Track failed fixes for reporting
    all_fix_attempts: dict[str, tuple[str | None, bool, str | None]] = {}

    if not remaining_fixes:
        logger.info("All errors already fixed in progress cache")
    else:
        logger.info("Processing %d error(s) with immediate validation", len(remaining_fixes))

        # Process fixes in parallel with limited concurrency
        batch_size = min(settings.max_llm_concurrency, len(remaining_fixes))
        fixes_list = list(remaining_fixes.items())

        for i in range(0, len(fixes_list), batch_size):
            batch = fixes_list[i : i + batch_size]
            batch_results: dict[str, tuple[str | None, bool, str | None]] = {}

            async def collect_result(
                original_content: str,
                fix_info: tuple["TypstBlock", str],
                results: dict[str, tuple[str | None, bool, str | None]],
            ) -> None:
                block, error_message = fix_info
                try:
                    fixed_content, is_valid, last_attempt = await fix_single_with_validation(
                        block=block,
                        initial_error_message=error_message,
                        model=settings.fixing_model,
                        logger_name=logger_name,
                        max_attempts=max_attempts,
                    )
                except Exception:
                    logger.exception("Exception during fix for content: %s", original_content[:50] + "...")
                    results[original_content] = (None, False, None)
                else:
                    results[original_content] = (fixed_content, is_valid, last_attempt)

            # Process all tasks in the batch using task group
            async with anyio.create_task_group() as tg:
                for original_content, fix_info in batch:
                    tg.start_soon(collect_result, original_content, fix_info, batch_results)

            # Process batch results and track all attempts
            batch_progress = False
            for original_content, (fixed_content, is_valid, last_attempt) in batch_results.items():
                all_fix_attempts[original_content] = (fixed_content, is_valid, last_attempt)
                if fixed_content is not None and is_valid:
                    # Find the corresponding block to get AST path and type
                    block, _ = remaining_fixes[original_content]
                    progress.fixes[original_content] = TypstFixEntry(
                        original_content=original_content,
                        fixed_content=fixed_content,
                        ast_path=block.ast_path,
                        block_type=block.type,
                    )
                    logger.info("Successfully fixed and validated Typst block: %s", original_content[:50] + "...")
                    batch_progress = True

            # Save progress after each batch
            if batch_progress:
                save_return_code = await progress.save_to_file(progress_file)
                progress_file_used = True
                logger.debug(
                    "Saved progress: %d fixes completed with return code %d", len(progress.fixes), save_return_code
                )

    # Apply fixes to markdown content
    if progress.fixes:
        # Convert TypstFixEntry back to simple mapping for reconstruction
        fixed_contents = {entry.original_content: entry.fixed_content for entry in progress.fixes.values()}

        # Extract all blocks (not just invalid ones) for reconstruction
        all_blocks = [result.block for result in validation_results]
        current_content = reconstruct_markdown_with_fixes(current_content, all_blocks, fixed_contents)

        logger.info("Applied %d Typst fix(es) to markdown content", len(progress.fixes))

    # Collect failed fixes for reporting
    failed_fixes: list[tuple[TypstBlock, str, str | None]] = []
    for original_content, (block, _) in unique_fixes.items():
        if original_content not in progress.fixes:
            # This fix failed
            last_attempt = all_fix_attempts.get(original_content, (None, False, None))[2]
            failed_fixes.append((block, original_content, last_attempt))

    all_fixed = len(progress.fixes) >= len(unique_fixes)
    progress_file_to_return = progress_file if progress_file_used or progress.fixes else None
    return current_content, all_fixed, progress_file_to_return, failed_fixes


async def fix_typst_errors(
    markdown_content: str,
    logger_name: str,
    max_fix_attempts: int = 3,
    progress_callback: Callable[[float, str | None], None] | None = None,
) -> tuple[str, bool, Path | None]:
    """
    Fix Typst errors in markdown content using LLM with immediate validation.

    Returns:
        (final_content, all_errors_fixed, progress_file)
    """
    logger = get_logger(logger_name)

    # Extract and validate Typst blocks
    typst_blocks = extract_typst_blocks(markdown_content)
    if not typst_blocks:
        logger.info("No Typst blocks found")
        if progress_callback:
            progress_callback(1.0, "no typst blocks found")
        return markdown_content, True, None

    if progress_callback:
        progress_callback(0.3, "validating typst blocks")

    validation_results = await validate_all_typst_blocks(typst_blocks, logger_name)
    invalid_results = get_invalid_blocks(validation_results)

    if not invalid_results:
        logger.info("All Typst blocks are valid")
        if progress_callback:
            progress_callback(1.0, "typst already valid")
        return markdown_content, True, None

    if progress_callback:
        progress_callback(0.5, "fixing typst errors")

    # Attempt to fix errors (with immediate validation per block)
    fixed_content, all_fixed, progress_file, failed_fixes = await _fix_blocks_batch(
        markdown_content, validation_results, logger_name, max_fix_attempts
    )

    if all_fixed:
        logger.info("All Typst errors fixed and validated successfully")
        if progress_callback:
            progress_callback(1.0, "typst fixes complete")
    else:
        logger.warning("Some Typst errors could not be fixed")
        if progress_callback:
            progress_callback(1.0, "typst fixes incomplete")

        # Display failed fixes in a Rich table
        if failed_fixes:
            console = Console()
            table = Table(title="[red]Failed Typst Fixes[/red]", show_header=True, header_style="bold")
            table.add_column("Line", style="cyan", width=8)
            table.add_column("Original Content", style="yellow", width=_MAX_PREVIEW_LENGTH)
            table.add_column("Last LLM Attempt", style="magenta", width=_MAX_PREVIEW_LENGTH)

            for block, original_content, last_attempt in failed_fixes:
                line_num = str(block.location) if block.location else "?"
                original_preview = (
                    (original_content[:_PREVIEW_TRUNCATE_AT] + "...")
                    if len(original_content) > _MAX_PREVIEW_LENGTH
                    else original_content
                )
                attempt_preview = (
                    (
                        (last_attempt[:_PREVIEW_TRUNCATE_AT] + "...")
                        if len(last_attempt) > _MAX_PREVIEW_LENGTH
                        else last_attempt
                    )
                    if last_attempt
                    else "[dim]No attempt[/dim]"
                )
                table.add_row(line_num, original_preview, attempt_preview)

            console.print(table)

    return fixed_content, all_fixed, progress_file
