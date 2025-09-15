"""LLM-based Typst error fixing using LiteLLM."""

import hashlib
import json
from pathlib import Path
from typing import cast

import anyio
import litellm
import orjson

from logs import get_logger

from .settings import settings
from .typst_parser import extract_typst_blocks, reconstruct_markdown_with_fixes
from .typst_validator import TypstValidationResult, get_invalid_blocks, validate_all_typst_blocks

ReMatch = str | tuple[str, str]

logger = get_logger(__name__)


def _get_progress_file_path(markdown_content: str) -> Path:
    """Generate a unique progress file path based on content hash."""
    content_hash = hashlib.sha256(markdown_content.encode()).hexdigest()[:16]
    return Path.cwd() / f".typst_fix_progress_{content_hash}.json"


def _save_progress(progress_file: Path, fixed_mapping: dict[str, str]) -> None:
    """Save the current progress to a file."""
    try:
        with progress_file.open("w", encoding="utf-8") as f:
            json.dump(fixed_mapping, f, ensure_ascii=False, indent=2)
    except (OSError, ValueError):
        pass  # Ignore save errors to avoid disrupting the main process


def _load_progress(progress_file: Path) -> dict[str, str]:
    """Load existing progress from a file."""
    if not progress_file.exists():
        return {}

    try:
        with progress_file.open("r", encoding="utf-8") as f:
            return cast("dict[str, str]", orjson.loads(f.read()))
    except (OSError, ValueError, json.JSONDecodeError):
        logger.exception("Failed to load progress from %s", progress_file)
        return {}  # Return empty dict if loading fails


def _cleanup_progress_file(progress_file: Path) -> None:
    """Remove the progress file after successful completion."""
    try:
        if progress_file.exists():
            progress_file.unlink()
    except OSError:
        pass  # Ignore cleanup errors


async def load_typst_instructions() -> str:
    """Load Typst instructions from the comprehensive instructions file."""
    async with await anyio.open_file(
        Path(__file__).parent / "prompts" / "comprehensive_typst_instructions.md", "r", encoding="utf-8"
    ) as f:
        return await f.read()


async def fix_single_typst_error(
    original_content: str, error_message: str, location: str, block_type: str, model: str
) -> str:
    """Fix a single Typst code error using LLM."""
    system_prompt = await load_typst_instructions()

    user_prompt = f"""Fix the following Typst code that has compilation errors:

Location: {location}
Block type: {block_type}

Original content:
```
{original_content}
```

Error message:
```
{error_message}
```

Please provide the fixed Typst code ONLY.
DO NOT emit any other text, not even Markdown formatting.
Your response should be the correct, drop-in replacement for the original content.
"""

    try:
        # Use the model to generate a fix
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

        response = await litellm.acompletion(  # pyright: ignore[reportUnknownMemberType]
            model=model,
            messages=messages,
        )

        # Parse the response to extract fixed content
        response_text = cast(
            "list[litellm.Choices]",
            cast("litellm.ModelResponse", response).choices,  # pyright: ignore[reportPrivateImportUsage]
        )[0].message.content  # type: ignore[reportUnknownMemberType]
    except (OSError, RuntimeError, ValueError, TypeError):
        # Return original content if fixing fails to avoid breaking the document
        return original_content
    else:
        return response_text or ""


async def fix_typst_errors(
    markdown_content: str, validation_results: list[TypstValidationResult], logger_name: str, max_attempts: int = 3
) -> tuple[str, bool]:
    """
    Fix Typst errors in markdown content using LLM with parallel processing and progress saving.

    Returns:
        (fixed_content, all_fixed): The fixed markdown content and whether all errors were resolved
    """
    logger = get_logger(logger_name)
    invalid_results = get_invalid_blocks(validation_results)

    if not invalid_results:
        return markdown_content, True

    logger.info("Attempting to fix %d Typst error(s) using LLM", len(invalid_results))

    current_content = markdown_content
    progress_file = _get_progress_file_path(markdown_content)

    # Load existing progress
    fixed_mapping: dict[str, str] = _load_progress(progress_file)
    if fixed_mapping:
        logger.info("Loaded %d existing fix(es) from progress file", len(fixed_mapping))

    # Group fixes by content to avoid duplicate work
    unique_fixes: dict[str, tuple[str, str, str]] = {}  # content -> (error_message, location, block_type)
    for result in invalid_results:
        content_key = result.block.content
        if content_key not in unique_fixes and content_key not in fixed_mapping:
            unique_fixes[content_key] = (
                result.error_message,
                result.block.location,
                result.block.block_type,
            )

    for attempt in range(max_attempts):
        remaining_fixes = {k: v for k, v in unique_fixes.items() if k not in fixed_mapping}

        if not remaining_fixes:
            break

        logger.info("Fix attempt %d/%d for %d error(s)", attempt + 1, max_attempts, len(remaining_fixes))

        # Process fixes in parallel with limited concurrency
        batch_size = min(settings.max_concurrency, len(remaining_fixes))
        fixes_list = list(remaining_fixes.items())

        async def fix_single_wrapper(original_content: str, error_info: tuple[str, str, str]) -> tuple[str, str | None]:
            error_message, location, block_type = error_info
            try:
                fixed_content = await fix_single_typst_error(
                    original_content=original_content,
                    error_message=error_message,
                    location=location,
                    block_type=block_type,
                    model=settings.model,
                )
            except (OSError, RuntimeError, ValueError, TypeError):
                logger.exception("Exception during fix for content: %s", original_content[:50] + "...")
                return original_content, None
            else:
                return original_content, fixed_content

        for i in range(0, len(fixes_list), batch_size):
            batch = fixes_list[i : i + batch_size]
            batch_results: dict[str, str | None] = {}

            async def collect_result(
                original_content: str, error_info: tuple[str, str, str], results: dict[str, str | None]
            ) -> None:
                _, fixed_content = await fix_single_wrapper(original_content, error_info)
                results[original_content] = fixed_content

            # Process all tasks in the batch using task group
            async with anyio.create_task_group() as tg:
                for original_content, error_info in batch:
                    tg.start_soon(collect_result, original_content, error_info, batch_results)

            # Process batch results
            batch_progress = False
            for original_content, fixed_content in batch_results.items():
                if fixed_content is not None and fixed_content != original_content:
                    fixed_mapping[original_content] = fixed_content
                    logger.info("Successfully fixed Typst block: %s", original_content[:50] + "...")
                    batch_progress = True
                elif fixed_content is not None:
                    logger.warning("LLM returned unchanged content for: %s", original_content[:50] + "...")

            # Save progress after each batch
            if batch_progress:
                _save_progress(progress_file, fixed_mapping)
                logger.debug("Saved progress: %d fixes completed", len(fixed_mapping))

        # If we made some progress, break
        if fixed_mapping:
            break

    # Apply fixes to markdown content
    if fixed_mapping:
        # Extract all blocks (not just invalid ones) for reconstruction
        all_blocks = [result.block for result in validation_results]
        current_content = reconstruct_markdown_with_fixes(current_content, all_blocks, fixed_mapping)

        logger.info("Applied %d Typst fix(es) to markdown content", len(fixed_mapping))

        # Clean up progress file on successful completion
        all_fixed = len(fixed_mapping) >= len(unique_fixes)
        if all_fixed:
            _cleanup_progress_file(progress_file)

    all_fixed = len(fixed_mapping) >= len(unique_fixes)
    return current_content, all_fixed


async def fix_typst_errors_iteratively(
    markdown_content: str, logger_name: str, max_iterations: int = 2, max_fix_attempts: int = 3
) -> tuple[str, bool]:
    """
    Iteratively fix Typst errors until all are resolved or max iterations reached.

    Returns:
        (final_content, all_errors_fixed)
    """
    logger = get_logger(logger_name)
    current_content = markdown_content

    for iteration in range(max_iterations):
        logger.info("Typst error fixing iteration %d/%d", iteration + 1, max_iterations)

        # Extract and validate Typst blocks
        typst_blocks = extract_typst_blocks(current_content)
        if not typst_blocks:
            logger.info("No Typst blocks found")
            return current_content, True

        validation_results = await validate_all_typst_blocks(typst_blocks, logger_name)
        invalid_results = get_invalid_blocks(validation_results)

        if not invalid_results:
            logger.info("All Typst blocks are valid")
            return current_content, True

        # Attempt to fix errors
        current_content, all_fixed = await fix_typst_errors(
            current_content, validation_results, logger_name, max_fix_attempts
        )

        if all_fixed:
            logger.info("All Typst errors fixed successfully")
            return current_content, True

    logger.warning("Could not fix all Typst errors after %d iterations", max_iterations)
    return current_content, False
