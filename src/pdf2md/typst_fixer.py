"""LLM-based Typst error fixing using LiteLLM."""

import hashlib
from pathlib import Path
from typing import cast

import anyio
import litellm

from logs import get_logger

from .fix_progress import TypstFixEntry, TypstFixProgress
from .prompt_loader import get_rendered_agent
from .settings import settings
from .typst_parser import TypstBlock, extract_typst_blocks, reconstruct_markdown_with_fixes
from .typst_validator import TypstValidationResult, get_invalid_blocks, validate_all_typst_blocks

ReMatch = str | tuple[str, str]

logger = get_logger(__name__)


def _get_progress_file_path(markdown_content: str) -> Path:
    """Generate a unique progress file path based on content hash."""
    content_hash = hashlib.sha256(markdown_content.encode()).hexdigest()[:16]
    return Path.cwd() / f".typst_fix_progress_{content_hash}.json"


async def fix_single_typst_error(block: "TypstBlock", error_message: str, model: str) -> str:
    """Fix a single Typst code error using LLM with proper block type context."""
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

        # Parse the response to extract fixed content
        response_text = cast(
            "list[litellm.Choices]",
            cast("litellm.ModelResponse", response).choices,  # pyright: ignore[reportPrivateImportUsage]
        )[0].message.content  # type: ignore[reportUnknownMemberType]
    except (TimeoutError, litellm.exceptions.InternalServerError) as e:
        # Only catch the same humble set as main module
        logger.warning("LLM error in fix_single_typst_error: %s", e)
        return block.content
    else:
        return response_text.strip() if response_text else block.content


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
    content_hash = hashlib.sha256(markdown_content.encode()).hexdigest()[:16]

    # Load existing progress
    progress = await TypstFixProgress.load_from_file(progress_file)
    if progress is None:
        progress = TypstFixProgress(content_hash=content_hash, fixes={})
    else:
        logger.info("Loaded %d existing fix(es) from progress file", len(progress.fixes))

    # Group fixes by content to avoid duplicate work, but keep block reference for type info
    unique_fixes: dict[str, tuple[TypstBlock, str]] = {}  # content -> (block, error_message)
    for result in invalid_results:
        content_key = result.block.content
        if content_key not in unique_fixes and content_key not in progress.fixes:
            unique_fixes[content_key] = (result.block, result.error_message)

    for attempt in range(max_attempts):
        remaining_fixes = {k: v for k, v in unique_fixes.items() if k not in progress.fixes}

        if not remaining_fixes:
            break

        logger.info("Fix attempt %d/%d for %d error(s)", attempt + 1, max_attempts, len(remaining_fixes))

        # Process fixes in parallel with limited concurrency
        batch_size = min(settings.max_concurrency, len(remaining_fixes))
        fixes_list = list(remaining_fixes.items())

        async def fix_single_wrapper(original_content: str, fix_info: tuple[TypstBlock, str]) -> tuple[str, str | None]:
            block, error_message = fix_info
            try:
                fixed_content = await fix_single_typst_error(
                    block=block,
                    error_message=error_message,
                    model=settings.fixing_model,
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
                original_content: str, fix_info: tuple["TypstBlock", str], results: dict[str, str | None]
            ) -> None:
                _, fixed_content = await fix_single_wrapper(original_content, fix_info)
                results[original_content] = fixed_content

            # Process all tasks in the batch using task group
            async with anyio.create_task_group() as tg:
                for original_content, fix_info in batch:
                    tg.start_soon(collect_result, original_content, fix_info, batch_results)

            # Process batch results
            batch_progress = False
            for original_content, fixed_content in batch_results.items():
                if fixed_content is not None and fixed_content != original_content:
                    # Find the corresponding block to get AST path and type
                    block, _ = remaining_fixes[original_content]
                    progress.fixes[original_content] = TypstFixEntry(
                        original_content=original_content,
                        fixed_content=fixed_content,
                        ast_path=block.ast_path,
                        block_type=block.type,
                    )
                    logger.info("Successfully fixed Typst block: %s", original_content[:50] + "...")
                    batch_progress = True
                elif fixed_content is not None:
                    logger.warning("LLM returned unchanged content for: %s", original_content[:50] + "...")

            # Save progress after each batch
            if batch_progress:
                save_return_code = await progress.save_to_file(progress_file)
                logger.debug(
                    "Saved progress: %d fixes completed with return code %d", len(progress.fixes), save_return_code
                )

        # If we made some progress, break
        if progress.fixes:
            break

    # Apply fixes to markdown content
    if progress.fixes:
        # Convert TypstFixEntry back to simple mapping for reconstruction
        fixed_contents = {entry.original_content: entry.fixed_content for entry in progress.fixes.values()}

        # Extract all blocks (not just invalid ones) for reconstruction
        all_blocks = [result.block for result in validation_results]
        current_content = reconstruct_markdown_with_fixes(current_content, all_blocks, fixed_contents)

        logger.info("Applied %d Typst fix(es) to markdown content", len(progress.fixes))

        # Clean up progress file on successful completion
        all_fixed = len(progress.fixes) >= len(unique_fixes)
        if all_fixed:
            progress.cleanup_file(progress_file)

    all_fixed = len(progress.fixes) >= len(unique_fixes)
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
