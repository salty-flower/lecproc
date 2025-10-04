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

from logs import get_logger

from .fix_progress import TypstFixEntry, TypstFixProgress
from .prompt_loader import get_rendered_agent
from .settings import settings
from .typst_parser import TypstBlock, extract_typst_blocks, reconstruct_markdown_with_fixes
from .typst_validator import TypstValidationResult, get_invalid_blocks, validate_all_typst_blocks

ReMatch = str | tuple[str, str]

logger = get_logger(__name__)


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
        logger.debug(
            "Falling back to current working directory for progress cache: %s", fallback_dir
        )
        return fallback_dir
    return progress_dir


def _get_progress_file_path(markdown_content: str) -> Path:
    """Generate a unique progress file path based on content hash."""

    content_hash = hashlib.sha256(markdown_content.encode()).hexdigest()[:16]
    return _progress_cache_dir() / f"typst_fix_progress_{content_hash}.json"


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

        response = response if isinstance(response, ModelResponse) else ModelResponse.model_validate(response)

        # Parse the response to extract fixed content
        response_text = cast(
            "list[litellm.Choices]",
            response.choices,
        )[0].message.content  # type: ignore[reportUnknownMemberType]
    except (TimeoutError, litellm.exceptions.InternalServerError) as e:
        # Only catch the same humble set as main module
        logger.warning("LLM error in fix_single_typst_error: %s", e)
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


async def fix_typst_errors(
    markdown_content: str,
    validation_results: list[TypstValidationResult],
    logger_name: str,
    max_attempts: int = 3,
) -> tuple[str, bool, dict[str, str], Path | None]:
    """
    Fix Typst errors in markdown content using LLM with parallel processing and progress saving.

    Returns:
        (fixed_content, all_fixed, fixed_contents):
            - fixed_content: the fixed markdown content string
            - all_fixed: whether we produced fixes for all invalid contents we attempted
            - fixed_contents: mapping from original content to fixed content applied
    """
    logger = get_logger(logger_name)
    invalid_results = get_invalid_blocks(validation_results)

    if not invalid_results:
        return markdown_content, True, {}, None

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
                progress_file_used = True
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

    all_fixed = len(progress.fixes) >= len(unique_fixes)
    fixed_contents_final = (
        {entry.original_content: entry.fixed_content for entry in progress.fixes.values()} if progress.fixes else {}
    )
    progress_file_to_return = progress_file if progress_file_used or progress.fixes else None
    return current_content, all_fixed, fixed_contents_final, progress_file_to_return


async def fix_typst_errors_iteratively(
    markdown_content: str,
    logger_name: str,
    max_iterations: int = 2,
    max_fix_attempts: int = 3,
    progress_callback: Callable[[float, str | None], None] | None = None,
) -> tuple[str, bool, set[Path]]:
    """
    Iteratively fix Typst errors until all are resolved or max iterations reached.

    Returns:
        (final_content, all_errors_fixed, progress_files)
    """
    logger = get_logger(logger_name)
    current_content = markdown_content
    used_progress_files: set[Path] = set()

    for iteration in range(max_iterations):
        logger.info("Typst error fixing iteration %d/%d", iteration + 1, max_iterations)

        if progress_callback and max_iterations:
            iteration_fraction = iteration / max_iterations
            progress_callback(iteration_fraction, f"iteration {iteration + 1}/{max_iterations}")

        # Extract and validate Typst blocks
        typst_blocks = extract_typst_blocks(current_content)
        if not typst_blocks:
            logger.info("No Typst blocks found")
            if progress_callback:
                progress_callback(1.0, "no typst blocks found")
            return current_content, True, used_progress_files

        validation_results = await validate_all_typst_blocks(typst_blocks, logger_name)
        invalid_results = get_invalid_blocks(validation_results)

        if not invalid_results:
            logger.info("All Typst blocks are valid")
            if progress_callback:
                progress_callback(1.0, "typst already valid")
            return current_content, True, used_progress_files

        # Attempt to fix errors
        current_content, _all_fixed_by_count, fixed_map, progress_file = await fix_typst_errors(
            current_content, validation_results, logger_name, max_fix_attempts
        )
        if progress_file is not None:
            used_progress_files.add(progress_file)

        # Re-validate only the blocks we already have, after applying content updates in-memory
        if fixed_map:
            updated_blocks = [
                block.model_copy(update={"content": fixed_map.get(block.content, block.content)})
                for block in typst_blocks
            ]
        else:
            updated_blocks = typst_blocks

        validation_results_after = await validate_all_typst_blocks(updated_blocks, logger_name)
        invalid_after = get_invalid_blocks(validation_results_after)

        if not invalid_after:
            logger.info("All Typst errors fixed successfully")
            if progress_callback:
                progress_callback(1.0, "typst fixes complete")
            return current_content, True, used_progress_files

    logger.warning("Could not fix all Typst errors after %d iterations", max_iterations)
    if progress_callback:
        progress_callback(1.0, "typst fixes incomplete")
    return current_content, False, used_progress_files
