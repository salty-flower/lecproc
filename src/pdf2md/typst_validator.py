"""Typst validation and parallel compilation utilities."""

import sys
from dataclasses import dataclass

import anyio
import typst
from anyio import to_thread

from logs import get_logger

from .typst_parser import TypstBlock
from .utils import check_typst_syntax


@dataclass
class TypstValidationResult:
    """Result of validating a Typst code block."""

    block: TypstBlock
    is_valid: bool
    error_message: str = ""
    warnings: list[typst.TypstWarning] | None = None


async def validate_typst_block(block: TypstBlock, logger_name: str) -> TypstValidationResult:
    """Validate a single Typst block asynchronously."""
    logger = get_logger(logger_name)

    try:
        # Run the validation in a thread pool to avoid blocking
        def _validate() -> tuple[bool, typst.TypstError | list[typst.TypstWarning] | str]:
            return check_typst_syntax(block.content)

        is_valid, diagnostics = await to_thread.run_sync(_validate)

        if is_valid:
            warnings = diagnostics if isinstance(diagnostics, list) else None
            return TypstValidationResult(block=block, is_valid=True, warnings=warnings)

        # Format error message from diagnostics
        if isinstance(diagnostics, typst.TypstError):
            error_msg = f"Typst compilation error in {block.location}:\n{diagnostics}"
        else:
            error_msg = f"Typst compilation error in {block.location}:\n{diagnostics}"

        return TypstValidationResult(block=block, is_valid=False, error_message=error_msg)

    except Exception as e:
        error_msg = f"Unexpected error validating Typst block in {block.location}: {e}"
        logger.exception("Typst validation error: %s", error_msg)
        return TypstValidationResult(block=block, is_valid=False, error_message=error_msg)


async def validate_all_typst_blocks(
    blocks: list[TypstBlock], logger_name: str, max_concurrency: int = sys.maxsize
) -> list[TypstValidationResult]:
    """Validate all Typst blocks in parallel."""
    if not blocks:
        return []

    logger = get_logger(logger_name)
    logger.info("Validating %d Typst block(s) in parallel", len(blocks))

    semaphore = anyio.Semaphore(max_concurrency)
    results: list[TypstValidationResult | Exception | None] = [None] * len(blocks)

    async def _validate_with_index(index: int, block: TypstBlock) -> None:
        try:
            async with semaphore:
                result = await validate_typst_block(block, logger_name)
                results[index] = result
        except (OSError, RuntimeError, ValueError, TypeError) as e:
            results[index] = e

    async with anyio.create_task_group() as tg:
        for i, block in enumerate(blocks):
            tg.start_soon(_validate_with_index, i, block)

    # Handle any exceptions that occurred during validation
    final_results: list[TypstValidationResult] = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error("Exception validating block %s: %s", blocks[i].location, result)
            final_results.append(
                TypstValidationResult(
                    block=blocks[i], is_valid=False, error_message=f"Exception during validation: {result}"
                )
            )
        elif result is not None:
            final_results.append(result)
        else:
            # This should not happen, but handle gracefully
            logger.error("Unexpected None result for block %s", blocks[i].location)
            final_results.append(
                TypstValidationResult(
                    block=blocks[i], is_valid=False, error_message="Unexpected None result during validation"
                )
            )

    # Log summary
    valid_count = sum(1 for r in final_results if r.is_valid)
    invalid_count = len(final_results) - valid_count

    if invalid_count == 0:
        logger.info("All %d Typst block(s) validated successfully", len(final_results))
    else:
        logger.warning("Typst validation: %d valid, %d invalid block(s)", valid_count, invalid_count)

    return final_results


def get_invalid_blocks(validation_results: list[TypstValidationResult]) -> list[TypstValidationResult]:
    """Get only the invalid validation results."""
    return [result for result in validation_results if not result.is_valid]


def format_validation_errors(invalid_results: list[TypstValidationResult]) -> str:
    """Format validation errors into a readable string for LLM feedback."""
    if not invalid_results:
        return ""

    error_parts: list[str] = []
    error_parts.append(f"Found {len(invalid_results)} Typst compilation error(s):")
    error_parts.append("")

    for i, result in enumerate(invalid_results, 1):
        error_parts.append(f"Error {i}: {result.block.location}")
        error_parts.append(f"Content: {result.block.content}")
        error_parts.append(f"Error: {result.error_message}")
        error_parts.append("")

    return "\n".join(error_parts)


async def has_any_typst_errors(
    blocks: list[TypstBlock], logger_name: str, max_concurrency: int = 8
) -> tuple[bool, str]:
    """
    Check if any Typst blocks have compilation errors.

    Returns:
        (has_errors, error_message): True if errors found, with formatted error message
    """
    if not blocks:
        return False, ""

    validation_results = await validate_all_typst_blocks(blocks, logger_name, max_concurrency)
    invalid_results = get_invalid_blocks(validation_results)

    if invalid_results:
        error_message = format_validation_errors(invalid_results)
        return True, error_message

    return False, ""
