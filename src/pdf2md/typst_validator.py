"""Typst validation and parallel compilation utilities."""

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path

import anyio

from logs import get_logger

from .typst_parser import TypstBlock


def _parse_warnings_from_stderr(stderr_output: str) -> list[str] | None:
    """Parse warning messages from typst compile stderr output."""
    if not stderr_output.strip():
        return None

    # Look for warning patterns in the output
    warning_lines: list[str] = []
    for line in stderr_output.split("\n"):
        if line.strip() and not line.startswith("error:"):  # noqa: SIM102
            # For now, consider non-error lines as potential warnings
            # This can be refined based on actual typst warning format
            if "warning:" in line.lower() or "warn:" in line.lower():
                warning_lines.append(line.strip())

    return warning_lines if warning_lines else None


def _format_typst_error(error_output: str, block_location: str) -> str:
    """Format typst compile error output for display."""
    if not error_output.strip():
        return f"Typst compilation error in {block_location}: Unknown error"

    # Extract relevant error information from the typst output
    # The output format is like:
    # error: unclosed delimiter
    #   ┌─ \\?\E:\lecproc\test_broken1.typ:1:26
    #   │
    # 1 │ #let x = unclosed_function(
    #   │                           ^

    error_lines: list[str] = []
    lines = error_output.split("\n")

    for i, line in enumerate(lines):
        if line.startswith("error:"):
            # Extract the error message
            error_desc = line[6:].strip()  # Remove "error:" prefix
            error_lines.append(f"Error: {error_desc}")

            # Look for location information in the next few lines
            for j in range(i + 1, min(i + 6, len(lines))):
                next_line = lines[j]
                if "┌─" in next_line and ".typ:" in next_line:
                    # Extract line:column info
                    match = re.search(r"\.typ:(\d+):(\d+)", next_line)
                    if match:
                        line_num, col_num = match.groups()
                        error_lines.append(f"  at line {line_num}, column {col_num}")
                    break

    if not error_lines:
        # Fallback if we can't parse the format
        first_error_line = next((line for line in lines if line.startswith("error:")), error_output.split("\n")[0])
        error_lines.append(first_error_line)

    formatted_error = "\n".join(error_lines)
    return f"Typst compilation error in {block_location}:\n{formatted_error}"


@dataclass
class TypstValidationResult:
    """Result of validating a Typst code block."""

    block: TypstBlock
    is_valid: bool
    error_message: str = ""
    warnings: list[str] | None = None


async def validate_typst_block(block: TypstBlock, logger_name: str) -> TypstValidationResult:
    """Validate a single Typst block asynchronously using subprocess call to typst compile."""
    logger = get_logger(logger_name)

    try:
        # Get the proper form for Typst compiler validation
        validation_content = block.get_validation_form()

        # Create a temporary file for validation
        async with anyio.NamedTemporaryFile(mode="w", suffix=".typ", delete=False, encoding="utf-8") as tmp_file:
            _ = await tmp_file.write(validation_content)
            tmp_file_path = str(tmp_file.name)

        try:
            # Run typst compile using subprocess
            process = await asyncio.create_subprocess_exec(
                "typst",
                "compile",
                tmp_file_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=Path(tmp_file_path).parent,
            )

            _, stderr = await process.communicate()

            # Clean up temporary file
            Path(tmp_file_path).unlink(missing_ok=True)

            match process.returncode:
                case 0:
                    # Compilation successful
                    warnings = _parse_warnings_from_stderr(stderr.decode("utf-8")) if stderr else None
                    return TypstValidationResult(block=block, is_valid=True, warnings=warnings)
                case _:
                    # Compilation failed - parse error from stderr
                    error_output = stderr.decode("utf-8") if stderr else "Unknown compilation error"
                    error_msg = _format_typst_error(error_output, block.location)
                    return TypstValidationResult(block=block, is_valid=False, error_message=error_msg)

        finally:
            # Ensure temp file is cleaned up even if an exception occurs
            Path(tmp_file_path).unlink(missing_ok=True)

    except Exception as e:
        error_msg = f"Unexpected error validating Typst block in {block.location}: {e}"
        logger.exception("Typst validation error: %s", error_msg)
        return TypstValidationResult(block=block, is_valid=False, error_message=error_msg)


async def validate_all_typst_blocks(
    blocks: list[TypstBlock], logger_name: str, max_concurrency: int = 8
) -> list[TypstValidationResult]:
    """Validate all Typst blocks in parallel using subprocess calls."""
    if not blocks:
        return []

    logger = get_logger(logger_name)
    logger.info("Validating %d Typst block(s) in parallel with max concurrency %d", len(blocks), max_concurrency)

    semaphore = asyncio.Semaphore(max_concurrency)
    results: list[TypstValidationResult | Exception | None] = [None] * len(blocks)

    async def _validate_with_index(index: int, block: TypstBlock) -> None:
        try:
            async with semaphore:
                result = await validate_typst_block(block, logger_name)
                results[index] = result
        except (OSError, RuntimeError, ValueError, TypeError, TimeoutError):
            logger.exception("Exception validating block %s", block.location)

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
