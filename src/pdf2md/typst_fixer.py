"""LLM-based Typst error fixing using LiteLLM."""

import re as re_
from typing import cast

import litellm
from regex import regex as re

from logs import get_logger

from .settings import settings
from .typst_parser import extract_typst_blocks, reconstruct_markdown_with_fixes
from .typst_validator import TypstValidationResult, get_invalid_blocks, validate_all_typst_blocks

ReMatch = str | tuple[str, str]


async def fix_single_typst_error(
    original_content: str, error_message: str, location: str, block_type: str, model: str
) -> str:
    """Fix a single Typst code error using LLM."""
    system_prompt = """You are an expert in Typst mathematical typesetting. Your task is to fix Typst compilation errors.

When given Typst code with compilation errors, you should:

1. Analyze the error message to understand what's wrong
2. Fix the syntax or semantic issues in the Typst code
3. Ensure the fixed code follows proper Typst syntax
4. Provide a brief explanation of what you fixed

Key Typst syntax reminders:
- Functions are called with # prefix: #sum, #frac, #sqrt, etc.
- Math expressions can use $ for inline: $x + y$
- Matrices use #mat(): #mat(delim: "(", 1, 2; 3, 4)
- Fractions use #frac(): #frac(numerator, denominator)
- Subscripts and superscripts: x_1, x^2
- Greek letters: alpha, beta, gamma, etc.
- Symbols: arrow.r, subset.eq, infinity, etc.

Only return the corrected Typst code - do not change the mathematical meaning, just fix syntax errors."""

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

Please provide the fixed Typst code and a brief explanation of what you changed."""

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

        return _extract_fixed_code((response_text or "").strip(), original_content)

    except (OSError, RuntimeError, ValueError, TypeError):
        # Return original content if fixing fails to avoid breaking the document
        return original_content


def _extract_fixed_code(response: str, original: str) -> str:
    """Extract the fixed code from the LLM response."""
    # Try to find code blocks marked with ```
    code_blocks: list[str] = re.findall(r"```(?:typ|typst)?\n(.*?)\n```", response, re_.DOTALL)
    if code_blocks:
        return code_blocks[0].strip()

    # Try to find code blocks without language specification
    code_blocks = re.findall(r"```\n(.*?)\n```", response, re_.DOTALL)
    if code_blocks:
        return code_blocks[0].strip()

    # If no code blocks found, try to extract everything after "Fixed:" or similar
    patterns = [
        r"(?:Fixed|Corrected|Solution):\s*\n(.*?)(?:\n\n|$)",
        r"(?:Here\'s the fix|The fix is):\s*\n(.*?)(?:\n\n|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re_.DOTALL | re_.IGNORECASE)
        if match:
            return match.group(1).strip()

    # If all else fails, return the original content
    # This prevents breaking the document further
    return original


async def fix_typst_errors(
    markdown_content: str, validation_results: list[TypstValidationResult], logger_name: str, max_attempts: int = 3
) -> tuple[str, bool]:
    """
    Fix Typst errors in markdown content using LLM.

    Returns:
        (fixed_content, all_fixed): The fixed markdown content and whether all errors were resolved
    """
    logger = get_logger(logger_name)
    invalid_results = get_invalid_blocks(validation_results)

    if not invalid_results:
        return markdown_content, True

    logger.info("Attempting to fix %d Typst error(s) using LLM", len(invalid_results))

    current_content = markdown_content

    # Group fixes by content to avoid duplicate work
    unique_fixes: dict[str, tuple[str, str, str]] = {}  # content -> (error_message, location, block_type)
    for result in invalid_results:
        content_key = result.block.content
        if content_key not in unique_fixes:
            unique_fixes[content_key] = (
                result.error_message,
                result.block.location,
                result.block.block_type,
            )

    # Fix each unique error
    fixed_mapping: dict[str, str] = {}

    for attempt in range(max_attempts):
        remaining_fixes = {k: v for k, v in unique_fixes.items() if k not in fixed_mapping}

        if not remaining_fixes:
            break

        logger.info("Fix attempt %d/%d for %d error(s)", attempt + 1, max_attempts, len(remaining_fixes))

        # Process fixes sequentially for now (can be parallelized later if needed)
        for original_content, (error_message, location, block_type) in remaining_fixes.items():
            try:
                fixed_content = await fix_single_typst_error(
                    original_content=original_content,
                    error_message=error_message,
                    location=location,
                    block_type=block_type,
                    model=settings.model,
                )
                if fixed_content != original_content:
                    fixed_mapping[original_content] = fixed_content
                    logger.info("Successfully fixed Typst block: %s", original_content[:50] + "...")
                else:
                    logger.warning("LLM returned unchanged content for: %s", original_content[:50] + "...")
            except (OSError, RuntimeError, ValueError, TypeError):
                logger.exception("Exception during fix for content: %s", original_content[:50] + "...")

        # If we made some progress, break
        if fixed_mapping:
            break

    # Apply fixes to markdown content
    if fixed_mapping:
        # Extract all blocks (not just invalid ones) for reconstruction
        all_blocks = [result.block for result in validation_results]
        current_content = reconstruct_markdown_with_fixes(current_content, all_blocks, fixed_mapping)

        logger.info("Applied %d Typst fix(es) to markdown content", len(fixed_mapping))

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
