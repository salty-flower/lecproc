"""Typst parsing and validation utilities for markdown documents."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import mistune

from logs import get_logger

if TYPE_CHECKING:
    from mistune.markdown import Markdown

logger = get_logger(__name__)


@dataclass
class TypstBlock:
    """Represents a Typst code block or formula in markdown."""

    content: str
    location: str  # Description of where this block was found
    block_type: str  # "formula" or "codeblock"
    line_start: int = -1
    line_end: int = -1


def _walk_ast_for_typst(tokens: list[dict[str, Any]] | str, line_counter: int = 1) -> list[TypstBlock]:
    """Walk AST tokens and extract Typst blocks."""
    typst_blocks: list[TypstBlock] = []
    if not isinstance(tokens, list):
        logger.warning("Expected list of typst tokens, got type %s:\n%s", type(tokens), str(tokens))
        return typst_blocks

    for token in tokens:
        token_type = str(token.get("type", ""))  # pyright: ignore[reportAny]

        match token_type:
            case "code_block":
                # Check for Typst code blocks
                info = str(token.get("info", "")).strip().lower()  # pyright: ignore[reportAny]
                if info in ("typ", "typst"):
                    code = str(token.get("raw", ""))  # pyright: ignore[reportAny]
                    block = TypstBlock(
                        content=code,
                        location=f"code block (line ~{line_counter})",
                        block_type="codeblock",
                        line_start=line_counter,
                        line_end=line_counter + code.count("\n"),
                    )
                    typst_blocks.append(block)

            case "block_math":
                # Always extract block math for compilation validation
                text = str(token.get("raw", ""))  # pyright: ignore[reportAny]
                block = TypstBlock(
                    content=text,
                    location=f"block formula (line ~{line_counter})",
                    block_type="formula",
                    line_start=line_counter,
                    line_end=line_counter + text.count("\n"),
                )
                typst_blocks.append(block)

            case "inline_math":
                # Always extract inline math for compilation validation
                text = str(token.get("raw", ""))  # pyright: ignore[reportAny]
                block = TypstBlock(
                    content=text,
                    location=f"inline formula (line ~{line_counter})",
                    block_type="formula",
                    line_start=line_counter,
                    line_end=line_counter,
                )
                typst_blocks.append(block)

            case _:
                pass

        # Recursively check children
        if "children" in token:
            child_blocks = _walk_ast_for_typst(token["children"], line_counter)  # pyright: ignore[reportAny]
            typst_blocks.extend(child_blocks)

        # Update line counter (rough approximation)
        if "raw" in token:
            line_counter += token["raw"].count("\n")  # pyright: ignore[reportAny]
        line_counter += 1

    return typst_blocks


def extract_typst_blocks(markdown_content: str) -> list[TypstBlock]:
    """Extract all Typst code blocks and formulas from markdown content."""
    parser: Markdown = mistune.create_markdown(renderer="ast", plugins=["math"])  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    # Parse to AST
    ast_tokens = parser(markdown_content)  # pyright: ignore[reportUnknownVariableType]

    # Walk AST and extract Typst blocks
    return _walk_ast_for_typst(ast_tokens)  # pyright: ignore[reportUnknownArgumentType]


def reconstruct_markdown_with_fixes(
    original_content: str, typst_blocks: list[TypstBlock], fixed_contents: dict[str, str]
) -> str:
    """Reconstruct markdown with fixed Typst content."""
    result = original_content

    # Sort blocks by line number in reverse order to avoid offset issues
    sorted_blocks = sorted(typst_blocks, key=lambda b: b.line_start, reverse=True)

    for block in sorted_blocks:
        if block.content in fixed_contents:
            fixed_content = fixed_contents[block.content]

            if block.block_type == "codeblock":
                # Replace code block content
                old_pattern = f"```typ\n{block.content}\n```"
                new_content = f"```typ\n{fixed_content}\n```"
                result = result.replace(old_pattern, new_content)
            elif block.block_type == "formula":
                # Replace formula content
                if block.location.startswith("inline"):
                    result = result.replace(f"${block.content}$", f"${fixed_content}$")
                else:
                    result = result.replace(f"$${block.content}$$", f"$${fixed_content}$$")

    return result
