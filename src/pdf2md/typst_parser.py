"""Typst parsing and validation utilities for markdown documents."""

from typing import TYPE_CHECKING, Any, Literal

import mistune
from pydantic import BaseModel, Field

from logs import get_logger

if TYPE_CHECKING:
    from mistune.markdown import Markdown

logger = get_logger(__name__)


class InlineTypstBlock(BaseModel):
    """Inline Typst math: $content$ in markdown, content only for validation."""

    type: Literal["inline"] = "inline"
    content: str = Field(description="Pure math content without $ delimiters")
    location: str = Field(description="Where this block was found")
    line_start: int = -1
    line_end: int = -1
    ast_path: list[int] = Field(default_factory=list, description="Path to AST node for precise replacement")
    context_before: list[str] = Field(default_factory=list, description="Context lines before this block")
    context_after: list[str] = Field(default_factory=list, description="Context lines after this block")

    def get_markdown_form(self) -> str:
        """Get the markdown representation."""
        return f"${self.content}$"

    def get_validation_form(self) -> str:
        """Get content for Typst compiler validation."""
        return self.content  # Pure content, no delimiters

    def get_context_for_llm(self) -> str:
        """Get context information for LLM (without AST path)."""
        context_parts = []
        if self.context_before:
            context_parts.append("Context before:")
            context_parts.extend(f"  {line}" for line in self.context_before)

        context_parts.append(f"Typst {self.type} content: {self.content}")

        if self.context_after:
            context_parts.append("Context after:")
            context_parts.extend(f"  {line}" for line in self.context_after)

        return "\n".join(context_parts)


class BlockTypstBlock(BaseModel):
    """Block Typst math: $$content$$ in markdown, single $ for validation."""

    type: Literal["block"] = "block"
    content: str = Field(description="Pure math content without $$ delimiters")
    location: str = Field(description="Where this block was found")
    line_start: int = -1
    line_end: int = -1
    ast_path: list[int] = Field(default_factory=list, description="Path to AST node for precise replacement")
    context_before: list[str] = Field(default_factory=list, description="Context lines before this block")
    context_after: list[str] = Field(default_factory=list, description="Context lines after this block")

    def get_markdown_form(self) -> str:
        """Get the markdown representation."""
        return f"$${self.content}$$"

    def get_validation_form(self) -> str:
        """Get content for Typst compiler validation."""
        return f"${self.content}$"  # Single dollars for block math validation

    def get_context_for_llm(self) -> str:
        """Get context information for LLM (without AST path)."""
        context_parts = []
        if self.context_before:
            context_parts.append("Context before:")
            context_parts.extend(f"  {line}" for line in self.context_before)

        context_parts.append(f"Typst {self.type} content: {self.content}")

        if self.context_after:
            context_parts.append("Context after:")
            context_parts.extend(f"  {line}" for line in self.context_after)

        return "\n".join(context_parts)


class CodeblockTypstBlock(BaseModel):
    """Typst code block: ```typst content ``` - direct compilation."""

    type: Literal["codeblock"] = "codeblock"
    content: str = Field(description="Full Typst code for direct compilation")
    location: str = Field(description="Where this block was found")
    line_start: int = -1
    line_end: int = -1
    ast_path: list[int] = Field(default_factory=list, description="Path to AST node for precise replacement")
    context_before: list[str] = Field(default_factory=list, description="Context lines before this block")
    context_after: list[str] = Field(default_factory=list, description="Context lines after this block")

    def get_markdown_form(self) -> str:
        """Get the markdown representation."""
        return f"```typst\n{self.content}\n```"

    def get_validation_form(self) -> str:
        """Get content for Typst compiler validation."""
        return self.content  # Direct compilation

    def get_context_for_llm(self) -> str:
        """Get context information for LLM (without AST path)."""
        context_parts = []
        if self.context_before:
            context_parts.append("Context before:")
            context_parts.extend(f"  {line}" for line in self.context_before)

        context_parts.append(f"Typst {self.type} content: {self.content}")

        if self.context_after:
            context_parts.append("Context after:")
            context_parts.extend(f"  {line}" for line in self.context_after)

        return "\n".join(context_parts)


# Discriminated union of all Typst block types
TypstBlock = InlineTypstBlock | BlockTypstBlock | CodeblockTypstBlock


def _extract_pure_content(raw_content: str, token_type: str) -> str:
    """Extract pure content from raw mistune token, removing delimiters."""
    content = raw_content.strip()

    match token_type:
        case "inline_math":
            # Remove single $ delimiters: "$x=y$" -> "x=y"
            if content.startswith("$") and content.endswith("$"):
                return content[1:-1]
        case "block_math":
            # Remove double $$ delimiters: "$$x=y$$" -> "x=y"
            if content.startswith("$$") and content.endswith("$$"):
                return content[2:-2]
        case _:
            # Not our business
            pass

    # For codeblocks or if delimiters not found, return as-is
    return content


def _extract_context_lines(
    markdown_content: str, line_start: int, line_end: int, context_lines: int
) -> tuple[list[str], list[str]]:
    """Extract context lines before and after a block."""
    lines = markdown_content.splitlines()
    total_lines = len(lines)

    # Convert to 0-based indexing (line_start/line_end are 1-based)
    start_idx = max(0, line_start - 1)
    end_idx = min(total_lines, line_end)

    # Extract context before
    context_before_start = max(0, start_idx - context_lines)
    context_before = lines[context_before_start:start_idx] if start_idx > 0 else []

    # Extract context after
    context_after_end = min(total_lines, end_idx + context_lines)
    context_after = lines[end_idx:context_after_end] if end_idx < total_lines else []

    return context_before, context_after


def _walk_ast_for_typst(
    tokens: list[dict[str, Any]] | str, markdown_content: str, ast_path: list[int] | None = None, line_counter: int = 1
) -> list[TypstBlock]:
    """Walk AST tokens and extract Typst blocks with AST paths and context."""
    from .settings import settings

    typst_blocks: list[TypstBlock] = []
    if not isinstance(tokens, list):
        logger.warning("Expected list of typst tokens, got type %s:\n%s", type(tokens), str(tokens))
        return typst_blocks

    if ast_path is None:
        ast_path = []

    for token_idx, token in enumerate(tokens):
        current_path = ast_path + [token_idx]
        token_type = str(token.get("type", ""))  # pyright: ignore[reportAny]

        block: TypstBlock
        match token_type:
            case "code_block":
                # Check for Typst code blocks
                info = str(token.get("info", "")).strip().lower()  # pyright: ignore[reportAny]
                if info in ("typ", "typst"):
                    code = str(token.get("raw", ""))  # pyright: ignore[reportAny]

                    # Extract context lines
                    context_before, context_after = _extract_context_lines(
                        markdown_content, line_counter, line_counter + code.count("\n"), settings.context_lines
                    )

                    block = CodeblockTypstBlock(
                        content=code,
                        location=f"code block (line ~{line_counter})",
                        line_start=line_counter,
                        line_end=line_counter + code.count("\n"),
                        ast_path=current_path,
                        context_before=context_before,
                        context_after=context_after,
                    )
                    typst_blocks.append(block)

            case "block_math":
                # Extract block math with pure content (no $$ delimiters)
                raw_text = str(token.get("raw", ""))  # pyright: ignore[reportAny]
                pure_content = _extract_pure_content(raw_text, "block_math")

                # Extract context lines
                context_before, context_after = _extract_context_lines(
                    markdown_content, line_counter, line_counter + raw_text.count("\n"), settings.context_lines
                )

                block = BlockTypstBlock(
                    content=pure_content,
                    location=f"block formula (line ~{line_counter})",
                    line_start=line_counter,
                    line_end=line_counter + raw_text.count("\n"),
                    ast_path=current_path,
                    context_before=context_before,
                    context_after=context_after,
                )
                typst_blocks.append(block)

            case "inline_math":
                # Extract inline math with pure content (no $ delimiters)
                raw_text = str(token.get("raw", ""))  # pyright: ignore[reportAny]
                pure_content = _extract_pure_content(raw_text, "inline_math")

                # Extract context lines
                context_before, context_after = _extract_context_lines(
                    markdown_content, line_counter, line_counter, settings.context_lines
                )

                block = InlineTypstBlock(
                    content=pure_content,
                    location=f"inline formula (line ~{line_counter})",
                    line_start=line_counter,
                    line_end=line_counter,
                    ast_path=current_path,
                    context_before=context_before,
                    context_after=context_after,
                )
                typst_blocks.append(block)

            case _:
                pass

        # Recursively check children with updated path
        if "children" in token:
            child_blocks = _walk_ast_for_typst(
                token["children"],
                markdown_content,
                current_path,
                line_counter,  # pyright: ignore[reportAny]
            )
            typst_blocks.extend(child_blocks)

        # Update line counter (rough approximation)
        if "raw" in token:
            line_counter += token["raw"].count("\n")  # pyright: ignore[reportAny]
        line_counter += 1

    return typst_blocks


def extract_typst_blocks(markdown_content: str) -> list[TypstBlock]:
    """Extract all Typst code blocks and formulas from markdown content with AST paths and context."""
    parser: Markdown = mistune.create_markdown(renderer="ast", plugins=["math"])  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    # Parse to AST
    ast_tokens = parser(markdown_content)  # pyright: ignore[reportUnknownVariableType]

    # Walk AST and extract Typst blocks with context
    return _walk_ast_for_typst(ast_tokens, markdown_content)  # pyright: ignore[reportUnknownArgumentType]


def _navigate_ast_path(ast: list[dict[str, Any]], path: list[int]) -> dict[str, Any] | None:
    """Navigate to a specific AST node using the path."""
    current = ast

    for index in path:
        if not isinstance(current, list) or index >= len(current):
            logger.warning("Invalid AST path %s: index %d out of bounds", path, index)
            return None

        current_node = current[index]
        if not isinstance(current_node, dict):
            logger.warning("Invalid AST path %s: expected dict at index %d", path, index)
            return None

        # For the last element in path, return the node
        if index == path[-1]:
            return current_node

        # Continue navigating into children
        if "children" in current_node:
            current = current_node["children"]  # pyright: ignore[reportAny]
        else:
            logger.warning("Invalid AST path %s: no children at index %d", path, index)
            return None

    return None


def _apply_ast_fixes(ast: list[dict[str, Any]], typst_blocks: list[TypstBlock], fixed_contents: dict[str, str]) -> None:
    """Apply fixes directly to AST nodes using their paths."""
    for block in typst_blocks:
        if block.content in fixed_contents:
            fixed_content = fixed_contents[block.content]

            # Navigate to the specific AST node
            node = _navigate_ast_path(ast, block.ast_path)
            if node is None:
                logger.warning("Could not find AST node at path %s for block: %s", block.ast_path, block.content[:50])
                continue

            # Update the raw content based on block type
            try:
                match block.type:
                    case "inline":
                        node["raw"] = f"${fixed_content}$"  # pyright: ignore[reportAny]
                    case "block":
                        node["raw"] = f"$${fixed_content}$$"  # pyright: ignore[reportAny]
                    case "codeblock":
                        node["raw"] = fixed_content  # pyright: ignore[reportAny]

                logger.debug(
                    "Applied AST fix at path %s: %s -> %s", block.ast_path, block.content[:30], fixed_content[:30]
                )
            except (KeyError, TypeError) as e:
                logger.warning("Failed to apply AST fix at path %s: %s", block.ast_path, e)


def _render_ast_to_markdown(ast: list[dict[str, Any]]) -> str:
    """Render modified AST back to markdown."""
    # Create a custom renderer that preserves the raw content
    from mistune.renderers.markdown import MarkdownRenderer

    class PreservingMarkdownRenderer(MarkdownRenderer):
        """Custom renderer that preserves raw content for math blocks."""

        def inline_math(self, text: str) -> str:  # pyright: ignore[reportAny]
            return text  # Raw content already includes $...$

        def block_math(self, text: str) -> str:  # pyright: ignore[reportAny]
            return text  # Raw content already includes $$...$$

        def code_block(self, code: str, info: str | None = None) -> str:  # pyright: ignore[reportAny]
            if info and info.strip().lower() in ("typ", "typst"):
                return f"```typst\n{code}\n```"
            return f"```{info or ''}\n{code}\n```"

    try:
        renderer = PreservingMarkdownRenderer()
        # Render using mistune's internal mechanisms
        # Note: This is a simplified approach and may not handle all edge cases
        result_parts = []

        for token in ast:
            token_type = token.get("type", "")  # pyright: ignore[reportAny]

            if token_type == "inline_math":
                result_parts.append(token.get("raw", ""))  # pyright: ignore[reportAny]
            elif token_type == "block_math":
                result_parts.append(token.get("raw", ""))  # pyright: ignore[reportAny]
            elif token_type == "code_block":
                result_parts.append(token.get("raw", ""))  # pyright: ignore[reportAny]
            else:
                # For other token types, we'd need more complex rendering logic
                # This is a limitation of the current approach
                result_parts.append(str(token.get("raw", "")))  # pyright: ignore[reportAny]

        return "\n".join(result_parts)

    except Exception as e:
        logger.warning("Failed to render AST to markdown: %s", e)
        # Fallback to string replacement
        return ""


def reconstruct_markdown_with_fixes(
    original_content: str, typst_blocks: list[TypstBlock], fixed_contents: dict[str, str]
) -> str:
    """Reconstruct markdown with fixed Typst content using AST-based replacement."""
    if not fixed_contents:
        return original_content

    try:
        # Parse original content to AST
        parser: Markdown = mistune.create_markdown(renderer="ast", plugins=["math"])  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        ast = parser(original_content)  # pyright: ignore[reportUnknownVariableType]

        if not isinstance(ast, list):
            logger.warning("Expected list AST, got %s. Falling back to string replacement", type(ast))
            raise ValueError("Invalid AST structure")

        # Apply fixes to AST
        _apply_ast_fixes(ast, typst_blocks, fixed_contents)

        # Render modified AST back to markdown
        result = _render_ast_to_markdown(ast)

        if result:
            logger.info("Successfully applied %d AST-based fixes", len(fixed_contents))
            return result
        else:
            raise ValueError("AST rendering failed")

    except Exception as e:
        logger.warning("AST-based reconstruction failed (%s), falling back to string replacement", e)

        # Fallback to the previous string-based approach
        result = original_content

        # Sort blocks by line number in reverse order to avoid offset issues
        sorted_blocks = sorted(typst_blocks, key=lambda b: b.line_start, reverse=True)

        for block in sorted_blocks:
            if block.content in fixed_contents:
                fixed_content = fixed_contents[block.content]

                # Get original markdown form and create fixed markdown form
                original_markdown = block.get_markdown_form()

                match block.type:
                    case "inline":
                        fixed_markdown = f"${fixed_content}$"
                    case "block":
                        fixed_markdown = f"$${fixed_content}$$"
                    case "codeblock":
                        fixed_markdown = f"```typst\n{fixed_content}\n```"

                # Replace the original markdown form with the fixed form
                result = result.replace(original_markdown, fixed_markdown, 1)

        logger.info("Applied %d fixes using string replacement fallback", len(fixed_contents))
        return result
