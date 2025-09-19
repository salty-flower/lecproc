"""Typst parsing and validation utilities for markdown documents."""

import abc
from typing import TYPE_CHECKING, Any, ClassVar, override

import mistune
from mistune.core import BlockState
from mistune.renderers.markdown import MarkdownRenderer
from pydantic import BaseModel, Field

from logs import get_logger

from .settings import settings

if TYPE_CHECKING:
    from mistune.markdown import Markdown

logger = get_logger(__name__)


class BaseTypstBlock(BaseModel, abc.ABC):  # pyright: ignore[reportUnsafeMultipleInheritance]: pydantic [documentation allows](https://docs.pydantic.dev/latest/concepts/models/#abstract-base-classes)
    """Base class for Typst blocks."""

    type: ClassVar[str]

    content: str
    location: str = Field(description="Where this block was found")
    line_start: int
    line_end: int
    ast_path: list[int] = Field(default_factory=list, description="Path to AST node for precise replacement")
    context_before: list[str] = Field(default_factory=list, description="Context lines before this block")
    context_after: list[str] = Field(default_factory=list, description="Context lines after this block")

    @abc.abstractmethod
    def get_markdown_form(self) -> str: ...

    @abc.abstractmethod
    def get_validation_form(self) -> str: ...

    def get_context_for_llm(self) -> str:
        """Get context information for LLM (without AST path)."""
        context_parts: list[str] = []
        if self.context_before:
            context_parts.append("Context before:")
            context_parts.extend(f"  {line}" for line in self.context_before)

        context_parts.append(f"Typst {self.type} content: {self.content}")

        if self.context_after:
            context_parts.append("Context after:")
            context_parts.extend(f"  {line}" for line in self.context_after)

        return "\n".join(context_parts)


class InlineTypstBlock(BaseTypstBlock):
    type: ClassVar[str] = "inline"
    content: str = Field(description="Pure math content without $ delimiters")

    @override
    def get_markdown_form(self) -> str:
        """Get the markdown representation."""
        return f"${self.content}$"

    @override
    def get_validation_form(self) -> str:
        """Get content for Typst compiler validation."""
        return f"${self.content}$"


class BlockTypstBlock(BaseTypstBlock):
    type: ClassVar[str] = "block"
    content: str = Field(description="Pure math content without $$ delimiters")

    @override
    def get_markdown_form(self) -> str:
        """Get the markdown representation."""
        return f"$${self.content}$$"

    @override
    def get_validation_form(self) -> str:
        """Get content for Typst compiler validation."""
        return f"$ {self.content} $"


class CodeblockTypstBlock(BaseTypstBlock):
    type: ClassVar[str] = "codeblock"
    content: str = Field(description="Full Typst code for direct compilation")

    @override
    def get_markdown_form(self) -> str:
        """Get the markdown representation."""
        return f"```typ\n{self.content}\n```"

    @override
    def get_validation_form(self) -> str:
        """Get content for Typst compiler validation."""
        return self.content


# Discriminated union of all Typst block types
TypstBlock = InlineTypstBlock | BlockTypstBlock | CodeblockTypstBlock


def _extract_pure_content(raw_content: str, token_type: str) -> str:
    """Extract pure content from raw mistune token, removing delimiters."""
    content = raw_content.strip()

    match token_type:
        case "inline_math":
            # Remove single $ delimiters: "$x=y$" -> "x=y"
            if content.startswith("$") and content.endswith("$"):
                return content[1:-1].strip()
        case "block_math":
            # Remove double $$ delimiters: "$$x=y$$" -> "x=y"
            if content.startswith("$$") and content.endswith("$$"):
                return content[2:-2].strip()
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

    typst_blocks: list[TypstBlock] = []
    if not isinstance(tokens, list):
        logger.warning("Expected list of typst tokens, got type %s:\n%s", type(tokens), str(tokens))
        return typst_blocks

    if ast_path is None:
        ast_path = []

    for token_idx, token in enumerate(tokens):
        current_path = [*ast_path, token_idx]
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
                token["children"],  # pyright: ignore[reportAny]
                markdown_content,
                current_path,
                line_counter,
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
            current = current_node["children"]
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

            node["raw"] = block.model_copy(update={"content": fixed_content}).get_markdown_form()


def _render_ast_to_markdown(ast: list[dict[str, Any]]) -> str:
    """Render modified AST back to markdown using mistune's built-in renderer."""

    # Create a markdown renderer with math plugin support
    class MathMarkdownRenderer(MarkdownRenderer):
        def inline_math(self, token: dict[str, Any], _: BlockState) -> str:
            return token.get("raw", f"${token.get('raw', '')}$")  # pyright: ignore[reportAny]

        def block_math(self, token: dict[str, Any], _: BlockState) -> str:
            return token.get("raw", f"$${token.get('raw', '')}$$")  # pyright: ignore[reportAny]

        @override
        def block_code(self, token: dict[str, Any], state: BlockState) -> str:
            # Handle Typst code blocks - ensure they use 'typ' not 'typst'
            attrs = token.get("attrs", {})  # pyright: ignore[reportAny]
            info = str(attrs.get("info", ""))  # pyright: ignore[reportAny]
            code = str(token["raw"])  # pyright: ignore[reportAny]

            # If it's a typst block, make sure it uses 'typ'
            if info and info.strip().lower() in ("typst", "typ"):
                if code and code[-1] != "\n":
                    code += "\n"
                return f"```typ\n{code}```\n\n"

            # For other code blocks, use the parent implementation
            return super().block_code(token, state)  # pyright: ignore[reportUnknownMemberType]

    # Create the markdown renderer
    renderer = MathMarkdownRenderer()

    # Create a block state (required for the renderer)
    state = BlockState()

    # Render the AST back to markdown
    return renderer(ast, state)


def reconstruct_markdown_with_fixes(
    original_content: str, typst_blocks: list[TypstBlock], fixed_contents: dict[str, str]
) -> str:
    """Reconstruct markdown with fixed Typst content using AST-based replacement."""
    if not fixed_contents:
        return original_content

    # Use AST-based reconstruction (proper approach)

    # Parse original content to AST
    parser: Markdown = mistune.create_markdown(renderer="ast", plugins=["math"])  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    ast = parser(original_content)  # pyright: ignore[reportUnknownVariableType]

    if not isinstance(ast, list):
        logger.warning("Expected list AST, got %s. Falling back to string replacement", type(ast))  # pyright: ignore[reportUnknownArgumentType]
        msg = "Invalid AST structure"
        raise TypeError(msg)

    # Apply fixes to AST
    _apply_ast_fixes(ast, typst_blocks, fixed_contents)  # pyright: ignore[reportUnknownArgumentType]

    # Render modified AST back to markdown
    result = _render_ast_to_markdown(ast)

    if result:
        logger.info("Successfully applied %d AST-based fixes", len(fixed_contents))
        return result
    msg = "AST rendering failed"
    raise ValueError(msg)
