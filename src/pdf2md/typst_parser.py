"""Typst parsing and validation utilities for markdown documents.

This module was migrated from Mistune to markdown-it-py + mdformat.
"""

import abc
from collections.abc import Callable, Iterable
from typing import ClassVar, override

from markdown_it import MarkdownIt
from markdown_it.token import Token
from mdformat.renderer import MDRenderer, RenderContext, RenderTreeNode
from mdit_py_plugins.dollarmath.index import dollarmath_plugin
from pydantic import BaseModel, Field

from logs import get_logger

from .settings import settings

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


def _normalize_info_string(info: str | None) -> str:
    """Normalize a code fence info string for Typst code blocks.

    Returns "typ" for variations of Typst language labels.
    """
    if not info:
        return ""
    lowered = info.strip().lower()
    if lowered in ("typst", "typ"):
        return "typ"
    return lowered


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


class TypstMdformatExtension:
    """Minimal mdformat parser extension for Typst math/code rendering."""

    CHANGES_AST: ClassVar[bool] = False

    @staticmethod
    def update_mdit(mdit: MarkdownIt) -> None:
        # Enable $...$ and $$...$$ support
        _: MarkdownIt = mdit.use(dollarmath_plugin, double_inline=True)

    # Renderer functions for dollarmath tokens
    @staticmethod
    def _render_math_inline(node: "RenderTreeNode", _: RenderContext) -> str:
        return f"${node.content}$"

    @staticmethod
    def _render_math_block(node: "RenderTreeNode", _: RenderContext) -> str:
        content = node.content.strip("\n$`")  # remove backticks and $ delimiters
        return f"$${content}$$"

    RENDERERS: ClassVar[dict[str, Callable[[RenderTreeNode, RenderContext], str]]] = {
        "math_inline": _render_math_inline,
        "inline_math": _render_math_inline,  # alias seen in some plugin versions
        "math_inline_double": _render_math_block,  # render inline-double as display
        "math_block": _render_math_block,
        "block_math": _render_math_block,  # alias seen in some plugin versions
        "math_block_label": _render_math_block,  # labeled display math
    }


class ProtocolMDRenderer(MDRenderer):
    """
    Override the __output__ class variable to ClassVar[str]
    to satisfy RendererProtocol constraints in markdown-it-py
    """

    __output__: ClassVar[str] = "md"  # type: ignore[misc]


def _build_mdformat_mdit() -> MarkdownIt:
    """Build a MarkdownIt instance configured like mdformat with our extension."""
    mdit = MarkdownIt(renderer_cls=ProtocolMDRenderer)
    # mdformat options container + behavior flags used by renderer
    mdit.options["mdformat"] = {}
    mdit.options["store_labels"] = True
    # Register our extension directly (no entry points needed)
    mdit.options["parser_extension"] = [TypstMdformatExtension]
    TypstMdformatExtension.update_mdit(mdit)
    return mdit


def _walk_tokens_for_typst(tokens: Iterable[Token], markdown_content: str) -> list[TypstBlock]:
    """Walk markdown-it tokens and extract Typst-related blocks.

    - Code fences with info strings "typ"/"typst" are treated as Typst code.
    - Inline and block dollar-math (via ``dollarmath``) are treated as Typst math.
    """

    typst_blocks: list[TypstBlock] = []

    for token_index, token in enumerate(tokens):
        ttype = token.type

        # Code fence: token.type == "fence"
        if ttype == "fence":
            info = _normalize_info_string(token.info)
            if info == "typ":
                code = token.content
                token_map = token.map
                # map is [start, end], 0-based lines of the block (inclusive start, exclusive end)
                # Convert to 1-based line numbers; try to be conservative for context
                line_start = (token_map[0] + 1) if token_map else 1
                line_end = (token_map[1]) if token_map else max(line_start, line_start + code.count("\n"))

                context_before, context_after = _extract_context_lines(
                    markdown_content, line_start, line_end, settings.context_lines
                )

                typst_blocks.append(
                    CodeblockTypstBlock(
                        content=code,
                        location=f"code block (line ~{line_start})",
                        line_start=line_start,
                        line_end=line_end,
                        ast_path=[token_index],
                        context_before=context_before,
                        context_after=context_after,
                    )
                )
            continue

        # Dollar math block via plugin: commonly "math_block" (also handle legacy "block_math")
        # also handle labeled variant "math_block_label"
        if ttype in ("math_block", "block_math", "math_block_label"):
            content = token.content
            token_map = token.map
            line_start = (token_map[0] + 1) if token_map else 1
            line_end = (token_map[1]) if token_map else max(line_start, line_start + content.count("\n"))

            context_before, context_after = _extract_context_lines(
                markdown_content, line_start, line_end, settings.context_lines
            )

            typst_blocks.append(
                BlockTypstBlock(
                    content=content.strip(),
                    location=f"block formula (line ~{line_start})",
                    line_start=line_start,
                    line_end=line_end,
                    ast_path=[token_index],
                    context_before=context_before,
                    context_after=context_after,
                )
            )
            continue

        # Inline tokens contain children; search for inline math
        if ttype == "inline":
            token_map = token.map
            line_start = (token_map[0] + 1) if token_map else 1
            # For inline, we'll use the starting line for location/context
            children: list[Token] = token.children or []
            math_seen = 0
            for child in children:
                ctype = child.type
                if ctype in ("math_inline", "inline_math", "math_inline_double"):
                    content = child.content

                    context_before, context_after = _extract_context_lines(
                        markdown_content, line_start, line_start, settings.context_lines
                    )

                    typst_blocks.append(
                        InlineTypstBlock(
                            content=content.strip(),
                            location=f"inline formula (line ~{line_start})",
                            line_start=line_start,
                            line_end=line_start,
                            # ast_path[1] stores the inline-math occurrence index on this line
                            ast_path=[token_index, math_seen],
                            context_before=context_before,
                            context_after=context_after,
                        )
                    )
                    math_seen += 1

    return typst_blocks


def extract_typst_blocks(markdown_content: str) -> list[TypstBlock]:
    """Extract all Typst code blocks and formulas using markdown-it-py with ``dollarmath``.

    Returns blocks with best-effort line numbers, context, and a simple token path.
    """
    md = _build_mdformat_mdit()
    tokens = md.parse(markdown_content)
    return _walk_tokens_for_typst(tokens, markdown_content)


def _apply_token_fixes(tokens: list[Token], typst_blocks: list[TypstBlock], fixed_contents: dict[str, str]) -> None:
    """Apply fixes directly into markdown-it token stream based on stored paths."""
    for block in typst_blocks:
        if block.content not in fixed_contents:
            continue
        new_content = fixed_contents[block.content]

        # Navigate to token via stored top-level index
        if not block.ast_path:
            continue
        top_index = block.ast_path[0]
        if top_index < 0 or top_index >= len(tokens):
            continue
        tok = tokens[top_index]

        match block:
            case CodeblockTypstBlock():
                if tok.type == "fence":
                    tok.content = new_content
                    tok.info = "typ"
            case BlockTypstBlock():
                if tok.type in ("math_block", "block_math", "math_block_label"):
                    tok.content = new_content
            case InlineTypstBlock():
                if tok.type == "inline" and tok.children:
                    occurrence = block.ast_path[1] if len(block.ast_path) > 1 else 0
                    seen = 0
                    for child in tok.children:
                        if child.type in ("math_inline", "inline_math", "math_inline_double"):
                            if seen == occurrence:
                                child.content = new_content
                                break
                            seen += 1


def reconstruct_markdown_with_fixes(
    original_content: str, typst_blocks: list[TypstBlock], fixed_contents: dict[str, str]
) -> str:
    """Reconstruct markdown by mutating the markdown-it token stream and rendering via mdformat."""
    if not fixed_contents:
        return original_content

    # Build mdformat-configured parser/renderer and parse
    md = _build_mdformat_mdit()
    tokens = md.parse(original_content)

    # Apply in-place fixes to tokens
    _apply_token_fixes(tokens, typst_blocks, fixed_contents)

    # Render back to Markdown using mdformat's MDRenderer
    rendered = str(md.renderer.render(tokens, md.options, {}))  # pyright: ignore[reportAny]

    # Return without an additional global formatting pass to minimize diffs
    logger.info("Successfully applied %d token-based fixes", len(fixed_contents))
    return rendered
