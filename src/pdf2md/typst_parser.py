"""Typst parsing and validation utilities for markdown documents."""

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mistune
from mistune.core import BaseRenderer

if TYPE_CHECKING:
    from mistune.markdown import Markdown


@dataclass
class TypstBlock:
    """Represents a Typst code block or formula in markdown."""

    content: str
    location: str  # Description of where this block was found
    block_type: str  # "formula" or "codeblock"
    line_start: int = -1
    line_end: int = -1


class TypstExtractor(BaseRenderer):
    """Mistune renderer that extracts Typst content from markdown."""

    def __init__(self) -> None:
        super().__init__()
        self.typst_blocks: list[TypstBlock] = []
        self._current_line: int = 1

    def text(self, text: str) -> str:
        # Count newlines to track line numbers
        self._current_line += text.count("\n")
        return text

    def code_block(self, code: str, info: str | None = None) -> str:
        """Extract Typst code blocks marked with 'typ' or 'typst' language."""
        if info and info.strip().lower() in ("typ", "typst"):
            block = TypstBlock(
                content=code,
                location=f"code block (line ~{self._current_line})",
                block_type="codeblock",
                line_start=self._current_line,
                line_end=self._current_line + code.count("\n"),
            )
            self.typst_blocks.append(block)

        self._current_line += code.count("\n") + 2  # +2 for the ``` lines
        return f"```{info or ''}\n{code}\n```"

    def inline_math(self, text: str) -> str:
        """Extract inline math that might be Typst formulas."""
        # Check if this looks like Typst syntax (contains # symbols, typical Typst functions)
        if self._looks_like_typst(text):
            block = TypstBlock(
                content=text,
                location=f"inline formula (line ~{self._current_line})",
                block_type="formula",
                line_start=self._current_line,
                line_end=self._current_line,
            )
            self.typst_blocks.append(block)
        return f"${text}$"

    def block_math(self, text: str) -> str:
        """Extract block math that might be Typst formulas."""
        # Check if this looks like Typst syntax
        if self._looks_like_typst(text):
            block = TypstBlock(
                content=text,
                location=f"block formula (line ~{self._current_line})",
                block_type="formula",
                line_start=self._current_line,
                line_end=self._current_line + text.count("\n"),
            )
            self.typst_blocks.append(block)

        self._current_line += text.count("\n") + 2  # +2 for the $$ lines
        return f"$${text}$$"

    def _looks_like_typst(self, text: str) -> bool:
        """Heuristic to determine if math content looks like Typst syntax."""
        # Typst-specific patterns
        typst_patterns = [
            r"#[a-zA-Z_][a-zA-Z0-9_]*",  # Typst function calls like #sum, #frac
            r"#\(",  # Typst expression blocks
            r"\$[^$]*\$",  # Typst inline math
            r"attach\(",  # Typst attach function
            r"cases\(",  # Typst cases function
            r"mat\(",  # Typst matrix function
        ]

        return any(re.search(pattern, text) for pattern in typst_patterns)

    # Default implementations for other elements
    def paragraph(self, text: str) -> str:
        self._current_line += text.count("\n") + 1
        return text + "\n\n"

    def heading(self, text: str, level: int, **_attrs: object) -> str:
        self._current_line += 1
        return "#" * level + " " + text + "\n\n"

    def list_item(self, text: str) -> str:
        return f"- {text}\n"

    def list(self, text: str, _ordered: bool, **_attrs: object) -> str:
        return text + "\n"

    def link(self, text: str, url: str, _title: str | None = None) -> str:
        return f"[{text}]({url})"

    def emphasis(self, text: str) -> str:
        return f"*{text}*"

    def strong(self, text: str) -> str:
        return f"**{text}**"

    def code(self, text: str) -> str:
        return f"`{text}`"

    def image(self, alt: str, url: str, _title: str | None = None) -> str:
        return f"![{alt}]({url})"

    def linebreak(self) -> str:
        self._current_line += 1
        return "\n"

    def thematic_break(self) -> str:
        self._current_line += 1
        return "---\n"


def extract_typst_blocks(markdown_content: str) -> list[TypstBlock]:
    """Extract all Typst code blocks and formulas from markdown content."""
    renderer = TypstExtractor()
    parser: Markdown = mistune.create_markdown(renderer=renderer, plugins=["math"])  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]

    # Parse the markdown (this will populate renderer.typst_blocks)
    _ = parser(markdown_content)  # pyright: ignore[reportUnknownVariableType]

    return renderer.typst_blocks


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
                old_pattern = f"```typ\n{re.escape(block.content)}\n```"
                new_content = f"```typ\n{fixed_content}\n```"
                result = result.replace(old_pattern, new_content)
            elif block.block_type == "formula":
                # Replace formula content
                if block.location.startswith("inline"):
                    result = result.replace(f"${block.content}$", f"${fixed_content}$")
                else:
                    result = result.replace(f"$${block.content}$$", f"$${fixed_content}$$")

    return result
