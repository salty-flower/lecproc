"""Models for tracking Typst fix progress with AST-based reconstruction."""

from pathlib import Path
from typing import Any

import orjson
from pydantic import BaseModel


class TypstFixEntry(BaseModel):
    """A single Typst fix entry with original and fixed content."""

    original_content: str
    fixed_content: str
    ast_path: list[int]  # Path to AST node for precise replacement
    block_type: str  # "inline", "block", or "codeblock"


class TypstFixProgress(BaseModel):
    """Complete fix progress with metadata."""

    content_hash: str
    fixes: dict[str, TypstFixEntry]  # content -> fix entry

    def save_to_file(self, progress_file: Path) -> None:
        """Save progress to file using orjson for performance."""
        try:
            with progress_file.open("wb") as f:
                f.write(orjson.dumps(self.model_dump(), option=orjson.OPT_INDENT_2))
        except (OSError, ValueError):
            pass  # Ignore save errors to avoid disrupting the main process

    @classmethod
    def load_from_file(cls, progress_file: Path) -> "TypstFixProgress | None":
        """Load progress from file, return None if not found or corrupted."""
        if not progress_file.exists():
            return None

        try:
            with progress_file.open("rb") as f:
                data = orjson.loads(f.read())
                return cls.model_validate(data)
        except (OSError, ValueError, orjson.JSONDecodeError):
            return None

    def cleanup_file(self, progress_file: Path) -> None:
        """Remove the progress file after successful completion."""
        try:
            if progress_file.exists():
                progress_file.unlink()
        except OSError:
            pass  # Ignore cleanup errors
