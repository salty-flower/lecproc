"""Models for tracking Typst fix progress with AST-based reconstruction."""

from pathlib import Path

import anyio
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

    async def save_to_file(self, progress_file: Path) -> int:
        """Save progress to file using orjson for performance."""

        async with await anyio.open_file(progress_file, "wb") as f:
            return await f.write(orjson.dumps(self.model_dump(), option=orjson.OPT_INDENT_2))

    @classmethod
    async def load_from_file(cls, progress_file: Path) -> "TypstFixProgress | None":
        """Load progress from file, return None if not found or corrupted."""
        if not progress_file.exists():
            return None

        async with await anyio.open_file(progress_file, "rb") as f:
            data = await f.read()
            return cls.model_validate(orjson.loads(data))

    def cleanup_file(self, progress_file: Path) -> None:
        """Remove the progress file after successful completion."""
        try:
            if progress_file.exists():
                progress_file.unlink()
        except OSError:
            pass  # Ignore cleanup errors
