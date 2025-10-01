from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from natsort import natsorted

from .settings import settings


@dataclass(frozen=True)
class InputFileMetadata:
    """Metadata describing a supported source document."""

    mime_type: str
    prompt_label: str
    category: Literal["pdf", "image"]


_PDF_METADATA = InputFileMetadata(
    mime_type="application/pdf",
    prompt_label="PDF file",
    category="pdf",
)

_PNG_METADATA = InputFileMetadata(
    mime_type="image/png",
    prompt_label="image file",
    category="image",
)

_JPEG_METADATA = InputFileMetadata(
    mime_type="image/jpeg",
    prompt_label="image file",
    category="image",
)

_WEBP_METADATA = InputFileMetadata(
    mime_type="image/webp",
    prompt_label="image file",
    category="image",
)

SUPPORTED_INPUT_FILE_TYPES: dict[str, InputFileMetadata] = {
    ".pdf": _PDF_METADATA,
    ".png": _PNG_METADATA,
    ".jpg": _JPEG_METADATA,
    ".jpeg": _JPEG_METADATA,
    ".webp": _WEBP_METADATA,
}


def supported_extensions_display() -> str:
    """Return a human-readable list of supported extensions."""

    return ", ".join(sorted(ext.lstrip(".") for ext in SUPPORTED_INPUT_FILE_TYPES))


def get_input_file_metadata(path: Path) -> InputFileMetadata | None:
    """Return metadata for the given path if it is a supported source file."""

    if not path.is_file():
        return None
    return SUPPORTED_INPUT_FILE_TYPES.get(path.suffix.lower())


def is_supported_input_file(path: Path) -> bool:
    return get_input_file_metadata(path) is not None


def discover_source_files(root: Path) -> list[Path]:  # type: ignore[return]
    """Return sorted list of supported source files under `root`.

    - If `root` is a single supported file, return it.
    - If `root` is a directory, return files in that directory (non-recursive)
      filtered by supported extensions.
    - If `root` does not exist or contains no supported files, return empty list.
    """

    base = root.resolve().absolute()
    match (base.is_file(), base.exists()):
        case (True, True):
            return [base] if is_supported_input_file(base) else []
        case (False, True):
            return natsorted(
                (p for p in base.glob("*") if is_supported_input_file(p)),
                key=lambda p: str(p).lower(),
            )
        case (_, False):
            return []


def output_path_for(source_path: Path) -> Path:
    return source_path.with_suffix(f".{settings.output_extension}")


def detect_output_collisions(source_files: Iterable[Path]) -> dict[Path, list[Path]]:
    """Return mapping of conflicting output paths to their source files."""

    grouped: dict[Path, list[Path]] = defaultdict(list)
    for source_path in source_files:
        grouped[output_path_for(source_path)].append(source_path)
    return {output_path: paths for output_path, paths in grouped.items() if len(paths) > 1}


def format_display_path(path: Path, base: Path) -> str:
    """Return a readable path for logs: relative to `base` when possible, else absolute."""

    target_abs = path.resolve().absolute()
    base_abs = base.resolve().absolute()
    try:
        return str(target_abs.relative_to(base_abs))
    except ValueError:
        return str(target_abs)
