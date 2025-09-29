from pathlib import Path

from natsort import natsorted

from .settings import settings


def is_pdf_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == ".pdf"


def discover_pdf_files(root: Path) -> list[Path]:  # type: ignore[return]
    """Return sorted list of PDF files under `root`.

    - If `root` is a single PDF file, return it.
    - If `root` is a directory, return files in that directory (non-recursive) filtered by .pdf suffix.
    - If `root` does not exist or is not a PDF/file, return empty list.
    """
    base = root.resolve().absolute()
    match (base.is_file(), base.exists()):
        case (True, True):
            return [base] if is_pdf_file(base) else []
        case (False, True):
            return natsorted((p for p in base.glob("*") if is_pdf_file(p)), key=lambda p: str(p).lower())
        case (_, False):
            return []


def output_path_for(pdf_path: Path) -> Path:
    return pdf_path.with_suffix(f".{settings.output_extension}")


def format_display_path(path: Path, base: Path) -> str:
    """Return a readable path for logs: relative to `base` when possible, else absolute."""
    target_abs = path.resolve().absolute()
    base_abs = base.resolve().absolute()
    try:
        return str(target_abs.relative_to(base_abs))
    except ValueError:
        return str(target_abs)
