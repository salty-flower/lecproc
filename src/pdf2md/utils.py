from pathlib import Path

import typst
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


def check_typst_syntax(code: str) -> tuple[bool, typst.TypstError | list[typst.TypstWarning] | str]:
    """Check Typst source for syntax/compilation errors using the typst Python package.

    Implementation notes:
    - Uses `typst.compile_with_warnings(input)` which is provided by typst-py.
    - If compilation raises `typst.TypstError` this is considered a syntax/compile error;
      the function returns (False, diagnostics) where diagnostics is a joined string
      containing the structured error message, hints and trace (when available).
    - If compilation succeeds (even with warnings) the function returns (True, "").
    - If the expected API is not present on the installed `typst` package, a RuntimeError is raised.

    Returns:
        (True, "") on full success (syntax OK)
        (True, "<warnings>") if compiled with warnings
        (False, "<diagnostics>") on syntax/compile error or other failures to validate
    """
    compile_with_warnings = typst.compile_with_warnings
    try:
        # typst.compile_with_warnings(input) -> (compiled_bytes_or_none, list_of_TypstWarning)
        # Encode string to bytes so it's treated as content, not file path
        _, warnings = compile_with_warnings(code.encode("utf-8"))
        # Warnings do not make the syntax invalid. Return success.
    except typst.TypstError as te:
        return False, te
    else:
        return True, warnings
