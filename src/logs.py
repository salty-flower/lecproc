import logging
from typing import Final

from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.traceback import install as install_rich_traceback

_CONFIGURED: Final[dict[str, bool]] = {"rich_logging": False}


def configure_rich_logging(
    level: int | str = logging.INFO, markup: bool = False
) -> None:
    """Idempotently configure Rich logging and tracebacks for the process.

    Safe to call multiple times; configuration will only be applied once.
    """
    if _CONFIGURED["rich_logging"]:
        return

    _ = install_rich_traceback(show_locals=False)
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, markup=markup)],
    )
    _CONFIGURED["rich_logging"] = True


def get_logger(name: str) -> logging.Logger:
    """Get a logger configured with Rich handler."""
    configure_rich_logging()
    return logging.getLogger(name)


def create_progress() -> Progress:
    """Create a pre-configured Rich Progress instance.

    Caller is responsible for entering the context manager:
        with create_progress() as progress: ...
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(pulse_style="cyan"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )


__all__ = [
    "configure_rich_logging",
    "get_logger",
    "create_progress",
    "TaskID",
]
