import base64
import logging
from asyncio import CancelledError
from pathlib import Path
from typing import Any, ClassVar, cast, override

import anyio
import litellm
from anyio import open_file
from litellm.exceptions import InternalServerError
from natsort import natsorted
from pydantic import computed_field
from pydantic_settings import CliPositionalArg, SettingsConfigDict
from rich.progress import Progress

from common_cli_settings import CommonCliSettings
from logs import TaskID, create_progress, get_logger

from .models import UserMessage, compose_pdf_user_messages
from .settings import settings


def format_display_path(path: Path, base: Path) -> str:
    """Return a readable path for logs: relative to `base` when possible, else absolute."""
    target_abs = path.resolve().absolute()
    base_abs = base.resolve().absolute()
    try:
        return str(target_abs.relative_to(base_abs))
    except ValueError:
        return str(target_abs)


def plan_processing(
    pdf_files: list[Path], overwrite: bool
) -> tuple[list[Path], list[Path], list[Path]]:
    """Plan which PDFs to process, which to skip, and which empty outputs to remove.

    Returns:
        - to_process: list of PDF files to convert
        - skipped: list of PDF files skipped due to existing non-empty outputs (when not overwriting)
        - removed_empty: list of empty output files that were deleted before processing begins
    """
    to_process: list[Path] = []
    skipped: list[Path] = []
    removed_empty: list[Path] = []

    for pdf_path in pdf_files:
        output_path = _output_path_for(pdf_path)
        try:
            if output_path.exists():
                try:
                    size = output_path.stat().st_size
                except OSError:
                    size = -1

                if size == 0:
                    try:
                        output_path.unlink(missing_ok=True)
                    except OSError:
                        pass
                    removed_empty.append(output_path)
                    to_process.append(pdf_path)
                    continue

                if not overwrite:
                    skipped.append(pdf_path)
                    continue

            to_process.append(pdf_path)
        except OSError:
            # If any unexpected FS issue occurs, err on the side of processing
            to_process.append(pdf_path)

    return to_process, skipped, removed_empty


class Cli(CommonCliSettings):
    """Bulk PDF → Markdown via LiteLLM document understanding.

    Naively sends the raw PDF (base64) to the model and writes the response
    to a `.md` file with the same basename next to the source PDF.
    """

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_file=".env", cli_parse_args=True
    )

    root_path: CliPositionalArg[Path]
    overwrite: bool = False

    @override
    def model_post_init(self, _context: Any) -> None:  # pyright: ignore[reportAny,reportExplicitAny]
        # suppress litellm logging
        litellm.suppress_debug_info = True
        logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    @computed_field
    @property
    def model(self) -> str:
        return settings.model

    @override
    async def cli_cmd(self) -> None:
        if not litellm.utils.supports_pdf_input(self.model, None):
            self.logger.error(
                "Model '%s' does not support PDF input. Aborting.", self.model
            )
            return

        pdf_files: list[Path] = discover_pdf_files(self.root_path)
        if not pdf_files:
            self.logger.warning("No PDF files found under %s", self.root_path)
            return
        else:
            self.logger.info(
                "Discovered %d PDF file(s) under %s", len(pdf_files), self.root_path
            )

        to_process, skipped, removed_empty = plan_processing(pdf_files, self.overwrite)

        if removed_empty:
            self.logger.info(
                "Removed %d empty output file(s):\n%s",
                len(removed_empty),
                ", ".join(
                    format_display_path(p, self.root_path) for p in removed_empty
                ),
            )

        if skipped:
            self.logger.info(
                "Skipping %d file(s) with existing non-empty output (use --overwrite to regenerate):\n%s",
                len(skipped),
                ", ".join(format_display_path(p, self.root_path) for p in skipped),
            )

        if not to_process:
            self.logger.info("Nothing to process after skip checks.")
            return
        else:
            self.logger.info(
                "Processing %d PDF file(s):\n%s",
                len(to_process),
                ", ".join(format_display_path(p, self.root_path) for p in to_process),
            )

        progress = create_progress()
        successes = 0
        failures = 0

        with progress:
            task_id: TaskID = progress.add_task(
                "Converting PDFs", total=len(to_process)
            )

            semaphore = anyio.Semaphore(settings.max_concurrency)
            async with anyio.create_task_group() as tg:
                for pdf_path in to_process:
                    tg.start_soon(
                        _convert_one,
                        pdf_path,
                        self.model,
                        progress,
                        task_id,
                        semaphore,
                        self.logger.name,
                    )

            # The per-task function updates the progress bar. For a quick summary,
            # re-scan outcomes by checking .md files existence.
            for pdf_path in to_process:
                output_path = _output_path_for(pdf_path)
                if output_path.exists() and output_path.stat().st_size > 0:
                    successes += 1
                else:
                    failures += 1

        if failures == 0:
            self.logger.info(
                "Completed %d/%d conversions successfully.", successes, len(to_process)
            )
        else:
            self.logger.warning(
                "Completed %d/%d conversions successfully, %d failed, %d skipped.",
                successes,
                len(to_process),
                failures,
                len(skipped),
            )


def is_pdf_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == ".pdf"


def discover_pdf_files(root: Path) -> list[Path]:
    """Return sorted list of PDF files under `root`.

    - If `root` is a single PDF file, return it.
    - If `root` is a directory, recursively search for PDFs by suffix.
    - If `root` does not exist or is not a PDF/file, return empty list.
    """
    base = root.resolve().absolute()
    if base.is_file():
        return [base] if is_pdf_file(base) else []
    if not base.exists():
        return []

    return natsorted(
        (p for p in base.rglob("*") if is_pdf_file(p)), key=lambda p: str(p).lower()
    )


def _output_path_for(pdf_path: Path) -> Path:
    return pdf_path.with_suffix(f".{settings.output_extension}")


async def _convert_one(
    pdf_path: Path,
    model: str,
    progress: Progress,
    task_id: TaskID,
    semaphore: anyio.Semaphore,
    logger_name: str,
) -> None:
    logger = get_logger(logger_name)

    try:
        output_path = _output_path_for(pdf_path)

        async with semaphore:
            async with await open_file(pdf_path, "rb") as f:
                pdf_bytes = await f.read()
            base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
            messages: list[UserMessage] = compose_pdf_user_messages(base64_pdf)

            # Enforce an overall per-request timeout on top of provider timeouts
            try:
                with anyio.fail_after(settings.request_timeout_s):
                    response = await litellm.acompletion(model=model, messages=messages)  # pyright: ignore[reportUnknownMemberType]
            except (TimeoutError, CancelledError):
                logger.error("Timed out after %s", settings.request_timeout_s)
                return
            except InternalServerError as e:
                logger.error("LLM API vendor boom: %s", e)
                return

            text = cast(
                list[litellm.Choices],
                cast(litellm.ModelResponse, response).choices,  # pyright: ignore[reportPrivateImportUsage]
            )[0].message.content

            # Write result
            async with await open_file(output_path, "w", encoding="utf-8") as f:
                _ = await f.write(text or "")

            logger.info("Converted: %s -> %s", pdf_path.name, output_path.name)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to convert %s: %s", pdf_path, exc)
    finally:
        progress.update(task_id, advance=1)


if __name__ == "__main__":
    _ = Cli.run_anyio()
