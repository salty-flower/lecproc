import base64
import contextlib
import logging
from asyncio import CancelledError
from pathlib import Path
from typing import Any, cast, override

import anyio
import litellm
from anyio import open_file
from litellm.exceptions import InternalServerError
from pydantic import computed_field
from pydantic_settings import CliPositionalArg
from rich.progress import Progress

from common_cli_settings import CommonCliSettings
from logs import TaskID, create_progress, get_logger

from .models import SystemMessage, UserMessage, compose_pdf_user_messages
from .settings import settings
from .utils import discover_pdf_files, format_display_path, output_path_for


def plan_processing(
    pdf_files: list[Path],
    overwrite: bool,
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
        output_path = output_path_for(pdf_path)
        try:
            if output_path.exists():
                try:
                    size = output_path.stat().st_size
                except OSError:
                    size = -1

                if size == 0:
                    with contextlib.suppress(OSError):
                        output_path.unlink(missing_ok=True)
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

    root_path: CliPositionalArg[Path]
    overwrite: bool = False
    general_context_file: str | None = "context.json"

    concurrency: int = settings.max_concurrency

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
        self.logger.info(
            "Discovered %d PDF file(s) under %s",
            len(pdf_files),
            self.root_path,
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
        self.logger.info(
            "Processing %d PDF file(s):\n%s",
            len(to_process),
            ", ".join(format_display_path(p, self.root_path) for p in to_process),
        )

        progress = create_progress()
        successes = 0
        failures = 0

        if (
            self.general_context_file
            and (self.root_path / self.general_context_file).exists()
        ):
            self.logger.info(
                "Loading general context from %s",
                self.general_context_file,
            )
            async with await open_file(
                self.root_path / self.general_context_file,
                "r",
                encoding="utf-8",
            ) as f:
                general_context = await f.read()
            self.logger.info(
                "Loaded general context with length %d. Headings: %s. Tail: %s",
                len(general_context),
                general_context[:100],
                general_context[-100:],
            )
        else:
            self.logger.info(
                "No general context file provided%s",
                f": {self.root_path / self.general_context_file} does not exist"
                if self.general_context_file
                else "",
            )
            general_context = None

        with progress:
            task_id: TaskID = progress.add_task(
                "Converting PDFs", total=len(to_process)
            )

            semaphore = anyio.Semaphore(self.concurrency)
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
                        general_context,
                    )

            # The per-task function updates the progress bar. For a quick summary,
            # re-scan outcomes by checking .md files existence.
            for pdf_path in to_process:
                output_path = output_path_for(pdf_path)
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


async def _convert_one(
    pdf_path: Path,
    model: str,
    progress: Progress,
    task_id: TaskID,
    semaphore: anyio.Semaphore,
    logger_name: str,
    general_context: str | None = None,
) -> None:
    logger = get_logger(logger_name)

    try:
        output_path = output_path_for(pdf_path)

        async with semaphore:
            async with await open_file(pdf_path, "rb") as f:
                pdf_bytes = await f.read()
            base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
            messages: list[UserMessage | SystemMessage] = compose_pdf_user_messages(
                pdf_path.name, base64_pdf, general_context
            )

            # Enforce an overall per-request timeout on top of provider timeouts
            try:
                with anyio.fail_after(settings.request_timeout_s):
                    response = await litellm.acompletion(model=model, messages=messages)  # pyright: ignore[reportUnknownMemberType]
            except (TimeoutError, CancelledError):
                logger.exception("Timed out after %s", settings.request_timeout_s)
                return
            except InternalServerError:
                logger.exception("LLM API vendor boom")
                return

            text = cast(
                "list[litellm.Choices]",
                cast("litellm.ModelResponse", response).choices,  # pyright: ignore[reportPrivateImportUsage]
            )[0].message.content

            # Write result
            if text:
                async with await open_file(output_path, "w", encoding="utf-8") as f:
                    _ = await f.write(text)

                logger.info("Converted: %s -> %s", pdf_path.name, output_path.name)
            else:
                logger.warning("No text returned for %s", pdf_path.name)
    except Exception:
        logger.exception("Failed to convert %s", pdf_path)
    finally:
        progress.update(task_id, advance=1)


if __name__ == "__main__":
    _ = Cli.run_anyio()
