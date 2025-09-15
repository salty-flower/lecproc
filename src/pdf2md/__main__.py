import base64
import contextlib
import logging
from asyncio import CancelledError
from pathlib import Path
from typing import Any, ClassVar, cast, override

import anyio
import litellm
from anyio import open_file
from litellm.exceptions import InternalServerError
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.vertex_ai.common_utils import VertexAIError
from pydantic import computed_field
from pydantic_settings import CliPositionalArg
from rich.progress import Progress

from common_cli_settings import CommonCliSettings
from logs import TaskID, create_progress, get_logger

from .models import SystemMessage, UserMessage, compose_pdf_user_messages
from .settings import settings
from .typst_fixer import fix_typst_errors_iteratively
from .typst_parser import extract_typst_blocks
from .typst_validator import has_any_typst_errors
from .utils import discover_pdf_files, format_display_path, output_path_for


async def load_context_file(context_path: Path) -> str:
    async with await open_file(context_path, "r", encoding="utf-8") as f:
        return await f.read()


def discover_context_file(root_path: Path, extensions: list[str], base_name: str = "context") -> Path | None:
    for ext in extensions:
        context_path = root_path / f"{base_name}.{ext}"
        if context_path.exists():
            return context_path
    return None


class RetryProgressCallback(CustomLogger):
    def __init__(self, progress: Progress, task_id: TaskID, pdf_name: str) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.progress: Progress = progress
        self.task_id: TaskID = task_id
        self.pdf_name: str = pdf_name
        self.retry_count: int = 0
        self.max_retries: int = settings.max_retry_attempts

    @override
    def log_pre_api_call(self, model: Any, messages: Any, kwargs: Any) -> None:  # pyright: ignore[reportAny]
        if self.retry_count > 0:
            self.progress.update(
                self.task_id,
                description=f"Converting PDFs (retry {self.retry_count}/{self.max_retries - 1} for {self.pdf_name})",
            )

    @override
    def log_failure_event(self, kwargs: Any, response_obj: Any, start_time: Any, end_time: Any) -> None:  # pyright: ignore[reportAny]
        self.retry_count += 1


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
    """Bulk PDF → Markdown with Typst formulas via LiteLLM document understanding.

    Two-phase process:
    1. Initial drafting: Sends the raw PDF (base64) to the model for Markdown conversion
    2. Typst validation: Parses output for Typst formulas/code blocks, validates syntax,
       and uses LLM to fix any compilation errors iteratively.

    Writes the final result to a `.md` file with the same basename next to the source PDF.
    """

    is_root: ClassVar[bool | None] = True

    root_path: CliPositionalArg[Path]
    overwrite: bool = False
    context_file_base: str = "context"
    context_extensions: ClassVar[list[str]] = ["json", "md", "txt"]

    concurrency: int = settings.max_concurrency

    @override
    def model_post_init(self, _context: Any) -> None:  # pyright: ignore[reportAny]
        # suppress litellm logging
        litellm.suppress_debug_info = True
        logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    @computed_field
    @property
    def model(self) -> str:
        return settings.model

    @override
    async def cli_cmd_async(self) -> None:
        # LiteLLM doesn't know about OpenRouter models capabilities yet, so we waive this check for OpenRouter models for now
        self.logger.info("Using %s", self.model)
        match self.model.split("/", 1):
            case ["openrouter", _]:
                # OpenRouter models are assumed to support PDF input
                pass
            case _:
                if not litellm.utils.supports_pdf_input(self.model, None):
                    self.logger.error("Model '%s' does not support PDF input. Aborting.", self.model)
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
                ", ".join(format_display_path(p, self.root_path) for p in removed_empty),
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

        context_file_path = discover_context_file(self.root_path, self.context_extensions, self.context_file_base)
        if context_file_path:
            self.logger.info(
                "Loading context from %s",
                format_display_path(context_file_path, self.root_path),
            )
            general_context = await load_context_file(context_file_path)
            self.logger.info(
                "Loaded context with length %d. Preview: %s... Tail: ...%s",
                len(general_context),
                general_context[:100].replace("\n", "\\n"),
                general_context[-100:].replace("\n", "\\n"),
            )
        else:
            extensions_str = ",".join(self.context_extensions)
            self.logger.info(
                "No context file found (searched for %s.{%s})",
                self.context_file_base,
                extensions_str,
            )
            general_context = None

        with progress:
            task_id: TaskID = progress.add_task("Converting PDFs", total=len(to_process))

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
            self.logger.info("Completed %d/%d conversions successfully.", successes, len(to_process))
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
        intermediate_path = output_path.with_suffix(".phase1.md")

        # Check if Phase 1 was already completed (intermediate file exists)
        text: str | None = None
        if intermediate_path.exists():
            try:
                async with await open_file(intermediate_path, "r", encoding="utf-8") as f:
                    text = await f.read()
                if text:
                    logger.info("Found existing Phase 1 output, skipping to Phase 2: %s", intermediate_path.name)
                else:
                    # Empty file, remove it and proceed with Phase 1
                    with contextlib.suppress(OSError):
                        intermediate_path.unlink(missing_ok=True)
                    text = None
            except (OSError, UnicodeDecodeError):
                # If we can't read the file, remove it and proceed with Phase 1
                with contextlib.suppress(OSError):
                    intermediate_path.unlink(missing_ok=True)
                text = None

        # Phase 1: Only run if we don't have existing intermediate content
        if text is None:
            async with semaphore:
                async with await open_file(pdf_path, "rb") as f:
                    pdf_bytes = await f.read()
                base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
                messages: list[UserMessage | SystemMessage] = await compose_pdf_user_messages(
                    pdf_path.name, base64_pdf, general_context
                )

                # Create retry progress callback
                retry_callback = RetryProgressCallback(progress, task_id, pdf_path.name)

                # Enforce an overall per-request timeout on top of provider timeouts and use built-in retry
                try:
                    with anyio.fail_after(settings.request_timeout_s):
                        response = await litellm.acompletion(  # pyright: ignore[reportUnknownMemberType]
                            model=model,
                            messages=messages,
                            num_retries=settings.max_retry_attempts - 1,
                            callbacks=[retry_callback],
                        )
                except (TimeoutError, CancelledError):
                    logger.error("Timed out after %s", settings.request_timeout_s)  # noqa: TRY400
                    return
                except (InternalServerError, VertexAIError):
                    logger.error("LLM API vendor boom")  # noqa: TRY400
                    return

                text = cast(
                    "list[litellm.Choices]",
                    cast("litellm.ModelResponse", response).choices,  # pyright: ignore[reportPrivateImportUsage]
                )[0].message.content

                # Save intermediate markdown after Phase 1 (before Typst validation)
                if text:
                    async with await open_file(intermediate_path, "w", encoding="utf-8") as f:
                        _ = await f.write(text)
                    logger.info("Phase 1 complete: Saved intermediate markdown to %s", intermediate_path.name)

        # Phase 2: Typst validation and fixing
        if text:
            logger.info("Phase 2: Validating and fixing Typst content for %s", pdf_path.name)

            # Extract Typst blocks from the generated markdown
            typst_blocks = extract_typst_blocks(text)

            if typst_blocks:
                logger.info("Found %d Typst block(s) in %s", len(typst_blocks), pdf_path.name)

                # Check for Typst compilation errors
                has_errors, _error_message = await has_any_typst_errors(typst_blocks, logger_name)

                if has_errors:
                    logger.warning("Found Typst errors in %s, attempting to fix", pdf_path.name)

                    # Attempt to fix errors iteratively
                    fixed_text, all_fixed = await fix_typst_errors_iteratively(
                        text, logger_name, max_iterations=2, max_fix_attempts=3
                    )

                    if all_fixed:
                        logger.info("Successfully fixed all Typst errors in %s", pdf_path.name)
                        text = fixed_text
                    else:
                        logger.warning(
                            "Could not fix all Typst errors in %s, proceeding with partial fixes", pdf_path.name
                        )
                        text = fixed_text
                else:
                    logger.info("All Typst blocks validated successfully for %s", pdf_path.name)
            else:
                logger.info("No Typst blocks found in %s", pdf_path.name)

            # Write result
            async with await open_file(output_path, "w", encoding="utf-8") as f:
                _ = await f.write(text)
            intermediate_path.unlink()

            logger.info("Converted: %s -> %s. Removed intermediate file", pdf_path.name, output_path.name)
        else:
            logger.warning("No text returned for %s", pdf_path.name)
    except Exception:
        logger.exception("Failed to convert %s", pdf_path)
    finally:
        progress.update(task_id, advance=1)


if __name__ == "__main__":
    _ = Cli.run_anyio_static()
