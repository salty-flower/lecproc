import base64
import contextlib
import logging
import math
from asyncio import CancelledError
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar, cast, override

import anyio
import litellm
from litellm.exceptions import InternalServerError, RateLimitError
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.vertex_ai.common_utils import VertexAIError
from pydantic import computed_field
from pydantic_settings import CliPositionalArg
from rich.progress import Progress

from common_cli_settings import CommonCliSettings
from logs import TaskID, create_progress, get_logger

from .models import compose_user_messages
from .rate_limit import execute_with_rate_limit_retry
from .settings import settings
from .typst_fixer import fix_typst_errors_iteratively
from .typst_parser import extract_typst_blocks
from .utils import (
    detect_output_collisions,
    discover_source_files,
    format_display_path,
    get_input_file_metadata,
    output_path_for,
    supported_extensions_display,
)


async def load_context_file(context_path: Path) -> str:
    return await anyio.Path(context_path).read_text(encoding="utf-8")


def discover_context_file(root_path: Path, extensions: list[str], base_name: str = "context") -> Path | None:
    for ext in extensions:
        context_path = root_path / f"{base_name}.{ext}"
        if context_path.exists():
            return context_path
    return None


_COMPLETION_THRESHOLD = 0.999
_WAIT_DESCRIPTION_MAX = 80


class ProgressTracker:
    """Track per-file conversion progress and surface it through a shared task."""

    def __init__(self, progress: Progress, task_id: TaskID, total_files: int) -> None:
        self._progress: Progress = progress
        self._task_id: TaskID = task_id
        self._total_files: int = total_files
        self._file_progress: dict[Path, float] = {}
        self._status: str | None = None

    def update(self, source_path: Path, stage: float, status: str | None = None) -> None:
        """Update the tracked stage for ``source_path`` and refresh the task display."""

        clamped = max(0.0, min(stage, 1.0))
        previous = self._file_progress.get(source_path, 0.0)
        clamped = max(clamped, previous)

        self._file_progress[source_path] = clamped

        if status is not None:
            self._status = status

        completed_total = sum(self._file_progress.values())
        finished = sum(1 for value in self._file_progress.values() if value >= _COMPLETION_THRESHOLD)
        description_base = self._status or "Converting documents"
        description = f"{description_base} ({finished}/{self._total_files})"

        self._progress.update(self._task_id, completed=completed_total, description=description)

    async def wait_for_rate_limit(self, seconds: float, description: str | None = None) -> None:
        """Display a temporary progress bar while waiting for rate limits to reset."""

        total = max(0.0, seconds)
        if total <= 0:
            return

        base_description = (description or "Waiting for rate limit reset").strip()
        if not base_description:
            base_description = "Waiting for rate limit reset"
        if len(base_description) > _WAIT_DESCRIPTION_MAX:
            base_description = base_description[: _WAIT_DESCRIPTION_MAX - 3] + "..."

        task_id = self._progress.add_task(base_description, total=total)
        start = anyio.current_time()

        try:
            while True:
                elapsed = anyio.current_time() - start
                remaining = total - elapsed
                if remaining <= 0:
                    self._progress.update(task_id, completed=total, description=f"{base_description} (resuming)")
                    break

                remaining_seconds = max(1, math.ceil(remaining))
                self._progress.update(
                    task_id,
                    completed=min(total, elapsed),
                    description=f"{base_description} ({remaining_seconds}s remaining)",
                )

                await anyio.sleep(min(1.0, remaining))
        finally:
            with contextlib.suppress(KeyError):
                self._progress.remove_task(task_id)


class RetryProgressCallback(CustomLogger):
    def __init__(self, on_retry: Callable[[int, int], None]) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.retry_count: int = 0
        self.max_retries: int = settings.max_retry_attempts
        self._on_retry: Callable[[int, int], None] = on_retry

    @override
    def log_pre_api_call(self, model: Any, messages: Any, kwargs: Any) -> None:  # pyright: ignore[reportAny]
        if self.retry_count > 0:
            self._on_retry(self.retry_count, self.max_retries - 1)

    @override
    def log_failure_event(self, kwargs: Any, response_obj: Any, start_time: Any, end_time: Any) -> None:  # pyright: ignore[reportAny]
        self.retry_count += 1


def plan_processing(
    source_files: list[Path],
    overwrite: bool,
) -> tuple[list[Path], list[Path], list[Path]]:
    """Plan which documents to process, which to skip, and which empty outputs to remove.

    Returns:
        - to_process: list of source files to convert
        - skipped: list of source files skipped due to existing non-empty outputs (when not overwriting)
        - removed_empty: list of empty output files that were deleted before processing begins

    """
    to_process: list[Path] = []
    skipped: list[Path] = []
    removed_empty: list[Path] = []

    for source_path in source_files:
        output_path = output_path_for(source_path)
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
                    to_process.append(source_path)
                    continue

                if not overwrite:
                    skipped.append(source_path)
                    continue

            to_process.append(source_path)
        except OSError:
            # If any unexpected FS issue occurs, err on the side of processing
            to_process.append(source_path)

    return to_process, skipped, removed_empty


class Cli(CommonCliSettings):
    """Bulk document → Markdown with Typst formulas via LiteLLM document understanding.

    Two-phase process:
    1. Initial drafting: Sends the raw document (base64) to the model for Markdown conversion
    2. Typst validation: Parses output for Typst formulas/code blocks, validates syntax,
       and uses LLM to fix any compilation errors iteratively.

    Writes the final result to a `.md` file with the same basename next to the source document.
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
        return settings.drafting_model

    @override
    async def cli_cmd_async(self) -> None:
        # LiteLLM doesn't know about OpenRouter models capabilities yet, so we waive this check for OpenRouter models for now
        self.logger.info("Using %s for drafting and %s for fixing", settings.drafting_model, settings.fixing_model)
        source_files: list[Path] = discover_source_files(self.root_path)
        if not source_files:
            self.logger.warning(
                "No supported files (%s) found under %s",
                supported_extensions_display(),
                self.root_path,
            )
            return

        has_pdf_inputs = False
        has_image_inputs = False
        for path in source_files:
            metadata = get_input_file_metadata(path)
            if metadata is None:
                # Should not happen because discover_source_files filters these out
                continue
            if metadata.category == "pdf":
                has_pdf_inputs = True
            elif metadata.category == "image":
                has_image_inputs = True

        match self.model.split("/", 1):
            case ["openrouter", _]:
                # OpenRouter models are assumed to support these inputs for now
                pass
            case _:
                if has_pdf_inputs and not litellm.utils.supports_pdf_input(self.model, None):
                    self.logger.error("Model '%s' does not support PDF input. Aborting.", self.model)
                    return
                if has_image_inputs and not litellm.utils.supports_vision(self.model, None):
                    self.logger.error("Model '%s' does not support image input. Aborting.", self.model)
                    return

        self.logger.info(
            "Discovered %d supported file(s) under %s",
            len(source_files),
            self.root_path,
        )

        collisions = detect_output_collisions(source_files)
        if collisions:
            collision_details = ", ".join(
                f"{format_display_path(output_path, self.root_path)} ← "
                + ", ".join(
                    format_display_path(src, self.root_path) for src in sorted(sources, key=lambda p: str(p).lower())
                )
                for output_path, sources in sorted(collisions.items(), key=lambda item: str(item[0]).lower())
            )
            self.logger.error(
                "Multiple source files would write to the same output. Rename conflicting files and retry: %s",
                collision_details,
            )
            return

        to_process, skipped, removed_empty = plan_processing(source_files, self.overwrite)

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
            "Processing %d file(s):\n%s",
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
            task_id: TaskID = progress.add_task("Converting documents", total=float(len(to_process)))
            tracker = ProgressTracker(progress, task_id, len(to_process))

            semaphore = anyio.Semaphore(self.concurrency)
            async with anyio.create_task_group() as tg:
                for source_path in to_process:
                    tg.start_soon(
                        _convert_one,
                        source_path,
                        self.model,
                        tracker,
                        semaphore,
                        self.logger.name,
                        general_context,
                    )

            # The per-task function updates the progress bar. For a quick summary,
            # re-scan outcomes by checking .md files existence.
            for source_path in to_process:
                output_path = output_path_for(source_path)
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


_PHASE_STAGE_START = 0.05
_PHASE_STAGE_DRAFTING = 0.15
_PHASE_STAGE_DRAFT_COMPLETE = 0.35
_PHASE_STAGE_VALIDATING = 0.45
_PHASE_STAGE_FIX_BASE = 0.5
_PHASE_STAGE_FIX_DONE = 0.9
_PHASE_STAGE_COMPLETE = 1.0


async def _convert_one(
    source_path: Path,
    model: str,
    tracker: ProgressTracker,
    semaphore: anyio.Semaphore,
    logger_name: str,
    general_context: str | None = None,
) -> None:
    logger = get_logger(logger_name)
    progress_files: set[Path] = set()

    def set_stage(stage: float, status: str | None = None) -> None:
        tracker.update(source_path, stage, status)

    set_stage(_PHASE_STAGE_START, f"Starting {source_path.name}")

    try:
        metadata = get_input_file_metadata(source_path)
        if metadata is None:
            logger.error("Skipping unsupported file type: %s", source_path.name)
            set_stage(_PHASE_STAGE_COMPLETE, f"Unsupported file: {source_path.name}")
            return

        output_path = output_path_for(source_path)
        intermediate_path = output_path.with_suffix(".phase1.md")

        # Check if Phase 1 was already completed (intermediate file exists)
        text: str | None = None
        if intermediate_path.exists():
            try:
                text = await anyio.Path(intermediate_path).read_text(encoding="utf-8")
                if text:
                    logger.info("Found existing Phase 1 output, skipping to Phase 2: %s", intermediate_path.name)
                    set_stage(_PHASE_STAGE_DRAFT_COMPLETE, f"Reusing draft for {source_path.name}")
                else:
                    with contextlib.suppress(OSError):
                        intermediate_path.unlink(missing_ok=True)
                    text = None
            except (OSError, UnicodeDecodeError):
                with contextlib.suppress(OSError):
                    intermediate_path.unlink(missing_ok=True)
                text = None

        # Phase 1: Only run if we don't have existing intermediate content
        if text is None:
            async with semaphore:
                raw_bytes = await anyio.Path(source_path).read_bytes()
                base64_data = base64.b64encode(raw_bytes).decode("utf-8")
                messages = await compose_user_messages(
                    source_path.name,
                    base64_data,
                    metadata,
                    general_context,
                )

                def handle_retry(retry_index: int, max_retries: int) -> None:
                    set_stage(
                        _PHASE_STAGE_DRAFTING,
                        f"Retry {retry_index}/{max_retries} for {source_path.name}",
                    )

                retry_callback = RetryProgressCallback(handle_retry)

                set_stage(_PHASE_STAGE_DRAFTING, f"Drafting {source_path.name}")

                async def _wait_for_reset(seconds: float, description: str | None) -> None:
                    wait_label = description or "Rate limited"
                    display = f"{source_path.name}: {wait_label}"
                    set_stage(_PHASE_STAGE_DRAFTING, display)
                    await tracker.wait_for_rate_limit(seconds, display)

                async def _request_completion() -> litellm.ModelResponse:
                    with anyio.fail_after(settings.request_timeout_s):
                        return await litellm.acompletion(  # pyright: ignore[reportUnknownMemberType]
                            model=model,
                            messages=messages,
                            num_retries=0,
                            callbacks=[retry_callback],
                        )

                try:
                    response = await execute_with_rate_limit_retry(
                        _request_completion,
                        logger=logger,
                        max_attempts=settings.max_retry_attempts,
                        wait_callback=_wait_for_reset,
                        context=f"{source_path.name} rate limit",
                    )
                except RateLimitError as error:
                    logger.error(  # noqa: TRY400
                        "Rate limit persisted for %s after %d attempts: %s",
                        source_path.name,
                        settings.max_retry_attempts,
                        error,
                    )
                    set_stage(_PHASE_STAGE_COMPLETE, f"Rate limit for {source_path.name}")
                    return
                except (TimeoutError, CancelledError):
                    logger.error("Timed out after %s", settings.request_timeout_s)  # noqa: TRY400
                    set_stage(_PHASE_STAGE_COMPLETE, f"Draft timed out for {source_path.name}")
                    return
                except (InternalServerError, VertexAIError):
                    logger.error("LLM API vendor boom")  # noqa: TRY400
                    set_stage(_PHASE_STAGE_COMPLETE, f"Draft failed for {source_path.name}")
                    return

                text = cast(
                    "list[litellm.Choices]",
                    cast("litellm.ModelResponse", response).choices,  # pyright: ignore[reportPrivateImportUsage]
                )[0].message.content

                if text:
                    _ = await anyio.Path(intermediate_path).write_text(text, encoding="utf-8")
                    logger.info("Phase 1 complete: Saved intermediate markdown to %s", intermediate_path.name)
                    set_stage(_PHASE_STAGE_DRAFT_COMPLETE, f"Draft complete for {source_path.name}")
                else:
                    logger.error("Received empty draft content for %s", source_path.name)
                    set_stage(_PHASE_STAGE_COMPLETE, f"Empty draft for {source_path.name}")
                    return

        all_fixed = False

        if text and settings.enable_fixing_phase:
            logger.info("Phase 2: Validating and fixing Typst content for %s", source_path.name)
            set_stage(_PHASE_STAGE_VALIDATING, f"Validating Typst for {source_path.name}")

            typst_blocks = extract_typst_blocks(text)

            if typst_blocks:
                logger.info("Found %d Typst block(s) in %s", len(typst_blocks), source_path.name)

                def iteration_progress(fraction: float, message: str | None = None) -> None:
                    span = _PHASE_STAGE_FIX_DONE - _PHASE_STAGE_FIX_BASE
                    stage = _PHASE_STAGE_FIX_BASE + max(0.0, min(fraction, 1.0)) * span
                    set_stage(
                        stage, f"{source_path.name}: {message}" if message else f"Fixing Typst for {source_path.name}"
                    )

                fixed_text, all_fixed, iteration_progress_files = await fix_typst_errors_iteratively(
                    text,
                    logger_name,
                    max_iterations=4,
                    max_fix_attempts=3,
                    progress_callback=iteration_progress,
                )
                progress_files.update(iteration_progress_files)

                text = fixed_text
                if all_fixed:
                    logger.info("Successfully fixed all Typst errors in %s", source_path.name)
                    set_stage(_PHASE_STAGE_FIX_DONE, f"Typst fixes complete for {source_path.name}")
                else:
                    logger.warning(
                        "Could not fix all Typst errors in %s, proceeding with partial fixes",
                        source_path.name,
                    )
                    set_stage(_PHASE_STAGE_FIX_DONE, f"Partial Typst fixes for {source_path.name}")
            else:
                logger.info("No Typst blocks found in %s", source_path.name)
                set_stage(_PHASE_STAGE_FIX_DONE, f"No Typst blocks in {source_path.name}")

            _ = await anyio.Path(output_path).write_text(text, encoding="utf-8")

            if all_fixed:
                for progress_file in progress_files:
                    with contextlib.suppress(OSError):
                        progress_file.unlink(missing_ok=True)

            logger.info("Converted: %s -> %s. Removed intermediate file", source_path.name, output_path.name)
            completion_status = (
                f"Completed {source_path.name}"
                if all_fixed
                else f"Completed with remaining Typst errors: {source_path.name}"
            )
            set_stage(_PHASE_STAGE_COMPLETE, completion_status)
        else:
            logger.info("Skipping fixing phase due to missing text or disabling fixing phase")
            if text:
                set_stage(_PHASE_STAGE_COMPLETE, f"Draft ready for {source_path.name}")
            else:
                set_stage(_PHASE_STAGE_COMPLETE, f"No output for {source_path.name}")
    except Exception:
        set_stage(_PHASE_STAGE_COMPLETE, f"Failed {source_path.name}")
        logger.exception("Failed to convert %s", source_path)
    finally:
        tracker.update(source_path, _PHASE_STAGE_COMPLETE)


if __name__ == "__main__":
    _ = Cli.run_anyio_static()
