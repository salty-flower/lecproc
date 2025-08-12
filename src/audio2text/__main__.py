import json
import dataclasses
from pathlib import Path
from typing import ClassVar
from collections.abc import Iterator, AsyncGenerator, Iterable

from pydantic_settings import CliApp, BaseSettings, CliPositionalArg, SettingsConfigDict
import trio
from pydantic import computed_field
from faster_whisper import WhisperModel
from aiofiles import open as aopen
from rich.progress import Progress
from faster_whisper.transcribe import Segment, TranscriptionInfo

from .settings import settings
from path_settings import path_settings
from logs import create_progress, configure_rich_logging, get_logger, TaskID


configure_rich_logging()
logger = get_logger("audio2text")


class Cli(BaseSettings):
    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_file=".env", cli_parse_args=True
    )
    media_path: CliPositionalArg[Path]
    save_path: CliPositionalArg[Path | None] = None

    def __await__(self):
        return self.cli_cmd().__await__()

    @computed_field
    @property
    def model(self) -> WhisperModel:
        logger.info(
            "Loading model '%s' (compute_type=%s) into %s",
            settings.model_path,
            settings.compute_type,
            path_settings.models_download_dir.absolute(),
        )
        return WhisperModel(
            model_size_or_path=settings.model_path,
            download_root=str(path_settings.models_download_dir.absolute()),
            compute_type=settings.compute_type,
        )

    async def cli_cmd(self) -> None:
        segments, info = self.model.transcribe(  # pyright: ignore[reportUnknownMemberType]
            str(self.media_path.absolute()), language="en"
        )

        # Log basic transcription info
        info_typed: TranscriptionInfo = info  # type: ignore[assignment]
        language: str | None = getattr(info_typed, "language", None)
        language_probability: float | None = getattr(
            info_typed, "language_probability", None
        )
        duration: float | None = getattr(info_typed, "duration", None)
        if language is not None:
            if language_probability is not None:
                logger.info(
                    "Detected language: %s (p=%.2f)",
                    language,
                    float(language_probability),
                )
            else:
                logger.info("Detected language: %s", language)
        if duration is not None:
            logger.info("Audio duration: %.2fs", float(duration))

        # Prepare output file if provided
        output_path = str(self.save_path) if self.save_path is not None else None
        if output_path:
            logger.info("Writing segments to %s", output_path)

        progress = create_progress()

        # If we know total duration, track progress against it. Otherwise, use an indeterminate bar.
        total_seconds: float | None = float(duration) if duration is not None else None
        with progress:
            task_id: TaskID = progress.add_task(
                "Transcribing",
                total=total_seconds if total_seconds is not None else None,
            )

            if output_path:
                async with aopen(output_path, "w") as f:
                    async for segment in _iterate_segments(
                        segments, progress, task_id, total_seconds
                    ):
                        logger.debug(
                            "[%s - %s] %s",
                            segment.start,
                            segment.end,
                            segment.text,
                        )
                        _ = await f.write(
                            json.dumps(dataclasses.asdict(segment)) + "\n"
                        )
            else:
                async for segment in _iterate_segments(
                    segments, progress, task_id, total_seconds
                ):
                    logger.debug(
                        "[%s - %s] %s",
                        segment.start,
                        segment.end,
                        segment.text,
                    )
        logger.info("Transcription complete")


async def _iterate_segments(
    segments: Iterable[Segment],
    progress: Progress,
    task_id: TaskID,
    total_seconds: float | None,
) -> AsyncGenerator[Segment, None]:
    """Async-iterate over segments while updating progress.

    The faster-whisper `segments` is a generator; wrap it for async consumption under Trio.
    """
    last_completed = 0.0

    segment_iterator: Iterator[Segment] = iter(segments)

    def _next_segment_blocking() -> Segment | None:
        try:
            return next(segment_iterator)
        except StopIteration:
            return None

    while True:
        # Run the blocking `next()` in a worker thread to avoid blocking Trio
        segment = await trio.to_thread.run_sync(_next_segment_blocking)
        if segment is None:
            break

        # Update progress based on end timestamp if total is known; otherwise just advance by 1
        if total_seconds is not None:
            current_completed = float(segment.end)
            if current_completed >= last_completed:
                progress.update(task_id, completed=current_completed)
                last_completed = current_completed
        else:
            progress.update(task_id, advance=1)

        yield segment


if __name__ == "__main__":
    trio.run(CliApp.run, Cli)
