from collections.abc import AsyncGenerator, Iterable, Iterator
from pathlib import Path
from typing import override

import orjson
from anyio import to_thread
from anyio.streams.file import FileWriteStream
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment
from pydantic import computed_field
from pydantic_settings import CliPositionalArg
from rich.progress import Progress

from common_cli_settings import CommonCliSettings
from logs import TaskID, create_progress
from path_settings import path_settings

from .settings import settings


class Cli(CommonCliSettings):
    media_path: CliPositionalArg[Path]
    save_path: CliPositionalArg[Path]
    language: str | None = None

    @computed_field
    @property
    def model(self) -> WhisperModel:
        self.logger.info(
            "Loading model '%s' (compute_type=%s) from %s",
            settings.model_path,
            settings.compute_type,
            path_settings.models_download_dir.absolute(),
        )
        return WhisperModel(
            model_size_or_path=settings.model_path,
            download_root=str(path_settings.models_download_dir.absolute()),
            compute_type=settings.compute_type,
        )

    @override
    async def cli_cmd_async(self) -> None:
        self.logger.info("Loading media from %s and will write segments to %s", self.media_path, self.save_path)
        segments, info = self.model.transcribe(  # pyright: ignore[reportUnknownMemberType]
            str(self.media_path.absolute()), language=self.language
        )

        self.logger.info(
            "Detected language: %s (p=%.2f)",
            info.language,
            info.language_probability,
        )
        self.logger.info("Audio duration: %.2fs", info.duration)

        progress = create_progress()

        with progress:
            task_id: TaskID = progress.add_task(
                "Transcribing",
                total=info.duration,
            )

            async with await FileWriteStream.from_path(self.save_path) as f:
                async for segment in _iterate_segments(segments, progress, task_id):
                    self.logger.debug(
                        "[%s - %s] %s",
                        segment.start,
                        segment.end,
                        segment.text,
                    )
                    await f.send(orjson.dumps(segment, option=orjson.OPT_APPEND_NEWLINE))
        self.logger.info("Transcription complete")


async def _iterate_segments(
    segments: Iterable[Segment], progress: Progress, task_id: TaskID
) -> AsyncGenerator[Segment]:
    """Async-iterate over segments while updating progress.

    The faster-whisper `segments` is a generator; wrap it for async consumption.
    """
    last_completed = 0.0

    segment_iterator: Iterator[Segment] = iter(segments)

    def _next_segment_blocking() -> Segment | None:
        try:
            return next(segment_iterator)
        except StopIteration:
            return None

    while True:
        # Run the blocking `next()` in a worker thread to avoid blocking the event loop
        segment: Segment | None = await to_thread.run_sync(_next_segment_blocking)
        if segment is None:
            break

        current_completed = float(segment.end)
        if current_completed >= last_completed:
            progress.update(task_id, completed=current_completed)
            last_completed = current_completed

        yield segment


if __name__ == "__main__":
    Cli.run_anyio_static()
