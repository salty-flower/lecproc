import gc
from collections.abc import AsyncGenerator, Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, override

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


@dataclass(slots=True)
class TranscriptionSummary:
    duration: float | None
    language: str | None
    language_probability: float | None


SegmentPayload = dict[str, Any]


class Cli(CommonCliSettings):
    media_path: CliPositionalArg[Path]
    save_path: CliPositionalArg[Path]
    language: str | None = None
    use_whisperx: bool = False

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

        if self.use_whisperx:
            segments, summary = self._transcribe_with_whisperx()
        else:
            segments, summary = self._transcribe_with_faster_whisper()

        if summary.language_probability is not None:
            self.logger.info(
                "Detected language: %s (p=%.2f)",
                summary.language,
                summary.language_probability,
            )
        elif summary.language is not None:
            self.logger.info("Detected language: %s", summary.language)
        else:
            self.logger.info("Detected language could not be determined")

        if summary.duration is not None:
            self.logger.info("Audio duration: %.2fs", summary.duration)
        else:
            self.logger.info("Audio duration could not be determined")

        progress = create_progress()

        with progress:
            task_id: TaskID = progress.add_task(
                "Transcribing",
                total=summary.duration,
            )

            async with await FileWriteStream.from_path(self.save_path) as f:
                async for segment in _iterate_segments(segments, progress, task_id):
                    self.logger.debug(
                        "[%s - %s] %s",
                        segment.get("start"),
                        segment.get("end"),
                        segment.get("text", ""),
                    )
                    await f.send(orjson.dumps(segment, option=orjson.OPT_APPEND_NEWLINE))
        self.logger.info("Transcription complete")

    def _transcribe_with_faster_whisper(self) -> tuple[Iterable[SegmentPayload], TranscriptionSummary]:
        segments, info = self.model.transcribe(  # pyright: ignore[reportUnknownMemberType]
            str(self.media_path.absolute()),
            language=self.language,
        )

        summary = TranscriptionSummary(
            duration=float(info.duration),
            language=info.language,
            language_probability=info.language_probability,
        )

        return _segments_asdict(segments), summary

    def _transcribe_with_whisperx(self) -> tuple[Iterable[SegmentPayload], TranscriptionSummary]:
        try:
            import whisperx  # type: ignore[import-not-found]  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - runtime guard
            message = "WhisperX support requires the 'whisperx' extra. Install it with `uv add whisperx`."
            raise RuntimeError(message) from exc

        import torch  # noqa: PLC0415

        device = "cuda"
        self.logger.info(
            "Loading WhisperX model '%s' on %s (compute_type=%s)",
            settings.whisperx_model,
            device,
            settings.compute_type,
        )

        model = whisperx.load_model(
            settings.whisperx_model,
            device,
            compute_type=settings.compute_type,
            download_root=str(path_settings.models_download_dir.absolute()),
        )
        try:
            audio = whisperx.load_audio(str(self.media_path.absolute()))

            transcribe_kwargs: dict[str, Any] = {"batch_size": settings.whisperx_batch_size}
            if self.language:
                transcribe_kwargs["language"] = self.language

            result = model.transcribe(audio, **transcribe_kwargs)
            segments: list[SegmentPayload] = list(result.get("segments", []))
            language = result.get("language", self.language)

            if settings.whisperx_align and language:
                self.logger.info("Running WhisperX alignment for language '%s'", language)
                model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
                try:
                    aligned_result = whisperx.align(
                        segments,
                        model_a,
                        metadata,
                        audio,
                        device,
                        return_char_alignments=settings.whisperx_return_char_alignments,
                    )
                finally:
                    if torch.cuda.is_available():  # pragma: no cover - optional GPU cleanup
                        torch.cuda.empty_cache()
                segments = list(aligned_result.get("segments", segments))
                language = aligned_result.get("language", language)
                del model_a
                del metadata
        finally:
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
            del model

        summary = TranscriptionSummary(
            duration=_infer_duration_from_segments(segments),
            language=language,
            language_probability=None,
        )

        return segments, summary


async def _iterate_segments(
    segments: Iterable[SegmentPayload], progress: Progress, task_id: TaskID
) -> AsyncGenerator[SegmentPayload]:
    """Async-iterate over segments while updating progress.

    The `segments` iterable may be backed by a generator (faster-whisper) or a list (WhisperX);
    wrap access in a worker thread so the event loop stays responsive.
    """
    last_completed = 0.0

    segment_iterator: Iterator[SegmentPayload] = iter(segments)

    def _next_segment_blocking() -> SegmentPayload | None:
        try:
            return next(segment_iterator)
        except StopIteration:
            return None

    while True:
        # Run the blocking `next()` in a worker thread to avoid blocking the event loop
        segment: SegmentPayload | None = await to_thread.run_sync(_next_segment_blocking)
        if segment is None:
            break

        end_time = segment.get("end")
        current_completed = float(end_time) if end_time is not None else last_completed
        if current_completed >= last_completed:
            progress.update(task_id, completed=current_completed)
            last_completed = current_completed

        yield segment


def _segments_asdict(segments: Iterable[Segment]) -> Iterator[SegmentPayload]:
    for segment in segments:
        yield asdict(segment)


def _infer_duration_from_segments(segments: Iterable[SegmentPayload]) -> float | None:
    max_end: float | None = None
    for segment in segments:
        end_time = segment.get("end")
        if end_time is None:
            continue
        end_value = float(end_time)
        if max_end is None or end_value > max_end:
            max_end = end_value
    return max_end


if __name__ == "__main__":
    Cli.run_anyio_static()
