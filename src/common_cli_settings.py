import logging
from asyncio.exceptions import CancelledError
from inspect import isclass
from typing import Any, ClassVar, cast, override

import anyio
from pydantic import computed_field
from pydantic_settings import BaseSettings, CliApp, SettingsConfigDict
from pydantic_settings.sources.types import _CliSubCommand

from logs import configure_rich_logging, get_logger

DEFAULT_MODEL_CONFIG: SettingsConfigDict = SettingsConfigDict(env_file=".env", arbitrary_types_allowed=True)


class CommonCliSettings(BaseSettings):
    """Base settings for CLI apps."""

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(**DEFAULT_MODEL_CONFIG)
    log_level: str = "INFO"

    is_root: ClassVar[bool | None] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:  # pyright: ignore[reportAny]  # noqa: ANN401
        """
        Set up `model_config` for subclasses based on their annotations.

        If the subclass declares any `CliSubCommand[...]` annotated fields,
        enable `cli_parse_args=True`
        so pydantic-settings will parse CLI arguments and support subcommands.
        Otherwise keep the default (no CLI parsing).
        """
        super().__init_subclass__(**kwargs)  # pyright: ignore[reportAny]

        if cls.is_root or cls.has_subcommand():
            cls.model_config = SettingsConfigDict(**DEFAULT_MODEL_CONFIG, cli_parse_args=True)  # pyright: ignore[reportCallIssue]

    @override
    def model_post_init(self, _context: Any) -> None:  # pyright: ignore[reportAny]
        configure_rich_logging(level=self.log_level)

    @classmethod
    def has_subcommand(cls) -> bool:
        for v in cls.model_fields.values():
            if v.metadata:
                m1 = cast("object", v.metadata[0])
                if isclass(m1) and issubclass(m1, _CliSubCommand):
                    return True
        return False

    @computed_field
    @property
    def logger(self) -> logging.Logger:
        return get_logger(type(self).__name__)

    async def cli_cmd_async(self) -> None: ...

    def cli_cmd(self) -> None:
        """Default method for top-level invocation.

        If this class declares subcommands, delegate to the pydantic-settings
        `CliApp.run_subcommand` helper. Otherwise run the async command path.
        """
        if self.has_subcommand():
            _ = CliApp.run_subcommand(self)
        else:
            self.run_anyio()

    @classmethod
    def run_anyio_static(cls) -> None:
        _ = CliApp.run(cls, cli_cmd_method_name="run_anyio")

    def run_anyio(self) -> None:
        """Instantiate CLI settings and run `cli_cmd_async` under AnyIO."""

        async def _main() -> None:
            try:
                return await self.cli_cmd_async()
            except CancelledError:
                self.logger.info("Cancellation received. Exiting...")
            except KeyboardInterrupt:
                self.logger.info("Keyboard interrupt received. Exiting...")
                raise

        return anyio.run(_main, backend="asyncio")
