import logging
from asyncio.exceptions import CancelledError
from inspect import isclass
from typing import Any, cast, override

import anyio
from pydantic import computed_field
from pydantic_settings import BaseSettings, CliApp, SettingsConfigDict
from pydantic_settings.sources.types import _CliSubCommand

from logs import configure_rich_logging, get_logger


class CommonCliSettings(BaseSettings):
    log_level: str = "INFO"

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
        cls().run_anyio()

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
