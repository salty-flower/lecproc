import logging
from typing import Any, ClassVar, override

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

from logs import configure_rich_logging, get_logger


class CommonCliSettings(BaseSettings):
    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_file=".env", cli_parse_args=True
    )
    log_level: str = "INFO"

    @override
    def model_post_init(self, _context: Any) -> None:  # pyright: ignore[reportAny,reportExplicitAny]
        configure_rich_logging(level=self.log_level)

    @computed_field
    @property
    def logger(self) -> logging.Logger:
        return get_logger(type(self).__name__)

    async def cli_cmd(self) -> None: ...

    def __await__(self):
        return self.cli_cmd().__await__()
