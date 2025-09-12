from typing import ClassVar

from pydantic import ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class DeepResearchSettings(BaseSettings):
    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(env_file=".env")

    model: str = "o4-mini-deep-research"
    openai_api_key: str

    citation_display_limit: int = 10


try:
    deep_research_settings = DeepResearchSettings()  # type: ignore[call-arg] # pyright: ignore[reportCallIssue]
except ValidationError:
    import sys

    from logs import get_logger

    logger = get_logger(__name__)
    logger.error("OpenAI API Key not set.")  # noqa: TRY400
    sys.exit(1)
