from typing import ClassVar

from pydantic_settings import BaseSettings, SettingsConfigDict


class DeepResearchSettings(BaseSettings):
    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(env_file=".env")

    model: str = "o4-mini-deep-research"
    openai_api_key: str

    citation_display_limit: int = 10


deep_research_settings = DeepResearchSettings()  # type: ignore[call-arg] # pyright: ignore[reportCallIssue]
