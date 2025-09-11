from pathlib import Path
from typing import ClassVar

from pydantic_settings import BaseSettings, SettingsConfigDict


class DeepResearchSettings(BaseSettings):
    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(env_file=".env")
    model: str = "o4-mini-deep-research"
    openai_api_key: str
    system_prompt_path: Path = Path(__file__).parent / "system_prompt.md"


deep_research_settings = DeepResearchSettings()  # type: ignore[call-arg] # pyright: ignore[reportCallIssue]
