from typing import ClassVar

from pydantic_settings import BaseSettings, SettingsConfigDict


class CloudLLMSettings(BaseSettings):
    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(env_file=".env", cli_parse_args=True)

    gemini_api_key: str | None = None


cloud_llm_settings = CloudLLMSettings()
