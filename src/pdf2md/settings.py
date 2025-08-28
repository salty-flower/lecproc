from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model: str = "gemini/gemini-2.5-pro"
    max_concurrency: int = 16
    request_timeout_s: float = 600.0
    output_extension: str = "md"
    max_retry_attempts: int = 3


settings = Settings()
