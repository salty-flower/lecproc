from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model: str = "openrouter/google/gemini-2.5-flash"
    max_concurrency: int = 16
    request_timeout_s: float = 600.0
    output_extension: str = "md"
    max_retry_attempts: int = 1


settings = Settings()
