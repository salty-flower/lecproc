from pathlib import Path
from typing import Any, override

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    drafting_model: str = "openrouter/google/gemini-2.5-pro"
    fixing_model: str = "openrouter/openrouter/sonoma-sky-alpha"
    max_concurrency: int = 16
    request_timeout_s: float = 600.0
    output_extension: str = "md"
    max_retry_attempts: int = 5

    @override
    def model_post_init(self, _context: Any) -> None:  # pyright: ignore[reportAny]
        self.system_prompt_path = self.system_prompt_path.resolve().absolute()

    system_prompt_path: Path = Path(__file__).parent / "system_prompt.md"


settings = Settings()
