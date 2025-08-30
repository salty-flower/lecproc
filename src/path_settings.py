from pathlib import Path
from typing import Any, override

from pydantic_settings import BaseSettings


class PathSettings(BaseSettings):
    models_download_dir: Path = Path(__file__).parent / ".." / "hf_downloads"

    @override
    def model_post_init(self, _context: Any) -> None:  # pyright: ignore[reportAny]
        self.models_download_dir = self.models_download_dir.resolve().absolute()
        self.models_download_dir.mkdir(parents=True, exist_ok=True)


path_settings = PathSettings()
