from pathlib import Path

from pydantic_settings import BaseSettings


class PathSettings(BaseSettings):
    models_download_dir: Path = Path(__file__).parent / ".." / "hf_downloads"


path_settings = PathSettings()
