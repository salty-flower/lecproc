from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_path: str = "deepdml/faster-whisper-large-v3-turbo-ct2"
    compute_type: str = "default"


settings = Settings()
