from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_path: str = "large-v3"
    compute_type: str = "default"


settings = Settings()
