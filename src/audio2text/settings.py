from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_path: str = "deepdml/faster-whisper-large-v3-turbo-ct2"
    compute_type: str = "default"
    whisperx_model: str = "large-v2"
    whisperx_batch_size: int = 16
    whisperx_align: bool = True
    whisperx_return_char_alignments: bool = False


settings = Settings()
