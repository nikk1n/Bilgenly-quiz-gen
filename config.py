from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    HF_TOKEN: str
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1  # keep at 1 — model can't be shared across processes

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()