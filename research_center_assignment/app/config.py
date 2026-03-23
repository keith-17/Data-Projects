from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_path: Path = BASE_DIR / "models" / "final_kmeans_pipeline.pkl"
    model_version: str = "1.0.0"

    model_config = SettingsConfigDict(env_file=str(BASE_DIR / ".env"))


settings = Settings()
