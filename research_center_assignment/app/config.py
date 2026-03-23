from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_path: str = "models/final_kmeans_pipeline.pkl"
    model_version: str = "1.0.0"

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()