from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Paths
    data_raw_path: Path = Path("data/raw/ds_salaries.csv")
    data_processed_path: Path = Path("data/processed")
    models_artifacts_path: Path = Path("models/artifacts")

    # Preprocessing
    test_size: float = 0.2
    random_state: int = 42
    iqr_cap_factor: float = 1.5


settings = Settings()
