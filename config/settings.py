from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Paths
    data_raw_path: Path = Path("data/raw/ds_salaries.csv")
    data_processed_path: Path = Path("data/processed")
    models_artifacts_path: Path = Path("models/artifacts")
    models_registry_path: Path = Path("models/registry")

    # Preprocessing
    test_size: float = 0.2
    random_state: int = 42
    iqr_cap_factor: float = 1.5

    # Decision Tree hyperparameter search space
    dt_max_depth_options: list[int | None] = [3, 5, 8, 12, 20, None]
    dt_min_samples_split_options: list[int] = [2, 5, 10, 20]
    dt_min_samples_leaf_options: list[int] = [1, 2, 4, 8]
    dt_cv_folds: int = 5

    # LLM provider — "ollama" for local dev, "groq" for production
    llm_provider: str

    # Ollama (local)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "phi4-mini"
    ollama_timeout: int = 120

    # Groq (production)
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "llama-3.3-70b-versatile"
    groq_timeout: int = 120

    # Supabase
    supabase_url: str
    supabase_anon_key: str
    supabase_service_role_key: str = ""

    # Dashboard → API
    api_base_url: str


settings = Settings()
