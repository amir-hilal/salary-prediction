import io
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: frozenset[str] = frozenset({
    "work_year",
    "experience_level",
    "employment_type",
    "job_title",
    "salary",
    "salary_currency",
    "salary_in_usd",
    "employee_residence",
    "remote_ratio",
    "company_location",
    "company_size",
})


def load_raw(path: Path) -> pd.DataFrame:
    """Load the raw CSV, drop the auto-index column if present, and validate schema."""
    df = pd.read_csv(path)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Raw dataset is missing required columns: {missing}")

    logger.info("load_raw | shape=%s | dtypes=%s", df.shape, df.dtypes.to_dict())
    return df


def load_raw_from_supabase() -> pd.DataFrame:
    """Download ds_salaries.csv from Supabase Storage and return as a DataFrame.

    Used in production (Streamlit Cloud / Koyeb) where the local file is not
    available.  Requires SUPABASE_URL and SUPABASE_ANON_KEY to be set.

    Raises:
        RuntimeError: if the download fails.
    """
    from supabase import create_client  # noqa: PLC0415

    from config.settings import settings  # noqa: PLC0415

    logger.info("load_raw | downloading ds_salaries.csv from Supabase Storage")
    try:
        client = create_client(settings.supabase_url, settings.supabase_anon_key)
        csv_bytes: bytes = client.storage.from_(settings.supabase_storage_bucket).download(
            "ds_salaries.csv"
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to download ds_salaries.csv from Supabase Storage: {exc}") from exc

    df = pd.read_csv(io.BytesIO(csv_bytes))
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Downloaded dataset is missing required columns: {missing}")

    logger.info("load_raw (supabase) | shape=%s", df.shape)
    return df
