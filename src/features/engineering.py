import logging

import pandas as pd

logger = logging.getLogger(__name__)

# ── Feature and target column lists ───────────────────────────────────────────
# Used by both training and inference — do not modify without updating both.
FEATURE_COLUMNS: list[str] = [
    "experience_level",  # ordinal integer (0–3)
    "employment_type",   # ordinal integer (0–3)
    "remote_ratio",      # numeric: 0, 50, 100
    "company_size",      # ordinal integer (0–2)
    "work_year",         # numeric (year integer)
    "job_family",        # label integer (0–5)
    "location_region",   # label integer (0–3)
    "is_us_company",     # binary integer (0 or 1)
]

TARGET_COLUMN: str = "salary_in_usd"

# ── Ordinal mappings (explicit, not sklearn-fitted) ───────────────────────────
# These are based on a well-defined natural order and are deterministic —
# no fitting required.
_EXPERIENCE_ORDER: dict[str, int] = {"EN": 0, "MI": 1, "SE": 2, "EX": 3}
_COMPANY_SIZE_ORDER: dict[str, int] = {"S": 0, "M": 1, "L": 2}
# employment_type has no true numeric order; assigned roughly by typical
# commitment level so the encoding at least produces a monotone-ish axis.
_EMPLOYMENT_TYPE_ORDER: dict[str, int] = {"FL": 0, "PT": 1, "CT": 2, "FT": 3}

# ── job_family ────────────────────────────────────────────────────────────────
# Keyword-based classification into 5 functional groups + "other".
# Order reflects increasing median salary (rough, used for label encoding).
_JOB_FAMILY_ORDER: dict[str, int] = {
    "other": 0,
    "analytics": 1,
    "data_science": 2,
    "data_engineering": 3,
    "ml_ai": 4,
    "leadership": 5,
}

_LEADERSHIP_KW = ("head", "director", "principal", "lead ", "manager", "vp ", "chief")
_ML_AI_KW = ("machine learning", "ml ", " ai ", "ai ", "deep learning", "nlp", "computer vision", "reinforcement")
_DATA_ENG_KW = ("data engineer", "analytics engineer", "etl", "data architect", "data infrastructure")
_ANALYTICS_KW = ("analyst", "bi ", "business intelligence", "reporting")
_DATA_SCI_KW = ("data scientist", "research scientist", "applied scientist", "applied data")


def _infer_job_family(title: str) -> str:
    t = title.lower()
    if any(kw in t for kw in _LEADERSHIP_KW):
        return "leadership"
    if any(kw in t for kw in _ML_AI_KW):
        return "ml_ai"
    if any(kw in t for kw in _DATA_ENG_KW):
        return "data_engineering"
    if any(kw in t for kw in _ANALYTICS_KW):
        return "analytics"
    if any(kw in t for kw in _DATA_SCI_KW):
        return "data_science"
    return "other"


# ── location_region ───────────────────────────────────────────────────────────
_NORTH_AMERICA: frozenset[str] = frozenset({"US", "CA", "MX"})
_WESTERN_EUROPE: frozenset[str] = frozenset({
    "GB", "DE", "FR", "NL", "ES", "PT", "CH", "IT", "BE", "SE",
    "NO", "DK", "AT", "IE", "FI", "PL", "CZ", "HU", "RO", "GR",
    "HR", "RS", "SK", "SI", "LT", "LV", "EE", "LU", "MT", "CY", "BG",
})
_ASIA_PACIFIC: frozenset[str] = frozenset({
    "IN", "JP", "AU", "SG", "CN", "KR", "NZ", "PK", "MY",
    "PH", "TH", "ID", "VN", "HK", "TW", "BD", "LK",
})

_LOCATION_REGION_ORDER: dict[str, int] = {
    "rest_of_world": 0,
    "asia_pacific": 1,
    "western_europe": 2,
    "north_america": 3,
}


def _map_location_region(country: str) -> str:
    if country in _NORTH_AMERICA:
        return "north_america"
    if country in _WESTERN_EUROPE:
        return "western_europe"
    if country in _ASIA_PACIFIC:
        return "asia_pacific"
    return "rest_of_world"


# ── Public engineering functions ──────────────────────────────────────────────

def add_job_family(df: pd.DataFrame) -> pd.DataFrame:
    """Derive job_family (integer label) from job_title using keyword matching."""
    result = df.copy()
    result["job_family"] = (
        result["job_title"]
        .map(_infer_job_family)
        .map(_JOB_FAMILY_ORDER)
    )
    logger.info("add_job_family | distribution=%s", result["job_family"].value_counts().to_dict())
    return result


def add_location_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive location_region (integer label) and is_us_company (0/1) from company_location."""
    result = df.copy()
    result["location_region"] = (
        result["company_location"]
        .map(_map_location_region)
        .map(_LOCATION_REGION_ORDER)
    )
    result["is_us_company"] = (result["company_location"] == "US").astype(int)
    logger.info(
        "add_location_features | region_distribution=%s | us_companies=%d",
        result["location_region"].value_counts().to_dict(),
        result["is_us_company"].sum(),
    )
    return result


def encode_ordinals(df: pd.DataFrame) -> pd.DataFrame:
    """Map ordinal categorical columns to integers using predefined orderings."""
    result = df.copy()
    result["experience_level"] = result["experience_level"].map(_EXPERIENCE_ORDER)
    result["company_size"] = result["company_size"].map(_COMPANY_SIZE_ORDER)
    result["employment_type"] = result["employment_type"].map(_EMPLOYMENT_TYPE_ORDER)
    return result


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature engineering pipeline.

    Returns a DataFrame containing FEATURE_COLUMNS + TARGET_COLUMN only.
    """
    logger.info("build_features | input shape=%s", df.shape)
    df = add_job_family(df)
    df = add_location_features(df)
    df = encode_ordinals(df)
    result = df[FEATURE_COLUMNS + [TARGET_COLUMN]]
    logger.info("build_features | output shape=%s", result.shape)
    return result
