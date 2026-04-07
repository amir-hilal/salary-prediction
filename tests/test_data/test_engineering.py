import pandas as pd
import pytest

from src.features.engineering import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    add_job_family,
    add_location_features,
    build_features,
    encode_ordinals,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_df(**overrides: object) -> pd.DataFrame:
    """Return a one-row DataFrame with sensible defaults for all engineering inputs."""
    row: dict = {
        "work_year": 2023,
        "experience_level": "SE",
        "employment_type": "FT",
        "job_title": "Data Scientist",
        "salary_in_usd": 120_000,
        "remote_ratio": 100,
        "company_location": "US",
        "company_size": "M",
    }
    row.update(overrides)
    return pd.DataFrame([row])


# ── add_job_family ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("title, expected_code", [
    ("Head of Data",              5),  # leadership
    ("Director of Engineering",   5),  # leadership — director keyword
    ("Machine Learning Engineer", 4),  # ml_ai
    ("AI Research Scientist",     4),  # ml_ai — "ai " keyword
    ("Data Engineer",             3),  # data_engineering
    ("Analytics Engineer",        3),  # data_engineering — analytics engineer keyword
    ("Business Analyst",          1),  # analytics
    ("BI Developer",              1),  # analytics — "bi " keyword
    ("Data Scientist",            2),  # data_science
    ("Applied Scientist",         2),  # data_science — applied scientist keyword
    ("Software Developer",        0),  # other — no matching keyword
])
def test_add_job_family_mapping(title: str, expected_code: int) -> None:
    df = _make_df(job_title=title)
    result = add_job_family(df)
    assert result.loc[0, "job_family"] == expected_code, (
        f"job_title='{title}' expected job_family={expected_code}, "
        f"got {result.loc[0, 'job_family']}"
    )


def test_add_job_family_does_not_mutate_input() -> None:
    df = _make_df()
    original_cols = list(df.columns)
    add_job_family(df)
    assert list(df.columns) == original_cols


def test_add_job_family_column_added() -> None:
    df = _make_df()
    result = add_job_family(df)
    assert "job_family" in result.columns


# ── add_location_features ─────────────────────────────────────────────────────


@pytest.mark.parametrize("country, expected_region, expected_us", [
    ("US", 3, 1),   # north_america, is_us_company=1
    ("CA", 3, 0),   # north_america, is_us_company=0
    ("GB", 2, 0),   # western_europe
    ("DE", 2, 0),   # western_europe
    ("IN", 1, 0),   # asia_pacific
    ("JP", 1, 0),   # asia_pacific
    ("AR", 0, 0),   # rest_of_world
    ("NG", 0, 0),   # rest_of_world
])
def test_add_location_features_region_and_us_flag(
    country: str, expected_region: int, expected_us: int
) -> None:
    df = _make_df(company_location=country)
    result = add_location_features(df)
    assert result.loc[0, "location_region"] == expected_region, (
        f"country='{country}' expected location_region={expected_region}"
    )
    assert result.loc[0, "is_us_company"] == expected_us, (
        f"country='{country}' expected is_us_company={expected_us}"
    )


def test_add_location_features_creates_both_columns() -> None:
    df = _make_df()
    result = add_location_features(df)
    assert "location_region" in result.columns
    assert "is_us_company" in result.columns


def test_add_location_features_does_not_mutate_input() -> None:
    df = _make_df()
    original_cols = list(df.columns)
    add_location_features(df)
    assert list(df.columns) == original_cols


# ── encode_ordinals ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("experience_level, expected", [
    ("EN", 0), ("MI", 1), ("SE", 2), ("EX", 3),
])
def test_encode_ordinals_experience_level(experience_level: str, expected: int) -> None:
    df = _make_df(experience_level=experience_level)
    result = encode_ordinals(df)
    assert result.loc[0, "experience_level"] == expected


@pytest.mark.parametrize("company_size, expected", [
    ("S", 0), ("M", 1), ("L", 2),
])
def test_encode_ordinals_company_size(company_size: str, expected: int) -> None:
    df = _make_df(company_size=company_size)
    result = encode_ordinals(df)
    assert result.loc[0, "company_size"] == expected


@pytest.mark.parametrize("employment_type, expected", [
    ("FL", 0), ("PT", 1), ("CT", 2), ("FT", 3),
])
def test_encode_ordinals_employment_type(employment_type: str, expected: int) -> None:
    df = _make_df(employment_type=employment_type)
    result = encode_ordinals(df)
    assert result.loc[0, "employment_type"] == expected


# ── build_features ────────────────────────────────────────────────────────────


def test_build_features_returns_only_expected_columns() -> None:
    df = _make_df()
    result = build_features(df)
    expected = set(FEATURE_COLUMNS) | {TARGET_COLUMN}
    assert set(result.columns) == expected


def test_build_features_no_extra_columns() -> None:
    df = _make_df()
    result = build_features(df)
    assert len(result.columns) == len(FEATURE_COLUMNS) + 1  # +1 for target


def test_build_features_preserves_row_count() -> None:
    rows = [_make_df(job_title=t) for t in ["Data Scientist", "ML Engineer", "Analyst"]]
    df = pd.concat(rows, ignore_index=True)
    result = build_features(df)
    assert len(result) == len(df)
