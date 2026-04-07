import pandas as pd
import pytest

from src.data.cleaning import cap_salary_outliers, clean, drop_leakage_columns


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def raw_df() -> pd.DataFrame:
    """Minimal DataFrame with all leakage columns present."""
    return pd.DataFrame({
        "work_year": [2022, 2023, 2024],
        "experience_level": ["SE", "MI", "EX"],
        "employment_type": ["FT", "FT", "PT"],
        "job_title": ["Data Scientist", "ML Engineer", "Head of Data"],
        "salary": [100_000, 80_000, 200_000],
        "salary_currency": ["USD", "GBP", "USD"],
        "salary_in_usd": [100_000, 95_000, 200_000],
        "employee_residence": ["US", "GB", "US"],
        "remote_ratio": [100, 0, 50],
        "company_location": ["US", "GB", "US"],
        "company_size": ["M", "S", "L"],
    })


@pytest.fixture()
def salary_df() -> pd.DataFrame:
    """DataFrame with 30 normal salaries and one extreme outlier."""
    normal = [80_000, 85_000, 90_000, 95_000, 100_000, 105_000, 110_000,
              115_000, 120_000, 125_000] * 3  # 30 rows
    return pd.DataFrame({"salary_in_usd": normal + [2_000_000]})


# ── drop_leakage_columns ──────────────────────────────────────────────────────


def test_drop_leakage_columns_removes_salary(raw_df: pd.DataFrame) -> None:
    result = drop_leakage_columns(raw_df)
    assert "salary" not in result.columns


def test_drop_leakage_columns_removes_salary_currency(raw_df: pd.DataFrame) -> None:
    result = drop_leakage_columns(raw_df)
    assert "salary_currency" not in result.columns


def test_drop_leakage_columns_removes_employee_residence(raw_df: pd.DataFrame) -> None:
    result = drop_leakage_columns(raw_df)
    assert "employee_residence" not in result.columns


def test_drop_leakage_columns_keeps_remaining_columns(raw_df: pd.DataFrame) -> None:
    result = drop_leakage_columns(raw_df)
    for col in ["work_year", "salary_in_usd", "company_location", "remote_ratio"]:
        assert col in result.columns


def test_drop_leakage_columns_does_not_mutate_input(raw_df: pd.DataFrame) -> None:
    original_columns = list(raw_df.columns)
    drop_leakage_columns(raw_df)
    assert list(raw_df.columns) == original_columns


def test_drop_leakage_columns_idempotent(raw_df: pd.DataFrame) -> None:
    """Calling twice on an already-cleaned frame must not raise."""
    once = drop_leakage_columns(raw_df)
    twice = drop_leakage_columns(once)
    assert list(once.columns) == list(twice.columns)


# ── cap_salary_outliers ───────────────────────────────────────────────────────


def test_cap_salary_outliers_caps_extreme_high(salary_df: pd.DataFrame) -> None:
    result = cap_salary_outliers(salary_df, factor=1.5)
    assert result["salary_in_usd"].max() < 2_000_000


def test_cap_salary_outliers_preserves_row_count(salary_df: pd.DataFrame) -> None:
    result = cap_salary_outliers(salary_df, factor=1.5)
    assert len(result) == len(salary_df)


def test_cap_salary_outliers_does_not_mutate_input(salary_df: pd.DataFrame) -> None:
    original_max = salary_df["salary_in_usd"].max()
    cap_salary_outliers(salary_df, factor=1.5)
    assert salary_df["salary_in_usd"].max() == original_max


def test_cap_salary_outliers_normal_values_unchanged(salary_df: pd.DataFrame) -> None:
    """Values well within the IQR fence must not be altered."""
    result = cap_salary_outliers(salary_df, factor=1.5)
    # The median value (100k) is never an outlier in this fixture
    assert 100_000 in result["salary_in_usd"].values


# ── clean ─────────────────────────────────────────────────────────────────────


def test_clean_drops_three_columns(raw_df: pd.DataFrame) -> None:
    result = clean(raw_df)
    assert result.shape[1] == raw_df.shape[1] - 3


def test_clean_preserves_row_count(raw_df: pd.DataFrame) -> None:
    result = clean(raw_df)
    assert len(result) == len(raw_df)
