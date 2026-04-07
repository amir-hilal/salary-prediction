import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Columns dropped because they are either redundant or cause data leakage.
# - salary / salary_currency: salary_in_usd is derived from them — keeping them
#   would trivially leak the target.
# - employee_residence: highly correlated with company_location (where the employer
#   pays); adds sparsity without independent signal.
_DROP_COLUMNS = ["salary", "salary_currency", "employee_residence"]


def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that are redundant or would cause data leakage."""
    to_drop = [col for col in _DROP_COLUMNS if col in df.columns]
    result = df.drop(columns=to_drop)
    logger.info(
        "drop_leakage_columns | dropped=%s | shape %s → %s",
        to_drop,
        df.shape,
        result.shape,
    )
    return result


def cap_salary_outliers(df: pd.DataFrame, factor: float = 1.5) -> pd.DataFrame:
    """Cap salary_in_usd outliers using the IQR fence method.

    Values below Q1 - factor*IQR are raised to the lower fence.
    Values above Q3 + factor*IQR are lowered to the upper fence.
    """
    result = df.copy()
    col = "salary_in_usd"

    q1 = result[col].quantile(0.25)
    q3 = result[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr

    n_lower = (result[col] < lower).sum()
    n_upper = (result[col] > upper).sum()

    result[col] = result[col].clip(lower=lower, upper=upper)

    logger.info(
        "cap_salary_outliers | lower_fence=%.0f upper_fence=%.0f | "
        "capped_low=%d capped_high=%d",
        lower,
        upper,
        n_lower,
        n_upper,
    )
    return result


def clean(df: pd.DataFrame, iqr_cap_factor: float = 1.5) -> pd.DataFrame:
    """Run the full cleaning pipeline and return a cleaned DataFrame."""
    logger.info("clean | input shape=%s", df.shape)
    df = drop_leakage_columns(df)
    df = cap_salary_outliers(df, factor=iqr_cap_factor)
    logger.info("clean | output shape=%s", df.shape)
    return df
