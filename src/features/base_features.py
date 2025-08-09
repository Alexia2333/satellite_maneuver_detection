
"""
Base feature engineering utilities for satellite maneuver detection.

These functions do not depend on any project-specific modules to keep them portable.
The typical usage is to call `generate_base_features` to augment an input DataFrame
with lag, difference, and rolling statistics for the selected numeric columns.
"""
from typing import List, Optional, Dict, Tuple
import pandas as pd
import numpy as np

def _ensure_sorted(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """
    Ensure the DataFrame is sorted by the time column and the index is monotonic.
    """
    if not df.index.is_monotonic_increasing or (time_col in df.columns):
        if time_col in df.columns:
            df = df.sort_values(time_col).reset_index(drop=True)
        else:
            df = df.sort_index()
    return df

def generate_base_features(
    df: pd.DataFrame,
    time_col: str,
    feature_cols: List[str],
    lags: List[int] = [1, 2, 3, 6, 12],
    rolling_windows: List[int] = [3, 6, 12],
    add_ratios: bool = False,
) -> pd.DataFrame:
    """
    Create common time-series features for numeric telemetry columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input data. Must contain `time_col` and the `feature_cols`.
    time_col : str
        Name of the timestamp column (or an increasing index if absent).
    feature_cols : List[str]
        Numeric columns to transform.
    lags : List[int], optional
        Lag steps to add, by default [1, 2, 3, 6, 12].
    rolling_windows : List[int], optional
        Rolling windows for mean/std/min/max, by default [3, 6, 12].
    add_ratios : bool, optional
        If True, add ratio features like x / rolling_mean(x).

    Returns
    -------
    pd.DataFrame
        A copy of the input with new feature columns appended.
    """
    df = df.copy()
    df = _ensure_sorted(df, time_col)

    for col in feature_cols:
        if col not in df.columns:
            continue

        # Lags and differences
        for L in lags:
            df[f"{col}_lag{L}"] = df[col].shift(L)
            df[f"{col}_diff{L}"] = df[col] - df[col].shift(L)

        # Rolling stats
        for W in rolling_windows:
            roll = df[col].rolling(W, min_periods=max(2, W // 2))
            df[f"{col}_rmean{W}"] = roll.mean()
            df[f"{col}_rstd{W}"]  = roll.std(ddof=0)
            df[f"{col}_rmin{W}"]  = roll.min()
            df[f"{col}_rmax{W}"]  = roll.max()
            if add_ratios:
                df[f"{col}_ratio_rmean{W}"] = df[col] / (df[f"{col}_rmean{W}"] + 1e-9)
                df[f"{col}_ratio_rstd{W}"]  = df[col] / (df[f"{col}_rstd{W}"] + 1e-9)

    # Replace inf and keep NaNs for later dropping/imputation
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df
