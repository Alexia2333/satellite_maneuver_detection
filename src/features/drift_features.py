
"""
Drift and cumulative-pressure features for GEO satellites (also usable for others).

The goal is to emphasize gradual deviations from a moving baseline and accumulating
"pressure" that often precedes maneuvers in station-keeping regimes.
"""
from typing import List
import pandas as pd
import numpy as np

def _zscore(x: pd.Series, eps: float = 1e-9) -> pd.Series:
    m = x.rolling(48, min_periods=12).mean()
    s = x.rolling(48, min_periods=12).std(ddof=0)
    return (x - m) / (s + eps)

def rolling_slope(x: pd.Series, window: int = 24) -> pd.Series:
    """
    Compute the slope from a rolling linear regression with intercept, using
    a simple closed-form over a sliding window. Window is the number of samples.
    """
    idx = np.arange(len(x))
    # Precompute sums in a rolling manner
    s1 = x.rolling(window, min_periods=max(4, window//3)).sum()
    s2 = pd.Series(idx, index=x.index).rolling(window, min_periods=max(4, window//3)).sum()
    s11 = (x**2).rolling(window, min_periods=max(4, window//3)).sum()
    s22 = (pd.Series(idx, index=x.index)**2).rolling(window, min_periods=max(4, window//3)).sum()
    s12 = (x * pd.Series(idx, index=x.index)).rolling(window, min_periods=max(4, window//3)).sum()
    n = x.rolling(window, min_periods=max(4, window//3)).count()
    denom = (n * s22 - s2**2).replace(0, np.nan)
    slope = (n * s12 - s2 * s1) / denom
    return slope

def generate_drift_features(
    df: pd.DataFrame,
    time_col: str,
    feature_cols: List[str],
    zscore_window: int = 48,
    slope_window: int = 24,
) -> pd.DataFrame:
    """
    Create drift-oriented features:
      - z-score vs. rolling mean/std (longer window, defaults for GEO cadence)
      - rolling slope as a measure of trend
      - cumulative absolute z-score ("pressure") as an integrator of stress

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    time_col : str
        Timestamp column name; used only for ordering.
    feature_cols : List[str]
        Numeric columns to transform.
    zscore_window : int, optional
        Window used to compute the rolling mean/std for z-score baseline.
    slope_window : int, optional
        Window for rolling slope trend.

    Returns
    -------
    pd.DataFrame
        DataFrame with drift features appended.
    """
    df = df.copy()
    if time_col in df.columns:
        df = df.sort_values(time_col).reset_index(drop=True)

    for col in feature_cols:
        if col not in df.columns:
            continue
        m = df[col].rolling(zscore_window, min_periods=max(8, zscore_window//4)).mean()
        s = df[col].rolling(zscore_window, min_periods=max(8, zscore_window//4)).std(ddof=0)
        z = (df[col] - m) / (s + 1e-9)
        df[f"{col}_z_{zscore_window}"] = z
        df[f"{col}_slope_{slope_window}"] = rolling_slope(df[col], slope_window)
        # Cumulative pressure: accumulate absolute deviation, but leak with a small decay
        decay = 0.001
        press = [np.nan] * len(df)
        acc = 0.0
        for i, v in enumerate(z.fillna(0.0).values):
            acc = acc * (1.0 - decay) + abs(float(v))
            press[i] = acc
        df[f"{col}_cum_pressure"] = press

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df
