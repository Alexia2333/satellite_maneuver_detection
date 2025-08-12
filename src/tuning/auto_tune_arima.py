# src/tuning/auto_tune_arima.py
import itertools
import warnings
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings("ignore")

# You can fill in according to your report
KNOWN_BEST_PARAMS: Dict[str, Dict[str, Tuple[int, int, int]]] = {
    "jason-1": {"mean_motion": (3, 1, 3), "eccentricity": (2, 1, 1), "inclination": (1, 1, 1)},
    "jason-2": {"mean_motion": (2, 2, 1), "eccentricity": (1, 0, 2), "inclination": (3, 0, 1)},
    "jason-3": {"mean_motion": (1, 1, 1), "eccentricity": (1, 1, 0), "inclination": (2, 1, 1)},
}


def get_stationarity(y: pd.Series, p_value: float = 0.05, max_d: int = 2) -> int:
    """
    Determine the order of differencing (d) needed for stationarity using the ADF test.
    """
    y = pd.Series(y).astype("float64").dropna()
    d = 0
    cur = y.copy()
    for _ in range(max_d + 1):
        try:
            _, p, *_ = adfuller(cur, autolag="AIC", maxlag=None)
        except Exception:
            p = 1.0
        if p < p_value:
            return d
        d += 1
        if d > max_d:
            return max_d
        cur = cur.diff().dropna()
    return min(d, max_d)


def _find_best_arima_order_impl(
    satellite_name: str,
    element_name: str,
    history: pd.Series,
    p_candidates=range(0, 5),
    q_candidates=range(0, 5),
    tail_len: int = 2000,
) -> Tuple[int, int, int]:
    """
    Internal function to find the best ARIMA(p,d,q) order for a given satellite and element.
    Uses known parameters if available, otherwise performs a guided search with constraints.
    """
    s_name_lower = str(satellite_name).strip().lower()
    element_name = str(element_name).strip()

    if s_name_lower in KNOWN_BEST_PARAMS and element_name in KNOWN_BEST_PARAMS[s_name_lower]:
        best_order = KNOWN_BEST_PARAMS[s_name_lower][element_name]
        print(f"  ✅ Found pre-computed optimal parameters from report: {best_order}")
        return best_order

    print("  No pre-computed parameters. Starting guided search...")

    history = pd.Series(history).astype("float64").dropna()
    if history.std() < 1e-12 or len(history) < 16:
        print("  [Warning] Data too flat/short. Returning safe default (1,0,0).")
        return (1, 0, 0)

    d = get_stationarity(history)
    print(f"  ADF test suggests d={d}")

    scaler = StandardScaler()
    y_scaled = pd.Series(
        scaler.fit_transform(history.values.reshape(-1, 1)).ravel(), index=history.index
    )
    y_eval = y_scaled.iloc[-tail_len:] if len(y_scaled) > tail_len else y_scaled

    pq_all = list(itertools.product(p_candidates, q_candidates))
    pq_grid = [(p, q) for (p, q) in pq_all if (p + q) <= 4 and not (p == 0 and q == 0)]
    pq_grid = pq_grid + ([(0, 0)] if (0, 0) not in pq_all else [])  # Ensure (0,d,0) is evaluated at least once

    best_aic = float("inf")
    best_order = None
    fail_cnt = 0
    trend_flag = "c" if d == 0 else "n"

    for p, q in tqdm(pq_grid, desc=f"Grid Search (d={d})", leave=False):
        order = (p, d, q)
        # Two-phase: try stable method first, then faster fallback
        tried = False
        try:
            model = ARIMA(
                y_eval,
                order=order,
                trend=trend_flag,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit()  # Default statespace MLE (more stable)
            tried = True
        except Exception:
            # If default fails and d=0, try CSS method (faster for ARMA)
            if d == 0:
                try:
                    model = ARIMA(
                        y_eval,
                        order=order,
                        trend=trend_flag,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    ).fit(method="css")
                    tried = True
                except Exception:
                    pass
        if not tried:
            fail_cnt += 1
            continue
        aic = model.aic
        if aic < best_aic:
            best_aic, best_order = aic, order

    if best_order is None:
        best_order = (1, min(d, 1), 0)
        print(f"  [Warning] Guided search failed. Falling back to safe default: {best_order}")
    else:
        print(f"  ✅ Guided search found best order: {best_order} (AIC: {best_aic:.2f})")

    return best_order


def _looks_like_series(x) -> bool:
    """
    Check if the input looks like a 1D series (pandas Series, numpy array, or list).
    """
    if isinstance(x, pd.Series):
        return True
    try:
        arr = np.asarray(x)
        return arr.ndim == 1 and arr.size > 0
    except Exception:
        return False


def find_best_arima_order(*args, **kwargs) -> Tuple[int, int, int]:
    """
    Compatibility function supporting two signatures:
    1) Old version: find_best_arima_order(history, satellite_name, element_name, ...)
    2) New version: find_best_arima_order(satellite_name, element_name, history, ...)

    Automatically detects the signature and forwards the call to _find_best_arima_order_impl
    without modifying the main script.
    """
    if len(args) < 3:
        raise TypeError(
            "find_best_arima_order requires at least 3 positional arguments. "
            "Expected either (history, satellite_name, element_name, ...) or (satellite_name, element_name, history, ...)."
        )

    a0, a1, a2 = args[0], args[1], args[2]

    # Old signature: first argument is the time series
    if _looks_like_series(a0):
        history = pd.Series(a0)
        satellite_name = a1
        element_name = a2
        p_candidates = kwargs.get("p_candidates", range(0, 5))
        q_candidates = kwargs.get("q_candidates", range(0, 5))
        tail_len = kwargs.get("tail_len", 2000)
        return _find_best_arima_order_impl(
            satellite_name, element_name, history, p_candidates, q_candidates, tail_len
        )

    # New signature: first argument is the satellite name, third is the time series
    satellite_name = a0
    element_name = a1
    history = pd.Series(a2)
    p_candidates = kwargs.get("p_candidates", range(0, 5))
    q_candidates = kwargs.get("q_candidates", range(0, 5))
    tail_len = kwargs.get("tail_len", 2000)
    return _find_best_arima_order_impl(
        satellite_name, element_name, history, p_candidates, q_candidates, tail_len
    )
