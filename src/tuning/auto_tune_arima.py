# src/tuning/auto_tune_arima.py
import itertools
import warnings
from typing import Tuple, Dict

import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings("ignore")

KNOWN_BEST_PARAMS: Dict[str, Dict[str, Tuple[int, int, int]]] = {
    "jason-1": {"mean_motion": (3, 1, 3), "eccentricity": (2, 1, 1), "inclination": (1, 1, 1)},
    "jason-2": {"mean_motion": (2, 2, 1), "eccentricity": (1, 0, 2), "inclination": (3, 0, 1)},
    "jason-3": {"mean_motion": (3, 1, 0), "eccentricity": (1, 1, 0), "inclination": (1, 0, 1)},
    "fengyun-2d": {"mean_motion": (2, 1, 1), "eccentricity": (1, 0, 2)},
    "fengyun-2f": {"mean_motion": (1, 0, 1)},
}

def get_stationarity(series: pd.Series) -> int:
    adf_result = adfuller(series.dropna())
    if adf_result[1] <= 0.05: return 0
    adf_result = adfuller(series.diff().dropna())
    if adf_result[1] <= 0.05: return 1
    return 2

def find_best_arima_order(
    history: pd.Series,
    satellite_name: str,
    element_name: str,
    p_range: range = range(0, 5),
    q_range: range = range(0, 5),
) -> Tuple[int, int, int]:
    """
    使用集成了您研究成果的智能方法寻找最优ARIMA阶数。
    """
    print(f"\n🔧 Intelligent auto-tuning for {satellite_name} - {element_name}...")
    
    s_name_lower = satellite_name.lower()
    if s_name_lower in KNOWN_BEST_PARAMS and element_name in KNOWN_BEST_PARAMS[s_name_lower]:
        best_order = KNOWN_BEST_PARAMS[s_name_lower][element_name]
        print(f"  ✅ Found pre-computed optimal parameters from report: {best_order}")
        return best_order

    print("  No pre-computed parameters. Starting guided search...")
    if history.std() < 1e-9:
        print("  [Warning] Data has zero variance. Returning (1, 0, 0).")
        return (1, 0, 0)

    d = get_stationarity(history)
    print(f"  ADF test suggests d={d}")

    scaler = StandardScaler()
    y_scaled = scaler.fit_transform(history.values.reshape(-1, 1)).flatten()
    
    best_aic, best_order = float("inf"), None
    pq_grid = list(itertools.product(p_range, q_range))
    iterator = tqdm(pq_grid, desc=f"Grid Search (d={d})", leave=False)

    for p, q in iterator:
        if p == 0 and q == 0: continue
        order = (p, d, q)
        try:
            model = ARIMA(y_scaled, order=order, trend='n').fit(method='css')
            if model.aic < best_aic:
                best_aic, best_order = model.aic, order
        except Exception:
            continue
    
    if best_order is None:
        best_order = (1, 1, 0)
        print(f"  [Warning] Guided search failed. Falling back to safe default: {best_order}")
    else:
        print(f"  ✅ Guided search found best order: {best_order} (AIC: {best_aic:.2f})")
            
    return best_order