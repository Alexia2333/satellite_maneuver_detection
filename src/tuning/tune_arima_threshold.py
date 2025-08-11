# src/tuning/tune_arima_threshold.py
from typing import List, Tuple, Iterable, Optional, Dict
from datetime import timedelta

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.models.arima_detector import EnhancedARIMADetector


# ---------- Event-level matching ----------
def _greedy_event_match(
    pred_times: List[pd.Timestamp],
    true_times: List[pd.Timestamp],
    window: pd.Timedelta,
) -> Tuple[int, int, int]:
    """
    Greedy 1-to-1 event match within +/- window.
    Returns (TP, FP, FN).
    """
    if not pred_times and not true_times:
        return 0, 0, 0
    if not pred_times:
        return 0, 0, len(true_times)
    if not true_times:
        return 0, len(pred_times), 0

    preds = sorted(set(pd.to_datetime(pred_times)))
    trues = sorted(set(pd.to_datetime(true_times)))

    tp = 0
    i = j = 0
    while i < len(preds) and j < len(trues):
        p, t = preds[i], trues[j]
        if p < t - window:
            i += 1
        elif p > t + window:
            j += 1
        else:
            # hit (1-to-1)
            tp += 1
            i += 1
            j += 1

    fp = len(preds) - tp
    fn = len(trues) - tp
    return tp, fp, fn


# ---------- Evaluation with caching ----------
def _evaluate_metrics(
    detector: EnhancedARIMADetector,
    series: pd.Series,
    true_maneuver_times: List[pd.Timestamp],
    factor: float,
    tol_days: int,
    cache: Dict[float, Tuple[float, float, float]],
) -> Tuple[float, float, float]:
    """
    Evaluate (precision, recall, f1) at a given factor.
    Uses a small cache to avoid repeated detect() calls.
    """
    f = float(factor)
    if f in cache:
        return cache[f]

    detector.cfg.threshold_factor = f
    df = detector.detect(series)
    pred_times = df.index[df["is_anomaly"]].to_list()
    tp, fp, fn = _greedy_event_match(pred_times, true_maneuver_times, timedelta(days=tol_days))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    cache[f] = (precision, recall, f1)
    return precision, recall, f1


# ---------- Main: F1-oriented threshold tuning ----------
def find_best_threshold_factor(
    detector_instance: EnhancedARIMADetector,
    val_data: pd.Series,
    true_maneuver_times: List[pd.Timestamp],
    factors_to_test: Optional[Iterable[float]] = None,
    tol_days: int = 2,
    coarse_min: float = 0.05,
    coarse_max: float = 12.0,
    coarse_num: int = 21,
    refine_rounds: int = 2,
) -> float:
    """
    F1-driven threshold tuning:
      1) Coarse search on a log-spaced grid
      2) Local refinement around the best coarse factor (log-domain neighborhood)

    Notes:
      - No deepcopy: we reuse the fitted detector and only change threshold_factor.
      - Cached evaluations to reduce repeated detect() calls.
      - Progress bars provided by tqdm.
    """
    cache: Dict[float, Tuple[float, float, float]] = {}

    # Coarse grid
    if factors_to_test is None:
        factors = np.unique(np.logspace(np.log10(coarse_min), np.log10(coarse_max), num=coarse_num))
    else:
        factors = np.unique(np.array(list(factors_to_test), dtype=float))

    best_f: Optional[float] = None
    best_tuple = (-1.0, -1.0, -1.0)  # (P, R, F1)

    print(f"[Threshold Tuning] Coarse search over {len(factors)} candidates...")
    for f in tqdm(factors, desc="Coarse Search", leave=False):
        p, r, f1 = _evaluate_metrics(detector_instance, val_data, true_maneuver_times, f, tol_days, cache)
        if f1 > best_tuple[2]:
            best_tuple = (p, r, f1)
            best_f = float(f)

    if best_f is None:
        best_f = 1.0
        detector_instance.cfg.threshold_factor = best_f
        print("[Warning] F1 search failed; falling back to 1.0")
        return best_f

    # Local refinements
    for round_id in range(refine_rounds):
        neigh = np.unique(
            np.clip(
                best_f * np.array([0.6, 0.75, 0.85, 0.95, 1.0, 1.05, 1.2, 1.4], dtype=float),
                coarse_min,
                coarse_max,
            )
        )
        improved = False
        print(f"[Threshold Tuning] Refinement round {round_id + 1} over {len(neigh)} candidates...")
        for f in tqdm(neigh, desc=f"Refine Round {round_id + 1}", leave=False):
            p, r, f1 = _evaluate_metrics(detector_instance, val_data, true_maneuver_times, f, tol_days, cache)
            if f1 > best_tuple[2] + 1e-12:
                best_tuple = (p, r, f1)
                best_f = float(f)
                improved = True
        if not improved:
            break

    detector_instance.cfg.threshold_factor = best_f
    print(
        f"[Threshold Tuning] Best factor for F1: {best_f:.3f} "
        f"(P={best_tuple[0]:.3f}, R={best_tuple[1]:.3f}, F1={best_tuple[2]:.3f})"
    )
    return best_f


# ---------- Compatibility wrapper ----------
def find_factor_for_target_recall(
    detector_instance: EnhancedARIMADetector,
    val_data: pd.Series,
    true_maneuver_times: List[pd.Timestamp],
    target_recall: float = 0.5,  # kept for signature compatibility; not used
    lo: float = 0.2,
    hi: float = 5.0,
    max_steps: int = 18,
    tol_days: int = 2,
    **kwargs,
) -> float:
    """
    Compatibility wrapper: regardless of legacy invocation, redirect to the F1-oriented search.
    """
    print("[Threshold Tuning] Using F1-oriented threshold tuning (compat wrapper).")
    return find_best_threshold_factor(
        detector_instance, val_data, true_maneuver_times, tol_days=tol_days
    )
