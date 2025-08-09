
"""
Metrics and threshold utilities, including temporal clustering
to avoid over-counting dense detections around a single maneuver.
"""
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, f1_score, average_precision_score

def temporal_cluster(
    timestamps: pd.Series,
    scores: pd.Series,
    window: int = 3
) -> pd.DataFrame:
    """
    Group nearby high-score points into clusters and keep the peak per cluster.

    Parameters
    ----------
    timestamps : pd.Series
        Series of timestamps or monotonically increasing integers.
    scores : pd.Series
        Anomaly scores or model probabilities aligned with timestamps.
    window : int
        Minimum separation (in number of samples) between clusters.

    Returns
    -------
    pd.DataFrame
        Columns: ["timestamp", "score", "cluster_id"]
    """
    idx = np.argsort(timestamps.values)
    t = timestamps.values[idx]
    s = scores.values[idx]
    clusters = []
    cur_id = -1
    last_t = None
    for ti, si in zip(t, s):
        if last_t is None or (ti - last_t) > window:
            cur_id += 1
        clusters.append((ti, si, cur_id))
        last_t = ti
    dfc = pd.DataFrame(clusters, columns=["timestamp", "score", "cluster_id"])
    # Keep the peak score per cluster
    dfc = dfc.loc[dfc.groupby("cluster_id")["score"].idxmax()].sort_values("timestamp").reset_index(drop=True)
    return dfc

def find_best_threshold_youden(
    y_true: np.ndarray,
    y_score: np.ndarray
) -> float:
    """
    Choose threshold by maximizing F1 via PR sweep.
    This is robust for imbalanced positive class.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1s = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx = f1s.argmax()
    # thresholds has length N-1; align index safely
    thr = thresholds[min(best_idx, len(thresholds)-1)] if len(thresholds) > 0 else 0.5
    return float(thr)

def evaluate_at_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thr: float
) -> dict:
    """
    Compute precision/recall/F1 and average precision at a fixed threshold.
    """
    y_pred = (y_score >= thr).astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    ap = average_precision_score(y_true, y_score)
    precision = (y_pred[y_true == 1].sum() / max(1, y_pred.sum())) if y_pred.sum() > 0 else 0.0
    recall = (y_pred[y_true == 1].sum() / max(1, (y_true == 1).sum())) if (y_true == 1).sum() > 0 else 0.0
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1), "ap": float(ap)}
