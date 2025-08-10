
"""
Metrics and threshold utilities, including temporal clustering
to avoid over-counting dense detections around a single maneuver.
"""
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
from datetime import timedelta
from typing import List, Dict, Tuple
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

def evaluate_detection(detected_epochs: List[pd.Timestamp], 
                         true_maneuvers: List[pd.Timestamp], 
                         matching_window: timedelta = timedelta(days=1)) -> Tuple[Dict, List]:
    """
    Evaluates detection performance by matching detected anomalies to ground truth events.

    Args:
        detected_epochs (List[pd.Timestamp]): A list of timestamps where anomalies were detected.
        true_maneuvers (List[pd.Timestamp]): A list of ground truth maneuver timestamps.
        matching_window (timedelta): The time window around a true maneuver to count a detection as a true positive.

    Returns:
        A tuple containing:
        - metrics (Dict): A dictionary with performance metrics (tp, fp, fn, precision, recall, f1).
        - matched_pairs (List): A list of (detection, true_maneuver) pairs.
    """
    if not detected_epochs:
        return {
            'tp': 0, 'fp': 0, 'fn': len(true_maneuvers),
            'precision': 0.0, 'recall': 0.0, 'f1': 0.0
        }, []

    if not true_maneuvers:
        return {
            'tp': 0, 'fp': len(detected_epochs), 'fn': 0,
            'precision': 0.0, 'recall': 0.0, 'f1': 0.0
        }, []

    # Sort both lists to ensure chronological order
    detected_epochs = sorted(list(set(detected_epochs)))
    true_maneuvers = sorted(list(set(true_maneuvers)))

    matched_pairs = []
    false_positives = list(detected_epochs)
    detected_true_events = []

    for maneuver_time in true_maneuvers:
        window_start = maneuver_time - matching_window
        window_end = maneuver_time + matching_window
        
        # Find any detections within the window of the true maneuver
        detections_in_window = [
            det for det in false_positives if window_start <= det <= window_end
        ]
        
        if detections_in_window:
            # If there's a match, take the closest detection as the true positive
            closest_detection = min(detections_in_window, key=lambda det: abs(det - maneuver_time))
            
            # This maneuver has been detected
            if maneuver_time not in detected_true_events:
                 detected_true_events.append(maneuver_time)
            
            # This detection is a true positive, so remove it from the list of false positives
            false_positives.remove(closest_detection)
            
            matched_pairs.append((closest_detection, maneuver_time))

    tp = len(detected_true_events)
    fp = len(false_positives)
    fn = len(true_maneuvers) - tp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / len(true_maneuvers) if len(true_maneuvers) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics = {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    return metrics, matched_pairs
