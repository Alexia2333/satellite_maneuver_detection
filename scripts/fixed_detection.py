"""
Fixed version with proper clustering for sparse time series
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
import argparse
import numpy as np
import pandas as pd
from typing import Tuple, List

# Ensure project root is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import SatelliteDataLoader
from src.data.feature_engineer import EnhancedSatelliteFeatureEngineer
from src.models.maneuver_xgb_detector import ManeuverXGBDetector, DetectorConfig


def build_labels_by_window(df: pd.DataFrame, time_col: str, events, window_min: int, out_col: str) -> pd.Series:
    """Binary labels: 1 within +/- window_min minutes of any event, else 0."""
    dt = pd.to_datetime(df[time_col], errors="coerce")
    if dt.isna().any():
        raise ValueError(f"Failed to parse timestamps in column '{time_col}'.")
    y = np.zeros(len(df), dtype=int)
    half = pd.to_timedelta(window_min, unit="m")
    for ev in events:
        mask = (dt >= ev - half) & (dt <= ev + half)
        y[mask.values] = 1
    return pd.Series(y, index=df.index, name=out_col)


def temporal_cluster_by_time(timestamps: pd.Series, scores: pd.Series, window_days: float = 4.0) -> pd.DataFrame:
    """
    Custom temporal clustering based on actual time gaps (not index).
    Groups detections that are within window_days of each other.
    """
    if len(timestamps) == 0:
        return pd.DataFrame()
    
    # Convert to datetime and sort
    ts = pd.to_datetime(timestamps).reset_index(drop=True)
    scores = scores.reset_index(drop=True)
    
    # Sort by time
    sorted_idx = ts.argsort()
    ts = ts[sorted_idx]
    scores = scores[sorted_idx]
    
    # Cluster based on time gaps
    clusters = []
    current_cluster = [0]
    
    for i in range(1, len(ts)):
        time_diff = (ts.iloc[i] - ts.iloc[i-1]).total_seconds() / 86400  # Convert to days
        
        if time_diff <= window_days:
            # Add to current cluster
            current_cluster.append(i)
        else:
            # Start new cluster
            clusters.append(current_cluster)
            current_cluster = [i]
    
    # Add the last cluster
    if current_cluster:
        clusters.append(current_cluster)
    
    # Select best detection from each cluster (highest score)
    result_indices = []
    cluster_ids = []
    
    for cluster_id, cluster_indices in enumerate(clusters):
        cluster_scores = scores.iloc[cluster_indices]
        best_idx = cluster_indices[cluster_scores.argmax()]
        result_indices.append(sorted_idx[best_idx])  # Map back to original index
        cluster_ids.append(cluster_id)
    
    return pd.DataFrame({
        'cluster_id': cluster_ids
    }, index=result_indices)


def find_threshold_for_target_recall(y_true, scores, target_recall=0.6) -> float:
    """Find threshold that achieves target recall."""
    from sklearn.metrics import recall_score
    
    # Sort scores to find thresholds
    unique_scores = np.unique(scores)
    # Sample up to 1000 thresholds
    if len(unique_scores) > 1000:
        thresholds = np.percentile(unique_scores, np.linspace(0, 100, 1000))
    else:
        thresholds = unique_scores
    
    best_threshold = thresholds[0]
    best_recall = 0
    
    for threshold in thresholds:
        preds = (scores >= threshold).astype(int)
        if preds.sum() == 0:  # Skip if no predictions
            continue
        recall = recall_score(y_true, preds)
        
        if recall >= target_recall:
            best_threshold = threshold
            best_recall = recall
            break
        elif recall > best_recall:
            best_threshold = threshold
            best_recall = recall
    
    if best_recall < target_recall:
        print(f"⚠️ Could not achieve target recall {target_recall:.2f}, best is {best_recall:.2f}")
    
    return best_threshold


def detect_with_custom_clustering(detector, X, timestamps, threshold=None, cluster_window_days=4.0):
    """
    Custom detection with time-based clustering.
    """
    # Get scores
    scores = detector.predict_scores_from_features(X)
    
    # Use custom threshold if provided
    if threshold is not None:
        thr = threshold
    else:
        thr = detector.fitted_threshold_
    
    # Get predictions
    preds = (scores >= thr).astype(int)
    ts = timestamps.iloc[-len(scores):].reset_index(drop=True)
    
    # Create detection dataframe
    det = pd.DataFrame({"timestamp": ts, "score": scores, "pred": preds})
    
    # Get positive detections
    pos = det[det["pred"] == 1].copy()
    
    if len(pos) == 0:
        det["cluster_id"] = np.nan
        return det
    
    # Apply temporal clustering
    clustered = temporal_cluster_by_time(
        pos["timestamp"],
        pos["score"],
        window_days=cluster_window_days
    )
    
    # Add cluster IDs
    pos = pos.reset_index(drop=True).loc[clustered.index]
    pos = pos.assign(cluster_id=clustered["cluster_id"].values)
    
    # Merge back
    out = det.merge(pos[["timestamp", "cluster_id"]], on="timestamp", how="left")
    return out


def load_satellite_config(name: str) -> dict:
    cfg_path = Path("configs") / f"{name}.json"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            if "hyperparams" in config and "scale_pos_weight" in config.get("hyperparams", {}):
                del config["hyperparams"]["scale_pos_weight"]
            return config
    return {}


def parse_cli():
    p = argparse.ArgumentParser(description="Run maneuver detection with fixed clustering.", add_help=True)
    p.add_argument("satellite_name", nargs="?", help="Satellite name, e.g., Fengyun-4A")
    p.add_argument("--target-recall", type=float, default=0.6, help="Target recall rate (default: 0.6)")
    p.add_argument("--cluster-window", type=float, default=4.0, 
                   help="Clustering window in days (default: 4.0)")
    return p.parse_args()


def plot_detection_timeline(det_df, X, ts, maneuver_times, label_window_min, scores, threshold, save_path):
    """Enhanced timeline plot with clear detection markers."""
    import matplotlib.pyplot as plt
    
    # Get cluster centers
    cluster_centers = (det_df.dropna(subset=['cluster_id'])
                           .sort_values(['cluster_id','score'], ascending=[True,False])
                           .drop_duplicates(subset=['cluster_id']))
    cluster_times = list(cluster_centers['timestamp'])
    
    # Match clusters to events
    half = pd.to_timedelta(label_window_min, unit="m")
    
    detected_events = []
    missed_events = []
    false_alarms = []
    
    for ev in maneuver_times:
        detected = any((pd.Timestamp(ct) >= ev - half) and (pd.Timestamp(ct) <= ev + half) for ct in cluster_times)
        if detected:
            detected_events.append(ev)
        else:
            missed_events.append(ev)
    
    for ct in cluster_times:
        matched = any((pd.Timestamp(ct) >= ev - half) and (pd.Timestamp(ct) <= ev + half) for ev in maneuver_times)
        if not matched:
            false_alarms.append(ct)
    
    # Calculate metrics
    tp = len(detected_events)
    fn = len(missed_events)
    fp = len(false_alarms)
    total = len(maneuver_times)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / total if total > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # Top plot: Full timeline
    ax1.plot(ts, scores, linewidth=0.8, label='Anomaly Score', color='blue', alpha=0.6)
    ax1.axhline(threshold, linestyle=':', alpha=0.7, label=f'Threshold ({threshold:.3f})', color='gray')
    
    # Mark events
    for ev in detected_events:
        ax1.axvline(ev, color='green', alpha=0.7, linewidth=2, linestyle='-')
    for ev in missed_events:
        ax1.axvline(ev, color='red', alpha=0.5, linewidth=1, linestyle='--')
    for ct in false_alarms:
        ax1.axvline(pd.Timestamp(ct), color='orange', alpha=0.5, linewidth=1, linestyle=':')
    
    ax1.set_ylabel('Anomaly Score')
    ax1.set_title(f'Full Timeline - Detected: {tp}/{total} (Recall: {recall:.1%}), False Alarms: {fp}, Precision: {precision:.1%}')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: Zoomed view of detections
    ax2.scatter(pd.to_datetime(cluster_centers['timestamp']), 
                cluster_centers['score'], 
                s=100, color='blue', zorder=5, label=f'Detected Clusters ({len(cluster_centers)})')
    
    # Show detection windows
    for ct in cluster_centers['timestamp']:
        ax2.axvspan(pd.Timestamp(ct) - pd.Timedelta(days=2), 
                   pd.Timestamp(ct) + pd.Timedelta(days=2), 
                   alpha=0.1, color='blue')
    
    # Show true event windows
    for ev in maneuver_times:
        ax2.axvspan(ev - half, ev + half, alpha=0.1, color='green')
        ax2.axvline(ev, color='green', alpha=0.3, linewidth=1)
    
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Detection Score')
    ax2.set_title('Detected Clusters vs True Events (green=truth, blue=detections)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "detected_events": detected_events,
        "missed_events": missed_events,
        "false_alarms": false_alarms
    }


def main():
    args = parse_cli()
    sat_name = args.satellite_name
    if not sat_name:
        sat_name = input("Satellite name: ").strip()
    
    target_recall = args.target_recall
    cluster_window = args.cluster_window
    
    print(f"🛰️ Processing: {sat_name}")
    print(f"🎯 Target recall: {target_recall:.0%}")
    print(f"📏 Cluster window: {cluster_window} days")

    # Load config
    cfg = load_satellite_config(sat_name)
    orbit = cfg.get("orbit", "auto")
    hyperparams = cfg.get("hyperparams", None)
    label_window_days = int(cfg.get("label_window_days", 2))
    label_window_min = int(cfg.get("label_window_min", label_window_days * 1440))
    
    # Prepare output dir
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("outputs", f"{sat_name.replace(' ', '_')}_{run_id}_fixed")
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    loader = SatelliteDataLoader(data_dir="data")
    tle_df, maneuver_times = loader.load_satellite_data(sat_name)
    tle_df = tle_df.sort_values("epoch").reset_index(drop=True)

    # Feature engineering
    eng = EnhancedSatelliteFeatureEngineer(
        target_column="mean_motion",
        additional_columns=["eccentricity", "inclination"],
        satellite_type=orbit
    ).fit(tle_df, satellite_name=sat_name)
    feat_df = eng.transform(tle_df)

    # Labels
    epoch_series = tle_df.loc[feat_df.index, "epoch"]
    y_series = build_labels_by_window(
        pd.DataFrame({"epoch": epoch_series}),
        "epoch", maneuver_times, label_window_min, "label"
    )
    feat_df["label"] = y_series.values

    # Prepare X/y/ts
    X = feat_df.drop(columns=["label"]).select_dtypes(include=[np.number])
    y = feat_df["label"].astype(int).values
    ts = tle_df.loc[feat_df.index, "epoch"]

    # Configure detector (with dummy temporal_window, we'll override it)
    det_cfg = DetectorConfig(
        time_col="epoch",
        label_col="label",
        orbit=orbit,
        threshold_mode="quantile",
        threshold_quantile=0.95,
        temporal_window=1,  # Will be overridden
        auto_tune=False,
        scale_pos_weight=10
    )
    det = ManeuverXGBDetector(det_cfg)
    
    if hyperparams:
        det.hyperparams = hyperparams

    # Train
    val_start = int(0.8 * len(X))
    orbit_for_params = orbit if orbit in {"GEO", "LEO"} else "GEO"
    det.fit_from_features(X, y, orbit=orbit_for_params, val_start=val_start)

    # Find optimal threshold
    val_scores = det.model.predict_proba(X.iloc[val_start:])[:, 1]
    val_y = y[val_start:]
    optimal_threshold = find_threshold_for_target_recall(val_y, val_scores, target_recall)
    
    print(f"📊 Original threshold: {det.fitted_threshold_:.4f}")
    print(f"📊 Optimal threshold for {target_recall:.0%} recall: {optimal_threshold:.4f}")

    # Detect with custom clustering
    all_scores = det.model.predict_proba(X)[:, 1]
    det_df = detect_with_custom_clustering(
        det, X, ts, 
        threshold=optimal_threshold,
        cluster_window_days=cluster_window
    )
    
    # Analyze results
    clusters = det_df.dropna(subset=['cluster_id'])
    n_clusters = len(clusters['cluster_id'].unique()) if len(clusters) > 0 else 0
    
    print(f"\n📈 Results:")
    print(f"   • Clusters detected: {n_clusters}")
    print(f"   • Positive predictions: {(det_df['pred']==1).sum()}")
    
    # Plot and save
    stats = plot_detection_timeline(
        det_df, X, ts, maneuver_times, label_window_min,
        all_scores, optimal_threshold,
        os.path.join(save_dir, 'timeline.png')
    )
    
    # Save results
    det_df.to_csv(os.path.join(save_dir, "detections.csv"), index=False)
    
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump({
            "threshold_used": float(optimal_threshold),
            "cluster_window_days": cluster_window,
            "target_recall": target_recall,
            "achieved_recall": stats["recall"],
            "precision": stats["precision"],
            "f1": stats["f1"],
            "tp": stats["tp"],
            "fp": stats["fp"],
            "fn": stats["fn"],
            "total_events": len(maneuver_times),
            "clusters_detected": n_clusters
        }, f, indent=2)
    
    print(f"\n✅ Detection complete!")
    print(f"   • Detected: {stats['tp']}/{len(maneuver_times)} events")
    print(f"   • Recall: {stats['recall']:.1%}")
    print(f"   • Precision: {stats['precision']:.1%}")
    print(f"   • F1 Score: {stats['f1']:.2f}")
    print(f"   • Results saved to: {save_dir}")


if __name__ == "__main__":
    main()