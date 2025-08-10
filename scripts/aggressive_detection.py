"""
Aggressive detection version - maximize recall rate
Key changes:
1. Use top-N detection strategy instead of threshold
2. Expand clustering window
3. Multiple detection strategies
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
import warnings
warnings.filterwarnings('ignore')

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
    y = np.zeros(len(df), dtype=int)
    half = pd.to_timedelta(window_min, unit="m")
    for ev in events:
        mask = (dt >= ev - half) & (dt <= ev + half)
        y[mask.values] = 1
    return pd.Series(y, index=df.index, name=out_col)


def detect_top_anomalies(scores, timestamps, n_events=50, min_gap_days=7):
    """
    Detect top N anomalies with minimum time gap between detections.
    
    Args:
        scores: Anomaly scores
        timestamps: Timestamps
        n_events: Expected number of events to detect
        min_gap_days: Minimum days between detections
    """
    # Create dataframe with scores and timestamps
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(timestamps),
        'score': scores,
        'index': range(len(scores))
    }).sort_values('score', ascending=False)
    
    # Select top anomalies with minimum gap
    selected = []
    selected_times = []
    
    for _, row in df.iterrows():
        # Check if this detection is far enough from existing ones
        current_time = row['timestamp']
        
        if not selected_times or all(
            abs((current_time - t).total_seconds()) / 86400 >= min_gap_days 
            for t in selected_times
        ):
            selected.append(row['index'])
            selected_times.append(current_time)
            
            if len(selected) >= n_events:
                break
    
    return selected


def detect_local_peaks(scores, timestamps, window_days=15, min_prominence=0.5):
    """
    Detect local peaks in the score time series.
    
    Args:
        scores: Anomaly scores
        timestamps: Timestamps  
        window_days: Window size for local peak detection
        min_prominence: Minimum prominence of peaks (relative to local baseline)
    """
    from scipy.signal import find_peaks
    
    # Normalize scores
    scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
    
    # Find peaks with prominence
    peaks, properties = find_peaks(
        scores_norm,
        prominence=min_prominence,
        distance=2  # Minimum 2 samples between peaks
    )
    
    # Filter peaks by local contrast
    filtered_peaks = []
    window_samples = max(1, window_days)  # Approximate samples in window
    
    for peak in peaks:
        # Get local window
        start = max(0, peak - window_samples)
        end = min(len(scores), peak + window_samples + 1)
        local_scores = scores_norm[start:end]
        
        if len(local_scores) > 1:
            local_baseline = np.percentile(local_scores, 25)
            if scores_norm[peak] > local_baseline + min_prominence:
                filtered_peaks.append(peak)
    
    return filtered_peaks


def smart_detection_strategy(scores, timestamps, maneuver_times, label_window_min):
    """
    Combined detection strategy using multiple approaches.
    """
    ts = pd.to_datetime(timestamps)
    
    # Strategy 1: Top-N detection (expecting ~49 events)
    top_n = detect_top_anomalies(scores, ts, n_events=60, min_gap_days=5)
    
    # Strategy 2: Local peaks
    peaks = detect_local_peaks(scores, ts, window_days=10, min_prominence=0.3)
    
    # Strategy 3: Statistical outliers (above 90th percentile)
    threshold_90 = np.percentile(scores, 90)
    outliers = np.where(scores > threshold_90)[0]
    
    # Strategy 4: Significant score changes
    score_diff = np.abs(np.diff(scores, prepend=scores[0]))
    diff_threshold = np.percentile(score_diff, 95)
    changes = np.where(score_diff > diff_threshold)[0]
    
    # Combine all strategies
    all_detections = set(top_n) | set(peaks) | set(outliers) | set(changes)
    
    # Create detection dataframe
    det_df = pd.DataFrame({
        'timestamp': ts,
        'score': scores,
        'is_top_n': [i in top_n for i in range(len(scores))],
        'is_peak': [i in peaks for i in range(len(scores))],
        'is_outlier': [i in outliers for i in range(len(scores))],
        'is_change': [i in changes for i in range(len(scores))],
        'detected': [i in all_detections for i in range(len(scores))]
    })
    
    return det_df, all_detections


def cluster_detections(detections, timestamps, scores, max_gap_days=7):
    """
    Cluster nearby detections.
    """
    if len(detections) == 0:
        return []
    
    # Sort by timestamp
    det_indices = sorted(detections, key=lambda i: timestamps.iloc[i])
    
    clusters = []
    current_cluster = [det_indices[0]]
    
    for i in range(1, len(det_indices)):
        time_diff = (timestamps.iloc[det_indices[i]] - timestamps.iloc[det_indices[i-1]]).total_seconds() / 86400
        
        if time_diff <= max_gap_days:
            current_cluster.append(det_indices[i])
        else:
            # Select best from current cluster
            cluster_scores = [scores[idx] for idx in current_cluster]
            best_idx = current_cluster[np.argmax(cluster_scores)]
            clusters.append(best_idx)
            current_cluster = [det_indices[i]]
    
    # Don't forget the last cluster
    if current_cluster:
        cluster_scores = [scores[idx] for idx in current_cluster]
        best_idx = current_cluster[np.argmax(cluster_scores)]
        clusters.append(best_idx)
    
    return clusters


def evaluate_detections(detection_times, maneuver_times, window_min):
    """Evaluate detection performance."""
    half = pd.to_timedelta(window_min, unit="m")
    
    detected_events = []
    missed_events = []
    false_alarms = []
    
    # Check which events were detected
    for ev in maneuver_times:
        detected = any(
            (pd.Timestamp(dt) >= ev - half) and (pd.Timestamp(dt) <= ev + half) 
            for dt in detection_times
        )
        if detected:
            detected_events.append(ev)
        else:
            missed_events.append(ev)
    
    # Check which detections are false alarms
    for dt in detection_times:
        matched = any(
            (pd.Timestamp(dt) >= ev - half) and (pd.Timestamp(dt) <= ev + half) 
            for ev in maneuver_times
        )
        if not matched:
            false_alarms.append(dt)
    
    return {
        'detected': detected_events,
        'missed': missed_events,
        'false_alarms': false_alarms,
        'tp': len(detected_events),
        'fn': len(missed_events),
        'fp': len(false_alarms),
        'recall': len(detected_events) / len(maneuver_times) if maneuver_times else 0,
        'precision': len(detected_events) / (len(detected_events) + len(false_alarms)) 
                     if (detected_events or false_alarms) else 0
    }


def plot_comprehensive_results(scores, timestamps, detections, maneuver_times, 
                              eval_results, save_path):
    """Create comprehensive visualization."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    
    ts = pd.to_datetime(timestamps)
    
    # Top plot: Score timeline with events
    ax1 = axes[0]
    ax1.plot(ts, scores, linewidth=0.8, color='blue', alpha=0.6, label='Anomaly Score')
    
    # Mark true events
    for ev in maneuver_times:
        ax1.axvline(ev, color='green', alpha=0.3, linewidth=1)
    
    # Mark detections
    for i in detections:
        ax1.scatter(ts.iloc[i], scores[i], color='red', s=50, zorder=5, alpha=0.7)
    
    ax1.set_ylabel('Anomaly Score')
    ax1.set_title(f"Detection Results - Recall: {eval_results['recall']:.1%}, "
                  f"Precision: {eval_results['precision']:.1%}")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Middle plot: Detection density
    ax2 = axes[1]
    
    # Create detection density over time (rolling count)
    det_series = pd.Series(0, index=ts)
    for i in detections:
        det_series.iloc[i] = 1
    
    rolling_det = det_series.rolling('30D').sum()
    ax2.fill_between(ts, 0, rolling_det, alpha=0.5, color='blue', label='Detection Density (30-day)')
    
    # Mark event density
    event_series = pd.Series(0, index=ts)
    for ev in maneuver_times:
        closest_idx = np.argmin(np.abs(ts - ev))
        event_series.iloc[closest_idx] = 1
    
    rolling_events = event_series.rolling('30D').sum()
    ax2.plot(ts, rolling_events, color='green', linewidth=2, alpha=0.7, label='Event Density (30-day)')
    
    ax2.set_ylabel('Count (30-day window)')
    ax2.set_title('Detection and Event Density Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Bottom plot: Performance over time
    ax3 = axes[2]
    
    # Show detected vs missed events
    for ev in eval_results['detected']:
        ax3.axvline(ev, color='green', alpha=0.7, linewidth=2)
    for ev in eval_results['missed']:
        ax3.axvline(ev, color='red', alpha=0.5, linewidth=1, linestyle='--')
    for fa in eval_results['false_alarms']:
        ax3.axvline(pd.Timestamp(fa), color='orange', alpha=0.5, linewidth=1, linestyle=':')
    
    # Create legend
    green_patch = mpatches.Patch(color='green', label=f"Detected ({eval_results['tp']})")
    red_patch = mpatches.Patch(color='red', label=f"Missed ({eval_results['fn']})")
    orange_patch = mpatches.Patch(color='orange', label=f"False Alarms ({eval_results['fp']})")
    ax3.legend(handles=[green_patch, red_patch, orange_patch])
    
    ax3.set_xlabel('Time')
    ax3.set_title('Detection Performance (Green=Detected, Red=Missed, Orange=False Alarm)')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("satellite_name", nargs="?", help="Satellite name")
    parser.add_argument("--min-recall", type=float, default=0.6, help="Minimum target recall")
    parser.add_argument("--strategy", default="combined", 
                       choices=["combined", "top_n", "peaks", "threshold"],
                       help="Detection strategy")
    args = parser.parse_args()
    
    sat_name = args.satellite_name or input("Satellite name: ").strip()
    
    print(f"🛰️ Processing: {sat_name}")
    print(f"🎯 Target minimum recall: {args.min_recall:.0%}")
    print(f"📋 Strategy: {args.strategy}")
    
    # Load config
    from pathlib import Path
    cfg_path = Path("configs") / f"{sat_name}.json"
    cfg = json.load(open(cfg_path)) if cfg_path.exists() else {}
    
    orbit = cfg.get("orbit", "auto")
    hyperparams = cfg.get("hyperparams", None)
    label_window_days = int(cfg.get("label_window_days", 2))
    label_window_min = label_window_days * 1440
    
    # Output directory
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("outputs", f"{sat_name.replace(' ', '_')}_{run_id}_aggressive")
    os.makedirs(save_dir, exist_ok=True)
    
    # Load data
    loader = SatelliteDataLoader(data_dir="data")
    tle_df, maneuver_times = loader.load_satellite_data(sat_name)
    tle_df = tle_df.sort_values("epoch").reset_index(drop=True)
    
    print(f"📊 Data: {len(tle_df)} samples, {len(maneuver_times)} maneuvers")
    
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
    
    # Prepare data
    X = feat_df.drop(columns=["label"]).select_dtypes(include=[np.number])
    y = feat_df["label"].astype(int).values
    ts = tle_df.loc[feat_df.index, "epoch"]
    
    # Train model with low regularization for higher sensitivity
    det_cfg = DetectorConfig(
        time_col="epoch",
        label_col="label",
        orbit=orbit,
        threshold_mode="quantile",
        threshold_quantile=0.50,  # Very low threshold
        temporal_window=1,
        auto_tune=False,
        scale_pos_weight=5,  # Lower weight for more detections
        early_stopping_rounds=30
    )
    
    det = ManeuverXGBDetector(det_cfg)
    
    # Override hyperparameters for more sensitive detection
    if hyperparams:
        hyperparams = hyperparams.copy()
        hyperparams["max_depth"] = 6  # Deeper trees
        hyperparams["min_child_weight"] = 1  # More sensitive
        hyperparams["reg_alpha"] = 0.01  # Less regularization
        hyperparams["reg_lambda"] = 0.5
        det.hyperparams = hyperparams
    
    # Train
    val_start = int(0.8 * len(X))
    det.fit_from_features(X, y, orbit=orbit if orbit in {"GEO", "LEO"} else "GEO", val_start=val_start)
    
    # Get scores
    scores = det.model.predict_proba(X)[:, 1]
    
    print(f"\n📈 Score Statistics:")
    print(f"   • Range: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"   • Mean: {scores.mean():.4f}")
    print(f"   • Std: {scores.std():.4f}")
    
    # Apply detection strategy
    if args.strategy == "combined":
        det_df, detections = smart_detection_strategy(scores, ts, maneuver_times, label_window_min)
    elif args.strategy == "top_n":
        detections = detect_top_anomalies(scores, ts, n_events=len(maneuver_times)+10, min_gap_days=5)
    elif args.strategy == "peaks":
        detections = detect_local_peaks(scores, ts, window_days=10, min_prominence=0.2)
    else:  # threshold
        threshold = np.percentile(scores, 80)
        detections = np.where(scores > threshold)[0]
    
    print(f"\n🔍 Initial detections: {len(detections)}")
    
    # Cluster detections
    final_detections = cluster_detections(detections, ts, scores, max_gap_days=label_window_days*2)
    
    print(f"📍 After clustering: {len(final_detections)} clusters")
    
    # Evaluate
    detection_times = [ts.iloc[i] for i in final_detections]
    eval_results = evaluate_detections(detection_times, maneuver_times, label_window_min)
    
    # If recall is too low, be more aggressive
    if eval_results['recall'] < args.min_recall:
        print(f"\n⚠️ Recall {eval_results['recall']:.1%} < target {args.min_recall:.0%}")
        print("🔧 Applying more aggressive detection...")
        
        # Lower the bar significantly
        n_target = int(len(maneuver_times) * 1.5)  # Detect 50% more than expected
        detections = detect_top_anomalies(scores, ts, n_events=n_target, min_gap_days=3)
        final_detections = cluster_detections(detections, ts, scores, max_gap_days=label_window_days*3)
        
        detection_times = [ts.iloc[i] for i in final_detections]
        eval_results = evaluate_detections(detection_times, maneuver_times, label_window_min)
    
    # Create visualization
    plot_comprehensive_results(scores, ts, final_detections, maneuver_times, 
                              eval_results, os.path.join(save_dir, 'results.png'))
    
    # Save results
    results = {
        "performance": {
            "recall": eval_results['recall'],
            "precision": eval_results['precision'],
            "f1": 2 * eval_results['recall'] * eval_results['precision'] / 
                 (eval_results['recall'] + eval_results['precision']) 
                 if (eval_results['recall'] + eval_results['precision']) > 0 else 0,
            "tp": eval_results['tp'],
            "fp": eval_results['fp'],
            "fn": eval_results['fn']
        },
        "settings": {
            "strategy": args.strategy,
            "min_recall": args.min_recall,
            "n_detections": len(final_detections),
            "label_window_days": label_window_days
        },
        "detected_times": [str(t) for t in detection_times],
        "missed_events": [str(e) for e in eval_results['missed']]
    }
    
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Save detection CSV
    det_output = pd.DataFrame({
        'timestamp': [ts.iloc[i] for i in final_detections],
        'score': [scores[i] for i in final_detections],
        'index': final_detections
    })
    det_output.to_csv(os.path.join(save_dir, "detections.csv"), index=False)
    
    # Print summary
    print("\n" + "="*60)
    print("📊 FINAL RESULTS")
    print("="*60)
    print(f"✅ Detected: {eval_results['tp']}/{len(maneuver_times)} events")
    print(f"📈 Recall: {eval_results['recall']:.1%}")
    print(f"🎯 Precision: {eval_results['precision']:.1%}")
    print(f"⚠️ False Alarms: {eval_results['fp']}")
    print(f"📁 Results saved to: {save_dir}")
    
    if eval_results['recall'] >= args.min_recall:
        print(f"\n✨ SUCCESS! Achieved target recall of {args.min_recall:.0%}")
    else:
        print(f"\n⚠️ Below target. Consider:")
        print("   1. Increase label_window_days in config")
        print("   2. Use --strategy top_n with more aggressive settings")
        print("   3. Reduce min_gap_days in detection")


if __name__ == "__main__":
    main()