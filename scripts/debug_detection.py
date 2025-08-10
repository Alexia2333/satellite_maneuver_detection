"""
Debug version to diagnose why recall is 0%
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple
from sklearn.metrics import precision_recall_curve, recall_score, precision_score

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


def analyze_scores_and_labels(scores, y, maneuver_times, timestamps, label_window_min):
    """Detailed analysis of scores and labels to diagnose the problem."""
    print("\n" + "="*60)
    print("📊 DIAGNOSTIC ANALYSIS")
    print("="*60)
    
    # Basic statistics
    print(f"\n1️⃣ Score Statistics:")
    print(f"   • Min score: {scores.min():.6f}")
    print(f"   • Max score: {scores.max():.6f}")
    print(f"   • Mean score: {scores.mean():.6f}")
    print(f"   • Std score: {scores.std():.6f}")
    print(f"   • Percentiles:")
    for p in [10, 25, 50, 75, 90, 95, 98, 99]:
        print(f"     - {p}%: {np.percentile(scores, p):.6f}")
    
    # Label statistics
    print(f"\n2️⃣ Label Statistics:")
    print(f"   • Total samples: {len(y)}")
    print(f"   • Positive samples: {y.sum()} ({100*y.sum()/len(y):.1f}%)")
    print(f"   • Negative samples: {len(y) - y.sum()} ({100*(1-y.sum()/len(y)):.1f}%)")
    
    # Score distribution by label
    pos_scores = scores[y == 1]
    neg_scores = scores[y == 0]
    
    print(f"\n3️⃣ Score Distribution by Label:")
    print(f"   Positive samples (y=1):")
    print(f"   • Count: {len(pos_scores)}")
    if len(pos_scores) > 0:
        print(f"   • Mean: {pos_scores.mean():.6f}")
        print(f"   • Max: {pos_scores.max():.6f}")
        print(f"   • 90th percentile: {np.percentile(pos_scores, 90):.6f}")
    
    print(f"   Negative samples (y=0):")
    print(f"   • Count: {len(neg_scores)}")
    if len(neg_scores) > 0:
        print(f"   • Mean: {neg_scores.mean():.6f}")
        print(f"   • 99th percentile: {np.percentile(neg_scores, 99):.6f}")
    
    # Check score separation
    if len(pos_scores) > 0 and len(neg_scores) > 0:
        overlap = np.sum(pos_scores[:, None] < neg_scores[None, :]) / (len(pos_scores) * len(neg_scores))
        print(f"\n4️⃣ Score Separation:")
        print(f"   • Overlap ratio: {overlap:.2%}")
        print(f"   • Best possible threshold: between {neg_scores.max():.6f} and {pos_scores.min():.6f}")
    
    # Analyze scores around events
    print(f"\n5️⃣ Scores Around Maneuver Events:")
    half = pd.to_timedelta(label_window_min, unit="m")
    ts = pd.to_datetime(timestamps)
    
    for i, ev in enumerate(maneuver_times[:5]):  # Check first 5 events
        # Find samples within window
        mask = (ts >= ev - half) & (ts <= ev + half)
        event_scores = scores[mask]
        event_labels = y[mask]
        
        print(f"\n   Event {i+1} at {ev}:")
        print(f"   • Samples in window: {mask.sum()}")
        if len(event_scores) > 0:
            print(f"   • Max score in window: {event_scores.max():.6f}")
            print(f"   • Mean score in window: {event_scores.mean():.6f}")
            print(f"   • Labels in window: {event_labels.sum()} positive")
    
    return pos_scores, neg_scores


def find_optimal_threshold(y_true, scores, target_recall=0.6):
    """Find threshold with detailed debugging."""
    from sklearn.metrics import precision_recall_curve, recall_score
    
    print(f"\n6️⃣ Finding Optimal Threshold for {target_recall:.0%} Recall:")
    
    # Get precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    
    # Find threshold for target recall
    idx = np.where(recall >= target_recall)[0]
    if len(idx) > 0:
        optimal_idx = idx[-1]  # Last index where recall >= target
        optimal_threshold = thresholds[optimal_idx]
        achieved_recall = recall[optimal_idx]
        achieved_precision = precision[optimal_idx]
        
        print(f"   • Optimal threshold: {optimal_threshold:.6f}")
        print(f"   • Achieved recall: {achieved_recall:.2%}")
        print(f"   • Achieved precision: {achieved_precision:.2%}")
    else:
        # Can't achieve target recall
        best_recall_idx = np.argmax(recall)
        optimal_threshold = thresholds[best_recall_idx] if best_recall_idx < len(thresholds) else scores.min()
        print(f"   • ⚠️ Cannot achieve {target_recall:.0%} recall")
        print(f"   • Best possible recall: {recall[best_recall_idx]:.2%}")
        print(f"   • Using threshold: {optimal_threshold:.6f}")
    
    # Test different thresholds
    print(f"\n   Testing different thresholds:")
    test_thresholds = [scores.min(), np.percentile(scores, 1), np.percentile(scores, 10), 
                       np.percentile(scores, 50), np.percentile(scores, 90), np.percentile(scores, 99)]
    
    for thr in test_thresholds:
        preds = (scores >= thr).astype(int)
        r = recall_score(y_true, preds)
        p = precision_score(y_true, preds) if preds.sum() > 0 else 0
        print(f"   • Threshold {thr:.6f}: Recall={r:.2%}, Precision={p:.2%}, Detections={preds.sum()}")
    
    return optimal_threshold


def load_satellite_config(name: str) -> dict:
    cfg_path = Path("configs") / f"{name}.json"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            if "hyperparams" in config and "scale_pos_weight" in config.get("hyperparams", {}):
                del config["hyperparams"]["scale_pos_weight"]
            return config
    return {}


def main():
    # Simple argument parsing
    import sys
    sat_name = sys.argv[1] if len(sys.argv) > 1 else input("Satellite name: ").strip()
    
    # Load config
    cfg = load_satellite_config(sat_name)
    orbit = cfg.get("orbit", "auto")
    hyperparams = cfg.get("hyperparams", None)
    label_window_days = int(cfg.get("label_window_days", 2)) 
    label_window_min = int(cfg.get("label_window_min", label_window_days * 1440))
    
    print(f"🛰️ Processing: {sat_name}")
    print(f"📍 Label window: ±{label_window_min/1440:.1f} days")

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
        "epoch",
        maneuver_times,
        label_window_min,
        "label"
    )
    feat_df["label"] = y_series.values

    # Prepare X/y
    X = feat_df.drop(columns=["label"]).select_dtypes(include=[np.number])
    y = feat_df["label"].astype(int).values
    ts = tle_df.loc[feat_df.index, "epoch"]

    # Train with very low threshold
    det_cfg = DetectorConfig(
        time_col="epoch",
        label_col="label",
        orbit=orbit,
        threshold_mode="quantile",
        threshold_quantile=0.90,  # Start with low quantile
        temporal_window=1,
        auto_tune=False,
        scale_pos_weight=10  # Lower weight to increase recall
    )
    det = ManeuverXGBDetector(det_cfg)
    
    if hyperparams:
        # Modify hyperparams for higher recall
        hyperparams["scale_pos_weight"] = 10
        det.hyperparams = hyperparams

    # Train
    val_start = int(0.8 * len(X))
    orbit_for_params = orbit if orbit in {"GEO", "LEO"} else "GEO"
    det.fit_from_features(X, y, orbit=orbit_for_params, val_start=val_start)

    # Get all scores
    all_scores = det.model.predict_proba(X)[:, 1]
    val_scores = all_scores[val_start:]
    val_y = y[val_start:]
    
    # Detailed analysis
    pos_scores, neg_scores = analyze_scores_and_labels(all_scores, y, maneuver_times, ts, label_window_min)
    
    # Find optimal threshold
    optimal_threshold = find_optimal_threshold(val_y, val_scores, target_recall=0.6)
    
    # Apply new threshold and test
    print(f"\n7️⃣ Testing with new threshold {optimal_threshold:.6f}:")
    det.fitted_threshold_ = optimal_threshold
    
    # Detect and analyze
    det_df = det.detect_from_features(X, timestamps=ts)
    clusters = det_df.dropna(subset=['cluster_id'])
    
    print(f"   • Detections: {len(clusters['cluster_id'].unique())} clusters")
    print(f"   • Detection rate: {len(det_df[det_df['pred']==1])} positive predictions")
    
    # Check actual detection performance
    half = pd.to_timedelta(label_window_min, unit="m")
    detected = 0
    for ev in maneuver_times:
        if len(clusters) > 0:
            has_detection = any((clusters['timestamp'] >= ev - half).values & 
                               (clusters['timestamp'] <= ev + half).values)
            if has_detection:
                detected += 1
    
    print(f"   • Events detected: {detected}/{len(maneuver_times)} = {100*detected/len(maneuver_times):.1f}%")
    
    # Plot score distribution
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Score histogram
    plt.subplot(2, 2, 1)
    plt.hist(neg_scores, bins=50, alpha=0.5, label='Negative', color='blue')
    if len(pos_scores) > 0:
        plt.hist(pos_scores, bins=50, alpha=0.5, label='Positive', color='red')
    plt.axvline(det.fitted_threshold_, color='green', linestyle='--', label=f'Threshold={det.fitted_threshold_:.4f}')
    plt.xlabel('Score')
    plt.ylabel('Count')
    plt.title('Score Distribution by Label')
    plt.legend()
    plt.yscale('log')
    
    # Subplot 2: Score over time with events
    plt.subplot(2, 2, 2)
    plt.plot(ts, all_scores, linewidth=0.5, alpha=0.7)
    plt.axhline(det.fitted_threshold_, color='red', linestyle='--', alpha=0.5)
    for ev in maneuver_times:
        plt.axvline(ev, color='green', alpha=0.3)
    plt.xlabel('Time')
    plt.ylabel('Score')
    plt.title('Scores Over Time (green lines = events)')
    
    # Subplot 3: Precision-Recall curve
    from sklearn.metrics import precision_recall_curve, precision_score
    plt.subplot(2, 2, 3)
    precision, recall, thresholds = precision_recall_curve(val_y, val_scores)
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Scores around first few events
    plt.subplot(2, 2, 4)
    for i, ev in enumerate(maneuver_times[:5]):
        window_mask = (pd.to_datetime(ts) >= ev - pd.Timedelta(days=5)) & \
                     (pd.to_datetime(ts) <= ev + pd.Timedelta(days=5))
        if window_mask.sum() > 0:
            window_ts = ts[window_mask]
            window_scores = all_scores[window_mask]
            plt.plot(window_ts, window_scores, label=f'Event {i+1}', alpha=0.7)
            plt.axvline(ev, linestyle='--', alpha=0.5)
    plt.xlabel('Time')
    plt.ylabel('Score')
    plt.title('Scores Around First 5 Events (±5 days)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('debug_analysis.png', dpi=150)
    plt.show()
    
    print("\n✅ Debug analysis complete. Check 'debug_analysis.png' for visualizations.")


if __name__ == "__main__":
    main()