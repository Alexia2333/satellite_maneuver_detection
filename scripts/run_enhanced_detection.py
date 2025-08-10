"""
Example script for using the Enhanced XGBoost Detector.
Demonstrates how to use the new detector with existing infrastructure.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import SatelliteDataLoader
from src.data.feature_engineer import EnhancedSatelliteFeatureEngineer
from src.models.enhanced_xgb_detector import (
    EnhancedManeuverXGBDetector, 
    EnhancedDetectorConfig,
    create_detector_from_config
)


def build_labels_by_window(df: pd.DataFrame, time_col: str, events, window_min: int, out_col: str) -> pd.Series:
    """Binary labels: 1 within +/- window_min minutes of any event, else 0."""
    dt = pd.to_datetime(df[time_col], errors="coerce")
    y = np.zeros(len(df), dtype=int)
    half = pd.to_timedelta(window_min, unit="m")
    for ev in events:
        mask = (dt >= ev - half) & (dt <= ev + half)
        y[mask.values] = 1
    return pd.Series(y, index=df.index, name=out_col)


def plot_detection_results(detector, X, timestamps, maneuver_times, eval_results, save_path):
    """Visualize detection results."""
    scores = detector.predict_scores_from_features(X)
    det_df = detector.detect_from_features(X, timestamps)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # Top plot: Scores and detections
    ax1.plot(timestamps, scores, linewidth=0.8, alpha=0.7, label='Anomaly Score')
    
    # Mark detected clusters
    clusters = det_df.dropna(subset=['cluster_id'])
    for cluster_id in clusters['cluster_id'].unique():
        cluster_data = clusters[clusters['cluster_id'] == cluster_id]
        best_idx = cluster_data['score'].idxmax()
        ax1.scatter(cluster_data.loc[best_idx, 'timestamp'], 
                   cluster_data.loc[best_idx, 'score'],
                   color='red', s=100, zorder=5)
    
    # Mark true events
    for ev in maneuver_times:
        ax1.axvline(ev, color='green', alpha=0.3, linewidth=1)
    
    ax1.set_ylabel('Anomaly Score')
    ax1.set_title(f'Detection Results - Strategy: {detector.config.detection_strategy} | '
                  f'Recall: {eval_results["recall"]:.1%} | Precision: {eval_results["precision"]:.1%}')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Bottom plot: Detection timeline
    ax2.eventplot([eval_results['detected_events']], colors='green', linewidths=2, 
                  label=f'Detected ({eval_results["tp"]})')
    ax2.eventplot([eval_results['missed_events']], colors='red', linewidths=2,
                  label=f'Missed ({eval_results["fn"]})')
    if eval_results['false_alarms']:
        ax2.eventplot([eval_results['false_alarms']], colors='orange', linewidths=2,
                      label=f'False Alarms ({eval_results["fp"]})')
    
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Events')
    ax2.set_title('Detection Performance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run enhanced maneuver detection")
    parser.add_argument("satellite_name", help="Satellite name (e.g., Fengyun-4A)")
    parser.add_argument("--strategy", default="top_n", 
                       choices=["threshold", "top_n", "peaks", "combined"],
                       help="Detection strategy (default: top_n)")
    parser.add_argument("--expected-events", type=int, default=None,
                       help="Expected number of maneuvers")
    parser.add_argument("--target-recall", type=float, default=0.6,
                       help="Target recall rate (default: 0.6)")
    parser.add_argument("--config", help="Path to configuration JSON file")
    parser.add_argument("--save-model", help="Path to save the trained model")
    parser.add_argument("--load-model", help="Path to load a pre-trained model")
    
    args = parser.parse_args()
    
    print(f"🛰️ Processing: {args.satellite_name}")
    print(f"📋 Detection strategy: {args.strategy}")
    
    # Load satellite configuration
    sat_config_path = Path("configs") / f"{args.satellite_name}.json"
    sat_config = {}
    if sat_config_path.exists():
        with open(sat_config_path, 'r') as f:
            sat_config = json.load(f)
    label_window_days = sat_config.get("label_window_days", 2)
    # Create detector configuration
    if args.config:
        # Load from specified config file
        detector = create_detector_from_config(args.config)
    else:
        # Create with parameters
        
        config = EnhancedDetectorConfig(
            detection_strategy=args.strategy,
            expected_events=args.expected_events,
            target_recall=args.target_recall,
            orbit=sat_config.get("orbit", "auto"),
            scale_pos_weight=5,  # Lower for better recall
            sensitive_hyperparams={
                "max_depth": 6,
                "min_child_weight": 1,
                "reg_alpha": 0.01,
                "reg_lambda": 0.5
            }
        )
        detector = EnhancedManeuverXGBDetector(config)
    
    # Load data
    loader = SatelliteDataLoader(data_dir="data")
    tle_df, maneuver_times = loader.load_satellite_data(args.satellite_name)
    tle_df = tle_df.sort_values("epoch").reset_index(drop=True)
    
    print(f"📊 Loaded {len(tle_df)} TLE records and {len(maneuver_times)} maneuvers")
    
    # Update expected events if not specified
    if not detector.config.expected_events:
        detector.config.expected_events = len(maneuver_times)
    
    # Feature engineering
    eng = EnhancedSatelliteFeatureEngineer(
        target_column="mean_motion",
        additional_columns=["eccentricity", "inclination"],
        satellite_type=detector.config.orbit
    ).fit(tle_df, satellite_name=args.satellite_name)
    feat_df = eng.transform(tle_df)
    
    # Create labels
    label_window_min = label_window_days * 1440
    epoch_series = tle_df.loc[feat_df.index, "epoch"]
    y_series = build_labels_by_window(
        pd.DataFrame({"epoch": epoch_series}),
        "epoch", maneuver_times, label_window_min, "label"
    )
    feat_df["label"] = y_series.values
    
    # Prepare data
    X = feat_df.drop(columns=["label"]).select_dtypes(include=[np.number])
    y = feat_df["label"].astype(int).values
    timestamps = tle_df.loc[feat_df.index, "epoch"]
    
    # Train or load model
    if args.load_model:
        print(f"📂 Loading model from {args.load_model}")
        detector.load_model(args.load_model)
    else:
        print("🎯 Training model...")
        val_start = int(0.8 * len(X))
        detector.fit_from_features(
            X, y, 
            orbit=detector.config.orbit if detector.config.orbit != "auto" else "GEO",
            val_start=val_start
        )
        
        if args.save_model:
            detector.save_model(args.save_model)
    
    # Detect maneuvers
    print("🔍 Detecting maneuvers...")
    detections_df = detector.detect_from_features(X, timestamps)
    
    # Evaluate performance
    eval_results = detector.evaluate_detections(
        detections_df, maneuver_times, label_window_min
    )
    
    # Create output directory
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path("outputs") / f"{args.satellite_name.replace(' ', '_')}_{args.strategy}_{run_id}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    detections_df.to_csv(save_dir / "detections.csv", index=False)
    
    with open(save_dir / "evaluation.json", 'w') as f:
        json.dump({
            "performance": {
                "recall": eval_results['recall'],
                "precision": eval_results['precision'],
                "f1": eval_results['f1'],
                "true_positives": eval_results['tp'],
                "false_positives": eval_results['fp'],
                "false_negatives": eval_results['fn']
            },
            "configuration": {
                "strategy": detector.config.detection_strategy,
                "expected_events": detector.config.expected_events,
                "target_recall": detector.config.target_recall,
                "min_gap_days": detector.config.min_gap_days,
                "cluster_window_days": detector.config.cluster_window_days
            },
            "detected_events": [str(ev) for ev in eval_results['detected_events']],
            "missed_events": [str(ev) for ev in eval_results['missed_events']],
            "false_alarms": [str(fa) for fa in eval_results['false_alarms']]
        }, f, indent=2)
    
    # Plot results
    plot_detection_results(
        detector, X, timestamps, maneuver_times, eval_results,
        save_dir / "detection_plot.png"
    )
    
    # Print summary
    print("\n" + "="*60)
    print("📊 DETECTION RESULTS")
    print("="*60)
    print(f"Strategy: {detector.config.detection_strategy}")
    print(f"Detected: {eval_results['tp']}/{len(maneuver_times)} events")
    print(f"Recall: {eval_results['recall']:.1%}")
    print(f"Precision: {eval_results['precision']:.1%}")
    print(f"F1 Score: {eval_results['f1']:.2f}")
    print(f"False Alarms: {eval_results['fp']}")
    print(f"\n📁 Results saved to: {save_dir}")
    
    # Print recommendations if recall is low
    if eval_results['recall'] < detector.config.target_recall:
        print(f"\n⚠️ Recall ({eval_results['recall']:.1%}) is below target ({detector.config.target_recall:.0%})")
        print("Recommendations:")
        print("  1. Try --strategy top_n for better recall")
        print("  2. Increase --expected-events parameter")
        print("  3. Decrease min_gap_days in configuration")
    else:
        print(f"\n✅ Target recall achieved!")


if __name__ == "__main__":
    main()