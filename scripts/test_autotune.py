"""
Test script for auto-tuning with enhanced detector.
Tests both the original auto_tune function and integration with enhanced detector.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import json
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import SatelliteDataLoader
from src.data.feature_engineer import EnhancedSatelliteFeatureEngineer
from src.tuning.auto_tune import random_search_xgb
from src.models.enhanced_xgb_detector import (
    EnhancedManeuverXGBDetector,
    EnhancedDetectorConfig
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


def test_basic_autotune(X, y, orbit="GEO"):
    """Test the basic auto_tune functionality."""
    print("\n" + "="*60)
    print("📊 TESTING BASIC AUTO-TUNE")
    print("="*60)
    
    # Define parameter grid for GEO satellites
    if orbit == "GEO":
        param_grid = {
            "max_depth": [3, 4, 5, 6],
            "learning_rate": [0.02, 0.05, 0.1],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.7, 0.85, 1.0],
            "reg_alpha": [0.0, 0.01, 0.1],
            "reg_lambda": [0.5, 1.0, 2.0],
            "n_estimators": [400, 600, 800],
            "min_child_weight": [1, 2, 3],
            "scale_pos_weight": [5, 10, 15]
        }
    else:
        param_grid = {
            "max_depth": [3, 4, 5],
            "learning_rate": [0.05, 0.1, 0.2],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "reg_alpha": [0.0, 0.1],
            "reg_lambda": [1.0, 2.0],
            "n_estimators": [300, 500],
            "min_child_weight": [1, 3],
            "scale_pos_weight": [5, 10]
        }
    
    print(f"Parameter grid for {orbit}:")
    for key, values in param_grid.items():
        print(f"  • {key}: {values}")
    
    print(f"\nRunning random search with 20 iterations...")
    
    # Run auto-tuning
    best_params, best_metric, best_threshold = random_search_xgb(
        X, y,
        param_grid=param_grid,
        n_iter=20,
        test_size=0.2,
        early_stopping_rounds=30,
        random_state=42,
        maximize="f1"
    )
    
    print(f"\n✅ Auto-tuning complete!")
    print(f"Best F1 score: {best_metric:.4f}")
    print(f"Best threshold: {best_threshold:.4f}")
    print(f"Best parameters:")
    for key, value in best_params.items():
        print(f"  • {key}: {value}")
    
    return best_params, best_metric, best_threshold


def test_detector_with_autotune(X, y, timestamps, maneuver_times, orbit="GEO", label_window_min=2880):
    """Test enhanced detector with auto-tuning."""
    print("\n" + "="*60)
    print("📊 TESTING ENHANCED DETECTOR WITH AUTO-TUNE")
    print("="*60)
    
    # Create configuration with auto_tune enabled
    config = EnhancedDetectorConfig(
        detection_strategy="top_n",
        expected_events=len(maneuver_times),
        target_recall=0.8,
        orbit=orbit,
        auto_tune=True,  # Enable auto-tuning
        n_tune_iter=15,  # Number of tuning iterations
        scale_pos_weight=10
    )
    
    print("Configuration:")
    print(f"  • Detection strategy: {config.detection_strategy}")
    print(f"  • Auto-tune enabled: {config.auto_tune}")
    print(f"  • Tuning iterations: {config.n_tune_iter}")
    print(f"  • Target recall: {config.target_recall:.0%}")
    
    # Create detector
    detector = EnhancedManeuverXGBDetector(config)
    
    # Train with auto-tuning
    print("\n🎯 Training with auto-tuning...")
    val_start = int(0.8 * len(X))
    
    # For auto-tuning to work, we need to modify the detector to use random_search_xgb
    # Let's do it manually here for testing
    if config.auto_tune:
        param_grid = {
            "max_depth": [4, 5, 6, 7],
            "learning_rate": [0.03, 0.05, 0.08, 0.1],
            "subsample": [0.8, 0.9, 1.0],
            "colsample_bytree": [0.8, 0.9, 1.0],
            "reg_alpha": [0.0, 0.01, 0.05],
            "reg_lambda": [0.5, 1.0, 1.5],
            "n_estimators": [600, 800, 1000],
            "min_child_weight": [1, 2],
            "scale_pos_weight": [5, 8, 10]
        }
        
        print("Running hyperparameter search...")
        best_params, best_f1, best_thr = random_search_xgb(
            X, y,
            param_grid=param_grid,
            n_iter=config.n_tune_iter,
            early_stopping_rounds=config.early_stopping_rounds,
            random_state=config.random_state,
            maximize="f1"
        )
        
        print(f"  • Best F1 from tuning: {best_f1:.4f}")
        print(f"  • Best threshold: {best_thr:.4f}")
        
        # Set the best parameters
        detector.hyperparams = best_params
    
    # Train detector
    detector.fit_from_features(X, y, orbit=orbit, val_start=val_start)
    
    # Detect maneuvers
    print("\n🔍 Detecting maneuvers...")
    detections_df = detector.detect_from_features(X, timestamps)
    
    # Evaluate
    eval_results = detector.evaluate_detections(
        detections_df, maneuver_times, label_window_min
    )
    
    print(f"\n📈 Results with auto-tuned parameters:")
    print(f"  • Detected: {eval_results['tp']}/{len(maneuver_times)} events")
    print(f"  • Recall: {eval_results['recall']:.1%}")
    print(f"  • Precision: {eval_results['precision']:.1%}")
    print(f"  • F1 Score: {eval_results['f1']:.3f}")
    print(f"  • False Alarms: {eval_results['fp']}")
    
    return detector, eval_results


def compare_tuned_vs_default(X, y, timestamps, maneuver_times, orbit="GEO", label_window_min=2880):
    """Compare performance with and without auto-tuning."""
    print("\n" + "="*60)
    print("📊 COMPARING DEFAULT VS AUTO-TUNED PARAMETERS")
    print("="*60)
    
    results = {}
    
    # Test 1: Default parameters
    print("\n1️⃣ Testing with DEFAULT parameters...")
    config_default = EnhancedDetectorConfig(
        detection_strategy="top_n",
        expected_events=len(maneuver_times),
        target_recall=0.8,
        orbit=orbit,
        auto_tune=False  # No auto-tuning
    )
    
    detector_default = EnhancedManeuverXGBDetector(config_default)
    val_start = int(0.8 * len(X))
    detector_default.fit_from_features(X, y, orbit=orbit, val_start=val_start)
    
    detections_default = detector_default.detect_from_features(X, timestamps)
    eval_default = detector_default.evaluate_detections(
        detections_default, maneuver_times, label_window_min
    )
    
    results['default'] = {
        'recall': eval_default['recall'],
        'precision': eval_default['precision'],
        'f1': eval_default['f1']
    }
    
    print(f"  Default results: Recall={eval_default['recall']:.1%}, "
          f"Precision={eval_default['precision']:.1%}, F1={eval_default['f1']:.3f}")
    
    # Test 2: Auto-tuned parameters
    print("\n2️⃣ Testing with AUTO-TUNED parameters...")
    
    # Get best parameters through tuning
    param_grid = {
        "max_depth": [4, 5, 6],
        "learning_rate": [0.03, 0.05, 0.08],
        "subsample": [0.85, 0.95],
        "colsample_bytree": [0.85, 0.95],
        "reg_alpha": [0.0, 0.01],
        "reg_lambda": [0.5, 1.0],
        "n_estimators": [600, 800],
        "min_child_weight": [1, 2],
        "scale_pos_weight": [5, 8, 10]
    }
    
    best_params, _, _ = random_search_xgb(
        X, y,
        param_grid=param_grid,
        n_iter=10,
        early_stopping_rounds=30,
        maximize="f1"
    )
    
    config_tuned = EnhancedDetectorConfig(
        detection_strategy="top_n",
        expected_events=len(maneuver_times),
        target_recall=0.8,
        orbit=orbit,
        auto_tune=False
    )
    
    detector_tuned = EnhancedManeuverXGBDetector(config_tuned)
    detector_tuned.hyperparams = best_params
    detector_tuned.fit_from_features(X, y, orbit=orbit, val_start=val_start)
    
    detections_tuned = detector_tuned.detect_from_features(X, timestamps)
    eval_tuned = detector_tuned.evaluate_detections(
        detections_tuned, maneuver_times, label_window_min
    )
    
    results['tuned'] = {
        'recall': eval_tuned['recall'],
        'precision': eval_tuned['precision'],
        'f1': eval_tuned['f1']
    }
    
    print(f"  Tuned results: Recall={eval_tuned['recall']:.1%}, "
          f"Precision={eval_tuned['precision']:.1%}, F1={eval_tuned['f1']:.3f}")
    
    # Comparison
    print("\n📊 COMPARISON SUMMARY:")
    print("="*40)
    print(f"{'Metric':<15} {'Default':<12} {'Tuned':<12} {'Improvement':<12}")
    print("-"*40)
    
    for metric in ['recall', 'precision', 'f1']:
        default_val = results['default'][metric]
        tuned_val = results['tuned'][metric]
        improvement = (tuned_val - default_val) / default_val * 100 if default_val > 0 else 0
        
        print(f"{metric.capitalize():<15} {default_val:<12.3f} {tuned_val:<12.3f} "
              f"{improvement:+.1f}%")
    
    return results


def main():
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="Test auto-tuning functionality")
    parser.add_argument("--satellite", default="Fengyun-4A", help="Satellite name")
    parser.add_argument("--quick", action="store_true", help="Quick test with fewer iterations")
    args = parser.parse_args()
    
    print(f"🛰️ Testing auto-tune with {args.satellite}")
    
    # Load data
    loader = SatelliteDataLoader(data_dir="data")
    tle_df, maneuver_times = loader.load_satellite_data(args.satellite)
    tle_df = tle_df.sort_values("epoch").reset_index(drop=True)
    
    print(f"📊 Loaded {len(tle_df)} TLE records and {len(maneuver_times)} maneuvers")
    
    # Feature engineering
    eng = EnhancedSatelliteFeatureEngineer(
        target_column="mean_motion",
        additional_columns=["eccentricity", "inclination"],
        satellite_type="GEO"
    ).fit(tle_df, satellite_name=args.satellite)
    feat_df = eng.transform(tle_df)
    
    # Create labels
    label_window_min = 2 * 1440  # 2 days in minutes
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
    
    print(f"\n📊 Data shape: X={X.shape}, y={y.shape}")
    print(f"   Positive samples: {y.sum()} ({100*y.sum()/len(y):.1f}%)")
    
    # Run tests
    if not args.quick:
        # Test 1: Basic auto-tune
        best_params, best_metric, best_threshold = test_basic_autotune(X, y, orbit="GEO")
        
        # Test 2: Detector with auto-tune
        detector, eval_results = test_detector_with_autotune(
            X, y, timestamps, maneuver_times, orbit="GEO", label_window_min=label_window_min
        )
        
        # Test 3: Compare tuned vs default
        comparison_results = compare_tuned_vs_default(
            X, y, timestamps, maneuver_times, orbit="GEO", label_window_min=label_window_min
        )
    else:
        print("\n⚡ Running quick test with fewer iterations...")
        # Quick test with minimal iterations
        param_grid = {
            "max_depth": [4, 5],
            "learning_rate": [0.05],
            "subsample": [0.9],
            "colsample_bytree": [0.9],
            "reg_alpha": [0.01],
            "reg_lambda": [1.0],
            "n_estimators": [600],
            "min_child_weight": [1],
            "scale_pos_weight": [8]
        }
        
        best_params, best_metric, best_threshold = random_search_xgb(
            X, y,
            param_grid=param_grid,
            n_iter=3,
            early_stopping_rounds=20,
            maximize="f1"
        )
        
        print(f"\n✅ Quick test complete!")
        print(f"   Best F1: {best_metric:.4f}")
        print(f"   Best threshold: {best_threshold:.4f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"autotune_test_{args.satellite}_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            "satellite": args.satellite,
            "timestamp": timestamp,
            "data_stats": {
                "n_samples": len(X),
                "n_features": X.shape[1],
                "n_positive": int(y.sum()),
                "n_maneuvers": len(maneuver_times)
            },
            "best_params": best_params if 'best_params' in locals() else None,
            "best_metric": best_metric if 'best_metric' in locals() else None,
            "best_threshold": best_threshold if 'best_threshold' in locals() else None
        }, f, indent=2, default=str)
    
    print(f"\n📁 Results saved to {results_file}")
    print("\n✅ All tests completed successfully!")


if __name__ == "__main__":
    main()