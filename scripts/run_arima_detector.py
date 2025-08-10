# scripts/run_arima_detector.py
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import SatelliteDataLoader
from src.models.arima_detector import EnhancedARIMADetector, DetectorConfig
from src.tuning.auto_tune_arima import find_best_arima_order
from src.tuning.tune_arima_threshold import find_best_threshold_factor
from src.utils.metrics import evaluate_detection

def prepare_time_series(tle_df: pd.DataFrame, element: str) -> pd.Series:
    """从原始TLE数据准备每日中位数时间序列。"""
    ts_data = tle_df[['epoch', element]].set_index('epoch')
    ts_data = ts_data.resample('D').median().interpolate(method='time').dropna()
    return ts_data[element]

def visualize_results(results_df, maneuvers, title, output_path):
    """创建并保存结果的可视化图表。"""
    print("📊 Creating visualization...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), sharex=True)
    ax1.plot(results_df.index, results_df['actual'], 'k-', label='Actual')
    if 'forecast' in results_df.columns:
        ax1.plot(results_df.index, results_df['forecast'], 'b--', label='Forecast', alpha=0.7)
    detections = results_df[results_df['is_anomaly']]
    ax1.scatter(detections.index, detections['actual'], color='red', s=80, marker='x', label='Detection', zorder=5)
    if maneuvers:
        ax1.axvline(maneuvers[0], color='green', linestyle=':', label='True Maneuver')
        for m in maneuvers[1:]: ax1.axvline(m, color='green', linestyle=':')
    ax1.set_title(title)
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True, alpha=0.5)

    ax2.plot(results_df.index, results_df['residual_scaled'], 'purple', label='Residual (Scaled)')
    ax2.plot(results_df.index, results_df['threshold_scaled'], 'r--', label='Threshold (Scaled)')
    ax2.scatter(detections.index, detections['residual_scaled'], color='red', s=80, marker='x', zorder=5)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Scaled Score')
    ax2.legend()
    ax2.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Enhanced ARIMA Maneuver Detection Pipeline")
    parser.add_argument("satellite", help="Satellite name")
    parser.add_argument("--element", default="mean_motion", help="Orbital element to analyze")
    parser.add_argument("--auto-tune", action="store_true", help="Enable auto-tuning of ARIMA order")
    parser.add_argument("--optimize-threshold", action="store_true", help="Enable threshold optimization")
    args = parser.parse_args()

    output_dir = Path("outputs") / f"{args.satellite.replace(' ', '_')}_Final_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}\n🛰️ Final Pipeline for: {args.satellite}\n{'='*60}")

    loader = SatelliteDataLoader(data_dir="data")
    tle_data, maneuver_times = loader.load_satellite_data(args.satellite)
    ts_data = prepare_time_series(tle_data, args.element)

    n = len(ts_data)
    train_end, val_end = int(n * 0.6), int(n * 0.8)
    train_data, val_data, test_data = ts_data.iloc[:train_end], ts_data.iloc[train_end:val_end], ts_data.iloc[val_end:]
    print(f"Data split: Train ({len(train_data)}), Validation ({len(val_data)}), Test ({len(test_data)})")

    order = (5, 1, 1) # Default
    if args.auto_tune:
        order = find_best_arima_order(train_data, args.satellite, args.element)
    
    config = DetectorConfig(order=order)
    detector = EnhancedARIMADetector(config=config)
    detector.fit(train_data)
    
    if args.optimize_threshold and not val_data.empty:
        val_maneuvers = [m for m in maneuver_times if val_data.index.min() <= m <= val_data.index.max()]
        if val_maneuvers:
            best_factor = find_best_threshold_factor(detector, val_data, val_maneuvers)
            detector.cfg.threshold_factor = best_factor

    print(f"\n📈 Running final detection on test set with factor={detector.cfg.threshold_factor:.2f}")
    results_df = detector.detect(test_data)
    
    detected_anomalies = results_df[results_df['is_anomaly']].index.tolist()
    test_maneuvers = [m for m in maneuver_times if not test_data.empty and test_data.index.min() <= m <= test_data.index.max()]
    metrics, _ = evaluate_detection(detected_anomalies, test_maneuvers, timedelta(days=2))
    
    print(f"\n📊 FINAL PERFORMANCE:")
    print(f"  F1: {metrics['f1']:.3f}, P: {metrics['precision']:.2%}, R: {metrics['recall']:.2%}")

    # --- THIS IS THE CORRECTED SECTION ---
    # 1. Create the title string first
    plot_title = f"Maneuver Detection for {args.satellite} - {args.element}"
    plot_path = output_dir / "detection_plot.png"
    
    # 2. Call the function with the correct 4 arguments
    visualize_results(results_df, test_maneuvers, plot_title, plot_path)
    # --- END OF CORRECTION ---
    
    report = {"satellite_name": args.satellite, "element": args.element, "best_order": list(order), "final_threshold_factor": detector.cfg.threshold_factor, "performance": metrics}
    with open(output_dir / 'summary_report.json', 'w') as f: json.dump(report, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to {output_dir}")

if __name__ == "__main__":
    main()