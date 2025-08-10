# scripts/run_fusion_detector.py
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
# --- 导入我们新的目标驱动优化器 ---
from src.tuning.tune_arima_threshold import find_factor_for_target_recall
from src.utils.metrics import evaluate_detection

ELEMENTS_TO_PROCESS = ['mean_motion', 'eccentricity', 'inclination']
TARGET_RECALL_LEVEL = 0.4 # 设定一个40%的召回率目标

def prepare_time_series(tle_df: pd.DataFrame, element: str) -> pd.Series:
    if element not in tle_df.columns:
        print(f"  [Warning] Element '{element}' not found. Skipping.")
        return None
    ts_data = tle_df[['epoch', element]].set_index('epoch')
    ts_data = ts_data.resample('D').median().interpolate(method='time').dropna()
    return ts_data[element]

def visualize_fusion_results(results_df, maneuvers, title, output_path):
    print("📊 Creating fusion visualization...")
    fig, axes = plt.subplots(len(ELEMENTS_TO_PROCESS) + 1, 1, figsize=(18, 14), sharex=True)
    
    for i, element in enumerate(ELEMENTS_TO_PROCESS):
        if f'actual_{element}' not in results_df.columns: continue
        ax = axes[i]
        ax.plot(results_df.index, results_df[f'actual_{element}'], 'k-', label=f'Actual {element}', linewidth=1)
        if maneuvers:
            for m in maneuvers: ax.axvline(m, color='green', linestyle=':', alpha=0.6)
        detections = results_df[results_df[f'is_anomaly_{element}']]
        ax.scatter(detections.index, detections[f'actual_{element}'], color='orange', marker='o', s=50, label=f'{element} Detection')
        ax.set_ylabel(element.replace('_', ' ').title())
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.4)

    ax_fusion = axes[-1]
    ax_fusion.plot(results_df.index, results_df['fused_score'], 'purple', label='Fused Anomaly Score')
    ax_fusion.axhline(1.0, color='red', linestyle='--', label='Fusion Threshold (1.0)')
    final_detections = results_df[results_df['is_final_anomaly']]
    ax_fusion.scatter(final_detections.index, final_detections['fused_score'], color='red', marker='x', s=100, label='Final Fused Detection', zorder=5)
    ax_fusion.set_ylabel('Fused Score')
    ax_fusion.set_xlabel('Date')
    ax_fusion.legend(loc='upper left')
    ax_fusion.grid(True, alpha=0.4)

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(output_path, dpi=200)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Multi-Element Fusion ARIMA Detection Pipeline")
    parser.add_argument("satellite", help="Satellite name")
    args = parser.parse_args()

    output_dir = Path("outputs") / f"{args.satellite.replace(' ', '_')}_Arima_Fusion_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}\n🛰️  Fusion Pipeline for: {args.satellite}\n{'='*60}")

    loader = SatelliteDataLoader(data_dir="data")
    tle_data, maneuver_times = loader.load_satellite_data(args.satellite)
    
    all_results = {}

    for element in ELEMENTS_TO_PROCESS:
        print(f"\n--- Processing Element: {element} ---")
        ts_data = prepare_time_series(tle_data, element)
        if ts_data is None: continue

        n = len(ts_data)
        train_end, val_end = int(n * 0.6), int(n * 0.8)
        train_data, val_data, test_data = ts_data.iloc[:train_end], ts_data.iloc[train_end:val_end], ts_data.iloc[val_end:]
        
        if len(train_data) < 50 or len(val_data) < 20 or len(test_data) < 20:
            print("  [Warning] Not enough data for a full split. Skipping element.")
            continue

        order = find_best_arima_order(train_data, args.satellite, element)
        config = DetectorConfig(order=order)
        detector = EnhancedARIMADetector(config=config)
        detector.fit(train_data)
        
        val_maneuvers = [m for m in maneuver_times if val_data.index.min() <= m <= val_data.index.max()]
        if val_maneuvers:
            # --- 使用新的、以召回率为目标的优化器 ---
            best_factor = find_factor_for_target_recall(detector, val_data, val_maneuvers, target_recall=TARGET_RECALL_LEVEL)
            detector.cfg.threshold_factor = best_factor
        else:
            print("  No maneuvers in validation set, using default threshold factor.")

        results_df = detector.detect(test_data)
        all_results[element] = results_df[['actual', 'score', 'is_anomaly']]
    
    if len(all_results) < 2:
        print("\n❌ Not enough elements processed for fusion. Exiting.")
        return

    print(f"\n--- Fusing results from {len(all_results)} elements ---")
    
    fusion_df = pd.DataFrame(index=all_results[list(all_results.keys())[0]].index)
    for element, df in all_results.items():
        fusion_df[f'actual_{element}'] = df['actual']
        fusion_df[f'score_{element}'] = df['score'].fillna(0)
        fusion_df[f'is_anomaly_{element}'] = df['is_anomaly']

    weights = {'mean_motion': 0.4, 'eccentricity': 0.4, 'inclination': 0.2}
    
    fusion_df['fused_score'] = 0.0
    for element in all_results.keys():
        fusion_df['fused_score'] += fusion_df[f'score_{element}'] * weights.get(element, 0.25)
    
    fusion_threshold = 1.0 
    fusion_df['is_final_anomaly'] = fusion_df['fused_score'] > fusion_threshold
    
    detected_anomalies = fusion_df[fusion_df['is_final_anomaly']].index.tolist()
    test_maneuvers = [m for m in maneuver_times if fusion_df.index.min() <= m <= fusion_df.index.max()]
    metrics, _ = evaluate_detection(detected_anomalies, test_maneuvers, timedelta(days=2))
    
    print(f"\n📊 FINAL FUSION PERFORMANCE:")
    print(f"  F1: {metrics['f1']:.3f}, P: {metrics['precision']:.2%}, R: {metrics['recall']:.2%}")

    plot_title = f"Multi-Element Fusion Detection for {args.satellite}"
    plot_path = output_dir / "fusion_detection_plot.png"
    visualize_fusion_results(fusion_df, test_maneuvers, plot_title, plot_path)
    
    report = {"satellite_name": args.satellite, "fusion_method": "Weighted Fusion", "weights": weights, "performance": metrics, "target_recall_for_tuning": TARGET_RECALL_LEVEL}
    with open(output_dir / 'fusion_summary_report.json', 'w') as f: json.dump(report, f, indent=2, default=str)
    
    print(f"\n✅ Fusion results saved to {output_dir}")

if __name__ == "__main__":
    main()