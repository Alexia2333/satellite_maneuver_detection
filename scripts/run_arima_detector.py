import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 添加项目路径
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import SatelliteDataLoader
from src.models.arima_detector import ARIMADetector
from src.tuning.auto_tune_arima import find_best_arima_order
from src.tuning.tune_arima_threshold import find_best_threshold_factor
from src.utils.metrics import evaluate_detection

def visualize_results(results_df: pd.DataFrame, maneuver_times: list, satellite_name: str, element: str, output_dir: Path):
    # (此函数无需修改)
    print("Creating visualization...")
    anomaly_points = results_df[results_df['is_anomaly'] == 1]
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.plot(results_df.index, results_df['actual'], label=f'Actual {element}', color='black', linewidth=1.5, alpha=0.8)
    ax.plot(results_df.index, results_df['forecast'], label='ARIMA Forecast', color='blue', linestyle='--', linewidth=1)
    ax_score = ax.twinx()
    ax_score.plot(results_df.index, results_df['score'], label='Anomaly Score (Residual)', color='orange', alpha=0.5, linewidth=1)
    ax_score.plot(results_df.index, results_df['threshold'], label='Dynamic Threshold', color='red', linestyle=':', linewidth=1.5)
    ax.scatter(anomaly_points.index, anomaly_points['actual'], color='magenta', s=100, label='Detected Anomaly', zorder=5, marker='x')
    for maneuver in maneuver_times:
        ax.axvline(maneuver, color='green', linestyle='--', linewidth=1.5, label='True Maneuver')
    ax.set_title(f'ARIMA Anomaly Detection for {satellite_name} - {element}', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Orbital Element Value', fontsize=12)
    ax_score.set_ylabel('Anomaly Score / Threshold', fontsize=12)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left')
    plt.tight_layout()
    output_path = output_dir / f"{satellite_name}_{element}_detection_plot.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"✅ Visualization saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Run Auto-Tuned ARIMA Maneuver Detection")
    parser.add_argument("satellite", help="Satellite name (e.g., Jason-1)")
    args = parser.parse_args()
    
    ELEMENT_TO_ANALYZE = "mean_motion"
    output_dir = Path("outputs") / f"{args.satellite.replace(' ', '_')}_ARIMA_{ELEMENT_TO_ANALYZE}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    loader = SatelliteDataLoader(data_dir="data")
    tle_data, maneuver_times = loader.load_satellite_data(args.satellite)
    
    ts_data = tle_data[['epoch', ELEMENT_TO_ANALYZE]].set_index('epoch').resample('D').median().interpolate(method='time').dropna()[ELEMENT_TO_ANALYZE]

    # --- 【核心修改】先净化，后划分 ---
    # 1. 在整个时间序列上进行离群点识别和插值
    print("\nCleaning the entire time series before splitting...")
    q1, q3 = ts_data.quantile(0.25), ts_data.quantile(0.75)
    iqr = q3 - q1
    lower_bound, upper_bound = q1 - 2.0 * iqr, q3 + 2.0 * iqr
    outlier_mask = (ts_data < lower_bound) | (ts_data > upper_bound)
    
    cleaned_ts_data = ts_data.copy()
    cleaned_ts_data[outlier_mask] = np.nan
    cleaned_ts_data = cleaned_ts_data.interpolate(method='time')
    print(f"Identified and interpolated {outlier_mask.sum()} outlier points from the entire dataset.")
    # ------------------------------------

    labels = pd.Series(0, index=cleaned_ts_data.index)
    for m_time in maneuver_times:
        mask = (cleaned_ts_data.index >= m_time - timedelta(days=1)) & (cleaned_ts_data.index <= m_time + timedelta(days=1))
        labels[mask] = 1
    
    # 2. 在净化后的数据上，进行分层时间划分
    print("\nPerforming stratified time-series split on cleaned data...")
    maneuver_indices = np.where(labels == 1)[0]
    
    if len(maneuver_indices) < 5:
        train_end_idx = int(len(cleaned_ts_data) * 0.7)
        val_end_idx = int(len(cleaned_ts_data) * 0.85)
    else:
        train_maneuver_count = int(len(maneuver_indices) * 0.6)
        val_maneuver_count = int(len(maneuver_indices) * 0.2)
        train_end_idx = maneuver_indices[train_maneuver_count]
        val_end_idx = maneuver_indices[train_maneuver_count + val_maneuver_count]

    train_data = cleaned_ts_data.iloc[:train_end_idx]
    val_data = cleaned_ts_data.iloc[train_end_idx:val_end_idx]
    test_data = cleaned_ts_data.iloc[val_end_idx:]
    
    train_labels = labels.iloc[:train_end_idx]
    val_labels = labels.iloc[train_end_idx:val_end_idx]
    test_labels = labels.iloc[val_end_idx:]

    print(f"Train set: {len(train_data)} points ({train_labels.sum()} maneuvers).")
    print(f"Validation set: {len(val_data)} points ({val_labels.sum()} maneuvers).")
    print(f"Test set: {len(test_data)} points ({test_labels.sum()} maneuvers).")
    
    # 3. 自动调优、训练、阈值优化、评估流程不变
    best_order = find_best_arima_order(train_data)
    if not best_order:
        print("❌ Auto-tuning failed. Aborting.")
        return
        
    detector = ARIMADetector(p=best_order[0], d=best_order[1], q=best_order[2])
    detector.fit(train_data)
    
    print("\n--- Tuning Threshold Factor on Validation Set ---")
    val_results = detector.detect(val_data)
    best_factor, _ = find_best_threshold_factor(scores=val_results['score'], true_labels=val_labels, threshold_base=val_results['threshold'] / 3.5)
    
    print(f"\n--- Final Evaluation on Test Set (using tuned factor={best_factor:.2f}) ---")
    results_df = detector.detect(test_data, threshold_factor=best_factor)
    
    detected_anomalies = results_df[results_df['is_anomaly'] == 1].index.tolist()
    test_maneuvers = [m for m in maneuver_times if test_data.index.min() <= m <= test_data.index.max()]
    eval_metrics, _ = evaluate_detection(detected_anomalies, test_maneuvers, matching_window=timedelta(days=1))
    
    print("\n📊 FINAL PERFORMANCE:")
    print(f"  - F1 Score:  {eval_metrics['f1']:.3f}, Precision: {eval_metrics['precision']:.2%}, Recall: {eval_metrics['recall']:.2%}")

    visualize_results(results_df, test_maneuvers, args.satellite, ELEMENT_TO_ANALYZE, output_dir)
    
    results_df.to_csv(output_dir / "detection_results.csv")
    report = {"satellite_name": args.satellite, "element": ELEMENT_TO_ANALYZE, "best_order": list(best_order), "best_threshold_factor": best_factor, "performance": eval_metrics}
    with open(output_dir / 'summary_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"✅ Detection results and report saved to {output_dir}")

if __name__ == "__main__":
    main()