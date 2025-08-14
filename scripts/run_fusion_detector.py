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
from src.tuning.tune_arima_threshold import find_factor_for_target_recall
from src.utils.metrics import evaluate_detection

ELEMENTS_TO_PROCESS = ['mean_motion', 'eccentricity', 'inclination']
TARGET_RECALL_LEVEL = 0.4 

def prepare_time_series(tle_df: pd.DataFrame, element: str) -> pd.Series:
    if element not in tle_df.columns:
        print(f"  [Warning] Element '{element}' not found. Skipping.")
        return None
    ts_data = tle_df[['epoch', element]].set_index('epoch')
    ts_data = ts_data.resample('D').median().interpolate(method='time').dropna()
    return ts_data[element]

def visualize_fusion_results(results_df, maneuvers, title, output_path):
    print(" Creating fusion visualization...")
    fig, axes = plt.subplots(len(ELEMENTS_TO_PROCESS) + 1, 1, figsize=(18, 14), sharex=True)
    
    for i, element in enumerate(ELEMENTS_TO_PROCESS):
        if f'actual_{element}' not in results_df.columns: continue
        ax = axes[i]
        ax.plot(results_df.index, results_df[f'actual_{element}'], 'k-', label=f'Actual {element}', linewidth=1)
        if maneuvers:
            maneuver_lines = [ax.axvline(m, color='green', linestyle=':', alpha=0.6) for m in maneuvers]
        detections = results_df[results_df[f'is_anomaly_{element}']]
        ax.scatter(detections.index, detections[f'actual_{element}'], color='orange', marker='o', s=50, label=f'{element} Detection')
        ax.set_ylabel(element.replace('_', ' ').title())
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.4)

    ax_fusion = axes[-1]
    ax_fusion.plot(results_df.index, results_df['fused_score'], 'purple', label='Fused Anomaly Score')
    ax_fusion.axhline(1.0, color='red', linestyle='--', label='Fusion Threshold (1.0)')

    initial_detections = results_df[results_df['is_final_anomaly']]
    ax_fusion.scatter(initial_detections.index, initial_detections['fused_score'], 
                      color='red', marker='x', s=100, label='Initial Fused Detection', zorder=5)

    if 'is_clustered_anomaly' in results_df.columns:
        final_clustered_detections = results_df[results_df['is_clustered_anomaly']]
        ax_fusion.scatter(final_clustered_detections.index, final_clustered_detections['fused_score'], 
                          edgecolor='green', facecolor='none', marker='o', s=200, linewidth=2, 
                          label='Final Clustered Event', zorder=10)

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
    
    print(f"\n{'='*60}\n  Fusion Pipeline for: {args.satellite}\n{'='*60}")

    loader = SatelliteDataLoader(data_dir="data")
    tle_data, maneuver_times = loader.load_satellite_data(args.satellite)
    
    all_results = {}
    individual_element_performance_list = []
    individual_element_events = {}

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
            best_factor = find_factor_for_target_recall(detector, val_data, val_maneuvers, target_recall=TARGET_RECALL_LEVEL)
            detector.cfg.threshold_factor = best_factor
        else:
            print("  No maneuvers in validation set, using default threshold factor.")

        results_df = detector.detect(test_data)
        all_results[element] = results_df[['actual', 'score', 'is_anomaly']]
        
        print(f"  Evaluating performance for single element: {element}")
        element_detections = results_df[results_df['is_anomaly']].index.tolist()
        test_maneuvers_element = [m for m in maneuver_times if test_data.index.min() <= m <= test_data.index.max()]
        
        element_metrics, element_events = evaluate_detection(element_detections, test_maneuvers_element, timedelta(days=2))
        
        element_metrics['element'] = element
        individual_element_performance_list.append(element_metrics)
        individual_element_events[element] = element_events
        
        print(f"  -> Single Element Performance (F1: {element_metrics['f1']:.3f}, P: {element_metrics['precision']:.2%}, R: {element_metrics['recall']:.2%})")
    
    if len(all_results) < 2:
        print("\n Not enough elements processed for fusion. Exiting.")
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
    
    FUSION_MIN_GAP_DAYS = 5
    fusion_df['is_clustered_anomaly'] = False 
    anomaly_points = fusion_df[fusion_df['is_final_anomaly']]
    
    if not anomaly_points.empty:
        anomalies = anomaly_points.sort_index()
        time_diffs = anomalies.index.to_series().diff().dt.days.fillna(FUSION_MIN_GAP_DAYS + 1)
        cluster_ids = (time_diffs > FUSION_MIN_GAP_DAYS).cumsum()
        
        for _, cluster in anomalies.groupby(cluster_ids):
            best_in_cluster = cluster['fused_score'].idxmax()
            fusion_df.loc[best_in_cluster, 'is_clustered_anomaly'] = True
    
    detected_anomalies = fusion_df[fusion_df['is_clustered_anomaly']].index.tolist()

    test_maneuvers = [m for m in maneuver_times if fusion_df.index.min() <= m <= fusion_df.index.max()]
    metrics, fusion_events = evaluate_detection(detected_anomalies, test_maneuvers, timedelta(days=2))
    
    print(f"\n📊 FINAL FUSION PERFORMANCE:")
    print(f"  F1: {metrics['f1']:.3f}, P: {metrics['precision']:.2%}, R: {metrics['recall']:.2%}")

    plot_title = f"Multi-Element Fusion Detection for {args.satellite}"
    plot_path = output_dir / "fusion_detection_plot.png"
    visualize_fusion_results(fusion_df, test_maneuvers, plot_title, plot_path)
    
    summary_report = {
        "metadata": {
            "satellite_name": args.satellite,
            "model_type": "ARIMA_Fusion",
            "timestamp": datetime.now().isoformat()
        },
        "fusion_config": {
            "fusion_method": "Weighted Fusion",
            "weights": weights,
            "target_recall_for_tuning": TARGET_RECALL_LEVEL
        },
        "fusion_performance": metrics,
        "individual_element_performance": individual_element_performance_list
    }

    events_report = {
        "metadata": {
            "satellite_name": args.satellite,
            "model_type": "ARIMA_Fusion",
            "timestamp": datetime.now().isoformat()
        },
        "fusion_events": fusion_events,
        "individual_element_events": individual_element_events
    }

    summary_report_path = output_dir / 'arima_fusion_report.json'
    with open(summary_report_path, 'w') as f:
        json.dump(summary_report, f, indent=2, default=str)
    print(f"Saved comprehensive ARIMA summary report to {summary_report_path}")

    events_report_path = output_dir / 'arima_fusion_events.json'
    with open(events_report_path, 'w') as f:
        json.dump(events_report, f, indent=2, default=str)
    print(f"Saved detailed event data to {events_report_path}")
    
    if individual_element_performance_list:
        individual_perf_df = pd.DataFrame(individual_element_performance_list)
        individual_perf_path_csv = output_dir / 'individual_element_performance.csv'
        individual_perf_df.to_csv(individual_perf_path_csv, index=False)
        print(f"Saved individual element performance to {individual_perf_path_csv}")
    
    print(f"\n Fusion results saved to {output_dir}")

if __name__ == "__main__":
    main()
