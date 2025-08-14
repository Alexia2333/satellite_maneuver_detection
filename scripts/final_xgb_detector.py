# scripts/final_xgb_detector.py
import os
import sys
from pathlib import Path
from datetime import datetime
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import SatelliteDataLoader
from src.data.feature_engineer import EnhancedSatelliteFeatureEngineer
from src.models.enhanced_xgb_detector import (
    EnhancedManeuverXGBDetector,
    EnhancedDetectorConfig
)
from src.tuning.auto_tune import random_search_xgb
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix
)


class FinalXGBDetector:
    def __init__(self, satellite_name: str, output_dir: str = "outputs"):
        self.satellite_name = satellite_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_name = f"{satellite_name.replace(' ', '_')}_{self.timestamp}_XGB"
        self.output_dir = Path(output_dir) / self.output_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.detector = None
        self.config = None
        self.training_history = {}
        self.evaluation_results = {}
        self.data_stats = {}
        
        print(f" Initializing Final XGB Detector for {satellite_name}")
        print(f" Output directory: {self.output_dir}")
    
    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, np.ndarray, pd.Series, List]:
        print("\n" + "="*60)
        print(" DATA LOADING AND PREPARATION")
        print("="*60)
        
        loader = SatelliteDataLoader(data_dir="data")
        tle_df, maneuver_times = loader.load_satellite_data(self.satellite_name)
        tle_df = tle_df.sort_values("epoch").reset_index(drop=True)
        
        print(f" Loaded {len(tle_df)} TLE records and {len(maneuver_times)} maneuvers")
        
        config_path = Path("configs") / f"{self.satellite_name}.json"
        orbit = "auto"
        label_window_days = 2
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                sat_config = json.load(f)
                orbit = sat_config.get("orbit", "auto")
                label_window_days = sat_config.get("label_window_days", 2)
        
        print("\n Engineering features...")
        eng = EnhancedSatelliteFeatureEngineer(
            target_column="mean_motion",
            additional_columns=["eccentricity", "inclination"],
            satellite_type=orbit
        ).fit(tle_df, satellite_name=self.satellite_name)
        feat_df = eng.transform(tle_df)
        
        label_window_min = label_window_days * 1440
        epoch_series = tle_df.loc[feat_df.index, "epoch"]
        y_series = self._build_labels(epoch_series, maneuver_times, label_window_min)
        feat_df["label"] = y_series.values
        
        X = feat_df.drop(columns=["label"]).select_dtypes(include=[np.number])
        y = feat_df["label"].astype(int).values
        timestamps = tle_df.loc[feat_df.index, "epoch"]
        
        self.data_stats = {
            "n_samples": len(X), "n_features": X.shape[1],
            "n_positive": int(y.sum()), "n_negative": int(len(y) - y.sum()),
            "positive_ratio": float(y.sum() / len(y)), "n_maneuvers": len(maneuver_times),
            "label_window_days": label_window_days, "orbit_type": orbit
        }
        
        print(f"\n Data Statistics:")
        for key, val in self.data_stats.items():
            print(f"  • {key.replace('_', ' ').title()}: {val:.1f}%" if "ratio" in key else f"  • {key.replace('_', ' ').title()}: {val}")
        
        return X, y, timestamps, maneuver_times
    
    def _build_labels(self, timestamps, events, window_min):
        dt = pd.to_datetime(timestamps)
        y = np.zeros(len(timestamps), dtype=int)
        half = pd.to_timedelta(window_min, unit="m")
        for ev in events:
            mask = (dt >= ev - half) & (dt <= ev + half)
            y[mask.values] = 1
        return pd.Series(y, index=timestamps.index, name="label")
    
    def auto_tune_hyperparameters(self, X, y, orbit="GEO") -> Dict:
        print("\n" + "="*60)
        print(" HYPERPARAMETER TUNING")
        print("="*60)
        
        param_grid = {
            "max_depth": [4, 6, 8], "learning_rate": [0.03, 0.05, 0.1],
            "subsample": [0.85, 0.95], "colsample_bytree": [0.85, 0.95],
            "reg_alpha": [0.0, 0.01, 0.05], "reg_lambda": [0.5, 1.0, 2.0],
            "n_estimators": [600, 800, 1000] if orbit == "GEO" else [400, 600, 800],
            "min_child_weight": [1, 2], "scale_pos_weight": [5, 10, 15]
        }
        
        print(f"Running random search with 25 iterations...")
        best_params, best_f1, best_threshold = random_search_xgb(
            X, y, param_grid=param_grid, n_iter=25, test_size=0.2,
            early_stopping_rounds=30, random_state=42, maximize="f1"
        )
        
        print(f"\n Tuning complete!")
        print(f"  • Best F1 score: {best_f1:.4f}")
        print(f"  • Best threshold: {best_threshold:.4f}")
        
        self.training_history['tuning'] = {
            "best_f1": float(best_f1), "best_threshold": float(best_threshold),
            "best_params": best_params
        }
        
        return best_params
    
    def train_detector(self, X, y, timestamps, maneuver_times, best_params=None):
        print("\n" + "="*60)
        print(" TRAINING DETECTOR")
        print("="*60)
        
        self.config = EnhancedDetectorConfig(
            detection_strategy="top_n", expected_events=len(maneuver_times),
            target_recall=0.8, orbit=self.data_stats.get("orbit_type", "GEO"),
            min_gap_days=5.0, cluster_window_days=7.0,
            scale_pos_weight=best_params.get("scale_pos_weight", 10) if best_params else 10
        )
        
        self.detector = EnhancedManeuverXGBDetector(self.config)
        if best_params:
            self.detector.hyperparams = best_params
        
        val_start = int(0.8 * len(X))
        print(f"\n Training with {val_start} samples and validating on {len(X)-val_start} samples...")
        
        self.detector.fit_from_features(X, y, orbit=self.config.orbit, val_start=val_start)
        
        self.training_history['training'] = {
            "train_samples": val_start, "val_samples": len(X) - val_start,
            "final_threshold": float(self.detector.fitted_threshold_)
        }
        
        print(f" Training complete! Final threshold: {self.detector.fitted_threshold_:.4f}")
        return val_start
    
    def evaluate_model(self, X, y, timestamps, maneuver_times, val_start):
        print("\n" + "="*60)
        print(" MODEL EVALUATION")
        print("="*60)
        
        scores = self.detector.predict_scores_from_features(X)
        detections_df = self.detector.detect_from_features(X, timestamps)
        label_window_min = self.data_stats.get("label_window_days", 2) * 1440
        eval_results = self.detector.evaluate_detections(detections_df, maneuver_times, label_window_min)
        
        val_scores, val_y = scores[val_start:], y[val_start:]
        fpr, tpr, _ = roc_curve(val_y, val_scores)
        roc_auc = auc(fpr, tpr)
        precision, recall, pr_thresholds = precision_recall_curve(val_y, val_scores)
        pr_auc = average_precision_score(val_y, val_scores)
        
        self.evaluation_results = {
            "detection_performance": eval_results,
            "validation_metrics": {"roc_auc": float(roc_auc), "pr_auc": float(pr_auc)},
            "curves": {
                "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(roc_auc)},
                "pr": {"precision": precision.tolist(), "recall": recall.tolist(), "auc": float(pr_auc)}
            }
        }
        
        print(f"\n Performance Summary:")
        print(f"  Recall: {eval_results['recall']:.1%}, Precision: {eval_results['precision']:.1%}, F1: {eval_results['f1']:.3f}")
        print(f"  ROC AUC: {roc_auc:.3f}, PR AUC: {pr_auc:.3f}")
        
        return detections_df, eval_results
    
    def create_visualizations(self, X, timestamps, detections_df, eval_results):
        print("\n" + "="*60)
        print(" CREATING VISUALIZATIONS")
        print("="*60)
        
        scores = self.detector.predict_scores_from_features(X)
        
        fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
        ax1 = axes[0]
        ax1.plot(timestamps, scores, linewidth=0.8, alpha=0.7, color='blue', label='Anomaly Score')
        ax1.axhline(self.detector.fitted_threshold_, color='red', linestyle='--', alpha=0.5, label=f'Threshold ({self.detector.fitted_threshold_:.3f})')
        
        initial_detections = detections_df[detections_df['pred'] == 1]
        ax1.scatter(initial_detections['timestamp'], initial_detections['score'], 
                    color='red', marker='x', s=100, zorder=5, label='Initial Detection')

        clusters = detections_df.dropna(subset=['cluster_id'])
        if not clusters.empty:
            best_detections = clusters.loc[clusters.groupby('cluster_id')['score'].idxmax()]
            ax1.scatter(best_detections['timestamp'], best_detections['score'], 
                        edgecolor='green', facecolor='none', marker='o', s=200, linewidth=2, 
                        label='Final Clustered Event', zorder=10)
        
        ax1.set_ylabel('Anomaly Score')
        ax1.set_title(f'{self.satellite_name} - XGBoost Detection Results (F1: {eval_results["f1"]:.3f})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[1]
        detected_events = [pd.to_datetime(ts) for ts in eval_results['detected_events']]
        missed_events = [pd.to_datetime(ts) for ts in eval_results['missed_events']]
        false_alarms = [pd.to_datetime(ts) for ts in eval_results['false_alarms']]

        if detected_events: ax2.plot(detected_events, [1]*len(detected_events), 'go', markersize=8, label=f'Detected (TP={len(detected_events)})')
        if missed_events: ax2.plot(missed_events, [1]*len(missed_events), 'rx', markersize=8, label=f'Missed (FN={len(missed_events)})')
        if false_alarms: ax2.plot(false_alarms, [1]*len(false_alarms), 'y|', markersize=12, label=f'False Alarm (FP={len(false_alarms)})')
        
        ax2.set_yticks([])
        ax2.set_xlabel('Date')
        ax2.set_title('Event Detection Timeline')
        ax2.legend()
        ax2.grid(True, axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'detection_plot.png', dpi=200)
        plt.close()
        
        print(f" Visualizations saved to {self.output_dir}")
    
    def save_all_results(self, X, timestamps, detections_df):
        print("\n" + "="*60)
        print(" SAVING RESULTS")
        print("="*60)
        
        scores = self.detector.predict_scores_from_features(X)
        scores_df = pd.DataFrame({
            'timestamp': timestamps,
            'score': scores
        })
        scores_df.to_csv(self.output_dir / 'scores_timeline.csv', index=False)
        print(f"Saved scores timeline to scores_timeline.csv")

        detections_df.to_csv(self.output_dir / 'detections.csv', index=False)

        results = {
            "metadata": {"satellite": self.satellite_name, "timestamp": self.timestamp},
            "data_statistics": self.data_stats,
            "training_history": self.training_history,
            "evaluation_results": self.evaluation_results,
            "configuration": self.config.__dict__ if self.config else {}
        }
        
        with open(self.output_dir / 'results.json', 'w', encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        

        model_path = self.output_dir / 'model.json'
        self.detector.save_model(str(model_path))
        
        print(f" All results saved to {self.output_dir}")

    def run(self, auto_tune=True):
        print("\n" + "="*70)
        print(f"🚀 STARTING FINAL XGB DETECTION FOR {self.satellite_name}")
        print("="*70)
        
        try:
            X, y, timestamps, maneuver_times = self.load_and_prepare_data()
            best_params = self.auto_tune_hyperparameters(X, y, self.data_stats.get("orbit_type", "GEO")) if auto_tune else None
            val_start = self.train_detector(X, y, timestamps, maneuver_times, best_params)
            detections_df, eval_results = self.evaluate_model(X, y, timestamps, maneuver_times, val_start)
            self.create_visualizations(X, timestamps, detections_df, eval_results)
            self.save_all_results(X, timestamps, detections_df)
            
            print(f"\n DETECTION COMPLETE FOR {self.satellite_name}")
            return True
            
        except Exception as e:
            print(f"\n Error during detection: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

def main():
    parser = argparse.ArgumentParser(description="Final XGBoost Maneuver Detection with Auto-tuning")
    parser.add_argument("satellite", nargs="?", help="Satellite name (e.g., Fengyun-4A)")
    parser.add_argument("--no-tune", action="store_true", help="Skip hyperparameter tuning")
    args = parser.parse_args()
    
    satellite_name = args.satellite or input("Enter satellite name (e.g., Fengyun-4A): ").strip()
    if not satellite_name:
        print(" Satellite name is required!")
        return
    
    detector = FinalXGBDetector(satellite_name=satellite_name)
    success = detector.run(auto_tune=not args.no_tune)
    
    if success:
        print(f"\n✨ Success! Check {detector.output_dir} for results.")
    else:
        print("\n Detection failed. Check the error messages above.")

if __name__ == "__main__":
    main()
