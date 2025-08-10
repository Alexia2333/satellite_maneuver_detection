"""
Final production-ready XGBoost maneuver detector with auto-tuning.
Automatically trains, tunes, and evaluates the model with comprehensive output.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import argparse
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
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
    confusion_matrix, classification_report
)


class FinalXGBDetector:
    """
    Final production XGBoost detector with comprehensive logging and evaluation.
    """
    
    def __init__(self, satellite_name: str, output_dir: str = "outputs"):
        self.satellite_name = satellite_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_name = f"{satellite_name.replace(' ', '_')}_{self.timestamp}_XGB"
        self.output_dir = Path(output_dir) / self.output_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.detector = None
        self.config = None
        self.training_history = {}
        self.evaluation_results = {}
        self.data_stats = {}
        
        print(f"🛰️ Initializing Final XGB Detector for {satellite_name}")
        print(f"📁 Output directory: {self.output_dir}")
    
    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, np.ndarray, pd.Series, List]:
        """Load satellite data and prepare features."""
        print("\n" + "="*60)
        print("📊 DATA LOADING AND PREPARATION")
        print("="*60)
        
        # Load data
        loader = SatelliteDataLoader(data_dir="data")
        tle_df, maneuver_times = loader.load_satellite_data(self.satellite_name)
        tle_df = tle_df.sort_values("epoch").reset_index(drop=True)
        
        print(f"✅ Loaded {len(tle_df)} TLE records and {len(maneuver_times)} maneuvers")
        
        # Determine orbit type from config or auto-detect
        config_path = Path("configs") / f"{self.satellite_name}.json"
        orbit = "auto"
        label_window_days = 2
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                sat_config = json.load(f)
                orbit = sat_config.get("orbit", "auto")
                label_window_days = sat_config.get("label_window_days", 2)
        
        # Feature engineering
        print("\n🔧 Engineering features...")
        eng = EnhancedSatelliteFeatureEngineer(
            target_column="mean_motion",
            additional_columns=["eccentricity", "inclination"],
            satellite_type=orbit
        ).fit(tle_df, satellite_name=self.satellite_name)
        feat_df = eng.transform(tle_df)
        
        # Create labels
        label_window_min = label_window_days * 1440
        epoch_series = tle_df.loc[feat_df.index, "epoch"]
        y_series = self._build_labels(epoch_series, maneuver_times, label_window_min)
        feat_df["label"] = y_series.values
        
        # Prepare final data
        X = feat_df.drop(columns=["label"]).select_dtypes(include=[np.number])
        y = feat_df["label"].astype(int).values
        timestamps = tle_df.loc[feat_df.index, "epoch"]
        
        # Store data statistics
        self.data_stats = {
            "n_samples": len(X),
            "n_features": X.shape[1],
            "n_positive": int(y.sum()),
            "n_negative": int(len(y) - y.sum()),
            "positive_ratio": float(y.sum() / len(y)),
            "n_maneuvers": len(maneuver_times),
            "label_window_days": label_window_days,
            "orbit_type": orbit
        }
        
        print(f"\n📊 Data Statistics:")
        print(f"  • Samples: {self.data_stats['n_samples']}")
        print(f"  • Features: {self.data_stats['n_features']}")
        print(f"  • Positive samples: {self.data_stats['n_positive']} ({100*self.data_stats['positive_ratio']:.1f}%)")
        print(f"  • Maneuvers: {self.data_stats['n_maneuvers']}")
        
        return X, y, timestamps, maneuver_times
    
    def _build_labels(self, timestamps, events, window_min):
        """Build binary labels for events."""
        dt = pd.to_datetime(timestamps)
        y = np.zeros(len(timestamps), dtype=int)
        half = pd.to_timedelta(window_min, unit="m")
        for ev in events:
            mask = (dt >= ev - half) & (dt <= ev + half)
            y[mask.values] = 1
        return pd.Series(y, index=timestamps.index, name="label")
    
    def auto_tune_hyperparameters(self, X, y, orbit="GEO") -> Dict:
        """Perform hyperparameter tuning."""
        print("\n" + "="*60)
        print("🔧 HYPERPARAMETER TUNING")
        print("="*60)
        
        # Define parameter grid based on orbit
        if orbit == "GEO":
            param_grid = {
                "max_depth": [4, 5, 6, 7],
                "learning_rate": [0.03, 0.05, 0.08],
                "subsample": [0.85, 0.95, 1.0],
                "colsample_bytree": [0.85, 0.95, 1.0],
                "reg_alpha": [0.0, 0.01, 0.05],
                "reg_lambda": [0.5, 1.0, 2.0],
                "n_estimators": [600, 800, 1000],
                "min_child_weight": [1, 2],
                "scale_pos_weight": [5, 8, 10, 12]
            }
        else:
            param_grid = {
                "max_depth": [4, 5, 6],
                "learning_rate": [0.05, 0.1],
                "subsample": [0.9, 1.0],
                "colsample_bytree": [0.9, 1.0],
                "reg_alpha": [0.0, 0.01],
                "reg_lambda": [1.0, 2.0],
                "n_estimators": [400, 600],
                "min_child_weight": [1, 2],
                "scale_pos_weight": [5, 10]
            }
        
        print(f"Running random search with 25 iterations...")
        print(f"Optimizing for F1 score...")
        
        best_params, best_f1, best_threshold = random_search_xgb(
            X, y,
            param_grid=param_grid,
            n_iter=25,
            test_size=0.2,
            early_stopping_rounds=30,
            random_state=42,
            maximize="f1"
        )
        
        print(f"\n✅ Tuning complete!")
        print(f"  • Best F1 score: {best_f1:.4f}")
        print(f"  • Best threshold: {best_threshold:.4f}")
        print(f"  • Best parameters:")
        for key, value in best_params.items():
            print(f"    - {key}: {value}")
        
        self.training_history['tuning'] = {
            "best_f1": float(best_f1),
            "best_threshold": float(best_threshold),
            "best_params": best_params,
            "param_grid": param_grid,
            "n_iterations": 25
        }
        
        return best_params
    
    def train_detector(self, X, y, timestamps, maneuver_times, best_params=None):
        """Train the enhanced detector."""
        print("\n" + "="*60)
        print("🎯 TRAINING DETECTOR")
        print("="*60)
        
        # Create configuration
        self.config = EnhancedDetectorConfig(
            detection_strategy="top_n",
            expected_events=len(maneuver_times),
            target_recall=0.8,
            orbit=self.data_stats.get("orbit_type", "GEO"),
            min_gap_days=5.0,
            cluster_window_days=7.0,
            scale_pos_weight=best_params.get("scale_pos_weight", 10) if best_params else 10
        )
        
        print(f"Configuration:")
        print(f"  • Strategy: {self.config.detection_strategy}")
        print(f"  • Expected events: {self.config.expected_events}")
        print(f"  • Target recall: {self.config.target_recall:.0%}")
        
        # Create detector
        self.detector = EnhancedManeuverXGBDetector(self.config)
        
        # Set hyperparameters if provided
        if best_params:
            self.detector.hyperparams = best_params
            print(f"  • Using auto-tuned hyperparameters")
        else:
            print(f"  • Using default hyperparameters")
        
        # Train
        val_start = int(0.8 * len(X))
        print(f"\n📊 Training with {val_start} training samples and {len(X)-val_start} validation samples...")
        
        self.detector.fit_from_features(
            X, y,
            orbit=self.config.orbit if self.config.orbit != "auto" else "GEO",
            val_start=val_start
        )
        
        # Store training info
        self.training_history['training'] = {
            "train_samples": val_start,
            "val_samples": len(X) - val_start,
            "train_positive": int(y[:val_start].sum()),
            "val_positive": int(y[val_start:].sum()),
            "final_threshold": float(self.detector.fitted_threshold_)
        }
        
        print(f"✅ Training complete!")
        print(f"  • Final threshold: {self.detector.fitted_threshold_:.4f}")
        
        return val_start
    
    def evaluate_model(self, X, y, timestamps, maneuver_times, val_start):
        """Comprehensive model evaluation."""
        print("\n" + "="*60)
        print("📈 MODEL EVALUATION")
        print("="*60)
        
        # Get predictions
        scores = self.detector.predict_scores_from_features(X)
        val_scores = scores[val_start:]
        val_y = y[val_start:]
        
        # Detection results
        detections_df = self.detector.detect_from_features(X, timestamps)
        label_window_min = self.data_stats.get("label_window_days", 2) * 1440
        eval_results = self.detector.evaluate_detections(
            detections_df, maneuver_times, label_window_min
        )
        
        # Calculate metrics
        from sklearn.metrics import roc_auc_score
        
        # ROC and PR curves
        fpr, tpr, roc_thresholds = roc_curve(val_y, val_scores)
        roc_auc = auc(fpr, tpr)
        
        precision, recall, pr_thresholds = precision_recall_curve(val_y, val_scores)
        pr_auc = average_precision_score(val_y, val_scores)
        
        # Confusion matrix at optimal threshold
        val_pred = (val_scores >= self.detector.fitted_threshold_).astype(int)
        cm = confusion_matrix(val_y, val_pred)
        
        # Store evaluation results
        self.evaluation_results = {
            "detection_performance": {
                "true_positives": eval_results['tp'],
                "false_positives": eval_results['fp'],
                "false_negatives": eval_results['fn'],
                "recall": eval_results['recall'],
                "precision": eval_results['precision'],
                "f1": eval_results['f1']
            },
            "validation_metrics": {
                "roc_auc": float(roc_auc),
                "pr_auc": float(pr_auc),
                "confusion_matrix": cm.tolist(),
                "threshold_used": float(self.detector.fitted_threshold_)
            },
            "curves": {
                "roc": {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                    "thresholds": roc_thresholds.tolist(),
                    "auc": float(roc_auc)
                },
                "pr": {
                    "precision": precision.tolist(),
                    "recall": recall.tolist(),
                    "thresholds": pr_thresholds.tolist(),
                    "auc": float(pr_auc)
                }
            }
        }
        
        print(f"\n📊 Performance Summary:")
        print(f"  Detection Performance:")
        print(f"    • Detected: {eval_results['tp']}/{len(maneuver_times)} events")
        print(f"    • Recall: {eval_results['recall']:.1%}")
        print(f"    • Precision: {eval_results['precision']:.1%}")
        print(f"    • F1 Score: {eval_results['f1']:.3f}")
        print(f"  Validation Metrics:")
        print(f"    • ROC AUC: {roc_auc:.3f}")
        print(f"    • PR AUC: {pr_auc:.3f}")
        
        return detections_df, eval_results
    
    def create_visualizations(self, X, timestamps, maneuver_times, detections_df, eval_results):
        """Create and save all visualizations."""
        print("\n" + "="*60)
        print("📊 CREATING VISUALIZATIONS")
        print("="*60)
        
        scores = self.detector.predict_scores_from_features(X)
        
        # 1. Main detection plot
        fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
        
        # Top: Score timeline
        ax1 = axes[0]
        ax1.plot(timestamps, scores, linewidth=0.8, alpha=0.7, color='blue', label='Anomaly Score')
        ax1.axhline(self.detector.fitted_threshold_, color='red', linestyle='--', 
                   alpha=0.5, label=f'Threshold ({self.detector.fitted_threshold_:.3f})')
        
        # Mark detected clusters
        clusters = detections_df.dropna(subset=['cluster_id'])
        for cluster_id in clusters['cluster_id'].unique():
            cluster_data = clusters[clusters['cluster_id'] == cluster_id]
            best_idx = cluster_data['score'].idxmax()
            ax1.scatter(cluster_data.loc[best_idx, 'timestamp'],
                       cluster_data.loc[best_idx, 'score'],
                       color='red', s=100, zorder=5)
        
        ax1.set_ylabel('Anomaly Score')
        ax1.set_title(f'{self.satellite_name} - XGBoost Detection Results\n'
                     f'Recall: {eval_results["recall"]:.1%}, Precision: {eval_results["precision"]:.1%}, '
                     f'F1: {eval_results["f1"]:.3f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Middle: Event timeline
        ax2 = axes[1]
        for ev in eval_results['detected_events']:
            ax2.axvline(ev, color='green', alpha=0.7, linewidth=2)
        for ev in eval_results['missed_events']:
            ax2.axvline(ev, color='red', alpha=0.5, linewidth=1, linestyle='--')
        for fa in eval_results['false_alarms']:
            ax2.axvline(pd.Timestamp(fa), color='orange', alpha=0.5, linewidth=1, linestyle=':')
        
        ax2.set_ylabel('Events')
        ax2.set_title(f'Detection Performance: TP={eval_results["tp"]}, FP={eval_results["fp"]}, FN={eval_results["fn"]}')
        ax2.grid(True, alpha=0.3)
        
        # Bottom: Detection density
        ax3 = axes[2]
        det_series = pd.Series(0, index=pd.to_datetime(timestamps))
        for ts_val in clusters['timestamp']:
            det_series.loc[ts_val] = 1
        rolling_det = det_series.rolling('30D').sum()
        ax3.fill_between(timestamps, 0, rolling_det, alpha=0.5, color='blue')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Detection Density (30-day)')
        ax3.set_title('Detection Frequency Over Time')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'detection_plot.png', dpi=200, bbox_inches='tight')
        plt.close()
        
        # 2. ROC and PR curves
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # ROC curve
        roc_data = self.evaluation_results['curves']['roc']
        ax1.plot(roc_data['fpr'], roc_data['tpr'], linewidth=2, 
                label=f'ROC (AUC = {roc_data["auc"]:.3f})')
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curve')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # PR curve
        pr_data = self.evaluation_results['curves']['pr']
        ax2.plot(pr_data['recall'], pr_data['precision'], linewidth=2,
                label=f'PR (AUC = {pr_data["auc"]:.3f})')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'curves.png', dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Visualizations saved to {self.output_dir}")
    
    def save_all_results(self, detections_df):
        """Save all results and models."""
        print("\n" + "="*60)
        print("💾 SAVING RESULTS")
        print("="*60)
        
        # 1. Save detection results
        detections_df.to_csv(self.output_dir / 'detections.csv', index=False)
        
        # 2. Save comprehensive results JSON
        results = {
            "metadata": {
                "satellite": self.satellite_name,
                "timestamp": self.timestamp,
                "output_name": self.output_name
            },
            "data_statistics": self.data_stats,
            "training_history": self.training_history,
            "evaluation_results": self.evaluation_results,
            "configuration": self.config.__dict__ if self.config else {}
        }
        
        with open(self.output_dir / 'results.json', 'w', encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        
        # 3. Save model
        model_path = self.output_dir / 'model.json'
        self.detector.save_model(str(model_path))
        
        # 4. Save feature names
        if hasattr(self.detector.model, 'feature_names_in_'):
            feature_names = self.detector.model.feature_names_in_.tolist()
            with open(self.output_dir / 'feature_names.json', 'w', encoding="utf-8") as f:
                json.dump(feature_names, f, indent=2)
        
        # 5. Create summary report
        self._create_summary_report()
        
        print(f"✅ All results saved to {self.output_dir}")
    
    def _create_summary_report(self):
        """Create a human-readable summary report."""
        report_path = self.output_dir / 'summary_report.txt'
        
        with open(report_path, 'w', encoding="utf-8") as f:
            f.write("="*70 + "\n")
            f.write(f"MANEUVER DETECTION REPORT - {self.satellite_name}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*70 + "\n\n")
            
            f.write("DATA SUMMARY\n")
            f.write("-"*40 + "\n")
            f.write(f"Total samples: {self.data_stats['n_samples']}\n")
            f.write(f"Features: {self.data_stats['n_features']}\n")
            f.write(f"Positive samples: {self.data_stats['n_positive']} ({100*self.data_stats['positive_ratio']:.1f}%)\n")
            f.write(f"Total maneuvers: {self.data_stats['n_maneuvers']}\n")
            f.write(f"Orbit type: {self.data_stats['orbit_type']}\n")
            f.write(f"Label window: ±{self.data_stats['label_window_days']} days\n\n")
            
            f.write("MODEL CONFIGURATION\n")
            f.write("-"*40 + "\n")
            f.write(f"Detection strategy: {self.config.detection_strategy}\n")
            f.write(f"Target recall: {self.config.target_recall:.0%}\n")
            f.write(f"Min gap between detections: {self.config.min_gap_days} days\n")
            f.write(f"Clustering window: {self.config.cluster_window_days} days\n\n")
            
            if 'tuning' in self.training_history:
                f.write("HYPERPARAMETER TUNING\n")
                f.write("-"*40 + "\n")
                f.write(f"Best F1 score: {self.training_history['tuning']['best_f1']:.4f}\n")
                f.write(f"Best threshold: {self.training_history['tuning']['best_threshold']:.4f}\n")
                f.write("Best parameters:\n")
                for key, value in self.training_history['tuning']['best_params'].items():
                    f.write(f"  • {key}: {value}\n")
                f.write("\n")
            
            f.write("PERFORMANCE METRICS\n")
            f.write("-"*40 + "\n")
            perf = self.evaluation_results['detection_performance']
            f.write(f"Detected events: {perf['true_positives']}/{self.data_stats['n_maneuvers']}\n")
            f.write(f"Recall: {perf['recall']:.1%}\n")
            f.write(f"Precision: {perf['precision']:.1%}\n")
            f.write(f"F1 Score: {perf['f1']:.3f}\n")
            f.write(f"False alarms: {perf['false_positives']}\n")
            f.write(f"Missed events: {perf['false_negatives']}\n\n")
            
            val_metrics = self.evaluation_results['validation_metrics']
            f.write("VALIDATION METRICS\n")
            f.write("-"*40 + "\n")
            f.write(f"ROC AUC: {val_metrics['roc_auc']:.3f}\n")
            f.write(f"PR AUC: {val_metrics['pr_auc']:.3f}\n")
            f.write(f"Threshold used: {val_metrics['threshold_used']:.4f}\n\n")
            
            f.write("="*70 + "\n")
            f.write("END OF REPORT\n")
    
    def run(self, auto_tune=True):
        """Run the complete detection pipeline."""
        print("\n" + "="*70)
        print(f"🚀 STARTING FINAL XGB DETECTION FOR {self.satellite_name}")
        print("="*70)
        
        try:
            # Load and prepare data
            X, y, timestamps, maneuver_times = self.load_and_prepare_data()
            
            # Auto-tune hyperparameters
            best_params = None
            if auto_tune:
                best_params = self.auto_tune_hyperparameters(
                    X, y, 
                    orbit=self.data_stats.get("orbit_type", "GEO")
                )
            
            # Train detector
            val_start = self.train_detector(X, y, timestamps, maneuver_times, best_params)
            
            # Evaluate model
            detections_df, eval_results = self.evaluate_model(
                X, y, timestamps, maneuver_times, val_start
            )
            
            # Create visualizations
            self.create_visualizations(X, timestamps, maneuver_times, detections_df, eval_results)
            
            # Save all results
            self.save_all_results(detections_df)
            
            print("\n" + "="*70)
            print(f"✅ DETECTION COMPLETE FOR {self.satellite_name}")
            print(f"📁 Results saved to: {self.output_dir}")
            print("="*70)
            
            return True
            
        except Exception as e:
            print(f"\n❌ Error during detection: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Final XGBoost Maneuver Detection with Auto-tuning"
    )
    parser.add_argument(
        "satellite",
        nargs="?",
        help="Satellite name (e.g., Fengyun-4A)"
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Skip hyperparameter tuning"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Output directory (default: outputs)"
    )
    
    args = parser.parse_args()
    
    # Get satellite name
    satellite_name = args.satellite
    if not satellite_name:
        satellite_name = input("Enter satellite name (e.g., Fengyun-4A): ").strip()
    
    if not satellite_name:
        print("❌ Satellite name is required!")
        return
    
    # Create and run detector
    detector = FinalXGBDetector(
        satellite_name=satellite_name,
        output_dir=args.output_dir
    )
    
    success = detector.run(auto_tune=not args.no_tune)
    
    if success:
        print(f"\n✨ Success! Check {detector.output_dir} for results.")
    else:
        print("\n❌ Detection failed. Check the error messages above.")


if __name__ == "__main__":
    main()