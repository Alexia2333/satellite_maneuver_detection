"""
Enhanced XGBoost-based maneuver detector with multiple detection strategies.

This detector extends the basic XGBoost detector with:
- Top-N anomaly detection
- Local peak detection
- Statistical outlier detection
- Smart clustering strategies
"""

from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Tuple, Literal
import json
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')


@dataclass
class EnhancedDetectorConfig:
    """Configuration for enhanced XGBoost detector."""
    # Basic settings (compatible with original DetectorConfig)
    time_col: str = "epoch"
    label_col: Optional[str] = "label"
    feature_cols: Optional[List[str]] = None
    orbit: str = "auto"
    use_drift: Optional[bool] = None
    threshold_mode: str = "quantile"
    threshold_quantile: float = 0.98
    temporal_window: int = 3
    early_stopping_rounds: int = 50
    random_state: int = 42
    auto_tune: bool = False
    n_tune_iter: int = 40
    scale_pos_weight: int = 5  # Lower for better recall
    
    # Enhanced detection settings
    detection_strategy: Literal["threshold", "top_n", "peaks", "combined"] = "top_n"
    expected_events: Optional[int] = None  # Expected number of maneuvers
    min_gap_days: float = 5.0  # Minimum days between detections
    cluster_window_days: float = 7.0  # Window for clustering nearby detections
    
    # Top-N strategy parameters
    top_n_multiplier: float = 1.2  # Detect N * multiplier events
    
    # Peak detection parameters
    peak_window_days: int = 10
    peak_prominence: float = 0.3
    
    # Outlier detection parameters
    outlier_percentile: float = 90
    
    # Target performance
    target_recall: float = 0.6
    
    # Model hyperparameters for sensitive detection
    sensitive_hyperparams: Dict = field(default_factory=lambda: {
        "max_depth": 6,
        "min_child_weight": 1,
        "reg_alpha": 0.01,
        "reg_lambda": 0.5
    })


class EnhancedManeuverXGBDetector:
    """
    Enhanced XGBoost maneuver detector with multiple detection strategies.
    
    Compatible with the original ManeuverXGBDetector interface while providing
    advanced detection capabilities for better recall.
    """
    
    def __init__(self, config: EnhancedDetectorConfig):
        self.config = config
        self.model: Optional[xgb.XGBClassifier] = None
        self.fitted_threshold_: Optional[float] = None
        self.profile_: Dict = {}
        self._detection_stats_: Dict = {}
    
    def fit_from_features(self, X: pd.DataFrame, y: np.ndarray, 
                          orbit: str = "GEO", val_start: Optional[int] = None):
        """
        Fit the XGBoost model using pre-engineered features.
        Compatible with original interface.
        """
        cfg = self.config
        
        # Ensure X is numeric-only
        X = X.select_dtypes(include=[np.number]).copy()
        
        # Prepare hyperparameters
        if hasattr(self, 'hyperparams') and self.hyperparams:
            params = self.hyperparams.copy()
        else:
            # Use sensitive hyperparameters for better recall
            params = {
                "max_depth": cfg.sensitive_hyperparams.get("max_depth", 6),
                "learning_rate": 0.05 if orbit == "GEO" else 0.1,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "reg_alpha": cfg.sensitive_hyperparams.get("reg_alpha", 0.01),
                "reg_lambda": cfg.sensitive_hyperparams.get("reg_lambda", 0.5),
                "n_estimators": 800 if orbit == "GEO" else 600,
                "min_child_weight": cfg.sensitive_hyperparams.get("min_child_weight", 1),
                "scale_pos_weight": cfg.scale_pos_weight,
            }
        
        if 'eval_metric' not in params:
            params['eval_metric'] = 'aucpr'
        
        # Train-validation split
        n = len(X)
        split = max(1, int(n * 0.8)) if val_start is None else int(val_start)
        
        # Create and train classifier
        clf = xgb.XGBClassifier(
            tree_method="hist",
            enable_categorical=False,
            objective="binary:logistic",
            **params
        )
        
        # Handle early stopping
        import inspect
        fit_sig = inspect.signature(xgb.XGBClassifier().fit)
        supports_callbacks = 'callbacks' in fit_sig.parameters
        supports_esr = 'early_stopping_rounds' in fit_sig.parameters
        
        fit_kwargs = dict(
            eval_set=[(X.iloc[split:], y[split:])],
            verbose=False,
        )
        
        if supports_callbacks:
            from xgboost.callback import EarlyStopping
            fit_kwargs['callbacks'] = [EarlyStopping(rounds=cfg.early_stopping_rounds, save_best=True)]
        elif supports_esr:
            fit_kwargs['early_stopping_rounds'] = cfg.early_stopping_rounds
        
        # Print class balance
        print(f"Training set - Positive: {y[:split].sum()}, Negative: {len(y[:split]) - y[:split].sum()}")
        print(f"Validation set - Positive: {y[split:].sum()}, Negative: {len(y[split:]) - y[split:].sum()}")
        
        # Fit model
        clf.fit(X.iloc[:split], y[:split], **fit_kwargs)
        self.model = clf
        self._val_start_ = split
        
        # Determine threshold based on strategy
        val_scores = clf.predict_proba(X.iloc[split:])[:, 1]
        
        if cfg.detection_strategy in ["threshold", "combined"]:
            # Find threshold for target recall
            self.fitted_threshold_ = self._find_threshold_for_recall(
                y[split:], val_scores, cfg.target_recall
            )
        else:
            # For top_n or peaks, threshold is less important
            self.fitted_threshold_ = float(np.percentile(val_scores, 50))
        
        # Store training profile
        self.profile_ = {
            "orbit": orbit,
            "params": clf.get_params(),
            "threshold_mode": cfg.threshold_mode,
            "threshold": float(self.fitted_threshold_),
            "detection_strategy": cfg.detection_strategy,
            "early_stopping_rounds": cfg.early_stopping_rounds,
        }
        
        return self
    
    def predict_scores_from_features(self, X: pd.DataFrame) -> np.ndarray:
        """Get anomaly scores for features."""
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        X = X.select_dtypes(include=[np.number])
        return self.model.predict_proba(X)[:, 1]
    
    def detect_from_features(self, X: pd.DataFrame, timestamps: pd.Series) -> pd.DataFrame:
        """
        Detect maneuvers using the configured strategy.
        Returns DataFrame with columns: timestamp, score, pred, cluster_id
        """
        scores = self.predict_scores_from_features(X)
        ts = timestamps.iloc[-len(scores):].reset_index(drop=True)
        
        # Apply detection strategy
        if self.config.detection_strategy == "top_n":
            detections = self._detect_top_n(scores, ts)
        elif self.config.detection_strategy == "peaks":
            detections = self._detect_peaks(scores, ts)
        elif self.config.detection_strategy == "combined":
            detections = self._detect_combined(scores, ts)
        else:  # threshold
            detections = self._detect_threshold(scores, ts)
        
        # Cluster detections
        clustered = self._cluster_detections(detections, ts, scores)
        
        # Create output DataFrame
        det_df = pd.DataFrame({
            "timestamp": ts,
            "score": scores,
            "pred": 0
        })
        
        # Mark detections
        det_df.loc[list(detections), "pred"] = 1
        
        # Add cluster IDs
        det_df["cluster_id"] = np.nan
        for cluster_id, indices in enumerate(clustered):
            for idx in indices:
                det_df.loc[idx, "cluster_id"] = cluster_id
        
        # Store detection statistics
        self._detection_stats_ = {
            "n_detections": len(detections),
            "n_clusters": len(clustered),
            "strategy": self.config.detection_strategy
        }
        
        return det_df
    
    def _detect_top_n(self, scores: np.ndarray, timestamps: pd.Series) -> set:
        """Detect top N anomalies with minimum gap."""
        n_events = self.config.expected_events or 50
        n_target = int(n_events * self.config.top_n_multiplier)
        
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(timestamps),
            'score': scores,
            'index': range(len(scores))
        }).sort_values('score', ascending=False)
        
        selected = []
        selected_times = []
        min_gap = pd.Timedelta(days=self.config.min_gap_days)
        
        for _, row in df.iterrows():
            current_time = row['timestamp']
            
            # Check minimum gap from existing detections
            if not selected_times or all(
                abs(current_time - t) >= min_gap for t in selected_times
            ):
                selected.append(row['index'])
                selected_times.append(current_time)
                
                if len(selected) >= n_target:
                    break
        
        return set(selected)
    
    def _detect_peaks(self, scores: np.ndarray, timestamps: pd.Series) -> set:
        """Detect local peaks in the score time series."""
        # Normalize scores
        scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        
        # Find peaks
        peaks, properties = find_peaks(
            scores_norm,
            prominence=self.config.peak_prominence,
            distance=2
        )
        
        # Filter by local contrast
        filtered_peaks = []
        window = self.config.peak_window_days
        
        for peak in peaks:
            start = max(0, peak - window)
            end = min(len(scores), peak + window + 1)
            local_scores = scores_norm[start:end]
            
            if len(local_scores) > 1:
                local_baseline = np.percentile(local_scores, 25)
                if scores_norm[peak] > local_baseline + self.config.peak_prominence:
                    filtered_peaks.append(peak)
        
        return set(filtered_peaks)
    
    def _detect_threshold(self, scores: np.ndarray, timestamps: pd.Series) -> set:
        """Simple threshold-based detection."""
        threshold = self.fitted_threshold_ or np.percentile(scores, self.config.threshold_quantile * 100)
        return set(np.where(scores >= threshold)[0])
    
    def _detect_combined(self, scores: np.ndarray, timestamps: pd.Series) -> set:
        """Combined detection using multiple strategies."""
        # Top-N detection
        top_n = self._detect_top_n(scores, timestamps)
        
        # Local peaks
        peaks = self._detect_peaks(scores, timestamps)
        
        # Statistical outliers
        outlier_threshold = np.percentile(scores, self.config.outlier_percentile)
        outliers = set(np.where(scores > outlier_threshold)[0])
        
        # Significant changes
        score_diff = np.abs(np.diff(scores, prepend=scores[0]))
        diff_threshold = np.percentile(score_diff, 95)
        changes = set(np.where(score_diff > diff_threshold)[0])
        
        # Combine all
        return top_n | peaks | outliers | changes
    
    def _cluster_detections(self, detections: set, timestamps: pd.Series, 
                           scores: np.ndarray) -> List[List[int]]:
        """Cluster nearby detections."""
        if not detections:
            return []
        
        # Sort by timestamp
        det_list = sorted(detections, key=lambda i: timestamps.iloc[i])
        
        clusters = []
        current_cluster = [det_list[0]]
        max_gap = pd.Timedelta(days=self.config.cluster_window_days)
        
        for i in range(1, len(det_list)):
            time_diff = timestamps.iloc[det_list[i]] - timestamps.iloc[det_list[i-1]]
            
            if time_diff <= max_gap:
                current_cluster.append(det_list[i])
            else:
                # Save current cluster (keep best detection)
                best_idx = max(current_cluster, key=lambda idx: scores[idx])
                clusters.append([best_idx])
                current_cluster = [det_list[i]]
        
        # Don't forget the last cluster
        if current_cluster:
            best_idx = max(current_cluster, key=lambda idx: scores[idx])
            clusters.append([best_idx])
        
        return clusters
    
    def _find_threshold_for_recall(self, y_true: np.ndarray, scores: np.ndarray, 
                                   target_recall: float) -> float:
        """Find threshold that achieves target recall."""
        from sklearn.metrics import recall_score
        
        unique_scores = np.unique(scores)
        if len(unique_scores) > 1000:
            thresholds = np.percentile(unique_scores, np.linspace(0, 100, 1000))
        else:
            thresholds = unique_scores
        
        best_threshold = thresholds[0]
        best_recall = 0
        
        for threshold in thresholds:
            preds = (scores >= threshold).astype(int)
            if preds.sum() == 0:
                continue
            recall = recall_score(y_true, preds)
            
            if recall >= target_recall:
                best_threshold = threshold
                break
            elif recall > best_recall:
                best_threshold = threshold
                best_recall = recall
        
        return best_threshold
    
    def evaluate_detections(self, detections_df: pd.DataFrame, 
                           maneuver_times: List, 
                           window_min: int) -> Dict:
        """
        Evaluate detection performance against ground truth.
        
        Args:
            detections_df: Output from detect_from_features
            maneuver_times: List of true maneuver timestamps
            window_min: Window in minutes for matching
        
        Returns:
            Dictionary with performance metrics
        """
        # Get cluster centers (one per cluster)
        clusters = detections_df.dropna(subset=['cluster_id'])
        if len(clusters) == 0:
            return {
                'tp': 0, 'fp': 0, 'fn': len(maneuver_times),
                'precision': 0.0, 'recall': 0.0, 'f1': 0.0
            }
        
        cluster_times = []
        for cluster_id in clusters['cluster_id'].unique():
            cluster_data = clusters[clusters['cluster_id'] == cluster_id]
            # Get detection with highest score in cluster
            best_idx = cluster_data['score'].idxmax()
            cluster_times.append(cluster_data.loc[best_idx, 'timestamp'])
        
        # Match detections to events
        half = pd.Timedelta(minutes=window_min)
        detected_events = []
        false_alarms = []
        
        for ct in cluster_times:
            matched = any(
                (pd.Timestamp(ct) >= ev - half) and (pd.Timestamp(ct) <= ev + half)
                for ev in maneuver_times
            )
            if matched:
                # Find which event was detected
                for ev in maneuver_times:
                    if (pd.Timestamp(ct) >= ev - half) and (pd.Timestamp(ct) <= ev + half):
                        if ev not in detected_events:
                            detected_events.append(ev)
                        break
            else:
                false_alarms.append(ct)
        
        missed_events = [ev for ev in maneuver_times if ev not in detected_events]
        
        tp = len(detected_events)
        fp = len(false_alarms)
        fn = len(missed_events)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / len(maneuver_times) if maneuver_times else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'detected_events': detected_events,
            'missed_events': missed_events,
            'false_alarms': false_alarms
        }
    
    def save_model(self, filepath: str):
        """Save the trained model and configuration."""
        if self.model is None:
            raise RuntimeError("No model to save. Train the model first.")
        
        # Save XGBoost model
        model_path = filepath.replace('.json', '_model.json')
        self.model.save_model(model_path)
        
        # Save configuration and metadata
        metadata = {
            'config': asdict(self.config),
            'profile': self.profile_,
            'threshold': self.fitted_threshold_,
            'detection_stats': self._detection_stats_,
            'model_path': os.path.basename(model_path)
        }
        
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a saved model and configuration."""
        with open(filepath, 'r') as f:
            metadata = json.load(f)
        
        # Load configuration
        self.config = EnhancedDetectorConfig(**metadata['config'])
        self.profile_ = metadata['profile']
        self.fitted_threshold_ = metadata['threshold']
        self._detection_stats_ = metadata.get('detection_stats', {})
        
        # Load XGBoost model
        model_path = filepath.replace('.json', '_model.json')
        if not os.path.exists(model_path):
            model_path = os.path.join(os.path.dirname(filepath), metadata['model_path'])
        
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        
        print(f"Model loaded from {filepath}")


def create_detector_from_config(config_path: Optional[str] = None, 
                               **kwargs) -> EnhancedManeuverXGBDetector:
    """
    Factory function to create detector with configuration.
    
    Args:
        config_path: Path to JSON configuration file
        **kwargs: Override configuration parameters
    
    Returns:
        Configured detector instance
    """
    config_dict = {}
    
    # Load from file if provided
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
    
    # Override with kwargs
    config_dict.update(kwargs)
    
    # Create configuration
    config = EnhancedDetectorConfig(**config_dict)
    
    # Create and return detector
    return EnhancedManeuverXGBDetector(config)