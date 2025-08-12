"""
Unified XGBoost-based maneuver detector (external-features friendly).

Supports:
  - Fit from raw df (legacy)
  - Fit from precomputed features (recommended)
"""
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

# Keep these imports for backward compatibility (raw-df path)
from src.features.base_features import generate_base_features
from src.features.drift_features import generate_drift_features
from src.utils.orbit_classification import classify_orbit
from src.utils.metrics import temporal_cluster, find_best_threshold_youden
from src.tuning.auto_tune import random_search_xgb


@dataclass
class DetectorConfig:
    time_col: str = "epoch"
    label_col: Optional[str] = "label"
    feature_cols: Optional[List[str]] = None
    orbit: str = "auto"                     # "auto" | "GEO" | "LEO"
    use_drift: Optional[bool] = None        # raw-df path only
    threshold_mode: str = "quantile"        # "quantile" | "pr_best"
    threshold_quantile: float = 0.98
    temporal_window: int = 3
    early_stopping_rounds: int = 50
    random_state: int = 42
    auto_tune: bool = False
    n_tune_iter: int = 40
    scale_pos_weight: int = 20


class ManeuverXGBDetector:
    """
    XGBoost maneuver detector.

    Two usage paths:
      A) Raw-df path: call fit(raw_df, ...)  -> detector builds features internally.
      B) External-features path: call fit_from_features(X, y).
    """

    def __init__(self, config: DetectorConfig):
        self.config = config
        self.model: Optional[xgb.XGBClassifier] = None
        self.fitted_threshold_: Optional[float] = None
        self.profile_: Dict = {}

    # ---------- Raw-df helpers (kept for backward compatibility) ----------
    def _infer_feature_cols(self, df: pd.DataFrame) -> List[str]:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if self.config.label_col in cols:
            cols.remove(self.config.label_col)
        if self.config.time_col in cols:
            cols.remove(self.config.time_col)
        return cols

    def build_features(self, df: pd.DataFrame, orbit_hint: Optional[str] = None) -> pd.DataFrame:
        cfg = self.config
        base_cols = self._infer_feature_cols(df) if cfg.feature_cols is None else cfg.feature_cols
        orbit = orbit_hint if orbit_hint else cfg.orbit
        use_drift = cfg.use_drift
        if use_drift is None:
            use_drift = True if orbit == "GEO" else False
        X = generate_base_features(df, cfg.time_col, base_cols)
        if use_drift:
            X = generate_drift_features(X, cfg.time_col, base_cols)
        return X.dropna().reset_index(drop=True)

    # ---------- External-features path ----------
    def fit_from_features(self, X: pd.DataFrame, y: np.ndarray, orbit: str = "GEO", val_start: Optional[int] = None):
        """
        Fit using a pre-engineered feature matrix X and labels y.
        """
        cfg = self.config

        # Guard: ensure X is numeric-only for xgboost 3.x
        X = X.select_dtypes(include=[np.number]).copy()

        # Auto tuning or defaults
        best_thr = None
        if cfg.auto_tune:
            grid = self._default_param_grid(orbit)
            best_params, best_metric, best_thr = random_search_xgb(
                X, y,
                param_grid=grid,
                n_iter=cfg.n_tune_iter,
                early_stopping_rounds=cfg.early_stopping_rounds,
                random_state=cfg.random_state,
                maximize="f1",
            )
            params = best_params
        else:
            if hasattr(self, 'hyperparams') and getattr(self, 'hyperparams'):
                print("   -> Using hyperparameters from config file.")
                params = self.hyperparams
            else:
                print("   -> Using default hyperparameters.")
                params = {
                    "max_depth": 4,
                    "learning_rate": 0.05 if orbit == "GEO" else 0.1,
                    "subsample": 0.9,
                    "colsample_bytree": 0.9,
                    "reg_alpha": 0.1 if orbit == "GEO" else 0.0,
                    "reg_lambda": 2.0,
                    "n_estimators": 600 if orbit == "GEO" else 400,
                    "min_child_weight": 3,
                    "scale_pos_weight": self.config.scale_pos_weight,
                }

        if 'eval_metric' not in params:
            params['eval_metric'] = 'aucpr'
        params.setdefault('scale_pos_weight', self.config.scale_pos_weight)

        # Holdout for early stopping
        n = len(X)
        split = max(1, int(n * 0.8)) if val_start is None else int(val_start)

        clf = xgb.XGBClassifier(
            tree_method="hist",
            enable_categorical=False,
            objective="binary:logistic",
            **params,
        )

        # Compatibility early stopping across xgboost versions
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

        # Print class balance for sanity
        print("train pos/neg:", int(y[:split].sum()), int(len(y[:split]) - y[:split].sum()))
        print("valid pos/neg:", int(y[split:].sum()), int(len(y[split:]) - y[split:].sum()))

        clf.fit(
            X.iloc[:split], y[:split],
            **fit_kwargs
        )
        self.model = clf
        self._val_start_ = split

        # Threshold
        scores = clf.predict_proba(X.iloc[split:])[:, 1]
        if cfg.threshold_mode == "pr_best":
            thr = find_best_threshold_youden(y[split:], scores)
        else:
            thr = float(np.quantile(scores, cfg.threshold_quantile))
        if best_thr is not None:
            thr = best_thr
        self.fitted_threshold_ = thr

        # Save profile
        self.profile_ = {
            "orbit": orbit,
            "params": clf.get_params(),
            "threshold_mode": cfg.threshold_mode,
            "threshold": float(thr),
            "early_stopping_rounds": cfg.early_stopping_rounds,
        }
        return self

    def predict_scores_from_features(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        return self.model.predict_proba(X)[:, 1]

    # ---------- Detection ----------
    def detect_from_features(self, X: pd.DataFrame, timestamps: pd.Series) -> pd.DataFrame:
        scores = self.predict_scores_from_features(X)
        thr = self.fitted_threshold_
        preds = (scores >= thr).astype(int)
        ts = timestamps.iloc[-len(scores):].reset_index(drop=True)
        det = pd.DataFrame({"timestamp": ts, "score": scores, "pred": preds})

        pos = det[det["pred"] == 1]
        if len(pos) == 0:
            det["cluster_id"] = np.nan
            return det

        clustered = temporal_cluster(
            timestamps=pd.Series(np.arange(len(pos))),
            scores=pos["score"],
            window=self.config.temporal_window,
        )
        pos = pos.reset_index(drop=True).loc[clustered.index]
        pos = pos.assign(cluster_id=clustered["cluster_id"].values)
        out = det.merge(pos[["timestamp", "cluster_id"]], on="timestamp", how="left")
        return out

    def plot_timeline_sliding_window(self, 
                                    X: pd.DataFrame, 
                                    timestamps: pd.Series, 
                                    maneuver_times: List[pd.Timestamp], 
                                    window_days: int = 2, 
                                    save_path: Optional[str] = None, 
                                    show: bool = False) -> Dict:
        """Plot scores and a time-based rolling-max (sliding window) together with maneuver windows and predicted clusters."""
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        
        # Data length verification
        if len(timestamps) < len(X):
            raise ValueError(f"timestamps length ({len(timestamps)}) < X length ({len(X)})")
        
        # Data length verification
        ts = pd.to_datetime(timestamps.iloc[-len(X):]).reset_index(drop=True)
        scores = self.predict_scores_from_features(X)
        s = pd.Series(scores, index=ts).sort_index()  # Ensure the time series is in order
        
        # Calculate the maximum value of the sliding window
        agg = s.rolling(f"{int(window_days)}D", min_periods=1).max()

        # Detect anomalies and clusters
        det_df = self.detect_from_features(X, timestamps=ts)
        cluster_centers = (det_df.dropna(subset=['cluster_id'])
                               .sort_values(['cluster_id','score'], ascending=[True, False])
                               .drop_duplicates(subset=['cluster_id']))
        cluster_times = list(cluster_centers['timestamp'])

        # Drawing
        half = pd.to_timedelta(window_days, unit='D')
        plt.figure(figsize=(12, 6))
        
        plt.plot(ts, s.values, linewidth=1, label='Score')
        plt.plot(ts, agg.values, linewidth=1, linestyle='--', 
                 label=f'Rolling max ({window_days}D)')
        
        if hasattr(self, 'fitted_threshold_') and self.fitted_threshold_ is not None:
            plt.axhline(self.fitted_threshold_, linestyle=':', alpha=0.8, label='Threshold')
        
        # Add maneuvering time window
        for ev in maneuver_times:
            plt.axvspan(ev - half, ev + half, alpha=0.1, color='red', label='Maneuver Window' if ev == maneuver_times[0] else "")
        
        # Add clustering timeline
        for ct in cluster_times:
            plt.axvline(ct, linestyle='--', alpha=0.7, color='green', 
                       label='Cluster Center' if ct == cluster_times[0] else "")
        
        plt.title('Maneuver Detection – Sliding Window View')
        plt.xlabel('Time')
        plt.ylabel('Anomaly Score')
        plt.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=180, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
        
        return {"cluster_times": cluster_times, "window_days": window_days}

    def _default_param_grid(self, orbit: str) -> Dict[str, List]:
        """Returns the default parameter grid based on the track type."""
        if orbit == "GEO":
            return {
                "max_depth": [3, 4, 5, 6],
                "learning_rate": [0.02, 0.05, 0.1],
                "subsample": [0.7, 0.9, 1.0],
                "colsample_bytree": [0.7, 0.9, 1.0],
                "reg_alpha": [0.0, 0.1, 0.5],
                "reg_lambda": [1.0, 2.0, 5.0],
                "n_estimators": [300, 500, 800, 1200],
                "min_child_weight": [1, 3, 5],
            }
        else:
            return {
                "max_depth": [3, 4, 5],
                "learning_rate": [0.05, 0.1, 0.2],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
                "reg_alpha": [0.0, 0.1],
                "reg_lambda": [1.0, 2.0],
                "n_estimators": [200, 400, 800],
                "min_child_weight": [1, 3],
            }

    def save_run_artifacts(self, save_dir: str) -> None:
        """Save the running results and configuration"""
        os.makedirs(save_dir, exist_ok=True)
        
        config_data = {
            "config": asdict(self.config), 
            "profile": getattr(self, 'profile_', None)
        }
        
        config_path = os.path.join(save_dir, "config_used.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)
        
        print(f"Artifacts saved to: {save_dir}")