
"""
Unified XGBoost-based maneuver detector.

This module is self-contained and does not import your legacy code.
It provides a configurable detector with:
  - Base and drift/pressure features
  - GEO/LEO-aware defaults (GEO enables drift features by default)
  - Optional automated hyperparameter tuning
  - Quantile or PR-based thresholding
  - Temporal clustering to reduce duplicate detections
"""
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Tuple
import json
import numpy as np
import pandas as pd
import xgboost as xgb

from ..features.base_features import generate_base_features
from ..features.drift_features import generate_drift_features
from ..utils.orbit_classification import classify_orbit
from ..utils.metrics import temporal_cluster, find_best_threshold_youden, evaluate_at_threshold
from ..tuning.auto_tune import random_search_xgb

@dataclass
class DetectorConfig:
    time_col: str = "timestamp"
    label_col: Optional[str] = "label"
    feature_cols: Optional[List[str]] = None  # If None, infer numeric columns
    orbit: str = "auto"  # "auto" | "GEO" | "LEO"
    use_drift: Optional[bool] = None  # If None, decide from orbit
    threshold_mode: str = "quantile"  # "quantile" | "pr_best"
    threshold_quantile: float = 0.98
    temporal_window: int = 3
    early_stopping_rounds: int = 50
    random_state: int = 42
    auto_tune: bool = False
    n_tune_iter: int = 40

class ManeuverXGBDetector:
    """
    Train and apply an XGBoost classifier to detect maneuver events.
    Requires labeled data for training (0/1 in `label_col`).
    """
    def __init__(self, config: DetectorConfig):
        self.config = config
        self.model: Optional[xgb.XGBClassifier] = None
        self.fitted_threshold_: Optional[float] = None
        self.profile_: Dict = {}

    def _infer_feature_cols(self, df: pd.DataFrame) -> List[str]:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove label and naive timestamp if present
        if self.config.label_col in cols:
            cols.remove(self.config.label_col)
        if self.config.time_col in cols:
            cols.remove(self.config.time_col)
        return cols

    def build_features(self, df: pd.DataFrame, orbit_hint: Optional[str] = None) -> pd.DataFrame:
        """
        Build base and drift features depending on the orbit class.
        """
        cfg = self.config
        if cfg.feature_cols is None:
            base_cols = self._infer_feature_cols(df)
        else:
            base_cols = cfg.feature_cols

        orbit = orbit_hint
        if orbit is None or orbit == "auto":
            orbit = classify_orbit(satellite_name=None) if cfg.orbit == "auto" else cfg.orbit
        use_drift = cfg.use_drift
        if use_drift is None:
            use_drift = True if orbit == "GEO" else False

        X = generate_base_features(df, cfg.time_col, base_cols)
        if use_drift:
            X = generate_drift_features(X, cfg.time_col, base_cols)

        X = X.copy()
        # Drop rows with NaNs introduced by lags/rolling
        X = X.dropna().reset_index(drop=True)
        return X

    def _default_param_grid(self, orbit: str) -> Dict[str, List]:
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

    def fit(self, df: pd.DataFrame, period_minutes: Optional[float] = None, sma_km: Optional[float] = None):
        """
        Fit the detector using labeled data.

        Parameters
        ----------
        df : pd.DataFrame
            Input data containing features and labels.
        period_minutes : Optional[float]
            If provided, helps classify orbit when config.orbit == "auto".
        sma_km : Optional[float]
            If provided, helps classify orbit when config.orbit == "auto".
        """
        cfg = self.config
        if cfg.label_col is None or cfg.label_col not in df.columns:
            raise ValueError("Label column is required for supervised training.")

        if cfg.orbit == "auto":
            orbit = classify_orbit(period_minutes=period_minutes, sma_km=sma_km)
        else:
            orbit = cfg.orbit

        X = self.build_features(df, orbit_hint=orbit)
        y = df.loc[X.index, cfg.label_col].astype(int).values

        if cfg.auto_tune:
            grid = self._default_param_grid(orbit)
            best_params, best_metric, best_thr = random_search_xgb(
                X, y, param_grid=grid, n_iter=cfg.n_tune_iter,
                early_stopping_rounds=cfg.early_stopping_rounds,
                random_state=cfg.random_state, maximize="f1"
            )
            params = best_params
        else:
            params = {
                "max_depth": 4,
                "learning_rate": 0.05 if orbit == "GEO" else 0.1,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "reg_alpha": 0.1 if orbit == "GEO" else 0.0,
                "reg_lambda": 2.0,
                "n_estimators": 600 if orbit == "GEO" else 400,
                "min_child_weight": 3,
            }
            best_thr = None

        clf = xgb.XGBClassifier(
            tree_method="hist",
            enable_categorical=False,
            objective="binary:logistic",
            **params
        )
        # Simple holdout for early stopping
        n = len(X)
        split = int(n * 0.8)
        clf.fit(
            X.iloc[:split], y[:split],
            eval_set=[(X.iloc[split:], y[split:])],
            eval_metric="aucpr",
            verbose=False,
            early_stopping_rounds=cfg.early_stopping_rounds
        )
        self.model = clf

        scores = clf.predict_proba(X.iloc[split:])[:, 1]
        if cfg.threshold_mode == "pr_best":
            thr = find_best_threshold_youden(y[split:], scores)
        else:
            thr = float(np.quantile(scores, cfg.threshold_quantile))
        if best_thr is not None:
            thr = best_thr  # prefer tuned threshold if tuning was run
        self.fitted_threshold_ = thr

        self.profile_ = {
            "orbit": orbit,
            "params": params,
            "threshold_mode": cfg.threshold_mode,
            "threshold": float(thr),
            "early_stopping_rounds": cfg.early_stopping_rounds,
        }
        return self

    def predict_scores(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict anomaly probabilities for the given data.
        `fit` must be called first.
        """
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        X = self.build_features(df, orbit_hint=self.profile_.get("orbit", "GEO"))
        scores = self.model.predict_proba(X)[:, 1]
        return scores

    def detect(self, df: pd.DataFrame, timestamps: pd.Series) -> pd.DataFrame:
        """
        Produce clustered detections with timestamps and scores.
        """
        scores = self.predict_scores(df)
        thr = self.fitted_threshold_
        preds = (scores >= thr).astype(int)
        ts = timestamps.iloc[-len(scores):].reset_index(drop=True)
        det = pd.DataFrame({"timestamp": ts, "score": scores, "pred": preds})
        # Cluster positives
        pos = det[det["pred"] == 1]
        if len(pos) == 0:
            return det.assign(cluster_id=pd.Series(dtype=int))
        clustered = temporal_cluster(
            timestamps=pd.Series(np.arange(len(pos))),  # index distance clustering
            scores=pos["score"],
            window=self.config.temporal_window
        )
        # Map cluster ids back to original indices
        pos = pos.reset_index(drop=True)
        pos = pos.loc[clustered.index]
        pos = pos.assign(cluster_id=clustered["cluster_id"].values)
        out = det.merge(pos[["timestamp", "cluster_id"]], on="timestamp", how="left")
        return out

    def save_run_artifacts(self, save_dir: str):
        """
        Save model parameters and threshold information for reproducibility.
        """
        os.makedirs(save_dir, exist_ok=True)
        # Save config/profile
        with open(f"{save_dir}/config_used.json", "w", encoding="utf-8") as f:
            json.dump({"config": asdict(self.config), "profile": self.profile_}, f, ensure_ascii=False, indent=2)
