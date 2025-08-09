from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Tuple
import json
import os  # <-- add this
import numpy as np
import pandas as pd
import xgboost as xgb

from src.features.base_features import generate_base_features
from src.features.drift_features import generate_drift_features
from src.utils.orbit_classification import classify_orbit
from src.utils.metrics import temporal_cluster, find_best_threshold_youden, evaluate_at_threshold
from src.tuning.auto_tune import random_search_xgb

# ... keep DetectorConfig as before ...

class ManeuverXGBDetector:
    # ...

    def build_features(self, df: pd.DataFrame, orbit_hint: Optional[str] = None) -> pd.DataFrame:
        """
        Build base and drift features depending on the orbit class.
        """
        cfg = self.config
        base_cols = self._infer_feature_cols(df) if cfg.feature_cols is None else cfg.feature_cols

        # Decide orbit simply: prefer hint; otherwise use config
        orbit = orbit_hint if orbit_hint else cfg.orbit
        use_drift = cfg.use_drift
        if use_drift is None:
            use_drift = True if orbit == "GEO" else False

        X = generate_base_features(df, cfg.time_col, base_cols)
        if use_drift:
            X = generate_drift_features(X, cfg.time_col, base_cols)
        X = X.dropna().reset_index(drop=True)
        return X

    # rest unchanged
