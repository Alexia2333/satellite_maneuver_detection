import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import json
import pandas as pd
import numpy as np

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import our modules
from src.data.loader import SatelliteDataLoader
from src.models.arima_detector import ARIMADetector
from src.data.feature_engineer import EnhancedSatelliteFeatureEngineer
from src.models.enhanced_xgb_detector import EnhancedManeuverXGBDetector, EnhancedDetectorConfig
from src.tuning.auto_tune_arima import find_best_arima_order
from src.utils.metrics import evaluate_detection

class FinalHybridDetector:
    """
    An integrated workflow for the ARIMA pre-filtered XGBoost detection model.
    """
    def __init__(self, satellite_name: str):
        self.satellite_name = satellite_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path("outputs") / f"{satellite_name.replace(' ', '_')}_Hybrid_{self.timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"🛰️ Initializing Hybrid Detector for {satellite_name}. Outputs will be in {self.output_dir}")

    def run(self):
        # 1. Load Data
        loader = SatelliteDataLoader(data_dir="data")
        tle_data, maneuver_times = loader.load_satellite_data(self.satellite_name)
        
        element = "mean_motion"
        ts_data = tle_data[['epoch', element]].set_index('epoch').resample('D').median().interpolate(method='time').dropna()[element]

        labels = pd.Series(0, index=ts_data.index)
        for m_time in maneuver_times:
            mask = (ts_data.index >= m_time - timedelta(days=1)) & (ts_data.index <= m_time + timedelta(days=1))
            labels[mask] = 1
        
        train_size = int(len(ts_data) * 0.7)
        train_ts = ts_data.iloc[:train_size]
        train_labels = labels.iloc[:train_size]

        # --- STAGE 1: ARIMA as a Signal Purifier ---
        print("\n" + "="*60)
        print("STAGE 1: ARIMA Pre-filtering")
        print("="*60)
        
        pure_train_data = train_ts.copy()
        pure_train_data[train_labels == 1] = np.nan
        pure_train_data = pure_train_data.interpolate(method='time')
        
        print(f"Training ARIMA on {len(pure_train_data)} 'normal' (interpolated) data points...")

        best_order = find_best_arima_order(pure_train_data)
        arima_detector = ARIMADetector(p=best_order[0], d=best_order[1], q=best_order[2])
        arima_detector.fit(pure_train_data)

        full_predictions = arima_detector.model_fit.predict(start=ts_data.index[0], end=ts_data.index[-1])
        residual_series = ts_data - full_predictions
        
        print("✅ ARIMA pre-filtering complete. Residual series generated.")

        # --- STAGE 2: XGBoost as a Residual Classifier ---
        print("\n" + "="*60)
        print("STAGE 2: XGBoost on Residuals")
        print("="*60)
        
        # 【CORE FIX】Create the DataFrame correctly, preserving the DatetimeIndex
        residual_df = pd.DataFrame({'mean_motion': residual_series})
        residual_df['epoch'] = residual_df.index
        
        # Feature Engineering on the residual signal
        eng = EnhancedSatelliteFeatureEngineer(
            target_column="mean_motion", additional_columns=[],
            satellite_type="GEO" if "fengyun" in self.satellite_name.lower() else "LEO"
        ).fit(residual_df, satellite_name=self.satellite_name)
        feat_df = eng.transform(residual_df)
        
        # Align labels with the feature-engineered DataFrame's index
        aligned_labels = labels.reindex(feat_df.index).dropna()
        feat_df = feat_df.reindex(aligned_labels.index)
        
        X = feat_df.select_dtypes(include=[np.number])
        y = aligned_labels.values
        timestamps = pd.Series(feat_df.index, name="epoch")

        train_size_xgb = int(len(X) * 0.7)
        X_train, X_test = X.iloc[:train_size_xgb], X.iloc[train_size_xgb:]
        y_train, y_test = y[:train_size_xgb], y[train_size_xgb:]
        
        config = EnhancedDetectorConfig(
            detection_strategy="top_n", 
            expected_events=len(maneuver_times),
            scale_pos_weight=10
        )
        xgb_detector = EnhancedManeuverXGBDetector(config)
        
        val_start_xgb = int(0.8 * len(X_train))
        xgb_detector.fit_from_features(X_train, y_train, orbit=config.orbit, val_start=val_start_xgb)
        
        detections_df = xgb_detector.detect_from_features(X, timestamps)
        
        test_maneuvers = [m for m in maneuver_times if timestamps.iloc[train_size_xgb:].min() <= m <= timestamps.iloc[train_size_xgb:].max()]
        eval_results = xgb_detector.evaluate_detections(detections_df.iloc[train_size_xgb:], test_maneuvers, window_min=1440)
        
        print("\n📊 FINAL HYBRID MODEL PERFORMANCE:")
        print(f"  - F1 Score:  {eval_results['f1']:.3f}")
        print(f"  - Precision: {eval_results['precision']:.2%}")
        print(f"  - Recall:    {eval_results['recall']:.2%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Hybrid ARIMA-XGBoost Detector")
    parser.add_argument("satellite", help="Satellite name (e.g., Jason-1)")
    args = parser.parse_args()
    
    detector = FinalHybridDetector(satellite_name=args.satellite)
    detector.run()