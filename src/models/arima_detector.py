# src/models/arima_detector.py
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field
import warnings

warnings.filterwarnings('ignore')

@dataclass
class DetectorConfig:
    order: Tuple[int, int, int] = (5, 1, 1)
    window_size: int = 50
    threshold_factor: float = 3.0
    min_gap_days: int = 3
    fallback_orders: List[Tuple[int, int, int]] = field(default_factory=lambda: [
        (5, 1, 1), (3, 1, 1), (1, 1, 1), (1, 1, 0)
    ])
    use_drift_awareness: bool = True       # Whether to enable the drift detection function
    drift_long_window: int = 30            # Window used to calculate the long-term "normal" center
    drift_short_window: int = 14           # Window for calculating accumulated drift pressure
    pressure_reduction_factor: float = 0.4 # The maximum reduction in the threshold value caused by drift pressure (e.g. 0.4 means a maximum reduction of 40%)
    min_adjustment_factor: float = 0.5     # The lower limit of the value adjustment factor to prevent the threshold from being too low低

class EnhancedARIMADetector:
    def __init__(self, config: Optional[DetectorConfig] = None):
        self.cfg = config or DetectorConfig()
        self.scaler = StandardScaler()
        self.train_series: Optional[pd.Series] = None
        self.baseline_resid_std: float = 1.0
        # --- Baseline for normalizing drift pressure ---
        self.pressure_baseline: float = 1.0

    def fit(self, train_series: pd.Series) -> "EnhancedARIMADetector":
        """
        'Training the detector now also includes calculating the baseline for drift pressure.
        """
        if train_series.empty:
            raise ValueError("Training data cannot be empty.")
        self.train_series = train_series.copy()
        self.scaler.fit(self.train_series.values.reshape(-1, 1))
        
        if self.cfg.use_drift_awareness:
            scaled_train_vals = self.scaler.transform(self.train_series.values.reshape(-1, 1)).flatten()
            short_term_drift = pd.Series(scaled_train_vals).diff().abs()
            cumulative_drift = short_term_drift.rolling(window=self.cfg.drift_short_window).sum().fillna(0)
            # Use the 75th percentile as a benchmark for "high stress," which is more robust than the mean.
            self.pressure_baseline = np.quantile(cumulative_drift[cumulative_drift > 0], 0.75)
            print(f"  Drift-Awareness enabled. Pressure baseline calculated: {self.pressure_baseline:.4f}")
        
        warmup_data = self.train_series.tail(self.cfg.window_size)
        if len(warmup_data) < 20: warmup_data = self.train_series
        scaled_warmup_values = self.scaler.transform(warmup_data.values.reshape(-1, 1))
        
        try:
            warmup_model = ARIMA(scaled_warmup_values, order=self.cfg.order).fit(method='css')
            self.baseline_resid_std = np.std(warmup_model.resid)
        except Exception:
            self.baseline_resid_std = 1.0
        print(f"✅ Detector 'fit' complete. Scaler is ready. Baseline residual std (scaled): {self.baseline_resid_std:.4f}")
        return self

    def _fit_on_window(self, window_data: np.ndarray) -> Optional[any]:
        if len(window_data) < 20 or np.std(window_data) < 1e-9: return None
        orders_to_try = [self.cfg.order] + self.cfg.fallback_orders
        for order in list(dict.fromkeys(orders_to_try)):
            try:
                model = ARIMA(window_data, order=order, trend='n')
                return model.fit(method='css')
            except Exception:
                continue
        return None

    def detect(self, test_series: pd.Series) -> pd.DataFrame:
        if self.train_series is None: raise RuntimeError("Detector has not been 'fit'. Call fit() first.")

        full_series = pd.concat([self.train_series, test_series])
        scaled_values = self.scaler.transform(full_series.values.reshape(-1, 1)).flatten()
        scaled_series = pd.Series(scaled_values, index=full_series.index)
        
        # --- Pre-calculate drift pressure and adjustment factors for the entire sequence ---
        adjustment_factors = pd.Series(1.0, index=full_series.index) 
        if self.cfg.use_drift_awareness and self.pressure_baseline > 0:
            short_term_drift = scaled_series.diff().abs()
            cumulative_drift = short_term_drift.rolling(window=self.cfg.drift_short_window).sum().fillna(0)
            
            # The greater the pressure, the smaller the adjustment factor
            normalized_pressure = cumulative_drift / self.pressure_baseline
            factor_reduction = self.cfg.pressure_reduction_factor * np.clip(normalized_pressure - 1.0, a_min=0, a_max=None)
            adjustment_factors = 1.0 - factor_reduction
            adjustment_factors = adjustment_factors.clip(lower=self.cfg.min_adjustment_factor, upper=1.0)
        
        results = []
        start_index = len(self.train_series)
        
        for i in range(start_index, len(full_series)):
            window_start = max(0, i - self.cfg.window_size)
            window_end = i
            data_window = scaled_series.iloc[window_start:window_end].values
            
            actual_scaled = scaled_series.iloc[i]
            model_fit = self._fit_on_window(data_window)
            
            if model_fit is None:
                forecast_scaled = data_window[-1] if len(data_window) > 0 else 0
                residual_scaled = np.abs(actual_scaled - forecast_scaled)
                threshold = self.baseline_resid_std * self.cfg.threshold_factor
            else:
                forecast_scaled = model_fit.forecast(steps=1)[0]
                residual_scaled = np.abs(actual_scaled - forecast_scaled)
                local_resid_std = np.std(model_fit.resid)
                threshold_base = max(local_resid_std, 0.5 * self.baseline_resid_std)
                threshold = threshold_base * self.cfg.threshold_factor

            # Applying drift-aware adjustment factors
            adjustment_factor = adjustment_factors.iloc[i]
            final_threshold = threshold * adjustment_factor

            score = residual_scaled / (final_threshold + 1e-12)
            is_anomaly = score > 1.0

            results.append((
                test_series.index[i-start_index], test_series.iloc[i-start_index],
                forecast_scaled, residual_scaled, final_threshold, score, is_anomaly
            ))

        results_df = pd.DataFrame(
            results,
            columns=["timestamp", "actual", "forecast_scaled", "residual_scaled", "threshold_scaled", "score", "is_anomaly"]
        ).set_index("timestamp")

        results_df = self._cluster_anomalies(results_df)
        
        valid_forecasts = results_df['forecast_scaled'].notna()
        if valid_forecasts.any():
            results_df.loc[valid_forecasts, 'forecast'] = self.scaler.inverse_transform(
                results_df.loc[valid_forecasts, ['forecast_scaled']].values
            )
            results_df['forecast'] = results_df['forecast'].fillna(method='ffill').fillna(method='bfill')

        return results_df
    
    def _cluster_anomalies(self, results_df: pd.DataFrame) -> pd.DataFrame:
        anomaly_points = results_df[results_df['is_anomaly']]
        if anomaly_points.empty: return results_df
        results_df['is_anomaly'] = False
        anomalies = anomaly_points.sort_index()
        time_diffs = anomalies.index.to_series().diff().dt.days.fillna(self.cfg.min_gap_days + 1)
        cluster_ids = (time_diffs > self.cfg.min_gap_days).cumsum()
        for _, cluster in anomalies.groupby(cluster_ids):
            best_in_cluster = cluster['score'].idxmax()
            results_df.loc[best_in_cluster, 'is_anomaly'] = True
        return results_df