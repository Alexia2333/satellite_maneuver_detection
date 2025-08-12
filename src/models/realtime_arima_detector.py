import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from typing import Dict
import warnings
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima.model.ARIMA', UserWarning)

class RealTimeARIMADetector:
    """
    An ARIMA-based detector optimized for real-time, single-point anomaly detection.
    Workflow:
    1. Fit once using a large amount of historical normal data.
    2. Call the `detect` method for each new incoming data point.
    """
    def __init__(self, p: int = 5, d: int = 1, q: int = 1, threshold_factor: float = 3.0):
        self.p, self.d, self.q = p, d, q
        self.threshold_factor = threshold_factor
        
        self.model_fit = None
        self.history = None
        self.residual_std = None
        self.series_name = None # <--- Added: used to store the original series name

    def fit(self, history: pd.Series):
        """
        Train the ARIMA model with historical data and establish a baseline of
        'normal' residuals.
        """
        if not isinstance(history.index, pd.DatetimeIndex):
            raise TypeError("Input series for fitting must have a DatetimeIndex.")
            
        print(f"Fitting ARIMA({self.p},{self.d},{self.q}) on {len(history)} historical data points...")
        self.history = history.copy()
        self.series_name = history.name # <--- [Key Fix 1] Store the original series name
        
        # Train the model
        model = ARIMA(self.history, order=(self.p, self.d, self.q))
        self.model_fit = model.fit()
        
        # Calculate and store the standard deviation of historical residuals
        residuals = self.model_fit.resid
        self.residual_std = np.std(residuals)
        
        print(f"Fit complete. Historical residual standard deviation = {self.residual_std:.6f}")
        return self

    def detect(self, new_observation: float, timestamp) -> Dict:
        """
        Detect whether a new data point is an anomaly.
        """
        if self.model_fit is None:
            raise RuntimeError("Detector has not been fitted. Call .fit() with historical data first.")
        
        forecast = self.model_fit.forecast(steps=1).iloc[0]
        
        residual = new_observation - forecast
        score = abs(residual) / self.residual_std if self.residual_std > 1e-9 else 0.0
        
        is_maneuver = score > self.threshold_factor
        
        # When creating the new data point, use the previously stored series name
        new_data_point = pd.Series([new_observation], index=[timestamp], name=self.series_name)
        self.model_fit = self.model_fit.append(new_data_point, refit=False)
        
        return {
            "forecast": forecast,
            "residual": residual,
            "score": score,
            "is_maneuver": is_maneuver
        }
