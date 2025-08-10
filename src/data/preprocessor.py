# src/data/preprocessor.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
from typing import Tuple, List

class XGBoostPreprocessor:
    def __init__(self, columns_to_process: List[str], resampling_freq: str = '1D', normalize: bool = True):
        self.columns_to_process = columns_to_process
        self.resampling_freq = resampling_freq
        self.normalize = normalize
        self.scaler = StandardScaler() if normalize else None
        self.stationarity_results = {}

    def fit(self, data: pd.DataFrame):
        df_resampled = self._resample_and_fill(data)
        if self.normalize and self.scaler:
            self.scaler.fit(df_resampled[self.columns_to_process])

    def transform(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        df_resampled = self._resample_and_fill(data)
        if not self.stationarity_results:
            for col in self.columns_to_process:
                is_stationary, p_value = self._check_stationarity(df_resampled[col])
                self.stationarity_results[col] = {'is_stationary': is_stationary, 'p_value': p_value}
        if self.normalize and self.scaler:
            df_resampled[self.columns_to_process] = self.scaler.transform(df_resampled[self.columns_to_process])
        return df_resampled, self.stationarity_results

    def _resample_and_fill(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.set_index(pd.to_datetime(df['epoch']))
        df_subset = df[self.columns_to_process]
        df_resampled = df_subset.resample(self.resampling_freq).mean()
        df_resampled = df_resampled.interpolate(method='linear')
        return df_resampled
    
    def _check_stationarity(self, series: pd.Series) -> Tuple[bool, float]:
        result = adfuller(series.dropna())
        return result[1] <= 0.05, result[1]