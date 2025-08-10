import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from typing import Dict
import warnings
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima.model.ARIMA', UserWarning)

class RealTimeARIMADetector:
    """
    一个为实时、单点异常检测优化的ARIMA检测器。
    工作流程：
    1. 用大量历史正常数据进行一次`fit`。
    2. 对每一个新数据点调用`detect`方法。
    """
    def __init__(self, p: int = 5, d: int = 1, q: int = 1, threshold_factor: float = 3.0):
        self.p, self.d, self.q = p, d, q
        self.threshold_factor = threshold_factor
        
        self.model_fit = None
        self.history = None
        self.residual_std = None
        self.series_name = None # <--- 新增：用于存储序列名称

    def fit(self, history: pd.Series):
        """
        用历史数据训练ARIMA模型，并建立“正常”残差的基线。
        """
        if not isinstance(history.index, pd.DatetimeIndex):
            raise TypeError("Input series for fitting must have a DatetimeIndex.")
            
        print(f"Fitting ARIMA({self.p},{self.d},{self.q}) on {len(history)} historical data points...")
        self.history = history.copy()
        self.series_name = history.name # <--- 【核心修正1】存储原始序列的名称
        
        # 训练模型
        model = ARIMA(self.history, order=(self.p, self.d, self.q))
        self.model_fit = model.fit()
        
        # 计算并存储历史残差的标准差
        residuals = self.model_fit.resid
        self.residual_std = np.std(residuals)
        
        print(f"Fit complete. Historical residual standard deviation = {self.residual_std:.6f}")
        return self

    def detect(self, new_observation: float, timestamp) -> Dict:
        """
        检测一个新数据点是否为异常。
        """
        if self.model_fit is None:
            raise RuntimeError("Detector has not been fitted. Call .fit() with historical data first.")
        
        forecast = self.model_fit.forecast(steps=1).iloc[0]
        
        residual = new_observation - forecast
        score = abs(residual) / self.residual_std if self.residual_std > 1e-9 else 0.0
        
        is_maneuver = score > self.threshold_factor
        
        # 【核心修正2】创建新数据点时，使用之前存储的名称
        new_data_point = pd.Series([new_observation], index=[timestamp], name=self.series_name)
        self.model_fit = self.model_fit.append(new_data_point, refit=False)
        
        return {
            "forecast": forecast,
            "residual": residual,
            "score": score,
            "is_maneuver": is_maneuver
        }