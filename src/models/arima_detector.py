import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

class ARIMADetector:
    """
    一个采用【混合波动率动态阈值】和优化流程的ARIMA检测器。
    """
    def __init__(self, p: int = 5, d: int = 1, q: int = 1):
        self.p, self.d, self.q = p, d, q
        self.model_fit = None

    def fit(self, history: pd.Series):
        """用历史数据一次性训练ARIMA模型。"""
        print(f"Fitting ARIMA({self.p},{self.d},{self.q}) on {len(history)} historical data points...")
        model = ARIMA(history, order=(self.p, self.d, self.q))
        
        try:
            self.model_fit = model.fit(method_kwargs={'maxiter': 200})
        except Exception:
            self.model_fit = model.fit()

        print("Fit complete.")
        return self

    def detect(self, test_series: pd.Series, threshold_factor: float = 2.5, window: int = 30) -> pd.DataFrame:
        """
        在测试集上检测异常，并使用高级的混合波动率动态阈值。
        """
        if self.model_fit is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        print("Forecasting on the test set...")
        predictions = self.model_fit.predict(start=test_series.index[0], end=test_series.index[-1])
        
        residuals = np.abs(test_series - predictions)
        
        # --- 【核心修改】集成你旧代码中更高级的自适应阈值逻辑 ---
        
        # 1. 计算全局残差的标准差，作为阈值的“下限”参考
        global_residual_std = residuals.std()
        
        # 2. 计算基于滚动窗口的局部标准差
        local_residual_std = residuals.rolling(window=window, min_periods=1).std().fillna(global_residual_std)
        
        # 3. 阈值的基础是“局部标准差”和“全局标准差的一半”中的较大值
        #    这可以防止在太平稳的时期阈值过低而产生误报
        threshold_base = np.maximum(local_residual_std, 0.5 * global_residual_std)

        dynamic_threshold = threshold_factor * threshold_base
        # -------------------------------------------------------------

        anomalies = (residuals > dynamic_threshold).astype(int)
        
        print(f"Detection complete. Found {anomalies.sum()} potential anomalies.")
        
        return pd.DataFrame({
            'actual': test_series,
            'forecast': predictions,
            'score': residuals,
            'threshold': dynamic_threshold,
            'is_anomaly': anomalies
        })