# src/models/hybrid/drift_aware_detector.py

import pandas as pd
import numpy as np
import xgboost as xgb

class DriftAwareDetector:
    """
    封装了我们最终优化的检测逻辑的探测器类。
    所有参数通过一个配置字典传入，方便管理。
    """
    def __init__(self, config: dict):
        print("   初始化DriftAwareDetector...")
        self.config = config
        self.model = None
        self.feature_names = None
        self.residual_threshold = None
        self.change_rate_threshold = None
        self.pressure_baseline = None

    def fit(self, train_data: pd.DataFrame):
        print("   🔧 正在训练模型并计算阈值...")
        X_train = train_data.drop(columns=['target'])
        y_train = train_data['target']
        self.feature_names = list(X_train.columns)

        self.model = xgb.XGBRegressor(**self.config['xgb_params'])
        self.model.fit(X_train, y_train)
        
        y_train_pred = self.model.predict(X_train)
        train_residuals = np.abs(y_train - y_train_pred)
        train_change_rates = np.abs(y_train - y_train_pred) / (np.abs(y_train) + 1e-8)

        self.residual_threshold = train_residuals.mean() + self.config['threshold_std_multiplier'] * train_residuals.std()
        self.change_rate_threshold = np.percentile(train_change_rates, self.config['threshold_percentile'])
        
        drift_features = [col for col in self.feature_names if any(keyword in col for keyword in ['drift', 'cumulative'])]
        if drift_features:
            train_drift_pressure = train_data[drift_features].abs().mean(axis=1)
            self.pressure_baseline = np.median(train_drift_pressure)

    def detect(self, test_data: pd.DataFrame) -> tuple:
        print("   🧪 正在应用增强检测逻辑...")
        if self.model is None: raise RuntimeError("模型尚未训练，请先调用fit方法。")

        X_test = test_data[self.feature_names]
        y_test = test_data['target']
        y_test_pred = self.model.predict(X_test)

        test_residuals = np.abs(y_test - y_test_pred)
        test_change_rates = np.abs(y_test - y_test_pred) / (np.abs(y_test) + 1e-8)
        
        strong_signal_mask = (test_residuals > self.residual_threshold) & (test_change_rates > self.change_rate_threshold)
        
        pressure_assisted_mask = pd.Series(False, index=test_data.index)
        normalized_pressure = None
        drift_features = [col for col in self.feature_names if any(keyword in col for keyword in ['drift', 'cumulative'])]
        
        if drift_features and self.pressure_baseline is not None and self.pressure_baseline > 0:
            test_drift_pressure = test_data[drift_features].abs().mean(axis=1)
            normalized_pressure = test_drift_pressure / self.pressure_baseline
            
            for i in range(len(test_residuals)):
                if normalized_pressure.iloc[i] > self.config['pressure_activation_threshold']:
                    reduction_factor = 1.0 - self.config['pressure_reduction_factor'] * min((normalized_pressure.iloc[i] - self.config['pressure_activation_threshold']), 1.0)
                    adjusted_threshold = self.residual_threshold * reduction_factor
                    if test_residuals[i] > adjusted_threshold:
                        pressure_assisted_mask.iloc[i] = True
        
        final_anomaly_mask = strong_signal_mask | pressure_assisted_mask
        initial_anomaly_indices = np.where(final_anomaly_mask)[0]
        initial_anomaly_timestamps = test_data.index[initial_anomaly_indices]
        
        return initial_anomaly_timestamps, { "residuals": test_residuals, "pressure": normalized_pressure }