# refactored_detector.py

import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import timedelta

# =====================================================================
# 特征工程函数 (从主脚本迁移过来，保持独立)
# =====================================================================
def create_drift_enhanced_features(tle_data, scaling_factor, target_col='mean_motion'):
    print("\n🛠️ 创建漂移增强特征")
    print("-" * 20)
    
    data = tle_data.set_index('epoch').copy()
    base_cols = ['mean_motion', 'eccentricity', 'inclination']
    available_cols = [col for col in base_cols if col in data.columns]
    
    if not available_cols:
        print(f"❌ 没有找到基础特征列: {base_cols}")
        return None
    
    print(f"   基础参数: {available_cols}")
    enhanced_data = data[available_cols].copy()
    
    lags = [1, 2, 3, 7]
    windows = [3, 7, 14]
    for col in available_cols:
        for lag in lags:
            enhanced_data[f'{col}_lag_{lag}'] = data[col].shift(lag)
        for window in windows:
            enhanced_data[f'{col}_rolling_mean_{window}'] = data[col].rolling(window, min_periods=window//2).mean()
            enhanced_data[f'{col}_rolling_std_{window}'] = data[col].rolling(window, min_periods=window//2).std()
        enhanced_data[f'{col}_diff_1'] = data[col].diff(1)
        enhanced_data[f'{col}_diff_7'] = data[col].diff(7)

    long_windows = [30, 60]
    for col in available_cols:
        for window in long_windows:
            long_mean = data[col].rolling(window, min_periods=window//3).mean()
            long_std = data[col].rolling(window, min_periods=window//3).std()
            enhanced_data[f'{col}_drift_from_{window}d'] = data[col] - long_mean
            enhanced_data[f'{col}_drift_zscore_{window}d'] = (data[col] - long_mean) / (long_std + 1e-8)

    trend_windows = [7, 14]
    for col in available_cols:
        for window in trend_windows:
            enhanced_data[f'{col}_trend_{window}d'] = data[col].rolling(window).apply(
                lambda x: (x[-1] - x[0]) / len(x) if len(x) > 1 else 0, raw=True)

    for col in available_cols:
        short_mean = data[col].rolling(7, min_periods=3).mean()
        short_drift = (data[col] - short_mean).abs()
        enhanced_data[f'{col}_cumulative_drift_7d'] = short_drift.rolling(7, min_periods=3).sum()
        enhanced_data[f'{col}_cumulative_drift_14d'] = short_drift.rolling(14, min_periods=7).sum()

    if len(available_cols) > 1:
        for window in [7, 14]:
            drift_cols = [f'{col}_drift_from_30d' for col in available_cols if f'{col}_drift_from_30d' in enhanced_data.columns]
            if drift_cols:
                enhanced_data[f'combined_drift_l2_{window}d'] = np.sqrt(enhanced_data[drift_cols].pow(2).sum(axis=1))

    # 创建目标变量（差分）
    mean_motion_diff = data[target_col].diff()
    enhanced_data['target'] = mean_motion_diff.shift(-1).abs()

    # ✅ 无需筛选小值：保留所有非NaN，直接做放大处理（重点！！）
    enhanced_data = enhanced_data.dropna(subset=['target'])

    # ✅ 数值放大 + log1p，提升数值可学习性
    enhanced_data['target'] = np.log1p(enhanced_data['target'] * 1e8)

    # 打印确认
    print("✅ 应用了 log1p(target * 1e8) 变换")
    print(enhanced_data['target'].describe())
    


    # 填充其他特征
    enhanced_data = enhanced_data.fillna(method='ffill', limit=3)
    enhanced_data = enhanced_data.fillna(method='bfill', limit=3)
    for col in enhanced_data.columns:
        if enhanced_data[col].isna().any():
            median_val = enhanced_data[col].median()
            enhanced_data[col].fillna(median_val if not pd.isna(median_val) else 0, inplace=True)

    print(f"✅ 特征创建完成")
    print(f"   特征数量: {len(enhanced_data.columns)}")
    print(f"   有效记录: {len(enhanced_data)}")
    print(f"   目标变量统计: mean={enhanced_data['target'].mean():.6f}, std={enhanced_data['target'].std():.6f}")
    return enhanced_data


# =====================================================================
# 时间序列聚类函数 (从主脚本迁移过来，保持独立)
# =====================================================================
def group_and_filter_anomalies(timestamps, max_gap=pd.Timedelta(days=1), min_group_size=2): # <--- 增加 min_group_size 参数
    if timestamps.empty:
        return []

    timestamps = pd.Series(sorted(timestamps))
    groups = []
    current_group = [timestamps.iloc[0]]

    for i in range(1, len(timestamps)):
        if (timestamps.iloc[i] - timestamps.iloc[i-1]) <= max_gap:
            current_group.append(timestamps.iloc[i])
        else:
            if len(current_group) >= min_group_size:
                groups.extend(current_group)
            current_group = [timestamps.iloc[i]]

    if len(current_group) >= min_group_size:
        groups.extend(current_group)

    return sorted(list(set(groups)))
# =====================================================================
# 核心探测器类
# =====================================================================
class DriftAwareDetector:
    def __init__(self, config: dict):
        """
        通过一个配置字典来初始化探测器
        """
        print("   初始化探测器...")
        self.config = config
        self.model = None
        self.feature_names = None
        self.residual_threshold = None
        self.change_rate_threshold = None
        self.pressure_baseline = None

    def fit(self, train_data: pd.DataFrame):
        """
        训练模型并计算所有必要的阈值
        """
        print("   🔧 训练XGBoost模型并计算阈值...")
        
        X_train = train_data.drop(columns=['target'])
        y_train = train_data['target']
        self.feature_names = list(X_train.columns)

        # 初始化并训练XGBoost模型
        self.model = xgb.XGBRegressor(**self.config['xgb_params'])
        self.model.fit(X_train, y_train)
        
        print(f"   训练R²: {self.model.score(X_train, y_train):.4f}")

        # 计算训练集残差和变化率
        y_train_pred = self.model.predict(X_train)
        train_residuals = np.abs(y_train - y_train_pred)
        train_change_rates = np.abs(y_train - y_train_pred) / (np.abs(y_train) + 1e-8)

        # 计算并存储阈值
        self.residual_threshold = train_residuals.mean() + self.config['threshold_std_multiplier'] * train_residuals.std()
        self.change_rate_threshold = np.percentile(train_change_rates, self.config['threshold_percentile'])
        
        # 计算并存储漂移压力基线
        drift_features = [col for col in self.feature_names if any(keyword in col for keyword in ['drift', 'cumulative'])]
        if drift_features:
            train_drift_pressure = train_data[drift_features].abs().mean(axis=1)
            self.pressure_baseline = np.median(train_drift_pressure)
        
        print(f"   残差阈值: {self.residual_threshold:.6f}")
        print(f"   变化率阈值: {self.change_rate_threshold:.6f}")
        print(f"   压力基线: {self.pressure_baseline:.6f}" if self.pressure_baseline is not None else "   无压力特征")

    def detect(self, test_data: pd.DataFrame) -> tuple:
        """
        在测试集上执行我们最终优化的检测逻辑
        """
        print("   🧪 应用增强检测逻辑...")
        if self.model is None:
            raise RuntimeError("模型尚未训练，请先调用fit方法。")

        X_test = test_data[self.feature_names]
        y_test = test_data['target']
        y_test_pred = self.model.predict(X_test)

        # 计算基础指标
        test_residuals = np.abs(y_test - y_test_pred)
        test_change_rates = np.abs(y_test - y_test_pred) / (np.abs(y_test) + 1e-8)
        
        # 路径A: "强信号" 检测
        strong_signal_mask = (test_residuals > self.residual_threshold) & (test_change_rates > self.change_rate_threshold)
        print(f"      - 强信号检测发现: {strong_signal_mask.sum()} 个异常")

        # 路径B: "压力辅助信号" 检测
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
            print(f"      - 压力辅助检测发现: {pressure_assisted_mask.sum()} 个异常")

        # 合并结果
        final_anomaly_mask = strong_signal_mask | pressure_assisted_mask
        initial_anomaly_indices = np.where(final_anomaly_mask)[0]
        initial_anomaly_timestamps = test_data.index[initial_anomaly_indices]
        
        # 返回原始检测结果及分析所需数据
        return initial_anomaly_timestamps, {
            'test_residuals': test_residuals,
            'normalized_pressure': normalized_pressure
        }