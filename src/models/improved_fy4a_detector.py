# src/models/hybrid/improved_fy4a_detector.py
"""
针对风云4A卫星轨道维持机动的改进检测器
基于enhanced_xgboost_detector.py改进，专门处理漂移累积模式
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional, List
from datetime import datetime, timedelta
import xgboost as xgb

# 导入基类
try:
    from .enhanced_xgboost_detector import ImprovedXGBoostDetector
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from enhanced_xgboost_detector import ImprovedXGBoostDetector

class DriftAwareXGBoostDetector(ImprovedXGBoostDetector):
    """
    漂移感知的XGBoost检测器
    专门针对GEO卫星的轨道维持机动模式优化
    """
    
    def __init__(self, target_column: str = 'mean_motion', 
                 xgb_params: Optional[Dict] = None,
                 threshold_quantile: float = 0.98,
                 drift_window: int = 30,
                 pressure_threshold: float = 0.7,
                 **kwargs):
        """
        Args:
            drift_window: 计算漂移的时间窗口（天）
            pressure_threshold: 漂移压力阈值（0-1）
        """
        
        # 调用父类初始化
        super().__init__(target_column=target_column, 
                        xgb_params=xgb_params,
                        threshold_quantile=threshold_quantile,
                        **kwargs)
        
        self.drift_window = drift_window
        self.pressure_threshold = pressure_threshold
        self.drift_features = []
        self.drift_pressure_history = None
        
    def fit(self, train_features: pd.DataFrame, **kwargs):
        """
        重写fit方法，在训练前计算漂移特征
        """
        # 识别漂移相关特征
        self.drift_features = [col for col in train_features.columns 
                              if any(keyword in col for keyword in 
                                    ['drift', 'cumulative', 'trend', 'pressure'])
                              and col != 'target']  # 排除目标列
        
        if self.drift_features:
            print(f"   发现 {len(self.drift_features)} 个漂移相关特征")
            
            # 计算训练集的漂移压力基线
            drift_values = train_features[self.drift_features].abs()
            self.drift_pressure_baseline = drift_values.mean(axis=1).quantile(0.75)
            print(f"   漂移压力基线: {self.drift_pressure_baseline:.6f}")
        
        # 调用父类的fit方法
        result = super().fit(train_features, **kwargs)
        
        # 确保阈值设置正确
        if self.residual_threshold == 0 or np.isnan(self.residual_threshold):
            print("   ⚠️ 检测到阈值为0，重新计算...")
            # 手动计算阈值
            X = train_features[self.feature_names]
            y = train_features[self.target_column]
            predictions = self.model.predict(X)
            residuals = np.abs(y - predictions)
            
            # 使用分位数方法
            self.residual_threshold = np.percentile(residuals, self.threshold_quantile * 100)
            print(f"   ✅ 重新计算的阈值: {self.residual_threshold:.6f}")
        
        return result
    
    def detect_anomalies(self, features_df: pd.DataFrame, 
                        return_scores: bool = False,
                        use_drift_adjustment: bool = True):
        """
        检测异常，可选择性地使用漂移调整
        """
        # 首先使用基础检测
        base_anomalies, base_scores = super().detect_anomalies(
            features_df, return_scores=True
        )
        
        if not use_drift_adjustment or not self.drift_features:
            if return_scores:
                return base_anomalies, base_scores
            return base_anomalies
        
        # 漂移压力调整
        print("   应用漂移压力调整...")
        
        # 计算当前漂移压力
        drift_pressure = self._calculate_drift_pressure(features_df)
        
        # 获取残差（从base_scores中提取）
        if len(base_scores) >= 2:
            residuals = base_scores[1]
        else:
            # 如果没有残差，重新计算
            predictions = self.model.predict(features_df[self.feature_names])
            residuals = np.abs(features_df[self.target_column] - predictions)
        
        # 动态调整阈值
        adjusted_anomalies = self._apply_drift_adjustment(
            residuals, drift_pressure, features_df.index
        )
        
        # 时序聚类（如果启用）
        if self.enable_temporal_clustering:
            adjusted_anomalies = self._temporal_clustering(
                adjusted_anomalies, features_df.index
            )
        
        if return_scores:
            return adjusted_anomalies, (drift_pressure, residuals)
        return adjusted_anomalies
    
    def _calculate_drift_pressure(self, features_df: pd.DataFrame) -> pd.Series:
        """
        计算漂移压力指标
        """
        if not self.drift_features:
            return pd.Series(0, index=features_df.index)
        
        # 使用可用的漂移特征
        available_drift_features = [f for f in self.drift_features 
                                   if f in features_df.columns]
        
        if not available_drift_features:
            return pd.Series(0, index=features_df.index)
        
        # 计算综合漂移压力（标准化后的均值）
        drift_values = features_df[available_drift_features].abs()
        
        # 使用滚动窗口平滑
        drift_pressure = drift_values.mean(axis=1)
        drift_pressure = drift_pressure.rolling(
            window=3, min_periods=1, center=True
        ).mean()
        
        # 标准化到0-1范围
        if hasattr(self, 'drift_pressure_baseline') and self.drift_pressure_baseline > 0:
            drift_pressure = drift_pressure / self.drift_pressure_baseline
            drift_pressure = drift_pressure.clip(0, 2)  # 限制最大值
        
        return drift_pressure
    
    def _apply_drift_adjustment(self, residuals: np.ndarray, 
                               drift_pressure: pd.Series,
                               index: pd.DatetimeIndex) -> List[int]:
        """
        基于漂移压力动态调整检测阈值
        """
        adjusted_anomalies = []
        
        # 将numpy数组转换为Series以便索引对齐
        residuals_series = pd.Series(residuals, index=index)
        
        for i, (timestamp, residual) in enumerate(residuals_series.items()):
            # 获取当前漂移压力
            pressure = drift_pressure.get(timestamp, 0)
            
            # 动态调整阈值
            # 高压力时降低阈值（更容易检测到异常）
            if pressure > self.pressure_threshold:
                # 线性调整：压力越高，阈值降低越多
                adjustment_factor = 1 - 0.3 * min((pressure - self.pressure_threshold) / 0.3, 1)
                adjusted_threshold = self.residual_threshold * adjustment_factor
            else:
                adjusted_threshold = self.residual_threshold
            
            # 检测异常
            if residual > adjusted_threshold:
                adjusted_anomalies.append(i)
        
        print(f"   漂移调整后: {len(adjusted_anomalies)} 个异常")
        
        return adjusted_anomalies
    
    def _temporal_clustering(self, anomaly_indices: List[int], 
                           index: pd.DatetimeIndex,
                           min_cluster_size: int = 2,
                           max_gap_days: int = 3) -> List[int]:
        """
        对检测到的异常进行时序聚类
        减少孤立的误报
        """
        if len(anomaly_indices) < min_cluster_size:
            return anomaly_indices
        
        # 转换为时间戳
        anomaly_times = index[anomaly_indices]
        
        # 聚类逻辑：相邻异常时间差小于max_gap_days则属于同一簇
        clusters = []
        current_cluster = [anomaly_indices[0]]
        
        for i in range(1, len(anomaly_indices)):
            time_diff = (anomaly_times[i] - anomaly_times[i-1]).days
            
            if time_diff <= max_gap_days:
                current_cluster.append(anomaly_indices[i])
            else:
                if len(current_cluster) >= min_cluster_size:
                    clusters.extend(current_cluster)
                current_cluster = [anomaly_indices[i]]
        
        # 处理最后一个簇
        if len(current_cluster) >= min_cluster_size:
            clusters.extend(current_cluster)
        
        print(f"   时序聚类后: {len(clusters)} 个异常（原始: {len(anomaly_indices)}）")
        
        return sorted(clusters)
    
    def get_drift_analysis(self, features_df: pd.DataFrame) -> Dict:
        """
        获取漂移分析报告
        """
        drift_pressure = self._calculate_drift_pressure(features_df)
        
        analysis = {
            'mean_pressure': float(drift_pressure.mean()),
            'max_pressure': float(drift_pressure.max()),
            'high_pressure_ratio': float((drift_pressure > self.pressure_threshold).mean()),
            'pressure_trend': float(drift_pressure.iloc[-7:].mean() - drift_pressure.iloc[:7].mean())
        }
        
        # 识别高压力期间
        high_pressure_mask = drift_pressure > self.pressure_threshold
        high_pressure_periods = []
        
        if high_pressure_mask.any():
            # 找到连续的高压力期间
            changes = high_pressure_mask.astype(int).diff()
            starts = features_df.index[changes == 1]
            ends = features_df.index[changes == -1]
            
            # 处理边界情况
            if high_pressure_mask.iloc[0]:
                starts = pd.Index([features_df.index[0]]).append(starts)
            if high_pressure_mask.iloc[-1]:
                ends = ends.append(pd.Index([features_df.index[-1]]))
            
            for start, end in zip(starts, ends):
                high_pressure_periods.append({
                    'start': start,
                    'end': end,
                    'duration_days': (end - start).days,
                    'max_pressure': float(drift_pressure[start:end].max())
                })
        
        analysis['high_pressure_periods'] = high_pressure_periods
        
        return analysis