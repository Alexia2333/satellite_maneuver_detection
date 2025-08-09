# src/models/hybrid/debug_detector.py
"""
调试版本的检测器，确保阈值正确设置
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
import xgboost as xgb

class DebugXGBoostDetector:
    """简化的调试检测器"""
    
    def __init__(self, target_column='target', threshold_percentile=95):
        self.target_column = target_column
        self.threshold_percentile = threshold_percentile
        self.model = None
        self.feature_names = []
        self.threshold = None
        
    def fit(self, train_data):
        """训练模型"""
        print("\n🔧 调试检测器训练...")
        
        # 准备数据
        X = train_data.drop(columns=[self.target_column])
        y = train_data[self.target_column]
        self.feature_names = list(X.columns)
        
        print(f"   特征数: {len(self.feature_names)}")
        print(f"   训练样本数: {len(X)}")
        print(f"   目标变量统计: mean={y.mean():.6f}, std={y.std():.6f}")
        
        # 简单的XGBoost模型
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        # 训练
        self.model.fit(X, y)
        
        # 计算训练集残差
        y_pred = self.model.predict(X)
        residuals = np.abs(y - y_pred)
        
        # 设置阈值
        self.threshold = np.percentile(residuals, self.threshold_percentile)
        
        print(f"   训练完成!")
        print(f"   残差统计: mean={residuals.mean():.6f}, std={residuals.std():.6f}")
        print(f"   检测阈值 ({self.threshold_percentile}%分位数): {self.threshold:.6f}")
        
        # 训练R2
        train_score = self.model.score(X, y)
        print(f"   训练R²: {train_score:.4f}")
        
        return self
        
    def detect(self, test_data):
        """检测异常"""
        print("\n🔍 调试检测...")
        
        X = test_data[self.feature_names]
        y = test_data[self.target_column]
        
        # 预测
        y_pred = self.model.predict(X)
        residuals = np.abs(y - y_pred)
        
        # 检测异常
        anomalies = residuals > self.threshold
        anomaly_indices = np.where(anomalies)[0]
        
        print(f"   测试样本数: {len(X)}")
        print(f"   残差范围: [{residuals.min():.6f}, {residuals.max():.6f}]")
        print(f"   检测阈值: {self.threshold:.6f}")
        print(f"   检测到异常: {len(anomaly_indices)} 个")
        
        # 打印前10个异常的详情
        if len(anomaly_indices) > 0:
            print("\n   前10个异常详情:")
            for i in anomaly_indices[:10]:
                print(f"     {test_data.index[i]}: 残差={residuals[i]:.6f}, 实际={y.iloc[i]:.6f}, 预测={y_pred[i]:.6f}")
        
        return test_data.index[anomaly_indices], residuals