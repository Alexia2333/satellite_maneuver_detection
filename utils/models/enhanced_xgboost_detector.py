# src/models/hybrid/enhanced_xgboost_detector.py

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional, List
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, recall_score, f1_score
import xgboost as xgb
from typing import Tuple, Dict, Optional, List

try:
    from .xgboost_detector import XGBoostAnomalyDetector
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from xgboost_detector import XGBoostAnomalyDetector

class ImprovedXGBoostDetector(XGBoostAnomalyDetector):
    """
    改进版XGBoost检测器 - 采用分位数阈值以提高稳健性
    """
    
    def __init__(self, target_column: str = 'mean_motion', xgb_params: Optional[Dict] = None, 
                 threshold_factor: float = 3.0,
                 threshold_quantile: float = 0.999,
                 enable_threshold_optimization: bool = False,
                 enable_temporal_clustering: bool = True, satellite_type: str = 'auto',
                 recall_boost: bool = True, ultra_deep: bool = True):
        
        super().__init__(target_column, xgb_params or {}, threshold_factor, 
                        enable_threshold_optimization, enable_temporal_clustering, satellite_type)

        self.ultra_deep = ultra_deep
        self.recall_boost = recall_boost
        self.multi_threshold_results = {}
        self.threshold_factor = threshold_factor
        self.threshold_quantile = threshold_quantile
        self.initial_xgb_params = xgb_params

    def _get_ultra_deep_params(self, satellite_type: str) -> dict:
        """【最终修正版】获取特定卫星的极深树参数配置"""
        
        print(f"   -> Getting ultra_deep_params for satellite type: {satellite_type}")
        
        # 为风云4A（GEO同步轨道维持）设计的参数
        if satellite_type == 'FY-4A' or satellite_type == 'FENGYUN':
            return {
                'objective': 'reg:squarederror',
                'n_estimators': 1000,
                'max_depth': 10,
                'learning_rate': 0.02,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'reg_alpha': 0.5,
                'reg_lambda': 0.5,
                'gamma': 0.1,
            }
            
        # --- 新增：为Jason系列（LEO轨道）设计的参数 ---
        elif satellite_type == 'JASON':
            return {
                'objective': 'reg:squarederror',
                'n_estimators': 500,
                'max_depth': 7, # LEO轨道模式可能稍简单，适当降低深度
                'learning_rate': 0.03,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'min_child_weight': 3,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
            }
            
        # --- 新增：一个通用的默认配置，确保函数永远不会返回None ---
        else:
            print(f"   -> Warning: No specific ultra_deep config for '{satellite_type}'. Using default.")
            return {
                'objective': 'reg:squarederror',
                'n_estimators': 300,
                'max_depth': 5,
                'learning_rate': 0.05,
                'random_state': 42,
            }

    @staticmethod
    def create_time_based_features(df: pd.DataFrame, maneuver_start: str = "2018-01-01", maneuver_end: str = "2022-12-31") -> pd.DataFrame:
        """根据机动窗口创建时间特征（保持不变）"""
        df_copy = df.copy()
        start_date = pd.to_datetime(maneuver_start)
        end_date = pd.to_datetime(maneuver_end)
        df_copy['is_maneuver_period'] = ((df_copy.index >= start_date) & (df_copy.index <= end_date)).astype(int)
        df_copy['days_from_maneuver_start'] = (df_copy.index - start_date).days
        return df_copy

    def fit(self, train_features: pd.DataFrame, y_true_val: Optional[pd.Series] = None, satellite_name: str = None, verbose: bool = True):
        if self.satellite_type == 'auto' and satellite_name:
            if 'Fengyun-4A' in satellite_name or 'FY-4A' in satellite_name: self.satellite_type = 'FY-4A'
            else: self.satellite_type = 'GEO'
        if self.ultra_deep: self.xgb_params = self._get_ultra_deep_params(self.satellite_type)
        else: self.xgb_params = self.initial_xgb_params or {}
        if verbose: print(f"🌲 Training with satellite-specific config: '{self.satellite_type}'")
        X_train = train_features.drop(columns=['target'])
        y_train = train_features['target']
        self.feature_names = [col for col in X_train.columns if col != self.target_column]
        split_idx = int(len(X_train) * 0.8)
        X_train_split, y_train_split = X_train.iloc[:split_idx], y_train.iloc[:split_idx]
        X_val_split, y_val_split = X_train.iloc[split_idx:], y_train.iloc[split_idx:]
        early_stopping_rounds = 80 
        self.model = xgb.XGBRegressor(**self.xgb_params, early_stopping_rounds=early_stopping_rounds)
        self.model.fit(
            X_train_split[self.feature_names], y_train_split, 
            eval_set=[(X_val_split[self.feature_names], y_val_split)], 
            verbose=False
        )
        train_predictions = self.model.predict(X_train[self.feature_names])
        self.train_residuals = y_train - train_predictions
        if verbose:
            train_r2 = 1 - (np.sum(self.train_residuals**2) / np.sum((y_train - np.mean(y_train))**2))
            print(f"   -> Training R²: {train_r2:.4f}")
        if self.enable_threshold_optimization and y_true_val is not None and len(y_true_val) == len(X_val_split):
            print("🔧 Optimizing threshold with ground truth labels...")
            # This part of logic is not used in the current k-fold script but kept for completeness
            self.optimized_threshold_factor = self._ground_truth_threshold_optimization(X_val_split[self.feature_names], y_val_split, y_true_val, verbose)
            self.residual_threshold = self.optimized_threshold_factor * np.std(self.train_residuals)
        else:
            # The logic used in our k-fold script
            abs_train_residuals = np.abs(self.train_residuals)
            self.residual_threshold = np.quantile(abs_train_residuals, self.threshold_quantile)
        if verbose:
            print(f"   -> Using quantile {self.threshold_quantile} for thresholding.")
            print(f"   -> Final Detection Threshold: {self.residual_threshold:.6f}")


    def _ground_truth_threshold_optimization(self, X_val: pd.DataFrame, y_val_pred_target: pd.Series, y_val_true: pd.Series, verbose: bool = False) -> float:
        """
        [重构] 使用真实标签(y_val_true)进行阈值优化，找到最佳F1分数的阈值因子。
        """
        val_predictions = self.model.predict(X_val)
        val_residuals = y_val_pred_target - val_predictions
        
        threshold_factors = np.arange(1.0, 5.0, 0.1)
        
        best_f1 = -1
        best_threshold_factor = self.threshold_factor
        results = []

        if np.sum(y_val_true) == 0:
            if verbose:
                print("   ⚠️ No true anomalies in validation set. Cannot optimize threshold.")
            return self.threshold_factor

        train_std_dev = np.std(self.train_residuals)

        for factor in threshold_factors:
            temp_threshold = factor * train_std_dev
            predicted_anomalies = (np.abs(val_residuals) > temp_threshold).astype(int)
            
            precision = precision_score(y_val_true, predicted_anomalies, zero_division=0)
            recall = recall_score(y_val_true, predicted_anomalies, zero_division=0)
            f1 = f1_score(y_val_true, predicted_anomalies, zero_division=0)
            
            results.append({'factor': factor, 'p': precision, 'r': recall, 'f1': f1, 'det': np.sum(predicted_anomalies)})
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold_factor = factor

        if verbose and results:
            print("   Ground Truth Threshold Optimization Results (Top 5 by F1):")
            sorted_results = sorted(results, key=lambda x: x['f1'], reverse=True)[:5]
            for r in sorted_results:
                print(f"   Factor={r['factor']:.1f}: F1={r['f1']:.3f}, P={r['p']:.3f}, R={r['r']:.3f}, Detections={r['det']}")
        
        return best_threshold_factor

    def detect_anomalies_with_sensitivity(self, features_df: pd.DataFrame, sensitivity: str = 'balanced') -> Tuple[list, np.ndarray]:
        """基于敏感度的异常检测 - 针对特定卫星类型调整因子"""
        
        if self.satellite_type == 'FY-4A':
            if sensitivity == 'high_recall':
                threshold_factor = max(1.0, self.optimized_threshold_factor * 0.6)
            elif sensitivity == 'high_precision':
                threshold_factor = self.optimized_threshold_factor * 1.5
            else: # balanced
                threshold_factor = self.optimized_threshold_factor
        else:
            if sensitivity == 'high_recall':
                threshold_factor = 1.5
            elif sensitivity == 'high_precision':
                threshold_factor = 3.5
            else:
                threshold_factor = self.optimized_threshold_factor

        original_threshold = self.residual_threshold
        self.residual_threshold = threshold_factor * np.std(self.train_residuals)
        
        print(f"   -> Using '{sensitivity}' sensitivity for '{self.satellite_type}' (Factor: {threshold_factor:.2f})")
        
        anomaly_indices, scores = super().detect_anomalies(features_df, return_scores=True)
        
        self.residual_threshold = original_threshold
        
        return anomaly_indices, scores

    def get_ultra_deep_summary(self) -> Dict[str, any]:
        """获取极深树模型的详细摘要"""
        base_summary = super().get_model_summary() if hasattr(super(), 'get_model_summary') else {}
        
        ultra_deep_info = {
            'ultra_deep_enabled': self.ultra_deep,
            'tree_depth': self.xgb_params.get('max_depth'),
            'n_estimators': self.xgb_params.get('n_estimators'),
            'growth_policy': self.xgb_params.get('grow_policy'),
            'satellite_type_config': self.satellite_type,
            'optimized_threshold_factor': f"{self.optimized_threshold_factor:.2f}",
        }
        
        base_summary.update(ultra_deep_info)
        return base_summary