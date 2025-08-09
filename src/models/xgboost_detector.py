# src/models/hybrid/xgboost_detector.py

import pandas as pd
import numpy as np
import xgboost as xgb
from typing import Dict, Optional, List, Tuple
from sklearn.metrics import recall_score, precision_score, f1_score, classification_report
from sklearn.cluster import DBSCAN
import warnings

# ==============================================================================
# 辅助函数
# ==============================================================================

def create_labels_for_split(df_index: pd.DatetimeIndex, maneuvers: list, window) -> pd.Series:
    """一个独立的标签创建函数。"""
    labels = pd.Series(0, index=df_index)
    if not maneuvers:
        return labels
    maneuver_datetimes = pd.to_datetime(maneuvers)
    for m_time in maneuver_datetimes:
        start = m_time - window
        end = m_time + window
        indices_in_window = (labels.index >= start) & (labels.index <= end)
        labels.loc[indices_in_window] = 1
    return labels


class XGBoostAnomalyDetector:
    # --- [修改1] 在构造函数中增加 early_stopping_rounds 参数 ---
    def __init__(self, target_column: str, xgb_params: Optional[Dict] = None, threshold_factor: float = 3.0,
                 enable_threshold_optimization: bool = True, enable_temporal_clustering: bool = True,
                 satellite_type: str = 'auto', early_stopping_rounds: int = 50):
        self.target_column = target_column
        self.model = None
        self.feature_names: List[str] = []
        self.enable_threshold_optimization = enable_threshold_optimization
        self.enable_temporal_clustering = enable_temporal_clustering
        self.xgb_params = xgb_params or {
            'objective': 'reg:squarederror', 'n_estimators': 200, 'learning_rate': 0.05,
            'max_depth': 5, 'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42
        }
        self.optimized_threshold_factor = threshold_factor
        self.residual_threshold = np.nan
        self.satellite_type = satellite_type
        # 保存 early stopping 参数
        self.early_stopping_rounds = early_stopping_rounds

    def fit(self, features_df, satellite_name=None, verbose=False, all_maneuvers=None, labeling_window=None):
        """
        Fits the XGBoost model using the older `early_stopping_rounds` syntax.
        """
        X = features_df.drop(columns=[self.target_column], errors='ignore')
        y = features_df[self.target_column]
        self.feature_names = X.columns.tolist()

        val_split_idx = int(len(X) * 0.7)
        X_train, X_val = X.iloc[:val_split_idx], X.iloc[val_split_idx:]
        y_train, y_val = y.iloc[:val_split_idx], y.iloc[val_split_idx:]

        if verbose:
            print(f"   -> 训练集大小: {len(X_train)}, 验证集大小: {len(X_val)}")
            print(f"   -> 特征数量: {len(self.feature_names)}")

        # --- 1. 计算 scale_pos_weight ---
        if all_maneuvers and labeling_window:
            y_true_labels_train = create_labels_for_split(y_train.index, all_maneuvers, labeling_window)
            count_negative = (y_true_labels_train == 0).sum()
            count_positive = (y_true_labels_train == 1).sum()
            scale_pos_weight = count_negative / count_positive if count_positive > 0 else 1
            print(f"   -> 检测到训练集中正负样本比例，设定 scale_pos_weight = {scale_pos_weight:.2f}")
        else:
            scale_pos_weight = 1 # 如果没有标签信息，则默认为1

        params = self.xgb_params.copy()
        if scale_pos_weight != 1:
            params['scale_pos_weight'] = scale_pos_weight
            
        # --- [修改2] 在模型初始化时传入 early_stopping_rounds ---
        self.model = xgb.XGBRegressor(**params, early_stopping_rounds=self.early_stopping_rounds)

        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        if verbose:
            train_r2 = self.model.score(X_train, y_train)
            val_r2 = self.model.score(X_val, y_val)
            print(f"   -> 训练集R²得分: {train_r2:.4f}")
            print(f"   -> 验证集R²得分: {val_r2:.4f}")
            # 打印最佳迭代次数
            if self.early_stopping_rounds:
                print(f"   -> Early stopping, best iteration is: {self.model.best_iteration}")


        # --- 2. 阈值优化 ---
        if self.enable_threshold_optimization and all_maneuvers and labeling_window:
            print("🔧 执行鲁棒性阈值优化 (目标: 最大化召回率)...")
            y_pred_val = self.model.predict(X_val)
            residuals_val = np.abs(y_val - y_pred_val)
            y_true_labels_val = create_labels_for_split(y_val.index, all_maneuvers, labeling_window)
            
            # 确保验证集里有正样本
            if y_true_labels_val.sum() == 0:
                print("   -> 警告: 验证集中没有正样本标签，无法进行阈值优化。将使用默认统计方法。")
                # Fallback if no positive labels in validation
                y_pred_train = self.model.predict(X_train)
                residuals_train = np.abs(y_train - y_pred_train)
                self.residual_threshold = np.mean(residuals_train) + self.optimized_threshold_factor * np.std(residuals_train)

            else:
                factors = np.linspace(0.1, 5.0, 50)
                results = []

                # 使用训练集的残差统计量作为基准，避免数据泄露
                train_residuals = np.abs(y_train - self.model.predict(X_train))
                train_mean_res = np.mean(train_residuals)
                train_std_res = np.std(train_residuals)

                for factor in factors:
                    threshold = train_mean_res + factor * train_std_res
                    pred_labels = (residuals_val > threshold).astype(int)
                    recall = recall_score(y_true_labels_val, pred_labels, zero_division=0)
                    precision = precision_score(y_true_labels_val, pred_labels, zero_division=0)
                    if recall > 0 or precision > 0: # 记录有意义的结果
                        results.append({'factor': factor, 'recall': recall, 'precision': precision})

                if not results:
                    print("   -> 警告: 无法在验证集上找到任何有效的阈值。将使用默认值。")
                    self.residual_threshold = train_mean_res + self.optimized_threshold_factor * train_std_res
                else:
                    results_df = pd.DataFrame(results)
                    # 优先保证高召回，其次是高精度
                    best_result = results_df.sort_values(by=['recall', 'precision'], ascending=[False, False]).iloc[0]
                    self.optimized_threshold_factor = best_result['factor']
                    self.residual_threshold = train_mean_res + self.optimized_threshold_factor * train_std_res
                    print(f"   -> 优化完成。发现最佳因子: {self.optimized_threshold_factor:.2f} (Recall={best_result['recall']:.3f}, Precision={best_result['precision']:.3f})")
        else:
             # 如果不进行优化，则基于训练集残差计算阈值
             y_pred_train = self.model.predict(X_train)
             residuals_train = np.abs(y_train - y_pred_train)
             self.residual_threshold = np.mean(residuals_train) + self.optimized_threshold_factor * np.std(residuals_train)


        return self

    def detect_anomalies(self, features_df, return_scores=False):
        # --- [修改3] 核心修复：使用self.feature_names确保预测特征与训练时一致 ---
        if not self.feature_names:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        
        # 确保预测数据包含所有需要的特征列
        missing_cols = set(self.feature_names) - set(features_df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in features_df: {missing_cols}")
            
        X = features_df[self.feature_names]
        # --- 修改结束 ---
        
        y = features_df[self.target_column]
        y_pred = self.model.predict(X)
        residuals = np.abs(y - y_pred)
        raw_anomaly_indices = np.where(residuals > self.residual_threshold)[0]
        print(f"   -> 原始异常检测数量: {len(raw_anomaly_indices)}")

        if self.enable_temporal_clustering and len(raw_anomaly_indices) > 0:
            # 将索引转换为DataFrame的iloc，而不是原始数组的索引
            clustering = DBSCAN(eps=3, min_samples=1).fit(raw_anomaly_indices.reshape(-1, 1))
            final_anomaly_ilocs = []
            for cluster_id in sorted(np.unique(clustering.labels_)):
                if cluster_id == -1: continue
                cluster_indices = raw_anomaly_indices[clustering.labels_ == cluster_id]
                # 获取残差值时，需要使用原始索引
                representative_idx = cluster_indices[np.argmax(residuals.iloc[cluster_indices])]
                final_anomaly_ilocs.append(representative_idx)
            final_anomaly_ilocs = sorted(final_anomaly_ilocs)
            print(f"   -> 时间聚类后异常数量: {len(final_anomaly_ilocs)}")
        else:
            final_anomaly_ilocs = raw_anomaly_indices.tolist()

        if return_scores:
            return final_anomaly_ilocs, residuals
        return final_anomaly_ilocs

    def get_feature_importance(self, top_n=15):
        if not self.model:
            raise RuntimeError("Model is not fitted yet.")
        importances = self.model.feature_importances_
        importance_df = pd.DataFrame({'feature': self.feature_names, 'importance': importances})
        return importance_df.sort_values(by='importance', ascending=False).head(top_n)

    def evaluate_performance(self, features_df):
        if not self.model:
            raise RuntimeError("Model is not fitted yet.")
        
        X = features_df[self.feature_names]
        y = features_df[self.target_column]
        
        y_pred = self.model.predict(X)
        r2 = self.model.score(X, y)
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        return {
            "r2_score": r2, "mse": mse, "mae": mae,
            "threshold_factor": self.optimized_threshold_factor,
            "threshold_value": self.residual_threshold,
            "satellite_type": self.satellite_type
        }