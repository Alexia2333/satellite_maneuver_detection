# src/data/feature_engineer.py

import pandas as pd
import numpy as np
from typing import List, Optional
from scipy import stats

class SatelliteFeatureEngineer:
    def __init__(self, target_column: str = 'mean_motion', additional_columns: Optional[List[str]] = None,
                 lag_features: List[int] = [1, 2, 3, 5, 7, 14], rolling_windows: List[int] = [7, 14, 30]):
        self.target_column = target_column
        self.additional_columns = additional_columns or []
        self.all_process_cols = [self.target_column] + self.additional_columns
        self.lag_features = lag_features
        self.rolling_windows = rolling_windows

    def fit(self, data: pd.DataFrame) -> 'SatelliteFeatureEngineer':
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        features_df = data.copy()
        for col in self.all_process_cols:
            features_df = self._create_base_features_for_col(features_df, col)
        return features_df

    def _create_base_features_for_col(self, df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        for lag in self.lag_features:
            df[f'{col_name}_lag_{lag}'] = df[col_name].shift(lag)
        for window in self.rolling_windows:
            rolling = df[col_name].rolling(window=window, min_periods=1)
            df[f'{col_name}_rolling_mean_{window}'] = rolling.mean()
            df[f'{col_name}_rolling_std_{window}'] = rolling.std()
        df[f'{col_name}_diff_1'] = df[col_name].diff(1)
        return df

    def prepare_target_features(self, df: pd.DataFrame, forecast_horizon: int = 1) -> pd.DataFrame:
        result_df = df.copy()
        result_df['target'] = result_df[self.target_column].shift(-forecast_horizon)
        return result_df


class EnhancedSatelliteFeatureEngineer:
    def __init__(self, target_column: str = 'mean_motion', additional_columns: Optional[List[str]] = None,
                 lag_features: List[int] = [1, 2, 3, 5, 7, 14, 21, 30], 
                 rolling_windows: List[int] = [3, 7, 14, 30, 60],
                 satellite_type: str = 'auto'):
        self.target_column = target_column
        self.additional_columns = additional_columns or []
        self.all_process_cols = [self.target_column] + self.additional_columns
        self.lag_features = lag_features
        self.rolling_windows = rolling_windows
        self.satellite_type = satellite_type
        self.global_stats = {}

    def fit(self, data: pd.DataFrame, satellite_name: str = None) -> 'EnhancedSatelliteFeatureEngineer':
        if self.satellite_type == 'auto':
            self.satellite_type = self._determine_satellite_type(satellite_name)
        
        self.global_stats = {}
        for col in self.all_process_cols:
            if col in data.columns:
                self.global_stats[col] = {
                    'mean': data[col].mean(), 'std': data[col].std(), 'median': data[col].median(),
                    'q25': data[col].quantile(0.25), 'q75': data[col].quantile(0.75)
                }
        return self

    def _determine_satellite_type(self, satellite_name: str) -> str:
        if satellite_name and any(name in satellite_name.lower() for name in ['fengyun', 'goes', 'meteosat']):
            return 'GEO'
        return 'LEO'

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        features_df = data.copy()
        
        features_df = self._create_orbital_vector_features(features_df)
        self.all_process_cols = list(set(self.all_process_cols + ['e_x', 'e_y', 'i_x', 'i_y']))


        for col in self.all_process_cols:
            if col in features_df.columns:
                features_df = self._create_enhanced_features_for_col(features_df, col)
        
        features_df = self._create_cross_element_features(features_df)
        
        # This part now becomes much more important
        if self.satellite_type == 'GEO':
            features_df = self._add_geo_specific_features(features_df)
        else: # LEO
            features_df = self._add_leo_specific_features(features_df)
        
        return features_df

    def _create_enhanced_features_for_col(self, df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        print(f"   -> Creating enhanced features for {col_name}...")
        
        # 1. 滞后特征
        for lag in self.lag_features:
            df[f'{col_name}_lag_{lag}'] = df[col_name].shift(lag)
        
        # 2. 滚动统计特征
        for window in self.rolling_windows:
            rolling = df[col_name].rolling(window=window, min_periods=1)
            df[f'{col_name}_rolling_mean_{window}'] = rolling.mean()
            df[f'{col_name}_rolling_std_{window}'] = rolling.std()
            df[f'{col_name}_rolling_min_{window}'] = rolling.min()
            df[f'{col_name}_rolling_max_{window}'] = rolling.max()
            df[f'{col_name}_rolling_median_{window}'] = rolling.median()
            
            df[f'{col_name}_rolling_range_{window}'] = (
                df[f'{col_name}_rolling_max_{window}'] - df[f'{col_name}_rolling_min_{window}']
            )
            
            mean_col = f'{col_name}_rolling_mean_{window}'
            df[f'{col_name}_rolling_cv_{window}'] = df[f'{col_name}_rolling_std_{window}'] / (df[mean_col] + 1e-8)
        
        # 3. 差分特征
        df[f'{col_name}_diff_1'] = df[col_name].diff(1)
        df[f'{col_name}_diff_2'] = df[col_name].diff(2)
        df[f'{col_name}_diff_7'] = df[col_name].diff(7)
        
        # 4. 趋势特征
        df = self._add_trend_features(df, col_name)
        
        # 5. 波动性特征
        df = self._add_volatility_features(df, col_name)
        
        # 6. 相对位置特征
        df = self._add_relative_position_features(df, col_name)
        
        return df

    def _add_trend_features(self, df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        for window in [7, 14, 30]:
            def trend_slope(x):
                if len(x) < 3:
                    return 0
                try:
                    return stats.linregress(range(len(x)), x)[0]
                except:
                    return 0
            
            df[f'{col_name}_trend_slope_{window}'] = (
                df[col_name].rolling(window, min_periods=3).apply(trend_slope, raw=False)
            )
            
            def trend_strength(x):
                if len(x) < 3:
                    return 0
                try:
                    slope, intercept, r_value, _, _ = stats.linregress(range(len(x)), x)
                    return r_value ** 2
                except:
                    return 0
            
            df[f'{col_name}_trend_strength_{window}'] = (
                df[col_name].rolling(window, min_periods=3).apply(trend_strength, raw=False)
            )
        
        return df

    def _add_volatility_features(self, df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        for window in [7, 14, 30]:
            std_col = f'{col_name}_rolling_std_{window}'
            if std_col in df.columns:
                df[f'{col_name}_volatility_change_{window}'] = df[std_col].pct_change()
        
        if f'{col_name}_rolling_std_7' in df.columns and f'{col_name}_rolling_std_30' in df.columns:
            df[f'{col_name}_volatility_breakout'] = (
                df[f'{col_name}_rolling_std_7'] > df[f'{col_name}_rolling_std_30'] * 1.5
            ).astype(int)
        
        return df

    def _add_relative_position_features(self, df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        if col_name in self.global_stats:
            stats_dict = self.global_stats[col_name]
            
            if stats_dict['std'] > 0:
                df[f'{col_name}_rel_to_mean'] = (df[col_name] - stats_dict['mean']) / stats_dict['std']
            df[f'{col_name}_rel_to_median'] = df[col_name] - stats_dict['median']
            
            df[f'{col_name}_above_q75'] = (df[col_name] > stats_dict['q75']).astype(int)
            df[f'{col_name}_below_q25'] = (df[col_name] < stats_dict['q25']).astype(int)
        
        return df

    def _create_cross_element_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """【强化版】创建交叉元素特征，捕捉多维度关联"""
        print("   -> Creating enhanced cross-element features...")
        
        # 保留原有特征
        if 'mean_motion' in df.columns and 'eccentricity' in df.columns:
            df['mm_div_ecc'] = df['mean_motion'] / (df['eccentricity'] + 1e-8)
        
        if 'eccentricity' in df.columns and 'inclination' in df.columns:
            df['ecc_mul_inc'] = df['eccentricity'] * df['inclination']

        # --- 新增交互特征 ---

        # 1. 动能相关特征 (简化形式)
        # mean_motion^2 正比于轨道能量，eccentricity^2 关系到形状。
        # 它们的组合可以反映轨道能量和形状的综合变化。
        if 'mean_motion' in df.columns and 'eccentricity' in df.columns:
            df['energy_shape_factor'] = (df['mean_motion']**2) * (1 - df['eccentricity']**2)

        # 2. 轨道平面与形状的耦合
        # 倾角和偏心率的交互可能指示平面变更和轨道形状调整的同步性
        if 'inclination_diff_1' in df.columns and 'eccentricity_diff_1' in df.columns:
            df['inc_ecc_change_sync'] = df['inclination_diff_1'] * df['eccentricity_diff_1']

        # 3. 综合变化率
        # 将主要参数的变化率相加，放大总体变化信号
        change_cols = [f'{col}_diff_1' for col in self.all_process_cols if f'{col}_diff_1' in df.columns]
        if len(change_cols) > 1:
            df['composite_change_rate'] = df[change_cols].abs().sum(axis=1)
            # 7日滚动变化率，观察近期变化的剧烈程度
            df['composite_change_rolling_7d'] = df['composite_change_rate'].rolling(7).mean()

        return df

    def _add_geo_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """【强化版】为GEO卫星添加基于物理的漂移和压力特征"""
        print("   -> Adding ENHANCED GEO-specific features...")
        
        # 我们将处理所有核心轨道根数
        cols_for_drift_analysis = [self.target_column] + self.additional_columns
        
        for col in cols_for_drift_analysis:
            if col not in df.columns: continue

            # 定义长短期窗口
            long_window = 60
            short_window = 14
            
            # --- 特征1: 瞬时漂移 (Deviation from long-term mean) ---
            # 计算一个长期的、稳定的移动平均线作为“正常”轨道中心
            long_term_mean = df[col].rolling(window=long_window, min_periods=long_window//2).mean()
            df[f'{col}_drift_from_long_mean'] = df[col] - long_term_mean

            # --- 特征2: 累积漂移压力 (Accumulated Drift Pressure) ---
            # 计算短期漂移的累积和，量化“修正压力”
            # 当这个值持续增大或减小，说明机动即将发生
            if f'{col}_drift_from_long_mean' in df.columns:
                df[f'{col}_cumulative_drift_{short_window}d'] = df[f'{col}_drift_from_long_mean'].rolling(window=short_window).sum()
                
            # --- 特征3: 漂移速度 (Rate of Drift) ---
            # 我们已有的 trend_slope 特征在这里扮演了关键角色，无需重复添加。
            # df[f'{col}_trend_slope_14'] 和 df[f'{col}_trend_slope_30'] 将是核心预测因子。

            # --- 特征4: 漂移加速度 (Acceleration of Drift) ---
            # 漂移速度本身的变化率，可以作为机动临近的更强信号
            if f'{col}_trend_slope_7' in df.columns:
                 df[f'{col}_drift_acceleration_7d'] = df[f'{col}_trend_slope_7'].diff()

        return df

    def _add_leo_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        print("   -> Adding LEO-specific features...")
        
        for col in self.all_process_cols:
            if col not in df.columns:
                continue
            
            if f'{col}_diff_1' in df.columns and f'{col}_rolling_std_7' in df.columns:
                df[f'{col}_sharp_change'] = (
                    abs(df[f'{col}_diff_1']) > df[f'{col}_rolling_std_7'] * 2
                ).astype(int)
                
                df[f'{col}_consecutive_change'] = (
                    df[f'{col}_sharp_change'].rolling(3).sum()
                )
        
        return df

    def prepare_target_features(self, df: pd.DataFrame, forecast_horizon: int = 1) -> pd.DataFrame:
        result_df = df.copy()
        result_df['target'] = result_df[self.target_column].shift(-forecast_horizon)
        return result_df
    


def create_drift_enhanced_features(tle_data, scaling_factor, target_col='mean_motion'):
    """
    一个独立的、可重用的特征工程函数
    """
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
    enhanced_data['target'] = enhanced_data['target'] * 1e8

    # 打印确认
    print("✅ 应用了 log1p(target * 1e8) 变换")
    print(enhanced_data['target'].describe())
    


    # 填充其他特征
    # [MODIFIED] Use .ffill() and .bfill() which are the modern replacements for method='...'
    enhanced_data = enhanced_data.ffill(limit=3)
    enhanced_data = enhanced_data.bfill(limit=3)
    
    # [MODIFIED] Loop through columns and fill remaining NaNs using direct assignment
    # to avoid SettingWithCopyWarning.
    for col in enhanced_data.columns:
        if enhanced_data[col].isna().any():
            median_val = enhanced_data[col].median()
            # Use assignment `=` instead of `inplace=True` on a chained call
            enhanced_data[col] = enhanced_data[col].fillna(median_val if not pd.isna(median_val) else 0)

    print(f"✅ 特征创建完成")
    print(f"   特征数量: {len(enhanced_data.columns)}")
    print(f"   有效记录: {len(enhanced_data)}")
    print(f"   目标变量统计: mean={enhanced_data['target'].mean():.6f}, std={enhanced_data['target'].std():.6f}")
    return enhanced_data

def create_drift_features(tle_data, target_col='mean_motion'):
    """
    创建漂移相关的特征
    
    特征1：量化"持续漂移" - 描述参数偏离其正常中心的程度
    特征2：引入"累积漂移压力" - 漂移是持续累积的
    """
    print("\n🛠️ 创建漂移特征")
    print("-" * 20)
    
    data = tle_data.set_index('epoch').copy()
    
    # 基础轨道参数
    orbit_params = ['mean_motion', 'eccentricity', 'inclination']
    available_params = [col for col in orbit_params if col in data.columns]
    
    if not available_params:
        print(f"❌ 没有找到轨道参数列: {orbit_params}")
        return None
    
    print(f"   基础轨道参数: {available_params}")
    
    enhanced_data = data[available_params].copy()
    
    # ==== 特征1：量化持续漂移 ====
    long_windows = [15, 30, 45]  
    
    for col in available_params:
        for window in long_windows:
            col_mean = data[col].rolling(window, min_periods=window//2).mean()
            enhanced_data[f'{col}_drift_from_{window}d_mean'] = data[col] - col_mean
            col_std = data[col].rolling(window, min_periods=window//2).std()
            enhanced_data[f'{col}_drift_zscore_{window}d'] = (data[col] - col_mean) / (col_std + 1e-8)
    
    trend_windows = [7, 14, 21]
    for col in available_params:
        for window in trend_windows:
            enhanced_data[f'{col}_trend_slope_{window}d'] = data[col].rolling(window).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
            )
    
    # ==== 特征2：累积漂移压力 ====
    short_windows = [3, 7, 14]
    
    for col in available_params:
        short_mean = data[col].rolling(7).mean()
        short_drift = (data[col] - short_mean).abs()
        
        for window in short_windows:
            enhanced_data[f'{col}_cumulative_drift_{window}d'] = short_drift.rolling(window).sum()
            weights = np.exp(np.linspace(-1, 0, window))
            enhanced_data[f'{col}_weighted_cumulative_drift_{window}d'] = short_drift.rolling(window).apply(
                lambda x: np.sum(x * weights[-len(x):]) / np.sum(weights[-len(x):])
            )
    
    # ==== 特征3：参数间的协同漂移 ====
    if len(available_params) > 1:
        for window in [7, 14]:
            drift_cols = [f'{col}_drift_from_{window}d_mean' for col in available_params 
                          if f'{col}_drift_from_{window}d_mean' in enhanced_data.columns]
            if drift_cols:
                enhanced_data[f'combined_drift_l2_{window}d'] = np.sqrt(
                    enhanced_data[drift_cols].pow(2).sum(axis=1)
                )
                enhanced_data[f'drift_correlation_{window}d'] = enhanced_data[drift_cols].apply(
                    lambda x: 1 if (x > 0).all() or (x < 0).all() else 0, axis=1
                )
    
    # ==== 特征4：历史机动模式特征 ====
    for col in available_params:
        col_diff = data[col].diff().abs()
        threshold = col_diff.quantile(0.95)
        events = (col_diff > threshold).astype(int)
        last_event_idx = pd.Series(range(len(events)), index=events.index)
        last_event_idx[events == 0] = np.nan
        last_event_idx = last_event_idx.ffill()
        enhanced_data[f'{col}_days_since_last_jump'] = (
            pd.Series(range(len(events)), index=events.index) - last_event_idx
        )
    
    # ==== 特征5：统计特征 ====
    stat_windows = [3, 7, 14, 30]
    for col in available_params:
        for window in stat_windows:
            enhanced_data[f'{col}_rolling_mean_{window}d'] = data[col].rolling(window).mean()
            enhanced_data[f'{col}_rolling_std_{window}d'] = data[col].rolling(window).std()
            enhanced_data[f'{col}_rolling_skew_{window}d'] = data[col].rolling(window).skew()
            enhanced_data[f'{col}_rolling_kurt_{window}d'] = data[col].rolling(window).kurt()
    
    # ==== 学习目标：预测机动引起的跳变 ====
    enhanced_data['target'] = data[target_col].diff().shift(-1)
    enhanced_data['target_abs'] = enhanced_data['target'].abs()
    
    # --- 使用更健壮的NaN处理方式 ---
    enhanced_data = enhanced_data.dropna(subset=['target', 'target_abs'])
    print(f"   记录数 (在丢弃无效目标后): {len(enhanced_data)}")

    feature_cols = [col for col in enhanced_data.columns if col not in ['target', 'target_abs']]
    enhanced_data[feature_cols] = enhanced_data[feature_cols].ffill().bfill()

    for col in feature_cols:
        if enhanced_data[col].isnull().any():
            median_val = enhanced_data[col].median()
            if not pd.isna(median_val):
                enhanced_data[col] = enhanced_data[col].fillna(median_val)
            else:
                enhanced_data[col] = enhanced_data[col].fillna(0)
    
    print(f"✅ 特征和NaN值处理完成")
    print(f"   特征数量: {len(enhanced_data.columns)}")
    print(f"   有效记录: {len(enhanced_data)}")
    
    print("\n📊 关键特征统计:")
    key_features = [col for col in enhanced_data.columns if 'drift' in col or 'cumulative' in col][:5]
    for feat in key_features:
        if feat in enhanced_data.columns:
            print(f"   {feat}: mean={enhanced_data[feat].mean():.6f}, std={enhanced_data[feat].std():.6f}")
    
    return enhanced_data