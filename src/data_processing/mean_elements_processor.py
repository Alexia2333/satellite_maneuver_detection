# src/data/mean_elements_processor.py
"""
平根数处理器 - 无监督机动检测的核心数据处理模块
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Dict, List, Optional, Union
import warnings

class MeanElementsProcessor:
    """
    专门处理TLE平根数的核心类 (【最终修复版 - 强制统一数据类型】)
    """
    
    def __init__(self, satellite_type: str = 'auto', 
                 interpolation_method: str = 'linear',
                 outlier_detection: bool = True):
        self.satellite_type = satellite_type
        self.interpolation_method = interpolation_method
        self.outlier_detection = outlier_detection
        self.processing_stats = {}
        # ... (其他初始化不变)

    def process_tle_data(self, tle_data: pd.DataFrame, 
                        satellite_name: str = None) -> pd.DataFrame:
        """
        处理原始TLE数据，提取和清理平根数
        """
        print(f"\n🛠️ Processing TLE data (Final Fix - Enforcing uniform dtype)...")
        
        if self.satellite_type == 'auto':
            self.satellite_type = self._detect_satellite_type(satellite_name)
        
        validated_data = self._validate_tle_data(tle_data)
        mean_elements = self._extract_mean_elements(validated_data)
        
        if self.outlier_detection:
            mean_elements = self._detect_and_mark_outliers(mean_elements)
        
        processed_data = self._ensure_time_continuity(mean_elements)
        
        if not processed_data.empty:
            processed_data = processed_data.interpolate(method=self.interpolation_method, limit_direction='both')
            processed_data.ffill(inplace=True)
            processed_data.bfill(inplace=True)

        if self.satellite_type == 'GEO':
            processed_data = self._apply_geo_specific_processing(processed_data)
            
        self._generate_processing_stats(tle_data, processed_data)
        
        print(f"✅ Mean elements processing completed")
        return processed_data

    def _extract_mean_elements(self, tle_data: pd.DataFrame) -> pd.DataFrame:
        """【已修复】提取平根数并强制统一数据类型为float64"""
        mean_elements = pd.DataFrame()
        mean_elements['epoch'] = tle_data['epoch']
        
        all_elements = ['mean_motion', 'eccentricity', 'inclination', 'argument of perigee', 
                        'right ascension', 'mean_anomaly']
        
        for element in all_elements:
            if element in tle_data.columns:
                clean_name = element.replace(' ', '_').replace('of_', '')
                # 【核心修复】强制将所有列转换为统一的 float64 类型，防止 reindex 出错
                mean_elements[clean_name] = tle_data[element].astype(np.float64)
        
        return mean_elements.set_index('epoch')

    def _ensure_time_continuity(self, data: pd.DataFrame) -> pd.DataFrame:
        """【保持原样】使用 reindex 进行重采样"""
        if data.empty or not isinstance(data.index, pd.DatetimeIndex):
            return data
        
        time_diffs = data.index.to_series().diff()
        median_interval = time_diffs.median()
        
        if pd.notna(median_interval) and median_interval.total_seconds() > 0:
            print("   -> Resampling data using reindex method...")

            data.index = data.index.normalize()
            regular_index = pd.date_range(start=data.index.min(), end=data.index.max(), freq='1D')
            return data.reindex(regular_index)
        print("原始 index 示例:", data.index[:5])    
        print("标准化后 index 示例:", data.index[:5])
        print("目标 index 示例:", regular_index[:5])    
        return data

    def _detect_and_mark_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        cleaned_data = data.copy()
        outlier_counts = {}
        for column in data.select_dtypes(include=[np.number]).columns:
            if data[column].notna().sum() < 10: continue
            Q1, Q3 = data[column].quantile(0.25), data[column].quantile(0.75)
            IQR = Q3 - Q1
            if IQR == 0: IQR = np.std(data[column]) * 0.1 if np.std(data[column]) > 0 else 1e-5
            lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            outliers = (data[column] < lower_bound) | (data[column] > upper_bound)
            outlier_count = outliers.sum()
            outlier_counts[column] = outlier_count
            if outlier_count > 0:
                print(f"   -> Found {outlier_count} outliers in '{column}', marking as NaN.")
                cleaned_data.loc[outliers, column] = np.nan
        self.processing_stats['outliers_detected'] = outlier_counts
        return cleaned_data

    def _detect_satellite_type(self, s_name: str) -> str:
        if not s_name: return 'LEO'
        s_name_lower = s_name.lower()
        if any(i in s_name_lower for i in ['fengyun', 'fy-', 'goes', 'meteosat']): return 'GEO'
        return 'LEO'

    def _validate_tle_data(self, tle_data: pd.DataFrame) -> pd.DataFrame:
        df = tle_data.copy().drop_duplicates(subset=['epoch'])
        if 'epoch' not in df.columns: raise ValueError("'epoch' column missing")
        df['epoch'] = pd.to_datetime(df['epoch'])
        return df.sort_values('epoch').reset_index(drop=True)

    def _apply_geo_specific_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if 'mean_motion' in df.columns and df['mean_motion'].notna().any():
            df['geo_drift'] = df['mean_motion'] - 1.00273790945
            if len(df) > 30:
                df['long_term_drift'] = df['geo_drift'].rolling(30, center=True, min_periods=1).mean()
        return df
    
    def _generate_processing_stats(self, original_data: pd.DataFrame, processed_data: pd.DataFrame):
        stats = {
            'original_records': len(original_data), 'processed_records': len(processed_data),
            'time_span_days': (processed_data.index.max() - processed_data.index.min()).days if not processed_data.empty else 0,
            'satellite_type': self.satellite_type, 'processing_method': 'mean_elements_only (fixed)',
            'columns_processed': list(processed_data.columns)
        }
        for column in processed_data.select_dtypes(include=[np.number]).columns:
            stats[f'{column}_mean'] = processed_data[column].mean()
            stats[f'{column}_std'] = processed_data[column].std()
            stats[f'{column}_missing_rate'] = processed_data[column].isnull().sum() / len(processed_data) if len(processed_data) > 0 else 0
        self.processing_stats.update(stats)
        print("\n" + self.get_processing_report())

    def get_processing_report(self) -> str:
        if not self.processing_stats: return "No processing statistics available."
        report = f"📊 Mean Elements Processing Report\n=====================================\n"
        report += f"Satellite Type: {self.processing_stats.get('satellite_type', 'N/A')}\n"
        # ... (rest of the function is the same)
        return report