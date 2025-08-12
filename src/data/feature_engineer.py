# src/data/feature_engineer.py

import pandas as pd
import numpy as np
from typing import List, Optional, Dict
from scipy import stats

class EnhancedSatelliteFeatureEngineer:
    """
Usage:
1. Initialization: `engineer = EnhancedSatelliteFeatureEngineer(...)`
2. Fitting: `engineer.fit(train_df, satellite_name='...')` (for learning global statistics and satellite type)
3. Transformation: `features = engineer.transform(df)` (for creating features)

Note: This class no longer creates the 'target' column or fills NaN values.
These steps should be handled in the main training process to maintain the generalizability of feature engineering.
    """

    EPSILON = 1e-9
    GEO_NAMES = ['fengyun', 'goes', 'meteosat']
    VOLATILITY_BREAKOUT_FACTOR = 1.5
    SHARP_CHANGE_STD_MULTIPLIER = 2.0
    JUMP_DETECTION_QUANTILE = 0.95

    def __init__(self,
                 target_column: str = 'mean_motion',
                 additional_columns: Optional[List[str]] = None,
                 lag_features: List[int] = [1, 2, 3, 5, 7, 14],
                 rolling_windows: List[int] = [3, 7, 14, 30],
                 trend_windows: List[int] = [7, 14, 30],
                 satellite_type: str = 'auto'):
        
        self.target_column = target_column
        self.additional_columns = additional_columns or []
        self.base_process_cols = [self.target_column] + self.additional_columns
        
        self.lag_features = lag_features
        self.rolling_windows = rolling_windows
        self.trend_windows = trend_windows
        
        self.satellite_type = satellite_type
        self.global_stats = {}
        self._all_process_cols = []

    def fit(self, data: pd.DataFrame, satellite_name: str = None) -> 'EnhancedSatelliteFeatureEngineer':
        """
        Learn global statistics based on input data and determine the satellite type.
        """
        print("Fitting Feature Engineer...")
        # 1. Determine the satellite type
        if self.satellite_type == 'auto':
            self.satellite_type = self._determine_satellite_type(satellite_name)
        print(f"   -> Determined satellite type: {self.satellite_type}")

        # 2. Calculate and store global statistics
        self.global_stats = {}
        for col in self.base_process_cols:
            if col in data.columns:
                series = data[col].dropna()
                self.global_stats[col] = {
                    'mean': series.mean(),
                    'std': series.std(),
                    'median': series.median(),
                    'q25': series.quantile(0.25),
                    'q75': series.quantile(0.75)
                }
        print(f"   -> Global stats computed for: {list(self.global_stats.keys())}")
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering steps to the input data.
        """
        print("Transforming data to create features...")
        features_df = data.copy()
        
        ## Use a list to collect the DataFrame of all new features and finally merge them together
        all_new_features = []


        vector_features_df = self._create_orbital_vector_features(features_df)
        if not vector_features_df.empty:

            self._all_process_cols = list(set(self.base_process_cols + vector_features_df.columns.tolist()))
            all_new_features.append(vector_features_df)
        else:
            self._all_process_cols = self.base_process_cols

        # Create enhanced timing features for each core parameter
        print(f"   -> Creating enhanced time-series features for {self._all_process_cols}...")
        for col in self._all_process_cols:
            if col in features_df.columns:
                col_features_df = self._create_enhanced_features_for_col(features_df, col)
                all_new_features.append(col_features_df)
        
        # Temporary complete feature DF, used to calculate cross features
        temp_full_df = pd.concat([features_df] + all_new_features, axis=1)

        #Creating Intersecting Element Features
        print("   -> Creating cross-element features...")
        cross_features_df = self._create_cross_element_features(temp_full_df)
        all_new_features.append(cross_features_df)

        # Add specific features based on satellite type
        if self.satellite_type == 'GEO':
            print("   -> Adding GEO-specific features...")
            geo_features_df = self._add_geo_specific_features(temp_full_df)
            all_new_features.append(geo_features_df)
        else:  # LEO
            print("   -> Adding LEO-specific features...")
            leo_features_df = self._add_leo_specific_features(temp_full_df)
            all_new_features.append(leo_features_df)

        # Finally, all features are merged
        print("   -> Finalizing feature set.")
        final_df = pd.concat([features_df] + all_new_features, axis=1)
        
        # Remove Duplicate Columns
        final_df = final_df.loc[:, ~final_df.columns.duplicated()]

        print(f"✅ Feature engineering complete. Final shape: {final_df.shape}")
        return final_df

    def _determine_satellite_type(self, satellite_name: str) -> str:
        if satellite_name and any(name in satellite_name.lower() for name in self.GEO_NAMES):
            return 'GEO'
        return 'LEO'

    def _create_enhanced_features_for_col(self, df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        """Create all time series features for a single column, returning a DataFrame containing only the new features."""
        new_cols = {}
        
        # Differential features
        for n in [1, 2, 7]:
            new_cols[f'{col_name}_diff_{n}'] = df[col_name].diff(n)

        # Hysteresis characteristics
        for lag in self.lag_features:
            new_cols[f'{col_name}_lag_{lag}'] = df[col_name].shift(lag)

        # Rolling statistical features
        for window in self.rolling_windows:
            rolling = df[col_name].rolling(window=window, min_periods=1)
            mean = rolling.mean()
            std = rolling.std()
            new_cols[f'{col_name}_rolling_mean_{window}'] = mean
            new_cols[f'{col_name}_rolling_std_{window}'] = std
            new_cols[f'{col_name}_rolling_cv_{window}'] = std / (mean + self.EPSILON)
        
        # Trend, Volatility, and Relative Position Characteristics
        new_cols.update(self._calculate_trend_features(df, col_name))
        new_cols.update(self._calculate_volatility_features(df, col_name, new_cols))
        new_cols.update(self._calculate_relative_position_features(df, col_name))
        
        return pd.DataFrame(new_cols, index=df.index)

    def _calculate_trend_features(self, df: pd.DataFrame, col_name: str) -> Dict[str, pd.Series]:
        """Calculate trend characteristics."""
        trend_cols = {}
        for window in self.trend_windows:
            def trend_slope(x):
                if x.count() < 3: return 0
                return stats.linregress(np.arange(len(x)), x.dropna())[0]

            trend_cols[f'{col_name}_trend_slope_{window}'] = df[col_name].rolling(window, min_periods=3).apply(trend_slope, raw=False)
        return trend_cols

    def _calculate_volatility_features(self, df: pd.DataFrame, col_name: str, existing_new_cols: Dict) -> Dict[str, pd.Series]:
        """Calculate volatility characteristics."""
        vol_cols = {}
        std_7_col = f'{col_name}_rolling_std_7'
        std_30_col = f'{col_name}_rolling_std_30'
        
        # Make sure the dependent columns are calculated
        series_std_7 = existing_new_cols.get(std_7_col)
        series_std_30 = existing_new_cols.get(std_30_col)
        
        if series_std_7 is not None and series_std_30 is not None:
            vol_cols[f'{col_name}_volatility_breakout'] = (series_std_7 > series_std_30 * self.VOLATILITY_BREAKOUT_FACTOR).astype(int)
        
        return vol_cols

    def _calculate_relative_position_features(self, df: pd.DataFrame, col_name: str) -> Dict[str, pd.Series]:
        """Calculate relative position features."""
        pos_cols = {}
        if col_name in self.global_stats:
            stats_dict = self.global_stats[col_name]
            if stats_dict['std'] > self.EPSILON:
                pos_cols[f'{col_name}_rel_to_mean'] = (df[col_name] - stats_dict['mean']) / stats_dict['std']
            pos_cols[f'{col_name}_above_q75'] = (df[col_name] > stats_dict['q75']).astype(int)
            pos_cols[f'{col_name}_below_q25'] = (df[col_name] < stats_dict['q25']).astype(int)
        return pos_cols
        
    def _create_cross_element_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Creates a cross element feature."""
        new_cols = {}
        if 'mean_motion' in df.columns and 'eccentricity' in df.columns:
            new_cols['mm_div_ecc'] = df['mean_motion'] / (df['eccentricity'] + self.EPSILON)
            new_cols['energy_shape_factor'] = (df['mean_motion']**2) * (1 - df['eccentricity']**2)
            
        if 'eccentricity' in df.columns and 'inclination' in df.columns:
            new_cols['ecc_mul_inc'] = df['eccentricity'] * df['inclination']
        
        if 'inclination_diff_1' in df.columns and 'eccentricity_diff_1' in df.columns:
            new_cols['inc_ecc_change_sync'] = df['inclination_diff_1'] * df['eccentricity_diff_1']
        
        return pd.DataFrame(new_cols, index=df.index)

    def _add_geo_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add drift and pressure features for GEO satellites."""
        new_cols = {}
        long_window = 60
        short_window = 14
        
        for col in self.base_process_cols:
            if col not in df.columns: continue
            
            long_term_mean = df[col].rolling(window=long_window, min_periods=long_window//2).mean()
            drift = df[col] - long_term_mean
            new_cols[f'{col}_drift_from_long_mean'] = drift
            new_cols[f'{col}_cumulative_drift_{short_window}d'] = drift.rolling(window=short_window).sum()
            
            slope_col = f'{col}_trend_slope_7'
            if slope_col in df.columns:
                new_cols[f'{col}_drift_acceleration_7d'] = df[slope_col].diff()
        
        return pd.DataFrame(new_cols, index=df.index)

    def _add_leo_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds sharp change and event history features for LEO satellites."""
        new_cols = {}
        for col in self.base_process_cols:
            if col not in df.columns: continue
            
            diff_1_col = f'{col}_diff_1'
            std_7_col = f'{col}_rolling_std_7'
            
            if diff_1_col in df.columns and std_7_col in df.columns:
                sharp_change = (abs(df[diff_1_col]) > df[std_7_col] * self.SHARP_CHANGE_STD_MULTIPLIER).astype(int)
                new_cols[f'{col}_sharp_change'] = sharp_change
                new_cols[f'{col}_consecutive_sharp_change_3d'] = sharp_change.rolling(3).sum()

            # Calculate the number of days since the last jump
            col_diff_abs = df[col].diff().abs()
            jump_threshold = col_diff_abs.quantile(self.JUMP_DETECTION_QUANTILE)
            events = (col_diff_abs > jump_threshold).astype(int)
            
            # Find the index of each event
            event_indices = pd.Series(np.arange(len(df)), index=df.index)
            event_indices[events == 0] = np.nan
            event_indices = event_indices.ffill()
            
            new_cols[f'{col}_days_since_last_jump'] = pd.Series(np.arange(len(df)), index=df.index) - event_indices

        return pd.DataFrame(new_cols, index=df.index)

    def _create_orbital_vector_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create an Orbital Vector feature (sin/cos) to handle angular periodicity."""
        new_cols = {}
        angle_cols = { "inclination": "inclination", "raan": "raan", "argp": "argument_perigee", "M": "mean_anomaly" }
        
        present_angles = {key: alias for key, alias in angle_cols.items() if alias in df.columns}
        
        def to_radians(s: pd.Series) -> pd.Series:
            return np.deg2rad(s) if s.abs().max() > 2 * np.pi else s

        for key, col_name in present_angles.items():
            angle_rad = to_radians(df[col_name])
            new_cols[f"{col_name}_sin"] = np.sin(angle_rad)
            new_cols[f"{col_name}_cos"] = np.cos(angle_rad)

        if "eccentricity" in df.columns and "argument_perigee" in df.columns:
            e = df["eccentricity"]
            argp_rad = to_radians(df["argument_perigee"])
            new_cols["e_cosw"] = e * np.cos(argp_rad)
            new_cols["e_sinw"] = e * np.sin(argp_rad)
            
        return pd.DataFrame(new_cols, index=df.index)