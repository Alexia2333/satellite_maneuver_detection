#!/usr/bin/env python
"""
Diagnostic script to debug why ARIMA detector isn't finding maneuvers.
Run this to understand the data and detection issues.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import SatelliteDataLoader


def diagnose_satellite_data(satellite_name="Jason-1", element="mean_motion"):
    """
    Comprehensive diagnostic of satellite data and detection issues.
    """
    print(f"\n{'='*60}")
    print(f"DIAGNOSTIC REPORT FOR {satellite_name}")
    print(f"{'='*60}")
    
    # Load data
    loader = SatelliteDataLoader()
    tle_data, maneuver_times = loader.load_satellite_data(satellite_name)
    
    if tle_data is None or tle_data.empty:
        print(f"❌ Failed to load data for {satellite_name}")
        return
    
    print(f"\n1. DATA OVERVIEW:")
    print(f"   - TLE records: {len(tle_data)}")
    print(f"   - Date range: {tle_data['epoch'].min()} to {tle_data['epoch'].max()}")
    print(f"   - Maneuvers: {len(maneuver_times)}")
    
    # Check element availability
    if element not in tle_data.columns:
        print(f"   ❌ Element '{element}' not found in data!")
        print(f"   Available columns: {list(tle_data.columns)}")
        return
    
    # Prepare time series
    ts_data = tle_data[['epoch', element]].set_index('epoch')
    ts_data = ts_data[element].resample('D').median().interpolate(method='time').dropna()
    
    print(f"\n2. TIME SERIES STATS:")
    print(f"   - Daily points: {len(ts_data)}")
    print(f"   - Mean: {ts_data.mean():.6f}")
    print(f"   - Std: {ts_data.std():.6f}")
    print(f"   - Min: {ts_data.min():.6f}")
    print(f"   - Max: {ts_data.max():.6f}")
    
    # Check for outliers
    q1, q3 = ts_data.quantile(0.25), ts_data.quantile(0.75)
    iqr = q3 - q1
    outliers = ((ts_data < q1 - 3*iqr) | (ts_data > q3 + 3*iqr)).sum()
    print(f"   - Outliers (3*IQR): {outliers}")
    
    # Analyze maneuvers
    print(f"\n3. MANEUVER ANALYSIS:")
    maneuver_datetimes = pd.to_datetime(maneuver_times)
    
    # Check which maneuvers fall within data range
    data_start, data_end = ts_data.index.min(), ts_data.index.max()
    valid_maneuvers = [m for m in maneuver_datetimes if data_start <= m <= data_end]
    print(f"   - Maneuvers in data range: {len(valid_maneuvers)}/{len(maneuver_times)}")
    
    if valid_maneuvers:
        print(f"   - First maneuver: {valid_maneuvers[0]}")
        print(f"   - Last maneuver: {valid_maneuvers[-1]}")
    
    # Check maneuver signal strength
    print(f"\n4. MANEUVER SIGNAL STRENGTH:")
    for i, m_time in enumerate(valid_maneuvers[:5]):  # Check first 5
        # Get values around maneuver
        before = ts_data[(ts_data.index >= m_time - timedelta(days=5)) & 
                        (ts_data.index < m_time)]
        after = ts_data[(ts_data.index > m_time) & 
                       (ts_data.index <= m_time + timedelta(days=5))]
        
        if len(before) > 0 and len(after) > 0:
            before_mean = before.mean()
            after_mean = after.mean()
            change = abs(after_mean - before_mean)
            change_pct = 100 * change / before_mean if before_mean != 0 else 0
            
            print(f"   Maneuver {i+1} ({m_time.date()}):")
            print(f"     Before: {before_mean:.6f}")
            print(f"     After:  {after_mean:.6f}")
            print(f"     Change: {change:.6f} ({change_pct:.2f}%)")
    
    # Test simple detection
    print(f"\n5. SIMPLE THRESHOLD DETECTION TEST:")
    
    # Calculate first differences
    diff = ts_data.diff().abs()
    diff_mean = diff.mean()
    diff_std = diff.std()
    
    # Try different thresholds
    for factor in [1.5, 2.0, 2.5, 3.0]:
        threshold = diff_mean + factor * diff_std
        detections = diff[diff > threshold]
        print(f"   Factor {factor}: {len(detections)} detections (threshold={threshold:.6f})")
    
    # Check data split
    print(f"\n6. DATA SPLIT CHECK:")
    
    # Create labels
    labels = pd.Series(0, index=ts_data.index)
    for m_time in maneuver_datetimes:
        mask = (ts_data.index >= m_time - timedelta(days=1)) & \
               (ts_data.index <= m_time + timedelta(days=1))
        labels[mask] = 1
    
    maneuver_indices = np.where(labels == 1)[0]
    
    if len(maneuver_indices) < 5:
        train_end = int(len(ts_data) * 0.7)
        val_end = int(len(ts_data) * 0.85)
    else:
        train_m = int(len(maneuver_indices) * 0.6)
        val_m = int(len(maneuver_indices) * 0.8)
        train_end = maneuver_indices[train_m]
        val_end = maneuver_indices[val_m]
    
    train_labels = labels.iloc[:train_end].sum()
    val_labels = labels.iloc[train_end:val_end].sum()
    test_labels = labels.iloc[val_end:].sum()
    
    print(f"   Train: {train_end} points, {train_labels} maneuver labels")
    print(f"   Val:   {val_end-train_end} points, {val_labels} maneuver labels")
    print(f"   Test:  {len(ts_data)-val_end} points, {test_labels} maneuver labels")
    
    # ARIMA residual test
    print(f"\n7. ARIMA RESIDUAL TEST:")
    from statsmodels.tsa.arima.model import ARIMA
    
    # Use a small sample for quick test
    sample_size = min(500, train_end)
    sample_data = ts_data.iloc[:sample_size]
    
    try:
        model = ARIMA(sample_data, order=(3, 1, 1))
        model_fit = model.fit(method='css', maxiter=50)
        
        # Get residuals
        residuals = model_fit.resid.dropna()
        res_std = np.std(np.abs(residuals))
        
        print(f"   Sample size: {sample_size}")
        print(f"   Residual std: {res_std:.6f}")
        print(f"   Typical threshold (2.5*std): {2.5*res_std:.6f}")
        
        # Check if maneuvers would be detected
        test_start = sample_size
        test_end = min(sample_size + 100, len(ts_data))
        test_sample = ts_data.iloc[test_start:test_end]
        
        predictions = model_fit.predict(start=test_sample.index[0], 
                                       end=test_sample.index[-1])
        test_residuals = np.abs(test_sample - predictions)
        
        threshold = 2.5 * res_std
        anomalies = test_residuals[test_residuals > threshold]
        
        print(f"   Test sample: {len(test_sample)} points")
        print(f"   Anomalies found: {len(anomalies)}")
        
        if len(anomalies) > 0:
            print(f"   Max residual: {test_residuals.max():.6f}")
            print(f"   Anomaly dates: {[str(d.date()) for d in anomalies.index[:3]]}")
        
    except Exception as e:
        print(f"   ❌ ARIMA test failed: {e}")
    
    # Recommendations
    print(f"\n8. RECOMMENDATIONS:")
    
    if outliers > len(ts_data) * 0.05:
        print("   ⚠️  High number of outliers - consider more aggressive cleaning")
    
    if test_labels == 0:
        print("   ⚠️  No maneuvers in test set - adjust split strategy")
    
    if len(valid_maneuvers) < len(maneuver_times) * 0.5:
        print("   ⚠️  Many maneuvers outside data range - check data completeness")
    
    # Try lower threshold
    print("\n   💡 Try running with:")
    print("      - Lower threshold factor (1.5-2.0)")
    print("      - Different element (try 'eccentricity' or 'inclination')")
    print("      - Disable clustering temporarily")
    
    print(f"\n{'='*60}\n")


def plot_diagnostic(satellite_name="Jason-1", element="mean_motion"):
    """
    Create diagnostic plots.
    """
    loader = SatelliteDataLoader()
    tle_data, maneuver_times = loader.load_satellite_data(satellite_name)
    
    if tle_data is None:
        return
    
    # Prepare data
    ts_data = tle_data[['epoch', element]].set_index('epoch')
    ts_data = ts_data[element].resample('D').median().interpolate(method='time').dropna()
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    # Plot 1: Full time series with maneuvers
    ax1 = axes[0]
    ax1.plot(ts_data.index, ts_data.values, 'b-', linewidth=1)
    
    maneuver_datetimes = pd.to_datetime(maneuver_times)
    for m in maneuver_datetimes:
        if ts_data.index.min() <= m <= ts_data.index.max():
            ax1.axvline(m, color='red', alpha=0.5, linestyle='--')
    
    ax1.set_title(f'{satellite_name} - {element}')
    ax1.set_ylabel('Value')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: First differences
    ax2 = axes[1]
    diff = ts_data.diff()
    ax2.plot(diff.index, diff.values, 'g-', linewidth=0.5)
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    # Mark large changes
    threshold = diff.std() * 2
    ax2.axhline(y=threshold, color='r', linestyle='--', alpha=0.5)
    ax2.axhline(y=-threshold, color='r', linestyle='--', alpha=0.5)
    
    ax2.set_title('First Differences')
    ax2.set_ylabel('Change')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Rolling statistics
    ax3 = axes[2]
    rolling_std = ts_data.rolling(window=30).std()
    ax3.plot(rolling_std.index, rolling_std.values, 'orange', linewidth=1)
    
    ax3.set_title('30-Day Rolling Standard Deviation')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Std Dev')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{satellite_name}_diagnostic.png', dpi=150)
    print(f"Diagnostic plot saved to {satellite_name}_diagnostic.png")
    plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Diagnose ARIMA detection issues")
    parser.add_argument("satellite", nargs="?", default="Jason-1", 
                       help="Satellite name")
    parser.add_argument("--element", default="mean_motion",
                       help="Orbital element to analyze")
    parser.add_argument("--plot", action="store_true",
                       help="Generate diagnostic plots")
    
    args = parser.parse_args()
    
    # Run diagnostic
    diagnose_satellite_data(args.satellite, args.element)
    
    # Create plots if requested
    if args.plot:
        plot_diagnostic(args.satellite, args.element)
