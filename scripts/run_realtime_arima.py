import os
import sys
from pathlib import Path
import pandas as pd
import argparse

# 添加项目路径
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import SatelliteDataLoader
from src.models.realtime_arima_detector import RealTimeARIMADetector

def main():
    parser = argparse.ArgumentParser(description="Run Real-Time ARIMA Maneuver Detection Simulation")
    parser.add_argument("satellite", help="Satellite name (e.g., Jason-3)")
    parser.add_argument("--element", default="mean_motion", help="Orbital element to analyze")
    args = parser.parse_args()

    print(f"--- Starting Real-Time ARIMA Simulation for {args.satellite} on '{args.element}' ---")

    # 1. 加载全部数据
    loader = SatelliteDataLoader(data_dir="data")
    tle_data, _ = loader.load_satellite_data(args.satellite)
    
    # 2. 预处理：重采样为每日数据并填充缺失值
    ts_data = tle_data[['epoch', args.element]].copy()
    ts_data = ts_data.set_index('epoch').resample('D').median()
    ts_data = ts_data.interpolate(method='time').dropna()
    
    if len(ts_data) < 50:
        print("Error: Not enough data to run simulation.")
        return

    # 3. 划分数据：大部分作为历史，最后15个点作为“新”数据来模拟实时检测
    history_data = ts_data[args.element].iloc[:-15]
    new_data_points = ts_data[args.element].iloc[-15:]

    # 4. 创建并训练检测器
    d_order = 2 if 'jason' in args.satellite.lower() else 1
    detector = RealTimeARIMADetector(p=5, d=d_order, q=1)
    detector.fit(history_data)

    # 5. 模拟实时检测过程
    print("\n--- Simulating Real-Time Detection on New Data Points ---")
    print("-" * 60)
    print(f"{'Timestamp':<25} | {'Actual':>12} | {'Forecast':>12} | {'Score':>8} | {'Decision'}")
    print("-" * 60)

    for timestamp, new_value in new_data_points.items():
        # 【核心修正】在调用 detect 时，同时传入 new_value 和 timestamp
        result = detector.detect(new_value, timestamp)
        
        decision = "MANEUVER" if result["is_maneuver"] else "Normal"
        
        print(f"{str(timestamp.date()):<25} | {new_value:12.6f} | {result['forecast']:12.6f} | {result['score']:8.2f} | {decision}")


if __name__ == "__main__":
    main()