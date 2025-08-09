# scripts/integrated_kfold_task.py
"""
【最终集成版 - V3 / 策略2】
采用分位数阈值法，并进行严格的无监督K折交叉验证。
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings('ignore')

# --- 路径设置 ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

# --- 导入模块 ---
try:
    from src.data.loader import SatelliteDataLoader
    from src.data.feature_engineer import EnhancedSatelliteFeatureEngineer
    from src.models.hybrid.enhanced_xgboost_detector import ImprovedXGBoostDetector
    from src.models.hybrid.xgboost_detector import create_labels_for_split
except ImportError as e:
    print(f"❌ 无法导入模块: {e}")
    sys.exit(1)

# --- 性能计算函数 (不变) ---
def calculate_performance(y_true, y_pred):
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1: tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    else:
        tp = np.sum((y_true == 1) & (y_pred == 1)); fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0)); tn = np.sum((y_true == 0) & (y_pred == 0))
    return {'precision': precision, 'recall': recall, 'f1': f1, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}

# --- 主执行函数 ---
def main():
    SATELLITE_NAME = "Fengyun-4A"
    TARGET_COLUMN = 'mean_motion'
    N_SPLITS = 5

    print(f"🚀 开始为卫星 {SATELLITE_NAME} 执行无监督K折交叉验证任务 (策略: 分位数阈值)")
    print("=" * 60)

    # 1. 数据加载
    print(" STEP 1: 加载原始数据")
    loader = SatelliteDataLoader(data_dir="data")
    tle_data, maneuver_times = loader.load_satellite_data(SATELLITE_NAME)
    if tle_data is None: return
    print(f"✅ 加载完成: {len(tle_data)} 条记录, {len(maneuver_times)} 次机动。")

    # 2. 特征工程
    print("\n STEP 2: 创建增强特征")
    features_input_data = tle_data.set_index('epoch').sort_index()
    engineer = EnhancedSatelliteFeatureEngineer(
        target_column=TARGET_COLUMN, additional_columns=['eccentricity', 'inclination'], satellite_type='GEO')
    engineer.fit(features_input_data, satellite_name=SATELLITE_NAME)
    features_df = engineer.transform(features_input_data)
    
    # 3. 修正学习目标
    print("\n STEP 3: 修正学习目标为“变化量”")
    features_df['target'] = features_df[TARGET_COLUMN].diff().shift(-1)
    final_data = features_df.dropna()
    print(f"✅ 最终数据集大小: {len(final_data)}")

    # 4. K折交叉验证
    print(f"\n STEP 4: 初始化 {N_SPLITS}-折 时间序列交叉验证")
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    results = []
    
    print("-" * 60)
    for fold, (train_index, test_index) in enumerate(tscv.split(final_data)):
        print(f"🔄 [第 {fold + 1}/{N_SPLITS} 折]")
        train_data = final_data.iloc[train_index]; test_data = final_data.iloc[test_index]
        print(f"   训练数据: {len(train_data)} 条, 测试数据: {len(test_data)} 条")

        # 【核心修改】初始化模型时使用 quantile 参数
        detector = ImprovedXGBoostDetector(
            target_column=TARGET_COLUMN,
            satellite_type='auto',
            ultra_deep=True,
            threshold_quantile=0.999 # <--- 在此调整分位数，这是现在的关键调优参数
        )

        print("   ⏳ 正在训练...")
        detector.fit(train_features=train_data, satellite_name=SATELLITE_NAME, verbose=True)
        
        print("   🎯 正在预测...")
        anomaly_indices, _ = detector.detect_anomalies(test_data, return_scores=True)
        anomaly_timestamps = test_data.index[anomaly_indices]

        # 验证性能
        y_true = create_labels_for_split(test_data.index, maneuver_times, window=timedelta(days=1))
        y_pred = pd.Series(0, index=test_data.index); y_pred.loc[anomaly_timestamps] = 1

        if y_true.sum() > 0:
            performance = calculate_performance(y_true, y_pred)
            results.append(performance)
            print(f"   📊 结果: F1={performance['f1']:.3f}, P={performance['precision']:.3f}, R={performance['recall']:.3f} (TP:{performance['tp']}, FP:{performance['fp']}, FN:{performance['fn']})")
        else:
            print("   ⚠️ 当前测试折无机动事件，跳过评估。")
        print("-" * 60)

    # 汇总最终结果
    if not results:
        print("❌ 未能在任何折中完成有效的性能评估。"); return

    print("\n" + "=" * 60); print("🏆 K折交叉验证最终性能总结"); print("=" * 60)
    results_df = pd.DataFrame(results)
    mean_perf = results_df.mean(); std_perf = results_df.std()
    print(f"基于 {len(results)} 个有效折的平均性能:")
    print(f"   - F1分数    : {mean_perf['f1']:.3f} ± {std_perf['f1']:.3f}")
    print(f"   - 精确率(P) : {mean_perf['precision']:.3f} ± {std_perf['precision']:.3f}")
    print(f"   - 召回率(R) : {mean_perf['recall']:.3f} ± {std_perf['recall']:.3f}")
    print("\n混淆矩阵项 (平均值):")
    print(f"   - 真正例 (TP): {mean_perf['tp']:.1f}, 假正例 (FP): {mean_perf['fp']:.1f}, 假负例 (FN): {mean_perf['fn']:.1f}")

if __name__ == "__main__":
    main()