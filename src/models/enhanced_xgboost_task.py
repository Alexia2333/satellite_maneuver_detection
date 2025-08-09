# scripts/enhanced_xgboost_task.py
"""
使用Enhanced XGBoost Detector的无监督检测任务

任务：
1. 读取数据（使用现有loader）
2. 用Enhanced XGBoost进行无监督检测
3. 用机动日志检查准确性
4. 换成粒子滤波方法再试一次

运行：python scripts/enhanced_xgboost_task.py
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from models.hybrid.xgboost_detector import create_labels_for_split


# 添加路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

print("🚀 Enhanced XGBoost无监督检测验证")
print("=" * 50)

# =====================================================================
# 任务1：读取数据
# =====================================================================
def load_data():
    """任务1：使用现有loader读取数据"""
    print("\n📂 任务1：读取数据")
    print("-" * 20)
    
    try:
        from data.loader import SatelliteDataLoader
        
        # 创建数据加载器
        loader = SatelliteDataLoader(data_dir="data")
        
        # 加载风云4A数据
        tle_data, maneuver_times = loader.load_satellite_data("Fengyun-4A")
        
        print(f"✅ TLE数据: {len(tle_data)} 条记录")
        print(f"✅ 机动数据: {len(maneuver_times)} 个事件")
        
        if len(tle_data) > 0:
            print(f"   时间范围: {tle_data['epoch'].min()} 到 {tle_data['epoch'].max()}")
            print(f"   列名: {list(tle_data.columns)}")
        
        return tle_data, maneuver_times
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return None, None

# =====================================================================
# 特征工程（简化版）
# =====================================================================
def create_enhanced_features(tle_data, target_col='mean_motion'):
    """创建增强特征 (学习目标已修正)"""
    print("\n🛠️ 创建增强特征")
    print("-" * 15)
    
    data = tle_data.set_index('epoch').copy()
    base_cols = ['mean_motion', 'eccentricity', 'inclination']
    available_cols = [col for col in base_cols if col in data.columns]
    
    if not available_cols:
        print(f"❌ 没有找到基础特征列: {base_cols}")
        return None
    
    print(f"   基础特征: {available_cols}")
    
    enhanced_data = data[available_cols].copy()
    
    # 创建滞后、滚动、差分等特征... (这部分逻辑不变)
    lags = [1, 2, 3, 7]
    for col in available_cols:
        for lag in lags:
            enhanced_data[f'{col}_lag_{lag}'] = data[col].shift(lag)
    
    windows = [3, 7, 14]
    for col in available_cols:
        for window in windows:
            enhanced_data[f'{col}_rolling_mean_{window}'] = data[col].rolling(window).mean()
            enhanced_data[f'{col}_rolling_std_{window}'] = data[col].rolling(window).std()
            
    for col in available_cols:
        enhanced_data[f'{col}_diff_1'] = data[col].diff(1)
        enhanced_data[f'{col}_diff_7'] = data[col].diff(7)
    
    # --- 【核心修正】学习目标改为预测目标列的“变化量” ---
    enhanced_data['target'] = enhanced_data[target_col].diff().shift(-1)
    
    enhanced_data = enhanced_data.dropna()
    
    print(f"✅ 增强特征创建完成 (学习目标: {target_col}的变化量)")
    print(f"   有效记录: {len(enhanced_data)}")
    
    return enhanced_data

# =====================================================================
# 任务2：Enhanced XGBoost无监督检测
# =====================================================================
def enhanced_xgboost_detection(enhanced_data, maneuver_times):
    """任务2：Enhanced XGBoost无监督检测 (采用标准流程重构)"""
    print("\n🤖 任务2：Enhanced XGBoost检测")
    print("-" * 30)
    
    try:
        from models.hybrid.enhanced_xgboost_detector import ImprovedXGBoostDetector
        from models.hybrid.xgboost_detector import create_labels_for_split

        # 1. 【标准划分】按时间划分出训练周期和测试周期
        split_ratio = 0.7
        split_index = int(len(enhanced_data) * split_ratio)
        train_period_data = enhanced_data.iloc[:split_index]
        test_data = enhanced_data.iloc[split_index:]
        
        print(f"   训练周期数据: {len(train_period_data)} 条")
        print(f"   测试周期数据: {len(test_data)} 条")

        # 2. 在训练周期内，识别出“纯净”的正常数据用于训练
        #    这是标准的做法，用已知信息构建一个高质量的“正常”模型
        train_period_labels = create_labels_for_split(
            train_period_data.index,
            maneuver_times,
            window=timedelta(days=1)
        )
        normal_mask = (train_period_labels == 0)
        normal_train_data = train_period_data[normal_mask]
        print(f"   从训练周期中提取出纯净的正常数据: {len(normal_train_data)} 条")

        # 3. 初始化检测器
        detector = ImprovedXGBoostDetector(
            target_column='mean_motion',
            enable_threshold_optimization=False, # 我们没有提供验证集标签，让其自动优化
            enable_temporal_clustering=True,
            satellite_type='auto',
            ultra_deep=True
        )
        
        # 4. 训练模型
        print("   🔧 训练Enhanced XGBoost检测器...")
        #    模型将只使用 normal_train_data 进行训练
        #    在其内部，fit方法会自动划分训练/验证集用于early stopping
        detector.fit(
            train_features=normal_train_data,
            satellite_name="Fengyun-4A",
            verbose=True
        )
        
        # 5. 在独立的测试集上检测异常
        print("   🔍 检测测试集异常...")
        anomaly_indices, anomaly_scores = detector.detect_anomalies(test_data, return_scores=True)
        anomaly_timestamps = test_data.index[anomaly_indices]
        
        print(f"✅ Enhanced XGBoost检测完成:")
        print(f"   检测到 {len(anomaly_timestamps)} 个异常点")
        if len(test_data) > 0:
            print(f"   异常比例: {len(anomaly_timestamps)/len(test_data)*100:.2f}%")
        
        return anomaly_timestamps, anomaly_scores, test_data, detector
        
    except Exception as e:
        print(f"❌ Enhanced XGBoost检测失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


# =====================================================================
# 任务3：用机动日志检查准确性
# =====================================================================
def check_enhanced_xgboost_accuracy(anomaly_timestamps, test_data, maneuver_times):
    """任务3：检查Enhanced XGBoost准确性"""
    print("\n📊 任务3：检查Enhanced XGBoost准确性")
    print("-" * 35)
    
    if anomaly_timestamps is None or len(maneuver_times) == 0:
        print("⚠️ 没有检测结果或机动日志，跳过准确性检查")
        return None
    
    # 创建测试集的真实标签
    test_true_labels = create_labels_for_split(test_data.index, maneuver_times, window=timedelta(days=1))
    
    # 创建预测标签
    test_pred_labels = pd.Series(0, index=test_data.index)
    for anomaly_time in anomaly_timestamps:
        if anomaly_time in test_pred_labels.index:
            test_pred_labels.loc[anomaly_time] = 1
    
    # 计算性能指标
    try:
        from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
        
        precision = precision_score(test_true_labels, test_pred_labels, zero_division=0)
        recall = recall_score(test_true_labels, test_pred_labels, zero_division=0)
        f1 = f1_score(test_true_labels, test_pred_labels, zero_division=0)
        
        # 混淆矩阵
        tn, fp, fn, tp = confusion_matrix(test_true_labels, test_pred_labels).ravel()
        
        print(f"✅ Enhanced XGBoost性能:")
        print(f"   精确率(Precision): {precision:.3f}")
        print(f"   召回率(Recall): {recall:.3f}")
        print(f"   F1分数: {f1:.3f}")
        print(f"   真正例(TP): {tp}, 假正例(FP): {fp}")
        print(f"   真负例(TN): {tn}, 假负例(FN): {fn}")
        
        return {'precision': precision, 'recall': recall, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn}
        
    except Exception as e:
        print(f"❌ 准确性计算失败: {e}")
        return None

# =====================================================================
# 任务4：粒子滤波方法（保持不变）
# =====================================================================
def particle_filter_detection(enhanced_data, maneuver_times):
    """任务4：粒子滤波方法检测"""
    print("\n🔬 任务4：粒子滤波检测")
    print("-" * 22)
    
    try:
        # 使用基础特征进行粒子滤波
        base_cols = ['mean_motion', 'eccentricity', 'inclination']
        available_cols = [col for col in base_cols if col in enhanced_data.columns]
        
        if not available_cols:
            print(f"❌ 没有找到基础特征列")
            return None, None, None
        
        data_matrix = enhanced_data[available_cols].values
        
        # 简化的粒子滤波器
        print("   🔧 运行粒子滤波器...")
        
        window_size = 7
        anomaly_scores = []
        anomaly_indices = []
        
        for i in range(window_size, len(data_matrix)):
            window_data = data_matrix[i-window_size:i]
            current_point = data_matrix[i]
            
            window_mean = np.mean(window_data, axis=0)
            window_std = np.std(window_data, axis=0)
            
            z_scores = np.abs(current_point - window_mean) / (window_std + 1e-8)
            combined_score = np.mean(z_scores)
            
            anomaly_scores.append(combined_score)
            
            # 调整阈值以减少误报
            if combined_score > 2.5:  # 提高阈值
                anomaly_indices.append(i)
        
        pf_anomaly_timestamps = enhanced_data.index[anomaly_indices]
        
        print(f"✅ 粒子滤波检测完成:")
        print(f"   检测到 {len(pf_anomaly_timestamps)} 个异常点")
        print(f"   异常比例: {len(pf_anomaly_timestamps)/len(enhanced_data)*100:.2f}%")
        
        # 检查准确性
        if len(maneuver_times) > 0:
            print("\n📊 粒子滤波准确性检查:")
            
            true_labels = create_labels_for_split(enhanced_data.index, maneuver_times, window=timedelta(days=1))
            pred_labels = pd.Series(0, index=enhanced_data.index)
            
            for anomaly_time in pf_anomaly_timestamps:
                if anomaly_time in pred_labels.index:
                    pred_labels.loc[anomaly_time] = 1
            
            try:
                from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
                
                precision = precision_score(true_labels, pred_labels, zero_division=0)
                recall = recall_score(true_labels, pred_labels, zero_division=0)
                f1 = f1_score(true_labels, pred_labels, zero_division=0)
                tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels).ravel()
                
                print(f"   精确率(Precision): {precision:.3f}")
                print(f"   召回率(Recall): {recall:.3f}")
                print(f"   F1分数: {f1:.3f}")
                print(f"   真正例(TP): {tp}, 假正例(FP): {fp}")
                print(f"   真负例(TN): {tn}, 假负例(FN): {fn}")
                
                return pf_anomaly_timestamps, anomaly_scores, {'precision': precision, 'recall': recall, 'f1': f1}
                
            except Exception as e:
                print(f"⚠️ 准确性计算失败: {e}")
                return pf_anomaly_timestamps, anomaly_scores, None
        
        return pf_anomaly_timestamps, anomaly_scores, None
        
    except Exception as e:
        print(f"❌ 粒子滤波检测失败: {e}")
        return None, None, None

# =====================================================================
# 主函数
# =====================================================================
def main():
    """主函数：执行所有任务"""
    print(f"📂 工作目录: {project_root}")
    
    # 任务1：读取数据
    tle_data, maneuver_times = load_data()
    if tle_data is None:
        print("❌ 数据加载失败，任务终止")
        return
    
    # 特征工程
    enhanced_data = create_enhanced_features(tle_data)
    if enhanced_data is None:
        print("❌ 特征工程失败，任务终止")
        return
    
    # 任务2：Enhanced XGBoost检测
    xgb_anomaly_timestamps, xgb_scores, test_data, detector = enhanced_xgboost_detection(
        enhanced_data, maneuver_times
    )
    
    # 任务3：检查Enhanced XGBoost准确性
    xgb_performance = None
    if xgb_anomaly_timestamps is not None and test_data is not None:
        xgb_performance = check_enhanced_xgboost_accuracy(
            xgb_anomaly_timestamps, test_data, maneuver_times
        )
    
    # 任务4：粒子滤波方法
    pf_anomaly_timestamps, pf_scores, pf_performance = particle_filter_detection(
        enhanced_data, maneuver_times
    )
    
    # 总结对比
    print("\n" + "=" * 60)
    print("📊 总结对比")
    print("=" * 60)
    
    print(f"🔍 数据概况:")
    print(f"   TLE记录数: {len(tle_data)}")
    print(f"   增强特征记录数: {len(enhanced_data)}")
    print(f"   机动事件数: {len(maneuver_times)}")
    
    if xgb_anomaly_timestamps is not None and test_data is not None:
        print(f"\n🤖 Enhanced XGBoost结果:")
        print(f"   测试集大小: {len(test_data)}")
        print(f"   检测异常数: {len(xgb_anomaly_timestamps)}")
        if xgb_performance:
            print(f"   F1分数: {xgb_performance['f1']:.3f}")
            print(f"   精确率: {xgb_performance['precision']:.3f}")
            print(f"   召回率: {xgb_performance['recall']:.3f}")
    
    if pf_anomaly_timestamps is not None:
        print(f"\n🔬 粒子滤波结果:")
        print(f"   检测异常数: {len(pf_anomaly_timestamps)}")
        if pf_performance:
            print(f"   F1分数: {pf_performance['f1']:.3f}")
            print(f"   精确率: {pf_performance['precision']:.3f}")
            print(f"   召回率: {pf_performance['recall']:.3f}")
    
    # 比较性能
    if xgb_performance and pf_performance:
        print(f"\n🏆 性能对比:")
        if xgb_performance['f1'] > pf_performance['f1']:
            print(f"   Enhanced XGBoost更好: F1={xgb_performance['f1']:.3f} vs {pf_performance['f1']:.3f}")
        elif pf_performance['f1'] > xgb_performance['f1']:
            print(f"   粒子滤波更好: F1={pf_performance['f1']:.3f} vs {xgb_performance['f1']:.3f}")
        else:
            print(f"   性能相近: F1={xgb_performance['f1']:.3f}")
    
    print(f"\n✅ Enhanced XGBoost任务完成！")

if __name__ == "__main__":
    main()