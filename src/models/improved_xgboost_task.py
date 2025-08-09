# scripts/improved_xgboost_task.py
"""
改进的XGBoost机动预测方案
针对风云4A卫星轨道维持机动的特征工程和检测优化

改进要点：
1. 特征工程：量化持续漂移和累积漂移压力
2. 学习目标：预测机动引起的状态跳变大小
3. 阈值优化：基于漂移累积模式动态调整
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

print("🚀 改进的XGBoost机动预测方案")
print("=" * 50)

# =====================================================================
# 改进的特征工程
# =====================================================================
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
    # 计算长期滚动均值（正常中心）
    long_windows = [15, 30, 45]  
    
    for col in available_params:
        for window in long_windows:
            # 长期均值
            col_mean = data[col].rolling(window, min_periods=window//2).mean()
            # 偏离程度
            enhanced_data[f'{col}_drift_from_{window}d_mean'] = data[col] - col_mean
            # 相对偏离（标准化）
            col_std = data[col].rolling(window, min_periods=window//2).std()
            enhanced_data[f'{col}_drift_zscore_{window}d'] = (data[col] - col_mean) / (col_std + 1e-8)
    
    # 漂移速度和方向（趋势斜率）
    trend_windows = [7, 14, 21]
    for col in available_params:
        for window in trend_windows:
            # 使用线性回归计算趋势斜率
            enhanced_data[f'{col}_trend_slope_{window}d'] = data[col].rolling(window).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
            )
    
    # ==== 特征2：累积漂移压力 ====
    # 短期漂移的累积值
    short_windows = [3, 7, 14]
    
    for col in available_params:
        # 首先计算短期漂移
        short_mean = data[col].rolling(7).mean()
        short_drift = (data[col] - short_mean).abs()
        
        # 累积漂移压力（滚动累加）
        for window in short_windows:
            enhanced_data[f'{col}_cumulative_drift_{window}d'] = short_drift.rolling(window).sum()
            # 加权累积（最近的漂移权重更大）
            weights = np.exp(np.linspace(-1, 0, window))
            enhanced_data[f'{col}_weighted_cumulative_drift_{window}d'] = short_drift.rolling(window).apply(
                lambda x: np.sum(x * weights[-len(x):]) / np.sum(weights[-len(x):])
            )
    
    # ==== 特征3：参数间的协同漂移 ====
    # 多个参数同时漂移可能预示着需要机动
    if len(available_params) > 1:
        # 计算所有参数的综合漂移指标
        for window in [7, 14]:
            drift_cols = [f'{col}_drift_from_{window}d_mean' for col in available_params 
                          if f'{col}_drift_from_{window}d_mean' in enhanced_data.columns]
            if drift_cols:
                # 综合漂移幅度（L2范数）
                enhanced_data[f'combined_drift_l2_{window}d'] = np.sqrt(
                    enhanced_data[drift_cols].pow(2).sum(axis=1)
                )
                # 综合漂移方向一致性
                enhanced_data[f'drift_correlation_{window}d'] = enhanced_data[drift_cols].apply(
                    lambda x: 1 if (x > 0).all() or (x < 0).all() else 0, axis=1
                )
    
    # ==== 特征4：历史机动模式特征 ====
    # 距离上次大幅变化的时间
    for col in available_params:
        # 检测大幅变化（可能的历史机动）
        col_diff = data[col].diff().abs()
        threshold = col_diff.quantile(0.95)
        
        # 创建事件标记
        events = (col_diff > threshold).astype(int)
        
        # 计算距离上次事件的时间
        last_event_idx = pd.Series(range(len(events)), index=events.index)
        last_event_idx[events == 0] = np.nan
        last_event_idx = last_event_idx.fillna(method='ffill')
        
        enhanced_data[f'{col}_days_since_last_jump'] = (
            pd.Series(range(len(events)), index=events.index) - last_event_idx
        )
    
    # ==== 特征5：统计特征 ====
    # 添加基础统计特征
    stat_windows = [3, 7, 14, 30]
    for col in available_params:
        for window in stat_windows:
            enhanced_data[f'{col}_rolling_mean_{window}d'] = data[col].rolling(window).mean()
            enhanced_data[f'{col}_rolling_std_{window}d'] = data[col].rolling(window).std()
            enhanced_data[f'{col}_rolling_skew_{window}d'] = data[col].rolling(window).skew()
            enhanced_data[f'{col}_rolling_kurt_{window}d'] = data[col].rolling(window).kurt()
    
    # ==== 学习目标：预测机动引起的跳变 ====
    # 目标是预测下一时刻的mean_motion变化量（机动会引起突变）
    enhanced_data['target'] = data[target_col].diff().shift(-1)
    
    # 添加目标的绝对值版本（用于检测任何方向的跳变）
    enhanced_data['target_abs'] = enhanced_data['target'].abs()
    
    # 删除空值
    # 步骤 1: 唯一需要删除的是没有学习目标(target)的行
    enhanced_data = enhanced_data.dropna(subset=['target', 'target_abs'])
    print(f"   记录数 (在丢弃无效目标后): {len(enhanced_data)}")

    # 步骤 2: 对特征列中的NaN值进行填充，而不是删除整行
    # 优先使用前一个时间点的值填充 (forward-fill)
    # 然后用后一个时间点的值填充，处理文件开头的NaN
    feature_cols = [col for col in enhanced_data.columns if col not in ['target', 'target_abs']]
    enhanced_data[feature_cols] = enhanced_data[feature_cols].ffill().bfill()

    # 步骤 3: 作为最后的保险，如果还有NaN，用每列的中位数填充
    for col in feature_cols:
        if enhanced_data[col].isnull().any():
            median_val = enhanced_data[col].median()
            if not pd.isna(median_val):
                enhanced_data[col] = enhanced_data[col].fillna(median_val)
            else:
                # 如果整列都是NaN，则用0填充
                enhanced_data[col] = enhanced_data[col].fillna(0)
    
    print(f"✅ 特征和NaN值处理完成")

    print(f"   记录数 (在丢弃无效目标后): {len(enhanced_data)}")
    
    print(f"✅ 漂移特征创建完成")
    print(f"   特征数量: {len(enhanced_data.columns)}")
    print(f"   有效记录: {len(enhanced_data)}")
    
    # 打印一些关键特征的统计信息
    print("\n📊 关键特征统计:")
    key_features = [col for col in enhanced_data.columns if 'drift' in col or 'cumulative' in col][:5]
    for feat in key_features:
        if feat in enhanced_data.columns:
            print(f"   {feat}: mean={enhanced_data[feat].mean():.6f}, std={enhanced_data[feat].std():.6f}")
    
    return enhanced_data

# =====================================================================
# 改进的XGBoost检测器
# =====================================================================
def improved_xgboost_detection(enhanced_data, maneuver_times):
    """改进的XGBoost检测器，针对轨道维持机动优化"""
    print("\n🤖 改进的XGBoost机动预测")
    print("-" * 30)
    
    try:
        from models.hybrid.enhanced_xgboost_detector import ImprovedXGBoostDetector
        from models.hybrid.xgboost_detector import create_labels_for_split
        
        # 1. 数据划分
        split_ratio = 0.8
        split_index = int(len(enhanced_data) * split_ratio)
        train_period_data = enhanced_data.iloc[:split_index]
        test_data = enhanced_data.iloc[split_index:]
        
        print(f"   训练周期: {len(train_period_data)} 条")
        print(f"   测试周期: {len(test_data)} 条")
        
        # 2. 提取正常数据用于训练
        train_labels = create_labels_for_split(
            train_period_data.index,
            maneuver_times,
            window=timedelta(days=1)
        )
        
        # 这里我们采用不同的策略：不只用正常数据训练
        # 而是用所有数据，但给机动期间的数据更高的权重
        
        # 3. 为机动检测优化的XGBoost参数
        xgb_params = {
            'objective': 'reg:squarederror',  # 预测跳变大小
            'n_estimators': 500,
            'learning_rate': 0.02,
            'max_depth': 8,  # 更深的树来捕捉复杂的漂移模式
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,  # 添加正则化
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42
        }
        
        # 4. 初始化改进的检测器
        detector = ImprovedXGBoostDetector(
            target_column='target_abs',  # 使用绝对值目标
            xgb_params=xgb_params,
            threshold_quantile=0.98,  # 使用分位数阈值
            enable_threshold_optimization=True,
            enable_temporal_clustering=True,
            satellite_type='FY-4A',
            ultra_deep=True
        )
        
        # 5. 训练模型
        print("   🔧 训练改进的XGBoost模型...")
        
        # 创建样本权重：机动期间的样本权重更高
        sample_weights = np.ones(len(train_period_data))
        maneuver_indices = train_labels[train_labels == 1].index
        for idx in maneuver_indices:
            if idx in train_period_data.index:
                loc = train_period_data.index.get_loc(idx)
                # 机动前后的样本也很重要
                for offset in range(-3, 4):  # 前后3天
                    if 0 <= loc + offset < len(sample_weights):
                        sample_weights[loc + offset] = 2.0
        
        # 使用加权训练
        detector.fit(
            train_features=train_period_data,
            satellite_name="Fengyun-4A",
            verbose=True,
        )
        
        # 6. 检测异常
        print("   🔍 检测测试集异常...")
        anomaly_indices, anomaly_scores = detector.detect_anomalies(test_data, return_scores=True)
        
        # 7. 基于漂移压力的后处理
        print("   🔧 基于漂移压力的后处理...")
        
        # 获取累积漂移特征
        drift_features = [col for col in test_data.columns if 'cumulative_drift' in col]
        if drift_features:
            # 计算综合漂移压力
            drift_pressure = test_data[drift_features].mean(axis=1)
            
            # 动态调整阈值：高漂移压力时降低检测阈值
            pressure_percentile = drift_pressure.rank(pct=True)
            adjusted_threshold = detector.residual_threshold * (1 - 0.3 * pressure_percentile)
            
            # 重新检测（考虑漂移压力）
            residuals = anomaly_scores    # 获取残差
            adjusted_anomalies = []
            
            for i, (idx, residual) in enumerate(zip(test_data.index, residuals)):
                if idx in drift_pressure.index:
                    threshold = adjusted_threshold.iloc[drift_pressure.index.get_loc(idx)]
                    if residual > threshold:
                        adjusted_anomalies.append(i)
            
            print(f"   调整前异常数: {len(anomaly_indices)}")
            print(f"   调整后异常数: {len(adjusted_anomalies)}")
            
            anomaly_indices = adjusted_anomalies
        
        anomaly_timestamps = test_data.index[anomaly_indices]
        
        print(f"✅ 检测完成:")
        print(f"   检测到 {len(anomaly_timestamps)} 个异常")
        print(f"   异常比例: {len(anomaly_timestamps)/len(test_data)*100:.2f}%")
        
        return anomaly_timestamps, anomaly_scores, test_data, detector
        
    except Exception as e:
        print(f"❌ 检测失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

# =====================================================================
# 评估函数
# =====================================================================
def evaluate_predictions(predictions, test_data, maneuver_times):
    """评估预测结果"""
    print("\n📊 评估预测结果")
    print("-" * 20)
    
    if predictions is None or len(predictions) == 0:
        print("⚠️ 没有预测结果")
        return None
    
    from models.hybrid.xgboost_detector import create_labels_for_split
    
    # 创建真实标签
    true_labels = create_labels_for_split(
        test_data.index,
        maneuver_times,
        window=timedelta(days=1)
    )
    
    # 创建预测标签
    pred_labels = pd.Series(0, index=test_data.index)
    for timestamp in predictions:
        if timestamp in pred_labels.index:
            pred_labels.loc[timestamp] = 1
    
    # 计算指标
    precision = precision_score(true_labels, pred_labels, zero_division=0)
    recall = recall_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)
    
    # 混淆矩阵
    tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels).ravel()
    
    print(f"✅ 性能指标:")
    print(f"   精确率: {precision:.3f}")
    print(f"   召回率: {recall:.3f}")
    print(f"   F1分数: {f1:.3f}")
    print(f"   真正例(TP): {tp}, 假正例(FP): {fp}")
    print(f"   真负例(TN): {tn}, 假负例(FN): {fn}")
    
    # 分析预测时机
    print("\n📅 预测时机分析:")
    if len(predictions) > 0 and len(maneuver_times) > 0:
        # 计算每个预测与最近机动的时间差
        time_diffs = []
        for pred_time in predictions:
            min_diff = min([abs((pred_time - m_time).days) for m_time in maneuver_times])
            time_diffs.append(min_diff)
        
        time_diffs = np.array(time_diffs)
        print(f"   提前预测（<3天）: {np.sum(time_diffs < 3)} 个")
        print(f"   准确预测（±1天）: {np.sum(time_diffs <= 1)} 个")
        print(f"   延迟预测（>3天）: {np.sum(time_diffs > 3)} 个")
        print(f"   平均时间差: {np.mean(time_diffs):.1f} 天")
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
    }

# =====================================================================
# 可视化函数
# =====================================================================
def visualize_results(enhanced_data, predictions, maneuver_times, detector):
    """可视化预测结果和漂移模式"""
    print("\n📈 生成可视化...")
    
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    fig.suptitle('Fengyun-4A 轨道维持机动预测分析', fontsize=16)
    
    # 1. Mean motion和预测
    ax1 = axes[0]
    ax1.plot(enhanced_data.index, enhanced_data['mean_motion'], 'b-', alpha=0.7, label='Mean Motion')
    
    # 标记真实机动
    for m_time in maneuver_times:
        if enhanced_data.index[0] <= m_time <= enhanced_data.index[-1]:
            ax1.axvline(m_time, color='red', alpha=0.5, linestyle='--', label='真实机动' if m_time == maneuver_times[0] else '')
    
    # 标记预测
    if predictions is not None:
        for pred_time in predictions:
            ax1.axvline(pred_time, color='green', alpha=0.5, linestyle=':', label='预测机动' if pred_time == predictions[0] else '')
    
    ax1.set_ylabel('Mean Motion')
    ax1.legend()
    ax1.set_title('Mean Motion 变化和机动事件')
    
    # 2. 漂移压力
    ax2 = axes[1]
    drift_cols = [col for col in enhanced_data.columns if 'cumulative_drift' in col and 'mean_motion' in col]
    if drift_cols:
        drift_pressure = enhanced_data[drift_cols[0]]
        ax2.plot(enhanced_data.index, drift_pressure, 'orange', alpha=0.7, label='累积漂移压力')
        ax2.set_ylabel('漂移压力')
        ax2.legend()
        ax2.set_title('Mean Motion 累积漂移压力')
    
    # 3. 预测目标（跳变大小）
    ax3 = axes[2]
    if 'target_abs' in enhanced_data.columns:
        ax3.plot(enhanced_data.index, enhanced_data['target_abs'], 'purple', alpha=0.5, label='实际跳变大小')
        ax3.set_ylabel('跳变大小')
        ax3.legend()
        ax3.set_title('Mean Motion 跳变大小（绝对值）')
    
    # 4. 异常分数
    ax4 = axes[3]
    if detector is not None and hasattr(detector, 'model'):
        # 这里简化处理，实际应该获取完整的异常分数
        ax4.set_ylabel('异常分数')
        ax4.set_title('模型异常分数')
    
    plt.tight_layout()
    
    # 保存图片
    output_dir = 'outputs/fengyun4a_improved'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'drift_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"   图片已保存到: {output_dir}/drift_analysis.png")
    
    plt.close()

# =====================================================================
# 主函数
# =====================================================================
def main():
    """主函数"""
    print(f"📂 工作目录: {project_root}")
    
    # 1. 加载数据
    print("\n📂 加载数据...")
    from data.loader import SatelliteDataLoader
    
    loader = SatelliteDataLoader(data_dir="data")
    tle_data, maneuver_times = loader.load_satellite_data("Fengyun-4A")
    
    if tle_data is None:
        print("❌ 数据加载失败")
        return
    
    print(f"✅ 加载 {len(tle_data)} 条TLE记录")
    print(f"✅ 加载 {len(maneuver_times)} 个机动事件")
    
    # 2. 创建漂移特征
    enhanced_data = create_drift_features(tle_data)
    if enhanced_data is None:
        print("❌ 特征创建失败")
        return
    
    # 3. 运行改进的XGBoost检测
    predictions, scores, test_data, detector = improved_xgboost_detection(
        enhanced_data, maneuver_times
    )
    
    # 4. 评估结果
    if predictions is not None:
        metrics = evaluate_predictions(predictions, test_data, maneuver_times)
        
        # 5. 可视化
        visualize_results(enhanced_data, predictions, maneuver_times, detector)
    
    print("\n✅ 改进方案执行完成！")
    
    # 6. 保存结果报告
    if predictions is not None and metrics is not None:
        report_path = 'outputs/fengyun4a_improved/performance_report.txt'
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write("Fengyun-4A 改进的机动预测结果\n")
            f.write("=" * 50 + "\n\n")
            f.write("1. 数据概况:\n")
            f.write(f"   TLE记录数: {len(tle_data)}\n")
            f.write(f"   增强特征数: {len(enhanced_data.columns)}\n")
            f.write(f"   机动事件数: {len(maneuver_times)}\n")
            f.write(f"\n2. 检测结果:\n")
            f.write(f"   测试集大小: {len(test_data)}\n")
            f.write(f"   检测异常数: {len(predictions)}\n")
            f.write(f"\n3. 性能指标:\n")
            for key, value in metrics.items():
                f.write(f"   {key}: {value}\n")
        
        print(f"\n📄 报告已保存到: {report_path}")

if __name__ == "__main__":
    main()