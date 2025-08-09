import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

print("🚀 改进的风云4A机动检测")
print("=" * 50)


def group_and_filter_anomalies(timestamps, max_gap=pd.Timedelta(days=1), min_group_size=2): # <--- 增加 min_group_size 参数
    if timestamps.empty:
        return []

    timestamps = pd.Series(sorted(timestamps))
    groups = []
    current_group = [timestamps.iloc[0]]

    for i in range(1, len(timestamps)):
        if (timestamps.iloc[i] - timestamps.iloc[i-1]) <= max_gap:
            current_group.append(timestamps.iloc[i])
        else:
            if len(current_group) >= min_group_size:
                groups.extend(current_group)
            current_group = [timestamps.iloc[i]]

    if len(current_group) >= min_group_size:
        groups.extend(current_group)

    return sorted(list(set(groups)))
# =====================================================================
# 增强的特征工程（基于原有create_enhanced_features改进）
# =====================================================================
def create_drift_enhanced_features(tle_data, target_col='mean_motion'):
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
    enhanced_data['target'] = np.log1p(enhanced_data['target'] * 1e8)

    # 打印确认
    print("✅ 应用了 log1p(target * 1e8) 变换")
    print(enhanced_data['target'].describe())
    


    # 填充其他特征
    enhanced_data = enhanced_data.fillna(method='ffill', limit=3)
    enhanced_data = enhanced_data.fillna(method='bfill', limit=3)
    for col in enhanced_data.columns:
        if enhanced_data[col].isna().any():
            median_val = enhanced_data[col].median()
            enhanced_data[col].fillna(median_val if not pd.isna(median_val) else 0, inplace=True)

    print(f"✅ 特征创建完成")
    print(f"   特征数量: {len(enhanced_data.columns)}")
    print(f"   有效记录: {len(enhanced_data)}")
    print(f"   目标变量统计: mean={enhanced_data['target'].mean():.6f}, std={enhanced_data['target'].std():.6f}")
    return enhanced_data


# =====================================================================
# 使用改进的检测器
# =====================================================================
def run_improved_detection(enhanced_data, maneuver_times):
    """运行改进的检测 - 使用简化方法"""
    print("\n🤖 运行改进的漂移感知检测")
    print("-" * 30)
    
    try:
        from models.hybrid.xgboost_detector import create_labels_for_split
        import xgboost as xgb
        
        # 1. 数据划分
        split_ratio = 0.7
        split_index = int(len(enhanced_data) * split_ratio)
        train_data = enhanced_data.iloc[:split_index]
        test_data = enhanced_data.iloc[split_index:]
        
        print(f"   训练集: {len(train_data)} 条")
        print(f"   测试集: {len(test_data)} 条")
        
        # 2. 提取正常数据训练
        train_labels = create_labels_for_split(
            train_data.index,
            maneuver_times,
            window=timedelta(days=1)
        )
        
        # 使用正常数据训练
        normal_mask = (train_labels == 0)
        normal_train_data = train_data[normal_mask]
        print(f"   正常训练数据: {len(normal_train_data)} 条")
        
        # 3. 准备训练数据
        X_train = normal_train_data.drop(columns=['target'])
        y_train = normal_train_data['target']
        feature_names = list(X_train.columns)
        
        print(f"   特征数: {len(feature_names)}")
        print(f"   目标统计: mean={y_train.mean():.6f}, std={y_train.std():.6f}, max={y_train.max():.6f}")
        
        # 4. 训练XGBoost模型
        print("   🔧 训练XGBoost模型...")
        model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        )
        
        model.fit(X_train, y_train)
        
        # 5. 计算训练集残差和阈值
        print("   📊 计算训练集性能和阈值...")
        y_train_pred = model.predict(X_train)
        train_residuals = np.abs(y_train - y_train_pred)
        
        epsilon = 1e-8 
        train_change_rates = np.abs(y_train - y_train_pred) / (np.abs(y_train) + epsilon)

        threshold_percentile = 99.5
        residual_threshold = train_residuals.mean() + 3 * train_residuals.std()
        change_rate_threshold_val = np.percentile(train_change_rates, 99.5)
        
        print(f"   训练R²: {model.score(X_train, y_train):.4f}")
        print(f"   残差阈值 ({threshold_percentile}%): {residual_threshold:.6f}")
        print(f"   变化率阈值 ({threshold_percentile}%): {change_rate_threshold_val:.6f}")
        
        # 6. 在测试集上检测
        print("   🔍 检测异常...")
        X_test = test_data[feature_names]
        y_test = test_data['target']
        
        y_test_log_pred = model.predict(X_test)
        y_test_pred = y_test_log_pred
        y_test_actual = y_test
        test_residuals = np.abs(y_test_actual - y_test_pred)
        # 基础检测
        anomaly_mask = test_residuals > residual_threshold
        anomaly_indices = np.where(anomaly_mask)[0]
        
        print(f"   基础检测: {len(anomaly_indices)} 个异常")
        
        # 7. 结合漂移压力的增强检测
        print("   🧪 应用结合漂移压力的增强检测...")

        # 准备测试数据
        X_test = test_data[feature_names]
        y_test = test_data['target']
        y_test_pred = model.predict(X_test)

        # 计算基础指标
        test_residuals = np.abs(y_test - y_test_pred)
        test_change_rates = np.abs(y_test - y_test_pred) / (np.abs(y_test) + epsilon)
        
        # 定义两个检测路径：
        # 路径A: "强信号" - 无论压力如何，只要残差和变化率都极高，就认为是异常
        strong_signal_mask = (test_residuals > residual_threshold) & (test_change_rates > change_rate_threshold_val)
        print(f"      - 强信号检测（AND逻辑）发现: {strong_signal_mask.sum()} 个异常")

        # 路径B: "压力辅助信号" - 在轨道漂移压力大的时期，适当放宽标准
        # 初始化一个全为False的mask
        pressure_assisted_mask = pd.Series(False, index=test_data.index)
        
        # 这部分代码会定义后续可视化和报告所需的变量
        drift_features = [col for col in feature_names if any(keyword in col for keyword in ['drift', 'cumulative'])]
        if drift_features:
            print("      - 计算漂移压力以辅助检测...")
            # 计算漂移压力
            train_drift_pressure = train_data[drift_features].abs().mean(axis=1)
            test_drift_pressure = test_data[drift_features].abs().mean(axis=1)

            # 使用训练集的中位数作为压力基线，更稳健
            pressure_baseline = np.median(train_drift_pressure)
            
            # 对测试集的压力进行标准化
            # 为避免除以0，如果基线为0则不进行标准化
            if pressure_baseline > 0:
                normalized_pressure = test_drift_pressure / pressure_baseline
            else:
                normalized_pressure = test_drift_pressure
            
            # 遍历每一个点，应用压力调整逻辑
            for i in range(len(test_residuals)):
                # 只在压力大于基线时，才考虑降低阈值
                if normalized_pressure.iloc[i] > 1.0: 
                    # 压力越大，阈值降低得越多，但最多降低30%
                    reduction_factor = 1.0 - 0.3 * min((normalized_pressure.iloc[i] - 1.0), 1.0)
                    adjusted_threshold = residual_threshold * reduction_factor
                    
                    # 如果点的残差超过了【调整后】的阈值，则认为异常
                    if test_residuals[i] > adjusted_threshold:
                        pressure_assisted_mask.iloc[i] = True
            
            print(f"      - 压力辅助检测发现: {pressure_assisted_mask.sum()} 个异常")
        
        # 合并两个路径的结果：满足任何一个路径的都是异常
        final_anomaly_mask = strong_signal_mask | pressure_assisted_mask
        initial_anomaly_indices = np.where(final_anomaly_mask)[0]

        # 为后续步骤（如图表和报告）保存必要的变量
        drift_pressure_test = test_drift_pressure if 'test_drift_pressure' in locals() else None

        # 8. 时序聚类与过滤
        initial_anomaly_timestamps = test_data.index[initial_anomaly_indices]
        
        if not initial_anomaly_timestamps.empty:
            print("   应用时序聚类和过滤...")
            clustered_timestamps = group_and_filter_anomalies(
                initial_anomaly_timestamps, 
                max_gap=pd.Timedelta(days=1.5), 
                min_group_size=3  # 要求一个机动事件至少产生2个连续的异常点
            )
            
            # 用过滤后的结果作为最终的异常时间戳
            anomaly_timestamps = clustered_timestamps
            print(f"   聚类和过滤后，最终异常事件: {len(anomaly_timestamps)} 个")
        else:
            anomaly_timestamps = pd.DatetimeIndex([]) # 如果没有初始异常点，则结果为空
            print("   没有发现符合条件的异常事件。")

        # 为了后续代码兼容，获取最终的索引
        anomaly_indices = [test_data.index.get_loc(ts) for ts in anomaly_timestamps if ts in test_data.index]
        
        drift_analysis = {}
        class SimpleDetector:
            def __init__(self, model, threshold, features):
                self.model = model
                self.residual_threshold = threshold
                self.feature_names = features
        
        
        detector = SimpleDetector(model, residual_threshold, feature_names)        
        # 9. 漂移分析
        
        if drift_features and drift_pressure_test is not None:
            # 使用相同的标准化方法
            if pressure_baseline > 0:
                drift_pressure_normalized = drift_pressure_test / pressure_baseline
            else:
                drift_pressure_normalized = drift_pressure_test
                
            drift_analysis = {
                'mean_pressure': float(drift_pressure_normalized.mean()),
                'max_pressure': float(drift_pressure_normalized.max()),
                'high_pressure_ratio': float((drift_pressure_normalized > 0.7).mean()),
                'pressure_trend': float(drift_pressure_normalized.iloc[-7:].mean() - drift_pressure_normalized.iloc[:7].mean()),
                'high_pressure_periods': []
            }
            
            # 找高压力期间
            high_pressure_mask = drift_pressure_normalized > 0.7
            if high_pressure_mask.any():
                changes = high_pressure_mask.astype(int).diff()
                starts = test_data.index[changes == 1]
                ends = test_data.index[changes == -1]
                
                if high_pressure_mask.iloc[0]:
                    starts = pd.Index([test_data.index[0]]).append(starts)
                if high_pressure_mask.iloc[-1]:
                    ends = ends.append(pd.Index([test_data.index[-1]]))
                    
                for start, end in zip(starts[:5], ends[:5]):  # 只取前5个
                    period_mask = (test_data.index >= start) & (test_data.index <= end)
                    drift_analysis['high_pressure_periods'].append({
                        'start': start,
                        'end': end,
                        'duration_days': (end - start).days,
                        'max_pressure': float(drift_pressure_normalized[period_mask].max())
                    })
        
        print(f"\n✅ 检测结果:")
        print(f"   检测异常数: {len(anomaly_timestamps)}")
        print(f"   异常比例: {len(anomaly_timestamps)/len(test_data)*100:.2f}%")
        
        if drift_analysis:
            print(f"\n📊 漂移分析:")
            print(f"   平均漂移压力: {drift_analysis['mean_pressure']:.3f}")
            print(f"   最大漂移压力: {drift_analysis['max_pressure']:.3f}")
            print(f"   高压力时间比例: {drift_analysis['high_pressure_ratio']*100:.1f}%")
        
        # 返回简化的结果
        class SimpleDetector:
            def __init__(self, model, threshold, features, pressure_baseline=1.0):
                self.model = model
                self.residual_threshold = threshold
                self.feature_names = features
                self.pressure_threshold = pressure_baseline  
                self.pressure_baseline = pressure_baseline
                
            def _calculate_drift_pressure(self, data):
                drift_feats = [col for col in self.feature_names 
                              if any(k in col for k in ['drift', 'cumulative'])]
                if drift_feats and all(col in data.columns for col in drift_feats):
                    pressure = data[drift_feats].abs().mean(axis=1)
                    # 标准化
                    if self.pressure_baseline > 0:
                        return pressure / self.pressure_baseline
                    return pressure
                return pd.Series(0, index=data.index)
        
        detector = SimpleDetector(model, residual_threshold, feature_names, 
                                 pressure_baseline if 'pressure_baseline' in locals() else 1.0)
        

        print("📊 y_train sample:", y_train.head())
        print("📊 y_train max:", y_train.max())

        return anomaly_timestamps, (anomaly_indices, test_residuals), test_data, detector, drift_analysis
        
    except Exception as e:
        print(f"❌ 检测失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None
def detect_anomalies_with_enhancements(y_true, y_pred, pressure_series,
                                       residual_threshold=0.18,
                                       change_rate_threshold=0.35,
                                       rolling_window=7,
                                       pressure_alpha=1.0,
                                       baseline_pressure=0.7):
    """
    综合 A/B/C 策略进行异常检测：
    A. 相对变化率
    B. 滑动残差平均
    C. 压力加权调整
    """
    residual = np.abs(y_true - y_pred)
    change_rate = np.abs(y_true - y_pred) / (np.abs(y_true) + 1e-6)
    rolling_residual = pd.Series(residual).rolling(rolling_window, center=True, min_periods=1).mean()
    pressure_weight = 1 + pressure_alpha * np.exp(np.clip(pressure_series - baseline_pressure, 0, 5))
    adjusted_residual = rolling_residual * pressure_weight

    # 综合检测逻辑
    anomaly_mask = (
        (residual > residual_threshold) |
        (change_rate > change_rate_threshold) |
        (adjusted_residual > residual_threshold)
    )
    return anomaly_mask, {
        'residual': residual,
        'change_rate': change_rate,
        'rolling_residual': rolling_residual,
        'adjusted_residual': adjusted_residual
    }
# =====================================================================
# 评估函数（复用原有代码）
# =====================================================================
def evaluate_detection(predictions, test_data, maneuver_times):
    """评估检测结果"""
    print("\n📊 评估检测性能")
    print("-" * 20)
    
    if predictions is None or len(predictions) == 0:
        print("⚠️ 没有预测结果")
        return None
    
    from models.hybrid.xgboost_detector import create_labels_for_split
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
    
    # 创建标签
    true_labels = create_labels_for_split(
        test_data.index,
        maneuver_times,
        window=timedelta(days=1)
    )
    
    pred_labels = pd.Series(0, index=test_data.index)
    for timestamp in predictions:
        if timestamp in pred_labels.index:
            pred_labels.loc[timestamp] = 1
    
    # 计算指标
    precision = precision_score(true_labels, pred_labels, zero_division=0)
    recall = recall_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels).ravel()
    
    print(f"✅ 性能指标:")
    print(f"   精确率: {precision:.3f}")
    print(f"   召回率: {recall:.3f}")
    print(f"   F1分数: {f1:.3f}")
    print(f"   TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
    }

# =====================================================================
# 可视化（改进版）
# =====================================================================
def visualize_drift_detection(enhanced_data, predictions, maneuver_times, 
                            drift_analysis, detector):
    """可视化漂移和检测结果 - 包含预测值对比"""
    print("\n📈 生成可视化...")
    
    # 创建两个图形
    
    # 图1：模型预测值 vs 实际值
    fig1, axes1 = plt.subplots(2, 1, figsize=(15, 10))
    fig1.suptitle('风云4A - 模型预测值 vs 实际值', fontsize=14)
    
    # 子图1：目标变量的预测值和实际值
    ax1 = axes1[0]
    if detector and hasattr(detector, 'model') and detector.model is not None:
        # 获取测试数据的最后部分用于展示
        display_data = enhanced_data.iloc[-200:]  # 最后200个点
        
        # 预测
        X_display = display_data[detector.feature_names]
        y_pred_log = detector.model.predict(X_display)
        y_pred = y_pred_log
        y_true = display_data['target']
        residuals = np.abs(y_true - y_pred)

        # 绘制
        ax1.plot(display_data.index, y_true, 'b-', alpha=0.7, linewidth=1.5, label='实际值')
        ax1.plot(display_data.index, y_pred, 'r--', alpha=0.7, linewidth=1.5, label='预测值')
        
        # 标记大的跳变点
        threshold = y_true.quantile(0.95)
        jumps = y_true > threshold
        ax1.scatter(display_data.index[jumps], y_true[jumps], 
                   color='orange', s=50, alpha=0.7, zorder=5, label='显著跳变')
        
        ax1.set_ylabel('Mean Motion 变化量(绝对值)')
        ax1.set_title('目标变量：预测值 vs 实际值')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 子图2：残差
    ax2 = axes1[1]
    if detector and hasattr(detector, 'model') and detector.model is not None:
        residuals = np.abs(y_true - y_pred)
        ax2.plot(display_data.index, residuals, 'g-', alpha=0.7, linewidth=1)
        
        # 标记检测阈值
        if hasattr(detector, 'residual_threshold'):
            ax2.axhline(y=detector.residual_threshold, color='red', 
                       linestyle='--', alpha=0.7, label=f'检测阈值: {detector.residual_threshold:.6f}')
        
        ax2.set_ylabel('预测残差(绝对值)')
        ax2.set_xlabel('时间')
        ax2.set_title('模型残差')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图1
    output_dir = 'outputs/fy4a_improved'
    os.makedirs(output_dir, exist_ok=True)
    output_path1 = os.path.join(output_dir, 'model_predictions.png')
    plt.savefig(output_path1, dpi=300, bbox_inches='tight')
    print(f"   已保存到: {output_path1}")
    plt.close()
    
    # 图2：机动事件检测对比
    fig2, axes2 = plt.subplots(3, 1, figsize=(15, 10))
    fig2.suptitle('风云4A - 机动事件检测对比', fontsize=14)
    
    # 子图1：Mean Motion和事件标记
    ax1 = axes2[0]
    ax1.plot(enhanced_data.index, enhanced_data['mean_motion'], 
             'b-', alpha=0.7, linewidth=1, label='Mean Motion')
    
    # 标记真实机动（红色虚线）
    for i, m_time in enumerate(maneuver_times):
        if enhanced_data.index[0] <= m_time <= enhanced_data.index[-1]:
            ax1.axvline(m_time, color='red', alpha=0.6, linestyle='--',
                       label='真实机动' if i == 0 else '')
    
    # 标记预测机动（绿色点线）
    if predictions is not None and len(predictions) > 0:
        for i, pred_time in enumerate(predictions):
            ax1.axvline(pred_time, color='green', alpha=0.6, linestyle=':',
                       label='预测机动' if i == 0 else '')
    
    ax1.set_ylabel('Mean Motion')
    ax1.set_title('轨道参数与机动事件')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 子图2：真实机动时间轴
    ax2 = axes2[1]
    # 创建真实机动的时间轴
    maneuver_in_range = [m for m in maneuver_times 
                        if enhanced_data.index[0] <= m <= enhanced_data.index[-1]]
    if maneuver_in_range:
        y_pos = [1] * len(maneuver_in_range)
        ax2.scatter(maneuver_in_range, y_pos, color='red', s=100, marker='v', label='真实机动')
        for i, m_time in enumerate(maneuver_in_range):
            ax2.text(m_time, 1.1, f'{i+1}', ha='center', va='bottom', fontsize=8)
    
    ax2.set_ylim(0.5, 1.5)
    ax2.set_ylabel('真实机动')
    ax2.set_yticks([])
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.legend()
    
    # 子图3：预测机动时间轴
    ax3 = axes2[2]
    if predictions is not None and len(predictions) > 0:
        y_pos = [1] * len(predictions)
        ax3.scatter(predictions, y_pos, color='green', s=100, marker='^', label='预测机动')
        for i, pred_time in enumerate(predictions[:20]):  # 只标记前20个
            ax3.text(pred_time, 1.1, f'{i+1}', ha='center', va='bottom', fontsize=8)
    
    ax3.set_ylim(0.5, 1.5)
    ax3.set_ylabel('预测机动')
    ax3.set_xlabel('时间')
    ax3.set_yticks([])
    ax3.grid(True, alpha=0.3, axis='x')
    ax3.legend()
    
    # 设置相同的x轴范围
    for ax in axes2:
        ax.set_xlim(enhanced_data.index[0], enhanced_data.index[-1])
    
    plt.tight_layout()
    
    # 保存图2
    output_path2 = os.path.join(output_dir, 'maneuver_detection_comparison.png')
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"   已保存到: {output_path2}")
    plt.close()
    
    # 额外：漂移压力分析图
    if detector is not None:
        fig3, ax = plt.subplots(1, 1, figsize=(15, 5))
        
        if hasattr(detector, '_calculate_drift_pressure'):
            drift_pressure = detector._calculate_drift_pressure(enhanced_data)
            ax.plot(enhanced_data.index, drift_pressure, 
                   'orange', alpha=0.7, linewidth=1.5, label='漂移压力')
            ax.axhline(y=detector.pressure_threshold, color='red', 
                      linestyle='--', alpha=0.5, label=f'压力阈值: {detector.pressure_threshold}')
            
            # 高压力区域填充
            ax.fill_between(enhanced_data.index, 0, drift_pressure, 
                           where=drift_pressure > detector.pressure_threshold,
                           color='red', alpha=0.2, label='高压力期')
            
            # 标记机动事件
            for m_time in maneuver_times:
                if enhanced_data.index[0] <= m_time <= enhanced_data.index[-1]:
                    ax.axvline(m_time, color='red', alpha=0.3, linestyle='--')
        
        ax.set_ylabel('漂移压力')
        ax.set_xlabel('时间')
        ax.set_title('累积漂移压力分析')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path3 = os.path.join(output_dir, 'drift_pressure_analysis.png')
        plt.savefig(output_path3, dpi=300, bbox_inches='tight')
        print(f"   已保存到: {output_path3}")
        plt.close()

# =====================================================================
# 主函数
# =====================================================================
def main():
    """主函数"""
    
    # 1. 加载数据（使用现有loader）
    print("\n📂 加载数据...")
    from data.loader import SatelliteDataLoader
    
    loader = SatelliteDataLoader(data_dir="data")
    tle_data, maneuver_times = loader.load_satellite_data("Fengyun-4A")
    
    if tle_data is None:
        print("❌ 数据加载失败")
        return
    
    # 2. 创建漂移增强特征
    enhanced_data = create_drift_enhanced_features(tle_data)
    if enhanced_data is None:
        print("❌ 特征创建失败")
        return
    
    # 3. 运行改进的检测
    predictions, scores, test_data, detector, drift_analysis = run_improved_detection(
        enhanced_data, maneuver_times
    )
    
    # 4. 评估性能
    if predictions is not None:
        metrics = evaluate_detection(predictions, test_data, maneuver_times)
        
        # 5. 可视化结果
        visualize_drift_detection(
            test_data, predictions, maneuver_times, 
            drift_analysis, detector
        )
        
        # 6. 生成报告
        generate_report(tle_data, enhanced_data, test_data, 
                       predictions, metrics, drift_analysis)
    
    print("\n✅ 改进检测任务完成！")

def generate_report(tle_data, enhanced_data, test_data, 
                   predictions, metrics, drift_analysis):
    """生成检测报告"""
    print("\n📄 生成报告...")
    
    output_dir = 'outputs/fy4a_improved'
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'detection_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("风云4A 漂移感知机动检测报告\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("1. 数据概况:\n")
        f.write(f"   原始TLE记录: {len(tle_data)}\n")
        f.write(f"   增强特征数: {len(enhanced_data.columns)}\n")
        f.write(f"   测试集大小: {len(test_data)}\n")
        f.write(f"   时间范围: {test_data.index[0]} 至 {test_data.index[-1]}\n")
        
        f.write(f"\n2. 检测结果:\n")
        if predictions is not None and len(predictions) > 0:
            f.write(f"   检测异常数: {len(predictions)}\n")
            f.write(f"   异常比例: {len(predictions)/len(test_data)*100:.2f}%\n")
        else:
            f.write(f"   检测异常数: 0\n")
            f.write(f"   异常比例: 0.00%\n")
        
        if metrics:
            f.write(f"\n3. 性能指标:\n")
            f.write(f"   精确率: {metrics['precision']:.3f}\n")
            f.write(f"   召回率: {metrics['recall']:.3f}\n")
            f.write(f"   F1分数: {metrics['f1']:.3f}\n")
            f.write(f"   真正例(TP): {metrics['tp']}\n")
            f.write(f"   假正例(FP): {metrics['fp']}\n")
            f.write(f"   假负例(FN): {metrics['fn']}\n")
            f.write(f"   真负例(TN): {metrics['tn']}\n")
        
        if drift_analysis:
            f.write(f"\n4. 漂移分析:\n")
            f.write(f"   平均漂移压力: {drift_analysis['mean_pressure']:.3f}\n")
            f.write(f"   最大漂移压力: {drift_analysis['max_pressure']:.3f}\n")
            f.write(f"   高压力时间比例: {drift_analysis['high_pressure_ratio']*100:.1f}%\n")
            f.write(f"   压力趋势: {drift_analysis['pressure_trend']:.3f}\n")
            
            if drift_analysis['high_pressure_periods']:
                f.write(f"\n   高压力期间:\n")
                for i, period in enumerate(drift_analysis['high_pressure_periods'][:5]):
                    f.write(f"     {i+1}. {period['start']} 至 {period['end']} ")
                    f.write(f"({period['duration_days']}天, 最大压力: {period['max_pressure']:.3f})\n")
        
        f.write(f"\n5. 检测时间列表:\n")
        if predictions is not None and len(predictions) > 0:
            f.write(f"   共检测到 {len(predictions)} 个异常\n")
            for i, pred_time in enumerate(predictions[:20]):
                f.write(f"   {i+1:2d}. {pred_time}\n")
            if len(predictions) > 20:
                f.write(f"   ... (还有 {len(predictions)-20} 个)\n")
        else:
            f.write("   无检测结果\n")
    
    print(f"   报告已保存到: {report_path}")

if __name__ == "__main__":
    main()