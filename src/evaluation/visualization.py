# src/evaluation/visualization.py

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_detection_results(test_data, predictions, maneuver_times, detector, config):
    """
    一个通用的、模块化的可视化函数，用于绘制所有分析图表。
    
    Args:
        test_data (pd.DataFrame): 包含目标和特征的测试数据集。
        predictions (list): 最终预测出的异常时间戳列表。
        maneuver_times (list): 真实的机动事件时间戳列表。
        detector (object): 我们训练好的探测器对象，内部包含模型和阈值。
        config (dict): 包含输出路径等信息的全局配置字典。
    """
    print("\n📈 生成可视化...")
    output_dir = config.get('output_dir', 'outputs/default')
    os.makedirs(output_dir, exist_ok=True)

    # --- 图1：模型预测 vs 实际值 ---
    fig1, axes1 = plt.subplots(2, 1, figsize=(15, 10))
    fig1.suptitle(f"{config['satellite_name']} - 模型预测与残差分析", fontsize=14)
    
    display_data = test_data.iloc[-250:] # 展示最近的250个点
    X_display = display_data[detector.feature_names]
    y_pred = detector.model.predict(X_display)
    y_true = display_data['target']
    
    # 子图1: 预测值 vs 实际值
    ax1 = axes1[0]
    ax1.plot(display_data.index, y_true, 'b-', alpha=0.7, linewidth=1.5, label='实际值')
    ax1.plot(display_data.index, y_pred, 'r--', alpha=0.7, linewidth=1.5, label='预测值')
    ax1.set_ylabel('目标变量')
    ax1.set_title('模型预测 vs 实际值')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    # 子图2: 残差
    ax2 = axes1[1]
    residuals = np.abs(y_true - y_pred)
    ax2.plot(display_data.index, residuals, 'g-', alpha=0.7, linewidth=1, label='预测残差')
    ax2.axhline(y=detector.residual_threshold, color='red', linestyle='--', alpha=0.7, label=f'残差阈值: {detector.residual_threshold:.4f}')
    ax2.set_ylabel('残差绝对值'); ax2.set_xlabel('时间'); ax2.set_title('模型残差'); ax2.legend(); ax2.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path1 = os.path.join(output_dir, 'model_predictions.png')
    plt.savefig(output_path1, dpi=300); plt.close(fig1)
    print(f"   已保存到: {output_path1}")

    # --- 图2：机动事件检测对比 ---
    fig2, axes2 = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    fig2.suptitle(f"{config['satellite_name']} - 机动事件检测对比", fontsize=14)

    # 子图1: Mean Motion
    ax1 = axes2[0]
    ax1.plot(test_data.index, test_data['mean_motion'], 'b-', alpha=0.7, linewidth=1, label='Mean Motion')
    ax1.set_ylabel('Mean Motion'); ax1.set_title('轨道参数与机动事件'); ax1.legend(); ax1.grid(True, alpha=0.3)
    
    # 子图2: 真实机动
    ax2 = axes2[1]
    maneuver_in_range = [m for m in maneuver_times if test_data.index[0] <= m <= test_data.index[-1]]
    if maneuver_in_range:
        ax2.scatter(maneuver_in_range, [1] * len(maneuver_in_range), c='red', s=100, marker='v', label='真实机动')
    ax2.set_ylim(0.5, 1.5); ax2.set_ylabel('真实机动'); ax2.set_yticks([]); ax2.grid(True, alpha=0.3, axis='x'); ax2.legend()
    
    # 子图3: 预测机动
    ax3 = axes2[2]
    if predictions:
        ax3.scatter(predictions, [1] * len(predictions), c='green', s=100, marker='^', label='预测机动')
    ax3.set_ylim(0.5, 1.5); ax3.set_ylabel('预测机动'); ax3.set_xlabel('时间'); ax3.set_yticks([]); ax3.grid(True, alpha=0.3, axis='x'); ax3.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path2 = os.path.join(output_dir, 'maneuver_detection_comparison.png')
    plt.savefig(output_path2, dpi=300); plt.close(fig2)
    print(f"   已保存到: {output_path2}")