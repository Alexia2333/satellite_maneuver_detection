# scripts/test_particle_filter_performance.py
"""
卫星机动检测粒子滤波器（Particle Filter）性能测试脚本

本脚本旨在评估 `ParticleFilterDetector` 在无监督模式下的性能。
流程:
1. 加载数据 (TLE 和真实机动日志)。
2. 使用 `MeanElementsProcessor` 准备平根数数据。
3. 将数据划分为训练集和测试集。
4. 在训练集上训练粒子滤波器检测器 (估计噪声协方差)。
5. 在测试集上进行机动检测。
6. 对比真实机动日志，评估检测性能。
7. 生成可视化图表和性能报告。
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
import json
import warnings

# --- 项目路径设置 ---
# 将项目根目录添加到系统路径，以便导入自定义模块
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

# --- 导入自定义模块 ---
from data.loader import SatelliteDataLoader
from data.mean_elements_processor import MeanElementsProcessor
from models.unsupervised.particle_filter import ParticleFilterDetector

warnings.filterwarnings('ignore', category=UserWarning) # 忽略一些非关键警告

class ParticleFilterExperiment:
    """封装粒子滤波器性能测试的实验类"""

    def __init__(self, config: dict):
        """
        初始化实验
        Args:
            config (dict): 包含所有实验参数的配置字典。
        """
        self.config = config
        self.loader = SatelliteDataLoader(data_dir=config.get('data_dir', 'data'))
        self.processor = MeanElementsProcessor(satellite_type='auto', outlier_detection=True)
        self.detector = None
        self.results = {}
        
        # 创建输出目录
        os.makedirs(self.config['output_dir'], exist_ok=True)
        print(f"🚀 初始化粒子滤波器实验，输出将保存至: {self.config['output_dir']}")

    def run(self):
        """执行完整的测试流程"""
        print("\n" + "="*60)
        print(f"🛰️  开始对卫星 '{self.config['satellite_name']}' 进行粒子滤波器性能测试")
        print("="*60)

        # 1. 加载数据
        tle_data, maneuver_times = self._load_data()
        
        # 2. 核心数据处理: 使用MeanElementsProcessor准备数据
        processed_data = self._process_data(tle_data)
        
        print("\n[*] 根据配置筛选输入数据...")
        
        if self.config.get('use_only_mean_motion', False):
            # 如果开关为True，只使用 'mean_motion'
            print("   -> 'use_only_mean_motion' 开关开启，模型将只使用 [mean_motion] 维度。")
            pf_input_data = processed_data[['mean_motion']]
        else:
            # 否则，使用多维核心根数
            print("   -> 'use_only_mean_motion' 关闭，使用多维核心根数。")
            core_elements = [
                'mean_motion', 'eccentricity', 'inclination', 
                'arg_perigee', 'raan', 'mean_anomaly'
            ]
            available_core_elements = [col for col in core_elements if col in processed_data.columns]
            print(f"   -> 模型将使用以下列进行滤波: {available_core_elements}")
            pf_input_data = processed_data[available_core_elements]

    

        # 3. 数据划分
        train_data, test_data = self._split_data(pf_input_data)
        
        # 4. 训练检测器 (无监督)
        self._train_detector(train_data)
        
        # 5. 在测试集上检测异常
        detected_anomalies, anomaly_scores = self._detect_anomalies(test_data)
        
        # 6. 评估结果
        metrics = self._evaluate(detected_anomalies, test_data, maneuver_times)
        
        # 7. 可视化和报告
        self._visualize_and_report(processed_data, test_data, detected_anomalies, anomaly_scores, maneuver_times, metrics)
        
        print("\n✅ 实验完成！")
        return self.results

    def _load_data(self):
        """加载TLE数据和机动日志"""
        print("\n[1/7] 📂 加载数据...")
        tle_data, maneuver_times = self.loader.load_satellite_data(self.config['satellite_name'])
    
        # 验证数据
        print(f"   -> TLE数据形状: {tle_data.shape}")
        print(f"   -> TLE数据列: {list(tle_data.columns[:10])}...")  # 显示前10列
    
        # 检查关键列是否存在
        required_cols = ['mean_motion', 'eccentricity', 'inclination', 'mean_anomaly']
        missing_cols = [col for col in required_cols if col not in tle_data.columns]
        if missing_cols:
            print(f"   -> ⚠️ 警告：缺少关键列: {missing_cols}")
    
        print(f"   -> 加载了 {len(tle_data)} 条TLE记录")
        print(f"   -> 加载了 {len(maneuver_times)} 个真实机动事件")
    
        # 显示时间范围
        if not tle_data.empty:
            print(f"   -> TLE时间范围: {tle_data['epoch'].min()} 到 {tle_data['epoch'].max()}")
        if maneuver_times:
            print(f"   -> 机动时间范围: {min(maneuver_times)} 到 {max(maneuver_times)}")
    
        self.results['total_tle_records'] = len(tle_data)
        self.results['total_maneuvers'] = len(maneuver_times)
        return tle_data, maneuver_times

    def _process_data(self, tle_data: pd.DataFrame) -> pd.DataFrame:
        """使用MeanElementsProcessor处理数据，为粒子滤波器准备输入"""
        print("\n[2/7] 🛠️  处理平根数数据...")
        processed_data = self.processor.process_tle_data(tle_data, self.config['satellite_name'])
        print(self.processor.get_processing_report())
        
        # 检查每列的NaN情况
        print("\n   -> 检查数据质量:")
        for col in processed_data.columns:
            nan_count = processed_data[col].isna().sum()
            nan_percent = (nan_count / len(processed_data)) * 100 if len(processed_data) > 0 else 0
            if nan_percent > 0:
                print(f"      - {col}: {nan_count} NaN值 ({nan_percent:.1f}%)")
    
        # 如果eccentricity全是NaN，尝试从原始数据恢复
        if 'eccentricity' in processed_data.columns and processed_data['eccentricity'].isna().all():
            print("   -> ⚠️ eccentricity列全是NaN，尝试从原始数据恢复...")
            if 'eccentricity' in tle_data.columns:
                # Ensure the length matches for direct assignment
                if len(tle_data) >= len(processed_data):
                     processed_data['eccentricity'] = tle_data['eccentricity'].values[:len(processed_data)]
                     print(f"      - 恢复后的eccentricity范围: [{processed_data['eccentricity'].min():.6f}, {processed_data['eccentricity'].max():.6f}]")
                else:
                    print("     -> 恢复失败：原始TLE数据长度小于处理后数据")

        # 移除全是NaN的列
        cols_before = len(processed_data.columns)
        processed_data = processed_data.dropna(axis=1, how='all')
        cols_after = len(processed_data.columns)
        if cols_before > cols_after:
            print(f"   -> 移除了 {cols_before - cols_after} 个全是NaN的列")
    
        # 填充剩余的NaN值
        print("   -> 正在填充数据处理过程中可能产生的NaN值...")
        processed_data.ffill(inplace=True)
        processed_data.bfill(inplace=True)
    
        if processed_data.isnull().values.any():
            warnings.warn("数据填充后仍存在NaN值，将使用0填充剩余空值。")
            processed_data.fillna(0, inplace=True)
    
        print(f"   -> 处理后得到 {len(processed_data)} 条有效数据")
        print(f"   -> 最终使用的列: {list(processed_data.columns)}")
    
        return processed_data

    def _split_data(self, data: pd.DataFrame) -> tuple:
        """按时间划分训练集和测试集"""
        print("\n[3/7] 📊 划分数据集...")
        split_ratio = self.config['train_split_ratio']
        split_index = int(len(data) * split_ratio)
        
        train_data = data.iloc[:split_index]
        test_data = data.iloc[split_index:]
        
        print(f"   -> 训练集: {len(train_data)} 条 ({split_ratio:.0%})")
        print(f"   -> 测试集: {len(test_data)} 条 ({1-split_ratio:.0%})")
        return train_data, test_data

    def _train_detector(self, train_data: pd.DataFrame):
        """在训练数据上拟合粒子滤波器检测器"""
        print("\n[4/7] 🧠 训练粒子滤波器检测器 (无监督)...")
    
        # 首先用一个临时检测器来估计分数分布
        temp_detector = ParticleFilterDetector(
            n_particles=100,  # 用较少的粒子快速估计
            anomaly_threshold=float('inf'), # 设置为无穷大以获取所有分数
            noise_scaling_factor=self.config.get('noise_scaling_factor', 1.0),
            auto_tune_threshold=False
        )
        temp_detector.fit(train_data)
    
        # 在训练数据上计算分数以了解分布
        _, train_scores = temp_detector.detect_anomalies(train_data)
    
        actual_threshold = self.config.get('anomaly_threshold', 12)
        
        if len(train_scores) > 0:
            # 基于分数分布估计阈值
            percentile = self.config.get('threshold_percentile', 95.0)
            estimated_threshold = np.percentile(train_scores, percentile)

            mean_score = np.mean(train_scores)
            std_score = np.std(train_scores)
            k_factor = self.config.get('threshold_std_dev_factor', 3.0) 
            
            estimated_threshold = mean_score + k_factor * std_score            
            print(f"   -> 训练集分数统计:")
            print(f"      - 范围: [{np.min(train_scores):.2f}, {np.max(train_scores):.2f}]")
            print(f"      - 均值: {np.mean(train_scores):.2f}")
            print(f"      - 标准差: {np.std(train_scores):.2f}")
            print(f"      - {percentile}%百分位数: {estimated_threshold:.2f}")
    
            # 使用估计的阈值，除非配置中指定了合理的值 (例如，一个负数)
            # 这是一个启发式规则：如果用户设置了一个看似随意的正值，我们宁愿相信自适应阈值
            if self.config.get('use_auto_threshold', True):
                actual_threshold = estimated_threshold
                print(f"   -> 'use_auto_threshold' 已开启, 使用自动估算的阈值: {actual_threshold:.2f}")
                print(f"   -> 使用配置的阈值: {actual_threshold:.2f}")
            else:
                print(f"   -> 'use_auto_threshold' 已关闭, 使用配置文件中的阈值: {actual_threshold:.2f}")
        else:
            print(f"   -> 无法计算训练集分数，使用配置的阈值: {actual_threshold:.2f}")
        
        # 创建最终的检测器
        self.detector = ParticleFilterDetector(
            n_particles=self.config.get('n_particles', 250),
            anomaly_threshold=actual_threshold,
            noise_scaling_factor=self.config.get('noise_scaling_factor', 1.0),
            auto_tune_threshold=False # 严格无监督
        )
    
        # 训练模型
        self.detector.fit(train_data)
        print("   -> 协方差矩阵 Q 和 R 已从训练数据中估计完成。")
        print(f"   -> 模型摘要: {self.detector.get_model_summary()}")


    def _detect_anomalies(self, test_data: pd.DataFrame) -> tuple:
        """在测试数据上执行异常检测"""
        print("\n[5/7] 🔍 在测试集上检测机动...")
        detected_anomalies, anomaly_scores = self.detector.detect_anomalies(test_data)
        
        # 添加调试信息
        if len(anomaly_scores) > 0:
            print(f"   -> 异常分数统计:")
            print(f"      - 最小值: {np.min(anomaly_scores):.2f}")
            print(f"      - 最大值: {np.max(anomaly_scores):.2f}")
            print(f"      - 平均值: {np.mean(anomaly_scores):.2f}")
            print(f"      - 标准差: {np.std(anomaly_scores):.2f}")
        print(f"      - 阈值: {self.detector.anomaly_threshold:.2f}")
    
        # 找出得分最高的时间点
        if len(anomaly_scores) > 10:
            top_scores_idx = np.argsort(anomaly_scores)[-10:]
            print(f"   -> 得分最高的10个时间点:")
            # score_index starts from the second element of test_data
            score_index_map = test_data.index[1:]
            for idx in top_scores_idx:
                if idx < len(score_index_map):
                    timestamp = score_index_map[idx]
                    score = anomaly_scores[idx]
                    print(f"      {timestamp}: {score:.2f}")        
        
        print(f"   -> 检测到 {len(detected_anomalies)} 个潜在机动事件。")
        self.results['detected_anomalies_count'] = len(detected_anomalies)
        return detected_anomalies, anomaly_scores

    def _evaluate(self, detected_anomalies: list, test_data: pd.DataFrame, maneuver_times: list) -> dict:
        """评估检测性能"""
        print("\n[6/7] 📈 评估检测性能...")
    
        # 调试信息：显示测试集的时间范围
        print(f"   -> 测试集时间范围: {test_data.index.min()} 到 {test_data.index.max()}")
    
        # 显示测试集中的真实机动
        true_maneuvers_in_test = []
        window = timedelta(days=self.config.get('label_window_days', 1))
    
        for m_time in maneuver_times:
            if test_data.index.min() <= m_time <= test_data.index.max():
                true_maneuvers_in_test.append(m_time)
    
        print(f"   -> 测试集中的真实机动时间 (前5个):")
        if not true_maneuvers_in_test:
            print("      无")
        for i, m_time in enumerate(true_maneuvers_in_test[:5]):
            print(f"      {i+1}. {m_time}")
    
        # 显示检测到的异常
        if detected_anomalies:
            print(f"   -> 检测到的异常时间 (前5个):")
            for i, d_time in enumerate(detected_anomalies[:5]):
                print(f"      {i+1}. {d_time}")
                # 检查最近的真实机动
                if true_maneuvers_in_test:
                    closest_maneuver = min(true_maneuvers_in_test,
                                          key=lambda x: abs((x - d_time).total_seconds()))
                    distance_days = abs((closest_maneuver - d_time).total_seconds()) / 86400
                    print(f"         最近的真实机动距离: {distance_days:.1f} 天")
        else:
            print("   -> 未检测到任何异常。")
            
        # 创建真实标签
        true_labels = pd.Series(0, index=test_data.index)
        true_maneuver_in_test_count = 0
    
        for m_time in maneuver_times:
            if test_data.index.min() <= m_time <= test_data.index.max():
                true_maneuver_in_test_count += 1
                mask = (true_labels.index >= m_time - window) & (true_labels.index <= m_time + window)
                true_labels[mask] = 1
    
        # 创建预测标签
        pred_labels = pd.Series(0, index=test_data.index)
        if detected_anomalies:
            valid_anomalies = [ts for ts in detected_anomalies if ts in pred_labels.index]
            if valid_anomalies:
                pred_labels.loc[valid_anomalies] = 1
    
        # 计算混淆矩阵元素
        tp = ((pred_labels == 1) & (true_labels == 1)).sum()
        fp = ((pred_labels == 1) & (true_labels == 0)).sum()
        fn = ((pred_labels == 0) & (true_labels == 1)).sum()
        tn = ((pred_labels == 0) & (true_labels == 0)).sum()
    
        # 计算性能指标
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
        metrics = {
            'TP': int(tp),
            'FP': int(fp),
            'FN': int(fn),
            'TN': int(tn),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        }
    
        print(f"   -> 测试集中的真实机动事件数: {true_maneuver_in_test_count}")
        print(f"   -> 带标签的时间点数 (窗口={window.days*2}天): {(true_labels == 1).sum()}")
        print(f"   -> 性能指标:")
        print(f"      - 精确率 (Precision): {precision:.3f}")
        print(f"      - 召回率 (Recall):    {recall:.3f}")
        print(f"      - F1分数 (F1-Score):   {f1:.3f}")
        print(f"      - TP: {tp}, FP: {fp}, FN: {fn}")
    
        self.results['performance_metrics'] = metrics
        return metrics

    def _visualize_and_report(self, all_data, test_data, anomalies, scores, maneuver_times, metrics):
        """生成并保存图表和报告"""
        print("\n[7/7] 📊 生成可视化图表和报告...")

        # --- 可视化 ---
        fig, axes = plt.subplots(2, 1, figsize=(18, 12), sharex=True)
        fig.suptitle(f"Particle Filter Maneuver Detection for {self.config['satellite_name']}", fontsize=16)

        # 图1: Mean Motion 和检测结果
        ax1 = axes[0]
        ax1.plot(all_data.index, all_data['mean_motion'], color='gray', alpha=0.5, label='Mean Motion (All Data)')
        ax1.plot(test_data.index, test_data['mean_motion'], 'b-', label='Mean Motion (Test Set)')
        
        # 标记真实机动
        true_man_in_test = [m for m in maneuver_times if test_data.index.min() <= m <= test_data.index.max()]
        if true_man_in_test:
             ax1.vlines(true_man_in_test, ymin=ax1.get_ylim()[0], ymax=ax1.get_ylim()[1], color='green', linestyle='--', label='True Maneuver', lw=2)
        
        # 标记检测到的机动
        if anomalies:
            valid_anomalies = [a for a in anomalies if a in test_data.index]
            if valid_anomalies:
                anomaly_values = test_data.loc[valid_anomalies]['mean_motion']
                ax1.scatter(valid_anomalies, anomaly_values, color='red', s=80, marker='^', label='Detected Maneuver', zorder=5)
        
        ax1.set_ylabel("Mean Motion (rev/day)")
        ax1.set_title("Maneuver Detection Results")
        ax1.legend()
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

        # 图2: 异常分数和阈值
        ax2 = axes[1]
        # 异常分数是从第二个时间点开始的，所以需要对齐
        if len(scores) > 0:
            score_index = test_data.index[1:len(scores)+1]
            ax2.plot(score_index, scores, 'r-', alpha=0.8, label='Anomaly Score')
            ax2.axhline(y=self.detector.anomaly_threshold, color='black', linestyle=':', label=f"Threshold ({self.detector.anomaly_threshold:.2f})", lw=2)
            ax2.set_yscale('log') # 使用对数坐标轴以便观察

        
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Anomaly Score (Negative Log-Likelihood)")
        ax2.set_title("Anomaly Scores Over Time")
        ax2.legend()
        ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        fig_path = os.path.join(self.config['output_dir'], f"{self.config['satellite_name']}_pf_detection_results.png")
        plt.savefig(fig_path, dpi=300)
        plt.close(fig)
        print(f"   -> 可视化图表已保存到: {fig_path}")

        # --- 生成报告 ---
        report = {
            "experiment_config": self.config,
            "data_summary": {
                "total_tle_records": self.results.get('total_tle_records'),
                "total_maneuvers": self.results.get('total_maneuvers'),
                "processed_records": len(all_data),
                "train_set_size": len(test_data.iloc[:int(len(test_data)*self.config['train_split_ratio'])])
            },
            "model_summary": self.detector.get_model_summary(),
            "performance_metrics": metrics,
            "detected_anomaly_timestamps": [ts.isoformat() for ts in anomalies]
        }
        
        report_path = os.path.join(self.config['output_dir'], f"{self.config['satellite_name']}_pf_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        print(f"   -> 详细JSON报告已保存到: {report_path}")

# =====================================================================
# 主执行函数
# =====================================================================
def main():
    """主函数，配置并运行实验"""
    
    # --- 实验配置 ---
    # 你可以在这里更改卫星名称和粒子滤波器参数
    experiment_config = {
        'satellite_name': 'Fengyun-4A', # 可更换为: 'Fengyun-4A', 'Sentinel-3A', 'Jason-2' 等
        'data_dir': 'data',
        'output_dir': 'outputs/particle_filter_test',
        
        # 数据划分配置
        'train_split_ratio': 0.7, # 70% 的数据用于训练
        
        # 粒子滤波器检测器配置
        'n_particles': 1000,
        #'anomaly_threshold': 3,  # 初始值，可能会被自适应阈值覆盖
        'threshold_std_dev_factor': 1, 

        'use_auto_threshold': True,      # 新增：明确控制是否使用自动阈值

        #'anomaly_threshold': 15.0, 

        'use_only_mean_motion': True, # 是否只使用平均运动进行检测（True/False）
        'noise_scaling_factor': 0.1,   # 增加噪声以提高敏感度

        # 评估配置
        'label_window_days': 2.5, # 增加窗口大小，GEO卫星机动可能持续较长
    }

    # 运行实验
    experiment = ParticleFilterExperiment(experiment_config)
    experiment.run()


if __name__ == "__main__":
    main()