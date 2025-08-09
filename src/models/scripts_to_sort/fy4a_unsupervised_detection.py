# scripts/fy4a_unsupervised_detection.py
"""
风云4A卫星无监督机动检测实验

该脚本实现了从有监督学习到无监督学习的转换，并在风云4A卫星数据上进行验证。
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 导入必要的模块
from src.data.loader import SatelliteDataLoader
from src.data.feature_engineer import create_drift_enhanced_features
from src.models.unsupervised.xgboost_unsupervised_adapter import (
    XGBoostUnsupervisedAdapter, create_unsupervised_detector
)
from src.evaluation.visualization import plot_detection_results
from src.evaluation.reporting import save_detection_report

# 导入有监督学习的标签创建函数（用于评估）
try:
    from src.models.hybrid.xgboost_detector import create_labels_for_split
except ImportError:
    def create_labels_for_split(indices, maneuver_times, window):
        """简化的标签创建函数"""
        labels = pd.Series(0, index=indices)
        for maneuver_time in maneuver_times:
            mask = (indices >= maneuver_time - window) & (indices <= maneuver_time + window)
            labels[mask] = 1
        return labels


class UnsupervisedExperiment:
    """无监督机动检测实验类"""
    
    def __init__(self, config):
        self.config = config
        self.loader = SatelliteDataLoader(data_dir=config['data_dir'])
        self.detector = None
        self.results = {}
        
    def run_experiment(self):
        """运行完整的无监督检测实验"""
        print("\n" + "="*60)
        print("🚀 风云4A卫星无监督机动检测实验")
        print("="*60)
        
        # 1. 加载数据
        tle_data, maneuver_times = self._load_data()
        
        # 2. 特征工程
        enhanced_data = self._create_features(tle_data)
        
        # 3. 数据划分
        train_data, val_data, test_data = self._split_data(enhanced_data)
        

        # 迭代式训练流程
        print("\n\n--- 阶段一：初始训练与粗筛 ---")
        initial_config = self.config.copy()
        initial_config['threshold_method'] = 'percentile'
        initial_config['threshold_percentile'] = 99.0 # 可以设置一个相对宽松的百分位

        initial_detector = create_unsupervised_detector(initial_config)
        initial_detector.fit(train_data, target_column='target')

        # 在训练集上找出疑似异常点
        suspected_anomalies_indices, _ = initial_detector.detect_anomalies(train_data, return_scores=True)
    
        print(f"\n   🧹 在训练集中发现 {len(suspected_anomalies_indices)} 个疑似异常点，将用于净化数据。")

        # ---- 阶段二：净化数据并进行最终训练 ----
        print("\n\n--- 阶段二：净化数据与最终训练 ---")
    
        # 从训练数据中移除第一阶段找到的异常点
        if len(suspected_anomalies_indices) > 0:
            clean_train_data = train_data.drop(suspected_anomalies_indices)
            print(f"   原始训练样本: {len(train_data)}, 净化后样本: {len(clean_train_data)}")
        else:
            clean_train_data = train_data
            print("   未发现可净化的异常点，使用原始数据进行最终训练。")












        # 4. 训练无监督检测器
        self._train_unsupervised_detector(clean_train_data)
        
        # 5. 在验证集上优化参数
        #self._optimize_on_validation(val_data, maneuver_times)
        
        # 6. 在测试集上检测
        anomalies = self._detect_on_test(test_data)
        
        # 7. 评估结果（如果有真实标签）
        metrics = self._evaluate_results(test_data, anomalies, maneuver_times)
        
        # 8. 可视化和报告
        self._visualize_results(test_data, anomalies, maneuver_times)
        self._save_report(metrics, anomalies)
        
        print("\n✅ 实验完成！")
        return self.results
    
    def _load_data(self):
        """加载卫星数据"""
        print("\n📁 加载数据...")
        tle_data, maneuver_times = self.loader.load_satellite_data(
            self.config['satellite_name']
        )
        print(f"   - TLE记录数: {len(tle_data)}")
        print(f"   - 已知机动数: {len(maneuver_times)}")
        return tle_data, maneuver_times
    
    def _create_features(self, tle_data):
        """创建增强特征"""
        print("\n🔧 特征工程...")
        enhanced_data = create_drift_enhanced_features(
            tle_data, 
            scaling_factor=self.config['target_scaling_factor']
        )
        print(f"   - 特征数: {len(enhanced_data.columns)}")
        print(f"   - 有效样本: {len(enhanced_data)}")
        return enhanced_data
    
    def _split_data(self, enhanced_data):
        """划分数据集"""
        print("\n📊 数据划分...")
        n = len(enhanced_data)
        train_end = int(n * self.config['train_ratio'])
        val_end = int(n * (self.config['train_ratio'] + self.config['val_ratio']))
        
        train_data = enhanced_data.iloc[:train_end]
        val_data = enhanced_data.iloc[train_end:val_end]
        test_data = enhanced_data.iloc[val_end:]
        
        print(f"   - 训练集: {len(train_data)} ({self.config['train_ratio']*100:.0f}%)")
        print(f"   - 验证集: {len(val_data)} ({self.config['val_ratio']*100:.0f}%)")
        print(f"   - 测试集: {len(test_data)} ({self.config['test_ratio']*100:.0f}%)")
        
        return train_data, val_data, test_data
    
    def _train_unsupervised_detector(self, train_data):
        """训练无监督检测器"""
        print("\n🤖 训练无监督检测器...")
        
        # 创建检测器
        detector_config = {
            'threshold_method': self.config['threshold_method'],
            'threshold_factor': self.config['threshold_factor'],
            'percentile': self.config['threshold_percentile'],
            'min_segment_size': self.config['min_segment_size'],
            'max_gap_days': self.config['max_gap_days'],
            'enable_drift_adjustment': self.config['enable_drift_adjustment']
        }
        
        self.detector = create_unsupervised_detector(detector_config)
        


        # 如果有预训练模型路径，加载它
        if self.config.get('pretrained_model_path'):
            self.detector.load_pretrained_model(self.config['pretrained_model_path'])
        
        # 训练检测器
        self.detector.fit(train_data, target_column='target')
        
    def _optimize_on_validation(self, val_data, maneuver_times):
        print("\n🔍 验证集参数优化...")
    
        # 检查阈值方法是否为 'percentile'
        if self.config.get('threshold_method') != 'percentile':
            print(f"   - 当前方法为 '{self.config.get('threshold_method')}'，跳过百分位优化。")
            self.results['validation_f1'] = 'N/A'
            self.results['best_threshold_percentile'] = 'N/A'
            return
    
        # 定义要测试的百分位范围
        percentiles_to_test = np.linspace(99.0, 99.95, 20)
        best_f1 = -1.0
        best_percentile = self.config.get('threshold_percentile', 99.5)
    
        # 保存原始检测器状态，以便循环结束后恢复
        original_percentile = self.detector.percentile
    
        print(f"   - 正在为 {len(percentiles_to_test)} 个不同的百分位评估F1分数...")
    
        # 循环测试每个百分位
        for p in percentiles_to_test:
            self.detector.percentile = p
            self.detector._calculate_dynamic_threshold()  # 根据新百分位计算新阈值
    
            anomalies, _ = self.detector.detect_anomalies(val_data, return_scores=True)
            val_labels = create_labels_for_split(val_data.index, maneuver_times, timedelta(days=1))
    
            if val_labels.sum() == 0:
                continue
    
            pred_labels = pd.Series(0, index=val_data.index)
            if len(anomalies) > 0:
                # 确保异常点在验证集的索引内
                valid_anomalies = [a for a in anomalies if a in pred_labels.index]
                if valid_anomalies:
                    pred_labels.loc[valid_anomalies] = 1
    
            # 计算性能指标
            tp = ((pred_labels == 1) & (val_labels == 1)).sum()
            fp = ((pred_labels == 1) & (val_labels == 0)).sum()
            fn = ((pred_labels == 0) & (val_labels == 1)).sum()
    
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
            if f1 > best_f1:
                best_f1 = f1
                best_percentile = p
    
        # 应用在验证集上找到的最佳参数
        self.detector.percentile = best_percentile
        self.detector._calculate_dynamic_threshold()
    
        print(f"   - 最佳百分位: {best_percentile:.4f}")
        print(f"   - 验证集最佳F1: {best_f1:.3f}")
    
        # 将结果存入字典
        self.results['best_threshold_percentile'] = best_percentile
        self.results['validation_f1'] = best_f1
        
    def _detect_on_test(self, test_data):
        """在测试集上检测异常"""
        print("\n🎯 测试集检测...")
        
        anomalies, scores = self.detector.detect_anomalies(
            test_data, 
            return_scores=True
        )
        
        print(f"   - 检测到异常点: {len(anomalies)}")
        print(f"   - 异常比例: {len(anomalies)/len(test_data)*100:.2f}%")
        
        self.results['test_anomalies'] = anomalies
        self.results['anomaly_scores'] = scores
        
        return anomalies
    
    def _evaluate_results(self, test_data, anomalies, maneuver_times):
        """评估检测结果"""
        print("\n📈 评估结果...")

        # 如果没有提供真实的机动时间，则跳过传统评估
        if not maneuver_times or len(maneuver_times) == 0:
            print("   - 未提供真实机动数据，跳过精确率/召回率评估。")
            print(f"   - 将所有 {len(anomalies)} 个检测到的异常点标记为 '待查点'。")
            
            # 将所有检测到的异常都视为误报（因为没有真实机动）
            metrics = {
                'tp': 0, 'fp': len(anomalies), 
                'tn': len(test_data) - len(anomalies), 'fn': 0,
                'precision': 0, 'recall': 0, 'f1': 0,
                'notes': "No ground truth maneuver data provided for evaluation."
            }
            self.results['test_metrics'] = metrics
            return metrics
        
        # --- 如果有真实数据，则执行以下原始逻辑 ---
        print("   - 发现真实机动数据，进行标准评估...")
        # 创建真实标签
        test_labels = create_labels_for_split(
            test_data.index,
            maneuver_times,
            timedelta(days=self.config['label_window_days'])
        )
        
        # 创建预测标签
        pred_labels = pd.Series(0, index=test_data.index)
        if len(anomalies) > 0:
            # 使用.unique()方法为Pandas索引去重，然后进行索引
            unique_anomalies = anomalies.unique()
            pred_labels.loc[unique_anomalies] = 1
        
        # 计算指标
        tp = ((pred_labels == 1) & (test_labels == 1)).sum()
        fp = ((pred_labels == 1) & (test_labels == 0)).sum()
        tn = ((pred_labels == 0) & (test_labels == 0)).sum()
        fn = ((pred_labels == 0) & (test_labels == 1)).sum()
        
        metrics = {
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'f1': 0
        }
        
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / \
                           (metrics['precision'] + metrics['recall'])
        
        print(f"   - 精确率: {metrics['precision']:.3f}")
        print(f"   - 召回率: {metrics['recall']:.3f}")
        print(f"   - F1分数: {metrics['f1']:.3f}")
        
        self.results['test_metrics'] = metrics
        return metrics
    
    def _visualize_results(self, test_data, anomalies, maneuver_times):
        """可视化检测结果"""
        print("\n📊 生成可视化...")
        
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # 1. Mean Motion时序图
        ax = axes[0]
        ax.plot(test_data.index, test_data['mean_motion'], 'b-', alpha=0.7, label='Mean Motion')
        
        if len(anomalies) > 0:
            # 使用 .unique() 来处理可能的重复异常点
            unique_anomalies = anomalies.unique()
            anomaly_data = test_data.loc[unique_anomalies]
            ax.scatter(anomaly_data.index, anomaly_data['mean_motion'], 
                      color='red', s=50, alpha=0.8, label='Detected Anomalies')
        
        # --- 修改在这里：正确的图例处理方式 ---
        # 仅当有真实机动数据时才绘制标记
        if maneuver_times and len(maneuver_times) > 0:
            # 标记只在第一次绘制时添加，以避免图例重复
            first_maneuver_plotted = False
            for maneuver_time in maneuver_times:
                if test_data.index[0] <= maneuver_time <= test_data.index[-1]:
                    if not first_maneuver_plotted:
                        ax.axvline(x=maneuver_time, color='green', linestyle='--', label='True Maneuvers')
                        first_maneuver_plotted = True
                    else:
                        ax.axvline(x=maneuver_time, color='green', linestyle='--')
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Mean Motion')
        ax.set_title('Mean Motion and Detected Anomalies')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 异常分数分布
        ax = axes[1]
        if 'anomaly_scores' in self.results:
            scores = self.results['anomaly_scores']
            ax.hist(scores, bins=50, alpha=0.7, color='blue', edgecolor='black')
            ax.axvline(x=self.detector.dynamic_threshold, color='red', 
                      linestyle='--', linewidth=2, label=f'Threshold: {self.detector.dynamic_threshold:.4f}')
            ax.set_xlabel('Anomaly Score (Residuals)')
            ax.set_ylabel('Frequency')
            ax.set_title('Anomaly Score Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 3. 漂移特征与异常的关系
        # 3. 漂移特征与异常的关系
        ax = axes[2]
        drift_features = [col for col in test_data.columns if 'drift' in col]
        if drift_features:
            drift_metric = test_data[drift_features[0]]
            ax.plot(test_data.index, drift_metric, 'g-', alpha=0.7, label=drift_features[0])
            
            if len(anomalies) > 0:
                unique_anomalies = anomalies.unique()
                
                drift_metric_unique_idx = drift_metric[~drift_metric.index.duplicated(keep='first')]
                
                valid_anomalies_to_plot = unique_anomalies.intersection(drift_metric_unique_idx.index)
                
                if not valid_anomalies_to_plot.empty:
                    y_values = drift_metric_unique_idx.loc[valid_anomalies_to_plot]
                    ax.scatter(valid_anomalies_to_plot, y_values,
                               color='red', s=50, alpha=0.8, label='Anomaly Points')

            ax.set_xlabel('Time')
            ax.set_ylabel('Drift Metric')
            ax.set_title('Drift Features and Anomaly Detection')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = os.path.join(self.config['output_dir'], 'unsupervised_detection_results.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   - 可视化已保存到: {output_path}")
        
    def _save_report(self, metrics, anomalies):
        """保存检测报告"""
        print("\n📄 生成报告...")
        
        report_path = os.path.join(self.config['output_dir'], 'unsupervised_detection_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("风云4A卫星无监督机动检测报告\n")
            f.write("="*60 + "\n\n")
            
            f.write("1. 实验配置:\n")
            f.write(f"   - 卫星名称: {self.config['satellite_name']}\n")
            f.write(f"   - 阈值方法: {self.config['threshold_method']}\n")


            if 'best_threshold_percentile' in self.results:
                f.write(f"   - 最佳阈值百分位: {self.results['best_threshold_percentile']:.4f}\n")
            elif 'best_threshold_factor' in self.results:
                f.write(f"   - 最佳阈值因子: {self.results['best_threshold_factor']:.2f}\n")



            f.write(f"   - 最小段大小: {self.config['min_segment_size']}\n")
            f.write(f"   - 最大间隔天数: {self.config['max_gap_days']}\n\n")
            
            f.write("2. 检测结果:\n")
            f.write(f"   - 检测到的异常数: {len(anomalies)}\n")
            f.write(f"   - 动态阈值: {self.detector.dynamic_threshold:.6f}\n\n")
            
            if metrics:
                f.write("3. 性能评估:\n")
                f.write(f"   - 精确率: {metrics['precision']:.3f}\n")
                f.write(f"   - 召回率: {metrics['recall']:.3f}\n")
                f.write(f"   - F1分数: {metrics['f1']:.3f}\n")
                f.write(f"   - 混淆矩阵:\n")
                f.write(f"     TP: {metrics['tp']}, FP: {metrics['fp']}\n")
                f.write(f"     FN: {metrics['fn']}, TN: {metrics['tn']}\n\n")
            
            f.write("4. 检测到的异常时间（前20个）:\n")
            for i, anomaly_time in enumerate(list(anomalies)[:20]):
                f.write(f"   {i+1:2d}. {anomaly_time}\n")
            
            if len(anomalies) > 20:
                f.write(f"   ... (还有 {len(anomalies)-20} 个)\n")
        
        print(f"   - 报告已保存到: {report_path}")


def main():
    """主函数"""
    # 实验配置
    config = {
        # 数据配置
        'satellite_name': 'Fengyun-4A',
        'data_dir': 'data',
        'output_dir': 'outputs/fy4a_unsupervised',
        
        # 数据划分
        'train_ratio': 0.6,
        'val_ratio': 0.2,
        'test_ratio': 0.2,
        
        # 特征工程
        'target_scaling_factor': 1e8,
        
        # 无监督检测器配置
        'threshold_method': 'percentile',  # 'std', 'percentile', 'mad'
        'threshold_factor': 3.5,
        'threshold_percentile': 95,
        'min_segment_size': 1,
        'max_gap_days': 1.5,
        'enable_drift_adjustment': True,
        
        # 评估配置
        'label_window_days': 1,
        
        # 可选：预训练模型路径
        'pretrained_model_path': None  # 如果有的话，填入路径
    }
    
    # 创建输出目录
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # 运行实验
    experiment = UnsupervisedExperiment(config)
    results = experiment.run_experiment()
    
    # 打印最终结果摘要
    print("\n" + "="*60)
    print("📊 实验结果摘要")
    print("="*60)
    
    if 'test_metrics' in results:
        metrics = results['test_metrics']
        if metrics.get('notes'):
            print(f"评估备注: {metrics['notes']}")
            print(f"  - 检测到的待查点数: {metrics['fp']}")
        else:
            print(f"最终性能:")
            print(f"  - 精确率: {metrics['precision']:.3f}")
            print(f"  - 召回率: {metrics['recall']:.3f}")
            print(f"  - F1分数: {metrics['f1']:.3f}")
    
    print(f"\n验证集最佳F1: {results.get('validation_f1', 'N/A')}")
    print(f"最佳阈值因子: {results.get('best_threshold_factor', 'N/A')}")
    
    return results


if __name__ == "__main__":
    main()