# scripts/compare_supervised_unsupervised.py
"""
有监督与无监督XGBoost机动检测方法对比

该脚本对比分析有监督和无监督两种方法在相同数据集上的表现。
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple

# 设置项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 导入必要的模块
from src.data.loader import SatelliteDataLoader
from src.data.feature_engineer import create_drift_enhanced_features
from src.models.unsupervised.xgboost_unsupervised_adapter import create_unsupervised_detector
from src.models.hybrid.enhanced_xgboost_detector import ImprovedXGBoostDetector
from src.models.hybrid.xgboost_detector import create_labels_for_split


class MethodComparison:
    """方法对比实验类"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.results = {
            'supervised': {},
            'unsupervised': {}
        }
        
    def run_comparison(self):
        """运行对比实验"""
        print("\n" + "="*70)
        print("🔬 有监督 vs 无监督 XGBoost 机动检测对比实验")
        print("="*70)
        
        # 1. 加载和准备数据
        tle_data, maneuver_times, enhanced_data = self._prepare_data()
        
        # 2. 数据划分
        train_data, test_data = self._split_data(enhanced_data)
        
        # 3. 运行有监督方法
        print("\n" + "-"*50)
        print("📚 运行有监督方法...")
        supervised_results = self._run_supervised(train_data, test_data, maneuver_times)
        self.results['supervised'] = supervised_results
        
        # 4. 运行无监督方法
        print("\n" + "-"*50)
        print("🔓 运行无监督方法...")
        unsupervised_results = self._run_unsupervised(train_data, test_data, maneuver_times)
        self.results['unsupervised'] = unsupervised_results
        
        # 5. 对比分析
        print("\n" + "-"*50)
        self._compare_results()
        
        # 6. 可视化对比
        self._visualize_comparison(test_data, maneuver_times)
        
        # 7. 生成对比报告
        self._generate_comparison_report()
        
        return self.results
    
    def _prepare_data(self):
        """准备数据"""
        print("\n📁 数据准备...")
        
        # 加载数据
        loader = SatelliteDataLoader(data_dir=self.config['data_dir'])
        tle_data, maneuver_times = loader.load_satellite_data(
            self.config['satellite_name']
        )
        
        # 特征工程
        enhanced_data = create_drift_enhanced_features(
            tle_data,
            scaling_factor=self.config['target_scaling_factor']
        )
        
        print(f"   - 总样本数: {len(enhanced_data)}")
        print(f"   - 特征数: {len(enhanced_data.columns)}")
        print(f"   - 已知机动数: {len(maneuver_times)}")
        
        return tle_data, maneuver_times, enhanced_data
    
    def _split_data(self, enhanced_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """划分数据集"""
        split_idx = int(len(enhanced_data) * self.config['train_ratio'])
        train_data = enhanced_data.iloc[:split_idx]
        test_data = enhanced_data.iloc[split_idx:]
        
        print(f"\n📊 数据划分:")
        print(f"   - 训练集: {len(train_data)} 样本")
        print(f"   - 测试集: {len(test_data)} 样本")
        
        return train_data, test_data
    
    def _run_supervised(self, train_data: pd.DataFrame, test_data: pd.DataFrame, 
                       maneuver_times: list) -> Dict:
        """运行有监督方法"""
        
        # 创建训练标签
        train_labels = create_labels_for_split(
            train_data.index,
            maneuver_times,
            timedelta(days=self.config['label_window_days'])
        )
        
        # 提取正常数据用于训练
        normal_train_data = train_data[train_labels == 0]
        
        print(f"   - 正常训练样本: {len(normal_train_data)}")
        print(f"   - 异常训练样本: {train_labels.sum()}")
        
        # 配置并训练有监督检测器
        xgb_params = {
            'objective': 'reg:squarederror',
            'n_estimators': 300,
            'learning_rate': 0.03,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        
        detector = ImprovedXGBoostDetector(
            target_column='target',
            xgb_params=xgb_params,
            threshold_quantile=0.995,
            enable_threshold_optimization=True,
            enable_temporal_clustering=True,
            satellite_type='FY-4A'
        )
        
        # 训练模型
        detector.fit(
            train_features=normal_train_data,
            satellite_name="Fengyun-4A",
            verbose=False
        )
        
        # 检测异常
        anomalies, scores = detector.detect_anomalies(test_data, return_scores=True)
        
        # 评估结果
        metrics = self._evaluate_predictions(test_data, anomalies, maneuver_times)
        
        return {
            'anomalies': anomalies,
            'scores': scores,
            'metrics': metrics,
            'detector': detector
        }
    
    def _run_unsupervised(self, train_data: pd.DataFrame, test_data: pd.DataFrame,
                         maneuver_times: list) -> Dict:
        """运行无监督方法"""
        
        # 创建无监督检测器
        detector = create_unsupervised_detector({
            'threshold_method': 'mad',
            'threshold_factor': 3.5,
            'min_segment_size': 3,
            'max_gap_days': 1.5,
            'enable_drift_adjustment': True
        })
        
        # 训练（使用所有训练数据）
        detector.fit(train_data, target_column='target')
        
        # 检测异常
        anomalies, scores = detector.detect_anomalies(test_data, return_scores=True)
        
        # 评估结果
        metrics = self._evaluate_predictions(test_data, anomalies, maneuver_times)
        
        return {
            'anomalies': anomalies,
            'scores': scores,
            'metrics': metrics,
            'detector': detector
        }
    
    def _evaluate_predictions(self, test_data: pd.DataFrame, anomalies: pd.Index,
                            maneuver_times: list) -> Dict:
        """评估预测结果"""
        
        # 创建真实标签
        test_labels = create_labels_for_split(
            test_data.index,
            maneuver_times,
            timedelta(days=self.config['label_window_days'])
        )
        
        # 创建预测标签
        pred_labels = pd.Series(0, index=test_data.index)
        pred_labels[anomalies] = 1
        
        # 计算混淆矩阵
        tp = ((pred_labels == 1) & (test_labels == 1)).sum()
        fp = ((pred_labels == 1) & (test_labels == 0)).sum()
        tn = ((pred_labels == 0) & (test_labels == 0)).sum()
        fn = ((pred_labels == 0) & (test_labels == 1)).sum()
        
        # 计算指标
        metrics = {
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'f1': 0,
            'accuracy': (tp + tn) / (tp + tn + fp + fn)
        }
        
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / \
                           (metrics['precision'] + metrics['recall'])
        
        return metrics
    
    def _compare_results(self):
        """对比两种方法的结果"""
        print("\n📊 性能对比:")
        print("-" * 50)
        print(f"{'指标':<10} {'有监督':<15} {'无监督':<15} {'差异':<10}")
        print("-" * 50)
        
        metrics_to_compare = ['precision', 'recall', 'f1', 'accuracy']
        
        for metric in metrics_to_compare:
            sup_val = self.results['supervised']['metrics'][metric]
            unsup_val = self.results['unsupervised']['metrics'][metric]
            diff = sup_val - unsup_val
            
            print(f"{metric:<10} {sup_val:<15.3f} {unsup_val:<15.3f} {diff:+.3f}")
        
        print("-" * 50)
        
        # 检测数量对比
        sup_count = len(self.results['supervised']['anomalies'])
        unsup_count = len(self.results['unsupervised']['anomalies'])
        
        print(f"\n检测到的异常数:")
        print(f"  - 有监督: {sup_count}")
        print(f"  - 无监督: {unsup_count}")
        
        # 计算重叠
        overlap = len(set(self.results['supervised']['anomalies']) & 
                     set(self.results['unsupervised']['anomalies']))
        
        print(f"  - 重叠异常: {overlap}")
        print(f"  - Jaccard相似度: {overlap / (sup_count + unsup_count - overlap):.3f}")
    
    def _visualize_comparison(self, test_data: pd.DataFrame, maneuver_times: list):
        """可视化对比结果"""
        print("\n📈 生成对比可视化...")
        
        fig, axes = plt.subplots(4, 1, figsize=(15, 16))
        
        # 1. Mean Motion 与两种方法的检测结果
        ax = axes[0]
        ax.plot(test_data.index, test_data['mean_motion'], 'k-', alpha=0.5, label='Mean Motion')
        
        # 有监督检测结果
        sup_anomalies = self.results['supervised']['anomalies']
        if len(sup_anomalies) > 0:
            sup_data = test_data.loc[sup_anomalies]
            ax.scatter(sup_data.index, sup_data['mean_motion'], 
                      color='blue', s=60, alpha=0.7, marker='o', label='有监督检测')
        
        # 无监督检测结果
        unsup_anomalies = self.results['unsupervised']['anomalies']
        if len(unsup_anomalies) > 0:
            unsup_data = test_data.loc[unsup_anomalies]
            ax.scatter(unsup_data.index, unsup_data['mean_motion'], 
                      color='red', s=40, alpha=0.7, marker='s', label='无监督检测')
        
        # 真实机动
        for maneuver_time in maneuver_times:
            if test_data.index[0] <= maneuver_time <= test_data.index[-1]:
                ax.axvline(x=maneuver_time, color='green', linestyle='--', 
                          alpha=0.5, linewidth=1)
        
        ax.set_xlabel('时间')
        ax.set_ylabel('Mean Motion')
        ax.set_title('Mean Motion 时序与检测结果对比')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 异常分数分布对比
        ax = axes[1]
        
        # 有监督分数
        sup_scores = self.results['supervised']['scores']
        ax.hist(sup_scores, bins=50, alpha=0.5, color='blue', 
                label='有监督', density=True)
        
        # 无监督分数
        unsup_scores = self.results['unsupervised']['scores']
        ax.hist(unsup_scores, bins=50, alpha=0.5, color='red', 
                label='无监督', density=True)
        
        ax.set_xlabel('异常分数')
        ax.set_ylabel('密度')
        ax.set_title('异常分数分布对比')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. 性能指标对比
        ax = axes[2]
        
        metrics = ['Precision', 'Recall', 'F1', 'Accuracy']
        sup_values = [self.results['supervised']['metrics'][m.lower()] for m in metrics]
        unsup_values = [self.results['unsupervised']['metrics'][m.lower()] for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax.bar(x - width/2, sup_values, width, label='有监督', color='blue', alpha=0.7)
        ax.bar(x + width/2, unsup_values, width, label='无监督', color='red', alpha=0.7)
        
        ax.set_xlabel('指标')
        ax.set_ylabel('值')
        ax.set_title('性能指标对比')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 4. 混淆矩阵对比
        ax = axes[3]
        ax.axis('off')
        
        # 创建混淆矩阵表格
        sup_m = self.results['supervised']['metrics']
        unsup_m = self.results['unsupervised']['metrics']
        
        table_data = [
            ['方法', 'TP', 'FP', 'TN', 'FN'],
            ['有监督', sup_m['tp'], sup_m['fp'], sup_m['tn'], sup_m['fn']],
            ['无监督', unsup_m['tp'], unsup_m['fp'], unsup_m['tn'], unsup_m['fn']]
        ]
        
        table = ax.table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        
        # 设置表格样式
        for i in range(len(table_data)):
            for j in range(len(table_data[0])):
                cell = table[(i, j)]
                if i == 0:
                    cell.set_facecolor('#40466e')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        ax.set_title('混淆矩阵对比', pad=20)
        
        plt.tight_layout()
        
        # 保存图像
        output_path = os.path.join(self.config['output_dir'], 'method_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   - 可视化已保存到: {output_path}")
    
    def _generate_comparison_report(self):
        """生成对比报告"""
        print("\n📄 生成对比报告...")
        
        report_path = os.path.join(self.config['output_dir'], 'comparison_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("有监督 vs 无监督 XGBoost 机动检测对比报告\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"卫星: {self.config['satellite_name']}\n")
            f.write(f"训练集比例: {self.config['train_ratio']*100:.0f}%\n\n")
            
            # 性能对比表
            f.write("1. 性能指标对比:\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'指标':<15} {'有监督':<15} {'无监督':<15} {'优势':<10}\n")
            f.write("-" * 50 + "\n")
            
            metrics = ['precision', 'recall', 'f1', 'accuracy']
            for metric in metrics:
                sup = self.results['supervised']['metrics'][metric]
                unsup = self.results['unsupervised']['metrics'][metric]
                better = "有监督" if sup > unsup else "无监督" if unsup > sup else "相同"
                f.write(f"{metric:<15} {sup:<15.3f} {unsup:<15.3f} {better:<10}\n")
            
            f.write("\n2. 检测结果对比:\n")
            sup_count = len(self.results['supervised']['anomalies'])
            unsup_count = len(self.results['unsupervised']['anomalies'])
            overlap = len(set(self.results['supervised']['anomalies']) & 
                         set(self.results['unsupervised']['anomalies']))
            
            f.write(f"   - 有监督检测数: {sup_count}\n")
            f.write(f"   - 无监督检测数: {unsup_count}\n")
            f.write(f"   - 重叠检测数: {overlap}\n")
            f.write(f"   - Jaccard相似度: {overlap / (sup_count + unsup_count - overlap):.3f}\n")
            
            f.write("\n3. 优缺点分析:\n")
            f.write("\n有监督方法:\n")
            f.write("   优点:\n")
            f.write("   - 可以利用已知的机动标签进行训练\n")
            f.write("   - 通常具有更高的精确率\n")
            f.write("   - 可以进行阈值优化\n")
            f.write("   缺点:\n")
            f.write("   - 需要标注数据\n")
            f.write("   - 可能对新型机动模式泛化能力有限\n")
            
            f.write("\n无监督方法:\n")
            f.write("   优点:\n")
            f.write("   - 不需要标注数据\n")
            f.write("   - 可以发现未知的异常模式\n")
            f.write("   - 更容易部署到新卫星\n")
            f.write("   缺点:\n")
            f.write("   - 阈值设置可能需要更多调试\n")
            f.write("   - 可能产生更多误报\n")
            
            f.write("\n4. 建议:\n")
            if self.results['supervised']['metrics']['f1'] > self.results['unsupervised']['metrics']['f1'] + 0.1:
                f.write("   - 当前数据集上，有监督方法表现明显更好\n")
                f.write("   - 建议在有充足标注数据时使用有监督方法\n")
            elif self.results['unsupervised']['metrics']['f1'] > self.results['supervised']['metrics']['f1'] + 0.1:
                f.write("   - 当前数据集上，无监督方法表现更好\n")
                f.write("   - 可能是由于训练数据中的标注不完整\n")
            else:
                f.write("   - 两种方法表现相近\n")
                f.write("   - 建议根据实际应用场景选择\n")
                f.write("   - 可以考虑集成两种方法的结果\n")
        
        print(f"   - 报告已保存到: {report_path}")


def main():
    """主函数"""
    config = {
        'satellite_name': 'Fengyun-4A',
        'data_dir': 'data',
        'output_dir': 'outputs/method_comparison',
        'train_ratio': 0.7,
        'target_scaling_factor': 1e8,
        'label_window_days': 1
    }
    
    # 创建输出目录
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # 运行对比实验
    comparison = MethodComparison(config)
    results = comparison.run_comparison()
    
    print("\n" + "="*70)
    print("✅ 对比实验完成！")
    print("="*70)
    
    return results


if __name__ == "__main__":
    main()