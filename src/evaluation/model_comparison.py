# src/evaluation/model_comparison.py
"""
无监督模型竞赛框架 (Model Bakeoff Framework)

这个模块实现了方案中推荐的"模型竞赛"概念，用于客观地评估和比较
不同无监督机动检测方法的性能。通过使用2020-2022年的风云4A机动日志
作为地面真实值，我们可以对各种无监督算法进行严格的基准测试。

Key Features:
- 统一的模型接口和评估框架
- 标准化的性能指标计算
- 自动化的模型选择和排名
- 详细的性能分析报告
- 可视化的结果对比
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Protocol
from datetime import datetime, timedelta
from dataclasses import dataclass
from abc import ABC, abstractmethod

import warnings
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, roc_curve, auc, confusion_matrix
)

# 导入我们的检测器
from src.models.unsupervised.particle_filter import ParticleFilterDetector
from src.models.unsupervised.lstm_vae import LSTMVAEDetector

@dataclass
class ModelPerformance:
    """模型性能结果数据类"""
    model_name: str
    precision: float
    recall: float
    f1_score: float
    auc_pr: float  # PR曲线下面积
    auc_roc: float  # ROC曲线下面积
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    total_detections: int
    total_true_events: int
    anomaly_scores: np.ndarray
    detection_timestamps: List[datetime]
    training_time: float
    detection_time: float
    model_params: Dict[str, Any]

class UnsupervisedDetector(Protocol):
    """无监督检测器的统一接口协议"""
    
    def fit(self, train_data: pd.DataFrame, 
            maneuver_labels: Optional[pd.Series] = None) -> None:
        """训练检测器"""
        ...
    
    def detect_anomalies(self, test_data: pd.DataFrame) -> Tuple[List[datetime], np.ndarray]:
        """检测异常，返回异常时刻和异常分数"""
        ...
    
    def get_model_summary(self) -> Dict[str, Any]:
        """获取模型摘要"""
        ...

class ModelBakeoff:
    """
    无监督模型竞赛框架
    
    这个类管理多个无监督检测器的训练、评估和比较，
    提供客观的性能评估和模型选择建议。
    """
    
    def __init__(self, 
                 tolerance_window: timedelta = timedelta(days=1),
                 min_time_between_events: timedelta = timedelta(hours=12)):
        """
        初始化竞赛框架
        
        Args:
            tolerance_window: 检测容忍窗口（检测在真实事件前后此时间内算正确）
            min_time_between_events: 连续事件间最小时间间隔
        """
        self.tolerance_window = tolerance_window
        self.min_time_between_events = min_time_between_events
        
        # 注册的模型
        self.models: Dict[str, UnsupervisedDetector] = {}
        self.model_configs: Dict[str, Dict] = {}
        
        # 评估结果
        self.results: Dict[str, ModelPerformance] = {}
        self.ground_truth_events: List[datetime] = []
        
        # 数据集信息
        self.train_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None
        self.train_labels: Optional[pd.Series] = None
        self.test_labels: Optional[pd.Series] = None
        
        print("🏁 Model Bakeoff Framework initialized")
    
    def register_model(self, name: str, model: UnsupervisedDetector, 
                      config: Optional[Dict] = None) -> None:
        """
        注册参赛模型
        
        Args:
            name: 模型名称
            model: 检测器实例
            config: 模型配置信息
        """
        self.models[name] = model
        self.model_configs[name] = config or {}
        print(f"✅ Registered model: {name}")
    
    def setup_data(self, train_data: pd.DataFrame, test_data: pd.DataFrame,
                   train_labels: pd.Series, test_labels: pd.Series) -> None:
        """
        设置训练和测试数据
        
        Args:
            train_data: 训练数据
            test_data: 测试数据  
            train_labels: 训练标签
            test_labels: 测试标签
        """
        self.train_data = train_data
        self.test_data = test_data
        self.train_labels = train_labels
        self.test_labels = test_labels
        
        # 提取真实事件时刻
        self.ground_truth_events = test_labels[test_labels == 1].index.tolist()
        
        print(f"📊 Data setup complete:")
        print(f"   -> Training data: {len(train_data)} records")
        print(f"   -> Test data: {len(test_data)} records")
        print(f"   -> Ground truth events: {len(self.ground_truth_events)}")
    
    def run_competition(self, verbose: bool = True) -> Dict[str, ModelPerformance]:
        """
        运行模型竞赛
        
        Args:
            verbose: 是否显示详细信息
            
        Returns:
            所有模型的性能结果
        """
        if not self.models:
            raise ValueError("No models registered for competition")
        
        if any(data is None for data in [self.train_data, self.test_data, 
                                        self.train_labels, self.test_labels]):
            raise ValueError("Data not setup. Call setup_data() first")
        
        print("\n🏆 Starting Model Competition...")
        print("=" * 50)
        
        for model_name, model in self.models.items():
            if verbose:
                print(f"\n🔧 Evaluating model: {model_name}")
            
            try:
                # 评估单个模型
                performance = self._evaluate_single_model(
                    model_name, model, verbose
                )
                self.results[model_name] = performance
                
                if verbose:
                    print(f"✅ {model_name} evaluation completed")
                    self._print_model_summary(performance)
                    
            except Exception as e:
                print(f"❌ Error evaluating {model_name}: {e}")
                if verbose:
                    import traceback
                    traceback.print_exc()
        
        # 生成竞赛报告
        if verbose:
            self._print_competition_report()
        
        return self.results
    
    def _evaluate_single_model(self, name: str, model: UnsupervisedDetector,
                              verbose: bool = True) -> ModelPerformance:
        """评估单个模型"""
        import time
        
        # 1. 训练模型
        start_time = time.time()
        model.fit(self.train_data, self.train_labels)
        training_time = time.time() - start_time
        
        if verbose:
            print(f"   -> Training completed in {training_time:.2f}s")
        
        # 2. 检测异常
        start_time = time.time()
        detection_timestamps, anomaly_scores = model.detect_anomalies(self.test_data)
        detection_time = time.time() - start_time
        
        if verbose:
            print(f"   -> Detection completed in {detection_time:.2f}s")
            print(f"   -> Detected {len(detection_timestamps)} anomalies")
        
        # 3. 计算性能指标
        performance = self._compute_performance_metrics(
            name, detection_timestamps, anomaly_scores,
            training_time, detection_time, model.get_model_summary()
        )
        
        return performance
    
    def _compute_performance_metrics(self, model_name: str,
                                   detection_timestamps: List[datetime],
                                   anomaly_scores: np.ndarray,
                                   training_time: float,
                                   detection_time: float,
                                   model_params: Dict) -> ModelPerformance:
        """计算详细的性能指标"""
        
        # 1. 创建二进制预测向量
        predictions = self._create_prediction_vector(detection_timestamps)
        
        # 2. 对齐预测和真实标签
        aligned_predictions, aligned_labels = self._align_predictions_labels(predictions)
        
        # 3. 计算基础指标
        tp = np.sum((aligned_predictions == 1) & (aligned_labels == 1))
        fp = np.sum((aligned_predictions == 1) & (aligned_labels == 0))
        fn = np.sum((aligned_predictions == 0) & (aligned_labels == 1))
        tn = np.sum((aligned_predictions == 0) & (aligned_labels == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # 4. 计算ROC和PR曲线
        try:
            # 对异常分数进行对齐
            aligned_scores = self._align_scores(anomaly_scores)
            
            if len(np.unique(aligned_labels)) > 1:  # 确保有正负样本
                auc_roc = roc_auc_score(aligned_labels, aligned_scores)
                
                # 计算PR AUC
                precision_curve, recall_curve, _ = precision_recall_curve(aligned_labels, aligned_scores)
                auc_pr = auc(recall_curve, precision_curve)
            else:
                auc_roc = 0.0
                auc_pr = 0.0
                
        except Exception as e:
            print(f"Warning: Could not compute AUC metrics for {model_name}: {e}")
            auc_roc = 0.0
            auc_pr = 0.0
        
        return ModelPerformance(
            model_name=model_name,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_pr=auc_pr,
            auc_roc=auc_roc,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            true_negatives=tn,
            total_detections=len(detection_timestamps),
            total_true_events=len(self.ground_truth_events),
            anomaly_scores=anomaly_scores,
            detection_timestamps=detection_timestamps,
            training_time=training_time,
            detection_time=detection_time,
            model_params=model_params
        )
    
    def _create_prediction_vector(self, detection_timestamps: List[datetime]) -> pd.Series:
        """创建二进制预测向量"""
        predictions = pd.Series(0, index=self.test_data.index)
        
        for det_time in detection_timestamps:
            # 在容忍窗口内标记为检测到
            window_start = det_time - self.tolerance_window
            window_end = det_time + self.tolerance_window
            
            mask = (predictions.index >= window_start) & (predictions.index <= window_end)
            predictions.loc[mask] = 1
        
        return predictions
    
    def _align_predictions_labels(self, predictions: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """对齐预测和标签"""
        # 确保索引对齐
        common_index = predictions.index.intersection(self.test_labels.index)
        
        aligned_predictions = predictions.reindex(common_index, fill_value=0).values
        aligned_labels = self.test_labels.reindex(common_index, fill_value=0).values
        
        return aligned_predictions, aligned_labels
    
    def _align_scores(self, anomaly_scores: np.ndarray) -> np.ndarray:
        """对齐异常分数"""
        # 由于异常分数可能比test_data短（序列处理），需要处理对齐
        if len(anomaly_scores) == len(self.test_data):
            return anomaly_scores
        elif len(anomaly_scores) < len(self.test_data):
            # 用最后一个值填充
            aligned_scores = np.full(len(self.test_data), anomaly_scores[-1])
            aligned_scores[:len(anomaly_scores)] = anomaly_scores
            return aligned_scores
        else:
            # 截断
            return anomaly_scores[:len(self.test_data)]
    
    def _print_model_summary(self, performance: ModelPerformance) -> None:
        """打印单个模型的性能摘要"""
        print(f"   📊 Performance Summary:")
        print(f"      Precision: {performance.precision:.3f}")
        print(f"      Recall: {performance.recall:.3f}")
        print(f"      F1-Score: {performance.f1_score:.3f}")
        print(f"      AUC-PR: {performance.auc_pr:.3f}")
        print(f"      AUC-ROC: {performance.auc_roc:.3f}")
        print(f"      Detections: {performance.total_detections}")
        print(f"      True Events: {performance.total_true_events}")
    
    def _print_competition_report(self) -> None:
        """打印竞赛总结报告"""
        print("\n" + "=" * 60)
        print("🏆 MODEL COMPETITION RESULTS")
        print("=" * 60)
        
        if not self.results:
            print("No results available")
            return
        
        # 按F1分数排序
        sorted_results = sorted(
            self.results.values(),
            key=lambda x: x.f1_score,
            reverse=True
        )
        
        print(f"\n📊 Performance Ranking (by F1-Score):")
        print("-" * 80)
        print(f"{'Rank':<4} {'Model':<20} {'F1':<6} {'Precision':<9} {'Recall':<7} {'AUC-PR':<7} {'Detections':<11}")
        print("-" * 80)
        
        for i, result in enumerate(sorted_results, 1):
            print(f"{i:<4} {result.model_name:<20} {result.f1_score:.3f}  "
                  f"{result.precision:.3f}     {result.recall:.3f}   "
                  f"{result.auc_pr:.3f}   {result.total_detections:<11}")
        
        # 推荐最佳模型
        if sorted_results:
            best_model = sorted_results[0]
            print(f"\n🥇 RECOMMENDED MODEL: {best_model.model_name}")
            print(f"   F1-Score: {best_model.f1_score:.3f}")
            print(f"   Balanced performance with {best_model.true_positives} true positives")
            print(f"   and {best_model.false_positives} false positives")
    
    def get_winner(self) -> Optional[str]:
        """获取获胜模型名称（基于F1分数）"""
        if not self.results:
            return None
        
        best_model = max(self.results.values(), key=lambda x: x.f1_score)
        return best_model.model_name
    
    def get_detailed_report(self) -> pd.DataFrame:
        """获取详细的比较报告"""
        if not self.results:
            return pd.DataFrame()
        
        report_data = []
        for result in self.results.values():
            report_data.append({
                'Model': result.model_name,
                'Precision': result.precision,
                'Recall': result.recall,
                'F1-Score': result.f1_score,
                'AUC-PR': result.auc_pr,
                'AUC-ROC': result.auc_roc,
                'True Positives': result.true_positives,
                'False Positives': result.false_positives,
                'False Negatives': result.false_negatives,
                'Total Detections': result.total_detections,
                'Training Time (s)': result.training_time,
                'Detection Time (s)': result.detection_time
            })
        
        return pd.DataFrame(report_data).sort_values('F1-Score', ascending=False)
    
    def save_results(self, filepath: str) -> None:
        """保存竞赛结果"""
        report_df = self.get_detailed_report()
        report_df.to_csv(filepath, index=False)
        print(f"✅ Results saved to: {filepath}")


# 可视化工具
class BakeoffVisualizer:
    """竞赛结果可视化工具"""
    
    @staticmethod
    def plot_performance_comparison(bakeoff: ModelBakeoff):
        """绘制性能比较图"""
        try:
            import matplotlib.pyplot as plt
            
            if not bakeoff.results:
                print("No results to plot")
                return
            
            models = list(bakeoff.results.keys())
            metrics = ['precision', 'recall', 'f1_score', 'auc_pr']
            metric_names = ['Precision', 'Recall', 'F1-Score', 'AUC-PR']
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            
            for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
                values = [getattr(bakeoff.results[model], metric) for model in models]
                
                bars = axes[i].bar(models, values, alpha=0.7)
                axes[i].set_title(f'{metric_name} Comparison')
                axes[i].set_ylabel(metric_name)
                axes[i].set_ylim(0, 1)
                
                # 添加数值标签
                for bar, value in zip(bars, values):
                    axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                f'{value:.3f}', ha='center', va='bottom')
                
                # 旋转x轴标签
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.suptitle('Model Performance Comparison', y=1.02, fontsize=16)
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for plotting")
    
    @staticmethod
    def plot_detection_timeline(bakeoff: ModelBakeoff, model_name: str):
        """绘制特定模型的检测时间线"""
        try:
            import matplotlib.pyplot as plt
            
            if model_name not in bakeoff.results:
                print(f"Model {model_name} not found in results")
                return
            
            result = bakeoff.results[model_name]
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
            
            # 上图：数据和检测结果
            test_data = bakeoff.test_data
            if 'mean_motion' in test_data.columns:
                ax1.plot(test_data.index, test_data['mean_motion'], 'b-', 
                        linewidth=1, alpha=0.7, label='Mean Motion')
            
            # 标记真实机动
            for event_time in bakeoff.ground_truth_events:
                ax1.axvline(x=event_time, color='green', alpha=0.8, 
                           linestyle='--', linewidth=2)
            
            # 标记检测到的异常
            for det_time in result.detection_timestamps:
                ax1.axvline(x=det_time, color='red', alpha=0.6, 
                           linestyle='-', linewidth=1)
            
            ax1.set_title(f'{model_name} - Detection Timeline')
            ax1.set_ylabel('Mean Motion')
            ax1.legend(['Data', 'True Maneuvers', 'Detections'])
            ax1.grid(True, alpha=0.3)
            
            # 下图：异常分数
            if len(result.anomaly_scores) > 0:
                score_times = test_data.index[:len(result.anomaly_scores)]
                ax2.plot(score_times, result.anomaly_scores, 'k-', 
                        linewidth=1, alpha=0.7, label='Anomaly Score')
                ax2.set_ylabel('Anomaly Score')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for plotting")


# 使用示例函数
def run_example_bakeoff():
    """运行示例竞赛"""
    print("🧪 Running Example Model Bakeoff...")
    
    # 生成模拟数据
    dates = pd.date_range('2020-01-01', '2020-12-31', freq='1D')
    n_samples = len(dates)
    
    # 正常数据 + 机动事件
    data = pd.DataFrame({
        'mean_motion': np.random.normal(1.0027, 0.00005, n_samples),
        'eccentricity': np.random.normal(0.001, 0.0002, n_samples),
        'inclination': np.random.normal(0.1, 0.01, n_samples)
    }, index=dates)
    
    # 添加机动
    maneuver_indices = [50, 150, 250, 300]
    labels = pd.Series(0, index=dates)
    for idx in maneuver_indices:
        if idx < len(data):
            data.iloc[idx]['mean_motion'] += np.random.normal(0, 0.001)
            labels.iloc[idx] = 1
    
    # 分割数据
    split_idx = len(data) // 2
    train_data, test_data = data.iloc[:split_idx], data.iloc[split_idx:]
    train_labels, test_labels = labels.iloc[:split_idx], labels.iloc[split_idx:]
    
    # 创建竞赛框架
    bakeoff = ModelBakeoff()
    
    # 注册模型（这里用简化版本进行演示）
    # 实际使用时应该注册真实的检测器
    
    # 设置数据
    bakeoff.setup_data(train_data, test_data, train_labels, test_labels)
    
    print("Example bakeoff setup completed. Ready for real models!")
    
    return bakeoff

if __name__ == "__main__":
    run_example_bakeoff()