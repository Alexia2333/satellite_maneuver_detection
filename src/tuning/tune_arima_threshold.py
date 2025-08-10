import numpy as np
import pandas as pd
from typing import Tuple
from ..utils.metrics import evaluate_detection # 使用相对导入

def find_best_threshold_factor(scores: pd.Series, 
                                 true_labels: pd.Series, 
                                 threshold_base: pd.Series) -> Tuple[float, float]:
    """
    通过遍历一系列阈值因子来寻找最优值，以最大化F1分数。

    Args:
        scores (pd.Series): 模型的异常分数（残差）。
        true_labels (pd.Series): 对应的真实标签 (0或1)。
        threshold_base (pd.Series): 用于乘以因子的动态阈值基准。

    Returns:
        A tuple containing:
        - best_factor (float): 产生最高F1分数的阈值因子。
        - best_f1 (float): 对应的最高F1分数。
    """
    print("Finding best threshold factor by optimizing F1 score...")
    
    best_f1 = -1.0
    best_factor = 3.5 # 默认值
    
    # 尝试从1.5到6.0的一系列阈值因子
    factors_to_try = np.arange(1.5, 6.0, 0.1)
    
    for factor in factors_to_try:
        # 计算当前因子下的动态阈值
        current_threshold = factor * threshold_base
        
        # 根据阈值生成预测
        predictions = (scores > current_threshold).astype(int)
        
        # 提取有标签的事件进行评估
        detected_epochs = scores.index[predictions == 1].tolist()
        true_maneuvers = true_labels.index[true_labels == 1].tolist()
        
        # 计算性能
        metrics, _ = evaluate_detection(detected_epochs, true_maneuvers)
        f1 = metrics['f1']
        
        if f1 > best_f1:
            best_f1 = f1
            best_factor = factor
            
    print(f"Best threshold factor found: {best_factor:.2f} (achieved F1 score: {best_f1:.3f})")
    return best_factor, best_f1