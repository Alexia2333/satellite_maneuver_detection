# src/tuning/tune_arima_threshold.py
import copy
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm  # <--- 1. 导入tqdm库
from datetime import timedelta

from src.models.arima_detector import EnhancedARIMADetector

def _greedy_event_match(pred_times: List[pd.Timestamp], true_times: List[pd.Timestamp], window: pd.Timedelta) -> Tuple[int, int, int]:
    """事件匹配的贪心算法，返回 (TP, FP, FN)。"""
    if not pred_times or not true_times:
        return 0, len(pred_times), len(true_times)
    
    pred_sorted, truth_sorted = sorted(list(set(pred_times))), sorted(list(set(true_times)))
    used_truth_indices, tp = set(), 0

    for p_time in pred_sorted:
        for i, t_time in enumerate(truth_sorted):
            if i not in used_truth_indices and abs(p_time - t_time) <= window:
                tp += 1
                used_truth_indices.add(i)
                break
    
    fp = len(pred_sorted) - tp
    fn = len(truth_sorted) - len(used_truth_indices)
    return tp, fp, fn

def find_factor_for_target_recall(
    detector_instance: EnhancedARIMADetector,
    val_data: pd.Series,
    true_maneuver_times: List[pd.Timestamp],
    target_recall: float = 0.5,
    n_trials: int = 50
) -> float:
    """
    寻找能满足目标召回率的最高阈值因子，并显示进度条。
    """
    print(f"\n🎯 Optimizing threshold for TARGET RECALL >= {target_recall:.0%}...")
    
    factors_to_test = np.linspace(0.2, 5.0, n_trials)
    valid_factors = []
    
    # --- 2. 在循环中加入tqdm进度条 ---
    # leave=False 表示进度条完成后会消失，保持日志整洁
    for factor in tqdm(factors_to_test, desc="Optimizing Threshold for Recall", leave=False):
        temp_detector = copy.deepcopy(detector_instance)
        temp_detector.cfg.threshold_factor = factor

        results_df = temp_detector.detect(val_data)
        predicted_times = results_df[results_df['is_anomaly']].index.tolist()
        
        tp, fp, fn = _greedy_event_match(predicted_times, true_maneuver_times, timedelta(days=2))
        
        current_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if current_recall >= target_recall:
            valid_factors.append(factor)
            
    if not valid_factors:
        print(f"  [Warning] No threshold factor achieved target recall. Falling back to the most sensitive setting.")
        return factors_to_test[0]
    else:
        best_factor = max(valid_factors)
        print(f"  ✅ Best factor to achieve recall >= {target_recall:.0%} is {best_factor:.2f}")
        return best_factor

# 同时为旧的F1优化器也加上进度条
def find_best_threshold_factor(detector_instance, val_data, true_maneuver_times, n_trials=25):
    """在验证集上优化阈值因子以最大化F1分数。"""
    print(f"\n🎯 Optimizing threshold factor for MAX F1...")
    factors_to_test = np.linspace(0.5, 4.0, n_trials)
    best_f1, best_factor = -1.0, detector_instance.cfg.threshold_factor
    
    for factor in tqdm(factors_to_test, desc="Optimizing Threshold for F1", leave=False):
        temp_detector = copy.deepcopy(detector_instance)
        temp_detector.cfg.threshold_factor = factor
        results_df = temp_detector.detect(val_data)
        predicted_times = results_df[results_df['is_anomaly']].index.tolist()
        tp, fp, fn = _greedy_event_match(predicted_times, true_maneuver_times, timedelta(days=2))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_factor = f1, factor
    
    print(f"✅ Best threshold factor for F1: {best_factor:.2f} (F1: {best_f1:.3f})")
    return best_factor