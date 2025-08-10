import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import warnings
from typing import Tuple
import itertools
from tqdm import tqdm 

def find_best_arima_order(history: pd.Series, p_range: range = range(0, 6), d_range: range = range(0, 3), q_range: range = range(0, 4)) -> Tuple[int, int, int]:
    """
    通过网格搜索寻找最优的ARIMA(p,d,q)阶数，并显示进度条。

    Args:
        history (pd.Series): 用于评估的历史时间序列数据。
        p_range, d_range, q_range: p, d, q各自的搜索范围。

    Returns:
        A tuple containing the best (p,d,q) order.
    """
    best_aic = float("inf")
    best_order = None
    
    # 【新增】生成所有 (p,d,q) 的组合
    all_orders = list(itertools.product(p_range, d_range, q_range))
    
    print(f"Starting grid search for best ARIMA order across {len(all_orders)} combinations...")
    
    # 忽略所有在模型拟合过程中可能出现的警告
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        
        # 【核心修改】使用tqdm包装组合列表，以创建进度条
        for order in tqdm(all_orders, desc="Searching ARIMA Orders"):
            try:
                model = ARIMA(history, order=order)
                model_fit = model.fit()
                
                if model_fit.aic < best_aic:
                    best_aic = model_fit.aic
                    best_order = order
                    
            except Exception:
                # 某些(p,d,q)组合可能导致模型无法拟合，直接跳过
                continue
                        
    print(f"\nGrid search complete. Best order found: {best_order} with AIC: {best_aic:.2f}")
    return best_order