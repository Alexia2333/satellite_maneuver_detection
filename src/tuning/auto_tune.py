"""
Lightweight automated hyperparameter tuning for XGBoost classifiers.

Uses random search over a bounded grid with early stopping on a validation split.
The objective is to maximize F1 (or AP) on the validation set.
"""
from typing import Dict, Tuple, Optional, List
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
import xgboost as xgb

# Fixed absolute import
from src.utils.metrics import find_best_threshold_youden, evaluate_at_threshold


def random_search_xgb(
    X, y,
    param_grid: Dict[str, List],
    n_iter: int = 30,
    test_size: float = 0.2,
    early_stopping_rounds: int = 50,
    random_state: int = 42,
    maximize: str = "f1"
) -> Tuple[dict, float, float]:
    """
    Random search over XGBClassifier hyperparameters.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    rng = np.random.default_rng(random_state)
    keys = list(param_grid.keys())
    best_params = None
    best_metric = -1.0
    best_thr = 0.5

    for _ in range(n_iter):
        params = {k: rng.choice(param_grid[k]) for k in keys}
        
        if 'eval_metric' not in params:
            params['eval_metric'] = 'aucpr'

        clf = xgb.XGBClassifier(
            tree_method="hist",
            enable_categorical=False,
            objective="binary:logistic",
            early_stopping_rounds=early_stopping_rounds, 
            **params
        )


        clf.fit(
            X_train, 
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        
        scores = clf.predict_proba(X_val)[:, 1]
        thr = find_best_threshold_youden(y_val, scores)

        if maximize == "ap":
            from sklearn.metrics import average_precision_score
            metric = average_precision_score(y_val, scores)
        else:
            metric = evaluate_at_threshold(y_val, scores, thr)["f1"]

        if metric > best_metric:
            best_metric = float(metric)
            best_params = dict(params)
            best_thr = float(thr)

    return best_params, best_metric, best_thr
