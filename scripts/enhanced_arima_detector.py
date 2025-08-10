#!/usr/bin/env python
"""
Enhanced ARIMA-based Satellite Maneuver Detector
Based on technical documentation and improved implementation
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

# Add project path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 这些依赖只在 main 中使用；如果你的工程没有它们，也不影响本 detector 的独立使用
try:
    from src.data.loader import SatelliteDataLoader
    from src.utils.metrics import evaluate_detection
except Exception:
    SatelliteDataLoader = None
    evaluate_detection = None


class EnhancedARIMADetector:
    """
    增强版 ARIMA 机动检测器：
    - 拟合失败自动回退（css-mle → css）
    - 动态更新：按间隔对最近短窗重拟合（失败忽略）
    - 自适应阈值：全局/本地波动融合 + 末段轻微降阈值 + 下界保护
    - 相邻异常聚类去重：相邻≤N天仅保留分数最高的一点
    """

    def __init__(
        self,
        p: int = 5,
        d: int = 1,
        q: int = 1,
        rolling_window: int = 30,
        adaptive_threshold: bool = True,
        threshold_factor: float = 2.5,
        min_threshold_factor: float = 1.5,
        max_threshold_factor: float = 4.0,
        min_gap_days: int = 3,
    ):
        self.p = int(p)
        self.d = int(d)
        self.q = int(q)
        self.rolling_window = int(rolling_window)
        self.adaptive_threshold = bool(adaptive_threshold)
        self.threshold_factor = float(threshold_factor)
        self.min_threshold_factor = float(min_threshold_factor)
        self.max_threshold_factor = float(max_threshold_factor)
        self.min_gap_days = int(min_gap_days)

        self.model_fit = None
        self.global_std: float = 0.0
        self.baseline_residuals: pd.Series | None = None

    # ------------------ 拟合 ------------------
    def fit(self, train_data: pd.Series):
        """在训练集上拟合 ARIMA，并记录基线残差与全局波动。"""
        y = self._ensure_series(train_data).dropna()
        print(f"Fitting ARIMA({self.p},{self.d},{self.q}) on {len(y)} training points...")

        # 平稳性报告（不强制）
        try:
            adf_stat, adf_p, *_ = adfuller(y)
            print(f"  ADF Statistic: {adf_stat:.4f}, p-value: {adf_p:.4f}")
        except Exception as e:
            print(f"  ADF check skipped: {e}")

        # 主拟合：css-mle，失败回退 css
        self.model_fit = self._fit_with_fallback(y)

        # 基线绝对残差与全局 std
        resid = np.abs(pd.Series(self.model_fit.resid, index=y.index[-len(self.model_fit.resid):]))
        self.baseline_residuals = resid.dropna()
        self.global_std = float(np.nanstd(self.baseline_residuals)) if len(self.baseline_residuals) else 0.0

        print(f"  Model AIC: {self.model_fit.aic:.2f}")
        print(f"  Baseline residual std: {self.global_std:.6f}")
        return self

    def _fit_with_fallback(self, series: pd.Series):
        """优先 css-mle，失败回退 css。"""
        for method, maxiter in (("css-mle", 200), ("css", 100)):
            try:
                model = ARIMA(series, order=(self.p, self.d, self.q))
                fit = model.fit(method=method, maxiter=maxiter)
                return fit
            except Exception:
                continue
        raise RuntimeError("ARIMA fit failed with both 'css-mle' and 'css'.")

    # ------------------ 检测 ------------------
    def detect(self, test_data: pd.Series, dynamic_update: bool = True) -> pd.DataFrame:
        """
        对测试集逐点检测。返回 DataFrame（索引为时间），包含列：
        ['actual','forecast','residual','threshold','score','is_anomaly']
        """
        if self.model_fit is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        y = self._ensure_series(test_data)
        print(f"\nDetecting anomalies on {len(y)} test points...")

        rw = self.rolling_window
        update_interval = max(10, rw // 3)
        last_update_i = 0

        results = {
            "actual": [],
            "forecast": [],
            "residual": [],
            "threshold": [],
            "score": [],
            "is_anomaly": [],
        }

        # 逐点处理
        for i, (t, val) in enumerate(y.items()):
            try:
                # 动态更新：每隔 update_interval，用最近 2*rw 窗口重拟合
                if dynamic_update and i > 0 and (i - last_update_i) >= update_interval:
                    win_start = max(0, i - 2 * rw)
                    recent = y.iloc[win_start:i].dropna()
                    if len(recent) > (self.p + self.d + self.q + 2):
                        try:
                            self.model_fit = self._fit_with_fallback(recent)
                            last_update_i = i
                        except Exception:
                            pass  # 忽略，继续用旧模型

                # 一步预测
                try:
                    yhat = float(self.model_fit.forecast(steps=1)[0])
                except Exception:
                    # 极端情况下，用上一实值回退
                    yhat = float(results["actual"][-1]) if results["actual"] else float(val)

                resid = abs(float(val) - yhat)

                # 自适应阈值（融合全局/本地波动）
                recent_resid = results["residual"][-rw:] if results["residual"] else [resid]
                th = self._calculate_adaptive_threshold(recent_resid, i, len(y))

                score = resid / max(th, 1e-12)
                is_anom = score > 1.0

                # 记录
                results["actual"].append(float(val))
                results["forecast"].append(yhat)
                results["residual"].append(resid)
                results["threshold"].append(float(th))
                results["score"].append(float(score))
                results["is_anomaly"].append(1 if is_anom else 0)

            except Exception as e:
                # 兜底：不中断流程
                prev_fore = results["forecast"][-1] if results["forecast"] else float(val)
                results["actual"].append(float(val))
                results["forecast"].append(float(prev_fore))
                results["residual"].append(0.0)
                results["threshold"].append(float(max(self.global_std, 1e-12) * max(self.threshold_factor, 1.0)))
                results["score"].append(0.0)
                results["is_anomaly"].append(0)
                print(f"  [warn] detect step {i} fallback due to: {e}")

        df = pd.DataFrame(results, index=y.index)

        # 相邻异常聚类去重（≤min_gap_days, 取分数最高）
        df = self._cluster_anomalies(df, min_gap_days=self.min_gap_days)

        print(f"Detection complete. Found {int(df['is_anomaly'].sum())} anomalies.")
        return df

    # ------------------ 阈值与聚类 ------------------
    def _calculate_adaptive_threshold(self, recent_residuals: list[float], position: int, total_len: int) -> float:
        """融合全局/本地波动并加下界保护。"""
        # 基线：全局 std × 因子
        factor = float(np.clip(self.threshold_factor, self.min_threshold_factor, self.max_threshold_factor))
        base = max(self.global_std, 1e-12) * factor

        if not self.adaptive_threshold or len(recent_residuals) < 5:
            th = base
        else:
            local_std = float(np.nanstd(recent_residuals))
            local_mean = float(np.nanmean(recent_residuals))
            vol_ratio = local_std / max(self.global_std, 1e-12)

            # 根据波动比微调因子（抑制过低阈值 / 降低误报）
            if vol_ratio > 1.5:
                adj = min(self.max_threshold_factor, factor * (1 + 0.3 * (vol_ratio - 1)))
            elif vol_ratio < 0.5:
                adj = max(self.min_threshold_factor, factor * (0.7 + 0.6 * vol_ratio))
            else:
                adj = factor

            # 融合（偏向全局，兼顾本地）
            th = 0.7 * base + 0.3 * (local_mean + adj * local_std)

        # 末段更敏感一点
        if position >= int(0.8 * total_len):
            th *= 0.95

        # 阈值下界保护：至少为全局 std 的一半（同量纲）
        return float(max(th, 0.5 * max(self.global_std, 1e-12)))

    def _cluster_anomalies(self, df: pd.DataFrame, min_gap_days: int = 3) -> pd.DataFrame:
        """将相邻≤min_gap_days的异常合并为1个：只保留簇内分数最高点。"""
        anom = df.index[df["is_anomaly"] == 1]
        if len(anom) == 0:
            return df.copy()

        keep_mask = pd.Series(0, index=df.index, dtype=int)

        # 逐簇扫描
        cluster = [anom[0]]
        for i in range(1, len(anom)):
            if (anom[i] - anom[i - 1]) <= timedelta(days=min_gap_days):
                cluster.append(anom[i])
            else:
                # 结束上一簇
                best_idx = df.loc[cluster, "score"].idxmax()
                keep_mask.loc[best_idx] = 1
                cluster = [anom[i]]

        # 最后一簇
        if cluster:
            best_idx = df.loc[cluster, "score"].idxmax()
            keep_mask.loc[best_idx] = 1

        out = df.copy()
        out["is_anomaly"] = keep_mask
        return out

    # ------------------ 小工具 ------------------
    @staticmethod
    def _ensure_series(series: pd.Series) -> pd.Series:
        if not isinstance(series, pd.Series):
            raise TypeError("Input must be a pandas Series.")
        if not isinstance(series.index, (pd.DatetimeIndex, pd.PeriodIndex)):
            try:
                series.index = pd.to_datetime(series.index)
            except Exception:
                raise ValueError("Series index must be datetime-like.")
        return series.sort_index().astype(float)

    # 便捷方法
    def set_threshold_factor(self, factor: float):
        self.threshold_factor = float(factor)


# ------------------ 阈值寻优（窗口统一 ±2 天） ------------------
def auto_tune_arima_order(train_data, max_p=7, max_d=2, max_q=3):
    """简易 AIC 网格搜索。"""
    print("\n🔧 Auto-tuning ARIMA parameters...")
    best_aic = np.inf
    best_order = (1, 1, 1)
    tested = 0
    for p in range(1, max_p + 1):
        for d in range(0, max_d + 1):
            for q in range(0, max_q + 1):
                if p + d + q > 10:
                    continue
                try:
                    fit = ARIMA(train_data, order=(p, d, q)).fit(method="css", maxiter=50)
                    if fit.aic < best_aic:
                        best_aic, best_order = fit.aic, (p, d, q)
                    tested += 1
                    if tested % 10 == 0:
                        print(f"  Tested {tested} models... Current best: {best_order} (AIC: {best_aic:.2f})")
                except Exception:
                    continue
    print(f"✅ Best ARIMA order: {best_order} (AIC: {best_aic:.2f})")
    return best_order


def find_optimal_threshold_factor(detector, val_data, val_labels, factor_range=(1.5, 4.0), n_trials=20):
    """
    在验证集上寻优阈值因子。窗口统一为 ±2 天。
    """
    print("\n🎯 Optimizing threshold factor on validation set (±2 days)...")
    factors = np.linspace(factor_range[0], factor_range[1], n_trials)
    best_f1, best_factor = 0.0, float(detector.threshold_factor)

    for f in factors:
        detector.set_threshold_factor(f)
        df = detector.detect(val_data, dynamic_update=False)
        pred_times = df.index[df["is_anomaly"] == 1].tolist()
        true_times = val_labels[val_labels == 1].index.tolist()

        if len(pred_times) == 0 and len(true_times) == 0:
            f1 = 1.0
        else:
            # 计数：±2 天匹配
            tp = 0
            used = set()
            for t in pred_times:
                for j, gt in enumerate(true_times):
                    if j in used:
                        continue
                    if abs((t - gt).days) <= 2:
                        tp += 1
                        used.add(j)
                        break
            fp = max(len(pred_times) - tp, 0)
            fn = max(len(true_times) - tp, 0)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        if f1 > best_f1:
            best_f1, best_factor = f1, float(f)

    print(f"✅ Best threshold factor: {best_factor:.2f} (F1: {best_f1:.3f})")
    detector.set_threshold_factor(best_factor)
    return best_factor, best_f1


# ------------------ 可视化 ------------------
def visualize_results(results_df, maneuver_times, satellite_name, element, output_dir):
    """
    Create comprehensive visualization of detection results
    """
    print("\n📊 Creating visualizations...")

    fig = plt.figure(figsize=(18, 12))

    # 1. 时序与检测
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(results_df.index, results_df["actual"], '-', label='Actual', linewidth=1.5, alpha=0.85)
    ax1.plot(results_df.index, results_df["forecast"], '--', label='ARIMA Forecast', linewidth=1.0, alpha=0.8)

    anomaly_points = results_df[results_df["is_anomaly"] == 1]
    ax1.scatter(anomaly_points.index, anomaly_points["actual"], s=80, marker='x', label='Detected', zorder=5)

    for m_time in maneuver_times:
        if results_df.index.min() <= m_time <= results_df.index.max():
            ax1.axvline(m_time, linestyle='--', alpha=0.45, linewidth=1.2)

    ax1.set_title(f'{satellite_name} - {element.replace("_", " ").title()} - ARIMA Detection', fontsize=14)
    ax1.set_ylabel('Value')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # 2. 残差与阈值
    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(results_df.index, results_df["residual"], '-', label='Residual', linewidth=1.0, alpha=0.8)
    ax2.plot(results_df.index, results_df["threshold"], '--', label='Adaptive Threshold', linewidth=1.2)
    ax2.fill_between(results_df.index, 0, results_df["threshold"], alpha=0.18)
    ax2.set_ylabel('Residual / Threshold')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    # 3. 分数
    ax3 = plt.subplot(3, 1, 3)
    ax3.plot(results_df.index, results_df["score"], label='Score', linewidth=1.0)
    ax3.axhline(y=1.0, linestyle='--', label='Decision', linewidth=1.2)
    for idx in anomaly_points.index:
        ax3.axvline(idx, alpha=0.25, linewidth=1.2)
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Score')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / f"{satellite_name}_{element}_enhanced_arima.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Visualization saved to {output_path}")


# ------------------ 可执行入口（可选使用；你的工程若不用可忽略） ------------------
def main():
    parser = argparse.ArgumentParser(description="Enhanced ARIMA Maneuver Detection")
    parser.add_argument("satellite", help="Satellite name (e.g., Fengyun-4A)")
    parser.add_argument("--element", default="mean_motion", help="Orbital element to analyze (default: mean_motion)")
    parser.add_argument("--auto-tune", action="store_true", help="Auto-tune ARIMA parameters")
    parser.add_argument("--optimize-threshold", action="store_true", help="Optimize threshold factor")
    parser.add_argument("--min-gap-days", type=int, default=3, help="Cluster nearby anomalies within N days")
    args = parser.parse_args()

    # 如果你的工程没有这些模块，main 可以不走；detector 不受影响
    if SatelliteDataLoader is None or evaluate_detection is None:
        print("This executable entry requires project modules (SatelliteDataLoader, evaluate_detection).")
        print("Detector class is ready for import/use.")
        return

    ELEMENT_TO_ANALYZE = args.element
    output_dir = Path("outputs") / f"{args.satellite.replace(' ', '_')}_Enhanced_ARIMA_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"🛰️ Enhanced ARIMA Detection for {args.satellite}")
    print(f"{'='*60}")

    loader = SatelliteDataLoader(data_dir="data")
    tle_data, maneuver_times = loader.load_satellite_data(args.satellite)

    if tle_data is None:
        print("❌ Failed to load data")
        return

    print(f"✅ Loaded {len(tle_data)} TLE records and {len(maneuver_times)} maneuvers")

    # 构造日尺度时序
    ts_data = tle_data[["epoch", ELEMENT_TO_ANALYZE]].set_index("epoch")
    ts_data = ts_data.resample("D").median().interpolate(method="time").dropna()
    ts_data = ts_data[ELEMENT_TO_ANALYZE]
    print(f"✅ Prepared time series with {len(ts_data)} daily points")

    # 标签（±2天窗口）
    labels = pd.Series(0, index=ts_data.index, dtype=int)
    for m_time in maneuver_times:
        mask = (ts_data.index >= m_time - timedelta(days=2)) & (ts_data.index <= m_time + timedelta(days=2))
        labels.loc[mask] = 1

    # 切分 6/2/2
    n = len(ts_data)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    train_data = ts_data.iloc[:train_end]
    val_data = ts_data.iloc[train_end:val_end]
    test_data = ts_data.iloc[val_end:]
    val_labels = labels.iloc[train_end:val_end]
    test_labels = labels.iloc[val_end:]

    # 自动寻优（可选）
    if args.auto_tune:
        p, d, q = auto_tune_arima_order(train_data)
    else:
        p, d, q = (5, 1, 1)
        print(f"\nUsing default ARIMA({p},{d},{q})")

    # 训练与阈值寻优
    detector = EnhancedARIMADetector(p=p, d=d, q=q, rolling_window=30, min_gap_days=args.min_gap_days)
    detector.fit(train_data)

    if args.optimize_threshold and len(val_data) > 0:
        best_factor, _ = find_optimal_threshold_factor(detector, val_data, val_labels)
        detector.set_threshold_factor(best_factor)

    # 测试集检测
    print(f"\n{'='*40}")
    print("📈 Running detection on test set...")
    print(f"{'='*40}")
    results_df = detector.detect(test_data, dynamic_update=True)

    # 评估（±2 天窗口）
    test_maneuvers = [m for m in maneuver_times if test_data.index.min() <= m <= test_data.index.max()]
    if evaluate_detection is not None and len(test_maneuvers) > 0:
        eval_metrics, matched_pairs = evaluate_detection(
            results_df.index[results_df["is_anomaly"] == 1].tolist(),
            test_maneuvers,
            matching_window=timedelta(days=2),
        )

        print(f"\n{'='*40}")
        print("📊 PERFORMANCE METRICS (±2 days)")
        print(f"{'='*40}")
        print(f"Precision:  {eval_metrics['precision']:.2%}")
        print(f"Recall:     {eval_metrics['recall']:.2%}")
        print(f"F1 Score:   {eval_metrics['f1']:.3f}")
        print(f"Detected:   {eval_metrics['tp']}/{len(test_maneuvers)} maneuvers")
        print(f"False Alarms: {eval_metrics['fp']}")

    # 可视化与保存
    visualize_results(results_df, test_maneuvers, args.satellite, ELEMENT_TO_ANALYZE, output_dir)
    results_df.to_csv(output_dir / "detection_results.csv")

    report = {
        "satellite_name": args.satellite,
        "element": ELEMENT_TO_ANALYZE,
        "arima_order": [p, d, q],
        "threshold_factor": detector.threshold_factor,
        "adaptive_threshold": detector.adaptive_threshold,
        "min_gap_days": detector.min_gap_days,
        "performance_window_days": 2,
        "detected_anomalies": [str(t) for t in results_df.index[results_df["is_anomaly"] == 1]],
        "test_maneuvers": [str(t) for t in test_maneuvers],
    }
    with open(output_dir / "detection_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n✅ Results saved to {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
