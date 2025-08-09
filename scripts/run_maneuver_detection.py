"""
Command-line entry point for the new maneuver detection pipeline.

This script does not import any legacy modules. It trains an XGBoost
classifier on labeled data and outputs detection results and artifacts.
"""
import argparse
import os
import json
from datetime import datetime
from typing import Optional

# Make repository root importable when running this file directly
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from src.models.maneuver_xgb_detector import ManeuverXGBDetector, DetectorConfig
from src.utils.orbit_classification import classify_orbit


def parse_args():
    p = argparse.ArgumentParser(description="Run maneuver detection with XGBoost.")
    p.add_argument("--data", required=True, help="Path to CSV file containing telemetry and labels.")
    p.add_argument("--time-col", default="timestamp", help="Timestamp column name.")
    p.add_argument("--label-col", default="label", help="Binary label column name (0/1).")
    p.add_argument("--orbit", default="auto", choices=["auto", "GEO", "LEO"], help="Orbit class or auto.")
    p.add_argument("--auto-tune", default="off", choices=["on", "off"], help="Enable automated hyperparameter tuning.")
    p.add_argument("--threshold", default="quantile:0.98", help="Threshold mode, e.g., 'quantile:0.98' or 'pr_best'.")
    p.add_argument("--save-dir", default=None, help="Directory to save outputs. Default uses outputs/<timestamp>.")
    p.add_argument("--temporal-window", type=int, default=3, help="Clustering window (samples).")
    return p.parse_args()


def parse_threshold(thr_arg: str):
    if thr_arg.startswith("quantile:"):
        q = float(thr_arg.split(":", 1)[1])
        return "quantile", q
    return "pr_best", None


def main():
    args = parse_args()
    df = pd.read_csv(args.data)

    if args.save_dir is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save_dir = os.path.join("outputs", f"run_{run_id}")
    os.makedirs(args.save_dir, exist_ok=True)

    thr_mode, q = parse_threshold(args.threshold)
    cfg = DetectorConfig(
        time_col=args.time_col,
        label_col=args.label_col,
        orbit=args.orbit,
        use_drift=None,  # decide by orbit
        threshold_mode="quantile" if thr_mode == "quantile" else "pr_best",
        threshold_quantile=q if q is not None else 0.98,
        temporal_window=args.temporal_window,
        auto_tune=(args.auto_tune == "on")
    )
    detector = ManeuverXGBDetector(cfg)
    detector.fit(df)

    det = detector.detect(df, timestamps=df[cfg.time_col])
    det.to_csv(os.path.join(args.save_dir, "detections.csv"), index=False)
    detector.save_run_artifacts(args.save_dir)
    print(f"Saved results to {args.save_dir}")


if __name__ == "__main__":
    main()
