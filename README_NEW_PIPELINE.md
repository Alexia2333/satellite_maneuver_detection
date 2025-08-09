
# New Maneuver Detection Pipeline (XGBoost, standalone)

This is a clean, self-contained detector that does not import legacy modules.
It adds GEO-oriented drift/pressure features by default, supports automated
hyperparameter tuning, and clusters dense positives into single events.

## Files
- `src/models/maneuver_xgb_detector.py`: Unified detector class.
- `src/features/base_features.py`: Base features (lags, diffs, rolling stats).
- `src/features/drift_features.py`: Drift/pressure features for GEO.
- `src/utils/orbit_classification.py`: GEO/LEO classification helpers.
- `src/utils/metrics.py`: Threshold utilities and temporal clustering.
- `src/tuning/auto_tune.py`: Lightweight random search with early stopping.
- `scripts/run_maneuver_detection.py`: New CLI entry point.

## Usage
```bash
python scripts/run_maneuver_detection.py       --data data/FY4A.csv       --time-col timestamp       --label-col label       --orbit auto       --auto-tune on       --threshold quantile:0.98       --save-dir outputs/FY4A_baseline
```

Notes:
- The script expects a binary `label` column for supervised training.
- For GEO, drift features are enabled by default.
- You can switch thresholding to `pr_best` via `--threshold pr_best`.
