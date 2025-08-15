# Satellite Maneuver Detection Using Time Series Anomaly Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the implementation of advanced time series anomaly detection techniques for identifying satellite orbital maneuvers from Two-Line Element (TLE) data. This project was developed as part of a Master of Data Science Research Project at the University of Adelaide.

## 🎯 Project Overview

The project addresses the critical need for automated satellite maneuver detection in space situational awareness applications. Using publicly available TLE data from NORAD, I developed two complementary approaches:

1. **Enhanced ARIMA Detector**: Unsupervised approach with drift-aware dynamic thresholding
2. **XGBoost Classifier**: Supervised machine learning approach with comprehensive feature engineering
3. **Multi-element Fusion Strategy**: Novel weighted fusion combining multiple orbital elements

### Key Satellites Analyzed
- **Jason Series** (LEO): Jason-1, Jason-2, Jason-3
- **Fengyun Series** (GEO): Fengyun-2D, Fengyun-2E, Fengyun-2F, Fengyun-2H, Fengyun-4A

## 📊 Key Results

- **XGBoost Model**: Average F1 scores of 0.612 (LEO) and 0.861 (GEO)
- **Performance Improvement**: 121% improvement over ARIMA for LEO satellites
- **Fusion Strategy**: Consistently outperforms single-element detection
- **Best Orbital Element**: Mean motion proves most effective for maneuver detection

## 📁 Repository Structure

```
satellite_maneuver_detection/
├── configs/                    # Configuration files
│   ├── Fengyun-4A.json
│   └── enhanced_config_example.json
├── data/                       # Data directory
│   ├── orbital_elements/       # TLE data in CSV format
│   ├── manoeuvres/            # Ground truth maneuver logs
│   └── processed/             # Processed datasets
├── src/                       # Source code modules
│   ├── data/                  # Data loading and preprocessing
│   ├── models/                # Model implementations
│   ├── features/              # Feature engineering
│   ├── utils/                 # Utility functions
│   └── tuning/                # Hyperparameter optimization
├── scripts/                   # Executable scripts
│   ├── run_maneuver_detection.py     # Main detection pipeline
│   ├── run_fusion_detector.py        # Multi-element fusion
│   ├── run_arima_detector.py         # ARIMA-based detection
│   └── final_xgb_detector.py         # XGBoost standalone detector
├── notebooks/                 # Jupyter notebooks for analysis
├── outputs/                   # Results and visualizations
└── results/                   # Generated reports and logs
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Git

### Setup
```bash
# Clone the repository
git clone https://github.com/Alexia2333/satellite_maneuver_detection.git
cd satellite_maneuver_detection

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
```
numpy
pandas
scikit-learn
xgboost
matplotlib
statsmodels
```

## 🚀 Quick Start

### 1. Basic XGBoost Detection
```bash
python scripts/run_maneuver_detection.py \
    --data data/orbital_elements/Fengyun-4A.csv \
    --time-col timestamp \
    --label-col label \
    --orbit auto \
    --auto-tune on \
    --threshold quantile:0.98 \
    --save-dir outputs/FY4A_baseline
```

### 2. Multi-element Fusion Detection
```bash
python scripts/run_fusion_detector.py \
    --satellite Fengyun-4A \
    --config configs/Fengyun-4A.json \
    --output-dir outputs/fusion_results
```

### 3. ARIMA-based Detection
```bash
python scripts/run_arima_detector.py \
    --satellite Jason-3 \
    --elements mean_motion,eccentricity,inclination \
    --auto-tune \
    --output-dir outputs/arima_results
```

## 📈 Key Features

### Enhanced ARIMA Detector
- **Drift-aware Thresholding**: Adapts to long-term orbital evolution
- **Dynamic Modeling**: Sliding window approach for time-varying parameters
- **Multi-element Support**: Analyzes mean motion, eccentricity, and inclination

### XGBoost Classifier
- **Advanced Feature Engineering**: Rolling statistics, lag features, and difference features
- **Detection Strategies**: "Top-N" and "peak detection" methods
- **Automated Hyperparameter Tuning**: Optuna-based optimization
- **Orbit-specific Features**: GEO-oriented drift and pressure features

### Multi-element Fusion
- **Weighted Score Combination**: Optimal weighting of individual element scores
- **Event Clustering**: Groups temporally close detections
- **Residual Gap Analysis**: Comprehensive post-detection analysis

## 📝 Data Format

### Input TLE Data (CSV)
```csv
timestamp,mean_motion,eccentricity,inclination,raan,arg_perigee,mean_anomaly
2018-01-01,1.00271234,0.0001234,98.1234,123.4567,234.5678,345.6789
```

### Ground Truth Maneuvers
```csv
date,satellite,maneuver_type
2018-06-15,Fengyun-4A,station_keeping
```

## 🔧 Configuration

The system uses JSON configuration files for different satellites and detection parameters:

```json
{
    "satellite_name": "Fengyun-4A",
    "orbit_type": "GEO",
    "elements": ["mean_motion", "eccentricity", "inclination"],
    "arima_params": {
        "p_range": [0, 3],
        "d_range": [0, 2],
        "q_range": [0, 3]
    },
    "xgboost_params": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1
    }
}
```

## 📊 Evaluation Metrics

The project emphasizes **Precision-Recall (PR) curve analysis** over single-point metrics:

- **Precision**: Fraction of detected events that are true maneuvers
- **Recall**: Fraction of true maneuvers that are detected
- **F1 Score**: Harmonic mean of precision and recall
- **AUPRC**: Area Under Precision-Recall Curve
- **Event-based Matching**: ±1 day tolerance for ground truth matching

## 🔬 Methodology

### 1. Data Preprocessing
- Outlier detection and removal
- Linear interpolation for missing values
- First-order differencing for stationarity
- Standardization/normalization

### 2. Model Architecture
- **ARIMA**: Auto-regressive Integrated Moving Average with dynamic thresholds
- **XGBoost**: Gradient boosting with extensive feature engineering
- **Fusion**: Weighted combination of multiple orbital elements

### 3. Computational Optimizations
- Guided ARIMA parameter search using ADF tests
- Two-phase threshold optimization (coarse + fine)
- Early stopping in hyperparameter tuning

## 📈 Performance Comparison

| Model | LEO F1 | GEO F1 | Improvement |
|-------|--------|--------|-------------|
| ARIMA Fusion | 0.277 | 0.527 | Baseline |
| XGBoost | 0.612 | 0.861 | +121% / +63% |

## 🎓 Academic Context

This project was completed as part of:
- **Program**: Master of Data Science Research Project
- **Institution**: University of Adelaide, School of Mathematical Sciences
- **Supervisor**: Dr. David Shorten
- **Focus Area**: Space Situational Awareness & Time Series Analysis

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@mastersthesis{lin2025satellite,
  title={Advanced Detection Techniques for Satellite Orbital Maneuvers Using Time Series Anomaly Detection},
  author={Lin, Yichao},
  year={2025},
  school={University of Adelaide},
  type={Master of Data Science Research Project}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

This is a completed academic project, but feedback and suggestions are welcome:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## 📞 Contact

**Yichao Lin**
- Student ID: 1908590
- Institution: University of Adelaide
- Email: [Contact through GitHub]

## 🙏 Acknowledgments

- Dr. David Shorten for project supervision and guidance
- University of Adelaide School of Mathematical Sciences
- NORAD for providing publicly available TLE data
- The space situational awareness research community

## 📋 Requirements

For detailed dependencies, see `requirements.txt`. Key packages:
- `numpy`, `pandas`: Data manipulation
- `scikit-learn`: Machine learning utilities
- `xgboost`: Gradient boosting implementation
- `statsmodels`: ARIMA modeling
- `matplotlib`: Visualization

---

**Note**: This repository contains the complete implementation for reproducible research. All scripts are documented and configuration-driven to facilitate independent verification of results.