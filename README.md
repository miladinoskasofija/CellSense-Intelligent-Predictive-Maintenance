# CellSense — Intelligent Machine Failure Detection & Predictive Maintenance

**An end-to-end Data Science and Machine Learning research project for industrial condition monitoring, machine failure classification, explainable AI, and anomaly detection.**

CellSense combines statistical analysis, physics-informed feature engineering, supervised machine learning, automated hyperparameter optimization, SHAP explainability, and Isolation Forest anomaly detection to investigate machine operating conditions.

The project is developed in Python using the **AI4I 2020 Predictive Maintenance Dataset** from the UCI Machine Learning Repository.

> **Project scope:** CellSense is a research prototype evaluated on synthetic industrial data. It performs observation-level machine failure classification, not time-to-failure forecasting or Remaining Useful Life (RUL) estimation.

---

## 1. Project Overview

Unexpected machine failures can lead to production interruptions, maintenance costs, and reduced equipment reliability.

Machine Learning offers an opportunity to analyze operational sensor measurements and identify patterns associated with equipment failures.

**CellSense** investigates how data-driven models and engineering knowledge can be combined to support industrial condition monitoring.

### Project Objectives

- Explore and analyze industrial operating measurements.
- Investigate statistical relationships between sensor readings and machine failures.
- Develop physics-informed features based on mechanical principles.
- Train and compare multiple classification algorithms.
- Address class imbalance using appropriate evaluation methods.
- Optimize model hyperparameters and decision thresholds.
- Explain model predictions using SHAP.
- Detect unusual operating conditions using Isolation Forest.
- Evaluate model generalization using an independent test dataset.
- Export trained models and experimental results for reproducibility.

---

## 2. Dataset

**Dataset:** AI4I 2020 Predictive Maintenance Dataset  
**Source:** UCI Machine Learning Repository  
**Dataset ID:** 601  
**Observations:** 10,000  
**Target:** Machine failure (binary classification)

**Dataset link:** https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

### Input Features

| Feature | Description |
|---|---|
| Type | Product quality category (L, M, H) |
| Air temperature | Ambient air temperature (K) |
| Process temperature | Process temperature (K) |
| Rotational speed | Rotational speed (rpm) |
| Torque | Applied torque (Nm) |
| Tool wear | Accumulated tool wear (min) |

### Target Variable

- **0 — No Failure:** Normal machine operation.
- **1 — Failure:** Machine failure occurrence.

The dataset also provides indicators for five failure mechanisms:

- TWF — Tool Wear Failure
- HDF — Heat Dissipation Failure
- PWF — Power Failure
- OSF — Overstrain Failure
- RNF — Random Failure

These failure indicators are used for exploratory analysis and post-training evaluation, **not as predictive input features**, to avoid target leakage.

### Class Imbalance

Machine failures represent approximately 3.4% of the dataset.

Because of this imbalance, accuracy alone is insufficient to evaluate model performance.

The project emphasizes **Average Precision, Precision, Recall, F1-score, F2-score, ROC-AUC, and confusion matrices**.

---

## 3. Technologies Used

**Programming & Data Analysis**
- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn

**Statistics**
- SciPy
- Statsmodels

**Machine Learning**
- Scikit-learn
- XGBoost
- LightGBM

**Optimization & Explainability**
- Optuna
- SHAP

**Anomaly Detection & Dimensionality Reduction**
- Isolation Forest
- Principal Component Analysis (PCA)

**Development Environment**
- Google Colab
- Jupyter Notebook

**Model Serialization**
- Joblib
- JSON
- CSV

---

## 4. Project Methodology

The project follows a structured experimental workflow:

1. **Data Quality Assessment** — Missing values, duplicates, data types, and descriptive statistics.
2. **Exploratory Data Analysis** — Sensor distributions, failure patterns, correlations, and failure mechanisms.
3. **Statistical Analysis** — Hypothesis testing, effect sizes, and multiple-testing correction.
4. **Feature Engineering** — Physics-informed features derived from operational measurements.
5. **Machine Learning** — Training and cross-validation of classification algorithms.
6. **Feature Ablation Study** — Comparison of original and engineered feature configurations.
7. **Hyperparameter Optimization** — Optuna-based model optimization.
8. **Threshold Optimization** — Selection of a decision threshold using training out-of-fold predictions.
9. **Model Evaluation** — Independent test evaluation, error analysis, and bootstrap confidence intervals.
10. **Explainable AI** — Global and local SHAP explanations.
11. **Anomaly Detection** — Isolation Forest and PCA-based analysis.
12. **Model Export** — Serialization, configuration management, and inference validation.

---

## 5. Exploratory Data Analysis & Statistical Testing

The exploratory analysis investigates:

- Machine failure distribution.
- Sensor measurements and their distributions.
- Potential outliers.
- Failure rates across product categories.
- Relationships between sensor readings and failure outcomes.
- Pearson and Spearman correlations.
- Individual failure mechanisms and their co-occurrence.

### Statistical Methods

The project includes:

| Method | Purpose |
|---|---|
| Shapiro-Wilk Test | Assess distribution normality |
| Mann-Whitney U Test | Compare sensor measurements between failure groups |
| Rank-biserial Correlation | Estimate nonparametric effect size |
| Chi-Square Test | Investigate categorical association |
| Cramer's V | Measure categorical association strength |
| Benjamini-Hochberg Correction | Control false discovery rate |

Statistical tests are interpreted as exploratory evidence of association rather than proof of causation.

---

## 6. Physics-Informed Feature Engineering

One of the main components of CellSense is the integration of mechanical relationships into the Machine Learning feature space.

Six additional features were investigated.

| Engineered Feature | Formula / Description |
|---|---|
| Angular Velocity | ω = 2π × RPM / 60 |
| Mechanical Power | P = Torque × Angular Velocity |
| Temperature Difference | Process Temperature − Air Temperature |
| Temperature Ratio | Process Temperature / Air Temperature |
| Tool Wear Load | Tool Wear × Torque |
| Torque Speed Ratio | Torque / Rotational Speed |

These features are intended to capture mechanical load, thermal behavior, and operating conditions that may not be fully represented by individual sensor readings.

Angular Velocity was excluded from the final engineered feature set because it is a deterministic linear transformation of rotational speed.

### Feature Ablation Study

Models were evaluated using two configurations:

- **Original Features:** Original sensor measurements and product type.
- **Physics-Informed Features:** Original features plus selected engineered features.

**Cross-validation Average Precision:**

| Model | Original Features | Physics-Informed Features |
|---|---:|---:|
| Logistic Regression | 0.4409 | 0.4449 |
| Random Forest | 0.7511 | 0.8853 |
| XGBoost | 0.7748 | 0.8883 |
| LightGBM | 0.7802 | 0.8846 |

Physics-informed feature engineering improved Average Precision across all four main classification algorithms.

For XGBoost, the improvement was approximately **14.65% relative to the original feature configuration**.

---

## 7. Machine Learning Models

Five algorithms were evaluated:

1. Dummy Classifier — Baseline
2. Logistic Regression
3. Random Forest
4. XGBoost
5. LightGBM

### Experimental Setup

- Stratified 80/20 train-test split.
- 8,000 training observations.
- 2,000 independent test observations.
- Stratified 5-fold cross-validation.
- Training-only preprocessing.
- Class imbalance handling.
- Original vs. engineered feature comparison.
- Average Precision as the primary model selection metric.

### Model Selection

**XGBoost with physics-informed features** achieved the highest mean cross-validation Average Precision among the evaluated configurations.

The model was subsequently optimized using Optuna.

---

## 8. Hyperparameter & Threshold Optimization

### Optuna Optimization

Hyperparameter optimization was conducted using:

- Tree-structured Parzen Estimator (TPE).
- 30 optimization trials.
- Stratified 5-fold cross-validation.
- Average Precision as the optimization objective.

The optimized XGBoost configuration achieved a cross-validation Average Precision of approximately **0.8978**, compared with **0.8883** before hyperparameter optimization.

### Decision Threshold Optimization

A separate threshold optimization experiment was performed using out-of-fold training predictions.

The final selected decision threshold was:

**0.8366**

The threshold was selected using the F2-score, which places greater emphasis on recall than precision.

This step enables explicit investigation of the trade-off between detected failures and false maintenance alarms.

---

## 9. Final Model Performance

The final optimized XGBoost model was evaluated on the independent test dataset containing 2,000 observations.

### Test Results

| Metric | Result |
|---|---:|
| Accuracy | **99.30%** |
| Balanced Accuracy | 91.12% |
| Precision | **96.55%** |
| Recall | **82.35%** |
| Specificity | 99.90% |
| F1-score | **88.89%** |
| F2-score | 84.85% |
| ROC-AUC | **98.09%** |
| Average Precision | **90.89%** |
| Matthews Correlation Coefficient | 0.8883 |

### Confusion Matrix

| Actual / Predicted | No Failure | Failure |
|---|---:|---:|
| No Failure | 1,930 | 2 |
| Failure | 12 | 56 |

### Key Findings

- Correctly identified **56 of 68 actual failures**.
- Missed **12 machine failures**.
- Generated only **2 false failure alarms**.
- Achieved high precision while maintaining a recall of approximately 82%.

These results indicate strong classification performance on the held-out synthetic dataset. They do not establish equivalent performance on real industrial equipment.

### Bootstrap Confidence Intervals

A stratified bootstrap procedure with 1,000 resamples was used to estimate performance uncertainty.

| Metric | Test Estimate | 95% Confidence Interval |
|---|---:|---|
| ROC-AUC | 0.9809 | [0.9614, 0.9963] |
| Average Precision | 0.9089 | [0.8511, 0.9600] |
| Recall | 0.8235 | [0.7206, 0.9118] |
| Precision | 0.9655 | [0.9138, 1.0000] |
| F1-score | 0.8889 | [0.8226, 0.9394] |

The confidence intervals highlight the uncertainty associated with evaluating a relatively small number of failure observations.

---

## 10. Explainable AI with SHAP

SHAP (SHapley Additive exPlanations) was used to investigate model behavior.

The explainability analysis includes:

- Global feature importance.
- SHAP summary visualizations.
- Feature dependence analysis.
- Local prediction explanations.
- Feature group contribution analysis.

### Most Influential Features

| Feature | Relative SHAP Contribution |
|---|---:|
| Tool wear | 22.70% |
| Tool Wear Load | 19.89% |
| Rotational speed | 16.05% |
| Mechanical Power | 8.23% |
| Torque Speed Ratio | 6.86% |

### Feature Group Contributions

| Feature Group | Relative Contribution |
|---|---:|
| Original Sensor Features | 49.52% |
| Physics-Informed Features | 47.50% |
| Categorical Features | 2.98% |

Physics-informed features accounted for approximately **47.5% of the aggregated absolute SHAP contributions**.

SHAP explains model predictions and should not be interpreted as evidence of physical causality.

---

## 11. Anomaly Detection with Isolation Forest

Isolation Forest was implemented as a complementary novelty detection approach.

Unlike supervised classification, the anomaly detector was trained exclusively on normal operating observations.

### Methodology

- Train Isolation Forest using normal training observations.
- Apply independent preprocessing.
- Generate anomaly scores.
- Evaluate detected anomalies against test failure labels.
- Visualize operating conditions using PCA.
- Analyze anomaly detection across failure mechanisms.
- Compare results with the supervised XGBoost classifier.

### Anomaly Detection Results

| Metric | Isolation Forest |
|---|---:|
| Precision | 27.03% |
| Recall | 29.41% |
| F1-score | 28.17% |
| ROC-AUC | 85.61% |
| Average Precision | 22.86% |

Isolation Forest flagged 74 observations as anomalous.

It identified one failure missed by the supervised classifier, but also produced 54 additional alerts on non-failure observations.

### Interpretation

Isolation Forest demonstrated a different detection behavior but substantially lower failure classification performance than XGBoost.

The results illustrate the distinction between detecting statistical anomalies and predicting labeled machine failures.

---

## 12. Project Structure

```text
CellSense-Predictive-Maintenance/
│
├── CellSense_Advanced.ipynb
│
├── cellsense_artifacts/
│   │
│   ├── models/
│   │   ├── supervised_model.joblib
│   │   ├── isolation_forest.joblib
│   │   ├── anomaly_preprocessor.joblib
│   │   └── anomaly_pca.joblib
│   │
│   ├── config/
│   │   ├── model_metadata.json
│   │   └── environment.csv
│   │
│   ├── results/
│   │   └── *.csv
│   │
│   ├── feature_engineering.py
│   └── requirements.txt
│
└── README.md
```

The notebook is the main research artifact. The exported files contain trained models, preprocessing components, configuration metadata, and experiment results.

---

## 13. How to Run the Project

### Option 1: Google Colab

1. Open `CellSense_Advanced.ipynb` in Google Colab.
2. Connect to a Python runtime.
3. Execute the notebook cells sequentially, starting from the environment setup.
4. Allow the required dependencies to install.
5. Run the experiments and inspect the generated results.

The dataset is loaded programmatically from the UCI Machine Learning Repository.

```python
from ucimlrepo import fetch_ucirepo

dataset = fetch_ucirepo(id=601)
```

**Note:** Hyperparameter optimization and SHAP computations may take additional execution time.

### Option 2: Local Jupyter Environment

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/CellSense-Predictive-Maintenance.git
cd CellSense-Predictive-Maintenance
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

**Windows**

```bash
.venv\Scripts\activate
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r cellsense_artifacts/requirements.txt
pip install notebook
```

Start Jupyter:

```bash
jupyter notebook
```

Open `CellSense_Advanced.ipynb` and execute the notebook cells.

The notebook includes a Google Colab-specific download cell, which can be skipped when running locally.

### Reproducibility

- Random seed: `42`
- Train-test split: 80/20
- Cross-validation: Stratified 5-fold
- Optuna trials: 30
- Bootstrap resamples: 1,000

Exact reproducibility may depend on Python and library versions.

---

## 14. Model Export and Inference

The trained supervised classification pipeline and anomaly detection components are exported using Joblib.

The project also exports:

- Model metadata.
- Selected feature names.
- Decision threshold.
- Feature engineering implementation.
- Evaluation metrics.
- Statistical test results.
- SHAP feature importance tables.
- Environment information.

The notebook validates model serialization by reloading the saved components and comparing their predictions against the original outputs.

For inference, raw operational measurements must first undergo the same feature engineering and preprocessing used during model development.

**Security note:** Only load Joblib model files from trusted sources.

---

## 15. Limitations

The project has several important limitations:

**Synthetic Data:** The AI4I dataset is synthetically generated and may not reflect the complexity of real manufacturing environments.

**Class Imbalance:** Machine failures are relatively rare, resulting in uncertainty around minority-class performance.

**No Temporal Failure Forecasting:** The dataset does not provide sufficient longitudinal machine histories to predict time-to-failure or Remaining Useful Life.

**Probability Calibration:** Model outputs have not been independently calibrated as real-world failure probabilities.

**Feature Dependencies:** Some engineered variables are mathematically related to original measurements, which can affect interpretability.

**External Validation:** The model has not been evaluated on an independent real-world industrial dataset.

**Production Readiness:** Real-time sensor integration, industrial communication protocols, monitoring infrastructure, and production deployment are outside the current scope.

---

## 16. Future Improvements

Potential future extensions include:

- Validation using real industrial sensor data.
- Time-series modeling and temporal feature extraction.
- Remaining Useful Life prediction with suitable longitudinal data.
- Additional anomaly detection algorithms.
- Probability calibration and cost-sensitive threshold selection.
- REST API development for model inference.
- Interactive monitoring dashboards.
- Real-time sensor integration using MQTT or OPC UA.
- MLOps practices, model monitoring, and automated retraining.
- Integration with industrial automation environments.

---

## 17. Conclusion

CellSense demonstrates an integrated approach to industrial machine failure classification using statistical analysis, mechanical feature engineering, machine learning, explainable AI, and anomaly detection.

The optimized XGBoost model achieved **99.30% accuracy, 96.55% precision, 82.35% recall, and 98.09% ROC-AUC** on the independent synthetic test dataset.

The experiments also demonstrated measurable improvements from physics-informed feature engineering and explored the complementary behavior of statistical anomaly detection.

The project provides a foundation for further research into intelligent manufacturing, machine condition monitoring, and data-driven maintenance decision support.

---

## 18. Acknowledgments

**Dataset:** AI4I 2020 Predictive Maintenance Dataset, UCI Machine Learning Repository.

The notebook was developed as an educational, AI-assisted research prototype using Python and open-source Data Science libraries.

---

**Project Status:** Completed Research Prototype  
**Primary Domain:** Data Science & Machine Learning  
**Application Domain:** Industrial Condition Monitoring & Predictive Maintenance
