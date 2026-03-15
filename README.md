# AI-Enabled Visa Status Prediction and Processing Time Estimator

A machine learning system that predicts H-1B visa processing times using historical U.S. Department of Labor LCA disclosure data.

---

## Project Motivation

Visa applicants often face long waiting times with very little visibility into how long their application might take. This project digs into historical LCA data to find patterns — seasonal trends, employer-specific behavior, geographic differences — and builds regression models that can give applicants a data-backed estimate of their processing timeline.

---

## Dataset

| | |
|---|---|
| **Source** | U.S. Department of Labor |
| **Dataset** | LCA Disclosure Data FY2026 Q1 |
| **Records** | 83,120 |
| **Original Features** | 98 |
| **Engineered Features** | 100+ |

> The dataset file (`LCA_Disclosure_Data_FY2026_Q1.xlsx`) is not included in this repo due to size. Download it from the [DOL website](https://www.dol.gov/agencies/eta/foreign-labor/performance).

---

## Project Structure

```
├── 00_main.py                  # Main script (all milestones)
├── eda_outputs/
│   ├── cleaned_data.csv
│   ├── SUMMARY_REPORT.txt
│   ├── milestone2/
│   │   ├── feature_engineered_data.csv
│   │   ├── state_features.csv
│   │   ├── employer_features.csv
│   │   └── soc_features.csv
│   └── *.png                   # EDA visualizations
├── model_results/
│   ├── model_comparison.csv
│   ├── feature_importance_*.csv
│   └── *.png                   # Model visualizations
├── requirements.txt
└── README.md
```

---

## Pipeline

### Milestone 1 — Data Collection & EDA ✅
- Loaded and cleaned raw LCA disclosure data
- Created target variable (`PROCESSING_DAYS = DECISION_DATE - RECEIVED_DATE`)
- Removed duplicates, invalid statuses, and outliers
- EDA across case status, visa class, states, monthly trends, and wages
- Statistical tests (t-test, ANOVA, Pearson correlation)

### Milestone 2 — Feature Engineering ✅
- State-level processing averages and application counts
- Employer-level processing history
- SOC occupation popularity tiers
- Log-transformed wage and state wage percentile
- Cyclical seasonal encoding (sin/cos) and H-1B cap season flag

### Milestone 3 — Predictive Modeling ✅
- Trained Linear Regression, Ridge, Random Forest, and Gradient Boosting
- Evaluated on MAE, RMSE, and R²
- Hyperparameter tuning on Random Forest via GridSearchCV (5-fold CV)
- Built a `predict_processing_time()` function that returns a point estimate + 90% confidence interval

### Milestone 4 — Deployment 🔲
- Streamlit web app with prediction interface
- Connect trained model to frontend input form

---

## Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | — | — | — |
| Ridge | — | — | — |
| Random Forest | — | — | — |
| Gradient Boosting | — | — | — |

> Results will be updated after final model evaluation.

---

## Tech Stack

- Python 3.14
- pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn
- joblib

---

## How to Run

**1. Clone the repo**
```bash
git clone https://github.com/Soham-Pasalkar/AI-Enabled-Visa-Status-Prediction-and-Processing-Time-Estimator.git
cd AI-Enabled-Visa-Status-Prediction-and-Processing-Time-Estimator
```

**2. Set up virtual environment**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add the dataset**

Download `LCA_Disclosure_Data_FY2026_Q1.xlsx` from the DOL website and place it in the project root.

**5. Run**
```bash
python 00_main.py
```

---

## Author

**Soham Pasalkar**
GitHub: [github.com/Soham-Pasalkar](https://github.com/Soham-Pasalkar)

---

## Status

| Milestone | Status |
|---|---|
| 1 — Data Collection & EDA | ✅ Complete |
| 2 — Feature Engineering | ✅ Complete |
| 3 — Predictive Modeling | ✅ Complete |
| 4 — Deployment | 🔲 Planned |
