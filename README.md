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
├── app.py                      # Streamlit web app (Milestone 4)
├── eda_outputs/
│   ├── cleaned_data.csv
│   ├── SUMMARY_REPORT.txt
│   ├── top_employers_analysis.csv
│   ├── top_occupations_analysis.csv
│   ├── milestone2/
│   │   ├── feature_engineered_data.csv
│   │   ├── state_features.csv
│   │   ├── employer_features.csv
│   │   └── soc_features.csv
│   └── *.png                   # EDA visualizations
├── model_results/
│   ├── model_comparison.csv
│   ├── feature_importance_*.csv
│   ├── label_encoders.pkl
│   ├── scaler.pkl
│   ├── feature_list.pkl
│   ├── best_model_tuned.pkl
│   └── *.png                   # Model visualizations
├── requirements.txt
└── README.md
```

---

## Pipeline

### Milestone 1 — Data Collection & EDA ✅
- Loaded and cleaned raw LCA disclosure data (83,120 records, 98 features)
- Created target variable (`PROCESSING_DAYS = DECISION_DATE - RECEIVED_DATE`)
- Removed duplicates, invalid statuses, negative/zero/outlier processing times (>365 days)
- EDA across case status, visa class, top states, monthly trends, and wages
- Statistical tests: t-test (full-time vs part-time), ANOVA (visa classes), Pearson correlation (wage vs processing time)
- Top employer and occupation breakdowns saved to CSV

### Milestone 2 — Feature Engineering ✅
- State-level processing averages, medians, and application counts
- Employer-level processing history and application volume
- SOC occupation popularity tiers (Rare / Uncommon / Common / Very Common) via quartile binning
- Log-transformed wage and within-state wage percentile ranking
- Cyclical seasonal encoding (sin/cos on month) and H-1B cap season flag (April)

### Milestone 3 — Predictive Modeling ✅
- Trained Linear Regression, Ridge, Random Forest, and Gradient Boosting on 22 engineered features
- Evaluated on MAE, RMSE, and R² with 5-fold cross-validation
- Hyperparameter tuning on Random Forest via GridSearchCV (n_estimators, max_depth, min_samples_split, min_samples_leaf)
- Best tuned model: MAE 0.597 days, R² 0.9864 on 15,151 test records
- `predict_processing_time()` function returns a point estimate + 90% confidence interval

### Milestone 4 — Deployment ✅
- Streamlit web app (`app.py`) with dark-themed prediction interface
- Sidebar inputs: worksite state, visa class, filing month, annual wage, position type, occupation
- Displays point estimate, 90% confidence interval, and delta vs. state average
- Interactive charts: processing time by filing month and top states by average processing time
- Loads trained model artifacts from `model_results/` at runtime

---

## Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 1.229 | 2.262 | 0.9411 |
| Ridge | 1.229 | 2.262 | 0.9411 |
| Gradient Boosting | 0.763 | 1.144 | 0.9849 |
| **Random Forest** ✅ | **0.597** | **1.092** | **0.9863** |

Best model: **Random Forest** (tuned) — MAE 0.597 days, R² 0.9864 across 15,151 test records.

---

## Tech Stack

- Python 3.14
- pandas, NumPy, SciPy
- Matplotlib, Seaborn
- scikit-learn
- joblib
- Streamlit

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

**5. Run the pipeline**
```bash
python 00_main.py
```

**6. Launch the app**
```bash
streamlit run app.py
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
| 4 — Deployment | ✅ Complete |
