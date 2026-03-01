# AI-Enabled Visa Status Prediction and Processing Time Estimator

A machine learning system that predicts H-1B visa processing time using
historical U.S. Department of Labor LCA disclosure data.

------------------------------------------------------------------------

## Project Motivation

Visa applicants often face uncertainty regarding processing timelines.

This project analyzes historical visa data and builds predictive models
to estimate processing duration.

------------------------------------------------------------------------

## Dataset

Source: U.S. Department of Labor\
Dataset: LCA Disclosure Data FY2026 Q1

Records: 83,120\
Features: 98 original → 100+ engineered

------------------------------------------------------------------------

## Project Pipeline

### Milestone 1 --- Data Collection & EDA (Completed)

-   Data cleaning and preprocessing\
-   Target variable creation\
-   Feature engineering\
-   Exploratory Data Analysis\
-   Statistical analysis

Output:

eda_outputs/

------------------------------------------------------------------------

### Milestone 2 --- Feature Engineering (Completed)

Created features:

-   State processing averages\
-   Employer processing averages\
-   SOC popularity\
-   Wage features\
-   Seasonal encoding

Output:

eda_outputs/milestone2/

------------------------------------------------------------------------

### Milestone 3 --- Predictive Modeling (In Progress)

Models:

-   Linear Regression\
-   Random Forest\
-   Gradient Boosting

------------------------------------------------------------------------

### Milestone 4 --- Deployment (Planned)

-   Streamlit web app\
-   Prediction interface

------------------------------------------------------------------------

## Tech Stack

Python\
Pandas\
NumPy\
Matplotlib\
Seaborn\
Scikit‑learn

------------------------------------------------------------------------

## How to Run

Install dependencies:

pip install pandas numpy matplotlib seaborn scikit-learn openpyxl

Run:

python 00_main.py

------------------------------------------------------------------------

## Author

Soham Pasalkar

GitHub: https://github.com/Soham-Pasalkar

------------------------------------------------------------------------

## Project Status

Milestone 1 --- Complete\
Milestone 2 --- Complete\
Milestone 3 --- In Progress\
Milestone 4 --- Planned
