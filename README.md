# Credit Risk Analysis — Loan Default Prediction

## Overview
End-to-end machine learning project predicting loan default risk using 
307,511 real loan applications from Home Credit. Built to demonstrate 
production-ready credit risk modeling with full explainability using SHAP.

## Key Results
- XGBoost model achieved AUC 0.7488 on holdout test set
- Outperformed Logistic Regression (0.7422) and Random Forest (0.6793)
- SHAP explainability layer provides audit trail for every credit decision
- Handles 8% class imbalance using SMOTE oversampling

## Key Findings
- External credit scores (EXT_SOURCE_2, EXT_SOURCE_3) are the strongest 
  predictors of default risk
- Applicants under 25 default at 12.3% vs only 5.1% for applicants over 55
- Cash loans carry higher default risk (8.5%) than revolving loans (5.5%)
- Employment stability matters more than income level for predicting default
- Every prediction can be explained using SHAP waterfall charts for 
  regulatory compliance

## Model Performance

| Model | AUC-ROC | Rank |
|-------|---------|------|
| XGBoost | 0.7488 | 1st |
| Logistic Regression | 0.7422 | 2nd |
| Random Forest | 0.6793 | 3rd |

## Project Structure

| Notebook | Description |
|----------|-------------|
| 01_data_loading | Load and explore 307k loan applications |
| 02_cleaning | Feature engineering, encoding, imputation |
| 03_eda | 5 charts exploring default risk patterns |
| 04_modeling | Train and compare 3 ML models |
| 05_explainability | SHAP feature importance and waterfall analysis |
| 06_insights | Business insights report |
| 07_dashboard | Interactive Plotly dashboard |

## Tools Used
Python, Pandas, NumPy, Scikit-learn, XGBoost, SHAP, 
Imbalanced-learn, Matplotlib, Seaborn, Plotly

## Data
This project uses the Home Credit Default Risk dataset from Kaggle.

Download here:
https://www.kaggle.com/competitions/home-credit-default-risk/data

After downloading, extract all CSV files into the data/ folder 
before running the notebooks.

Files needed:
- application_train.csv
- application_test.csv

## How to Run
1. Clone this repo
2. Download dataset from Kaggle link above
3. Run pip install -r requirements.txt
4. Open notebooks in order 01 through 07
5. Open dashboard/credit_risk_dashboard.html in browser 
   for interactive visualization

## Why This Project Matters
Credit risk modeling is the core of every lending business. This project
demonstrates not just model building but the full production workflow:
handling class imbalance, comparing multiple models honestly, and providing
regulatory-compliant explainability through SHAP. A model that cannot
explain its decisions cannot be deployed in a real financial institution.
