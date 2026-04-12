# UNICEF-DS — Child Malnutrition & Gender Inequality Analysis

## Overview

This project analyzes the relationships between malnutrition, gender inequality, education, WASH (Water, Sanitation, and Hygiene), and other socioeconomic indicators using OLS regression and XGBoost machine learning models. All data sources are from UNICEF and related international datasets.

## Data Sources

All data sourced from [UNICEF Data](https://data.unicef.org/):

| Dataset | Source | Key Variables |
|---------|--------|---------------|
| **Malnutrition** | JME 2025 (stunting, wasting, overweight, overlapping) | National prevalence, gender/wealth/urban-rural disaggregation |
| **Gender Inequality** | UNICEF child marriage, FGM, adolescent health databases | Child marriage by 15/18, FGM prevalence, ANC4, contraceptive use |
| **WASH** | JMP WASH in Schools 2024 | Basic water, sanitation, hygiene (national/urban/rural) |
| **Education** | UNICEF education dataset | Primary completion, literacy rates (male/female) |
| **Child Health** | Child Health Coverage Database 2025 | Diarrhoea treatment, pneumonia care, breastfeeding, birthweight |


## Repository structure

```text
UNICEF-DS/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                # raw source datasets
│   └── processed/          # cleaned datasets used for analysis
├── scripts/
│   ├── data_preparation/   # cleaning and merge scripts
│   ├── exploration/        # exploratory visualisation scripts
│   ├── modelling/          # OLS and final XGBoost v2 scripts
│   └── legacy/             # archived scripts retained for reference
├── notebooks/              # exploratory notebooks
├── outputs/
│   ├── exploration/        # exploratory figures
│   ├── ols/                # OLS model outputs
│   └── xgboost_v2/         # final XGBoost v2 outputs
└── report/                 # final report files
```


## Data Exploration

Exploratory analysis is conducted separately for:

- Global malnutrition distributions
- Gender inequality indicators
- Education indicators
- WASH indicators

Outputs are stored in `outputs/exploration/`.

## OLS Modelling

A staged OLS comparison is used as the main inferential framework:

- **Model 1:** `outcome ~ income group`
- **Model 2:** `outcome ~ female child marriage by age 18`
- **Model 3:** `outcome ~ female child marriage by age 18 + female literacy`
- **Model 4:** `outcome ~ female child marriage by age 18 + female literacy + basic sanitation + income group`

All models are estimated on the same complete-case sample (n = 75) for fair comparison of adjusted R², AIC, and BIC.

## XGBoost v2 Modelling (Final Predictive Framework)

The final predictive framework is implemented in:

- `xgboost_v2_cleaning.py`
- `Xgboost_v2_analysis.py`

Key features:

- Uses a broader predictor set across gender, education, WASH, and child health coverage
- Applies a minimum predictor coverage threshold (n ≥ 90)
- One-hot encodes income group
- Evaluates performance using true nested cross-validation
- Interprets feature importance using SHAP

## Main Outputs Used in the Report

### OLS
- Model comparison table
- Grouped coefficient plot
- Coefficient summaries

### XGBoost v2
- `xgboost_v2_performance_table.csv`
- `xgboost_v2_shap_rankings.csv`
- `xgboost_v2_shap_bar_stunting_national.png`
- `xgboost_v2_shap_dependence_stunting_national.png`

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
````

### 2. Run the Core Pipeline

To replicate the analysis, execute the scripts in the **exact order** listed below. This ensures that data is cleaned and harmonized before being fed into the models.

**Data Preparation & Cleaning:**
```bash
python scripts/data_preparation/malnutrition_analysis.py
python scripts/data_preparation/gender_inequality_analysis.py
python scripts/data_preparation/education_datacleaning.py
python scripts/data_preparation/WASH_datacleaning.py
python scripts/data_preparation/xgboost_v2_cleaning.py
````

## Notes on Legacy Files

Older XGBoost scripts (`xgboost_analysis.py`, `xgboost_cleaning.py`) and other exploratory scripts are retained for reference only.

The final XGBoost results used in the report are generated exclusively by:

- `xgboost_v2_cleaning.py`
- `Xgboost_v2_analysis.py`

## Team
- Jiayi Qin
- Omar Elekiaby
- David Reeder
- Sulaiman Alsalami
- Saud Binlebdah
