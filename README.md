# UNICEF-DS — Child Malnutrition & Gender Inequality Analysis

Data science coursework exploring the relationship between **gender inequality**, **WASH (Water, Sanitation & Hygiene)**, **education**, and **child malnutrition** using UNICEF datasets.

## Research Question

> Does gender inequality — measured through child marriage, adolescent health access, and FGM prevalence — explain variation in child malnutrition (stunting, wasting, overweight) across countries, beyond what income alone predicts?

## Data Sources

All data sourced from [UNICEF Data](https://data.unicef.org/):

| Dataset | Source | Key Variables |
|---------|--------|---------------|
| **Malnutrition** | JME 2025 (stunting, wasting, overweight, overlapping) | National prevalence, gender/wealth/urban-rural disaggregation |
| **Gender Inequality** | UNICEF child marriage, FGM, adolescent health databases | Child marriage by 15/18, FGM prevalence, ANC4, contraceptive use |
| **WASH** | JMP WASH in Schools 2024 | Basic water, sanitation, hygiene (national/urban/rural) |
| **Education** | UNICEF education dataset | Primary completion, literacy rates (male/female) |
| **Child Health** | Child Health Coverage Database 2025 | Diarrhoea treatment, pneumonia care, breastfeeding, birthweight |

## Project Structure

```
UNICEF-DS/
├── Malnutrition Datasets/          # Raw malnutrition data (JME 2025)
├── WASH/                           # WASH data cleaning pipeline
│   ├── WASH_datacleaning.py        # Cleans JMP WASH data (year=2023)
│   └── outputs/                    # wash_clean_data.csv
├── outputs/                        # All cleaned CSVs and analysis outputs
│   └── xgboost/                    # XGBoost model plots and metrics
│
├── malnutrition_analysis.py        # Malnutrition data extraction pipeline
├── gender_inequality_analysis.py   # Gender data cleaning & ISO mapping
├── gender-malnutrition.py          # OLS regression (Models 1-3), SHAP, Lasso
├── education_cleaning.py           # Education data cleaning
├── xgboost_cleaning.py             # Merges all datasets for XGBoost
├── xgboost_analysis.py             # XGBoost predictive model + SHAP
│
├── WASH_visualisation.ipynb                    # WASH data visualisations
├── education_visualisation.ipynb               # Education visualisations
├── xgboost_datasets_visualisation.ipynb        # Merged dataset EDA + model results
├── Adolescent_cleaning_and_visulisation.ipynb  # Adolescent health cleaning
├── Child marriage cleaning and visulisation.ipynb  # Child marriage cleaning
└── FGM_cleaning_and_visulisation.ipynb         # FGM cleaning & choropleth
```

## Analytical Framework

The project follows a **four-model incremental approach**:

1. **Model 1 (Baseline)**: Malnutrition ~ Income Group
2. **Model 2 (Gender)**: Malnutrition ~ Gender Inequality Indicators
3. **Model 3 (Integrated)**: Malnutrition ~ Income + Gender + WASH + Education + Health
4. **XGBoost**: Non-linear predictive model with SHAP interpretability

## Key Findings

- **Income group** is the strongest single predictor of stunting (SHAP = 3.03)
- **Low birthweight** is the top predictor for wasting
- **Female primary completion** is the top predictor for overweight
- WASH indicators are absorbed by income group due to high collinearity
- XGBoost CV R-squared: Stunting 0.58, Wasting 0.34, Overweight 0.24

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run data cleaning pipelines
python malnutrition_analysis.py
python gender_inequality_analysis.py
python education_cleaning.py
python WASH/WASH_datacleaning.py

# Merge datasets and run XGBoost
python xgboost_cleaning.py
python xgboost_analysis.py
```

## Team

Data Science Coursework — UNICEF Child Wellbeing Analysis
