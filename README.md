# UNICEF-DS — Child Malnutrition & Gender Inequality Analysis

Data science coursework exploring the relationship between **gender inequality**, **WASH (Water, Sanitation & Hygiene)**, **education**, and **child malnutrition** using UNICEF datasets.

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
├── data/                               # All raw source data
│   ├── Child_marriage.csv
│   ├── education-dataset.xlsx
│   ├── Child-Health-Coverage-Database-November-2025.xlsx
│   ├── JMP-WASH-in-schools-2024-data-by-country.xlsx
│   └── Malnutrition Datasets/          # JME 2025 (stunting, wasting, etc.)
│
├── scripts/                            # All Python cleaning & analysis scripts
│   ├── WASH_datacleaning.py            # Cleans JMP WASH data (year=2023)
│   ├── education_cleaning.py           # Education data cleaning
│   ├── gender_inequality_analysis.py   # Gender data cleaning & ISO mapping
│   ├── gender-malnutrition.py          # OLS regression (Models 1-3), SHAP
│   ├── malnutrition_analysis.py        # Malnutrition data extraction pipeline
│   ├── xgboost_cleaning.py             # Merges all datasets for XGBoost
│   └── xgboost_analysis.py             # XGBoost predictive model + SHAP
│
├── notebooks/                          # Jupyter notebooks (cleaning & viz)
│   ├── WASH_visualisation.ipynb
│   ├── education_visualisation.ipynb
│   ├── xgboost_datasets_visualisation.ipynb
│   ├── Adolescent_cleaning_and_visulisation.ipynb
│   ├── Child marriage cleaning and visulisation.ipynb
│   └── FGM_cleaning_and_visulisation.ipynb
│
├── outputs/                            # All generated outputs (CSVs, PNGs)
│   └── xgboost/                        # XGBoost model plots and metrics
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Analytical Framework

The project follows a **four-model incremental approach**:

1. **Model 1 (Baseline)**: Malnutrition ~ Income Group
2. **Model 2 (Gender)**: Malnutrition ~ Gender Inequality Indicators
3. **Model 3 (Integrated)**: Malnutrition ~ Income + Gender + WASH + Education + Health
4. **XGBoost**: Non-linear predictive model with SHAP interpretability

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run data cleaning pipelines
python scripts/malnutrition_analysis.py
python scripts/gender_inequality_analysis.py
python scripts/education_cleaning.py
python scripts/WASH_datacleaning.py

# Merge datasets and run XGBoost
python scripts/xgboost_cleaning.py
python scripts/xgboost_analysis.py
```

## Team
- Omar Elekiaby
- Saud Binlebdah
- David Reeder
- Sulaiman Alsalami
- Jiayi Qin
