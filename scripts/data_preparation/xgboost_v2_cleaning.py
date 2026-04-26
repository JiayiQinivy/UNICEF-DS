"""
xgboost_v2_cleaning.py
=============================================================
Builds the XGBoost analytical dataset from scratch.

Inputs (all previously validated):
  outputs/malnutrition_modelling_sample.csv      (163 countries)
  outputs/gender_inequality_analysis.csv
  education_clean.csv
  WASH/outputs/wash_clean_data.csv
  data/Child-Health-Coverage-Database-November-2025.xlsx

Merge key: ISO (ISO3 country code)

Output:
  XGBoost_v2/outputs/xgboost_v2_dataset.csv

Predictor selection:
  Minimum coverage threshold: n >= 90 countries
  Excluded due to low coverage (n=73):
    - anc4_15_19_pct
    - modern_contraceptive_pct
  Excluded due to very low coverage (n=30):
    - fgm_prevalence_pct

Predictor domains:
  Gender:    female_married_by_15, female_married_by_18,
             marriage_gap_18, anc4_15_19_pct,
             modern_contraceptive_pct
  Education: literacy_f, completion_primary_f,
             oos_upsec_f (if available)
  WASH:      wat_bas_nat, san_bas_nat, hyg_bas_nat
  Coverage:  diarrhoea_care_pct, pneumonia_care_pct
  Control:   income_group

Outcomes:
  stunting_national, wasting_national, overweight_national
=============================================================
"""

import os
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR     = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR   = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAL_CSV      = os.path.join(ROOT_DIR, "outputs",
                             "malnutrition_modelling_sample.csv")
GENDER_CSV   = os.path.join(ROOT_DIR, "outputs",
                             "gender_inequality_analysis.csv")
EDU_CSV      = os.path.join(ROOT_DIR, "education_clean.csv")
WASH_CSV     = os.path.join(ROOT_DIR, "WASH", "outputs",
                             "wash_clean_data.csv")
COVERAGE_XLS = os.path.join(ROOT_DIR, "data",
                             "Child-Health-Coverage-Database-November-2025.xlsx")

OUTCOMES = [
    "stunting_national",
    "wasting_national",
    "overweight_national",
]

# ── Gender predictors to retain ────────────────────────────────
GENDER_COLS = [
    "female_married_by_15",
    "female_married_by_18",
    "marriage_gap_18",
    # anc4_15_19_pct excluded: n=73 < coverage threshold (n>=90)
    # modern_contraceptive_pct excluded: n=73 < coverage threshold
]

# ── Education predictors to retain ────────────────────────────
EDU_COLS = [
    "literacy_f",
    "completion_primary_f",
    "oos_upsec_f",          # may not exist; handled gracefully
]

# ── WASH predictors ────────────────────────────────────────────
WASH_COLS = [
    "wat_bas_nat",
    "san_bas_nat",
    "hyg_bas_nat",
]


# ──────────────────────────────────────────────────────────────
# SECTION 1: LOAD MALNUTRITION BASE
# ──────────────────────────────────────────────────────────────
def load_malnutrition():
    df = pd.read_csv(MAL_CSV)
    keep = ["ISO", "country", "unicef_region", "income_group"] + OUTCOMES
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()
    print(f"  Malnutrition base loaded: {len(df)} countries")
    return df

# ──────────────────────────────────────────────────────────────
# SECTION 2: LOAD GENDER
# ──────────────────────────────────────────────────────────────
def load_gender():
    df = pd.read_csv(GENDER_CSV)
    keep = ["ISO"] + [c for c in GENDER_COLS if c in df.columns]
    df = df[keep].copy()
    available = [c for c in GENDER_COLS if c in df.columns]
    print(f"  Gender predictors available: {available}")
    return df


# ──────────────────────────────────────────────────────────────
# SECTION 3: LOAD EDUCATION
# ──────────────────────────────────────────────────────────────
def load_education():
    df = pd.read_csv(EDU_CSV)
    keep = ["ISO"] + [c for c in EDU_COLS if c in df.columns]
    df = df[keep].copy()
    available = [c for c in EDU_COLS if c in df.columns]
    print(f"  Education predictors available: {available}")
    return df


# ──────────────────────────────────────────────────────────────
# SECTION 4: LOAD WASH
# ──────────────────────────────────────────────────────────────
def load_wash():
    df = pd.read_csv(WASH_CSV)

    # Standardise ISO column name
    if "iso3" in df.columns:
        df = df.rename(columns={"iso3": "ISO"})

    # Keep latest year per country if year column exists
    year_cols = [c for c in df.columns if "year" in c.lower()]
    if year_cols:
        df = (df.sort_values(year_cols[0])
                .groupby("ISO", as_index=False)
                .last())

    keep = ["ISO"] + [c for c in WASH_COLS if c in df.columns]
    df = df[keep].copy()
    available = [c for c in WASH_COLS if c in df.columns]
    print(f"  WASH predictors available: {available}")
    return df


# ──────────────────────────────────────────────────────────────
# SECTION 5: LOAD CHILD HEALTH COVERAGE (Long format)
# ──────────────────────────────────────────────────────────────
def load_coverage():
    """
    Read DIARCARE and PNEUCARE from the Long-format sheet.
    Filter: Stratifier == 'National', Level == 'National'
    Take the most recent year per country.
    """
    df = pd.read_excel(COVERAGE_XLS, sheet_name="Long")

    # Standardise column names
    df.columns = df.columns.str.strip()

    # Filter to national-level only
    national = df[
        (df["Stratifier"] == "National") &
        (df["Level"] == "National")
    ].copy()

    # Keep only the two indicators we need
    indicators = ["DIARCARE", "PNEUCARE"]
    national = national[national["Indicator"].isin(indicators)].copy()

    # Convert value to numeric
    national["Value"] = pd.to_numeric(national["Value"], errors="coerce")

    # Keep latest year per country per indicator
    national["Latest Year"] = pd.to_numeric(
        national["Latest Year"], errors="coerce")
    national = (national
                .sort_values("Latest Year")
                .groupby(["ISO Code", "Indicator"], as_index=False)
                .last())

    # Pivot to wide format
    wide = national.pivot(
        index="ISO Code",
        columns="Indicator",
        values="Value"
    ).reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={
        "ISO Code":  "ISO",
        "DIARCARE":  "diarrhoea_care_pct",
        "PNEUCARE":  "pneumonia_care_pct",
    })

    print(f"  Coverage: diarrhoea_care_pct n="
          f"{wide['diarrhoea_care_pct'].notna().sum()}, "
          f"pneumonia_care_pct n="
          f"{wide['pneumonia_care_pct'].notna().sum()}")
    return wide


# ──────────────────────────────────────────────────────────────
# SECTION 6: MERGE ALL
# ──────────────────────────────────────────────────────────────
def build_dataset():
    print("=" * 60)
    print("XGBOOST V2 — DATA CLEANING")
    print("=" * 60)

    # Check all files exist
    for label, path in [
        ("Malnutrition", MAL_CSV),
        ("Gender",       GENDER_CSV),
        ("Education",    EDU_CSV),
        ("WASH",         WASH_CSV),
        ("Coverage",     COVERAGE_XLS),
    ]:
        status = "found" if os.path.exists(path) else "NOT FOUND"
        print(f"  {label:15s}: {status}")
    print()

    mal      = load_malnutrition()
    gender   = load_gender()
    edu      = load_education()
    wash     = load_wash()
    coverage = load_coverage()

    # Sequential left merges on ISO
    df = mal.copy()
    for src, name in [
        (gender,   "gender"),
        (edu,      "education"),
        (wash,     "WASH"),
        (coverage, "coverage"),
    ]:
        before = len(df)
        df = df.merge(src, on="ISO", how="left")
        print(f"  After merging {name:12s}: {len(df)} rows "
              f"(was {before})")

    # Drop duplicate columns from merges (e.g. country_x / country_y)
    for suffix in ["_x", "_y"]:
        dup_cols = [c for c in df.columns if c.endswith(suffix)]
        for col in dup_cols:
            base = col[:-2]
            if base in df.columns:
                df = df.drop(columns=[col])
            else:
                df = df.rename(columns={col: base})

    print(f"\n  Final dataset shape: {df.shape}")

    # ── Coverage summary ──────────────────────────────────────
    print("\n  Predictor coverage summary:")
    predictor_cols = (
        [c for c in GENDER_COLS if c in df.columns] +
        [c for c in EDU_COLS    if c in df.columns] +
        [c for c in WASH_COLS   if c in df.columns] +
        ["diarrhoea_care_pct", "pneumonia_care_pct"]
    )
    for col in predictor_cols:
        if col in df.columns:
            n = df[col].notna().sum()
            print(f"    {col:35s}: {n:3d} / {len(df)}")

    # ── Coverage threshold filter ─────────────────────────────
    MIN_COVERAGE = 90
    predictor_cols = (
            [c for c in GENDER_COLS if c in df.columns] +
            [c for c in EDU_COLS if c in df.columns] +
            [c for c in WASH_COLS if c in df.columns] +
            ["diarrhoea_care_pct", "pneumonia_care_pct"]
    )

    print(f"\n  Applying coverage threshold (n >= {MIN_COVERAGE}):")
    retained = []
    excluded = []
    for col in predictor_cols:
        n = df[col].notna().sum()
        if n >= MIN_COVERAGE:
            retained.append(col)
            print(f"    RETAINED  {col:<35}: n={n}")
        else:
            excluded.append(col)
            print(f"    EXCLUDED  {col:<35}: n={n} < {MIN_COVERAGE}")

    # Keep only outcome cols + metadata + retained predictors
    meta_cols = ["ISO", "country", "unicef_region", "income_group"]
    keep_cols = meta_cols + OUTCOMES + retained
    df = df[[c for c in keep_cols if c in df.columns]]
    print(f"\n  Final predictor set: {len(retained)} predictors")

    # ── Save ──────────────────────────────────────────────────
    out_path = os.path.join(OUTPUT_DIR, "xgboost_v2_dataset.csv")
    df.to_csv(out_path, index=False)
    print(f"\n  Saved: xgboost_v2_dataset.csv "
          f"({len(df)} countries, {len(df.columns)} columns)")

    return df


if __name__ == "__main__":
    build_dataset()
