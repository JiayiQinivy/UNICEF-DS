"""
wash_cleaning.py
=============================================================
WASH Data Cleaning Pipeline

Source: JMP WASH in Schools 2024
        JMP-WASH-in-schools-2024-data-by-country.xlsx
Sheet:  WASH (main sheet — the one visible in Excel)

Column mapping (confirmed from raw header rows):
  Col 0  = country name
  Col 1  = year
  Col 7  = Water   — Basic    (TOTAL/national)
  Col 8  = Water   — Limited  (TOTAL/national)
  Col 9  = Water   — None     (TOTAL/national)
  Col 25 = Sanitation — Basic    (TOTAL/national)
  Col 26 = Sanitation — Limited  (TOTAL/national)
  Col 27 = Sanitation — None     (TOTAL/national)
  Col 43 = Hygiene    — Basic    (TOTAL/national)
  Col 44 = Hygiene    — Limited  (TOTAL/national)
  Col 45 = Hygiene    — None     (TOTAL/national)
  Col 61 = SDG Region
  Col 64 = UNICEF Reporting Region
  Col 67 = ISO3 code

Value encoding in this sheet:
  '>99'  = reported as ">99%"  -> converted to 99.5
  '<1'   = reported as "<1%"   -> converted to 0.5
  '-'    = no data             -> converted to NaN
  numeric strings / floats     -> converted to float

Strategy:
  - Read the WASH main sheet (not the hidden Water/Sanitation/Hygiene
    Data sheets used by the previous cleaning script)
  - Filter to year = 2023 (latest year, all 191 countries present)
  - Extract national-level basic/limited/no-service for water,
    sanitation, and hygiene
  - Convert all string placeholders to numeric or NaN
  - Rename iso3 -> ISO to match the project merge key
  - Output: one row per country, wide format, ready for Model 3

Output file: outputs/wash_clean_data.csv
=============================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

# ===== PATHS (relative to repo root) =====
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR      = os.path.dirname(SCRIPT_DIR)
FILE          = os.path.join(ROOT_DIR, "data", "JMP-WASH-in-schools-2024-data-by-country.xlsx")
OUTPUT_FOLDER = os.path.join(ROOT_DIR, "outputs")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
# ==========================================

# Column indices in the WASH main sheet (0-indexed, confirmed from header)
COL_MAP = {
    67: "ISO",
    0:  "country",
    1:  "year",
    61: "sdg_region",
    64: "unicef_reporting_region",
    7:  "wat_bas_nat",
    8:  "wat_lim_nat",
    9:  "wat_none_nat",
    25: "san_bas_nat",
    26: "san_lim_nat",
    27: "san_none_nat",
    43: "hyg_bas_nat",
    44: "hyg_lim_nat",
    45: "hyg_none_nat",
}

INDICATOR_COLS = [
    "wat_bas_nat", "wat_lim_nat", "wat_none_nat",
    "san_bas_nat", "san_lim_nat", "san_none_nat",
    "hyg_bas_nat", "hyg_lim_nat", "hyg_none_nat",
]


def convert_value(val):
    """
    Convert JMP string-encoded values to float.
      '>99' -> 99.5  (reported as above 99%, use 99.5 as midpoint)
      '<1'  -> 0.5   (reported as below 1%, use 0.5 as midpoint)
      '-'   -> NaN   (no data)
      numeric -> float
    """
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s in ("-", ""):
        return np.nan
    if s == ">99":
        return 99.5
    if s == "<1":
        return 0.5
    try:
        return float(s)
    except ValueError:
        return np.nan


def clean_wash():
    raw = pd.read_excel(FILE, sheet_name="WASH", header=None)

    # Skip the two header rows (rows 0 and 1), keep data rows
    data = raw.iloc[2:].copy()
    data.columns = range(data.shape[1])

    data_2023 = data[data[1] == 2023].copy()

    # Extract and rename required columns
    col_indices = list(COL_MAP.keys())
    df = data_2023[col_indices].copy()
    df.columns = list(COL_MAP.values())

    # Clean text columns
    df["ISO"]     = df["ISO"].astype(str).str.strip()
    df["country"] = df["country"].astype(str).str.strip()
    df["sdg_region"] = df["sdg_region"].astype(str).str.strip()
    df["unicef_reporting_region"] = (
        df["unicef_reporting_region"].astype(str).str.strip()
        .replace("nan", np.nan))
    df["year"] = 2023

    # Drop rows with missing ISO
    df = df[df["ISO"].notna() & (df["ISO"] != "nan")].copy()

    # Convert all indicator columns
    for col in INDICATOR_COLS:
        df[col] = df[col].apply(convert_value)

    # Fill missing unicef_reporting_region for known territories
    REGION_FILL = {
        "BMU": "North America",
        "CYM": "Latin America and Caribbean",
        "HKG": "East Asia and Pacific",
        "MAC": "East Asia and Pacific",
        "GIB": "Europe and Central Asia",
    }
    for iso, region in REGION_FILL.items():
        mask = (df["ISO"] == iso) & (df["unicef_reporting_region"].isna())
        df.loc[mask, "unicef_reporting_region"] = region

    core = ["wat_bas_nat", "san_bas_nat", "hyg_bas_nat"]
    df["all_basic_missing"] = df[core].isna().all(axis=1)

    df = df.reset_index(drop=True)

    df[INDICATOR_COLS] = df[INDICATOR_COLS].round(2)

    # ── Save ───────────────────────────────────────────────────
    out_path = os.path.join(OUTPUT_FOLDER, "wash_clean_data.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path}  ({df.shape[0]} countries)")

    return df


if __name__ == "__main__":
    clean_wash()
    print("PIPELINE COMPLETE")