import os
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
FILE          = os.path.join(SCRIPT_DIR, "JMP-WASH-in-schools-2024-data-by-country.xlsx")
OUTPUT_FOLDER = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


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
    '>99' -> 99.5, '<1' -> 0.5, '-' -> NaN.
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
    print("=" * 60)
    print("WASH DATA CLEANING PIPELINE")
    print("=" * 60)

    raw = pd.read_excel(FILE, sheet_name="WASH", header=None)
    print(f"  Raw shape: {raw.shape}")

    data = raw.iloc[2:].copy()
    data.columns = range(data.shape[1])

    data_2023 = data[data[1] == 2023].copy()
    print(f"  Rows with year=2023: {len(data_2023)} countries")

    col_indices = list(COL_MAP.keys())
    df = data_2023[col_indices].copy()
    df.columns = list(COL_MAP.values())


    df["ISO"]     = df["ISO"].astype(str).str.strip()
    df["country"] = df["country"].astype(str).str.strip()
    df["sdg_region"] = df["sdg_region"].astype(str).str.strip()
    df["unicef_reporting_region"] = (
        df["unicef_reporting_region"].astype(str).str.strip()
        .replace("nan", np.nan))
    df["year"] = 2023


    df = df[df["ISO"].notna() & (df["ISO"] != "nan")].copy()


    for col in INDICATOR_COLS:
        df[col] = df[col].apply(convert_value)

    # Fill missing regions for known territories
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

    # Flag countries with all 3 basic indicators missing
    core = ["wat_bas_nat", "san_bas_nat", "hyg_bas_nat"]
    df["all_basic_missing"] = df[core].isna().all(axis=1)
    n_flagged = df["all_basic_missing"].sum()
    print(f"  Flagged {n_flagged} countries with all 3 basic indicators missing")

    df = df.reset_index(drop=True)

    # Coverage report
    print(f"\n  {'-'*50}")
    print(f"  COVERAGE SUMMARY (year = 2023)")
    print(f"  {'-'*50}")
    print(f"  Total countries : {len(df)}")
    for col in INDICATOR_COLS:
        print(f"  {col:20s}: {df[col].notna().sum():3d} countries")

    core = ["wat_bas_nat", "san_bas_nat", "hyg_bas_nat"]
    n_all = df[core].notna().all(axis=1).sum()
    print(f"\n  With ALL 3 core indicators: {n_all} countries")

    print(f"\n  Value range check:")
    all_ok = True
    for col in INDICATOR_COLS:
        data_col = df[col].dropna()
        if len(data_col) == 0:
            continue
        out_range = data_col[(data_col < 0) | (data_col > 100)]
        if len(out_range) > 0:
            print(f"    {col}: {len(out_range)} values out of [0,100]")
            all_ok = False
    if all_ok:
        print(f"    All values within [0,100] range")

    print(f"\n  Missing values:")
    miss = df.isna().sum()
    for col, n in miss[miss > 0].items():
        print(f"    {col:30s}: {n}")

    print(f"\n  Descriptive statistics:")
    print(df[core].describe().round(2).to_string())

    df[INDICATOR_COLS] = df[INDICATOR_COLS].round(2)

    out_path = os.path.join(OUTPUT_FOLDER, "wash_clean_data.csv")
    df.to_csv(out_path, index=False)
    print(f"\n  Saved -> {out_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"\nPIPELINE COMPLETE")

    return df


if __name__ == "__main__":
    clean_wash()