import os
import pandas as pd
import numpy as np

FILE = "JMP-WASH-in-schools-2024-data-by-country.xlsx"

# ── Key columns to keep per sheet ──────────────────────────────────────
WATER_COLS = [
    "iso3", "name_unsd", "year",
    "wat_bas_nat", "wat_lim_nat", "wat_none_nat",
    "wat_bas_urb", "wat_lim_urb", "wat_none_urb",
    "wat_bas_rur", "wat_lim_rur", "wat_none_rur",
]

SANITATION_COLS = [
    "iso3", "name_unsd", "year",
    "san_bas_nat", "san_lim_nat", "san_none_nat",
    "san_bas_urb", "san_lim_urb", "san_none_urb",
    "san_bas_rur", "san_lim_rur", "san_none_rur",
]

HYGIENE_COLS = [
    "iso3", "name_unsd", "year",
    "hyg_bas_nat", "hyg_lim_nat", "hyg_none_nat",
    "hyg_bas_urb", "hyg_lim_urb", "hyg_none_urb",
    "hyg_bas_rur", "hyg_lim_rur", "hyg_none_rur",
]


def _clean_sheet(df, keep_cols, label):
    """Generic cleaning applied to every WASH sheet."""
    # 1. Select relevant columns
    df = df[[c for c in keep_cols if c in df.columns]].copy()

    # 2. Use name_unsd as the country name (always populated) and rename
    df = df.rename(columns={"name_unsd": "country"})

    # 3. Drop rows with no country or iso3
    df = df.dropna(subset=["country", "iso3"])

    # 4. Strip whitespace from text columns
    df["country"] = df["country"].str.strip()
    df["iso3"] = df["iso3"].str.strip()

    # 5. Convert numeric columns — replace any placeholder strings with NaN
    num_cols = [c for c in df.columns if c not in ("iso3", "country", "year")]
    df[num_cols] = df[num_cols].replace(["-", "–", "−", "x", ".."], np.nan)
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    # 6. Ensure year is integer
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    # 7. Reset index
    df = df.reset_index(drop=True)

    # 8. Reorder: ISO first, then country, year, then the rest
    ordered = ["iso3", "country", "year"] + [
        c for c in df.columns if c not in ("iso3", "country", "year")
    ]
    df = df[ordered]

    print(f"  {label}: {len(df)} rows, {df.columns.tolist()}")
    return df


def clean_wash():
    print("Cleaning WASH Data ...")

    # ── Read sheets ────────────────────────────────────────────────────
    water = pd.read_excel(FILE, sheet_name="Water Data")
    sanitation = pd.read_excel(FILE, sheet_name="Sanitation Data")
    hygiene = pd.read_excel(FILE, sheet_name="Hygiene Data")
    regions = pd.read_excel(FILE, sheet_name="Regions")

    # ── Build region lookup (one row per iso3-year) ────────────────────
    regions = regions[["iso3", "year", "region_sdg"]].copy()
    regions["iso3"] = regions["iso3"].astype(str).str.strip()
    regions["region_sdg"] = regions["region_sdg"].str.strip()

    # ── Clean each sheet ───────────────────────────────────────────────
    water_clean = _clean_sheet(water, WATER_COLS, "Water")
    sanitation_clean = _clean_sheet(sanitation, SANITATION_COLS, "Sanitation")
    hygiene_clean = _clean_sheet(hygiene, HYGIENE_COLS, "Hygiene")

    # ── Merge all three on iso3 + year ─────────────────────────────────
    merged = water_clean.merge(
        sanitation_clean.drop(columns=["country"]),
        on=["iso3", "year"],
        how="outer",
    ).merge(
        hygiene_clean.drop(columns=["country"]),
        on=["iso3", "year"],
        how="outer",
    )

    # Fill country from any available side
    if "country" not in merged.columns:
        merged["country"] = np.nan
    merged["country"] = merged["country"].fillna(
        merged["iso3"].map(
            water_clean.drop_duplicates("iso3").set_index("iso3")["country"]
        )
    )

    # ── Attach SDG region ──────────────────────────────────────────────
    merged = merged.merge(regions, on=["iso3", "year"], how="left")

    # ── Reorder final columns ──────────────────────────────────────────
    front = ["iso3", "country", "region_sdg", "year"]
    rest = [c for c in merged.columns if c not in front]
    merged = merged[front + sorted(rest)]

    # ── Drop rows that have no data at all across all service columns ──
    service_cols = [c for c in merged.columns if c not in front]
    merged = merged.dropna(subset=service_cols, how="all")
    merged = merged.reset_index(drop=True)

    # ── Save ───────────────────────────────────────────────────────────
    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/wash_clean.csv"
    merged.to_csv(out_path, index=False)
    print(f"\nSaved merged WASH data -> {out_path}")
    print(f"  {len(merged)} rows x {len(merged.columns)} columns")
    print(f"  Countries: {merged['iso3'].nunique()}")
    print(f"  Years: {merged['year'].min()} - {merged['year'].max()}")


if __name__ == "__main__":
    clean_wash()
