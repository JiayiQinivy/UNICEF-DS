"""
education_cleaning.py
=============================================================
Education Data Cleaning Pipeline

Source : education-dataset.xlsx, sheet '10. Edu'
Period : Out-of-school rates 2012-2018
         Completion rates    2012-2018
         Literacy rate       2010-2018

Route B Rationale:
  Education indicators are selected to double-reinforce the
  gender inequality argument:

  (1) Female absolute indicators (oos_upsec_f, completion_primary_f,
      literacy_f) capture women's educational deprivation as a
      direct pathway to child malnutrition — educated mothers
      have better feeding practices, healthcare utilisation, and
      household bargaining power (Smith & Haddad, 2015).

  (2) Education gender gap indicators (oos_upsec_gap,
      completion_primary_gap) measure relative gender inequality
      in education, complementing the marriage gender gap already
      captured in the gender inequality dataset.

  Together these two types of indicator provide converging evidence
  that gender inequality in education — both in absolute female
  deprivation and relative female disadvantage — is associated with
  child malnutrition.

  Male indicators and learning outcome indicators are excluded:
  - Male indicators are redundant given female and gap indicators
  - Learning outcomes have low country coverage (< 90 countries)
    which would severely reduce Model 3 sample size

Output: education_clean.csv  (same directory as this script)
=============================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

# ===== USER INPUT =====
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FILE = os.path.join(SCRIPT_DIR, "data", "education-dataset.xlsx")
SHEET  = "10. Edu"
OUTPUT = "education_clean.csv"   # saved in same folder as this script
# ======================

COL_MAP = {
    1:  "country",
    # Out-of-school rates — female and male (male needed for gap derivation)
    6:  "oos_primary_m",         8:  "oos_primary_f",
    14: "oos_upsec_m",           16: "oos_upsec_f",
    # Completion rates — female and male (male needed for gap derivation)
    18: "completion_primary_m",  20: "completion_primary_f",
    # Literacy — female only (no male needed, no gap derived)
    44: "literacy_f",
}

ISO_LOOKUP = {
    "Afghanistan":"AFG","Albania":"ALB","Algeria":"DZA","Andorra":"AND",
    "Angola":"AGO","Anguilla":"AIA","Antigua and Barbuda":"ATG",
    "Argentina":"ARG","Armenia":"ARM","Australia":"AUS","Austria":"AUT",
    "Azerbaijan":"AZE","Bahamas":"BHS","Bahrain":"BHR","Bangladesh":"BGD",
    "Barbados":"BRB","Belarus":"BLR","Belgium":"BEL","Belize":"BLZ",
    "Benin":"BEN","Bhutan":"BTN","Bolivia (Plurinational State of)":"BOL",
    "Bosnia and Herzegovina":"BIH","Botswana":"BWA","Brazil":"BRA",
    "British Virgin Islands":"VGB","Brunei Darussalam":"BRN",
    "Bulgaria":"BGR","Burkina Faso":"BFA","Burundi":"BDI",
    "Cabo Verde":"CPV","Cape Verde":"CPV","Cambodia":"KHM",
    "Cameroon":"CMR","Canada":"CAN","Central African Republic":"CAF",
    "Chad":"TCD","Chile":"CHL","China":"CHN","Colombia":"COL",
    "Comoros":"COM","Congo":"COG","Democratic Republic of the Congo":"COD",
    "Cook Islands":"COK","Costa Rica":"CRI","Croatia":"HRV","Cuba":"CUB",
    "Cyprus":"CYP","Czech Republic":"CZE","Czechia":"CZE",
    "Cote d'Ivoire":"CIV","Côte d'Ivoire":"CIV",
    "Democratic People's Republic of Korea":"PRK","Denmark":"DNK",
    "Djibouti":"DJI","Dominica":"DMA","Dominican Republic":"DOM",
    "Ecuador":"ECU","Egypt":"EGY","El Salvador":"SLV",
    "Equatorial Guinea":"GNQ","Eritrea":"ERI","Estonia":"EST",
    "Eswatini":"SWZ","Swaziland":"SWZ","Ethiopia":"ETH","Fiji":"FJI",
    "Finland":"FIN","France":"FRA","Gabon":"GAB","Gambia":"GMB",
    "Georgia":"GEO","Germany":"DEU","Ghana":"GHA","Greece":"GRC",
    "Grenada":"GRD","Guatemala":"GTM","Guinea":"GIN",
    "Guinea-Bissau":"GNB","Guinea Bissau":"GNB","Guyana":"GUY",
    "Haiti":"HTI","Holy See":"VAT","Honduras":"HND","Hungary":"HUN",
    "Iceland":"ISL","India":"IND","Indonesia":"IDN",
    "Iran (Islamic Republic of)":"IRN","Iraq":"IRQ","Ireland":"IRL",
    "Israel":"ISR","Italy":"ITA","Jamaica":"JAM","Japan":"JPN",
    "Jordan":"JOR","Kazakhstan":"KAZ","Kenya":"KEN","Kiribati":"KIR",
    "Kosovo":"XKX","Kosovo under UNSC res. 1244":"XKX","Kuwait":"KWT",
    "Kyrgyzstan":"KGZ","Lao People's Democratic Republic":"LAO",
    "Latvia":"LVA","Lebanon":"LBN","Lesotho":"LSO","Liberia":"LBR",
    "Libya":"LBY","Liechtenstein":"LIE","Lithuania":"LTU",
    "Luxembourg":"LUX","Madagascar":"MDG","Malawi":"MWI",
    "Malaysia":"MYS","Maldives":"MDV","Mali":"MLI","Malta":"MLT",
    "Marshall Islands":"MHL","Mauritania":"MRT","Mauritius":"MUS",
    "Mexico":"MEX","Micronesia (Federated States of)":"FSM",
    "Republic of Moldova":"MDA","Moldova":"MDA","Monaco":"MCO",
    "Mongolia":"MNG","Montenegro":"MNE","Montserrat":"MSR",
    "Morocco":"MAR","Mozambique":"MOZ","Myanmar":"MMR","Namibia":"NAM",
    "Nauru":"NRU","Nepal":"NPL","Netherlands":"NLD",
    "Netherlands (Kingdom of the)":"NLD","New Zealand":"NZL",
    "Nicaragua":"NIC","Niger":"NER","Nigeria":"NGA","Niue":"NIU",
    "North Macedonia":"MKD","Norway":"NOR","Oman":"OMN","Pakistan":"PAK",
    "Palau":"PLW","Palestine":"PSE","State of Palestine":"PSE",
    "Panama":"PAN","Papua New Guinea":"PNG","Paraguay":"PRY",
    "Peru":"PER","Philippines":"PHL","Poland":"POL","Portugal":"PRT",
    "Qatar":"QAT","Republic of Korea":"KOR","Romania":"ROU",
    "Russian Federation":"RUS","Rwanda":"RWA",
    "Saint Kitts and Nevis":"KNA","Saint Lucia":"LCA",
    "Saint Vincent and the Grenadines":"VCT","Samoa":"WSM",
    "San Marino":"SMR","Sao Tome and Principe":"STP",
    "Saudi Arabia":"SAU","Senegal":"SEN","Serbia":"SRB",
    "Seychelles":"SYC","Sierra Leone":"SLE","Singapore":"SGP",
    "Slovakia":"SVK","Slovenia":"SVN","Solomon Islands":"SLB",
    "Somalia":"SOM","South Africa":"ZAF","South Sudan":"SSD",
    "Spain":"ESP","Sri Lanka":"LKA","Sudan":"SDN","Suriname":"SUR",
    "Sweden":"SWE","Switzerland":"CHE","Syrian Arab Republic":"SYR",
    "Tajikistan":"TJK","United Republic of Tanzania":"TZA",
    "Thailand":"THA","Timor-Leste":"TLS","Togo":"TGO","Tokelau":"TKL",
    "Tonga":"TON","Trinidad and Tobago":"TTO","Tunisia":"TUN",
    "Turkey":"TUR","Türkiye":"TUR","Turks and Caicos Islands":"TCA",
    "Turkmenistan":"TKM","Tuvalu":"TUV","Uganda":"UGA","Ukraine":"UKR",
    "United Arab Emirates":"ARE","United Kingdom":"GBR",
    "United Kingdom of Great Britain and Northern Ireland":"GBR",
    "United States of America":"USA","United States":"USA",
    "Uruguay":"URY","Uzbekistan":"UZB","Vanuatu":"VUT",
    "Venezuela (Bolivarian Republic of)":"VEN","Viet Nam":"VNM",
    "Wallis and Futuna Islands":"WLF","Yemen":"YEM",
    "Zambia":"ZMB","Zimbabwe":"ZWE",
}

NON_COUNTRY_PATTERNS = (
    "SUMMARY|East Asia|Europe and Central|Eastern Europe|Western Europe|"
    "Latin America|Middle East|North America|South Asia|Sub-Saharan|"
    "Eastern and Southern|West and Central|Least developed|World|"
    "NOTES|DEFINITIONS|MAIN DATA|Out of school rate|Completion rate|"
    "Proportion of children|Youth literacy|Data refer|advisable|"
    "International Standard|UNESCO|Demographic and Health|"
    "United Nations Statistics|nan"
)


def clean_education():
    print("=" * 60)
    print("EDUCATION DATA CLEANING PIPELINE")
    print("=" * 60)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path  = os.path.join(script_dir, FILE)
    out_path   = os.path.join(script_dir, OUTPUT)

    print(f"\n  Reading: {file_path}")
    if not os.path.exists(file_path):
        print(f"  File not found: {file_path}")
        return

    raw = pd.read_excel(file_path, sheet_name=SHEET, header=None)
    print(f"  Raw shape: {raw.shape}")

    # ── Step 1: Extract required columns ──────────────────────
    data = raw.iloc[8:, list(COL_MAP.keys())].copy()
    data.columns = list(COL_MAP.values())

    # ── Step 2: Clean country column ──────────────────────────
    data["country"] = data["country"].astype(str).str.strip()

    # ── Step 3: Remove non-country rows ───────────────────────
    data = data[data["country"].str.len() <= 60]
    data = data[~data["country"].str.contains(
        NON_COUNTRY_PATTERNS, case=False, na=True)]
    data = data[data["country"] != "nan"]
    data = data.reset_index(drop=True)
    print(f"  Rows after cleaning: {len(data)}")

    # ── Step 4: Convert numeric columns ───────────────────────
    num_cols = [c for c in data.columns if c != "country"]
    for col in num_cols:
        data[col] = data[col].replace(
            ["−", "-", "–", "x", "..", ""], np.nan)
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # ── Step 5: Add ISO3 codes ─────────────────────────────────
    data["ISO"] = data["country"].map(ISO_LOOKUP)
    unmatched = data[data["ISO"].isna()]["country"].dropna().unique()
    if len(unmatched) > 0:
        print(f"  Unmatched ({len(unmatched)}): {list(unmatched)}")
    data = data.dropna(subset=["ISO"]).reset_index(drop=True)

    # ── Step 6: Derive gender gap indicators ──────────────────
    # completion_primary_gap < 0: girls complete less than boys
    # oos_upsec_gap > 0: more girls out of school than boys
    data["completion_primary_gap"] = (
        data["completion_primary_f"] - data["completion_primary_m"])
    data["oos_upsec_gap"] = (
        data["oos_upsec_f"] - data["oos_upsec_m"])

    # ── Step 7: Drop male columns (no longer needed) ──────────
    # Male columns were only needed to compute gender gaps
    data = data.drop(columns=["oos_primary_m", "oos_upsec_m",
                               "completion_primary_m"])

    # ── Step 8: Reorder final columns ─────────────────────────
    # Female absolute indicators first, then gender gap indicators
    final_cols = [
        "ISO", "country",
        # Female absolute indicators
        "oos_primary_f",          # Female primary OOS rate
        "oos_upsec_f",            # Female upper-secondary OOS rate
        "completion_primary_f",   # Female primary completion rate
        "literacy_f",             # Female youth literacy rate
        # Education gender gap indicators
        "completion_primary_gap", # Female minus male completion rate
        "oos_upsec_gap",          # Female minus male upper-sec OOS rate
    ]
    data = data[final_cols]

    # ── Coverage report ────────────────────────────────────────
    print(f"\n  {'─'*50}")
    print(f"  COVERAGE SUMMARY")
    print(f"  {'─'*50}")
    print(f"  Total countries: {len(data)}")

    print(f"\n  Female absolute indicators:")
    for col in ["oos_primary_f", "oos_upsec_f",
                "completion_primary_f", "literacy_f"]:
        print(f"    {col:30s}: {data[col].notna().sum():3d} countries")

    print(f"\n  Gender gap indicators (derived):")
    for col in ["completion_primary_gap", "oos_upsec_gap"]:
        d = data[col].dropna()
        print(f"    {col:30s}: {len(d):3d} countries  "
              f"(mean={d.mean():+.2f}, min={d.min():.2f}, "
              f"max={d.max():.2f})")

    # ── Value range check ──────────────────────────────────────
    print(f"\n  Value range check:")
    all_ok = True
    for col in ["oos_primary_f", "oos_upsec_f",
                "completion_primary_f", "literacy_f"]:
        d = data[col].dropna()
        if len(d) == 0:
            continue
        out = d[(d < 0) | (d > 100)]
        if len(out) > 0:
            print(f"    {col}: {len(out)} values out of [0,100]")
            all_ok = False
    if all_ok:
        print(f"    All female indicators within [0,100]")

    # ── Missing values ─────────────────────────────────────────
    print(f"\n  Missing values:")
    miss = data.isna().sum()
    for col, n in miss[miss > 0].items():
        if col not in ("ISO", "country"):
            print(f"    {col:30s}: {n}")

    # ── Descriptive statistics ─────────────────────────────────
    print(f"\n  Descriptive statistics:")
    print(data[["oos_primary_f", "oos_upsec_f",
                "completion_primary_f", "literacy_f",
                "completion_primary_gap",
                "oos_upsec_gap"]].describe().round(2).to_string())

    # ── Save ───────────────────────────────────────────────────
    data.to_csv(out_path, index=False)
    print(f"\n  Saved -> {out_path}")
    print(f"  Shape: {data.shape}")
    print(f"  Columns: {list(data.columns)}")
    print(f"\nPIPELINE COMPLETE")


if __name__ == "__main__":
    clean_education()