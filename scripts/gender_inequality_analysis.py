"""
gender_inequality_analysis.py
=============================================================
Gender Inequality Data Cleaning and Analysis Pipeline

Reads three existing files from the UNICEF-DS root folder:
  1. Adolescent_Long_clean.csv  -- already cleaned, has ISO column
  2. Child_marriage.csv         -- raw, needs cleaning
  3. FGM_clean_english.csv      -- already cleaned, needs ISO added

Output:
  outputs/gender_inequality_analysis.csv  -- one row per country
  outputs/gender_distributions.png
  outputs/child_marriage_gender_gap.png
  outputs/fgm_analysis.png
  outputs/adolescent_health.png
  outputs/gender_correlation_heatmap.png
=============================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
warnings.filterwarnings("ignore")

# ===== USER INPUT =====
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR      = os.path.dirname(SCRIPT_DIR)
DATA_FOLDER   = os.path.join(ROOT_DIR, "data")
OUTPUT_FOLDER = os.path.join(ROOT_DIR, "outputs")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

ADOLESCENT_FILE     = os.path.join(DATA_FOLDER, "Adolescent_Long_clean.csv")
CHILD_MARRIAGE_FILE = os.path.join(DATA_FOLDER, "Child_marriage.csv")
FGM_FILE            = os.path.join(DATA_FOLDER, "FGM_clean_english.csv")
# ======================


# ──────────────────────────────────────────────────────────────
# ISO3 LOOKUP TABLE
# ──────────────────────────────────────────────────────────────

ISO_LOOKUP = {
    "Afghanistan": "AFG", "Albania": "ALB", "Algeria": "DZA",
    "Andorra": "AND", "Angola": "AGO", "Anguilla": "AIA",
    "Antigua and Barbuda": "ATG", "Argentina": "ARG", "Armenia": "ARM",
    "Australia": "AUS", "Austria": "AUT", "Azerbaijan": "AZE",
    "Bahamas": "BHS", "Bahrain": "BHR", "Bangladesh": "BGD",
    "Barbados": "BRB", "Belarus": "BLR", "Belgium": "BEL",
    "Belize": "BLZ", "Benin": "BEN", "Bhutan": "BTN",
    "Bolivia (Plurinational State of)": "BOL", "Bolivia": "BOL",
    "Bosnia and Herzegovina": "BIH", "Botswana": "BWA", "Brazil": "BRA",
    "British Virgin Islands": "VGB", "Brunei Darussalam": "BRN",
    "Bulgaria": "BGR", "Burkina Faso": "BFA", "Burundi": "BDI",
    "Cabo Verde": "CPV", "Cape Verde": "CPV",
    "Cambodia": "KHM", "Cameroon": "CMR", "Canada": "CAN",
    "Central African Republic": "CAF", "Central African Rep.": "CAF",
    "Chad": "TCD", "Chile": "CHL", "China": "CHN",
    "Colombia": "COL", "Comoros": "COM", "Comoro Islands": "COM",
    "Congo": "COG", "Democratic Republic of the Congo": "COD",
    "Cook Islands": "COK", "Costa Rica": "CRI", "Croatia": "HRV",
    "Cuba": "CUB", "Cyprus": "CYP", "Czech Republic": "CZE",
    "Czechia": "CZE",
    "Cote d'Ivoire": "CIV", "Côte d'Ivoire": "CIV", "Ivory Coast": "CIV",
    "Democratic People's Republic of Korea": "PRK", "North Korea": "PRK",
    "Denmark": "DNK", "Djibouti": "DJI", "Dominica": "DMA",
    "Dominican Republic": "DOM", "Ecuador": "ECU", "Egypt": "EGY",
    "El Salvador": "SLV", "Equatorial Guinea": "GNQ", "Eritrea": "ERI",
    "Estonia": "EST", "Eswatini": "SWZ", "Swaziland": "SWZ",
    "Ethiopia": "ETH", "Fiji": "FJI", "Finland": "FIN", "France": "FRA",
    "Gabon": "GAB", "Gambia": "GMB", "Georgia": "GEO",
    "Germany": "DEU", "Ghana": "GHA", "Greece": "GRC",
    "Grenada": "GRD", "Guatemala": "GTM", "Guinea": "GIN",
    "Guinea-Bissau": "GNB", "Guinea Bissau": "GNB",
    "Guyana": "GUY", "Haiti": "HTI", "Holy See": "VAT",
    "Honduras": "HND", "Hungary": "HUN", "Iceland": "ISL",
    "India": "IND", "Indonesia": "IDN",
    "Iran (Islamic Republic of)": "IRN", "Iran": "IRN",
    "Iraq": "IRQ", "Ireland": "IRL", "Israel": "ISR",
    "Italy": "ITA", "Jamaica": "JAM", "Japan": "JPN",
    "Jordan": "JOR", "Kazakhstan": "KAZ", "Kenya": "KEN",
    "Kiribati": "KIR", "Kosovo": "XKX",
    "Kosovo under UNSC res. 1244": "XKX",
    "Kuwait": "KWT", "Kyrgyzstan": "KGZ",
    "Lao People's Democratic Republic": "LAO", "Laos": "LAO",
    "Latvia": "LVA", "Lebanon": "LBN", "Lesotho": "LSO",
    "Liberia": "LBR", "Libya": "LBY", "Liechtenstein": "LIE",
    "Lithuania": "LTU", "Luxembourg": "LUX",
    "Madagascar": "MDG", "Malawi": "MWI", "Malaysia": "MYS",
    "Maldives": "MDV", "Mali": "MLI", "Malta": "MLT",
    "Marshall Islands": "MHL", "Mauritania": "MRT",
    "Mauritius": "MUS", "Mexico": "MEX",
    "Micronesia (Federated States of)": "FSM",
    "Republic of Moldova": "MDA", "Moldova": "MDA",
    "Monaco": "MCO", "Mongolia": "MNG", "Montenegro": "MNE",
    "Montserrat": "MSR", "Morocco": "MAR", "Mozambique": "MOZ",
    "Myanmar": "MMR", "Namibia": "NAM", "Nauru": "NRU",
    "Nepal": "NPL", "Netherlands": "NLD",
    "Netherlands (Kingdom of the)": "NLD",
    "New Zealand": "NZL", "Nicaragua": "NIC", "Niger": "NER",
    "Nigeria": "NGA", "Niue": "NIU", "North Macedonia": "MKD",
    "Norway": "NOR", "Oman": "OMN", "Pakistan": "PAK",
    "Palau": "PLW", "Palestine": "PSE", "State of Palestine": "PSE",
    "Panama": "PAN", "Papua New Guinea": "PNG", "Paraguay": "PRY",
    "Peru": "PER", "Philippines": "PHL", "Poland": "POL",
    "Portugal": "PRT", "Qatar": "QAT",
    "Republic of Korea": "KOR", "South Korea": "KOR",
    "Romania": "ROU", "Russian Federation": "RUS", "Russia": "RUS",
    "Rwanda": "RWA", "Saint Kitts and Nevis": "KNA",
    "Saint Lucia": "LCA", "Saint Vincent and the Grenadines": "VCT",
    "Samoa": "WSM", "San Marino": "SMR",
    "Sao Tome and Principe": "STP",
    "Saudi Arabia": "SAU", "Senegal": "SEN", "Serbia": "SRB",
    "Seychelles": "SYC", "Sierra Leone": "SLE",
    "Singapore": "SGP", "Slovakia": "SVK", "Slovenia": "SVN",
    "Solomon Islands": "SLB", "Somalia": "SOM",
    "South Africa": "ZAF", "South Sudan": "SSD",
    "Spain": "ESP", "Sri Lanka": "LKA", "Sudan": "SDN",
    "Suriname": "SUR", "Sweden": "SWE", "Switzerland": "CHE",
    "Syrian Arab Republic": "SYR", "Syria": "SYR",
    "Tajikistan": "TJK",
    "United Republic of Tanzania": "TZA", "Tanzania": "TZA",
    "Thailand": "THA", "Timor-Leste": "TLS", "East Timor": "TLS",
    "Togo": "TGO", "Tokelau": "TKL", "Tonga": "TON",
    "Trinidad and Tobago": "TTO", "Tunisia": "TUN",
    "Turkey": "TUR", "Türkiye": "TUR",
    "Turks and Caicos Islands": "TCA",
    "Turkmenistan": "TKM", "Tuvalu": "TUV",
    "Uganda": "UGA", "Ukraine": "UKR",
    "United Arab Emirates": "ARE", "United Kingdom": "GBR",
    "United Kingdom of Great Britain and Northern Ireland": "GBR",
    "United States of America": "USA", "United States": "USA",
    "Uruguay": "URY", "Uzbekistan": "UZB", "Vanuatu": "VUT",
    "Venezuela (Bolivarian Republic of)": "VEN", "Venezuela": "VEN",
    "Viet Nam": "VNM", "Vietnam": "VNM",
    "Wallis and Futuna Islands": "WLF",
    "Yemen": "YEM", "Zambia": "ZMB", "Zimbabwe": "ZWE",
}


def add_iso(df, country_col):
    """Map country names to ISO3 codes and report unmatched rows."""
    df = df.copy()
    df["ISO"] = df[country_col].map(ISO_LOOKUP)
    unmatched = df[df["ISO"].isna()][country_col].dropna().unique()
    if len(unmatched) > 0:
        print(f"  Unmatched countries ({len(unmatched)}): "
              f"{', '.join(str(x) for x in unmatched[:10])}"
              f"{'...' if len(unmatched) > 10 else ''}")
    return df


# ──────────────────────────────────────────────────────────────
# SECTION 1: ADOLESCENT HEALTH INDICATORS
# Reads Adolescent_Long_clean.csv (already cleaned by notebook)
# ──────────────────────────────────────────────────────────────

def load_adolescent():
    """
    Load Adolescent_Long_clean.csv.
    Columns: ISO, Country or Area,
             ANC4_15_19_yrs(%), CPMODHS_15_19_yrs(%), ORS_15_19_yrs(%)
    """
    print(f"\n{'='*60}")
    print("LOADING ADOLESCENT HEALTH INDICATORS")
    print(f"{'='*60}")
    print(f"  Reading: {ADOLESCENT_FILE}")

    if not os.path.exists(ADOLESCENT_FILE):
        print(f"  File not found: {ADOLESCENT_FILE}")
        return pd.DataFrame()

    df = pd.read_csv(ADOLESCENT_FILE)
    print(f"  Raw shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    # Standardise column names
    df = df.rename(columns={
        "Country or Area":      "country",
        "ANC4_15_19_yrs(%)":    "anc4_15_19_pct",
        "CPMODHS_15_19_yrs(%)": "modern_contraceptive_pct",
        "ORS_15_19_yrs(%)":     "ors_usage_pct",
    })

    # ISO column already present; add if missing
    if "ISO" not in df.columns:
        print("  ISO column missing, adding via lookup")
        df = add_iso(df, "country")

    df = df.dropna(subset=["ISO"])
    print(f"  Countries: {len(df)}")
    for col in ["anc4_15_19_pct", "modern_contraceptive_pct",
                "ors_usage_pct"]:
        if col in df.columns:
            print(f"  {col:35s}: {df[col].notna().sum()} countries")

    keep = ["ISO", "country"] + [
        c for c in ["anc4_15_19_pct", "modern_contraceptive_pct",
                    "ors_usage_pct"] if c in df.columns]
    return df[keep]


# ──────────────────────────────────────────────────────────────
# SECTION 2: CHILD MARRIAGE
# Reads Child_marriage.csv (raw file, cleans here)
# ──────────────────────────────────────────────────────────────

def load_child_marriage():
    """
    Load and clean Child_marriage.csv.
    - Rename 'Countries and areas' to 'country'
    - Replace '-' with NaN, coerce numerics
    - Derive marriage_gap_18 and early_marriage_ratio
    - Add ISO3 codes
    """
    print(f"\n{'='*60}")
    print("LOADING CHILD MARRIAGE INDICATORS")
    print(f"{'='*60}")
    print(f"  Reading: {CHILD_MARRIAGE_FILE}")

    if not os.path.exists(CHILD_MARRIAGE_FILE):
        print(f"  File not found: {CHILD_MARRIAGE_FILE}")
        return pd.DataFrame()

    df = pd.read_csv(CHILD_MARRIAGE_FILE, encoding="utf-8-sig")
    print(f"  Raw shape: {df.shape}")

    df = df.rename(columns={"Countries and areas": "country"})
    df = df.replace("-", np.nan)
    df = df[df["country"].notna()].copy()

    for col in ["female_married_by_15", "female_married_by_18",
                "male_married_by_18"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["marriage_gap_18"] = (
        df["female_married_by_18"] - df["male_married_by_18"])
    df["early_marriage_ratio"] = (
        df["female_married_by_15"] / df["female_married_by_18"])

    df = add_iso(df, "country")
    df = df.dropna(subset=["ISO"])

    print(f"  Countries with ISO: {len(df)}")
    for col in ["female_married_by_18", "female_married_by_15",
                "male_married_by_18", "marriage_gap_18",
                "early_marriage_ratio"]:
        if col in df.columns:
            print(f"  {col:35s}: {df[col].notna().sum()} countries")

    keep = [c for c in ["ISO", "country",
                         "female_married_by_15", "female_married_by_18",
                         "male_married_by_18", "marriage_gap_18",
                         "early_marriage_ratio"] if c in df.columns]
    return df[keep]


# ──────────────────────────────────────────────────────────────
# SECTION 3: FGM
# Reads FGM_clean_english.csv (already cleaned by notebook)
# ──────────────────────────────────────────────────────────────

def load_fgm():
    """
    Load FGM_clean_english.csv produced by the FGM notebook.
    Columns: country, fgm_prevalence_total, fgm_urban, fgm_rural,
             fgm_wealth_poorest, fgm_wealth_second, fgm_wealth_middle,
             fgm_wealth_fourth, fgm_wealth_richest

    Renames to standardised names, adds ISO3, derives:
      fgm_wealth_gap      = poorest - richest
      fgm_urban_rural_gap = rural - urban

    MNAR structural missingness: ~183 of ~213 countries will have NaN.
    This is expected. FGM is only surveyed in Sub-Saharan Africa and
    parts of the Middle East.
    """
    print(f"\n{'='*60}")
    print("LOADING FGM INDICATORS")
    print(f"{'='*60}")
    print(f"  Reading: {FGM_FILE}")

    if not os.path.exists(FGM_FILE):
        print(f"  File not found: {FGM_FILE}")
        return pd.DataFrame()

    df = pd.read_csv(FGM_FILE)
    print(f"  Raw shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    # Remove annotation rows and regional aggregates that survived
    # the notebook cleaning (e.g. "Contact us:", "East Asia and Pacific")
    df["country"] = df["country"].astype(str).str.strip()
    exclude_patterns = (
        "Contact us|East Asia|Eastern Europe|Europe and Central|"
        "Latin America|North America|Prepared by|SUMMARY|Source:|"
        "Division of|UNICEF|Analytics|Notes:|Data refer|World|"
        "Africa|Least developed|nan"
    )
    df = df[~df["country"].str.contains(exclude_patterns,
                                        case=False, na=False)]
    df = df[df["country"].str.len() <= 60]
    print(f"  Rows after cleaning annotation rows: {len(df)}")

    # Rename from FGM notebook output to standardised names
    df = df.rename(columns={
        "fgm_prevalence_total": "fgm_prevalence_pct",
        "fgm_urban":            "fgm_urban_pct",
        "fgm_rural":            "fgm_rural_pct",
        "fgm_wealth_poorest":   "fgm_wealth_Q1_poorest",
        "fgm_wealth_second":    "fgm_wealth_Q2",
        "fgm_wealth_middle":    "fgm_wealth_Q3",
        "fgm_wealth_fourth":    "fgm_wealth_Q4",
        "fgm_wealth_richest":   "fgm_wealth_Q5_richest",
    })

    df = add_iso(df, "country")
    df = df.dropna(subset=["ISO"])

    # Derived indicators
    if ("fgm_wealth_Q1_poorest" in df.columns
            and "fgm_wealth_Q5_richest" in df.columns):
        df["fgm_wealth_gap"] = (
            df["fgm_wealth_Q1_poorest"] - df["fgm_wealth_Q5_richest"])

    if "fgm_urban_pct" in df.columns and "fgm_rural_pct" in df.columns:
        df["fgm_urban_rural_gap"] = (
            df["fgm_rural_pct"] - df["fgm_urban_pct"])

    total     = len(df)
    with_data = df["fgm_prevalence_pct"].notna().sum()
    print(f"  Total countries (incl. structural NaN): {total}")
    print(f"  Countries with FGM data: {with_data}")
    print(f"  Structural missingness (MNAR): {total - with_data}")

    keep = [c for c in ["ISO", "country",
                         "fgm_prevalence_pct",
                         "fgm_urban_pct", "fgm_rural_pct",
                         "fgm_wealth_Q1_poorest", "fgm_wealth_Q2",
                         "fgm_wealth_Q3", "fgm_wealth_Q4",
                         "fgm_wealth_Q5_richest",
                         "fgm_wealth_gap", "fgm_urban_rural_gap"]
            if c in df.columns]
    return df[keep]


# ──────────────────────────────────────────────────────────────
# SECTION 4: BUILD GENDER MASTER DATASET
# ──────────────────────────────────────────────────────────────

def build_gender_master(adolescent_df, child_marriage_df, fgm_df):
    print(f"\n{'=' * 60}")
    print("BUILDING GENDER INEQUALITY MASTER DATASET")
    print(f"{'=' * 60}")

    # child_marriage是base，直接用，不再重复合并
    master = child_marriage_df.copy()
    print(f"  Base (child_marriage): {len(master)} countries")

    shared_meta = {"country"}

    def safe_merge(master, df, name, how="outer"):
        if df is None or df.empty:
            print(f"  {name} is empty, skipped")
            return master
        join_cols = ["ISO"] + [c for c in df.columns
                                if c not in shared_meta]
        join_cols = list(dict.fromkeys(join_cols))
        result = master.merge(df[join_cols], on="ISO", how=how)
        print(f"  + {name:15s} ({how}): master now {len(result)} countries")
        return result

    # 只merge adolescent和fgm，不再merge child_marriage
    master = safe_merge(master, adolescent_df, "adolescent", how="outer")
    master = safe_merge(master, fgm_df, "fgm", how="left")

    # Resolve duplicate country columns
    if "country_x" in master.columns or "country_y" in master.columns:
        col_x = master.get("country_x", pd.Series(dtype=str))
        col_y = master.get("country_y", pd.Series(dtype=str))
        master["country"] = col_x.fillna(col_y)
        master = master.drop(
            columns=[c for c in ["country_x", "country_y"]
                        if c in master.columns])


    # Coverage report
    all_indicators = [
        "female_married_by_18", "female_married_by_15",
        "male_married_by_18",   "marriage_gap_18",
        "early_marriage_ratio",
        "anc4_15_19_pct",       "modern_contraceptive_pct",
        "ors_usage_pct",
        "fgm_prevalence_pct",   "fgm_wealth_gap",
        "fgm_urban_rural_gap",
    ]
    print(f"\n  {'─'*50}")
    print(f"  COVERAGE SUMMARY")
    print(f"  {'─'*50}")
    print(f"  Total countries: {len(master)}")
    for col in all_indicators:
        if col in master.columns:
            print(f"  {col:40s}: {master[col].notna().sum():3d}")

    avail = [c for c in all_indicators if c in master.columns]
    has_any = master[avail].notna().any(axis=1).sum()
    print(f"\n  Countries with >= 1 gender indicator: {has_any}")

    return master


# ──────────────────────────────────────────────────────────────
# SECTION 5: VISUALISATIONS
# ──────────────────────────────────────────────────────────────

def plot_distributions(master):
    """Distribution + KDE + Shapiro-Wilk for all core indicators."""
    print("\n  Generating distribution plots...")
    from scipy.stats import gaussian_kde

    indicators = [
        ("female_married_by_18",     "Female Child Marriage by 18 (%)"),
        ("female_married_by_15",     "Female Child Marriage by 15 (%)"),
        ("marriage_gap_18",          "Marriage Gender Gap (F - M, pp)"),
        ("anc4_15_19_pct",           "Antenatal Care 4+ visits, 15-19 yrs (%)"),
        ("modern_contraceptive_pct", "Modern Contraceptive Use, 15-19 yrs (%)"),
        ("fgm_prevalence_pct",       "FGM Prevalence (%)"),
    ]
    panels = [(c, l) for c, l in indicators
              if c in master.columns
              and master[c].notna().sum() > 5]
    if not panels:
        return

    n_cols = 3
    n_rows = (len(panels) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(6 * n_cols, 5 * n_rows))
    axes = axes.flatten()

    for i, (col, label) in enumerate(panels):
        ax   = axes[i]
        data = master[col].dropna()
        ax.hist(data, bins=20, color="#4e79a7", alpha=0.70,
                edgecolor="white", density=True)
        kde = gaussian_kde(data)
        x   = np.linspace(data.min(), data.max(), 200)
        ax.plot(x, kde(x), color="#e15759", lw=2, label="KDE")

        if len(data) <= 5000:
            w, p   = stats.shapiro(data)
            normal = p > 0.05
            clr    = "#2ca02c" if normal else "#d62728"
            txt    = (f"Shapiro-Wilk\nW={w:.3f}, p={p:.4f}\n"
                      f"{'Normal' if normal else 'Non-normal'}")
            ax.annotate(txt, xy=(0.97, 0.95), xycoords="axes fraction",
                        fontsize=8, ha="right", va="top", color=clr,
                        bbox=dict(boxstyle="round,pad=0.3",
                                   fc="white", alpha=0.85))
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xlabel("Value (%)", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.set_xlim(left=0)
        ax.legend(fontsize=8)

    for j in range(len(panels), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Distribution of Gender Inequality Indicators\n"
        "(Shapiro-Wilk test guides parametric vs non-parametric choice)",
        fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = os.path.join(OUTPUT_FOLDER, "gender_distributions.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: gender_distributions.png")


def plot_child_marriage_gender_gap(master):
    """Boxplot female vs male + Top 15 countries by gender gap."""
    print("\n  Generating child marriage gender gap plots...")
    sns.set_theme(style="whitegrid", palette="pastel")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    need = [c for c in ["female_married_by_18", "male_married_by_18"]
            if c in master.columns]
    if need:
        df_melted = master.dropna(subset=need, how="all").melt(
            id_vars=["ISO"], value_vars=need,
            var_name="Sex", value_name="Marriage_Rate"
        ).dropna(subset=["Marriage_Rate"])
        df_melted["Sex"] = df_melted["Sex"].replace({
            "female_married_by_18": "Female (by 18)",
            "male_married_by_18":   "Male (by 18)",
        })
        sns.boxplot(data=df_melted, x="Sex", y="Marriage_Rate",
                    hue="Sex", legend=False, width=0.4,
                    ax=axes[0], palette=["#e15759", "#4e79a7"])
        axes[0].set_title("Distribution of Child Marriage Rates by Sex",
                           fontsize=13, fontweight="bold")
        axes[0].set_ylabel("Percentage (%)", fontsize=10)
        axes[0].set_xlabel("")

    if "marriage_gap_18" in master.columns:
        top15 = (master.dropna(subset=["marriage_gap_18"])
                       .nlargest(15, "marriage_gap_18"))
        sns.barplot(data=top15, x="marriage_gap_18", y="country",
                    hue="country", legend=False,
                    ax=axes[1], palette="Reds_r")
        axes[1].set_title(
            "Top 15 Countries: Marriage Gender Gap (Female - Male, by 18)",
            fontsize=13, fontweight="bold")
        axes[1].set_xlabel("Gender Gap (percentage points)", fontsize=10)
        axes[1].set_ylabel("")

    plt.tight_layout()
    out = os.path.join(OUTPUT_FOLDER, "child_marriage_gender_gap.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: child_marriage_gender_gap.png")


def plot_fgm_analysis(master):
    """Top 15 FGM countries + wealth quintile gradient."""
    print("\n  Generating FGM analysis plots...")
    if "fgm_prevalence_pct" not in master.columns:
        return
    df_fgm = master.dropna(subset=["fgm_prevalence_pct"]).copy()
    if df_fgm.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    top15 = df_fgm.nlargest(15, "fgm_prevalence_pct")
    axes[0].barh(top15["country"], top15["fgm_prevalence_pct"],
                 color="#e15759", alpha=0.85)
    axes[0].invert_yaxis()
    axes[0].set_title("Top 15 Countries by FGM Prevalence",
                       fontsize=13, fontweight="bold")
    axes[0].set_xlabel("FGM Prevalence (%)", fontsize=10)
    axes[0].set_xlim(0, 105)
    for i, (_, row) in enumerate(top15.iterrows()):
        axes[0].text(row["fgm_prevalence_pct"] + 0.5, i,
                     f"{row['fgm_prevalence_pct']:.1f}%",
                     va="center", fontsize=8)

    wealth_cols = [c for c in ["fgm_wealth_Q1_poorest", "fgm_wealth_Q2",
                                "fgm_wealth_Q3", "fgm_wealth_Q4",
                                "fgm_wealth_Q5_richest"]
                   if c in df_fgm.columns]
    if wealth_cols:
        top10   = df_fgm.nlargest(10, "fgm_prevalence_pct")
        x_ticks = ["Q1\n(Poorest)", "Q2", "Q3", "Q4",
                   "Q5\n(Richest)"][:len(wealth_cols)]
        palette = plt.cm.get_cmap("tab10", len(top10))
        for i, (_, row) in enumerate(top10.iterrows()):
            vals = [row[c] for c in wealth_cols]
            axes[1].plot(range(len(wealth_cols)), vals,
                          marker="o", lw=1.8, label=row["country"],
                          color=palette(i), alpha=0.85)
        axes[1].set_xticks(range(len(wealth_cols)))
        axes[1].set_xticklabels(x_ticks, fontsize=9)
        axes[1].set_title(
            "FGM Prevalence by Wealth Quintile (Top 10 countries)",
            fontsize=13, fontweight="bold")
        axes[1].set_ylabel("FGM Prevalence (%)", fontsize=10)
        axes[1].set_ylim(bottom=0)
        axes[1].legend(fontsize=7, loc="upper right",
                        title="Country", title_fontsize=8)
        axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(OUTPUT_FOLDER, "fgm_analysis.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fgm_analysis.png")


def plot_adolescent_health(master):
    """Boxplot of adolescent indicators + ANC4 vs contraceptive scatter."""
    print("\n  Generating adolescent health plots...")
    adol_cols = [c for c in ["anc4_15_19_pct",
                              "modern_contraceptive_pct",
                              "ors_usage_pct"]
                 if c in master.columns]
    if not adol_cols:
        return

    labels = {
        "anc4_15_19_pct":          "Antenatal Care 4+",
        "modern_contraceptive_pct": "Modern Contraceptive",
        "ors_usage_pct":           "ORS Usage",
    }
    sns.set_theme(style="whitegrid", palette="muted")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    df_melted = master.dropna(subset=adol_cols, how="all").melt(
        id_vars=["ISO"], value_vars=adol_cols,
        var_name="Indicator", value_name="Value"
    ).dropna(subset=["Value"])
    df_melted["Indicator"] = df_melted["Indicator"].map(labels)
    sns.boxplot(data=df_melted, y="Indicator", x="Value",
                orient="h", width=0.5, ax=axes[0], palette="Set2")
    axes[0].set_title(
        "Adolescent Health Indicators (15-19 yrs)\nDistribution across Countries",
        fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Percentage (%)", fontsize=10)
    axes[0].set_ylabel("")

    if ("anc4_15_19_pct" in master.columns
            and "modern_contraceptive_pct" in master.columns):
        df_sc = master.dropna(
            subset=["anc4_15_19_pct", "modern_contraceptive_pct"])
        axes[1].scatter(df_sc["anc4_15_19_pct"],
                         df_sc["modern_contraceptive_pct"],
                         color="#4e79a7", alpha=0.7, s=55,
                         edgecolors="white", linewidth=0.5)
        if len(df_sc) > 3:
            m, b, r, p, _ = stats.linregress(
                df_sc["anc4_15_19_pct"],
                df_sc["modern_contraceptive_pct"])
            x_line = np.linspace(df_sc["anc4_15_19_pct"].min(),
                                  df_sc["anc4_15_19_pct"].max(), 100)
            axes[1].plot(x_line, m * x_line + b, color="#e15759",
                          lw=1.8, label=f"r={r:.2f}, p={p:.3f}")
            axes[1].legend(fontsize=9)
        axes[1].set_xlabel("Antenatal Care 4+ visits (%)", fontsize=10)
        axes[1].set_ylabel("Modern Contraceptive Use (%)", fontsize=10)
        axes[1].set_title(
            "Antenatal Care vs Modern Contraceptive Use (15-19 yrs)",
            fontsize=12, fontweight="bold")
        axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(OUTPUT_FOLDER, "adolescent_health.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: adolescent_health.png")


def plot_correlation_heatmap(master):
    """Spearman correlation heatmap of all gender indicators."""
    print("\n  Generating correlation heatmap...")
    indicator_cols = [
        "female_married_by_18", "female_married_by_15",
        "male_married_by_18",   "marriage_gap_18",
        "early_marriage_ratio",
        "anc4_15_19_pct",       "modern_contraceptive_pct",
        "ors_usage_pct",
        "fgm_prevalence_pct",   "fgm_wealth_gap",
        "fgm_urban_rural_gap",
    ]
    avail = [c for c in indicator_cols if c in master.columns]
    if len(avail) < 2:
        return

    corr = master[avail].corr(method="spearman")
    short = {
        "female_married_by_18":    "F.Married.18",
        "female_married_by_15":    "F.Married.15",
        "male_married_by_18":      "M.Married.18",
        "marriage_gap_18":         "MarriageGap",
        "early_marriage_ratio":    "EarlyRatio",
        "anc4_15_19_pct":          "ANC4",
        "modern_contraceptive_pct":"Contraceptive",
        "ors_usage_pct":           "ORS",
        "fgm_prevalence_pct":      "FGM",
        "fgm_wealth_gap":          "FGM.WealthGap",
        "fgm_urban_rural_gap":     "FGM.UrbanRural",
    }
    corr = corr.rename(index=short, columns=short)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, ax=ax, annot=True, fmt=".2f",
                annot_kws={"size": 8}, cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, square=True,
                linewidths=0.5, mask=mask,
                cbar_kws={"shrink": 0.8, "label": "Spearman r"})
    ax.set_title(
        "Spearman Correlation Matrix — Gender Inequality Indicators\n"
        "(Lower triangle; pairwise complete cases)",
        fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    out = os.path.join(OUTPUT_FOLDER, "gender_correlation_heatmap.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: gender_correlation_heatmap.png")


# ──────────────────────────────────────────────────────────────
# SECTION 6: SUMMARY STATISTICS
# ──────────────────────────────────────────────────────────────

def print_summary(master):
    print(f"\n{'='*60}")
    print("DESCRIPTIVE STATISTICS")
    print(f"{'='*60}")
    cols = [
        "female_married_by_18", "female_married_by_15",
        "male_married_by_18",   "marriage_gap_18",
        "early_marriage_ratio",
        "anc4_15_19_pct",       "modern_contraceptive_pct",
        "ors_usage_pct",        "fgm_prevalence_pct",
    ]
    avail = [c for c in cols if c in master.columns]
    print(master[avail].describe().round(2).to_string())

    print(f"\n  Output files:")
    for f in sorted(os.listdir(OUTPUT_FOLDER)):
        if f.endswith((".csv", ".png")):
            sz = os.path.getsize(os.path.join(OUTPUT_FOLDER, f))
            print(f"  {f:55s} {sz/1024:6.1f} KB")
    print("=" * 60)


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("GENDER INEQUALITY ANALYSIS PIPELINE")
    print("=" * 60)

    adolescent_df     = load_adolescent()
    child_marriage_df = load_child_marriage()
    fgm_df            = load_fgm()

    master = build_gender_master(
        adolescent_df, child_marriage_df, fgm_df)

    if master.empty:
        print("Gender master is empty. Check file paths.")
        return

    out_path = os.path.join(OUTPUT_FOLDER,
                             "gender_inequality_analysis.csv")
    master.to_csv(out_path, index=False)
    print(f"\n  Final CSV saved -> {out_path}")
    print(f"  Shape: {master.shape}")
    print(f"  Columns: {list(master.columns)}")

    plot_distributions(master)
    plot_child_marriage_gender_gap(master)
    plot_fgm_analysis(master)
    plot_adolescent_health(master)
    plot_correlation_heatmap(master)

    print_summary(master)
    print("\nPIPELINE COMPLETE")


if __name__ == "__main__":
    main()