import os
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR   = os.path.dirname(_SCRIPT_DIR)

# ====================================================================
# FUNCTIONS COPIED FROM Malnutrition Datasets/Semi-Final.py
# ====================================================================
COUNTRY_COL = "CountryName"
YEAR_COL = "CMRS_year*"
NATIONAL_COL = "National_r"

HEADER_KEYWORDS = ["ISO3Code", "CountryName", "UNICEF_Reporting_Sub_Region", "ISO Code", "ISO"]

def detect_header(df):
    for i in range(min(20, len(df))):
        row = df.iloc[i]
        row_str = [str(v).lower() if pd.notna(v) else "" for v in row]
        if any(any(k.lower() in v for k in HEADER_KEYWORDS) for v in row_str):
            return i
    return None

def extract_target_columns(df):
    header_row = detect_header(df)
    if header_row is None:
        return pd.DataFrame()

    df.columns = df.iloc[header_row]
    df = df.iloc[header_row + 1:]

    df.columns = df.columns.astype(str).str.strip()
    df = df.loc[:, ~df.columns.duplicated()]
    
    # Handle variations in Country column names across datasets
    country_col_name = None
    for possible_name in ["CountryName", "Countries and areas", "Country"]:
        if possible_name in df.columns:
            country_col_name = possible_name
            break
            
    if not country_col_name:
        return pd.DataFrame()

    # Anchor rows to valid country entries
    country_col = df[country_col_name]
    valid_idx = country_col[country_col.notna()].index
    df = df.loc[valid_idx]
    
    # Rename country column to standard COUNTRY_COL
    if country_col_name != COUNTRY_COL:
        df = df.rename(columns={country_col_name: COUNTRY_COL})

    # Find the year column
    year_col_name = None
    for possible_name in ["CMRS_year*", "Year", "Year(s) of data collection"]:
        if possible_name in df.columns:
            year_col_name = possible_name
            break
            
    if year_col_name and year_col_name != YEAR_COL:
         df = df.rename(columns={year_col_name: YEAR_COL})

    # Find National value column
    # In the new child health database, the national value is usually a specific column or the last numeric column
    # For now, let's grab the standard target columns if they exist
    cols_to_keep = [COUNTRY_COL]
    if YEAR_COL in df.columns:
        cols_to_keep.append(YEAR_COL)
    
    # If National_r exists, grab it. Otherwise, we'll need custom extraction logic per file later.
    if NATIONAL_COL in df.columns:
        cols_to_keep.append(NATIONAL_COL)
        
    # Return all columns for now so we can extract custom values if needed
    return df

def clean_data(df, target_val_col):
    df = df.dropna(subset=[COUNTRY_COL])

    col_data = df[target_val_col]
    if isinstance(col_data, pd.DataFrame):
        col_data = col_data.iloc[:, 0]
    if not isinstance(col_data, pd.Series):
        col_data = pd.Series(col_data)

    df[target_val_col] = pd.to_numeric(col_data, errors="coerce")

    if YEAR_COL in df.columns:
        df = df.sort_values(YEAR_COL).groupby(COUNTRY_COL).tail(1)

    return df[[COUNTRY_COL, target_val_col]]

# ====================================================================
# CUSTOM EXTRACTION LOGIC
# ====================================================================

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
    "Brunei Darussalam": "BRN", "Bulgaria": "BGR", "Burkina Faso": "BFA",
    "Burundi": "BDI", "Cabo Verde": "CPV", "Cambodia": "KHM",
    "Cameroon": "CMR", "Canada": "CAN", "Central African Republic": "CAF",
    "Chad": "TCD", "Chile": "CHL", "China": "CHN", "Colombia": "COL",
    "Comoros": "COM", "Congo": "COG", "Democratic Republic of the Congo": "COD",
    "Costa Rica": "CRI", "Croatia": "HRV", "Cuba": "CUB", "Cyprus": "CYP",
    "Czechia": "CZE", "Côte d'Ivoire": "CIV",
    "Democratic People's Republic of Korea": "PRK", "Denmark": "DNK",
    "Djibouti": "DJI", "Dominica": "DMA", "Dominican Republic": "DOM",
    "Ecuador": "ECU", "Egypt": "EGY", "El Salvador": "SLV",
    "Equatorial Guinea": "GNQ", "Eritrea": "ERI", "Estonia": "EST",
    "Eswatini": "SWZ", "Ethiopia": "ETH", "Fiji": "FJI", "Finland": "FIN",
    "France": "FRA", "Gabon": "GAB", "Gambia": "GMB", "Georgia": "GEO",
    "Germany": "DEU", "Ghana": "GHA", "Greece": "GRC", "Grenada": "GRD",
    "Guatemala": "GTM", "Guinea": "GIN", "Guinea-Bissau": "GNB",
    "Guyana": "GUY", "Haiti": "HTI", "Honduras": "HND", "Hungary": "HUN",
    "Iceland": "ISL", "India": "IND", "Indonesia": "IDN",
    "Iran (Islamic Republic of)": "IRN", "Iraq": "IRQ", "Ireland": "IRL",
    "Israel": "ISR", "Italy": "ITA", "Jamaica": "JAM", "Japan": "JPN",
    "Jordan": "JOR", "Kazakhstan": "KAZ", "Kenya": "KEN", "Kiribati": "KIR",
    "Kuwait": "KWT", "Kyrgyzstan": "KGZ",
    "Lao People's Democratic Republic": "LAO", "Latvia": "LVA",
    "Lebanon": "LBN", "Lesotho": "LSO", "Liberia": "LBR", "Libya": "LBY",
    "Lithuania": "LTU", "Luxembourg": "LUX", "Madagascar": "MDG",
    "Malawi": "MWI", "Malaysia": "MYS", "Maldives": "MDV", "Mali": "MLI",
    "Malta": "MLT", "Marshall Islands": "MHL", "Mauritania": "MRT",
    "Mauritius": "MUS", "Mexico": "MEX",
    "Micronesia (Federated States of)": "FSM", "Republic of Moldova": "MDA",
    "Monaco": "MCO", "Mongolia": "MNG", "Montenegro": "MNE",
    "Morocco": "MAR", "Mozambique": "MOZ", "Myanmar": "MMR", "Namibia": "NAM",
    "Nauru": "NRU", "Nepal": "NPL", "Netherlands": "NLD", "New Zealand": "NZL",
    "Nicaragua": "NIC", "Niger": "NER", "Nigeria": "NGA", "Niue": "NIU",
    "North Macedonia": "MKD", "Norway": "NOR", "Oman": "OMN", "Pakistan": "PAK",
    "Palau": "PLW", "Palestine": "PSE", "Panama": "PAN",
    "Papua New Guinea": "PNG", "Paraguay": "PRY", "Peru": "PER",
    "Philippines": "PHL", "Poland": "POL", "Portugal": "PRT", "Qatar": "QAT",
    "Republic of Korea": "KOR", "Romania": "ROU", "Russian Federation": "RUS",
    "Rwanda": "RWA", "Saint Kitts and Nevis": "KNA", "Saint Lucia": "LCA",
    "Saint Vincent and the Grenadines": "VCT", "Samoa": "WSM",
    "Sao Tome and Principe": "STP", "Saudi Arabia": "SAU", "Senegal": "SEN",
    "Serbia": "SRB", "Seychelles": "SYC", "Sierra Leone": "SLE",
    "Singapore": "SGP", "Slovakia": "SVK", "Slovenia": "SVN",
    "Solomon Islands": "SLB", "Somalia": "SOM", "South Africa": "ZAF",
    "South Sudan": "SSD", "Spain": "ESP", "Sri Lanka": "LKA", "Sudan": "SDN",
    "Suriname": "SUR", "Sweden": "SWE", "Switzerland": "CHE",
    "Syrian Arab Republic": "SYR", "Tajikistan": "TJK",
    "United Republic of Tanzania": "TZA", "Thailand": "THA",
    "Timor-Leste": "TLS", "Togo": "TGO", "Tonga": "TON",
    "Trinidad and Tobago": "TTO", "Tunisia": "TUN", "Türkiye": "TUR",
    "Turkmenistan": "TKM", "Tuvalu": "TUV", "Uganda": "UGA", "Ukraine": "UKR",
    "United Arab Emirates": "ARE", "United Kingdom": "GBR",
    "United States of America": "USA", "Uruguay": "URY", "Uzbekistan": "UZB",
    "Vanuatu": "VUT", "Venezuela (Bolivarian Republic of)": "VEN",
    "Viet Nam": "VNM", "Yemen": "YEM", "Zambia": "ZMB", "Zimbabwe": "ZWE",
}

def get_iso(country_name):
    return ISO_LOOKUP.get(country_name.strip(), np.nan)

def extract_child_health_db():
    file_path = os.path.join(_ROOT_DIR, "data", "Child-Health-Coverage-Database-November-2025.xlsx")
    
    # We want DIARCARE, PNEUCARE, ITN
    targets = {
        "DIARCARE": "diarrhoea_care_pct",
        "PNEUCARE": "pneumonia_care_pct",
        "ITN": "itn_use_pct"
    }
    
    results = []
    
    for sheet, new_col_name in targets.items():
        df = pd.read_excel(file_path, sheet_name=sheet, header=None)
        df = extract_target_columns(df)
        
        if df.empty:
            continue
            
        # In this specific DB, the 8th column (index 8) is the National Total value
        # The columns are usually: ISO, Country, Region, ProgRegion, Income, Year, Source, SourceLong, Total, Male, Female...
        target_val_col = df.columns[8] 
        
        df_clean = clean_data(df, target_val_col)
        df_clean = df_clean.rename(columns={target_val_col: new_col_name})
        results.append(df_clean)
        
    return results


def extract_malnutrition_datasets():
    results = []
    
    # 1. Exclusive Breastfeeding
    ebf_df = pd.read_excel(os.path.join(_ROOT_DIR, "data", "Malnutrition Datasets", "Breastfeeding", "UNICEF_Expanded_Global_Databases_ExclusiveBF_2025.xlsx"), sheet_name="Exclusive Breastfeeding", header=None)
    ebf_idx = detect_header(ebf_df)
    if ebf_idx is not None:
        ebf_df.columns = ebf_df.iloc[ebf_idx]
        ebf_df = ebf_df.iloc[ebf_idx + 1:]
        # Find the National point estimate. Usually col 10 or specifically named "National_Point Estimate"
        target_col = [c for c in ebf_df.columns if str(c).lower().startswith("national")]
        if target_col:
            target_col = target_col[0]
        else:
            target_col = ebf_df.columns[10] # Fallback
            
        ebf_clean = ebf_df.dropna(subset=["Countries and areas", target_col])
        ebf_clean["CountryName"] = ebf_clean["Countries and areas"]
        
        # Get latest year
        year_col = [c for c in ebf_clean.columns if "Year*" in str(c)]
        if year_col:
            ebf_clean["Year"] = pd.to_numeric(ebf_clean[year_col[0]], errors="coerce")
            ebf_clean = ebf_clean.sort_values("Year").groupby("CountryName").tail(1)
            
        ebf_clean["exclusive_bf_pct"] = pd.to_numeric(ebf_clean[target_col], errors="coerce")
        results.append(ebf_clean[["CountryName", "exclusive_bf_pct"]])
        
    # 2. Low Birthweight
    lbw_df = pd.read_excel(os.path.join(_ROOT_DIR, "data", "Malnutrition Datasets", "Birthweight", "UNICEF-WHO-LBW-estimates-2023.xlsx"), sheet_name="country prevalence", header=8)
    if "ISO3" in lbw_df.columns and "Estimate" in lbw_df.columns:
        # The Estimate column specifies "Point Estimate", "Lower Bound", "Upper Bound"
        # We want the Point Estimate for the latest year (2020)
        lbw_clean = lbw_df[lbw_df["Estimate"] == "LBW estimate"].copy()
        if 2020 in lbw_clean.columns:
            lbw_clean = lbw_clean.dropna(subset=["Countries and areas", 2020])
            lbw_clean["CountryName"] = lbw_clean["Countries and areas"]
            lbw_clean["low_birthweight_pct"] = pd.to_numeric(lbw_clean[2020], errors="coerce")
            results.append(lbw_clean[["CountryName", "low_birthweight_pct"]])
            
    return results
def build_final_dataset():
    # 1. Start with the existing base dataset that has Gender + existing Malnutrition
    base_df = pd.read_csv(os.path.join(_ROOT_DIR, "outputs", "final_analytical_dataset.csv"))
    
    # Create a mapping DataFrame of all ISOs to Country Names from the base dataset
    # This helps ensure perfect merging
    iso_map = base_df[["ISO", "country"]].drop_duplicates()
    
    # 2. Get new data
    health_dfs = extract_child_health_db()
    mal_dfs = extract_malnutrition_datasets()
    
    all_new_dfs = health_dfs + mal_dfs
    
    # 3. Merge everything
    final_df = base_df.copy()
    
    for new_df in all_new_dfs:
        # Add ISO code to the new data
        new_df["ISO"] = new_df[COUNTRY_COL].apply(get_iso)
        
        # Drop the country name column to avoid conflicts, keep only ISO and the new value
        new_col = [c for c in new_df.columns if c not in [COUNTRY_COL, "ISO"]][0]
        merge_df = new_df[["ISO", new_col]].dropna(subset=["ISO", new_col])
        
        # Merge via left join to keep base dataset intact
        final_df = final_df.merge(merge_df, on="ISO", how="left")
        

    # 4. We also need to bring in WASH and Education directly into this final dataset
    # so xgboost_malnutrition.py doesn't have to do it.
    wash_df = pd.read_csv(os.path.join(_ROOT_DIR, "outputs", "wash_clean_data.csv"))
    if "iso3" in wash_df.columns:
        wash_df = wash_df.rename(columns={"iso3": "ISO"})
    
    edu_df = pd.read_csv(os.path.join(_ROOT_DIR, "outputs", "outputs/education_clean.csv"))
    
    wash_cols = ["wat_bas_nat", "san_bas_nat", "hyg_bas_nat"]
    edu_cols = ["completion_primary_f", "literacy_f"]
    
    if "year" in wash_df.columns:
        wash_df = wash_df.sort_values("year").groupby("ISO").tail(1)
        
    final_df = final_df.merge(wash_df[["ISO"] + [c for c in wash_cols if c in wash_df.columns]], on="ISO", how="left")
    final_df = final_df.merge(edu_df[["ISO"] + [c for c in edu_cols if c in edu_df.columns]], on="ISO", how="left")
    
    # Save it
    out_path = os.path.join(_ROOT_DIR, "outputs", "xgboost_final_dataset.csv")
    final_df.to_csv(out_path, index=False)

if __name__ == "__main__":
    build_final_dataset()
