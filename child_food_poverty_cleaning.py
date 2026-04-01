import os
import pandas as pd
import numpy as np

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
    df = df.copy()
    if 'ISO' not in df.columns:
        df["ISO"] = df[country_col].map(ISO_LOOKUP)
    return df

def clean_food_poverty():
    print("Cleaning Child Food Poverty Data...")

    raw_severe = pd.read_excel('UNICEF_Expanded_Global_Databases_child_food_poverty_2024_2.xlsx', sheet_name='Severe_food_poverty', header=None)

    header_row = 6
    data_start = 8

    header = raw_severe.iloc[header_row].tolist()
    subheader = raw_severe.iloc[header_row + 1].tolist()

    combined_headers = []
    current_main = None
    for m, s in zip(header, subheader):
        if pd.notna(m) and isinstance(m, str) and not m.startswith('Unnamed'):
            current_main = m.strip()

        if pd.notna(s) and isinstance(s, str) and current_main:
            combined_headers.append(f"{current_main}_{s.strip()}")
        elif pd.notna(m) and isinstance(m, str) and not m.startswith('Unnamed'):
             combined_headers.append(m.strip())
        else:
            combined_headers.append("Unknown")

    col_idx = {name: i for i, name in enumerate(combined_headers) if name != "Unknown"}

    keep_cols = ['Countries and areas', 'ISO', 'Data Source Years', 'National_Point Estimate', 'Male_Point Estimate', 'Female_Point Estimate']

    keep_indices = [col_idx[c] for c in keep_cols if c in col_idx]
    actual_keep_cols = [c for c in keep_cols if c in col_idx]

    data_severe = raw_severe.iloc[data_start:, keep_indices].copy()
    data_severe.columns = actual_keep_cols

    data_severe = data_severe.rename(columns={
        'Countries and areas': 'country',
        'ISO': 'ISO',
        'Data Source Years': 'year',
        'National_Point Estimate': 'severe_food_poverty_national',
        'Male_Point Estimate': 'severe_food_poverty_male',
        'Female_Point Estimate': 'severe_food_poverty_female'
    })

    data_severe = data_severe.dropna(subset=['country'])

    raw_moderate = pd.read_excel('UNICEF_Expanded_Global_Databases_child_food_poverty_2024_2.xlsx', sheet_name='Moderate_food_poverty', header=None)

    header = raw_moderate.iloc[header_row].tolist()
    subheader = raw_moderate.iloc[header_row + 1].tolist()

    combined_headers_mod = []
    current_main = None
    for m, s in zip(header, subheader):
        if pd.notna(m) and isinstance(m, str) and not m.startswith('Unnamed'):
            current_main = m.strip()

        if pd.notna(s) and isinstance(s, str) and current_main:
            combined_headers_mod.append(f"{current_main}_{s.strip()}")
        elif pd.notna(m) and isinstance(m, str) and not m.startswith('Unnamed'):
             combined_headers_mod.append(m.strip())
        else:
            combined_headers_mod.append("Unknown")

    col_idx_mod = {name: i for i, name in enumerate(combined_headers_mod) if name != "Unknown"}

    keep_cols_mod = ['Countries and areas', 'ISO', 'Data Source Years', 'National_Point Estimate', 'Male_Point Estimate', 'Female_Point Estimate']
    keep_indices_mod = [col_idx_mod[c] for c in keep_cols_mod if c in col_idx_mod]
    actual_keep_cols_mod = [c for c in keep_cols_mod if c in col_idx_mod]

    data_moderate = raw_moderate.iloc[data_start:, keep_indices_mod].copy()
    data_moderate.columns = actual_keep_cols_mod

    data_moderate = data_moderate.rename(columns={
        'Countries and areas': 'country',
        'ISO': 'ISO',
        'Data Source Years': 'year',
        'National_Point Estimate': 'moderate_food_poverty_national',
        'Male_Point Estimate': 'moderate_food_poverty_male',
        'Female_Point Estimate': 'moderate_food_poverty_female'
    })

    data_moderate = data_moderate.dropna(subset=['country'])

    data = pd.merge(data_severe, data_moderate, on=['country', 'ISO', 'year'], how='outer')

    data = data.replace('−', np.nan).replace('x', np.nan)

    num_cols = ['severe_food_poverty_national', 'severe_food_poverty_male', 'severe_food_poverty_female',
                'moderate_food_poverty_national', 'moderate_food_poverty_male', 'moderate_food_poverty_female']

    num_cols = [c for c in num_cols if c in data.columns]

    data[num_cols] = data[num_cols].apply(pd.to_numeric, errors='coerce')
    data = data.reset_index(drop=True)

    if 'ISO' in data.columns:
        data['ISO'] = data['ISO'].fillna(data['country'].map(ISO_LOOKUP))
    else:
        data = add_iso(data, 'country')

    cols = ['ISO'] + [col for col in data.columns if col != 'ISO']
    data = data[cols]

    data = data.dropna(subset=['ISO'])

    data = data.sort_values(['ISO', 'year'], ascending=[True, False]).drop_duplicates(subset=['ISO'], keep='first')

    out_path = 'outputs/child_food_poverty_clean.csv'
    os.makedirs('outputs', exist_ok=True)
    data.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    clean_food_poverty()
