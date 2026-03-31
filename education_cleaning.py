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
    """Map country names to ISO3 codes."""
    df = df.copy()
    df["ISO"] = df[country_col].map(ISO_LOOKUP)
    return df

def clean_education():
    print("Cleaning Education Data...")
    raw = pd.read_excel('education-dataset.xlsx', sheet_name='10. Edu', header=None)
    data = raw.iloc[8:, [1, 2, 4, 6, 8, 18, 20, 42, 44]].copy()
    data.columns = [
        'country',
        'oos_preprimary_m', 'oos_preprimary_f',
        'oos_primary_m',    'oos_primary_f',
        'completion_primary_m', 'completion_primary_f',
        'literacy_m', 'literacy_f'
    ]
    data = data.dropna(subset=['country'])
    data = data.replace('−', np.nan).replace('x', np.nan)
    num_cols = data.columns[1:]
    data[num_cols] = data[num_cols].apply(pd.to_numeric, errors='coerce')
    data = data.reset_index(drop=True)

    data = add_iso(data, 'country')

    cols = ['ISO'] + [col for col in data.columns if col != 'ISO']
    data = data[cols]

    data = data.dropna(subset=['ISO'])

    out_path = 'outputs/education_clean.csv'
    os.makedirs('outputs', exist_ok=True)
    data.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    clean_education()
