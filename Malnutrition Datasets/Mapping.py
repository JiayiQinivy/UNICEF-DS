import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# ===== USER INPUT =====
BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))
TARGET_FOLDER = "Diets"

EXCLUDE_SHEETS = ["Cover","Latest Regional Global", "Notes"]

TARGET_COLUMNS = ["Countries and areas", "Data Source Year*", "National"]

HEADER_KEYWORDS = ["ISO", "Countries and areas", "UNICEF Regions"]
# ======================

FOLDER_PATH = os.path.join(BASE_FOLDER, TARGET_FOLDER)


def find_excel_files(folder):
    files = []
    for root, dirs, filenames in os.walk(folder):
        for f in filenames:
            if f.endswith(".xlsx"):
                files.append(os.path.join(root, f))
    return files


def detect_header(df):
    for i in range(min(20, len(df))):
        row = df.iloc[i]
        row_str = [str(v).lower() if pd.notna(v) else "" for v in row]
        if any(any(k.lower() in v for k in HEADER_KEYWORDS) for v in row_str):
            return i
    return None


def extract_data(file_path):
    all_data = []

    raw_sheets = pd.read_excel(file_path, sheet_name=None, header=None)

    for sheet_name, df in raw_sheets.items():

        if any(x.lower() in sheet_name.lower() for x in EXCLUDE_SHEETS):
            continue

        header_row = detect_header(df)

        if header_row is None:
            continue

        df = pd.read_excel(file_path, sheet_name=sheet_name, header=header_row)
        df.columns = df.columns.astype(str).str.strip()
        df = df.ffill()

        # Match columns
        col_map = {}
        for target in TARGET_COLUMNS:
            for col in df.columns:
                if target.lower() in col.lower():
                    col_map[target] = col

        if len(col_map) < 3:
            continue

        sub = df[[col_map[t] for t in TARGET_COLUMNS]].copy()
        sub.columns = TARGET_COLUMNS

        all_data.append(sub)

    if all_data:
        return pd.concat(all_data, ignore_index=True)

    return pd.DataFrame()


def clean_and_select_latest(df):
    df = df.dropna(subset=["Countries and areas", "National"])

    df["Data Source Year*"] = pd.to_numeric(df["Data Source Year*"], errors="coerce")
    df["National"] = pd.to_numeric(df["National"], errors="coerce")

    df = df.dropna(subset=["Data Source Year*", "National"])

    df = df.sort_values("Data Source Year*")
    df = df.groupby("Countries and areas").tail(1)

    return df


def standardise_country_names(df):
    name_map = {
        "United States": "United States of America",
        "Russia": "Russian Federation",
        "Congo": "Democratic Republic of the Congo",
        "DR Congo": "Democratic Republic of the Congo",
        "Iran": "Iran (Islamic Republic of)",
        "Syria": "Syrian Arab Republic",
        "Vietnam": "Viet Nam",
        "Tanzania": "United Republic of Tanzania",
        "Bolivia": "Bolivia (Plurinational State of)",
        "Venezuela": "Venezuela (Bolivarian Republic of)",
        "South Korea": "Republic of Korea",
        "North Korea": "Democratic People's Republic of Korea",
        "Laos": "Lao People's Democratic Republic",
        "Moldova": "Republic of Moldova",
        "Brunei": "Brunei Darussalam",
        "Ivory Coast": "Côte d'Ivoire",
        "Cape Verde": "Cabo Verde",
        "Swaziland": "Eswatini",
        "UK": "United Kingdom",
    }

    df["Countries and areas"] = df["Countries and areas"].replace(name_map)
    return df


def plot_heatmap(df):
    # Load Natural Earth shapefile directly from URL
    world = gpd.read_file(
        "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
    )

    df = standardise_country_names(df)

    merged = world.merge(
        df,
        how="left",
        left_on="ADMIN",
        right_on="Countries and areas"
    )

    fig, ax = plt.subplots(figsize=(15, 10))

    # Plot map without legend
    merged.plot(
        column="National",
        ax=ax,
        legend=False,
        missing_kwds={"color": "lightgrey"}
    )

    # Create a colorbar that matches map height
    sm = plt.cm.ScalarMappable(
        cmap="viridis",  # or your preferred colormap
        norm=plt.Normalize(vmin=merged["National"].min(), vmax=merged["National"].max())
    )
    sm._A = []  # Required for ScalarMappable

    cbar = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.02)  # fraction controls width, pad controls spacing
    cbar.ax.set_ylabel("National Value", rotation=90, labelpad=10)

    ax.set_title("Global Heatmap based on National Values")
    ax.axis("off")

    plt.show()

def main():
    files = find_excel_files(FOLDER_PATH)

    all_data = []
    for f in files:
        df = extract_data(f)
        if not df.empty:
            all_data.append(df)

    if not all_data:
        print("No data found")
        return

    combined = pd.concat(all_data, ignore_index=True)

    latest = clean_and_select_latest(combined)

    plot_heatmap(latest)


if __name__ == "__main__":
    main()