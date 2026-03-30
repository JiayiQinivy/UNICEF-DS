import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# ===== USER INPUT =====
BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))
TARGET_FOLDERS = ["Diets", "Breastfeeding", "Child Food Enviroments 25", "Infant and Yound Child", "Iodized Salt"]

EXCLUDE_SHEETS = ["Cover", "Notes"]

TARGET_COLUMNS = ["Countries and areas", "Data Source Year*", "National"]

HEADER_KEYWORDS = ["ISO", "Countries and areas", "UNICEF Regions"]
# ======================


def detect_header(df):
    for i in range(min(20, len(df))):
        row = df.iloc[i]
        row_str = [str(v).lower() if pd.notna(v) else "" for v in row]
        if any(any(k.lower() in v for k in HEADER_KEYWORDS) for v in row_str):
            return i
    return None


def extract_data_per_sheet(file_path):
    """Return a dict: {sheet_name: dataframe}"""
    raw_sheets = pd.read_excel(file_path, sheet_name=None, header=None)
    sheet_data = {}

    for sheet_name, df in raw_sheets.items():
        if any(x.lower() in sheet_name.lower() for x in EXCLUDE_SHEETS):
            continue

        header_row = detect_header(df)
        if header_row is None:
            continue

        df = pd.read_excel(file_path, sheet_name=sheet_name, header=header_row)
        df.columns = df.columns.astype(str).str.strip()
        df = df.ffill()

        col_map = {}
        for target in TARGET_COLUMNS:
            for col in df.columns:
                if target.lower() in col.lower():
                    col_map[target] = col

        if len(col_map) < 3:
            continue

        sub = df[[col_map[t] for t in TARGET_COLUMNS]].copy()
        sub.columns = TARGET_COLUMNS
        sheet_data[sheet_name] = sub

    return sheet_data


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


def plot_heatmap(df, save_path, cmap_min, cmap_max, title):
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

    merged.plot(column="National", ax=ax, legend=False,
                missing_kwds={"color": "lightgrey"})

    sm = plt.cm.ScalarMappable(
        cmap="viridis",
        norm=plt.Normalize(vmin=cmap_min, vmax=cmap_max)
    )
    sm._A = []

    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.set_ylabel("National Value", rotation=90, labelpad=10)

    ax.set_title(title, fontsize=16)
    ax.axis("off")

    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def main():
    for target in TARGET_FOLDERS:
        FOLDER_PATH = os.path.join(BASE_FOLDER, target)
        if not os.path.exists(FOLDER_PATH):
            print(f"Folder not found: {FOLDER_PATH}")
            continue

        files = [f for f in os.listdir(FOLDER_PATH) if f.endswith(".xlsx")]

        all_national_values = []
        all_sheets_data = []

        for f in files:
            full_path = os.path.join(FOLDER_PATH, f)
            sheets = extract_data_per_sheet(full_path)
            for sheet_name, df in sheets.items():
                latest = clean_and_select_latest(df)
                if latest.empty:
                    continue
                all_national_values.extend(latest["National"].tolist())
                all_sheets_data.append((target, f, sheet_name, latest))

        if not all_sheets_data:
            print(f"No data found in folder {target}")
            continue

        cmap_min = min(all_national_values)
        cmap_max = max(all_national_values)

        for folder_name, f, sheet_name, df in all_sheets_data:
            file_base = os.path.splitext(f)[0]
            sheet_safe = sheet_name.replace(" ", "_").replace("/", "_")
            save_file = os.path.join(BASE_FOLDER, folder_name, f"{file_base}_{sheet_safe}.pdf")
            plot_heatmap(df, save_file, cmap_min, cmap_max, title=f"{file_base} - {sheet_name}")
            print(f"Saved heatmap: {save_file}")


if __name__ == "__main__":
    main()