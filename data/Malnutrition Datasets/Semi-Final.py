import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))
TARGET_FOLDERS = ["Diets", "Infant and Yound Child", "Breastfeeding", "Iodized Salt"]
EXCLUDE_SHEETS = ["Cover", "Notes"]

COUNTRY_COL = "CountryName"
YEAR_COL = "CMRS_year*"
NATIONAL_COL = "National_r"
MALE_COL = "male_r"
FEMALE_COL = "female_r"
URBAN_COL = "urban_r"
RURAL_COL = "rural_r"

TARGET_COLUMNS = [COUNTRY_COL, YEAR_COL, NATIONAL_COL, MALE_COL, FEMALE_COL, URBAN_COL, RURAL_COL]
HEADER_KEYWORDS = ["ISO3Code", "CountryName", "UNICEF_Reporting_Sub_Region"]

def get_unique_file_labels(files):
    tokenised = {}
    for f in files:
        name = os.path.splitext(f)[0]
        tokens = name.replace("-", " ").replace("_", " ").split()
        tokenised[f] = tokens
    common_tokens = set(tokenised[files[0]])
    for tokens in tokenised.values():
        common_tokens &= set(tokens)
    unique_names = {}
    for f, tokens in tokenised.items():
        unique = [t for t in tokens if t not in common_tokens]
        unique_names[f] = "_".join(unique) if unique else os.path.splitext(f)[0]
    return unique_names

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

    if COUNTRY_COL not in df.columns:
        return pd.DataFrame()

    # Anchor rows to valid country entries
    country_col = df[COUNTRY_COL]
    valid_idx = country_col[country_col.notna()].index
    df = df.loc[valid_idx]

    cols = [col for col in TARGET_COLUMNS if col in df.columns]
    return df[cols]

def clean_data(df):
    df = df.dropna(subset=[COUNTRY_COL])

    numeric_cols = [col for col in df.columns if col != COUNTRY_COL]

    for col in numeric_cols:
        col_data = df[col]

        if isinstance(col_data, pd.DataFrame):
            col_data = col_data.iloc[:, 0]

        if not isinstance(col_data, pd.Series):
            col_data = pd.Series(col_data)

        df[col] = pd.to_numeric(col_data, errors="coerce")

    if YEAR_COL in df.columns:
        df = df.sort_values(YEAR_COL).groupby(COUNTRY_COL).tail(1)

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
    df[COUNTRY_COL] = df[COUNTRY_COL].replace(name_map)
    return df

def plot_heatmap(df, save_path, cmap_min, cmap_max, title):
    if NATIONAL_COL not in df.columns:
        return
    world = gpd.read_file("https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip")
    df = standardise_country_names(df)
    merged = world.merge(df, how="left", left_on="ADMIN", right_on=COUNTRY_COL)
    fig, ax = plt.subplots(figsize=(15, 10))
    merged.plot(column=NATIONAL_COL, ax=ax, legend=False, missing_kwds={"color": "lightgrey"})
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=cmap_min, vmax=cmap_max))
    sm._A = []
    fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    ax.set_title(title)
    ax.axis("off")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

def plot_top_bottom_bar(df, save_path, title):
    if NATIONAL_COL not in df.columns:
        return
    df_sorted = df.dropna(subset=[NATIONAL_COL]).sort_values(NATIONAL_COL)
    bottom = df_sorted.head(10)
    top = df_sorted.tail(10)
    combined = pd.concat([bottom, top])
    fig, ax = plt.subplots(figsize=(12, 8))
    y = range(len(combined))
    ax.barh(y, combined[NATIONAL_COL])
    ax.set_yticks(y)
    ax.set_yticklabels(combined[COUNTRY_COL])
    for i, v in enumerate(combined[NATIONAL_COL]):
        ax.text(v, i, f" {v:.1f}", va="center")
    ax.axhline(9.5)
    ax.set_title(title)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

def plot_gender_difference(df, save_path, title):
    if MALE_COL not in df.columns or FEMALE_COL not in df.columns:
        return
    df = df.dropna(subset=[MALE_COL, FEMALE_COL]).copy()
    df["Diff"] = (df[MALE_COL] - df[FEMALE_COL]).abs()
    df_top = df.sort_values("Diff", ascending=False).head(10).sort_values("Diff")
    fig, ax = plt.subplots(figsize=(12, 8))
    y = range(len(df_top))
    width = 0.3

    # Base bars
    ax.barh([i - width for i in y], df_top[MALE_COL], height=width, label="Male")
    ax.barh(y, df_top[FEMALE_COL], height=width, label="Female")

    # Difference bar (centered)
    ax.barh([i + width for i in y], df_top["Diff"], height=width, label="Abs Diff")
    ax.set_yticks(y)
    ax.set_yticklabels(df_top[COUNTRY_COL])

    # Labels
    for i, (m, f, d) in enumerate(zip(df_top[MALE_COL], df_top[FEMALE_COL], df_top["Diff"])):
        ax.text(m, i - width, f" {m:.1f}", va="center")
        ax.text(f, i, f" {f:.1f}", va="center")
        ax.text(d, i + width, f" {d:.1f}", va="center")
    ax.set_title(title + " (Male vs Female with Difference)")
    ax.legend()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

def plot_urban_rural_difference(df, save_path, title):
    if URBAN_COL not in df.columns or RURAL_COL not in df.columns:
        return
    df = df.dropna(subset=[URBAN_COL, RURAL_COL]).copy()
    df["Diff"] = (df[URBAN_COL] - df[RURAL_COL]).abs()
    df_top = df.sort_values("Diff", ascending=False).head(10).sort_values("Diff")
    fig, ax = plt.subplots(figsize=(12, 8))
    y = range(len(df_top))
    width = 0.3

    # Base bars
    ax.barh([i - width for i in y], df_top[URBAN_COL], height=width, label="Urban")
    ax.barh(y, df_top[RURAL_COL], height=width, label="Rural")

    # Difference bar
    ax.barh([i + width for i in y], df_top["Diff"], height=width, label="Abs Diff")
    ax.set_yticks(y)
    ax.set_yticklabels(df_top[COUNTRY_COL])

    # Labels
    for i, (u, r, d) in enumerate(zip(df_top[URBAN_COL], df_top[RURAL_COL], df_top["Diff"])):
        ax.text(u, i - width, f" {u:.1f}", va="center")
        ax.text(r, i, f" {r:.1f}", va="center")
        ax.text(d, i + width, f" {d:.1f}", va="center")
    ax.set_title(title + " (Urban vs Rural with Difference)")
    ax.legend()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

def main():
    for folder in TARGET_FOLDERS:
        folder_path = os.path.join(BASE_FOLDER, folder)
        if not os.path.exists(folder_path):
            continue

        files = [f for f in os.listdir(folder_path) if f.endswith(".xlsx")]
        if not files:
            continue

        unique_names = get_unique_file_labels(files)
        all_national = []
        all_data = []

        for f in files:
            file_path = os.path.join(folder_path, f)
            xls = pd.ExcelFile(file_path)

            for sheet_name in xls.sheet_names:
                if sheet_name in EXCLUDE_SHEETS:
                    continue

                df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
                df = extract_target_columns(df)
                if df.empty:
                    continue

                df = clean_data(df)
                all_data.append((f, sheet_name, df))

                if NATIONAL_COL in df.columns:
                    all_national.extend(df[NATIONAL_COL].dropna().tolist())

        cmap_min = min(all_national) if all_national else 0
        cmap_max = max(all_national) if all_national else 1

        for f, sheet, df in all_data:
            file_base = unique_names[f]
            sheet_safe = sheet.replace(" ", "_").replace("/", "_")
            title = f"{file_base} - {sheet}"

            if NATIONAL_COL in df.columns:
                heat = os.path.join(folder_path, f"{file_base}_{sheet_safe}_map.pdf")
                bar = os.path.join(folder_path, f"{file_base}_{sheet_safe}_bar.pdf")
                plot_heatmap(df, heat, cmap_min, cmap_max, title)
                plot_top_bottom_bar(df, bar, title + " (Top & Bottom 10)")
                print(f"Saved: {heat}")
                print(f"Saved: {bar}")

            if MALE_COL in df.columns and FEMALE_COL in df.columns:
                gap = os.path.join(folder_path, f"{file_base}_{sheet_safe}_gender_diff.pdf")
                plot_gender_difference(df, gap, title)
                print(f"Saved: {gap}")

            if URBAN_COL in df.columns and RURAL_COL in df.columns:
                ur_gap = os.path.join(folder_path, f"{file_base}_{sheet_safe}_urban_rural_diff.pdf")
                plot_urban_rural_difference(df, ur_gap, title)
                print(f"Saved: {ur_gap}")

if __name__ == "__main__":
    main()