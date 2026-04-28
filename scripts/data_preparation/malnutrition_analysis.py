import os
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
warnings.filterwarnings("ignore")

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
BASE_FOLDER   = os.path.join(_SCRIPT_DIR, "Malnutrition") #"data", "Malnutrition Datasets"
OUTPUT_FOLDER = os.path.join(_SCRIPT_DIR, "outputs")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

FILES = {
    "stunting":    "jme_expand_database_stunting_2025.xlsx",
    "wasting":     "jme_expand_database_wasting_2025.xlsx",
    "overweight":  "jme_expand_database_overweight_2025.xlsx",
    "overlapping": "jme_expand_database_overlapping_2025.xlsx",
    "model":       "jme_database_country_model_2025.xlsx",
}


def load_jme_standard(filepath, indicator):
    if not os.path.exists(filepath):
        print(f"  File not found: {filepath}")
        return pd.DataFrame()

    df = pd.read_excel(filepath, sheet_name="Trend", header=8)
    df.columns = df.columns.astype(str).str.strip()

    df_latest = df[df["LatestSource"] == "Latest Source"].copy()


    col_map = {
        "ISO3Code":                    "ISO",
        "CountryName":                 "country",
        "UNICEF_Reporting_Sub_Region": "unicef_region",
        "WB_Latest":                   "income_group",
        "CMRS_year":                   "data_year",
        "National_r":                  f"{indicator}_national",
        "National_ll":                 f"{indicator}_national_ll",
        "National_ul":                 f"{indicator}_national_ul",
        "male_r":                      f"{indicator}_male",
        "female_r":                    f"{indicator}_female",
        "urban_r":                     f"{indicator}_urban",
        "rural_r":                     f"{indicator}_rural",
        "Q1_r":                        f"{indicator}_Q1_poorest",
        "Q2_r":                        f"{indicator}_Q2",
        "Q3_r":                        f"{indicator}_Q3",
        "Q4_r":                        f"{indicator}_Q4",
        "Q5_r":                        f"{indicator}_Q5_richest",
    }
    existing = {k: v for k, v in col_map.items()
                if k in df_latest.columns}
    df_out = df_latest[list(existing.keys())].rename(
        columns=existing).copy()


    skip = {"ISO", "country", "unicef_region", "income_group"}
    for col in df_out.columns:
        if col not in skip:
            df_out[col] = pd.to_numeric(df_out[col], errors="coerce")


    m, f = f"{indicator}_male", f"{indicator}_female"
    if m in df_out.columns and f in df_out.columns:
        df_out[f"{indicator}_gender_gap"] = df_out[m] - df_out[f]

    q1, q5 = f"{indicator}_Q1_poorest", f"{indicator}_Q5_richest"
    if q1 in df_out.columns and q5 in df_out.columns:
        df_out[f"{indicator}_wealth_gap"] = df_out[q1] - df_out[q5]

    u, r = f"{indicator}_urban", f"{indicator}_rural"
    if u in df_out.columns and r in df_out.columns:
        df_out[f"{indicator}_urban_rural_gap"] = df_out[r] - df_out[u]

    return df_out


def load_overlapping(filepath):
    if not os.path.exists(filepath):
        print(f"  File not found: {filepath}")
        return pd.DataFrame()

    df = pd.read_excel(filepath, sheet_name="Trend", header=8)
    df.columns = df.columns.astype(str).str.strip()

    df_latest = df[df["Latest_Estimate"] == "Latest Estimate"].copy()

    if df_latest.empty:
        return pd.DataFrame()


    col_map = {
        "ISO3Code":                    "ISO",
        "CountryName":                 "country",
        "UNICEF_Reporting_Sub_Region": "unicef_region",
        "CMRS_year":                   "data_year",
        # 6 nutritional status categories
        "ANT_WHZ_NE2_ONLY_r":   "wasted_only_pct",
        "ANT_HAZWHZ_NE2_NE2_r": "wasted_and_stunted_pct",
        "ANT_HAZ_NE2_ONLY_r":   "stunted_only_pct",
        "ANT_HAZWHZ_NE2_PO2_r": "stunted_and_overweight_pct",
        "ANT_WHZ_PO2_ONLY_r":   "overweight_only_pct",
        "ANT_FREE_r":            "free_from_malnutrition_pct",
    }
    existing = {k: v for k, v in col_map.items()
                if k in df_latest.columns}
    df_out = df_latest[list(existing.keys())].rename(
        columns=existing).copy()


    skip = {"ISO", "country", "unicef_region"}
    for col in df_out.columns:
        if col not in skip:
            df_out[col] = pd.to_numeric(df_out[col], errors="coerce")

    # Total affected = 100 - free from malnutrition
    if "free_from_malnutrition_pct" in df_out.columns:
        df_out["any_malnutrition_pct"] = (
            100 - df_out["free_from_malnutrition_pct"])

    return df_out


def load_country_model(filepath):
    if not os.path.exists(filepath):
        print(f"  File not found: {filepath}")
        return pd.DataFrame()

    xl = pd.ExcelFile(filepath)
    sheet_prefix = {
        "Stunting Prevalence":   "stunting_model",
        "Overweight Prevalence": "overweight_model",
    }

    all_dfs = []
    for sheet, prefix in sheet_prefix.items():
        if sheet not in xl.sheet_names:
            continue

        df = pd.read_excel(filepath, sheet_name=sheet)
        df.columns = df.columns.astype(str).str.strip()

        col_map = {
            "ISO Code":                     "ISO",
            "Country or Area":              "country",
            "Year":                         "model_year",
            "Both Sexes - Point Estimates": f"{prefix}_national",
            "Both Sexes - Lower Limit":     f"{prefix}_national_ll",
            "Both Sexes - Upper Limit":     f"{prefix}_national_ul",
            "Male - Point Estimate":        f"{prefix}_male",
            "Female - Point Estimate":      f"{prefix}_female",
            "Male - Lower Limit":           f"{prefix}_male_ll",
            "Male - Upper Limit":           f"{prefix}_male_ul",
            "Female - Lower Limit":         f"{prefix}_female_ll",
            "Female - Upper Limit":         f"{prefix}_female_ul",
        }
        existing = {k: v for k, v in col_map.items()
                    if k in df.columns}
        df = df[list(existing.keys())].rename(columns=existing).copy()

        for col in df.columns:
            if col not in {"ISO", "country"}:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        # Most recent year per country
        df = (df.sort_values("model_year")
                .groupby("ISO").tail(1)
                .reset_index(drop=True))


        m, f = f"{prefix}_male", f"{prefix}_female"
        if m in df.columns and f in df.columns:
            df[f"{prefix}_gender_gap"] = df[m] - df[f]

        all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()


    result = all_dfs[0]
    for df in all_dfs[1:]:
        drop_dupe = [c for c in ["country", "model_year"]
                     if c in df.columns and c in result.columns]
        result = result.merge(
            df.drop(columns=drop_dupe, errors="ignore"),
            on="ISO", how="outer")

    return result


def build_master(stunting_df, wasting_df, overweight_df,
                 overlapping_df, model_df):
    master = stunting_df.copy()

    shared_meta = {"country", "unicef_region", "income_group", "data_year"}

    def safe_merge(master, df, name):
        if df.empty:
            return master
        join_cols = ["ISO"] + [c for c in df.columns
                                if c not in shared_meta]
        join_cols = list(dict.fromkeys(join_cols))
        master = master.merge(df[join_cols], on="ISO", how="outer")
        return master

    master = safe_merge(master, wasting_df,     "wasting")
    master = safe_merge(master, overweight_df,  "overweight")
    master = safe_merge(master, overlapping_df, "overlapping")

    if model_df is not None and not model_df.empty:
        drop_dupe = [c for c in ["country"] if c in model_df.columns]
        master = master.merge(
            model_df.drop(columns=drop_dupe, errors="ignore"),
            on="ISO", how="left")

    if ("stunting_national" in master.columns
            and "overweight_national" in master.columns):
        master["double_burden_index"] = (
            master["stunting_national"] + master["overweight_national"])


    score_cols = [c for c in ["stunting_national",
                               "wasting_national",
                               "overweight_national"]
                  if c in master.columns]
    if len(score_cols) >= 2:
        master["malnutrition_composite_score"] = (
            master[score_cols].mean(axis=1))


    if ("stunting_national" in master.columns
            and "overweight_national" in master.columns):
        both_avail = master[["stunting_national",
                              "overweight_national"]].notna().all(axis=1)
        master["nutrition_transition_flag"] = np.where(
            both_avail,
            (master["overweight_national"]
             > master["stunting_national"]).astype(int),
            np.nan)

    out = os.path.join(OUTPUT_FOLDER, "malnutrition_master.csv")
    master.to_csv(out, index=False)
    print(f"Saved {out}  ({len(master)} countries)")

    modelling_sample = master.dropna(subset=["stunting_national",
                                             "wasting_national",
                                             "overweight_national"])
    modelling_sample.to_csv(
        os.path.join(OUTPUT_FOLDER, "malnutrition_modelling_sample.csv"),
        index=False)

    return master



def get_world():
    try:
        return gpd.read_file(
            "https://naturalearth.s3.amazonaws.com/110m_cultural/"
            "ne_110m_admin_0_countries.zip")
    except Exception as e:
        print(f"  ️  Cannot load world shapefile: {e}")
        return None


def plot_four_panel_map(master):
    world = get_world()
    if world is None:
        return

    merged = world.merge(master, how="left",
                          left_on="SOV_A3", right_on="ISO")

    panels = [
        ("stunting_national",
         "Stunting (%)\nChronic Malnutrition (HAZ < −2SD)", "YlOrRd"),
        ("wasting_national",
         "Wasting (%)\nAcute Malnutrition (WHZ < −2SD)", "YlOrRd"),
        ("overweight_national",
         "Overweight (%)\nOver-nutrition (WHZ > +2SD)", "YlGn"),
        ("wasted_and_stunted_pct",
         "Wasted & Stunted (%)\nDouble Deficit (Overlapping)", "PuRd"),
    ]
    panels = [(c, t, cm) for c, t, cm in panels
              if c in merged.columns
              and merged[c].notna().sum() > 0]

    fig, axes = plt.subplots(2, 2, figsize=(22, 14))
    axes = axes.flatten()

    for i, (col, title, cmap) in enumerate(panels):
        ax   = axes[i]
        data = merged[col].dropna()
        vmin = float(data.quantile(0.02))
        vmax = float(data.quantile(0.98))

        merged.plot(column=col, ax=ax, cmap=cmap,
                    vmin=vmin, vmax=vmax, legend=False,
                    missing_kwds={"color": "#e0e0e0"})

        sm = plt.cm.ScalarMappable(
            cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm._A = []
        cbar = fig.colorbar(sm, ax=ax, fraction=0.025,
                             pad=0.02, shrink=0.6)
        cbar.ax.set_ylabel("Prevalence (%)", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

        ax.legend(
            handles=[mpatches.Patch(color="#e0e0e0", label="No data")],
            loc="lower left", fontsize=8, framealpha=0.8)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.axis("off")
        ax.annotate(f"n = {merged[col].notna().sum()} countries",
                    xy=(0.02, 0.04), xycoords="axes fraction",
                    fontsize=8, color="#555")

    for j in range(len(panels), 4):
        axes[j].set_visible(False)

    fig.suptitle(
        "Global Child Malnutrition — Core Indicators\n"
        "Source: UNICEF Joint Malnutrition Estimates 2025",
        fontsize=14, fontweight="bold", y=1.005)
    plt.tight_layout()
    out = os.path.join(OUTPUT_FOLDER, "malnutrition_4panel_map.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_composite_map(master):
    world = get_world()
    if world is None:
        return

    merged = world.merge(master, how="left",
                          left_on="SOV_A3", right_on="ISO")

    col  = "malnutrition_composite_score"
    data = merged[col].dropna()
    vmin = float(data.quantile(0.02))
    vmax = float(data.quantile(0.98))

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    merged.plot(
        column=col, ax=ax, cmap="YlOrRd",
        vmin=vmin, vmax=vmax, legend=False,
        missing_kwds={"color": "#d4d4d4"},
        linewidth=0.15, edgecolor="#aaaaaa")

    sm = plt.cm.ScalarMappable(
        cmap="YlOrRd", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax,
                        fraction=0.012,
                        pad=0.01,
                        shrink=0.85)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("Composite score (%)", fontsize=14, labelpad=3)

    ax.legend(
        handles=[mpatches.Patch(color="#d4d4d4", label="No data")],
        loc="lower left", fontsize=12, framealpha=0.8,
        handlelength=1, handleheight=0.8,
        borderpad=0.3, labelspacing=0.2)

    ax.set_title(
        f"Global Malnutrition Composite Score (n=137, UNICEF JME 2025)",
        fontsize=16, fontweight="bold", pad=3)

    ax.axis("off")

    fig.subplots_adjust(left=0.01, right=0.88,
                        top=0.92, bottom=0.01)

    out = os.path.join(OUTPUT_FOLDER, "report_composite_map.pdf")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: report_composite_map.png")



def plot_overlapping_stacked_bar(master):
    """
    Stacked bar chart of the 6 nutritional status categories
    by UNICEF region — showing the 'nutritional landscape'.
    This is a unique visualisation using the overlapping data.
    """
    cat_cols = [
        ("wasted_only_pct",            "Wasted Only",             "#e15759"),
        ("wasted_and_stunted_pct",     "Wasted & Stunted",        "#9c3d54"),
        ("stunted_only_pct",           "Stunted Only",            "#f28e2b"),
        ("stunted_and_overweight_pct", "Stunted & Overweight",    "#76b7b2"),
        ("overweight_only_pct",        "Overweight Only",         "#59a14f"),
        ("free_from_malnutrition_pct", "Free from Malnutrition",  "#bab0ac"),
    ]
    avail = [(c, l, cl) for c, l, cl in cat_cols
             if c in master.columns]

    if not avail or "unicef_region" not in master.columns:
        return

    df = master.dropna(subset=["unicef_region"]).copy()
    avail_cols = [c for c, _, _ in avail]
    df_clean   = df.dropna(subset=avail_cols, how="all")

    region_means = (df_clean.groupby("unicef_region")[avail_cols]
                            .mean().round(1))
    region_means = region_means.sort_values(
        "stunted_only_pct" if "stunted_only_pct" in region_means.columns
        else avail_cols[0], ascending=False)

    fig, ax = plt.subplots(figsize=(13, 6))
    bottom  = np.zeros(len(region_means))
    bars    = []

    for col, label, color in avail:
        if col not in region_means.columns:
            continue
        vals = region_means[col].values
        bar  = ax.bar(region_means.index, vals,
                      bottom=bottom, color=color, label=label,
                      edgecolor="white", linewidth=0.5)
        bars.append(bar)
        bottom += vals

    ax.set_xlabel("UNICEF Region", fontsize=11)
    ax.set_ylabel("Children under 5 (%)", fontsize=11)
    ax.set_title(
        "Nutritional Status of Children Under 5 by UNICEF Region\n"
        "Source: UNICEF Overlapping Malnutrition Estimates 2025",
        fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_ylim(0, 105)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.tight_layout()

    out = os.path.join(OUTPUT_FOLDER,
                       "nutritional_status_stacked_bar.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_double_burden(master):
    """Scatter: Stunting vs Overweight coloured by UNICEF region."""
    need = ["stunting_national", "overweight_national", "unicef_region"]
    if not all(c in master.columns for c in need):
        return

    df = master.dropna(subset=need).copy()
    if df.empty:
        return

    regions = sorted(df["unicef_region"].dropna().unique())
    palette = plt.cm.get_cmap("Set2", max(len(regions), 3))
    rcolors = {r: palette(i) for i, r in enumerate(regions)}

    fig, ax = plt.subplots(figsize=(10, 8))
    for region, grp in df.groupby("unicef_region"):
        ax.scatter(grp["stunting_national"],
                   grp["overweight_national"],
                   c=[rcolors.get(region, "grey")],
                   label=region, s=65, alpha=0.78,
                   edgecolors="white", linewidth=0.5)

    lim = max(df["stunting_national"].max(),
              df["overweight_national"].max()) + 5
    ax.plot([0, lim], [0, lim], "k--", lw=0.9, alpha=0.35,
            label="Overweight = Stunting")

    med_s = df["stunting_national"].median()
    med_o = df["overweight_national"].median()
    ax.axvline(med_s, color="#aaa", lw=0.7, ls=":")
    ax.axhline(med_o, color="#aaa", lw=0.7, ls=":")
    ax.annotate("Double burden\nzone",
                xy=(med_s + 0.5, med_o + 0.3),
                fontsize=9, color="#cc3333", style="italic")

    ax.set_xlabel("Stunting Rate (%) — Chronic Undernutrition",
                  fontsize=11)
    ax.set_ylabel("Overweight Rate (%) — Over-nutrition", fontsize=11)
    ax.set_title(
        "Double Burden of Malnutrition: Stunting vs Overweight\n"
        "by UNICEF Region",
        fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9,
              title="UNICEF Region", title_fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUTPUT_FOLDER, "double_burden_scatter.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_distributions(master):
    from scipy import stats
    from scipy.stats import gaussian_kde

    candidates = [
        ("stunting_national",    "Stunting (%)"),
        ("wasting_national",     "Wasting (%)"),
        ("overweight_national",  "Overweight (%)"),
        ("wasted_and_stunted_pct", "Wasted & Stunted (%)"),
    ]
    panels = [(c, l) for c, l in candidates
              if c in master.columns
              and master[c].notna().sum() > 5]
    if not panels:
        return

    fig, axes = plt.subplots(1, len(panels),
                              figsize=(5 * len(panels), 5))
    if len(panels) == 1:
        axes = [axes]

    for ax, (col, label) in zip(axes, panels):
        data = master[col].dropna()
        ax.hist(data, bins=25, color="#4e79a7", alpha=0.70,
                edgecolor="white", density=True)
        kde = gaussian_kde(data)
        x   = np.linspace(data.min(), data.max(), 200)
        ax.plot(x, kde(x), color="#e15759", lw=2, label="KDE")

        if len(data) <= 5000:
            w, p   = stats.shapiro(data)
            normal = p > 0.05
            clr    = "#2ca02c" if normal else "#d62728"
            txt    = (f"Shapiro-Wilk\nW={w:.3f}, p={p:.4f}\n"
                      f"{'Normal ✓' if normal else 'Non-normal ✗'}")
            ax.annotate(txt,
                        xy=(0.97, 0.95), xycoords="axes fraction",
                        fontsize=8.5, ha="right", va="top", color=clr,
                        bbox=dict(boxstyle="round,pad=0.3",
                                   fc="white", alpha=0.85))

        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Prevalence (%)", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.set_xlim(left=0)
        ax.legend(fontsize=8)

    fig.suptitle(
        "Distribution of Child Malnutrition Indicators\n"
        "(Shapiro-Wilk guides parametric vs non-parametric choice)",
        fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = os.path.join(OUTPUT_FOLDER, "malnutrition_distributions.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_regional_boxplot(master):
    if "unicef_region" not in master.columns:
        return

    candidates = [
        ("stunting_national",    "Stunting (%)"),
        ("wasting_national",     "Wasting (%)"),
        ("overweight_national",  "Overweight (%)"),
    ]
    panels = [(c, l) for c, l in candidates
              if c in master.columns
              and master[c].notna().sum() > 5]
    if not panels:
        return

    df = master.dropna(subset=["unicef_region"]).copy()
    if "stunting_national" in df.columns:
        region_order = (df.groupby("unicef_region")["stunting_national"]
                          .median().sort_values(ascending=False)
                          .index.tolist())
    else:
        region_order = sorted(df["unicef_region"].unique())

    colors = ["#4e79a7", "#f28e2b", "#59a14f", "#b07aa1"]
    fig, axes = plt.subplots(1, len(panels), figsize=(15, 6))
    if len(panels) == 1:
        axes = [axes]

    for ax, (col, label), clr in zip(axes, panels, colors):
        data_per_region = [
            df.loc[df["unicef_region"] == r, col].dropna().values
            for r in region_order
        ]
        bp = ax.boxplot(data_per_region, patch_artist=True,
                        widths=0.5,
                        medianprops=dict(color="black", linewidth=2))
        for patch in bp["boxes"]:
            patch.set_facecolor(clr)
            patch.set_alpha(0.70)

        ax.set_xticks(range(1, len(region_order) + 1))
        ax.set_xticklabels(region_order, rotation=38,
                            ha="right", fontsize=14)
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.set_ylabel("Prevalence (%)", fontsize=14)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Child Malnutrition by UNICEF Region",
        fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = os.path.join(OUTPUT_FOLDER,
                       "malnutrition_regional_boxplot.pdf")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_summary(master):
    pass


def main():
    def fp(key):
        return os.path.join(BASE_FOLDER, FILES[key])


    stunting_df    = load_jme_standard(fp("stunting"),   "stunting")
    wasting_df     = load_jme_standard(fp("wasting"),    "wasting")
    overweight_df  = load_jme_standard(fp("overweight"), "overweight")
    overlapping_df = load_overlapping(fp("overlapping"))
    model_df       = load_country_model(fp("model"))


    master = build_master(stunting_df, wasting_df, overweight_df,
                          overlapping_df, model_df)

    if master.empty:
        print(" Master dataset is empty. Check file paths.")
        return


    for name, df in [("stunting",    stunting_df),
                     ("wasting",     wasting_df),
                     ("overweight",  overweight_df),
                     ("overlapping", overlapping_df)]:
        if not df.empty:
            out = os.path.join(OUTPUT_FOLDER, f"{name}_clean.csv")
            df.to_csv(out, index=False)


    plot_four_panel_map(master)
    plot_overlapping_stacked_bar(master)
    plot_double_burden(master)
    plot_distributions(master)
    plot_regional_boxplot(master)
    plot_composite_map(master)

    print("PIPELINE COMPLETE")


if __name__ == "__main__":
    main()
