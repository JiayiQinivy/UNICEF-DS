"""
WASH Exploratory Visualisation

Source: outputs/wash_clean_data.csv
Outputs: outputs/

Selected indicators: wat_bas_nat, san_bas_nat, hyg_bas_nat
Rationale: Basic-service tier is the JMP policy benchmark and has
complete national coverage.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")

# ===== USER INPUT =====
SCRIPT_DIR       = os.path.dirname(os.path.abspath(__file__))
WASH_CSV         = os.path.join(SCRIPT_DIR, "outputs", "wash_clean_data.csv")
MALNUTRITION_CSV = os.path.join(SCRIPT_DIR, "outputs",
                                 "malnutrition_modelling_sample.csv")
OUTPUT_DIR       = SCRIPT_DIR
OUTPUT_CSV_DIR   = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)

# The three selected WASH indicators for Model 3
SELECTED_WASH = ["wat_bas_nat", "san_bas_nat", "hyg_bas_nat"]
# ======================

OUTCOMES = ["stunting_national", "wasting_national", "overweight_national"]
OUTCOME_LABELS = {
    "stunting_national":   "Stunting (%)",
    "wasting_national":    "Wasting (%)",
    "overweight_national": "Overweight (%)",
}

ALL_WASH_COLS = [
    "wat_bas_nat", "wat_lim_nat", "wat_none_nat",
    "san_bas_nat", "san_lim_nat", "san_none_nat",
    "hyg_bas_nat", "hyg_lim_nat", "hyg_none_nat",
]

WASH_LABELS = {
    "wat_bas_nat":  "Basic Water Access (%)",
    "wat_lim_nat":  "Limited Water Access (%)",
    "wat_none_nat": "No Water Service (%)",
    "san_bas_nat":  "Basic Sanitation Access (%)",
    "san_lim_nat":  "Limited Sanitation Access (%)",
    "san_none_nat": "No Sanitation Service (%)",
    "hyg_bas_nat":  "Basic Hygiene Access (%)",
    "hyg_lim_nat":  "Limited Hygiene Access (%)",
    "hyg_none_nat": "No Hygiene Service (%)",
}

REGION_LABELS = {
    "Sub-Saharan Africa":           "Sub-Saharan Africa",
    "South Asia":                   "South Asia",
    "East Asia and Pacific":        "East Asia & Pacific",
    "Europe and Central Asia":      "Europe & Central Asia",
    "Latin America and Caribbean":  "Latin America & Carib.",
    "Middle East and North Africa": "Middle East & N. Africa",
    "North America":                "North America",
}

REGION_COLORS = {
    "Sub-Saharan Africa":        "#ff7f0e",
    "South Asia":                "#d62728",
    "East Asia & Pacific":       "#2ca02c",
    "Europe & Central Asia":     "#1f77b4",
    "Latin America & Carib.":    "#17becf",
    "Middle East & N. Africa":   "#8c564b",
    "North America":             "#7f7f7f",
}


# ──────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────

def load_data():
    wash   = pd.read_csv(WASH_CSV)
    mal    = pd.read_csv(MALNUTRITION_CSV)
    merged = mal.merge(
        wash.drop(columns=["country", "year", "sdg_region",
                            "all_basic_missing"], errors="ignore"),
        on="ISO", how="inner")
    merged["region"] = (
        merged["unicef_reporting_region"]
        .map(REGION_LABELS)
        .fillna(merged["unicef_reporting_region"]))
    print(f"  Merged dataset: {len(merged)} countries")
    return wash, merged


# ──────────────────────────────────────────────────────────────
# SECTION 1: INDICATOR SELECTION
# ──────────────────────────────────────────────────────────────

def indicator_selection(wash, merged):
    """
    Rank all 9 WASH indicators by Spearman correlation with
    malnutrition outcomes. Report collinearity among selected
    indicators. Save wash_selected.csv.
    """
    print(f"\n{'='*60}")
    print("SECTION 1: INDICATOR SELECTION")
    print(f"{'='*60}")

    # Full ranking
    n_tests = len(ALL_WASH_COLS) * len(OUTCOMES)
    bonf    = 0.05 / n_tests

    rows = []
    for col in ALL_WASH_COLS:
        if col not in merged.columns:
            continue
        n = merged[col].notna().sum()
        if n < 15:
            continue
        total_r, n_bonf, details = 0, 0, []
        for outcome in OUTCOMES:
            sub = merged[[col, outcome]].dropna()
            if len(sub) < 10:
                details.append("n/a")
                continue
            r, p = spearmanr(sub[col], sub[outcome])
            total_r += abs(r)
            if p < bonf:
                n_bonf += 1
            s = ("***" if p < 0.001 else
                 ("**"  if p < 0.01  else
                  ("*"   if p < 0.05  else "ns")))
            details.append(f"{r:+.3f}{s}")
        avg_r = total_r / len(OUTCOMES)
        rows.append({
            "indicator":   col,
            "avg_abs_r":   round(avg_r, 3),
            "n_bonf":      n_bonf,
            "n_countries": n,
            "stunting_r":  details[0] if details else "n/a",
            "wasting_r":   details[1] if len(details) > 1 else "n/a",
            "overweight_r":details[2] if len(details) > 2 else "n/a",
        })

    rank_df = pd.DataFrame(rows).sort_values(
        "avg_abs_r", ascending=False).reset_index(drop=True)
    rank_df.index += 1

    print(f"\n  Bonferroni threshold: p < {bonf:.5f}")
    print(f"\n  {'Rank':<5} {'Indicator':<20} {'avg|r|':>7} "
          f"{'Bonf':>5} {'n':>5}  "
          f"stunting | wasting | overweight")
    print(f"  {'─'*80}")
    for rank, row in rank_df.iterrows():
        marker = " <-- selected" if row["indicator"] in SELECTED_WASH else ""
        print(f"  #{rank:<4} {row['indicator']:<20} "
              f"{row['avg_abs_r']:>7.3f} "
              f"{row['n_bonf']:>4}/3 "
              f"{row['n_countries']:>5}  "
              f"{row['stunting_r']} | "
              f"{row['wasting_r']} | "
              f"{row['overweight_r']}"
              f"{marker}")

    # Collinearity among selected indicators
    print(f"\n  Collinearity among selected indicators (Spearman r):")
    avail = [c for c in SELECTED_WASH if c in merged.columns]
    corr  = merged[avail].corr(method="spearman").round(3)
    print(corr.to_string())
    print(f"\n  Note: all pairs r > 0.80, but each represents a")
    print(f"  distinct WASH dimension (water / sanitation / hygiene)")
    print(f"  with independent causal pathways to malnutrition.")

    # Coverage of selected indicators
    print(f"\n  Selected indicators coverage:")
    for col in SELECTED_WASH:
        if col in merged.columns:
            n = merged[col].notna().sum()
            print(f"    {col:<20}: {n} countries")

    # Estimate Model 3 sample size
    GENDER = ["female_married_by_18", "female_married_by_15",
              "marriage_gap_18", "anc4_15_19_pct",
              "modern_contraceptive_pct"]
    avail_g = [c for c in GENDER if c in merged.columns]
    avail_w = [c for c in SELECTED_WASH if c in merged.columns]
    test_cols = (["stunting_national", "income_group"] +
                 avail_g + avail_w)
    n_est = merged[
        [c for c in test_cols if c in merged.columns]
    ].dropna().shape[0]
    print(f"\n  Estimated Model 3 sample")
    print(f"  (gender + all 3 WASH indicators): n = {n_est} countries")

    # Save wash_selected.csv
    keep = ["ISO", "country"] + SELECTED_WASH
    if "unicef_reporting_region" in wash.columns:
        keep += ["unicef_reporting_region"]
    wash_sel = wash[[c for c in keep if c in wash.columns]].copy()
    out_path  = os.path.join(OUTPUT_CSV_DIR, "wash_selected.csv")
    wash_sel.to_csv(out_path, index=False)
    print(f"\n  Saved: wash_selected.csv")
    print(f"  Columns: {list(wash_sel.columns)}")

    return rank_df


# ──────────────────────────────────────────────────────────────
# FIGURE 1: CORRELATION HEATMAP
# ──────────────────────────────────────────────────────────────

def plot_correlation_heatmap(merged):
    """
    Spearman r heatmap: all 9 WASH indicators vs 3 malnutrition
    outcomes. Rows sorted by avg|r|. Selected indicators highlighted.
    """
    print("\n  Generating Figure 1: Correlation heatmap...")

    #indicators = [c for c in ALL_WASH_COLS if c in merged.columns]
    indicators = [c for c in SELECTED_WASH if c in merged.columns]
    r_mat = pd.DataFrame(index=indicators, columns=OUTCOMES, dtype=float)
    p_mat = pd.DataFrame(index=indicators, columns=OUTCOMES, dtype=float)
    n_mat = pd.DataFrame(index=indicators, columns=OUTCOMES, dtype=int)

    for ind in indicators:
        for outcome in OUTCOMES:
            sub = merged[[ind, outcome]].dropna()
            if len(sub) < 10:
                r_mat.loc[ind, outcome] = np.nan
                p_mat.loc[ind, outcome] = np.nan
                n_mat.loc[ind, outcome] = len(sub)
            else:
                r, p = spearmanr(sub[ind], sub[outcome])
                r_mat.loc[ind, outcome] = round(r, 3)
                p_mat.loc[ind, outcome] = p
                n_mat.loc[ind, outcome] = len(sub)

    avg_r = r_mat.abs().mean(axis=1).sort_values(ascending=False)
    r_mat = r_mat.loc[avg_r.index]
    p_mat = p_mat.loc[avg_r.index]
    n_mat = n_mat.loc[avg_r.index]

    annot = pd.DataFrame(index=r_mat.index, columns=OUTCOMES, dtype=str)
    for ind in r_mat.index:
        for outcome in OUTCOMES:
            r = r_mat.loc[ind, outcome]
            p = p_mat.loc[ind, outcome]
            n = n_mat.loc[ind, outcome]
            if pd.isna(r):
                annot.loc[ind, outcome] = f"n={n}"
            else:
                s = ("***" if p < 0.001 else
                     ("**"  if p < 0.01  else
                      ("*"   if p < 0.05  else "")))
                annot.loc[ind, outcome] = f"{r:.2f}{s}\nn={n}"

    r_display = r_mat.rename(index=WASH_LABELS, columns=OUTCOME_LABELS)
    a_display = annot.rename(index=WASH_LABELS, columns=OUTCOME_LABELS)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        r_display.astype(float),
        annot=a_display, fmt="",
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        ax=ax, linewidths=0.8, annot_kws={"size": 18, "weight": "bold"},
        cbar_kws={"label": "Spearman r", "shrink": 0.75})

    ax.set_title(
        "Spearman Correlation: WASH Indicators vs Child Malnutrition\n"
        " * p<0.05  ** p<0.01  *** p<0.001  ",
        fontsize=16, fontweight="bold", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=15, ha="right", fontsize=14)
    plt.yticks(rotation=0, fontsize=14)
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "wash_correlation_heatmap.pdf")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: wash_correlation_heatmap.png")


# ──────────────────────────────────────────────────────────────
# FIGURE 2: SCATTER PLOTS (3 panels, one per selected indicator)
# ──────────────────────────────────────────────────────────────

def plot_scatter_plots(merged):
    """
    One scatter plot per selected WASH indicator vs stunting.
    Points coloured by UNICEF region.
    """
    print("\n  Generating Figure 2: Scatter plots...")

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    axes = [ax]

    for ax, wash_col in zip(axes, ["san_bas_nat"]):
        outcome = "stunting_national"
        sub = merged[[wash_col, outcome, "region"]].dropna()
        regions = sorted(sub["region"].dropna().unique())

        for region in regions:
            grp   = sub[sub["region"] == region]
            color = REGION_COLORS.get(region, "#999999")
            ax.scatter(grp[wash_col], grp[outcome],
                       c=color, s=45, alpha=0.78,
                       edgecolors="white", linewidth=0.4,
                       label=region)

        x = sub[wash_col].values
        y = sub[outcome].values
        m, b = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, m * x_line + b,
                color="black", lw=1.4, ls="--", alpha=0.65)

        r, p = spearmanr(x, y)
        s = ("***" if p < 0.001 else
             ("**"  if p < 0.01  else
              ("*"   if p < 0.05  else "")))
        ax.annotate(f"r = {r:+.3f}{s}\nn = {len(sub)}",
                    xy=(0.05, 0.95), xycoords="axes fraction",
                    fontsize=10, va="top",
                    bbox=dict(boxstyle="round,pad=0.3",
                               fc="white", alpha=0.85))

        ax.set_xlabel(WASH_LABELS.get(wash_col, wash_col), fontsize=14)
        ax.set_ylabel(OUTCOME_LABELS["stunting_national"], fontsize=14)
        ax.set_title(WASH_LABELS.get(wash_col, wash_col),
                     fontsize=12, fontweight="bold")
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=14)

    handles = [mpatches.Patch(color=REGION_COLORS.get(r, "#999"),
                               label=r)
               for r in sorted(REGION_COLORS.keys())
               if r in merged["region"].values]
    ax.legend(handles=handles, loc="upper right",
              fontsize=10, framealpha=0.85, ncol=1,
              handlelength=1, handleheight=0.8,
              borderpad=0.5, labelspacing=0.3)

    fig.suptitle(
        "WASH Basic Access Indicators vs Child Stunting\n"
        "(Spearman r; points coloured by UNICEF Region)",
        fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "wash_scatter_plots.pdf")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: wash_scatter_plots.png")


# ──────────────────────────────────────────────────────────────
# FIGURE 3: REGIONAL BOXPLOT
# ──────────────────────────────────────────────────────────────

def plot_regional_boxplot(merged):
    """
    Boxplot of all three selected WASH indicators by UNICEF region.
    Regions sorted by median stunting (descending).
    """
    print("\n  Generating Figure 3: Regional boxplot...")

    region_stunt = (merged.groupby("region")["stunting_national"]
                           .median()
                           .sort_values(ascending=False))
    region_order = region_stunt.index.tolist()

    panels = [
        ("wat_bas_nat", "Basic Water Access (%)"),
        ("san_bas_nat", "Basic Sanitation Access (%)"),
        ("hyg_bas_nat", "Basic Hygiene Access (%)"),
    ]
    colors = ["#4e79a7", "#59a14f", "#e15759"]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for ax, (col, label), color in zip(axes, panels, colors):
        data_per_region = [
            merged.loc[merged["region"] == r, col].dropna().values
            for r in region_order
        ]

        bp = ax.boxplot(data_per_region,
                        patch_artist=True,
                        widths=0.55,
                        medianprops=dict(color="black", linewidth=2))
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.70)

        for i, (r, vals) in enumerate(
                zip(region_order, data_per_region), start=1):
            ax.text(i, max(0, ax.get_ylim()[0] - 3),
                    f"n={len(vals)}",
                    ha="center", va="top", fontsize=7, color="#555")

        ax.set_xticks(range(1, len(region_order) + 1))
        ax.set_xticklabels(region_order, rotation=35,
                            ha="right", fontsize=8)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_ylabel("Percentage (%)", fontsize=9)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "WASH Basic Access Indicators by UNICEF Region\n"
        "(Regions sorted by median stunting rate, highest first)",
        fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "wash_regional_boxplot.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: wash_regional_boxplot.png")


def main():
    print("=" * 60)
    print("WASH ANALYSIS: INDICATOR SELECTION + VISUALISATION")
    print("=" * 60)

    for path, label in [(WASH_CSV,         "wash_clean_data.csv"),
                         (MALNUTRITION_CSV, "malnutrition_modelling_sample.csv")]:
        status = "found" if os.path.exists(path) else "NOT FOUND"
        print(f"  {label}: {status}")

    if not (os.path.exists(WASH_CSV) and
            os.path.exists(MALNUTRITION_CSV)):
        print("  One or more input files missing. Check paths.")
        return

    wash, merged = load_data()

    # Section 1: Indicator selection
    indicator_selection(wash, merged)

    # Section 2: Visualisations
    print(f"\n{'='*60}")
    print("SECTION 2: VISUALISATION")
    print(f"{'='*60}")

    plot_correlation_heatmap(merged)
    plot_scatter_plots(merged)
    plot_regional_boxplot(merged)

    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print("\nOutput files:")
    files = ["wash_correlation_heatmap.png",
             "wash_scatter_plots.png",
             "wash_regional_boxplot.png"]
    for f in files:
        full = os.path.join(OUTPUT_DIR, f)
        if os.path.exists(full):
            sz = os.path.getsize(full)
            print(f"  {f:45s} {sz/1024:6.1f} KB")
    csv_out = os.path.join(OUTPUT_CSV_DIR, "wash_selected.csv")
    if os.path.exists(csv_out):
        sz = os.path.getsize(csv_out)
        print(f"  {'outputs/wash_selected.csv':45s} {sz/1024:6.1f} KB")


if __name__ == "__main__":
    main()