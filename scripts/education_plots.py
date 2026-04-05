"""
education_visualisation.py
=============================================================
Education Indicators — Visualisation

Generates three figures for the Data Exploration chapter:

  Figure 1: Correlation heatmap
            Spearman r between all 6 education indicators
            and 3 malnutrition outcomes, with significance stars
            and pairwise n. Sorted by avg|r|.

  Figure 2: Scatter plots
            oos_upsec_f and completion_primary_f vs stunting,
            points coloured by UNICEF region, with regression
            line and Spearman r annotation.

  Figure 3: Regional boxplot
            oos_upsec_f and completion_primary_f distributions
            by UNICEF region, sorted by median stunting rate.

Inputs (same directory as this script):
  education_clean.csv

Also reads (from ../outputs/):
  malnutrition_modelling_sample.csv  (for region and outcomes)

Outputs (same directory as this script):
  education_correlation_heatmap.png
  education_scatter_plots.png
  education_regional_boxplot.png
=============================================================
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
EDUCATION_CSV    = os.path.join(SCRIPT_DIR, "outputs/education_clean.csv")
MALNUTRITION_CSV = os.path.join(SCRIPT_DIR, "outputs",
                                 "malnutrition_modelling_sample.csv")
OUTPUT_DIR       = SCRIPT_DIR  # save figures alongside this script
# ======================

OUTCOMES = ["stunting_national", "wasting_national", "overweight_national"]
OUTCOME_LABELS = {
    "stunting_national":   "Stunting (%)",
    "wasting_national":    "Wasting (%)",
    "overweight_national": "Overweight (%)",
}

EDU_INDICATORS = [
    "oos_upsec_f",
    "completion_primary_f",
    "literacy_f",
    "oos_primary_f",
    "oos_upsec_gap",
    "completion_primary_gap",
]
EDU_LABELS = {
    "oos_upsec_f":           "Female Upper-Secondary OOS Rate (%)",
    "completion_primary_f":  "Female Primary Completion Rate (%)",
    "literacy_f":            "Female Youth Literacy Rate (%)",
    "oos_primary_f":         "Female Primary OOS Rate (%)",
    "oos_upsec_gap":         "Upper-Secondary OOS Gender Gap (F-M, pp)",
    "completion_primary_gap":"Primary Completion Gender Gap (F-M, pp)",
}

REGION_LABELS = {
    "SA":   "South Asia",
    "WCA":  "West/Central Africa",
    "ESA":  "East/Southern Africa",
    "MENA": "Middle East & N. Africa",
    "EAP":  "East Asia & Pacific",
    "LAC":  "Latin America & Carib.",
    "EECA": "E. Europe & Central Asia",
    "WE":   "Western Europe",
    "NA ":  "North America",
}

REGION_COLORS = {
    "South Asia":              "#d62728",
    "West/Central Africa":     "#ff7f0e",
    "East/Southern Africa":    "#e377c2",
    "Middle East & N. Africa": "#8c564b",
    "East Asia & Pacific":     "#2ca02c",
    "Latin America & Carib.":  "#17becf",
    "E. Europe & Central Asia":"#1f77b4",
    "Western Europe":          "#7f7f7f",
    "North America":           "#bcbd22",
}


# ──────────────────────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────────────────────

def load_data():
    edu = pd.read_csv(EDUCATION_CSV)
    mal = pd.read_csv(MALNUTRITION_CSV)

    merged = mal.merge(
        edu.drop(columns=["country"], errors="ignore"),
        on="ISO", how="inner")

    # Map region codes to readable labels
    merged["region"] = (merged["unicef_region"]
                        .map(REGION_LABELS)
                        .fillna(merged["unicef_region"]))

    print(f"  Merged dataset: {len(merged)} countries")
    print(f"  Regions: {sorted(merged['region'].dropna().unique())}")
    return merged


# ──────────────────────────────────────────────────────────────
# FIGURE 1: CORRELATION HEATMAP
# ──────────────────────────────────────────────────────────────

def plot_correlation_heatmap(merged):
    """
    Spearman correlation heatmap between all 6 education indicators
    and 3 malnutrition outcomes.
    Indicators sorted by average |r| across outcomes.
    Each cell shows r, significance stars, and pairwise n.
    """
    print("\n  Generating Figure 1: Correlation heatmap...")

    # Compute correlations and sort by avg|r|
    indicators = [i for i in EDU_INDICATORS if i in merged.columns]
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

    # Sort rows by average |r|
    avg_r = r_mat.abs().mean(axis=1).sort_values(ascending=False)
    r_mat = r_mat.loc[avg_r.index]
    p_mat = p_mat.loc[avg_r.index]
    n_mat = n_mat.loc[avg_r.index]

    # Build annotation strings
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

    r_display = r_mat.rename(index=EDU_LABELS, columns=OUTCOME_LABELS)
    annot_display = annot.rename(index=EDU_LABELS, columns=OUTCOME_LABELS)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        r_display.astype(float),
        annot=annot_display, fmt="",
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        ax=ax, linewidths=0.8, annot_kws={"size": 9},
        cbar_kws={"label": "Spearman r", "shrink": 0.75})
    ax.set_title(
        "Spearman Correlation: Education Indicators vs Child Malnutrition\n"
        "Sorted by avg |r|  |  * p<0.05  ** p<0.01  *** p<0.001",
        fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=15, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "outputs/education_correlation_heatmap.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: education_correlation_heatmap.png")


# ──────────────────────────────────────────────────────────────
# FIGURE 2: SCATTER PLOTS
# ──────────────────────────────────────────────────────────────

def plot_scatter_plots(merged):
    """
    Scatter plots for the two strongest female indicators
    (oos_upsec_f and completion_primary_f) vs stunting.
    Points coloured by UNICEF region.
    Includes OLS regression line and Spearman r annotation.
    """
    print("\n  Generating Figure 2: Scatter plots...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    panels = [
        ("oos_upsec_f",          "stunting_national"),
        ("completion_primary_f", "stunting_national"),
    ]

    for ax, (edu_col, outcome) in zip(axes, panels):
        sub = merged[[edu_col, outcome, "region"]].dropna()

        regions = sorted(sub["region"].dropna().unique())

        for region in regions:
            grp   = sub[sub["region"] == region]
            color = REGION_COLORS.get(region, "#999999")
            ax.scatter(grp[edu_col], grp[outcome],
                       c=color, s=45, alpha=0.78,
                       edgecolors="white", linewidth=0.4,
                       label=region)

        # Regression line
        x = sub[edu_col].values
        y = sub[outcome].values
        m, b = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, m * x_line + b,
                color="black", lw=1.4, ls="--", alpha=0.65)

        # Spearman annotation
        r, p = spearmanr(x, y)
        s = "***" if p < 0.001 else ("**" if p < 0.01 else
            ("*" if p < 0.05 else ""))
        ax.annotate(f"r = {r:+.3f}{s}\nn = {len(sub)}",
                    xy=(0.05, 0.95), xycoords="axes fraction",
                    fontsize=9, va="top",
                    bbox=dict(boxstyle="round,pad=0.3",
                               fc="white", alpha=0.85))

        ax.set_xlabel(EDU_LABELS.get(edu_col, edu_col), fontsize=10)
        ax.set_ylabel(OUTCOME_LABELS.get(outcome, outcome), fontsize=10)
        ax.set_title(
            f"{EDU_LABELS.get(edu_col, edu_col)}\nvs "
            f"{OUTCOME_LABELS.get(outcome, outcome)}",
            fontsize=10, fontweight="bold")
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=8)

    # Shared legend (regions)
    handles = [mpatches.Patch(color=REGION_COLORS.get(r, "#999"),
                               label=r)
               for r in sorted(REGION_COLORS.keys())
               if r in merged["region"].values]
    fig.legend(handles=handles, loc="lower center",
               ncol=3, fontsize=8, title="UNICEF Region",
               title_fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.12))

    fig.suptitle(
        "Female Education Indicators vs Child Stunting\n"
        "(Spearman r; points coloured by UNICEF Region)",
        fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "outputs/education_scatter_plots.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: education_scatter_plots.png")


# ──────────────────────────────────────────────────────────────
# FIGURE 3: REGIONAL BOXPLOT
# ──────────────────────────────────────────────────────────────

def plot_regional_boxplot(merged):
    """
    Boxplot of oos_upsec_f and completion_primary_f by UNICEF region.
    Regions sorted by median stunting rate (descending) to align
    with the malnutrition burden narrative.
    """
    print("\n  Generating Figure 3: Regional boxplot...")

    # Sort regions by median stunting
    region_stunt = (merged.groupby("region")["stunting_national"]
                           .median()
                           .sort_values(ascending=False))
    region_order = region_stunt.index.tolist()

    panels = [
        ("oos_upsec_f",         "Female Upper-Secondary OOS Rate (%)"),
        ("completion_primary_f","Female Primary Completion Rate (%)"),
    ]

    colors = ["#e15759", "#4e79a7"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

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

        # Annotate n per region
        for i, (r, vals) in enumerate(
                zip(region_order, data_per_region), start=1):
            ax.text(i, ax.get_ylim()[0] - 2, f"n={len(vals)}",
                    ha="center", va="top", fontsize=7, color="#555")

        ax.set_xticks(range(1, len(region_order) + 1))
        ax.set_xticklabels(region_order, rotation=35,
                            ha="right", fontsize=8)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_ylabel("Percentage (%)", fontsize=9)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Female Education Indicators by UNICEF Region\n"
        "(Regions sorted by median stunting rate, highest first)",
        fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "outputs/education_regional_boxplot.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: education_regional_boxplot.png")


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("EDUCATION VISUALISATION")
    print("=" * 60)

    for path, label in [(EDUCATION_CSV,    "outputs/education_clean.csv"),
                         (MALNUTRITION_CSV, "malnutrition_modelling_sample.csv")]:
        status = "found" if os.path.exists(path) else "NOT FOUND"
        print(f"  {label}: {status}")

    if not (os.path.exists(EDUCATION_CSV) and
            os.path.exists(MALNUTRITION_CSV)):
        print("  One or more input files missing. Check paths.")
        return

    merged = load_data()

    plot_correlation_heatmap(merged)
    plot_scatter_plots(merged)
    plot_regional_boxplot(merged)

    print(f"\n{'='*60}")
    print("VISUALISATION COMPLETE")
    print(f"{'='*60}")
    print("\nOutput files:")
    for f in ["outputs/education_correlation_heatmap.png",
              "outputs/education_scatter_plots.png",
              "outputs/education_regional_boxplot.png"]:
        full = os.path.join(OUTPUT_DIR, f)
        if os.path.exists(full):
            sz = os.path.getsize(full)
            print(f"  {f:45s} {sz/1024:6.1f} KB")


if __name__ == "__main__":
    main()