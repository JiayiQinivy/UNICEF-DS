"""
OLS Sensitivity Analysis for Stunting Model 4

Tests whether the sanitation finding is robust to alternative
indicator choices within each domain.

Four specifications (stunting only):
  Main:   female_married_by_18 + literacy_f    + san_bas_nat + income_group
  Spec A: female_married_by_18 + oos_upsec_f   + san_bas_nat + income_group
  Spec B: marriage_gap_18      + literacy_f    + san_bas_nat + income_group
  Spec C: female_married_by_18 + literacy_f    + wat_bas_nat + income_group

Each specification uses its own complete-case sample.

Input:  Model3/model3_analytical_dataset.csv
Output: Model3/model3_sensitivity_results.csv
        Model3/model3_sensitivity_plot.png
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CSV   = os.path.join(SCRIPT_DIR, "model3_analytical_dataset.csv")
OUTPUT_DIR = SCRIPT_DIR──
SPECIFICATIONS = {
    "Main (literacy_f / san_bas_nat)":
        {
            "formula": "stunting_national ~ female_married_by_18"
                       " + literacy_f + san_bas_nat + C(income_group)",
            "wash_pred":  "san_bas_nat",
            "edu_pred":   "literacy_f",
            "gender_pred":"female_married_by_18",
        },
    "Spec A (oos_upsec_f / san_bas_nat)":
        {
            "formula": "stunting_national ~ female_married_by_18"
                       " + oos_upsec_f + san_bas_nat + C(income_group)",
            "wash_pred":  "san_bas_nat",
            "edu_pred":   "oos_upsec_f",
            "gender_pred":"female_married_by_18",
        },
    "Spec B (literacy_f / san_bas_nat / marriage_gap)":
        {
            "formula": "stunting_national ~ marriage_gap_18"
                       " + literacy_f + san_bas_nat + C(income_group)",
            "wash_pred":  "san_bas_nat",
            "edu_pred":   "literacy_f",
            "gender_pred":"marriage_gap_18",
        },
    "Spec C (literacy_f / wat_bas_nat)":
        {
            "formula": "stunting_national ~ female_married_by_18"
                       " + literacy_f + wat_bas_nat + C(income_group)",
            "wash_pred":  "wat_bas_nat",
            "edu_pred":   "literacy_f",
            "gender_pred":"female_married_by_18",
        },
}

WASH_LABELS = {
    "san_bas_nat": "Basic Sanitation Access",
    "wat_bas_nat": "Basic Water Access",
}

SIG_STARS = {
    0.001: "***",
    0.01:  "**",
    0.05:  "*",
    1.0:   "",
}

def sig_label(p):
    for threshold, stars in SIG_STARS.items():
        if p < threshold:
            return stars
    return ""



def run_sensitivity(df):
    print("=" * 60)
    print("OLS SENSITIVITY ANALYSIS — Stunting")
    print("=" * 60)

    results = []

    for spec_name, spec in SPECIFICATIONS.items():
        formula      = spec["formula"]
        wash_pred    = spec["wash_pred"]
        edu_pred     = spec["edu_pred"]
        gender_pred  = spec["gender_pred"]

        vars_needed = [
            "stunting_national", "income_group",
            wash_pred, edu_pred, gender_pred
        ]
        vars_needed = [v for v in vars_needed if v in df.columns]

        sub = df[vars_needed].dropna().copy()
        n   = len(sub)

        model = smf.ols(formula, data=sub).fit()

        wash_coef  = model.params.get(wash_pred, np.nan)
        wash_se   = model.bse.get(wash_pred, np.nan)
        wash_p    = model.pvalues.get(wash_pred, np.nan)
        wash_ci_lo = wash_coef - 1.96 * wash_se
        wash_ci_hi = wash_coef + 1.96 * wash_se
        wash_sig   = sig_label(wash_p)

        adj_r2 = model.rsquared_adj
        aic    = model.aic

        print(f"\n  {spec_name}")
        print(f"    n = {n}")
        print(f"    WASH predictor: {WASH_LABELS.get(wash_pred, wash_pred)}")
        print(f"    β = {wash_coef:.3f}  SE = {wash_se:.3f}  "
              f"p = {wash_p:.4f} {wash_sig}")
        print(f"    95% CI: [{wash_ci_lo:.3f}, {wash_ci_hi:.3f}]")
        print(f"    Adj. R² = {adj_r2:.3f}  AIC = {aic:.1f}")

        results.append({
            "Specification":  spec_name,
            "N":              n,
            "Gender":         gender_pred,
            "Education":      edu_pred,
            "WASH":           wash_pred,
            "WASH_label":     WASH_LABELS.get(wash_pred, wash_pred),
            "Beta":           round(wash_coef, 3),
            "SE":             round(wash_se, 3),
            "CI_lo":          round(wash_ci_lo, 3),
            "CI_hi":          round(wash_ci_hi, 3),
            "p_value":        round(wash_p, 4),
            "Sig":            wash_sig,
            "Adj_R2":         round(adj_r2, 3),
            "AIC":            round(aic, 1),
        })

    return pd.DataFrame(results)



def plot_sensitivity(results_df):
    fig, ax = plt.subplots(figsize=(9, 5))

    y_labels = []
    for i, row in results_df.iterrows():
        spec_short = row["Specification"]
        label = f"{spec_short}\n(n={row['N']}, {row['WASH_label']})"
        y_labels.append(label)

    y_pos = np.arange(len(results_df))[::-1]

    colors = {"san_bas_nat": "#59a14f", "wat_bas_nat": "#76b7b2"}

    for y, (_, row) in zip(y_pos, results_df.iterrows()):
        color = colors.get(row["WASH"], "#888888")
        sig = row["p_value"] < 0.05
        ci_lo, ci_hi, coef = row["CI_lo"], row["CI_hi"], row["Beta"]

        ax.plot([ci_lo, ci_hi], [y, y], color=color, alpha=0.8, linewidth=2.5)
        ax.scatter(coef, y, color=color, s=100, zorder=3,
                   edgecolors="black" if sig else "none")

        ax.text(
            ci_lo - 0.015, y,
            f"β={coef:.3f}\np={row['p_value']:.3f}",
            va="center", ha="right",
            fontsize=8, color="#333"
        )

        if sig and row["Sig"]:
            ax.text(ci_hi + 0.01, y, row["Sig"], va="center",
                    ha="left", fontsize=10, color=color, fontweight="bold")

    ax.axvline(0, color="#333", lw=1, ls="--", alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=9)

    current_xlim = ax.get_xlim()
    ax.set_xlim(current_xlim[0] - 0.05, current_xlim[1] + 0.05)

    ax.set_xlabel("OLS Coefficient (WASH predictor)", fontsize=10, labelpad=10)
    ax.set_title("Sensitivity Analysis: WASH Coefficient for Stunting",
                 fontsize=12, fontweight="bold", pad=20)

    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.2, linestyle=":")

    # Expand left margin for long labels
    plt.subplots_adjust(left=0.35, right=0.9, top=0.85, bottom=0.15)

    out = os.path.join(OUTPUT_DIR, "model3_sensitivity_plot.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"\n Saved: model3_sensitivity_plot.png")



def main():
    df = pd.read_csv(DATA_CSV)
    print(f"  Dataset loaded: {df.shape[0]} countries, "
          f"{df.shape[1]} columns\n")

    results_df = run_sensitivity(df)

    # Save CSV
    out_csv = os.path.join(OUTPUT_DIR, "model3_sensitivity_results.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"\n  Saved: model3_sensitivity_results.csv")

    # Plot
    plot_sensitivity(results_df)

    # Summary
    print("\n" + "=" * 60)
    print("SENSITIVITY SUMMARY")
    print("=" * 60)
    print(f"\n{'Specification':<45} {'n':>4} {'Beta':>7} "
          f"{'p':>7} {'Sig':>4} {'Adj.R²':>7}")
    print("─" * 75)
    for _, row in results_df.iterrows():
        spec = row["Specification"][:44]
        print(f"  {spec:<44} {row['N']:>4} "
              f"{row['Beta']:>7.3f} {row['p_value']:>7.4f} "
              f"{row['Sig']:>4} {row['Adj_R2']:>7.3f}")


if __name__ == "__main__":
    main()
