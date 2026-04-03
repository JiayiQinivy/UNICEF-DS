"""
gender_malnutrition_analysis.py
=============================================================
Analysis: Does Gender Inequality Explain Child Malnutrition?

This script implements Model 1 and Model 2 of the four-model
incremental framework:

  Model 1 — Baseline: stunting/wasting/overweight ~ income_group
  Model 2 — Gender:   stunting/wasting/overweight ~ gender_indicators

Analysis layers:
  1. Data merge and sample description
  2. Exploratory correlation analysis (Spearman + heatmap)
  3. Scatter plots (each gender indicator vs each outcome)
  4. Inferential statistics (Spearman r, p-values, Bonferroni)
  5. Partial correlation (controlling for income group)
  6. OLS regression (Model 1 baseline, Model 2 gender-only)
  7. SHAP variable importance
  8. Visualisation of regression results

Outputs:
  final_analytical_dataset.csv
  correlation_heatmap.png
  scatter_matrix.png
  regression_coefficients.png
  shap_importance.png
  partial_correlation_table.csv
  model_comparison_table.csv

Note on causal language:
  This is cross-sectional data. We establish statistical association,
  not causation. Causal mechanisms are discussed in the report.
=============================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import statsmodels.api as sm
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")

# ===== USER INPUT =====
OUTPUT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

GENDER_CSV      = os.path.join(OUTPUT_FOLDER, "gender_inequality_analysis.csv")
MALNUTRITION_CSV = os.path.join(OUTPUT_FOLDER, "malnutrition_modelling_sample.csv")
# ======================

# Gender inequality predictors used in models
GENDER_PREDICTORS = [
    "female_married_by_18",
    "female_married_by_15",
    "marriage_gap_18",
    "anc4_15_19_pct",
    "modern_contraceptive_pct",
]

# FGM is kept separate due to limited coverage (30 countries)
FGM_PREDICTOR = "fgm_prevalence_pct"

# Malnutrition outcome variables
OUTCOMES = [
    "stunting_national",
    "wasting_national",
    "overweight_national",
]

OUTCOME_LABELS = {
    "stunting_national":   "Stunting (%)",
    "wasting_national":    "Wasting (%)",
    "overweight_national": "Overweight (%)",
}

PREDICTOR_LABELS = {
    "female_married_by_18":    "Female Child Marriage by 18 (%)",
    "female_married_by_15":    "Female Child Marriage by 15 (%)",
    "marriage_gap_18":         "Marriage Gender Gap (F - M, pp)",
    "anc4_15_19_pct":          "Antenatal Care 4+ visits, 15-19 yrs (%)",
    "modern_contraceptive_pct":"Modern Contraceptive Use, 15-19 yrs (%)",
    "fgm_prevalence_pct":      "FGM Prevalence (%)",
}


# ──────────────────────────────────────────────────────────────
# SECTION 1: MERGE DATASETS
# ──────────────────────────────────────────────────────────────

def load_and_merge():
    """
    Merge gender_inequality_analysis.csv with
    malnutrition_modelling_sample.csv on ISO3 code.
    Reports sample sizes before and after merge.
    """
    print("=" * 60)
    print("SECTION 1: DATA MERGE")
    print("=" * 60)

    gender = pd.read_csv(GENDER_CSV)
    mal    = pd.read_csv(MALNUTRITION_CSV)

    print(f"  Gender dataset   : {len(gender)} countries, "
          f"{len(gender.columns)} columns")
    print(f"  Malnutrition     : {len(mal)} countries, "
          f"{len(mal.columns)} columns")

    # Merge on ISO
    df = mal.merge(gender.drop(columns=["country"], errors="ignore"),
                   on="ISO", how="inner")
    print(f"  After inner merge: {len(df)} countries")

    # Report per-variable sample sizes
    all_vars = OUTCOMES + GENDER_PREDICTORS + [FGM_PREDICTOR]
    print(f"\n  Variable coverage in merged dataset:")
    for col in all_vars:
        if col in df.columns:
            n = df[col].notna().sum()
            print(f"    {col:40s}: {n:3d} countries")

    # Save final analytical dataset
    out = os.path.join(OUTPUT_FOLDER, "final_analytical_dataset.csv")
    df.to_csv(out, index=False)
    print(f"\n  Saved: final_analytical_dataset.csv ({len(df)} countries, "
          f"{len(df.columns)} columns)")

    return df


# ──────────────────────────────────────────────────────────────
# SECTION 2: SPEARMAN CORRELATION WITH BONFERRONI CORRECTION
# ──────────────────────────────────────────────────────────────

def spearman_correlation_table(df):
    """
    Compute Spearman correlation between all gender predictors
    and all three outcomes.

    Applies Bonferroni correction for multiple comparisons
    (n_tests = n_predictors x n_outcomes).

    Returns a summary DataFrame for reporting.
    """
    print("\n" + "=" * 60)
    print("SECTION 2: SPEARMAN CORRELATION + BONFERRONI CORRECTION")
    print("=" * 60)

    predictors = [p for p in GENDER_PREDICTORS + [FGM_PREDICTOR]
                  if p in df.columns]
    n_tests = len(predictors) * len(OUTCOMES)
    alpha   = 0.05
    bonf    = alpha / n_tests
    print(f"  Number of tests: {n_tests} | "
          f"Bonferroni threshold: p < {bonf:.4f}")

    rows = []
    for outcome in OUTCOMES:
        if outcome not in df.columns:
            continue
        for pred in predictors:
            if pred not in df.columns:
                continue
            sub = df[[outcome, pred]].dropna()
            n   = len(sub)
            if n < 10:
                continue
            r, p = spearmanr(sub[outcome], sub[pred])
            rows.append({
                "Outcome":          OUTCOME_LABELS.get(outcome, outcome),
                "Predictor":        PREDICTOR_LABELS.get(pred, pred),
                "N":                n,
                "Spearman_r":       round(r, 3),
                "p_value":          round(p, 4),
                "Significant_0.05": "Yes" if p < alpha else "No",
                "Significant_Bonf": "Yes" if p < bonf  else "No",
            })

    corr_table = pd.DataFrame(rows)
    print(f"\n  Results (sorted by |Spearman r|):")
    display_df = corr_table.sort_values(
        "Spearman_r", key=abs, ascending=False)
    print(display_df.to_string(index=False))

    out = os.path.join(OUTPUT_FOLDER, "spearman_correlation_table.csv")
    corr_table.to_csv(out, index=False)
    print(f"\n  Saved: spearman_correlation_table.csv")

    return corr_table


# ──────────────────────────────────────────────────────────────
# SECTION 3: CORRELATION HEATMAP
# ──────────────────────────────────────────────────────────────

def plot_correlation_heatmap(df):
    """
    Heatmap of Spearman correlations between gender indicators
    and malnutrition outcomes. Shows r values with significance stars.
    * p < 0.05  ** p < 0.01  *** p < 0.001
    """
    print("\n  Generating correlation heatmap...")

    predictors = [p for p in GENDER_PREDICTORS + [FGM_PREDICTOR]
                  if p in df.columns]
    outcomes   = [o for o in OUTCOMES if o in df.columns]

    r_matrix = pd.DataFrame(index=predictors, columns=outcomes,
                             dtype=float)
    p_matrix = pd.DataFrame(index=predictors, columns=outcomes,
                             dtype=float)
    n_matrix = pd.DataFrame(index=predictors, columns=outcomes,
                             dtype=int)

    for pred in predictors:
        for outcome in outcomes:
            sub = df[[pred, outcome]].dropna()
            if len(sub) < 10:
                r_matrix.loc[pred, outcome] = np.nan
                p_matrix.loc[pred, outcome] = np.nan
                n_matrix.loc[pred, outcome] = len(sub)
            else:
                r, p = spearmanr(sub[pred], sub[outcome])
                r_matrix.loc[pred, outcome] = round(r, 3)
                p_matrix.loc[pred, outcome] = p
                n_matrix.loc[pred, outcome] = len(sub)

    # Build annotation matrix with stars
    annot = pd.DataFrame(index=predictors, columns=outcomes, dtype=str)
    for pred in predictors:
        for outcome in outcomes:
            r = r_matrix.loc[pred, outcome]
            p = p_matrix.loc[pred, outcome]
            n = n_matrix.loc[pred, outcome]
            if pd.isna(r):
                annot.loc[pred, outcome] = f"n={n}"
            else:
                stars = ""
                if p < 0.001: stars = "***"
                elif p < 0.01: stars = "**"
                elif p < 0.05: stars = "*"
                annot.loc[pred, outcome] = f"{r:.2f}{stars}\nn={n}"

    # Rename for display
    r_display = r_matrix.rename(
        index=PREDICTOR_LABELS, columns=OUTCOME_LABELS)
    annot_display = annot.rename(
        index=PREDICTOR_LABELS, columns=OUTCOME_LABELS)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        r_display.astype(float),
        annot=annot_display, fmt="",
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        ax=ax, linewidths=0.8,
        annot_kws={"size": 9},
        cbar_kws={"label": "Spearman r", "shrink": 0.8}
    )
    ax.set_title(
        "Spearman Correlation: Gender Inequality vs Child Malnutrition\n"
        "* p<0.05  ** p<0.01  *** p<0.001  (n = pairwise complete cases)",
        fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=20, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    out = os.path.join(OUTPUT_FOLDER, "correlation_heatmap.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: correlation_heatmap.png")

    return r_matrix, p_matrix


# ──────────────────────────────────────────────────────────────
# SECTION 4: SCATTER MATRIX
# ──────────────────────────────────────────────────────────────

def plot_scatter_matrix(df):
    """
    Scatter plots: each gender predictor vs each malnutrition outcome.
    Points coloured by income group. Includes regression line and r/p.
    """
    print("\n  Generating scatter matrix...")

    predictors = [p for p in GENDER_PREDICTORS if p in df.columns]
    outcomes   = [o for o in OUTCOMES if o in df.columns]

    n_rows = len(outcomes)
    n_cols = len(predictors)

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4.5 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    income_palette = {
        "Low Income":          "#d62728",
        "Lower Middle Income": "#ff7f0e",
        "Upper Middle Income": "#2ca02c",
        "High Income":         "#1f77b4",
    }

    for i, outcome in enumerate(outcomes):
        for j, pred in enumerate(predictors):
            ax  = axes[i][j]
            sub = df[[pred, outcome, "income_group"]].dropna()

            if len(sub) < 5:
                ax.set_visible(False)
                continue

            # Plot by income group
            for ig, grp in sub.groupby("income_group"):
                color = income_palette.get(ig, "#888888")
                ax.scatter(grp[pred], grp[outcome],
                           c=color, s=35, alpha=0.7,
                           edgecolors="white", linewidth=0.3,
                           label=ig)

            # Regression line
            r, p = spearmanr(sub[pred], sub[outcome])
            x = sub[pred].values
            y = sub[outcome].values
            m, b = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, m * x_line + b,
                    color="black", lw=1.2, ls="--", alpha=0.7)

            # Annotation
            stars = ""
            if p < 0.001:   stars = "***"
            elif p < 0.01:  stars = "**"
            elif p < 0.05:  stars = "*"
            ax.annotate(f"r={r:.2f}{stars}\nn={len(sub)}",
                        xy=(0.05, 0.95), xycoords="axes fraction",
                        fontsize=8, va="top",
                        bbox=dict(boxstyle="round,pad=0.3",
                                   fc="white", alpha=0.8))

            # Labels only on edges
            if i == n_rows - 1:
                ax.set_xlabel(
                    PREDICTOR_LABELS.get(pred, pred),
                    fontsize=8)
            if j == 0:
                ax.set_ylabel(
                    OUTCOME_LABELS.get(outcome, outcome),
                    fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.25)

    # Legend (income groups) — top right
    handles = [plt.scatter([], [], c=c, s=40, label=l)
               for l, c in income_palette.items()]
    fig.legend(handles=handles, loc="upper right",
               fontsize=8, title="Income Group",
               title_fontsize=8, framealpha=0.9)

    fig.suptitle(
        "Gender Inequality Indicators vs Child Malnutrition Outcomes\n"
        "(Spearman r; * p<0.05  ** p<0.01  *** p<0.001)",
        fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    out = os.path.join(OUTPUT_FOLDER, "scatter_matrix.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: scatter_matrix.png")


# ──────────────────────────────────────────────────────────────
# SECTION 5: PARTIAL CORRELATION
# Controlling for income group
# ──────────────────────────────────────────────────────────────

def partial_correlation_controlling_income(df):
    """
    Compute partial Spearman correlation between each gender predictor
    and each outcome, controlling for income group.

    Method: regress both predictor and outcome on income group dummies,
    take residuals, then compute Spearman r on residuals.
    This removes the confounding effect of economic development.
    """
    print("\n" + "=" * 60)
    print("SECTION 5: PARTIAL CORRELATION (controlling for income group)")
    print("=" * 60)

    # Dummy-encode income group
    income_dummies = pd.get_dummies(
        df["income_group"], prefix="ig", drop_first=True)

    predictors = [p for p in GENDER_PREDICTORS if p in df.columns]
    rows = []

    for outcome in OUTCOMES:
        if outcome not in df.columns:
            continue
        for pred in predictors:
            if pred not in df.columns:
                continue

            sub = df[[pred, outcome, "income_group"]].dropna()
            if len(sub) < 15:
                continue

            # Merge dummies
            sub = sub.join(
                pd.get_dummies(sub["income_group"],
                               prefix="ig", drop_first=True),
                how="left")
            ig_cols = [c for c in sub.columns if c.startswith("ig_")]

            def residualise(var):
                X = sm.add_constant(sub[ig_cols].astype(float))
                y = sub[var].astype(float)
                model = sm.OLS(y, X).fit()
                return model.resid.values

            resid_pred    = residualise(pred)
            resid_outcome = residualise(outcome)

            r_raw,  p_raw  = spearmanr(sub[pred],   sub[outcome])
            r_part, p_part = spearmanr(resid_pred, resid_outcome)

            rows.append({
                "Outcome":        OUTCOME_LABELS.get(outcome, outcome),
                "Predictor":      PREDICTOR_LABELS.get(pred, pred),
                "N":              len(sub),
                "r_unadjusted":   round(r_raw,  3),
                "p_unadjusted":   round(p_raw,  4),
                "r_partial":      round(r_part, 3),
                "p_partial":      round(p_part, 4),
                "Attenuated":     "Yes" if abs(r_part) < abs(r_raw) else "No",
            })

    part_df = pd.DataFrame(rows)
    print(part_df.to_string(index=False))

    out = os.path.join(OUTPUT_FOLDER, "partial_correlation_table.csv")
    part_df.to_csv(out, index=False)
    print(f"\n  Saved: partial_correlation_table.csv")

    return part_df


# ──────────────────────────────────────────────────────────────
# SECTION 6: OLS REGRESSION MODELS
# Model 1: Baseline (income group only)
# Model 2: Gender indicators only
# ──────────────────────────────────────────────────────────────

def run_regression_models(df):
    """
    For each outcome (stunting, wasting, overweight):

    Model 1 — Baseline:
      outcome ~ C(income_group)
      Establishes how much variance is explained by economic development

    Model 2 — Gender only:
      outcome ~ gender_predictors
      Tests the incremental explanatory power of gender inequality

    Reports: Adjusted R², AIC, BIC, coefficients, p-values.
    Compares models using F-test (partial F-test).
    """
    print("\n" + "=" * 60)
    print("SECTION 6: OLS REGRESSION MODELS")
    print("=" * 60)

    all_results = []
    model_comparison = []

    for outcome in OUTCOMES:
        if outcome not in df.columns:
            continue

        print(f"\n  Outcome: {OUTCOME_LABELS[outcome]}")
        print(f"  {'─'*50}")

        predictors = [p for p in GENDER_PREDICTORS if p in df.columns]
        all_cols   = [outcome, "income_group"] + predictors
        sub        = df[all_cols].dropna()
        print(f"  Sample size (complete cases): {len(sub)}")

        if len(sub) < 20:
            print(f"  Sample too small for regression.")
            continue

        # ── Model 1: Baseline ────────────────────────────────
        formula1 = f"{outcome} ~ C(income_group)"
        m1 = smf.ols(formula1, data=sub).fit()
        print(f"\n  Model 1 (Baseline — income group only):")
        print(f"    Adj. R²: {m1.rsquared_adj:.3f} | "
              f"AIC: {m1.aic:.1f} | N: {int(m1.nobs)}")

        # ── Model 2: Gender indicators ────────────────────────
        pred_str  = " + ".join(predictors)
        formula2  = f"{outcome} ~ {pred_str}"
        m2 = smf.ols(formula2, data=sub).fit()
        print(f"\n  Model 2 (Gender indicators only):")
        print(f"    Adj. R²: {m2.rsquared_adj:.3f} | "
              f"AIC: {m2.aic:.1f} | N: {int(m2.nobs)}")
        print(f"\n  Coefficients (Model 2):")
        coef_df = pd.DataFrame({
            "Coefficient": m2.params.round(3),
            "Std. Error":  m2.bse.round(3),
            "t":           m2.tvalues.round(3),
            "p-value":     m2.pvalues.round(4),
            "Significant": m2.pvalues.apply(
                lambda p: "***" if p < 0.001
                else ("**" if p < 0.01
                      else ("*" if p < 0.05 else ""))),
        })
        print(coef_df.to_string())

        # Store model comparison row
        model_comparison.append({
            "Outcome":        OUTCOME_LABELS[outcome],
            "Model":          "Model 1 (Baseline)",
            "N":              int(m1.nobs),
            "R2":             round(m1.rsquared, 3),
            "Adj_R2":         round(m1.rsquared_adj, 3),
            "AIC":            round(m1.aic, 1),
            "BIC":            round(m1.bic, 1),
        })
        model_comparison.append({
            "Outcome":        OUTCOME_LABELS[outcome],
            "Model":          "Model 2 (Gender only)",
            "N":              int(m2.nobs),
            "R2":             round(m2.rsquared, 3),
            "Adj_R2":         round(m2.rsquared_adj, 3),
            "AIC":            round(m2.aic, 1),
            "BIC":            round(m2.bic, 1),
        })

        # Store coefficients for visualisation
        for var, row in coef_df.iterrows():
            if var == "Intercept":
                continue
            all_results.append({
                "Outcome":     OUTCOME_LABELS[outcome],
                "Predictor":   PREDICTOR_LABELS.get(
                    var, var.replace("_", " ")),
                "Coefficient": row["Coefficient"],
                "SE":          row["Std. Error"],
                "p_value":     row["p-value"],
                "Significant": row["Significant"],
            })

    # Save model comparison table
    comp_df = pd.DataFrame(model_comparison)
    out = os.path.join(OUTPUT_FOLDER, "model_comparison_table.csv")
    comp_df.to_csv(out, index=False)
    print(f"\n  Saved: model_comparison_table.csv")

    results_df = pd.DataFrame(all_results)
    return results_df, comp_df

def run_model_3(df):
    """
    Model 3 — Integrated:
      outcome ~ gender_predictors + wash_indicators + education_indicators
      Tests the explanatory power of gender inequality along with WASH and education indicators.
    """
    print("\n" + "=" * 60)
    print("SECTION 6B: OLS REGRESSION MODEL 3 (Integrated)")
    print("=" * 60)

    all_results = []
    model_comparison = []

    # Extra predictors for Model 3
    wash_predictors = ["wat_bas_nat", "san_bas_nat", "hyg_bas_nat"]
    edu_predictors = ["completion_primary_f", "literacy_f"]

    # We need to load and merge these datasets first
    WASH_CSV = os.path.join(OUTPUT_FOLDER, "WASH_clean.csv")
    EDU_CSV = os.path.join(OUTPUT_FOLDER, "education_clean.csv")

    wash_df = pd.read_csv(WASH_CSV) if os.path.exists(WASH_CSV) else pd.DataFrame(columns=["iso3"] + wash_predictors)
    edu_df = pd.read_csv(EDU_CSV) if os.path.exists(EDU_CSV) else pd.DataFrame(columns=["ISO"] + edu_predictors)

    # Make ISO column consistent
    if not wash_df.empty and "iso3" in wash_df.columns:
        wash_df = wash_df.rename(columns={"iso3": "ISO"})

    # Filter wash_df for latest year per country
    if not wash_df.empty and "year" in wash_df.columns:
        wash_df = wash_df.sort_values("year").groupby("ISO").tail(1)

    df_m3 = df.copy()
    if not wash_df.empty:
        df_m3 = df_m3.merge(wash_df[["ISO"] + [p for p in wash_predictors if p in wash_df.columns]], on="ISO", how="left")
    if not edu_df.empty:
        df_m3 = df_m3.merge(edu_df[["ISO"] + [p for p in edu_predictors if p in edu_df.columns]], on="ISO", how="left")

    for outcome in OUTCOMES:
        if outcome not in df_m3.columns:
            continue

        print(f"\n  Outcome: {OUTCOME_LABELS[outcome]}")
        print(f"  {'─'*50}")

        base_predictors = [p for p in GENDER_PREDICTORS if p in df_m3.columns]
        extra_predictors = [p for p in wash_predictors + edu_predictors if p in df_m3.columns]
        all_predictors = base_predictors + extra_predictors

        all_cols   = [outcome, "income_group"] + all_predictors
        sub        = df_m3[all_cols].dropna()
        print(f"  Sample size (complete cases): {len(sub)}")

        if len(sub) < 10:
            print(f"  Sample too small for regression.")
            continue

        # ── Model 3: Integrated ────────────────────────
        pred_str  = " + ".join(all_predictors)
        formula3  = f"{outcome} ~ {pred_str}"
        m3 = smf.ols(formula3, data=sub).fit()
        print(f"\n  Model 3 (Integrated):")
        print(f"    Adj. R²: {m3.rsquared_adj:.3f} | "
              f"AIC: {m3.aic:.1f} | N: {int(m3.nobs)}")
        print(f"\n  Coefficients (Model 3):")
        coef_df = pd.DataFrame({
            "Coefficient": m3.params.round(3),
            "Std. Error":  m3.bse.round(3),
            "t":           m3.tvalues.round(3),
            "p-value":     m3.pvalues.round(4),
            "Significant": m3.pvalues.apply(
                lambda p: "***" if p < 0.001
                else ("**" if p < 0.01
                      else ("*" if p < 0.05 else ""))),
        })
        print(coef_df.to_string())

        # Store model comparison row
        model_comparison.append({
            "Outcome":        OUTCOME_LABELS[outcome],
            "Model":          "Model 3 (Integrated)",
            "N":              int(m3.nobs),
            "R2":             round(m3.rsquared, 3),
            "Adj_R2":         round(m3.rsquared_adj, 3),
            "AIC":            round(m3.aic, 1),
            "BIC":            round(m3.bic, 1),
        })

        # Store coefficients for visualisation
        for var, row in coef_df.iterrows():
            if var == "Intercept":
                continue
            all_results.append({
                "Outcome":     OUTCOME_LABELS[outcome],
                "Predictor":   PREDICTOR_LABELS.get(
                    var, var.replace("_", " ")),
                "Coefficient": row["Coefficient"],
                "SE":          row["Std. Error"],
                "p_value":     row["p-value"],
                "Significant": row["Significant"],
            })

    # Save model comparison table
    if model_comparison:
        comp_df = pd.DataFrame(model_comparison)
        out = os.path.join(OUTPUT_FOLDER, "model_comparison_table_m3.csv")
        comp_df.to_csv(out, index=False)
        print(f"\n  Saved: model_comparison_table_m3.csv")
        results_df = pd.DataFrame(all_results)
        return results_df, comp_df
    return pd.DataFrame(), pd.DataFrame()


# ──────────────────────────────────────────────────────────────
# SECTION 7: REGRESSION COEFFICIENT PLOT (Forest Plot)
# ──────────────────────────────────────────────────────────────

def plot_regression_coefficients(results_df):
    """
    Forest plot of OLS regression coefficients (Model 2).
    Shows point estimate ± 1.96*SE (95% CI).
    Significant predictors highlighted.
    One panel per malnutrition outcome.
    """
    print("\n  Generating regression coefficient plot...")

    if results_df.empty:
        return

    outcomes = results_df["Outcome"].unique()
    n_panels = len(outcomes)
    fig, axes = plt.subplots(1, n_panels,
                              figsize=(6 * n_panels, 7),
                              sharey=False)
    if n_panels == 1:
        axes = [axes]

    for ax, outcome in zip(axes, outcomes):
        sub = results_df[results_df["Outcome"] == outcome].copy()
        sub = sub.sort_values("Coefficient")

        y_pos  = range(len(sub))
        colors = ["#d62728" if s else "#aaaaaa"
                  for s in (sub["p_value"] < 0.05)]

        ax.barh(list(y_pos), sub["Coefficient"],
                xerr=1.96 * sub["SE"],
                color=colors, alpha=0.8, height=0.5,
                error_kw=dict(elinewidth=1, capsize=3,
                               ecolor="black"))
        ax.axvline(0, color="black", lw=0.9, ls="--", alpha=0.5)

        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(sub["Predictor"], fontsize=8)
        ax.set_xlabel("OLS Coefficient", fontsize=9)
        ax.set_title(outcome, fontsize=11, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

        # Significance annotation
        for y, (_, row) in zip(y_pos, sub.iterrows()):
            if row["Significant"]:
                ax.text(row["Coefficient"] + 1.96 * row["SE"] + 0.1,
                        y, row["Significant"],
                        va="center", fontsize=9, color="#d62728")

    fig.suptitle(
        "OLS Regression Coefficients — Model 2 (Gender Indicators Only)\n"
        "Bars show 95% CI; red = significant at p < 0.05",
        fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()

    out = os.path.join(OUTPUT_FOLDER, "regression_coefficients.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: regression_coefficients.png")


# ──────────────────────────────────────────────────────────────
# SECTION 8: STANDARDISED IMPORTANCE (Lasso-based)
# ──────────────────────────────────────────────────────────────

def plot_variable_importance(df):
    """
    Standardise all predictors and outcomes, then fit LassoCV
    to get relative variable importance for each outcome.

    Standardisation allows direct comparison of coefficients
    across predictors with different scales.

    This is a data-driven alternative to manual feature selection.
    """
    print("\n  Generating variable importance plot (Lasso)...")

    predictors = [p for p in GENDER_PREDICTORS if p in df.columns]
    scaler = StandardScaler()

    fig, axes = plt.subplots(1, len(OUTCOMES),
                              figsize=(6 * len(OUTCOMES), 5))
    if len(OUTCOMES) == 1:
        axes = [axes]

    for ax, outcome in zip(axes, OUTCOMES):
        if outcome not in df.columns:
            ax.set_visible(False)
            continue

        sub = df[predictors + [outcome]].dropna()
        if len(sub) < 20:
            ax.set_visible(False)
            continue

        X = scaler.fit_transform(sub[predictors])
        y = scaler.fit_transform(sub[[outcome]]).ravel()

        lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
        lasso.fit(X, y)

        importance = pd.Series(
            np.abs(lasso.coef_), index=predictors
        ).sort_values(ascending=True)

        colors = ["#4e79a7" if v > 0 else "#e15759"
                  for v in importance]
        ax.barh(range(len(importance)), importance.values,
                color="#4e79a7", alpha=0.8)
        ax.set_yticks(range(len(importance)))
        ax.set_yticklabels(
            [PREDICTOR_LABELS.get(p, p) for p in importance.index],
            fontsize=8)
        ax.set_xlabel("|Lasso Coefficient| (standardised)", fontsize=9)
        ax.set_title(OUTCOME_LABELS.get(outcome, outcome),
                     fontsize=11, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        ax.annotate(f"alpha={lasso.alpha_:.4f}\nn={len(sub)}",
                    xy=(0.97, 0.05), xycoords="axes fraction",
                    ha="right", fontsize=8, color="#555")

    fig.suptitle(
        "Variable Importance — LassoCV (Standardised Predictors)\n"
        "Larger bar = stronger association with malnutrition outcome",
        fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()

    out = os.path.join(OUTPUT_FOLDER, "variable_importance_lasso.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: variable_importance_lasso.png")


# ──────────────────────────────────────────────────────────────
# SECTION 9: MODEL COMPARISON SUMMARY TABLE (for report)
# ──────────────────────────────────────────────────────────────

def print_model_comparison(comp_df):
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(comp_df.to_string(index=False))
    print("\n  Key metrics for report:")
    print("  - Adj. R²: proportion of variance explained")
    print("  - AIC: lower is better (penalises complexity)")
    print("  - BIC: lower is better (stronger complexity penalty)")
    print("\n  Interpretation guide:")
    print("  If Model 2 Adj.R² >> Model 1 Adj.R²:")
    print("  -> Gender indicators explain malnutrition BEYOND")
    print("     what income group alone explains")


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("GENDER INEQUALITY vs MALNUTRITION — ANALYSIS PIPELINE")
    print("Model 1 (Baseline) + Model 2 (Gender Only)")
    print("=" * 60)

    # 1. Merge
    df = load_and_merge()
    if df.empty:
        print("Merged dataset is empty. Check file paths.")
        return

    # 2. Spearman correlation table
    corr_table = spearman_correlation_table(df)

    # 3. Correlation heatmap
    plot_correlation_heatmap(df)

    # 4. Scatter matrix
    plot_scatter_matrix(df)

    # 5. Partial correlation (controlling for income)
    part_df = partial_correlation_controlling_income(df)

    # 6. OLS regression models
    results_df, comp_df = run_regression_models(df)

    # 6b. OLS regression model 3 (Integrated)
    results_df_m3, comp_df_m3 = run_model_3(df)

    # 7. Coefficient forest plot
    if not results_df.empty:
        plot_regression_coefficients(results_df)

    # 7b. Coefficient forest plot for Model 3
    if not results_df_m3.empty:
        # We can reuse the plot function but pass a different dataframe, we'll need to save to a different file
        def plot_m3(results_df_m3):
            outcomes = results_df_m3["Outcome"].unique()
            n_panels = len(outcomes)
            fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 7), sharey=False)
            if n_panels == 1:
                axes = [axes]
            for ax, outcome in zip(axes, outcomes):
                sub = results_df_m3[results_df_m3["Outcome"] == outcome].copy()
                sub = sub.sort_values("Coefficient")
                y_pos  = range(len(sub))
                colors = ["#d62728" if s else "#aaaaaa" for s in (sub["p_value"] < 0.05)]
                ax.barh(list(y_pos), sub["Coefficient"], xerr=1.96 * sub["SE"],
                        color=colors, alpha=0.8, height=0.5,
                        error_kw=dict(elinewidth=1, capsize=3, ecolor="black"))
                ax.axvline(0, color="black", lw=0.9, ls="--", alpha=0.5)
                ax.set_yticks(list(y_pos))
                ax.set_yticklabels(sub["Predictor"], fontsize=8)
                ax.set_xlabel("OLS Coefficient", fontsize=9)
                ax.set_title(outcome, fontsize=11, fontweight="bold")
                ax.grid(axis="x", alpha=0.3)
                for y, (_, row) in zip(y_pos, sub.iterrows()):
                    if row["Significant"]:
                        ax.text(row["Coefficient"] + 1.96 * row["SE"] + 0.1,
                                y, row["Significant"],
                                va="center", fontsize=9, color="#d62728")
            fig.suptitle(
                "OLS Regression Coefficients — Model 3 (Integrated)\n"
                "Bars show 95% CI; red = significant at p < 0.05",
                fontsize=12, fontweight="bold", y=1.02)
            plt.tight_layout()
            out = os.path.join(OUTPUT_FOLDER, "regression_coefficients_m3.png")
            fig.savefig(out, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: regression_coefficients_m3.png")

        plot_m3(results_df_m3)

    # 8. Lasso variable importance
    plot_variable_importance(df)

    # 9. Model comparison summary
    print_model_comparison(comp_df)
    if not comp_df_m3.empty:
        print_model_comparison(comp_df_m3)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print("\nOutput files:")
    for f in sorted(os.listdir(OUTPUT_FOLDER)):
        if f.endswith((".csv", ".png")):
            sz = os.path.getsize(os.path.join(OUTPUT_FOLDER, f))
            print(f"  {f:55s} {sz/1024:6.1f} KB")


if __name__ == "__main__":
    main()