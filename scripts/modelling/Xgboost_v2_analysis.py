"""
xgboost_v2_analysis.py
=============================================================
XGBoost predictive analysis with SHAP interpretation.

Input:
  XGBoost_v2/outputs/xgboost_v2_dataset.csv

Outputs:
  xgboost_v2_cv_results.csv
  xgboost_v2_shap_rankings.csv
  xgboost_v2_shap_bar_{outcome}.png
  xgboost_v2_shap_beeswarm_stunting_national.png
  xgboost_v2_shap_dependence_stunting_national.png
  xgboost_v2_performance_table.csv

Modelling strategy:
  - Three separate XGBoost models (one per outcome)
  - income_group one-hot encoded (dummy_na=True)
  - Predictor coverage threshold: n >= 90 countries
  - Excluded: anc4_15_19_pct (n=73), modern_contraceptive_pct (n=73),
              fgm_prevalence_pct (n=30)
  - TRUE nested CV: cross_validate(search, ...) so each outer fold
    runs its own inner RandomizedSearchCV independently
  - SHAP values computed on full dataset using best estimator
  - Missing values handled natively by XGBoost (no imputation)
  - Dependence plots for stunting: oos_upsec_f and san_bas_nat
=============================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
from xgboost import XGBRegressor
from sklearn.model_selection import (KFold, cross_validate,
                                     RandomizedSearchCV)

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_PATH = os.path.join(OUTPUT_DIR, "xgboost_v2_dataset.csv")

# ── Outcomes ───────────────────────────────────────────────────
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

# ── Predictors (all pass n >= 90 threshold) ────────────────────
# income_group handled separately via one-hot encoding
PREDICTOR_COLS = [
    # Gender
    "female_married_by_15",
    "female_married_by_18",
    "marriage_gap_18",
    # Education
    "literacy_f",
    "completion_primary_f",
    "oos_upsec_f",
    # WASH
    "wat_bas_nat",
    "san_bas_nat",
    "hyg_bas_nat",
    # Coverage
    "diarrhoea_care_pct",
    "pneumonia_care_pct",
]

PREDICTOR_LABELS = {
    "female_married_by_15":    "Child Marriage by 15 (F)",
    "female_married_by_18":    "Child Marriage by 18 (F)",
    "marriage_gap_18":         "Marriage Gender Gap",
    "literacy_f":              "Female Literacy Rate",
    "completion_primary_f":    "Primary Completion (F)",
    "oos_upsec_f":             "Upper-Sec OOS Rate (F)",
    "wat_bas_nat":             "Basic Water Access",
    "san_bas_nat":             "Basic Sanitation Access",
    "hyg_bas_nat":             "Basic Hygiene Access",
    "diarrhoea_care_pct":      "Diarrhoea Care",
    "pneumonia_care_pct":      "Pneumonia Care",
    # income dummies
    "income_low":              "Income: Low",
    "income_lower_middle":     "Income: Lower Middle",
    "income_upper_middle":     "Income: Upper Middle",
    "income_high":             "Income: High",
    "income_missing":          "Income: Unknown",
    # aggregated label for bar plot
    "income_group":            "Income Group",
}

DOMAIN_COLORS = {
    "female_married_by_15":    "#e15759",
    "female_married_by_18":    "#e15759",
    "marriage_gap_18":         "#e15759",
    "literacy_f":              "#4e79a7",
    "completion_primary_f":    "#4e79a7",
    "oos_upsec_f":             "#4e79a7",
    "wat_bas_nat":             "#59a14f",
    "san_bas_nat":             "#59a14f",
    "hyg_bas_nat":             "#59a14f",
    "diarrhoea_care_pct":      "#f28e2b",
    "pneumonia_care_pct":      "#f28e2b",
    "income_group":            "#aaaaaa",
}

# OLS Adj.R² from Model 4 (n=75) for reference
OLS_ADJ_R2 = {
    "stunting_national":   0.505,
    "wasting_national":    0.264,
    "overweight_national": 0.305,
}

# ── Hyperparameter search space ────────────────────────────────
PARAM_DIST = {
    "n_estimators":     [100, 200, 300, 500],
    "max_depth":        [3, 4, 5, 6],
    "learning_rate":    [0.01, 0.05, 0.1, 0.2],
    "subsample":        [0.6, 0.7, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
    "min_child_weight": [1, 3, 5],
    "reg_alpha":        [0, 0.1, 0.5],
    "reg_lambda":       [1, 2, 5],
}


# ──────────────────────────────────────────────────────────────
# SECTION 1: DATA PREPARATION
# ──────────────────────────────────────────────────────────────
def prepare_data(df, outcome):
    """
    For a given outcome:
      - Select all available predictors
      - One-hot encode income_group with dummy_na=True
        Column names: income_low, income_lower_middle,
                      income_upper_middle, income_high, income_missing
      - Drop rows where outcome is missing
      - Return X (DataFrame), y (Series), feature_names (list)
    """
    available_preds = [c for c in PREDICTOR_COLS if c in df.columns]
    cols_needed = [outcome] + available_preds + ["income_group"]
    cols_needed = [c for c in cols_needed if c in df.columns]

    sub = df[cols_needed].copy()
    sub = sub.dropna(subset=[outcome])

    # One-hot encode income_group
    income_map = {
        "Low Income":          "low",
        "Lower Middle Income": "lower_middle",
        "Upper Middle Income": "upper_middle",
        "High Income":         "high",
    }
    if "income_group" in sub.columns:
        sub["income_group"] = sub["income_group"].map(income_map)
        dummies = pd.get_dummies(
            sub["income_group"],
            prefix="income",
            dummy_na=True
        )
        # Standardise NaN column name
        rename_map = {c: "income_missing"
                      for c in dummies.columns
                      if "nan" in c.lower()}
        dummies = dummies.rename(columns=rename_map)
        sub = pd.concat(
            [sub.drop(columns=["income_group"]), dummies],
            axis=1
        )

    income_dummy_cols = [c for c in sub.columns
                         if c.startswith("income_")]
    feature_names = available_preds + income_dummy_cols

    X = sub[feature_names]
    y = sub[outcome]

    return X, y, feature_names


# ──────────────────────────────────────────────────────────────
# SECTION 2: TRUE NESTED CV
# ──────────────────────────────────────────────────────────────
def nested_cv_evaluate(X, y, outcome_label, random_state=42):
    """
    True nested cross-validation:
      Inner loop: RandomizedSearchCV (5-fold) per outer fold
      Outer loop: 10-fold via cross_validate(search, ...)

    Passing the search object directly to cross_validate ensures
    each outer fold runs its own independent inner search,
    so test observations never influence parameter selection.

    After nested CV, the search is refit on the full dataset
    to obtain a best_estimator for SHAP computation.
    """
    print(f"\n  Running nested CV for {outcome_label}...")

    base_xgb = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        enable_categorical=False,
        random_state=random_state,
        n_jobs=-1,
    )

    inner_cv = KFold(n_splits=5, shuffle=True,
                     random_state=random_state)
    search = RandomizedSearchCV(
        estimator=base_xgb,
        param_distributions=PARAM_DIST,
        n_iter=40,
        scoring="r2",
        cv=inner_cv,
        random_state=random_state,
        n_jobs=-1,
        refit=True,
    )

    # TRUE nested CV: search object passed to cross_validate
    outer_cv = KFold(n_splits=10, shuffle=True,
                     random_state=random_state)
    cv_res = cross_validate(
        search, X, y,
        cv=outer_cv,
        scoring={
            "r2":   "r2",
            "rmse": "neg_root_mean_squared_error",
        },
        return_estimator=False,
        n_jobs=-1,
    )

    cv_r2    = cv_res["test_r2"].mean()
    cv_r2_sd = cv_res["test_r2"].std()
    cv_rmse  = (-cv_res["test_rmse"]).mean()

    print(f"    Nested 10-fold CV R²:   {cv_r2:.3f} ± {cv_r2_sd:.3f}")
    print(f"    Nested 10-fold CV RMSE: {cv_rmse:.3f}")

    # Refit on full data for SHAP
    print(f"    Refitting on full data for SHAP...")
    search.fit(X, y)
    best_params = search.best_params_
    print(f"    Best params: {best_params}")

    return search.best_estimator_, cv_r2, cv_r2_sd, cv_rmse, best_params


# ──────────────────────────────────────────────────────────────
# SECTION 3: SHAP BAR PLOT
# ──────────────────────────────────────────────────────────────
def plot_shap_bar(shap_values, feature_names, outcome, cv_r2):
    """
    Horizontal bar chart of mean |SHAP| values.
    Income dummies are aggregated into a single 'Income Group' bar.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)

    # Aggregate income dummies
    income_cols = [f for f in feature_names if f.startswith("income_")]
    income_shap = sum(
        mean_abs[feature_names.index(c)]
        for c in income_cols
        if c in feature_names
    )

    rows = []
    for feat, val in zip(feature_names, mean_abs):
        if feat.startswith("income_"):
            continue
        rows.append({
            "feature":   feat,
            "mean_shap": val,
            "label":     PREDICTOR_LABELS.get(feat, feat),
            "color":     DOMAIN_COLORS.get(feat, "#888888"),
        })
    rows.append({
        "feature":   "income_group",
        "mean_shap": income_shap,
        "label":     "Income Group",
        "color":     "#aaaaaa",
    })

    shap_df = (pd.DataFrame(rows)
               .sort_values("mean_shap", ascending=True)
               .reset_index(drop=True))

    fig, ax = plt.subplots(
        figsize=(7.0, 0.45 * len(shap_df) + 1.5))

    bars = ax.barh(
        shap_df["label"],
        shap_df["mean_shap"],
        color=shap_df["color"],
        alpha=0.88,
        edgecolor="white",
        linewidth=0.5,
    )
    for bar, val in zip(bars, shap_df["mean_shap"]):
        ax.text(
            val + shap_df["mean_shap"].max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}",
            va="center", ha="left", fontsize=12
        )

    ax.set_xlabel("Mean |SHAP value|", fontsize=12)
    ax.set_title(
        f"{OUTCOME_LABELS[outcome]} — Feature Importance (SHAP)\n"
        f"Nested 10-fold CV R² = {cv_r2:.3f}",
        fontsize=14, fontweight="bold", pad=8
    )
    ax.grid(axis="x", alpha=0.25, linestyle=":")
    ax.spines[["top", "right"]].set_visible(False)

    domain_info = [
        ("Gender",    "#e15759"),
        ("Education", "#4e79a7"),
        ("WASH",      "#59a14f"),
        ("Coverage",  "#f28e2b"),
        ("Control",   "#aaaaaa"),
    ]
    handles = [mpatches.Patch(color=c, alpha=0.88, label=d)
               for d, c in domain_info]
    ax.legend(handles=handles, loc="lower right",
              fontsize=12, title="Domain", title_fontsize=8,
              framealpha=0.9)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR,
                       f"xgboost_v2_shap_bar_{outcome}.pdf")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: xgboost_v2_shap_bar_{outcome}.png")

    return shap_df


# ──────────────────────────────────────────────────────────────
# SECTION 4: SHAP BEESWARM (stunting only)
# ──────────────────────────────────────────────────────────────
def plot_shap_beeswarm(shap_values, X, feature_names, outcome, cv_r2):
    """
    SHAP beeswarm showing direction and distribution per feature.
    """
    feature_labels = [PREDICTOR_LABELS.get(f, f) for f in feature_names]
    X_labeled = X.copy()
    X_labeled.columns = feature_labels

    explanation = shap.Explanation(
        values=shap_values,
        data=X_labeled.values,
        feature_names=feature_labels,
    )

    plt.figure(figsize=(7.5, 0.45 * len(feature_names) + 1.5))
    shap.plots.beeswarm(explanation,
                        max_display=len(feature_names),
                        show=False)
    plt.title(
        f"{OUTCOME_LABELS[outcome]} — SHAP Beeswarm Plot\n"
        f"Nested 10-fold CV R² = {cv_r2:.3f}",
        fontsize=11, fontweight="bold", pad=8
    )
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR,
                       f"xgboost_v2_shap_beeswarm_{outcome}.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Saved: xgboost_v2_shap_beeswarm_{outcome}.png")


# ──────────────────────────────────────────────────────────────
# SECTION 5: SHAP DEPENDENCE PLOTS (stunting only)
# ──────────────────────────────────────────────────────────────
def plot_shap_dependence(shap_values, X, feature_names, outcome, cv_r2):
    """
    Two-panel SHAP dependence plot for stunting (stacked vertically):
      Top:    oos_upsec_f  — top education predictor
      Bottom: san_bas_nat  — OLS-significant WASH predictor

    Points coloured by income_high dummy to show structural
    interaction between infrastructure access and income level.
    Quadratic trend line added for visual guidance.
    """
    targets = ["oos_upsec_f", "san_bas_nat"]
    target_labels = {
        "oos_upsec_f": "Upper-Sec OOS Rate, Female (%)",
        "san_bas_nat": "Basic Sanitation Access (%)",
    }
    point_colors = {
        "oos_upsec_f": "#4e79a7",
        "san_bas_nat": "#59a14f",
    }

    available = [t for t in targets if t in feature_names]
    if not available:
        print("    Dependence plot: target features not found.")
        return

    n_panels = len(available)
    # Change to vertical layout: n_panels rows, 1 column
    fig, axes = plt.subplots(
        n_panels, 1,                       # rows, columns
        figsize=(5.5, 4.2 * n_panels),     # wider, and taller per panel
    )
    if n_panels == 1:
        axes = [axes]

    # income_high as interaction variable
    interact_col = "income_high"
    if interact_col in feature_names:
        interact_idx  = feature_names.index(interact_col)
        interact_vals = X.iloc[:, interact_idx].values
        use_interact  = True
    else:
        use_interact = False

    for ax, feat in zip(axes, available):
        feat_idx  = feature_names.index(feat)
        x_vals    = X.iloc[:, feat_idx].values
        shap_vals = shap_values[:, feat_idx]

        mask      = ~np.isnan(x_vals.astype(float))
        x_plot    = x_vals[mask].astype(float)
        shap_plot = shap_vals[mask]

        if use_interact:
            c_plot = interact_vals[mask].astype(float)
            sc = ax.scatter(
                x_plot, shap_plot,
                c=c_plot, cmap="coolwarm",
                alpha=0.65, s=35, edgecolors="none",
                vmin=0, vmax=1,
            )
            # Colorbar - for vertical layout, keep at side of each subplot
            cbar = fig.colorbar(sc, ax=ax, pad=0.02,
                                fraction=0.046)
            cbar.set_label("High Income (1=yes)", fontsize=10)
            cbar.set_ticks([0, 1])
            cbar.ax.tick_params(labelsize=10)
        else:
            ax.scatter(
                x_plot, shap_plot,
                color=point_colors.get(feat, "#888"),
                alpha=0.65, s=35, edgecolors="none"
            )

        # Quadratic trend line
        sort_idx = np.argsort(x_plot)
        z        = np.polyfit(x_plot[sort_idx],
                              shap_plot[sort_idx], 2)
        p        = np.poly1d(z)
        x_line   = np.linspace(x_plot.min(), x_plot.max(), 200)
        ax.plot(x_line, p(x_line),
                color="#333333", lw=1.5, ls="--", alpha=0.7,
                label="Quadratic trend")

        ax.axhline(0, color="#666", lw=0.8, ls=":", alpha=0.5)
        ax.set_xlabel(target_labels[feat], fontsize=12)
        ax.set_ylabel(
            "SHAP value\n(impact on stunting prediction)",
            fontsize=12)
        ax.set_title(
            PREDICTOR_LABELS.get(feat, feat),
            fontsize=14, fontweight="bold")
        ax.grid(alpha=0.2, linestyle=":")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=14, framealpha=0.8)

    fig.suptitle(
        f"{OUTCOME_LABELS[outcome]} — SHAP Dependence Plots\n"
        f"Nested 10-fold CV R² = {cv_r2:.3f}  \n  "
        f"Colour = Income: High (1) vs Others (0)",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR,
                       f"xgboost_v2_shap_dependence_{outcome}.pdf")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: xgboost_v2_shap_dependence_{outcome}.png")

# ──────────────────────────────────────────────────────────────
# SECTION 6: PERFORMANCE TABLE
# ──────────────────────────────────────────────────────────────
def save_performance_table(cv_results_list):
    """
    Save CSV comparing OLS Adj.R² and XGBoost nested CV R².

    IMPORTANT: these metrics are not directly comparable.
    OLS uses adjusted R² on a complete-case sample (n=75) with
    a parsimonious 3-predictor specification.
    XGBoost uses nested cross-validated R² on the full sample
    (n=163) with an 11-predictor specification.
    The table is provided for descriptive reference only.
    """
    rows = []
    for outcome, res in cv_results_list:
        rows.append({
            "Outcome":      OUTCOME_LABELS[outcome],
            "OLS_n":        75,
            "OLS_Adj_R2":   OLS_ADJ_R2[outcome],
            "XGB_n":        res["n"],
            "XGB_CV_R2":    round(res["cv_r2"], 3),
            "XGB_CV_R2_SD": round(res["cv_r2_sd"], 3),
            "XGB_CV_RMSE":  round(res["cv_rmse"], 3),
        })
    perf_df = pd.DataFrame(rows)
    out = os.path.join(OUTPUT_DIR,
                       "xgboost_v2_performance_table.csv")
    perf_df.to_csv(out, index=False)
    print(f"\n  Saved: xgboost_v2_performance_table.csv")
    return perf_df


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("XGBOOST V2 — ANALYSIS")
    print("  Nested CV | One-hot income | Dependence plots")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)
    print(f"\n  Dataset loaded: {df.shape[0]} countries, "
          f"{df.shape[1]} columns")

    cv_results_list = []
    all_metrics     = []
    shap_rankings   = []

    for outcome in OUTCOMES:
        print(f"\n{'─'*60}")
        print(f"  OUTCOME: {OUTCOME_LABELS[outcome]}")
        print(f"{'─'*60}")

        X, y, feature_names = prepare_data(df, outcome)
        n_domain = len([f for f in feature_names
                        if not f.startswith("income_")])
        n_income = len([f for f in feature_names
                        if f.startswith("income_")])
        print(f"  Features: {n_domain} domain predictors "
              f"+ {n_income} income dummies = {len(feature_names)} total")
        print(f"  Working sample: n={len(y)}")

        # Nested CV
        best_model, cv_r2, cv_r2_sd, cv_rmse, best_params = \
            nested_cv_evaluate(X, y, OUTCOME_LABELS[outcome])

        cv_results_list.append((outcome, {
            "n": len(y),
            "cv_r2": cv_r2,
            "cv_r2_sd": cv_r2_sd,
            "cv_rmse": cv_rmse,
        }))

        all_metrics.append({
            "Outcome":    OUTCOME_LABELS[outcome],
            "N":          len(y),
            "CV_R2":      round(cv_r2, 3),
            "CV_R2_SD":   round(cv_r2_sd, 3),
            "CV_RMSE":    round(cv_rmse, 3),
            "OLS_Adj_R2": OLS_ADJ_R2[outcome],
            **{f"param_{k}": v for k, v in best_params.items()},
        })

        # SHAP
        print(f"\n  Computing SHAP values...")
        explainer   = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X)

        # Bar plot (all outcomes)
        shap_df = plot_shap_bar(
            shap_values, feature_names, outcome, cv_r2)

        # Beeswarm + dependence (stunting only)
        if outcome == "stunting_national":
            plot_shap_beeswarm(
                shap_values, X, feature_names, outcome, cv_r2)
            plot_shap_dependence(
                shap_values, X, feature_names, outcome, cv_r2)

        # Rankings
        for _, row in shap_df.iterrows():
            shap_rankings.append({
                "Outcome":       OUTCOME_LABELS[outcome],
                "Feature":       row["feature"],
                "Label":         row["label"],
                "Mean_Abs_SHAP": round(float(row["mean_shap"]), 4),
            })

    # Performance table
    perf_df = save_performance_table(cv_results_list)

    # Save CSVs
    pd.DataFrame(all_metrics).to_csv(
        os.path.join(OUTPUT_DIR, "xgboost_v2_cv_results.csv"),
        index=False)
    (pd.DataFrame(shap_rankings)
       .sort_values(["Outcome", "Mean_Abs_SHAP"],
                    ascending=[True, False])
       .to_csv(os.path.join(OUTPUT_DIR,
                            "xgboost_v2_shap_rankings.csv"),
               index=False))

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"\n{'Outcome':<20} {'N':>5} {'CV R²':>8} "
          f"{'±SD':>7} {'CV RMSE':>9} {'OLS Adj.R²':>12}")
    print("─" * 65)
    for m in all_metrics:
        print(f"  {m['Outcome']:<18} {m['N']:>5} "
              f"{m['CV_R2']:>8.3f} {m['CV_R2_SD']:>7.3f} "
              f"{m['CV_RMSE']:>9.3f} {m['OLS_Adj_R2']:>12.3f}")

    print("\n" + "=" * 60)
    print("TOP SHAP PREDICTORS PER OUTCOME")
    print("=" * 60)
    shap_rank_df = pd.DataFrame(shap_rankings)
    for outcome in OUTCOMES:
        top = (shap_rank_df[
                   shap_rank_df["Outcome"] == OUTCOME_LABELS[outcome]]
               .sort_values("Mean_Abs_SHAP", ascending=False)
               .head(5))
        print(f"\n  {OUTCOME_LABELS[outcome]}:")
        for _, row in top.iterrows():
            print(f"    {row['Label']:<35} {row['Mean_Abs_SHAP']:.4f}")

    print("\n  All outputs saved to XGBoost_v2/outputs/")


if __name__ == "__main__":
    main()