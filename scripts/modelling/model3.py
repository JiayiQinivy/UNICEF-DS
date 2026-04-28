import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")

SCRIPT_DIR       = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR         = os.path.dirname(SCRIPT_DIR)

MALNUTRITION_CSV = os.path.join(ROOT_DIR, "outputs",
                                 "malnutrition_modelling_sample.csv")
GENDER_CSV       = os.path.join(ROOT_DIR, "outputs",
                                 "gender_inequality_analysis.csv")
EDUCATION_CSV    = os.path.join(ROOT_DIR, "education_clean.csv")
WASH_CSV         = os.path.join(ROOT_DIR, "WASH", "outputs",
                                 "wash_selected.csv")
OUTPUT_FOLDER    = SCRIPT_DIR
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


OUTCOMES = ["stunting_national", "wasting_national", "overweight_national"]
OUTCOME_LABELS = {
    "stunting_national":   "Stunting (%)",
    "wasting_national":    "Wasting (%)",
    "overweight_national": "Overweight (%)",
}

GENDER_PRED = "female_married_by_18"
EDU_PRED    = "literacy_f"
WASH_PRED   = "san_bas_nat"

PREDICTOR_LABELS = {
    "female_married_by_18": "Female Marriage by 18 (%)",
    "literacy_f":           "Female Literacy Rate (%)",
    "san_bas_nat":          "Basic Sanitation Access (%)",
}

GROUP_COLORS = {
    "Gender":    "#e15759",
    "Education": "#4e79a7",
    "WASH":      "#59a14f",
}



def load_and_merge():
    print("=" * 60)
    print("SECTION 1: DATA MERGE")
    print("=" * 60)

    files = {
        "Malnutrition":  MALNUTRITION_CSV,
        "Gender":        GENDER_CSV,
        "Education":     EDUCATION_CSV,
        "WASH":          WASH_CSV,
    }
    for name, path in files.items():
        status = "found" if os.path.exists(path) else "NOT FOUND"
        print(f"  {name:15s}: {status}")

    if not all(os.path.exists(p) for p in files.values()):
        print("\n  One or more files missing. Check paths above.")
        return pd.DataFrame()

    mal  = pd.read_csv(MALNUTRITION_CSV)
    gen  = pd.read_csv(GENDER_CSV)
    edu  = pd.read_csv(EDUCATION_CSV)
    wash = pd.read_csv(WASH_CSV)

    df = mal.copy()
    for src, name in [(gen, "gender"), (edu, "education"),
                       (wash, "wash")]:
        drop_cols = [c for c in src.columns
                     if c in ("country", "year", "sdg_region",
                               "unicef_reporting_region",
                               "all_basic_missing")
                     and c != "ISO"]
        df = df.merge(
            src.drop(columns=drop_cols, errors="ignore"),
            on="ISO", how="left")
        print(f"  After merging {name:12s}: {len(df)} countries")

    # Resolve duplicate country columns
    if "country_x" in df.columns:
        df["country"] = df["country_x"].fillna(
            df.get("country_y", pd.Series(dtype=str)))
        df = df.drop(
            columns=[c for c in df.columns
                     if c in ("country_x", "country_y")],
            errors="ignore")

    # Model 4 complete-case subsample
    m4_cols = OUTCOMES + ["income_group",
                           GENDER_PRED, EDU_PRED, WASH_PRED]
    avail   = [c for c in m4_cols if c in df.columns]
    n_m4    = df[avail].dropna().shape[0]
    print(f"\n  Model 4 complete-case sample: n = {n_m4} countries")
    print(f"  (all four models use this same subsample)")

    print(f"\n  Predictor coverage in merged dataset:")
    for col in [GENDER_PRED, EDU_PRED, WASH_PRED]:
        if col in df.columns:
            print(f"    {col:30s}: {df[col].notna().sum():3d} / {len(df)}")

    # Income group distribution
    sample_idx = df[avail].dropna().index
    print(f"\n  Income group distribution (Model 4 sample, n={n_m4}):")
    ig = df.loc[sample_idx, "income_group"].value_counts()
    for grp, cnt in ig.items():
        print(f"    {grp:25s}: {cnt}")

    out = os.path.join(OUTPUT_FOLDER, "model3_analytical_dataset.csv")
    df.to_csv(out, index=False)
    print(f"\n  Saved: model3_analytical_dataset.csv "
          f"({len(df)} countries, {len(df.columns)} columns)")

    return df



def run_models(df):
    print("\n" + "=" * 60)
    print("SECTION 2: MODEL COMPARISON (Models 1-4)")
    print("=" * 60)

    m4_cols = OUTCOMES + ["income_group",
                           GENDER_PRED, EDU_PRED, WASH_PRED]
    avail   = [c for c in m4_cols if c in df.columns]
    sub     = df[avail].dropna().copy()
    n       = len(sub)
    print(f"\n  Sample (all models): n = {n} countries")

    all_comparison = []
    all_coefs      = []

    for outcome in OUTCOMES:
        if outcome not in sub.columns:
            continue

        print(f"\n  {'─'*55}")
        print(f"  Outcome: {OUTCOME_LABELS[outcome]}")
        print(f"  {'─'*55}")

        m1 = smf.ols(
            f"{outcome} ~ C(income_group)",
            data=sub).fit()
        m2 = smf.ols(
            f"{outcome} ~ {GENDER_PRED}",
            data=sub).fit()
        m3 = smf.ols(
            f"{outcome} ~ {GENDER_PRED} + {EDU_PRED}",
            data=sub).fit()
        m4 = smf.ols(
            f"{outcome} ~ {GENDER_PRED} + {EDU_PRED}"
            f" + {WASH_PRED} + C(income_group)",
            data=sub).fit()

        for label, model in [
            ("Model 1 (Baseline — income only)", m1),
            ("Model 2 (Gender only)",             m2),
            ("Model 3 (Gender + Education)",      m3),
            ("Model 4 (Full multivariate)",       m4),
        ]:
            print(f"\n  {label}:")
            print(f"    Adj.R²={model.rsquared_adj:.3f}  "
                  f"AIC={model.aic:.1f}  N={int(model.nobs)}")

        print(f"\n  Model 4 Coefficients:")
        coef_df = pd.DataFrame({
            "Coefficient": m4.params.round(3),
            "Std.Error":   m4.bse.round(3),
            "t":           m4.tvalues.round(2),
            "p-value":     m4.pvalues.round(4),
            "Sig":         m4.pvalues.apply(
                lambda p: "***" if p < 0.001
                else ("**" if p < 0.01
                      else ("*" if p < 0.05 else ""))),
        })
        print(coef_df.to_string())

        print(f"\n  Delta (Model 4 vs Model 2): "
              f"Adj.R²={m4.rsquared_adj-m2.rsquared_adj:+.3f}  "
              f"AIC={m4.aic-m2.aic:+.1f}")
        print(f"  Delta (Model 4 vs Model 3): "
              f"Adj.R²={m4.rsquared_adj-m3.rsquared_adj:+.3f}  "
              f"AIC={m4.aic-m3.aic:+.1f}")

        for model_name, model_obj in [
            ("Model 1 (Baseline)",           m1),
            ("Model 2 (Gender)",             m2),
            ("Model 3 (Gender+Education)",   m3),
            ("Model 4 (Full multivariate)",  m4),
        ]:
            all_comparison.append({
                "Outcome": OUTCOME_LABELS[outcome],
                "Model":   model_name,
                "N":       int(model_obj.nobs),
                "R2":      round(model_obj.rsquared, 3),
                "Adj_R2":  round(model_obj.rsquared_adj, 3),
                "AIC":     round(model_obj.aic, 1),
                "BIC":     round(model_obj.bic, 1),
            })

        for var, row in coef_df.iterrows():
            if "Intercept" in str(var) or "income_group" in str(var):
                continue
            group = ("Gender"    if var == GENDER_PRED else
                     "Education" if var == EDU_PRED    else "WASH")
            all_coefs.append({
                "Outcome":     OUTCOME_LABELS[outcome],
                "Predictor":   PREDICTOR_LABELS.get(
                    var, str(var).replace("_", " ")),
                "Variable":    var,
                "Coefficient": row["Coefficient"],
                "SE":          row["Std.Error"],
                "p_value":     row["p-value"],
                "Sig":         row["Sig"],
                "Group":       group,
            })

    comp_df  = pd.DataFrame(all_comparison)
    coefs_df = pd.DataFrame(all_coefs)

    comp_df.to_csv(
        os.path.join(OUTPUT_FOLDER, "model3_model_comparison.csv"),
        index=False)
    coefs_df.to_csv(
        os.path.join(OUTPUT_FOLDER, "model3_coefficients.csv"),
        index=False)
    print(f"\n  Saved: model3_model_comparison.csv")
    print(f"  Saved: model3_coefficients.csv")

    print(f"\n{'='*60}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(comp_df.to_string(index=False))

    return comp_df, coefs_df, sub



def plot_model_comparison(comp_df):
    print("\n  Generating Figure 1: Model comparison plot...")

    outcomes = [OUTCOME_LABELS[o] for o in OUTCOMES]
    models   = ["Model 1 (Baseline)",
                "Model 2 (Gender)",
                "Model 3 (Gender+Education)",
                "Model 4 (Full multivariate)"]
    colors   = ["#aec6cf", "#e15759", "#4e79a7", "#59a14f"]

    x     = np.arange(len(outcomes))
    width = 0.18
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (model, color) in enumerate(zip(models, colors)):
        vals = []
        for o in outcomes:
            match = comp_df.loc[
                (comp_df["Outcome"] == o) &
                (comp_df["Model"]   == model), "Adj_R2"]
            vals.append(float(match.values[0]) if len(match) else 0)

        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, vals, width, label=model,
                      color=color, alpha=0.88,
                      edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            if abs(val) > 0.02:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{val:.2f}",
                        ha="center", va="bottom",
                        fontsize=7.5, color="#333")

    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(outcomes, fontsize=11)
    ax.set_ylabel("Adjusted R²", fontsize=11)
    ax.set_ylim(bottom=min(-0.05, comp_df["Adj_R2"].min() - 0.05))
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(
        "Model Comparison: Adjusted R² by Malnutrition Outcome\n"
        "(All models on same complete-case sample, n same for all)",
        fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()

    out = os.path.join(OUTPUT_FOLDER, "model3_model_comparison_plot.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: model3_model_comparison_plot.png")



def plot_grouped_coefficient_forest(coefs_df):
    """Grouped coefficient plot for Model 4 OLS coefficients."""
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    print("\n  Generating grouped coefficient plot...")

    if coefs_df is None or coefs_df.empty:
        print("  No coefficients to plot.")
        return

    df = coefs_df.copy()

    required_cols = {
        "Outcome", "Predictor", "Coefficient", "SE", "p_value", "Sig"
    }
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        print(f"  Missing required columns: {sorted(missing_cols)}")
        return

    # Predictor order: top to bottom
    pred_order = [
        PREDICTOR_LABELS[GENDER_PRED],
        PREDICTOR_LABELS[EDU_PRED],
        PREDICTOR_LABELS[WASH_PRED],
    ]

    # Outcome order
    outcome_order = [OUTCOME_LABELS[o] for o in OUTCOMES if OUTCOME_LABELS[o] in df["Outcome"].unique()]
    if len(outcome_order) == 0:
        print("  No recognised outcomes found.")
        return

    df = df[df["Predictor"].isin(pred_order) & df["Outcome"].isin(outcome_order)].copy()
    if df.empty:
        print("  No matching retained predictors found.")
        return

    if {"CI_lower", "CI_upper"}.issubset(df.columns):
        df["CI_lower_plot"] = df["CI_lower"]
        df["CI_upper_plot"] = df["CI_upper"]
    else:
        df["CI_lower_plot"] = df["Coefficient"] - 1.96 * df["SE"]
        df["CI_upper_plot"] = df["Coefficient"] + 1.96 * df["SE"]

    x_min = np.nanmin(df["CI_lower_plot"].values)
    x_max = np.nanmax(df["CI_upper_plot"].values)
    pad = max((x_max - x_min) * 0.10, 0.025)
    x_min -= pad
    x_max += pad

    outcome_colors = {
        OUTCOME_LABELS["stunting_national"]: "#4e79a7",  # steel blue
        OUTCOME_LABELS["wasting_national"]: "#f28e2b",  # orange
        OUTCOME_LABELS["overweight_national"]: "#9c755f",  # brown
    }

    offsets = {
        outcome_order[0]: -0.18 if len(outcome_order) >= 1 else 0.0,
        outcome_order[1]:  0.00 if len(outcome_order) >= 2 else 0.0,
        outcome_order[2]:  0.18 if len(outcome_order) >= 3 else 0.0,
    }

    fig, ax = plt.subplots(figsize=(7, 5.2))

    y_base = np.arange(len(pred_order))

    for y in y_base:
        ax.axhline(y, color="#eeeeee", lw=0.8, zorder=0)

    # Plot all outcome-predictor combinations
    for pred_idx, pred in enumerate(pred_order):
        pred_df = df[df["Predictor"] == pred].copy()

        for outcome in outcome_order:
            row = pred_df[pred_df["Outcome"] == outcome]
            if row.empty:
                continue

            row = row.iloc[0]
            y = pred_idx + offsets[outcome]

            coef = row["Coefficient"]
            ci_lo = row["CI_lower_plot"]
            ci_hi = row["CI_upper_plot"]
            p_val = row["p_value"]
            sig = row["Sig"] if pd.notna(row["Sig"]) else ""

            color = outcome_colors.get(outcome, "#444444")
            is_sig = pd.notna(p_val) and p_val < 0.05

            # CI whisker
            ax.plot(
                [ci_lo, ci_hi], [y, y],
                color=color,
                alpha=0.85 if is_sig else 0.45,
                linewidth=1.8,
                solid_capstyle="round",
                zorder=2
            )

            # End caps
            cap_half = 0.055
            ax.plot([ci_lo, ci_lo], [y - cap_half, y + cap_half],
                    color=color, alpha=0.85 if is_sig else 0.45,
                    linewidth=1.1, zorder=2)
            ax.plot([ci_hi, ci_hi], [y - cap_half, y + cap_half],
                    color=color, alpha=0.85 if is_sig else 0.45,
                    linewidth=1.1, zorder=2)

            # Dot
            if is_sig:
                ax.scatter(
                    coef, y,
                    s=72,
                    color=color,
                    edgecolors="white",
                    linewidth=0.6,
                    zorder=3
                )
            else:
                ax.scatter(
                    coef, y,
                    s=68,
                    facecolors="white",
                    edgecolors=color,
                    linewidth=1.7,
                    zorder=3
                )

            # Significance stars
            if isinstance(sig, str) and sig.strip() and is_sig:
                offset = max(abs(ci_hi - ci_lo) * 0.08, 0.012 * (x_max - x_min))
                #star_x = min(ci_hi + offset, x_max - 0.02 * (x_max - x_min))
                star_x = x_max - (x_max - x_min) * 0.03
                ax.text(
                    star_x, y, sig.strip(),
                    va="center", ha="left",
                    fontsize=10,
                    color=color,
                    fontweight="bold",
                    zorder=4
                )

    ax.axvline(0, color="#444444", lw=1.0, ls="--", alpha=0.65, zorder=1)

    # Axes
    ax.set_xlim(x_min, x_max)
    ax.set_yticks(y_base)
    ax.set_yticklabels(pred_order, fontsize=11)
    ax.invert_yaxis()

    ax.set_xlabel("OLS Coefficient", fontsize=12)
    ax.set_title(
        "Model 4 — Grouped OLS Coefficients with 95% Confidence Intervals",
        fontsize=12,
        fontweight="bold",
        pad=10
    )

    ax.grid(axis="x", alpha=0.22, linestyle=":", linewidth=0.8)
    ax.grid(axis="y", alpha=0.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend for outcomes
    outcome_handles = [
        Line2D([0], [0],
               marker='o', color=outcome_colors[o], label=o,
               markerfacecolor=outcome_colors[o], markersize=7, linewidth=1.8)
        for o in outcome_order
    ]

    # Legend for significance encoding
    sig_handles = [
        Line2D([0], [0],
               marker='o', color="#555555", label="p < 0.05",
               markerfacecolor="#555555", markersize=7, linewidth=0),
        Line2D([0], [0],
               marker='o', color="#555555", label="p ≥ 0.05",
               markerfacecolor="white", markeredgecolor="#555555",
               markersize=7, linewidth=0),
    ]

    legend1 = ax.legend(
        handles=outcome_handles,
        title="Outcome",
        loc="upper right",
        bbox_to_anchor=(1.0, 0.90),
        fontsize=10,
        title_fontsize=10,
        framealpha=0.92
    )
    ax.add_artist(legend1)

    ax.legend(
        handles=sig_handles,
        title="Point style",
        loc="lower right",
        fontsize=9,
        title_fontsize=9,
        framealpha=0.92
    )

    plt.tight_layout()

    out = os.path.join(OUTPUT_FOLDER, "model3_grouped_coefficient_plot.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: model3_grouped_coefficient_plot.png")



def plot_lasso_importance(sub):
    print("\n  Generating Figure 3: Lasso variable importance...")

    predictors = [GENDER_PRED, EDU_PRED, WASH_PRED]
    groups     = {GENDER_PRED: "Gender",
                  EDU_PRED:    "Education",
                  WASH_PRED:   "WASH"}
    scaler = StandardScaler()

    fig, axes = plt.subplots(1, len(OUTCOMES),
                              figsize=(5 * len(OUTCOMES), 5))
    if len(OUTCOMES) == 1:
        axes = [axes]

    for ax, outcome in zip(axes, OUTCOMES):
        if outcome not in sub.columns:
            ax.set_visible(False)
            continue

        data = sub[predictors + [outcome]].dropna()
        if len(data) < 10:
            ax.set_visible(False)
            continue

        X = scaler.fit_transform(data[predictors])
        y = scaler.fit_transform(data[[outcome]]).ravel()

        lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
        lasso.fit(X, y)

        importance = pd.Series(
            np.abs(lasso.coef_), index=predictors
        ).sort_values(ascending=True)

        bar_colors = [GROUP_COLORS.get(groups.get(p, ""), "#888")
                      for p in importance.index]

        ax.barh(range(len(importance)), importance.values,
                color=bar_colors, alpha=0.85)
        ax.set_yticks(range(len(importance)))
        ax.set_yticklabels(
            [PREDICTOR_LABELS.get(p, p) for p in importance.index],
            fontsize=9)
        ax.set_xlabel("|Lasso Coef.| (standardised)", fontsize=9)
        ax.set_title(OUTCOME_LABELS.get(outcome, outcome),
                     fontsize=11, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        ax.annotate(f"alpha={lasso.alpha_:.4f}, n={len(data)}",
                    xy=(0.97, 0.04), xycoords="axes fraction",
                    ha="right", fontsize=8, color="#555")

    legend_handles = [
        mpatches.Patch(color=c, alpha=0.85, label=g)
        for g, c in GROUP_COLORS.items()
    ]
    fig.legend(handles=legend_handles, loc="upper right",
               fontsize=9, title="Predictor Group",
               title_fontsize=9, framealpha=0.9)
    fig.suptitle(
        "Model 4 — Variable Importance via LassoCV\n"
        "Standardised predictors  |  Gender | Education | WASH",
        fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()

    out = os.path.join(OUTPUT_FOLDER, "model3_lasso_importance.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: model3_lasso_importance.png")



def main():
    print("=" * 60)
    print("MODEL 3 — FULL MULTIVARIATE ANALYSIS")
    print(f"  Gender predictor:    {GENDER_PRED}")
    print(f"  Education predictor: {EDU_PRED}")
    print(f"  WASH predictor:      {WASH_PRED}")
    print("=" * 60)

    df = load_and_merge()
    if df.empty:
        print("Merge failed. Check file paths.")
        return

    comp_df, coefs_df, sub = run_models(df)
    plot_model_comparison(comp_df)
    #plot_coefficient_forest(coefs_df)
    plot_grouped_coefficient_forest(coefs_df)
    plot_lasso_importance(sub)

    print("\n" + "=" * 60)
    print("MODEL 3 COMPLETE")
    print("=" * 60)
    print("\nOutput files:")
    for f in ["model3_analytical_dataset.csv",
              "model3_model_comparison.csv",
              "model3_coefficients.csv",
              "model3_model_comparison_plot.png",
              "model3_coefficient_plot.png",
              "model3_lasso_importance.png"]:
        full = os.path.join(OUTPUT_FOLDER, f)
        if os.path.exists(full):
            sz = os.path.getsize(full)
            print(f"  {f:45s} {sz/1024:6.1f} KB")


if __name__ == "__main__":
    main()

