import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import xgboost as xgb
import shap
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, train_test_split

warnings.filterwarnings("ignore")

# ── Chart style ───────────────────────────────────────────
UNICEF_BLUE   = "#1CABE2"
UNICEF_DARK   = "#374EA2"
UNICEF_GREEN  = "#00833D"
UNICEF_YELLOW = "#FFC20E"
UNICEF_RED    = "#E2231A"
UNICEF_GREY   = "#D8D1C9"

OUTCOME_COLORS = {
    "stunting_national":   UNICEF_BLUE,
    "wasting_national":    UNICEF_RED,
    "overweight_national": UNICEF_YELLOW,
}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#FAFAFA",
    "axes.edgecolor":   "#CCCCCC",
    "axes.grid":        True,
    "grid.color":       "#E0E0E0",
    "grid.linestyle":   "--",
    "grid.alpha":       0.6,
    "font.family":      "sans-serif",
    "font.size":        11,
    "axes.titlesize":   14,
    "axes.titleweight": "bold",
    "axes.labelsize":   12,
    "figure.dpi":       150,
})
# ──────────────────────────────────────────────────────────

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR   = os.path.dirname(_SCRIPT_DIR)

OUTPUT_FOLDER = os.path.join(_ROOT_DIR, "outputs", "xgboost")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

MALNUTRITION_CSV = os.path.join(_ROOT_DIR, "outputs", "xgboost_final_dataset.csv")
WASH_CSV = os.path.join(_ROOT_DIR, "outputs", "wash_clean_data.csv")
EDU_CSV = os.path.join(_ROOT_DIR, "outputs", "education_clean.csv")

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

GENDER_PREDICTORS = [
    "female_married_by_18",
    "female_married_by_15",
    "marriage_gap_18",
    "anc4_15_19_pct",
    "modern_contraceptive_pct",
]
WASH_PREDICTORS = ["wat_bas_nat", "san_bas_nat", "hyg_bas_nat"]
EDU_PREDICTORS = ["completion_primary_f", "literacy_f"]
HEALTH_PREDICTORS = [
    "diarrhoea_care_pct",
    "pneumonia_care_pct",
    "exclusive_bf_pct",
    "low_birthweight_pct"
]

PREDICTOR_LABELS = {
    "female_married_by_18":    "Female Child Marriage by 18 (%)",
    "female_married_by_15":    "Female Child Marriage by 15 (%)",
    "marriage_gap_18":         "Marriage Gender Gap (F-M)",
    "anc4_15_19_pct":          "Antenatal Care (4+ visits)",
    "modern_contraceptive_pct":"Modern Contraceptives",
    "wat_bas_nat":             "Basic Water",
    "san_bas_nat":             "Basic Sanitation",
    "hyg_bas_nat":             "Basic Hygiene",
    "completion_primary_f":    "Female Primary Completion",
    "literacy_f":              "Female Literacy",
    "diarrhoea_care_pct":      "Diarrhoea Treatment (%)",
    "pneumonia_care_pct":      "Pneumonia Care Seeking (%)",
    "exclusive_bf_pct":        "Exclusive Breastfeeding (%)",
    "low_birthweight_pct":     "Low Birthweight (%)",
    "income_group":            "Income Group"
}

def load_and_merge_data():
    print("=" * 60)
    print("DATA LOADING & MERGING")
    print("=" * 60)

    df = pd.read_csv(MALNUTRITION_CSV)
    print(f"Base Dataset: {len(df)} countries")

    wash_df = pd.read_csv(WASH_CSV) if os.path.exists(WASH_CSV) else pd.DataFrame(columns=["ISO"] + WASH_PREDICTORS)
    if "iso3" in wash_df.columns:
        wash_df = wash_df.rename(columns={"iso3": "ISO"})

    edu_df = pd.read_csv(EDU_CSV) if os.path.exists(EDU_CSV) else pd.DataFrame(columns=["ISO"] + EDU_PREDICTORS)

    if "year" in wash_df.columns:
        wash_df = wash_df.sort_values("year").groupby("ISO").tail(1)

    df_m3 = df.copy()
    if not wash_df.empty:
        df_m3 = df_m3.merge(wash_df[["ISO"] + [p for p in WASH_PREDICTORS if p in wash_df.columns]], on="ISO", how="left")
    if not edu_df.empty:
        df_m3 = df_m3.merge(edu_df[["ISO"] + [p for p in EDU_PREDICTORS if p in edu_df.columns]], on="ISO", how="left")

    print(f"Merged Dataset: {len(df_m3)} countries")
    return df_m3

def run_xgboost_analysis(df):
    metrics_list = []
    shap_importance_list = []
    print("\n" + "=" * 60)
    print("XGBOOST MODELING & SHAP INTERPRETABILITY")
    print("=" * 60)

    all_predictors = GENDER_PREDICTORS + WASH_PREDICTORS + EDU_PREDICTORS + HEALTH_PREDICTORS + ["income_group"]

    if "income_group" in df.columns:
        df["income_group"] = df["income_group"].astype("category")

    for outcome in OUTCOMES:
        if outcome not in df.columns:
            continue

        print(f"\nTarget Outcome: {OUTCOME_LABELS[outcome]}")
        print("-" * 50)

        sub = df.dropna(subset=[outcome]).copy()

        current_predictors = [p for p in all_predictors if p in sub.columns]

        X = sub[current_predictors]
        y = sub[outcome]

        print(f"Sample Size (N): {len(sub)} countries")

        model = xgb.XGBRegressor(
            missing=np.nan,
            enable_categorical=True,
            max_depth=2,
            learning_rate=0.05,
            n_estimators=300,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=1.0,
            reg_lambda=5.0,
            min_child_weight=5,
            random_state=42,
            early_stopping_rounds=30,
        )

        # Use 20% holdout for early stopping
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        print(f"  Early stopping: best iteration = {model.best_iteration}")

        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))

        print(f"Performance Metrics (full data):")
        print(f"  R-squared (R\u00b2): {r2:.3f}")
        print(f"  RMSE:           {rmse:.3f}")

        print("Calculating SHAP values...")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X)

        # Rename features in the SHAP explanation object for better plotting labels
        shap_values.feature_names = [PREDICTOR_LABELS.get(f, f) for f in current_predictors]

        # Create summary plot
        fig, ax = plt.subplots(figsize=(12, 7))
        shap.summary_plot(shap_values, X, show=False)
        ax.set_title(f"SHAP Feature Importance: {OUTCOME_LABELS[outcome]}",
                     fontsize=16, fontweight="bold", pad=20)
        ax.set_xlabel("SHAP value (impact on model output)", fontsize=12)
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.tight_layout()

        # Save plot
        plot_path = os.path.join(OUTPUT_FOLDER, f"xgboost_shap_{outcome}.png")
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved SHAP plot to: {plot_path}")

        print("Generating XGBoost native importance plots (gain, weight, cover)...")

        model.get_booster().feature_names = [PREDICTOR_LABELS.get(f, f) for f in current_predictors]

        accent = OUTCOME_COLORS.get(outcome, UNICEF_BLUE)
        for imp_type in ['gain', 'weight', 'cover']:
            fig, ax = plt.subplots(figsize=(11, 7))
            xgb.plot_importance(model, importance_type=imp_type, show_values=False,
                              title=f"XGBoost Feature Importance ({imp_type.capitalize()}): {OUTCOME_LABELS[outcome]}",
                              xlabel=f"F score ({imp_type})",
                              color=accent, alpha=0.85, ax=ax)
            ax.set_facecolor("#FAFAFA")
            for spine in ax.spines.values():
                spine.set_visible(False)
            fig.tight_layout()
            imp_path = os.path.join(OUTPUT_FOLDER, f"xgboost_importance_{imp_type}_{outcome}.png")
            fig.savefig(imp_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

        print(f"Saved native importance plots to {OUTPUT_FOLDER}/xgboost_importance_*_{outcome}.png")

        metrics_list.append({
            "Outcome": OUTCOME_LABELS[outcome],
            "N": len(sub),
            "R2": round(r2, 3),
            "RMSE": round(rmse, 3)
        })

        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            "Outcome": OUTCOME_LABELS[outcome],
            "Indicator": [PREDICTOR_LABELS.get(f, f) for f in current_predictors],
            "Mean_Abs_SHAP": mean_abs_shap
        })
        
        importance_df = importance_df.sort_values("Mean_Abs_SHAP", ascending=False)
        shap_importance_list.append(importance_df)

        fig, ax = plt.subplots(figsize=(11, 7))
        plot_df = importance_df.iloc[::-1]

        # Gradient color: strongest feature darkest
        norm_vals = plot_df["Mean_Abs_SHAP"] / plot_df["Mean_Abs_SHAP"].max()
        bar_colors = [to_rgba(accent, 0.35 + 0.65 * v) for v in norm_vals]

        bars = ax.barh(plot_df["Indicator"], plot_df["Mean_Abs_SHAP"],
                       color=bar_colors, edgecolor="white", linewidth=0.5, height=0.7)

        ax.set_title(f"Average Impact on {OUTCOME_LABELS[outcome]}\n(Mean |SHAP| Value)",
                     fontsize=16, fontweight="bold", pad=18)
        ax.set_xlabel("Mean Absolute Impact (Percentage Points)", fontsize=12)
        ax.xaxis.grid(True)
        ax.yaxis.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        for bar in bars:
            w = bar.get_width()
            ax.text(w + plot_df["Mean_Abs_SHAP"].max() * 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    f"{w:.2f}", ha="left", va="center", fontsize=10, fontweight="bold")

        fig.tight_layout()
        bar_plot_path = os.path.join(OUTPUT_FOLDER, f"xgboost_importance_bar_{outcome}.png")
        fig.savefig(bar_plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved Importance Bar chart to: {bar_plot_path}")

        print("Running 10-Fold Cross Validation...")
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        
        cv_model = xgb.XGBRegressor(
            missing=np.nan,
            enable_categorical=True,
            max_depth=2,
            learning_rate=0.05,
            n_estimators=model.best_iteration + 1,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=1.0,
            reg_lambda=5.0,
            min_child_weight=5,
            random_state=42,
        )
        cv_preds = cross_val_predict(cv_model, X, y, cv=kf)
        cv_r2 = r2_score(y, cv_preds)
        cv_rmse = np.sqrt(mean_squared_error(y, cv_preds))
        
        print(f"  CV R-squared (R\u00b2): {cv_r2:.3f}")
        print(f"  CV RMSE:           {cv_rmse:.3f}")
        print(f"  Overfit gap (Train R\u00b2 - CV R\u00b2): {r2 - cv_r2:.3f}")
        
        metrics_list[-1]["CV_R2"] = round(cv_r2, 3)
        metrics_list[-1]["CV_RMSE"] = round(cv_rmse, 3)

        fig, ax = plt.subplots(figsize=(9, 9))
        ax.scatter(y, cv_preds, alpha=0.7, color=accent, edgecolor="white",
                   s=60, linewidth=0.6, zorder=3)

        min_val = min(y.min(), cv_preds.min()) - 1
        max_val = max(y.max(), cv_preds.max()) + 1
        ax.plot([min_val, max_val], [min_val, max_val],
                color="#555555", ls="--", lw=1.8, alpha=0.7, label="Perfect Prediction", zorder=2)
        ax.fill_between([min_val, max_val],
                        [min_val - 5, max_val - 5],
                        [min_val + 5, max_val + 5],
                        alpha=0.06, color=accent, zorder=1)

        ax.set_title(f"Actual vs Predicted: {OUTCOME_LABELS[outcome]}\n(10-Fold Cross-Validation, R\u00b2 = {cv_r2:.3f})",
                     fontsize=16, fontweight="bold", pad=18)
        ax.set_xlabel(f"Actual {OUTCOME_LABELS[outcome]}", fontsize=13)
        ax.set_ylabel(f"Predicted {OUTCOME_LABELS[outcome]}", fontsize=13)
        ax.legend(fontsize=11, loc="upper left", framealpha=0.9)
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.set_aspect("equal")
        for spine in ax.spines.values():
            spine.set_visible(False)

        fig.tight_layout()
        scatter_path = os.path.join(OUTPUT_FOLDER, f"xgboost_actual_vs_pred_{outcome}.png")
        fig.savefig(scatter_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved Actual vs Pred plot to: {scatter_path}")

    if metrics_list:
        pd.DataFrame(metrics_list).to_csv(os.path.join(OUTPUT_FOLDER, "xgboost_performance_metrics.csv"), index=False)
        print("\nSaved: outputs/xgboost_performance_metrics.csv")

    if shap_importance_list:
        pd.concat(shap_importance_list, ignore_index=True).to_csv(os.path.join(OUTPUT_FOLDER, "xgboost_shap_importance_rankings.csv"), index=False)
        print("Saved: outputs/xgboost_shap_importance_rankings.csv")


df = load_and_merge_data()
run_xgboost_analysis(df)
print("\n" + "=" * 60)
print("XGBOOST ANALYSIS COMPLETE")
print("=" * 60)