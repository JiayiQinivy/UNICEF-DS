"""
UNICEF Child Food Poverty Data Visualization
=============================================
Reads the UNICEF Expanded Global Databases on Child Food Poverty (2024)
and produces several charts summarising severe and moderate food poverty
across regions, income groups, and individual countries.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np


FILE = "UNICEF_Expanded_Global_Databases_child_food_poverty_2024_2.xlsx"


summary = pd.read_excel(FILE, sheet_name="Latest Regional Global", header=None)


summary.columns = range(summary.shape[1])
summary = summary.rename(columns={1: "Region", 3: "Period", 4: "Severe", 5: "Moderate"})


summary = summary.dropna(subset=["Severe", "Moderate"], how="all")
summary = summary[summary["Severe"].apply(lambda x: str(x).replace(".", "").replace("-", "").strip().isdigit())]


summary["Severe"] = pd.to_numeric(summary["Severe"], errors="coerce")
summary["Moderate"] = pd.to_numeric(summary["Moderate"], errors="coerce")
summary["Region"] = summary["Region"].str.strip()


unicef_regions = [
    "East Asia and the Pacific",
    "Eastern and Southern Africa",
    "Middle East and North Africa",
    "South Asia",
    "West and Central Africa",
]
region_df = summary[summary["Region"].isin(unicef_regions)].copy()


income_groups = ["Low Income", "Lower Middle Income", "Upper Middle Income"]
income_df = summary[summary["Region"].isin(income_groups)].copy()


global_df = summary[summary["Region"] == "Global"]


severe_raw = pd.read_excel(FILE, sheet_name="Severe_food_poverty", header=None)


col_names = severe_raw.iloc[8].tolist()
data = severe_raw.iloc[9:].copy()
data.columns = col_names
data = data.reset_index(drop=True)


latest = data[data["LatestSource"] == "Latest Source"].copy()
latest["National_r"] = pd.to_numeric(latest["National_r"], errors="coerce")
latest = latest.dropna(subset=["National_r"])
latest = latest.sort_values("National_r", ascending=False)


mod_raw = pd.read_excel(FILE, sheet_name="Moderate_food_poverty", header=None)
mod_col_names = mod_raw.iloc[8].tolist()
mod_data = mod_raw.iloc[9:].copy()
mod_data.columns = mod_col_names
mod_data = mod_data.reset_index(drop=True)

mod_latest = mod_data[mod_data["LatestSource"] == "Latest Source"].copy()
mod_latest["National_r"] = pd.to_numeric(mod_latest["National_r"], errors="coerce")
mod_latest = mod_latest.dropna(subset=["National_r"])


plt.style.use("seaborn-v0_8-whitegrid")
fig = plt.figure(figsize=(20, 22))
fig.suptitle(
    "UNICEF Child Food Poverty – Global Overview (2017-2023)",
    fontsize=20,
    fontweight="bold",
    y=0.98,
)


ax1 = fig.add_subplot(3, 2, 1)
x = np.arange(len(region_df))
width = 0.35
bars1 = ax1.barh(x - width / 2, region_df["Severe"], width, label="Severe", color="#d62728")
bars2 = ax1.barh(x + width / 2, region_df["Moderate"], width, label="Moderate", color="#ff7f0e")
ax1.set_yticks(x)
ax1.set_yticklabels(region_df["Region"], fontsize=9)
ax1.set_xlabel("Prevalence (%)")
ax1.set_title("Severe vs Moderate Food Poverty\nby UNICEF Region", fontweight="bold")
ax1.legend(loc="lower right")
ax1.bar_label(bars1, fmt="%.0f%%", padding=3, fontsize=8)
ax1.bar_label(bars2, fmt="%.0f%%", padding=3, fontsize=8)
ax1.invert_yaxis()


ax2 = fig.add_subplot(3, 2, 2)
x2 = np.arange(len(income_df))
bars3 = ax2.bar(x2 - width / 2, income_df["Severe"], width, label="Severe", color="#d62728")
bars4 = ax2.bar(x2 + width / 2, income_df["Moderate"], width, label="Moderate", color="#ff7f0e")
ax2.set_xticks(x2)
ax2.set_xticklabels(income_df["Region"], fontsize=9)
ax2.set_ylabel("Prevalence (%)")
ax2.set_title("Food Poverty by World Bank\nIncome Group", fontweight="bold")
ax2.legend()
ax2.bar_label(bars3, fmt="%.0f%%", padding=3, fontsize=9)
ax2.bar_label(bars4, fmt="%.0f%%", padding=3, fontsize=9)


ax3 = fig.add_subplot(3, 2, (3, 4))
top20 = latest.head(20)
colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(top20)))
bars5 = ax3.barh(range(len(top20)), top20["National_r"], color=colors)
ax3.set_yticks(range(len(top20)))
ax3.set_yticklabels(top20["CountryName"], fontsize=9)
ax3.set_xlabel("Prevalence (%)")
ax3.set_title("Top 20 Countries – Severe Child Food Poverty (Latest Estimate)", fontweight="bold")
ax3.invert_yaxis()
ax3.bar_label(bars5, fmt="%.1f%%", padding=3, fontsize=8)


ax4 = fig.add_subplot(3, 2, 5)
ax4.hist(latest["National_r"], bins=20, color="#1f77b4", edgecolor="white", alpha=0.85)
ax4.axvline(latest["National_r"].median(), color="#d62728", linestyle="--", linewidth=2,
            label=f'Median: {latest["National_r"].median():.1f}%')
ax4.axvline(latest["National_r"].mean(), color="#2ca02c", linestyle="--", linewidth=2,
            label=f'Mean: {latest["National_r"].mean():.1f}%')
ax4.set_xlabel("Severe Food Poverty Prevalence (%)")
ax4.set_ylabel("Number of Countries")
ax4.set_title("Distribution of Severe Food Poverty\nAcross Countries", fontweight="bold")
ax4.legend()


ax5 = fig.add_subplot(3, 2, 6)

merged = pd.merge(
    latest[["CountryName", "National_r"]].rename(columns={"National_r": "Severe"}),
    mod_latest[["CountryName", "National_r"]].rename(columns={"National_r": "Moderate"}),
    on="CountryName",
    how="inner",
)
ax5.scatter(merged["Severe"], merged["Moderate"], alpha=0.6, edgecolors="k", linewidths=0.5, s=50, c="#1f77b4")

max_val = max(merged["Severe"].max(), merged["Moderate"].max()) + 5
ax5.plot([0, max_val], [0, max_val], "k--", alpha=0.3, label="1:1 line")
ax5.set_xlabel("Severe Food Poverty (%)")
ax5.set_ylabel("Moderate Food Poverty (%)")
ax5.set_title("Severe vs Moderate Food Poverty\n(Country-Level Latest Estimates)", fontweight="bold")


for _, row in merged.nlargest(5, "Severe").iterrows():
    ax5.annotate(row["CountryName"], (row["Severe"], row["Moderate"]),
                 fontsize=7, alpha=0.8, textcoords="offset points", xytext=(5, 5))
ax5.legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("food_poverty_charts.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n[OK] Visualisation saved to food_poverty_charts.png")
