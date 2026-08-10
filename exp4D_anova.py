# =============================================================================
# Experiment 4D: PERFORM ANOVA ON DIABETES DATASETS
# -----------------------------------------------------------------------------
# AIM: Perform One-Way ANOVA on the UCI and Pima Diabetes datasets to analyze
#      differences between group means, visualized with boxplots annotated
#      with the F-statistic and p-value.
# RUN AS: A single Google Colab cell.
# =============================================================================

!pip install -q pandas numpy matplotlib seaborn scipy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway

uci_diabetes = pd.read_csv("uci_diabetes.csv")
pima_diabetes = pd.read_csv("pima_diabetes.csv")

numerical_columns = ["Glucose", "BloodPressure", "BMI"]

anova_results = {}
for col in numerical_columns:
    f_stat, p_value = f_oneway(uci_diabetes[col], pima_diabetes[col])
    anova_results[col] = {"F-statistic": f_stat, "P-value": p_value}

anova_df = pd.DataFrame(anova_results).T
print("ANOVA Results:\n", anova_df)

# =============================== VISUALIZATION ===============================
combined = pd.concat([
    uci_diabetes[numerical_columns].assign(Dataset="UCI"),
    pima_diabetes[numerical_columns].assign(Dataset="Pima"),
])

fig, axes = plt.subplots(1, len(numerical_columns), figsize=(14, 5))
for ax, col in zip(axes, numerical_columns):
    sns.boxplot(data=combined, x="Dataset", y=col, ax=ax, palette=["#4C72B0", "#DD8452"])
    row = anova_df.loc[col]
    sig = "significant" if row["P-value"] < 0.05 else "not significant"
    ax.set_title(f"{col}\nF={row['F-statistic']:.3f}, p={row['P-value']:.4f} ({sig})", fontsize=9)

plt.tight_layout()
plt.savefig("exp4d_anova.png", dpi=130)
plt.show()

print("\n[RESULT] One-Way ANOVA completed and visualized for Glucose, BloodPressure, and BMI.")

# =============================================================================
# RESULT
# One-Way ANOVA compared group means of Glucose, Blood Pressure, and BMI across
# the UCI and Pima datasets. The boxplots, annotated with F- and p-values, show
# which features differ significantly between the two cohorts (equivalent to
# the two-sample T-test result here, since only two groups are compared).
# =============================================================================
