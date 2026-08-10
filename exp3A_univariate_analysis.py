# =============================================================================
# Experiment 3A: STATISTICAL ANALYSIS USING DIABETES DATASETS - UNIVARIATE
# -----------------------------------------------------------------------------
# AIM: Analyze the UCI Diabetes and Pima Indians Diabetes datasets using
#      univariate statistics (Mean, Median, Mode, Variance, Std Dev, Skewness,
#      Kurtosis), visualized as comparative bar charts.
#
# DATA: uci_diabetes.csv, pima_diabetes.csv (both 768 rows x 9 cols: Glucose,
#       BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction,
#       Age, Outcome). pima_diabetes.csv is the real Pima Indians Diabetes
#       dataset (Sigillito, Johns Hopkins, 1990). uci_diabetes.csv here is a
#       synthetic same-schema cohort generated to demonstrate a two-dataset
#       comparison exactly as the manual's procedure requires - replace it
#       with your own UCI file if you have one for submission.
# RUN AS: A single Google Colab cell.
# =============================================================================

!pip install -q pandas numpy scipy matplotlib seaborn

import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import seaborn as sns

uci_diabetes = pd.read_csv("uci_diabetes.csv")
pima_diabetes = pd.read_csv("pima_diabetes.csv")

print("UCI Diabetes Dataset Sample:\n", uci_diabetes.head())
print("\nPima Indians Diabetes Dataset Sample:\n", pima_diabetes.head())

numerical_columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
                      "DiabetesPedigreeFunction", "Age"]


def univariate_analysis(df, columns):
    stats = {}
    for col in columns:
        stats[col] = {
            "Mean": np.mean(df[col]),
            "Median": np.median(df[col]),
            "Mode": df[col].mode()[0],
            "Variance": np.var(df[col], ddof=1),
            "Standard Deviation": np.std(df[col], ddof=1),
            "Skewness": skew(df[col]),
            "Kurtosis": kurtosis(df[col]),
        }
    return pd.DataFrame(stats).T


uci_stats = univariate_analysis(uci_diabetes, numerical_columns)
pima_stats = univariate_analysis(pima_diabetes, numerical_columns)

print("\nUCI Diabetes Dataset Statistics:\n", uci_stats)
print("\nPima Indians Diabetes Dataset Statistics:\n", pima_stats)

# =============================== VISUALIZATION ===============================
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

x = np.arange(len(numerical_columns))
width = 0.35
axes[0, 0].bar(x - width/2, uci_stats["Mean"], width, label="UCI")
axes[0, 0].bar(x + width/2, pima_stats["Mean"], width, label="Pima")
axes[0, 0].set_xticks(x); axes[0, 0].set_xticklabels(numerical_columns, rotation=40, ha="right")
axes[0, 0].set_title("Mean by Feature"); axes[0, 0].legend()

axes[0, 1].bar(x - width/2, uci_stats["Standard Deviation"], width, label="UCI")
axes[0, 1].bar(x + width/2, pima_stats["Standard Deviation"], width, label="Pima")
axes[0, 1].set_xticks(x); axes[0, 1].set_xticklabels(numerical_columns, rotation=40, ha="right")
axes[0, 1].set_title("Standard Deviation by Feature"); axes[0, 1].legend()

axes[1, 0].bar(x - width/2, uci_stats["Skewness"], width, label="UCI")
axes[1, 0].bar(x + width/2, pima_stats["Skewness"], width, label="Pima")
axes[1, 0].axhline(0, color="black", linewidth=0.8)
axes[1, 0].set_xticks(x); axes[1, 0].set_xticklabels(numerical_columns, rotation=40, ha="right")
axes[1, 0].set_title("Skewness by Feature"); axes[1, 0].legend()

sns.histplot(uci_diabetes["Glucose"], kde=True, color="#4C72B0", label="UCI", ax=axes[1, 1], alpha=0.5)
sns.histplot(pima_diabetes["Glucose"], kde=True, color="#DD8452", label="Pima", ax=axes[1, 1], alpha=0.5)
axes[1, 1].set_title("Glucose Distribution: UCI vs. Pima")
axes[1, 1].legend()

plt.tight_layout()
plt.savefig("exp3a_univariate_comparison.png", dpi=130)
plt.show()

print("\n[RESULT] Univariate statistical analysis completed and visualized for both datasets.")

# =============================================================================
# RESULT
# The univariate analysis of the UCI and Pima Indians Diabetes datasets reveals
# differences in central tendency, dispersion, and distribution. The bar charts
# and overlaid histogram make these differences visually explicit.
# =============================================================================
