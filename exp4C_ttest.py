# =============================================================================
# Experiment 4C: PERFORMING T-TEST ON DIABETES DATASETS
# -----------------------------------------------------------------------------
# AIM: Perform an independent T-test on the UCI and Pima Diabetes datasets to
#      compare the means of Glucose, BloodPressure, and BMI, visualized with
#      bar charts of means +/- standard error and significance annotation.
# RUN AS: A single Google Colab cell.
# =============================================================================

!pip install -q pandas numpy matplotlib scipy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

uci_diabetes = pd.read_csv("uci_diabetes.csv")
pima_diabetes = pd.read_csv("pima_diabetes.csv")

numerical_columns = ["Glucose", "BloodPressure", "BMI"]

t_test_results = {}
for col in numerical_columns:
    t_stat, p_value = ttest_ind(uci_diabetes[col], pima_diabetes[col], equal_var=False)
    t_test_results[col] = {"T-statistic": t_stat, "P-value": p_value}

t_test_df = pd.DataFrame(t_test_results).T
print("T-test Results:\n", t_test_df)

# =============================== VISUALIZATION ===============================
fig, axes = plt.subplots(1, len(numerical_columns), figsize=(14, 4.5))

for ax, col in zip(axes, numerical_columns):
    means = [uci_diabetes[col].mean(), pima_diabetes[col].mean()]
    sems = [uci_diabetes[col].std() / np.sqrt(len(uci_diabetes)),
            pima_diabetes[col].std() / np.sqrt(len(pima_diabetes))]
    p_value = t_test_df.loc[col, "P-value"]
    bars = ax.bar(["UCI", "Pima"], means, yerr=sems, capsize=6,
                   color=["#4C72B0", "#DD8452"])
    sig = "significant (p<0.05)" if p_value < 0.05 else "not significant (p\u22650.05)"
    ax.set_title(f"{col}\np={p_value:.4f} - {sig}", fontsize=10)
    ax.set_ylabel("Mean \u00b1 SEM")

plt.tight_layout()
plt.savefig("exp4c_ttest.png", dpi=130)
plt.show()

print("\n[RESULT] T-test completed and visualized for Glucose, BloodPressure, and BMI.")

# =============================================================================
# RESULT
# The T-test compares means of Glucose, Blood Pressure, and BMI between the UCI
# and Pima datasets. Bars annotated with p-values show which feature means
# differ significantly between the two cohorts.
# =============================================================================
