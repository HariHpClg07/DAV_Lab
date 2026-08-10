# =============================================================================
# Experiment 4B: HYPOTHESIS TESTING - Z-TEST ON UCI DIABETES DATASET
# -----------------------------------------------------------------------------
# AIM: Perform a Z-test on the UCI Diabetes dataset to determine whether the
#      mean Glucose level significantly differs from a population mean (100),
#      visualized with the sampling distribution and rejection region.
# RUN AS: A single Google Colab cell.
# =============================================================================

!pip install -q pandas numpy matplotlib statsmodels scipy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from statsmodels.stats.weightstats import ztest

uci_diabetes = pd.read_csv("uci_diabetes.csv")
POP_MEAN = 100
ALPHA = 0.05

# H0: mean Glucose = 100   H1: mean Glucose != 100
z_stat, p_value = ztest(uci_diabetes["Glucose"], value=POP_MEAN)

print(f"Sample Mean Glucose: {uci_diabetes['Glucose'].mean():.2f}")
print(f"Z-Statistic: {z_stat:.4f}")
print(f"P-Value: {p_value:.6f}")

if p_value < ALPHA:
    conclusion = "Reject the null hypothesis: mean Glucose is significantly different from 100."
else:
    conclusion = "Fail to reject the null hypothesis: no significant difference from 100."
print(conclusion)

# =============================== VISUALIZATION ===============================
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Left: sample distribution with population mean marker
axes[0].hist(uci_diabetes["Glucose"], bins=25, color="#4C72B0", alpha=0.7, density=True)
axes[0].axvline(POP_MEAN, color="green", linestyle="--", linewidth=2, label=f"H0 mean = {POP_MEAN}")
axes[0].axvline(uci_diabetes["Glucose"].mean(), color="red", linestyle="-", linewidth=2,
                label=f"Sample mean = {uci_diabetes['Glucose'].mean():.1f}")
axes[0].set_title("Glucose Distribution vs. Hypothesized Mean")
axes[0].set_xlabel("Glucose"); axes[0].legend(fontsize=8)

# Right: standard normal curve with critical region and observed Z shaded
z_crit = norm.ppf(1 - ALPHA / 2)
x = np.linspace(-12, 12, 1000)
y = norm.pdf(x)
axes[1].plot(x, y, color="black")
axes[1].fill_between(x, y, where=(x <= -z_crit) | (x >= z_crit), color="red", alpha=0.3,
                      label=f"Rejection region (\u03b1={ALPHA})")
axes[1].axvline(z_stat, color="blue", linewidth=2, label=f"Observed Z = {z_stat:.2f}")
axes[1].set_title("Z-Test: Standard Normal Distribution")
axes[1].set_xlim(-12, 12)
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig("exp4b_ztest.png", dpi=130)
plt.show()

print("\n[RESULT] Z-test completed and visualized -", conclusion)

# =============================================================================
# RESULT
# The Z-test shows the mean Glucose level in the UCI Diabetes dataset is
# significantly different from 100 (p < 0.05). The observed Z-statistic falls
# far into the rejection region, visually confirming the numerical result.
# =============================================================================
