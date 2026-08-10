# =============================================================================
# Experiment 4A: DATA VISUALIZATION - NORMAL CURVES ON UCI DIABETES DATASET
# -----------------------------------------------------------------------------
# AIM: Visualize the distribution of key numerical attributes in the UCI
#      Diabetes dataset using histograms + KDE + overlaid normal curves.
# RUN AS: A single Google Colab cell.
# =============================================================================

!pip install -q pandas numpy matplotlib seaborn scipy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

uci_diabetes = pd.read_csv("uci_diabetes.csv")

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
features = ["Glucose", "BMI", "BloodPressure", "Age"]

for ax, feature in zip(axes.ravel(), features):
    sns.histplot(uci_diabetes[feature], kde=True, stat="density", ax=ax, color="#4C72B0")
    x = np.linspace(uci_diabetes[feature].min(), uci_diabetes[feature].max(), 100)
    ax.plot(x, norm.pdf(x, uci_diabetes[feature].mean(), uci_diabetes[feature].std()),
            'r', linewidth=2, label="Normal curve")
    ax.set_title(f"Normal Curve - {feature}")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("exp4a_normal_curves.png", dpi=130)
plt.show()

print("[RESULT] Normal curves plotted for Glucose, BMI, BloodPressure, and Age.")

# =============================================================================
# RESULT
# The normal curves overlaid on histograms of Glucose, BMI, BloodPressure, and
# Age show each feature's spread relative to a theoretical normal distribution
# with the same mean and standard deviation, revealing which features are
# close to normal and which are skewed.
# =============================================================================
