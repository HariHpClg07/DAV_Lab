# =============================================================================
# Experiment 3D: COMPARISON OF ANALYSIS RESULTS BETWEEN UCI AND PIMA DATASETS
# -----------------------------------------------------------------------------
# AIM: Compare the univariate, bivariate, and multiple-regression results of
#      the UCI and Pima Indians Diabetes datasets, visualized as a dashboard.
# RUN AS: A single Google Colab cell.
# =============================================================================

!pip install -q pandas numpy matplotlib scikit-learn scipy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score

uci_diabetes = pd.read_csv("uci_diabetes.csv")
pima_diabetes = pd.read_csv("pima_diabetes.csv")

# --- Recompute the headline metrics from experiments 3A-3C for this dataset pair ---
uci_glucose_mean, pima_glucose_mean = uci_diabetes["Glucose"].mean(), pima_diabetes["Glucose"].mean()
uci_glucose_std, pima_glucose_std = uci_diabetes["Glucose"].std(), pima_diabetes["Glucose"].std()
uci_glucose_skew, pima_glucose_skew = skew(uci_diabetes["Glucose"]), skew(pima_diabetes["Glucose"])


def r2_glucose_bmi(df):
    X, Y = df[["Glucose"]], df["BMI"]
    model = LinearRegression().fit(X, Y)
    return r2_score(Y, model.predict(X))


def logistic_accuracy(df):
    features = ["Glucose", "BloodPressure", "BMI", "Age"]
    X, Y = df[features], df["Outcome"]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000).fit(X_train, Y_train)
    return accuracy_score(Y_test, model.predict(X_test))


uci_r2, pima_r2 = r2_glucose_bmi(uci_diabetes), r2_glucose_bmi(pima_diabetes)
uci_acc, pima_acc = logistic_accuracy(uci_diabetes), logistic_accuracy(pima_diabetes)

print(f"Glucose Mean:      UCI={uci_glucose_mean:.2f}, Pima={pima_glucose_mean:.2f}")
print(f"Glucose Std Dev:   UCI={uci_glucose_std:.2f}, Pima={pima_glucose_std:.2f}")
print(f"Glucose Skewness:  UCI={uci_glucose_skew:.4f}, Pima={pima_glucose_skew:.4f}")
print(f"Linear R\u00b2 (Glucose->BMI): UCI={uci_r2:.4f}, Pima={pima_r2:.4f}")
print(f"Logistic Accuracy (Outcome): UCI={uci_acc:.4f}, Pima={pima_acc:.4f}")

# =============================== VISUALIZATION ===============================
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
labels = ["UCI", "Pima"]

axes[0, 0].bar(labels, [uci_glucose_mean, pima_glucose_mean], color=["#4C72B0", "#DD8452"])
axes[0, 0].set_title("Glucose Mean"); axes[0, 0].set_ylabel("mg/dL")

axes[0, 1].bar(labels, [uci_glucose_std, pima_glucose_std], color=["#4C72B0", "#DD8452"])
axes[0, 1].set_title("Glucose Standard Deviation")

axes[1, 0].bar(labels, [uci_r2, pima_r2], color=["#4C72B0", "#DD8452"])
axes[1, 0].set_title("Linear Regression R\u00b2 (Glucose \u2192 BMI)")
axes[1, 0].set_ylim(0, max(0.1, uci_r2, pima_r2) * 1.3)

axes[1, 1].bar(labels, [uci_acc, pima_acc], color=["#4C72B0", "#DD8452"])
axes[1, 1].set_title("Logistic Regression Accuracy (Outcome)")
axes[1, 1].set_ylim(0, 1)

plt.tight_layout()
plt.savefig("exp3d_comparison_dashboard.png", dpi=130)
plt.show()

print("\n[RESULT] Comparison of statistical and model results between datasets completed and visualized.")

# =============================================================================
# RESULT
# The comparison dashboard summarises univariate (mean, std dev), bivariate
# (linear R\u00b2), and classification (logistic accuracy) results side by side,
# making dataset-level differences immediately visible.
# =============================================================================
