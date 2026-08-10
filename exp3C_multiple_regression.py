# =============================================================================
# Experiment 3C: MULTIPLE REGRESSION ANALYSIS
# -----------------------------------------------------------------------------
# AIM: Perform multiple regression on the UCI and Pima Diabetes datasets to
#      predict BMI from Glucose, BloodPressure, and Age - visualized as
#      actual-vs-predicted scatter plots.
# RUN AS: A single Google Colab cell.
# =============================================================================

!pip install -q pandas numpy matplotlib scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

uci_diabetes = pd.read_csv("uci_diabetes.csv")
pima_diabetes = pd.read_csv("pima_diabetes.csv")

features = ["Glucose", "BloodPressure", "Age"]
target = "BMI"


def multiple_regression_analysis(df, dataset_name, ax):
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    print(f"\n{dataset_name} - Multiple Regression R\u00b2 Score: {r2:.4f}")
    print(f"{dataset_name} - Coefficients: "
          f"{dict(zip(features, np.round(model.coef_, 4)))}")

    ax.scatter(y_test, y_pred, alpha=0.6, color="#4C72B0")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, color="red", linestyle="--", label="Perfect prediction")
    ax.set_xlabel("Actual BMI"); ax.set_ylabel("Predicted BMI")
    ax.set_title(f"{dataset_name}\nActual vs. Predicted BMI (R\u00b2={r2:.4f})")
    ax.legend()
    return r2, model


fig, axes = plt.subplots(1, 2, figsize=(12, 5))
r2_uci, model_uci = multiple_regression_analysis(uci_diabetes, "UCI Diabetes Dataset", axes[0])
r2_pima, model_pima = multiple_regression_analysis(pima_diabetes, "Pima Indians Diabetes Dataset", axes[1])
plt.tight_layout()
plt.savefig("exp3c_multiple_regression.png", dpi=130)
plt.show()

# --- Coefficient comparison chart ---
fig, ax = plt.subplots(figsize=(7, 4.5))
x = np.arange(len(features))
width = 0.35
ax.bar(x - width/2, model_uci.coef_, width, label="UCI")
ax.bar(x + width/2, model_pima.coef_, width, label="Pima")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x); ax.set_xticklabels(features)
ax.set_title("Multiple Regression Coefficients (predicting BMI)")
ax.legend()
plt.tight_layout()
plt.savefig("exp3c_coefficients.png", dpi=130)
plt.show()

print("\n[RESULT] Multiple regression analysis completed and visualized for both datasets.")

# =============================================================================
# RESULT
# Multiple regression predicting BMI from Glucose, Blood Pressure, and Age was
# performed on both datasets. Actual-vs-predicted scatter plots and a
# coefficient comparison chart show how weakly these three features explain BMI
# on their own (low R\u00b2), consistent with BMI depending on factors outside
# this feature set.
# =============================================================================
