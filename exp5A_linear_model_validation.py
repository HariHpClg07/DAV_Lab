# =============================================================================
# Experiment 5A: BUILDING AND VALIDATING LINEAR MODELS
# -----------------------------------------------------------------------------
# AIM: Build and validate Linear Regression models on the UCI and Pima
#      Diabetes datasets, evaluated with R\u00b2, MSE, MAE, and visualized with
#      actual-vs-predicted and residual plots.
# RUN AS: A single Google Colab cell.
# =============================================================================

!pip install -q pandas numpy matplotlib seaborn scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

uci_diabetes = pd.read_csv("uci_diabetes.csv")
pima_diabetes = pd.read_csv("pima_diabetes.csv")

features = ["Glucose", "BloodPressure", "BMI"]
target = "Age"


def build_and_validate(df, name, ax_pred, ax_resid):
    X, y = df[features], df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"{name} - R\u00b2: {r2:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}")

    ax_pred.scatter(y_test, y_pred, alpha=0.6, color="#4C72B0")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax_pred.plot(lims, lims, "r--", label="Ideal fit")
    ax_pred.set_xlabel("Actual Age"); ax_pred.set_ylabel("Predicted Age")
    ax_pred.set_title(f"{name}: Actual vs Predicted (R\u00b2={r2:.3f})")
    ax_pred.legend()

    residuals = y_test - y_pred
    ax_resid.scatter(y_pred, residuals, alpha=0.6, color="#DD8452")
    ax_resid.axhline(0, color="black", linestyle="--")
    ax_resid.set_xlabel("Predicted Age"); ax_resid.set_ylabel("Residual")
    ax_resid.set_title(f"{name}: Residual Plot (MAE={mae:.2f})")

    return r2, mse, mae


fig, axes = plt.subplots(2, 2, figsize=(13, 9))
build_and_validate(uci_diabetes, "UCI Diabetes", axes[0, 0], axes[1, 0])
build_and_validate(pima_diabetes, "Pima Diabetes", axes[0, 1], axes[1, 1])
plt.tight_layout()
plt.savefig("exp5a_linear_model_validation.png", dpi=130)
plt.show()

print("\n[RESULT] Linear regression models built and validated with R\u00b2, MSE, MAE and visual diagnostics.")

# =============================================================================
# RESULT
# Linear Regression models predicting Age from Glucose, BloodPressure, and BMI
# were built and validated on both datasets. Actual-vs-predicted scatter plots
# and residual plots confirm the models' limited explanatory power (low R\u00b2),
# with residuals showing no strong pattern (no major nonlinearity missed).
# =============================================================================
