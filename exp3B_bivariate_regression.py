# =============================================================================
# Experiment 3B: BIVARIATE ANALYSIS - LINEAR AND LOGISTIC REGRESSION MODELING
# -----------------------------------------------------------------------------
# AIM: Perform bivariate analysis on the UCI and Pima Diabetes datasets using
#      Linear Regression (Glucose -> BMI) and Logistic Regression (predicting
#      Outcome), each visualized.
# RUN AS: A single Google Colab cell.
# =============================================================================

!pip install -q pandas numpy matplotlib seaborn scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix

uci_diabetes = pd.read_csv("uci_diabetes.csv")
pima_diabetes = pd.read_csv("pima_diabetes.csv")

# =============================== LINEAR REGRESSION ============================
def linear_regression_analysis(df, x_column, y_column, ax, label):
    X = df[[x_column]]
    Y = df[y_column]
    model = LinearRegression()
    model.fit(X, Y)
    Y_pred = model.predict(X)
    r2 = r2_score(Y, Y_pred)
    print(f"\nLinear Regression (Predicting {y_column} using {x_column}) - {label}:")
    print(f"R\u00b2 Score: {r2:.4f}")

    ax.scatter(X, Y, color="blue", alpha=0.5, label="Actual Data", s=15)
    order = np.argsort(X[x_column].values)
    ax.plot(X[x_column].values[order], Y_pred[order], color="red", linewidth=2, label="Regression Line")
    ax.set_xlabel(x_column); ax.set_ylabel(y_column)
    ax.set_title(f"{label}: {x_column} vs. {y_column} (R\u00b2={r2:.4f})")
    ax.legend()
    return r2


fig, axes = plt.subplots(1, 2, figsize=(13, 5))
r2_uci = linear_regression_analysis(uci_diabetes, "Glucose", "BMI", axes[0], "UCI Diabetes")
r2_pima = linear_regression_analysis(pima_diabetes, "Glucose", "BMI", axes[1], "Pima Diabetes")
plt.tight_layout()
plt.savefig("exp3b_linear_regression.png", dpi=130)
plt.show()

# ============================== LOGISTIC REGRESSION ============================
def logistic_regression_analysis(df, features, target, ax, label):
    X = df[features]
    Y = df[target]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    print(f"\nLogistic Regression (Predicting {target} using {features}) - {label}:")
    print(f"Accuracy Score: {accuracy:.4f}")

    cm = confusion_matrix(Y_test, Y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
    ax.set_title(f"{label} Confusion Matrix (Acc={accuracy:.3f})")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    return accuracy


features = ["Glucose", "BloodPressure", "BMI", "Age"]
target = "Outcome"

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
acc_uci = logistic_regression_analysis(uci_diabetes, features, target, axes[0], "UCI Diabetes")
acc_pima = logistic_regression_analysis(pima_diabetes, features, target, axes[1], "Pima Diabetes")
plt.tight_layout()
plt.savefig("exp3b_logistic_regression.png", dpi=130)
plt.show()

print("\n[RESULT] Bivariate linear and logistic regression modeling completed and visualized.")

# =============================================================================
# RESULT
# Linear Regression revealed the (weak) relationship between Glucose and BMI in
# both datasets. Logistic Regression predicted diabetes Outcome from Glucose,
# BloodPressure, BMI and Age, with confusion matrices confirming classification
# performance on held-out test data.
# =============================================================================
