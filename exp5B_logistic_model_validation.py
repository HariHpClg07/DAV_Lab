# =============================================================================
# Experiment 5B: BUILDING AND VALIDATING LOGISTIC MODELS
# -----------------------------------------------------------------------------
# AIM: Build and validate Logistic Regression models predicting diabetes
#      Outcome on the UCI and Pima datasets, evaluated with accuracy,
#      precision, recall, F1, and visualized with confusion matrices and
#      ROC curves.
# RUN AS: A single Google Colab cell.
# =============================================================================

!pip install -q pandas numpy matplotlib seaborn scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              confusion_matrix, roc_curve, auc)

uci_diabetes = pd.read_csv("uci_diabetes.csv")
pima_diabetes = pd.read_csv("pima_diabetes.csv")

features = ["Glucose", "BloodPressure", "BMI", "Age"]
target = "Outcome"


def build_and_validate(df, name, ax_cm, ax_roc):
    X, y = df[features], df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    print(f"{name} - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm, cbar=False)
    ax_cm.set_title(f"{name} Confusion Matrix"); ax_cm.set_xlabel("Predicted"); ax_cm.set_ylabel("Actual")

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    ax_roc.plot(fpr, tpr, color="#4C72B0", linewidth=2, label=f"AUC = {roc_auc:.3f}")
    ax_roc.plot([0, 1], [0, 1], "k--", label="Chance")
    ax_roc.set_xlabel("False Positive Rate"); ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title(f"{name} ROC Curve"); ax_roc.legend()

    return acc, prec, rec, f1, roc_auc


fig, axes = plt.subplots(2, 2, figsize=(12, 10))
build_and_validate(uci_diabetes, "UCI Diabetes", axes[0, 0], axes[1, 0])
build_and_validate(pima_diabetes, "Pima Diabetes", axes[0, 1], axes[1, 1])
plt.tight_layout()
plt.savefig("exp5b_logistic_model_validation.png", dpi=130)
plt.show()

print("\n[RESULT] Logistic regression models built and validated with accuracy, precision, recall, F1, and ROC/AUC.")

# =============================================================================
# RESULT
# Logistic Regression models predicting diabetes Outcome were built and
# validated on both datasets. Confusion matrices show class-wise prediction
# errors, and ROC curves with AUC scores summarise overall discriminative
# ability beyond a single accuracy number.
# =============================================================================
