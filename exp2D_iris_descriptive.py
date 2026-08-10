# =============================================================================
# Experiment 2D: EXPLORING DESCRIPTIVE ANALYTICS USING THE IRIS DATASET
# -----------------------------------------------------------------------------
# AIM: Explore descriptive analytics on the Iris dataset using Pandas and
#      Seaborn - histograms, boxplots, and a pairplot.
# RUN AS: A single Google Colab cell.
# =============================================================================

!pip install -q pandas matplotlib seaborn scikit-learn

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# --- Load dataset (built into scikit-learn; no upload needed) ---
iris = load_iris(as_frame=True)
df = iris.frame.copy()
df.columns = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)",
              "petal width (cm)", "target"]
df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
df = df.drop(columns=["target"])

# --- Basic info and summary statistics ---
print("Basic Information:")
df.info()
print("\nSummary Statistics:\n", df.describe())
print("\nSpecies Count:\n", df["species"].value_counts())

# --- Histograms of all four features ---
df.hist(figsize=(8, 6), edgecolor="black", color="#4C72B0")
plt.suptitle("Feature Distributions")
plt.tight_layout()
plt.savefig("exp2d_iris_histograms.png", dpi=130)
plt.show()

# --- Boxplot: Sepal Length by species ---
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="species", y="sepal length (cm)", palette="Set2")
plt.title("Sepal Length Comparison Across Species")
plt.tight_layout()
plt.savefig("exp2d_iris_boxplot.png", dpi=130)
plt.show()

# --- Pairplot: all feature relationships coloured by species ---
sns.pairplot(df, hue="species", palette="Set2")
plt.savefig("exp2d_iris_pairplot.png", dpi=130)
plt.show()

print("\n[RESULT] Descriptive analytics on the Iris dataset completed and visualized.")

# =============================================================================
# RESULT
# Descriptive analytics on the Iris dataset were successfully demonstrated using
# Pandas and Seaborn. Histograms show feature distributions, the boxplot shows
# sepal length separating by species, and the pairplot shows petal measurements
# give the cleanest species separation.
# =============================================================================
