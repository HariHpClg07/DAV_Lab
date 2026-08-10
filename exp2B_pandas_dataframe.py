# =============================================================================
# Experiment 2B: EXPLORING PANDAS DATAFRAME OPERATIONS
# -----------------------------------------------------------------------------
# AIM: Explore DataFrame operations - loading, inspection, missing values,
#      transformations, filtering, grouping, sorting, saving - visualized.
#
# DATA: sales_data.csv (40-row synthetic sales dataset: OrderID, Product,
#       Category, Price, Quantity, Region). Upload it alongside this notebook,
#       or regenerate it with the snippet at the bottom of this cell.
# RUN AS: A single Google Colab cell.
# =============================================================================

!pip install -q pandas matplotlib seaborn numpy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Regenerate sales_data.csv if not already uploaded ---
import os
if not os.path.exists("sales_data.csv"):
    rng = np.random.default_rng(3)
    sales = pd.DataFrame({
        "OrderID": range(1001, 1041),
        "Product": rng.choice(["Laptop", "Monitor", "Keyboard", "Mouse", "Headset"], 40),
        "Category": rng.choice(["Electronics", "Accessories"], 40),
        "Price": rng.integers(15, 1200, 40),
        "Quantity": rng.integers(1, 12, 40),
        "Region": rng.choice(["North", "South", "East", "West"], 40),
    })
    sales.loc[rng.choice(40, 4, replace=False), "Price"] = np.nan
    sales.to_csv("sales_data.csv", index=False)

# --- 1. Load and inspect ---
df = pd.read_csv("sales_data.csv")
print("First 5 rows:\n", df.head())
print("\nLast 5 rows:\n", df.tail())
df.info()
print("\nSummary statistics:\n", df.describe())

# --- 2. Handle missing values ---
df["Price"] = df["Price"].fillna(df["Price"].mean())

# --- 3. New column + Series operations ---
df["TotalValue"] = df["Price"] * df["Quantity"]
series = df["Price"]
print("\nSeries + 10 (first 5):\n", (series + 10).head())

# --- 4. Filter rows on multiple conditions ---
filtered_df = df[(df["Price"] > 200) & (df["Quantity"] < 8)]
print("\nFiltered rows (Price>200 & Quantity<8):", len(filtered_df))

# --- 5. Grouping and aggregation ---
grouped = df.groupby("Region")["TotalValue"].mean().sort_values(ascending=False)
print("\nMean order value by region:\n", grouped)

# --- 6. Sorting ---
df_sorted = df.sort_values(by="TotalValue", ascending=False)

# --- 7. Boolean masking ---
masked_df = df[df["TotalValue"] > df["TotalValue"].median()]

# --- 8. Remove duplicates, drop remaining NaNs, save subset ---
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
subset_df = df[["OrderID", "Product", "TotalValue"]]
subset_df.to_csv("filtered_sales.csv", index=False)

print(f"\nTotal sum: {df['TotalValue'].sum():.2f}")
print(f"Mean: {df['TotalValue'].mean():.2f}")
print(f"Standard Deviation: {df['TotalValue'].std():.2f}")

# =============================== VISUALIZATION ===============================
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

grouped.plot(kind="bar", ax=axes[0, 0], color="#4C72B0")
axes[0, 0].set_title("Mean Order Value by Region (groupby)")
axes[0, 0].set_ylabel("Mean Total Value")
axes[0, 0].tick_params(axis='x', rotation=0)

sns.boxplot(data=df, x="Product", y="TotalValue", ax=axes[0, 1])
axes[0, 1].set_title("Order Value Distribution by Product")
axes[0, 1].tick_params(axis='x', rotation=30)

top10 = df_sorted.head(10)
axes[1, 0].barh(top10["OrderID"].astype(str), top10["TotalValue"], color="#DD8452")
axes[1, 0].invert_yaxis()
axes[1, 0].set_title("Top 10 Orders by Total Value (sort_values)")
axes[1, 0].set_xlabel("Total Value")

sns.scatterplot(data=df, x="Price", y="Quantity", hue="Category",
                 style=df["TotalValue"] > df["TotalValue"].median(), ax=axes[1, 1])
axes[1, 1].set_title("Price vs Quantity (shape = above-median mask)")

plt.tight_layout()
plt.savefig("exp2b_pandas_dataframe.png", dpi=130)
plt.show()

print("\n[RESULT] Pandas DataFrame operations demonstrated and visualized successfully.")

# =============================================================================
# RESULT
# Loading, inspecting, cleaning, transforming, filtering, grouping, sorting, and
# exporting a DataFrame were demonstrated. Each operation is paired with a chart
# that visually confirms its effect on the data.
# =============================================================================
