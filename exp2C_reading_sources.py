# =============================================================================
# Experiment 2C: READING DATA FROM TEXT FILES, EXCEL, AND THE WEB
# -----------------------------------------------------------------------------
# AIM: Read and process data from CSV, Excel, and a web-based source using
#      Pandas, then visualize each source.
# RUN AS: A single Google Colab cell.
# =============================================================================

!pip install -q pandas matplotlib seaborn openpyxl numpy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Regenerate local files if not already uploaded ---
if not os.path.exists("sales_data.csv") or not os.path.exists("sales_data.xlsx"):
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
    sales.to_excel("sales_data.xlsx", index=False)

# --- 1. Read from a text/CSV file ---
text_df = pd.read_csv("sales_data.csv")

# --- 2. Read from an Excel file ---
excel_df = pd.read_excel("sales_data.xlsx", sheet_name="Sheet1")

# --- 3. Read from a web-based source (a well-known public CSV on GitHub) ---
web_df = pd.read_csv(
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
    header=None,
    names=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
           "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"],
)

print("CSV source:\n", text_df.head())
print("\nExcel source:\n", excel_df.head())
print("\nWeb source:\n", web_df.head())

# --- 4. Handle missing values ---
text_df = text_df.ffill()
excel_df = excel_df.bfill()
web_df = web_df.dropna()

# --- 5. Save processed data ---
text_df.to_csv("processed_text.csv", index=False)
excel_df.to_excel("processed_excel.xlsx", index=False)

# =============================== VISUALIZATION ===============================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

text_df["Product"].value_counts().plot(kind="bar", ax=axes[0], color="#4C72B0")
axes[0].set_title("CSV Source: Product Counts")
axes[0].tick_params(axis='x', rotation=30)

excel_df.groupby("Region")["Price"].mean().plot(kind="bar", ax=axes[1], color="#DD8452")
axes[1].set_title("Excel Source: Mean Price by Region")
axes[1].tick_params(axis='x', rotation=0)

sns.histplot(web_df["Glucose"], kde=True, ax=axes[2], color="#55A868")
axes[2].set_title("Web Source: Glucose Distribution")

plt.tight_layout()
plt.savefig("exp2c_reading_sources.png", dpi=130)
plt.show()

print("\n[RESULT] Successfully read and visualized data from CSV, Excel, and web sources.")

# =============================================================================
# RESULT
# Data was successfully read from a text/CSV file, an Excel spreadsheet, and a
# live web-based CSV source, then cleaned and visualized to confirm each source
# loaded correctly.
# =============================================================================
