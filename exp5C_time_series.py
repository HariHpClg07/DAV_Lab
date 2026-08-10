# =============================================================================
# Experiment 5C: TIME SERIES ANALYSIS
# -----------------------------------------------------------------------------
# AIM: Perform time series analysis on a diabetes-related sequence (Glucose
#      readings treated as a sequential series), showing trend, seasonality,
#      moving-average smoothing, and ARIMA forecasting.
# RUN AS: A single Google Colab cell.
# =============================================================================

!pip install -q pandas numpy matplotlib statsmodels

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

diabetes_data = pd.read_csv("diabetes_timeseries.csv")
print(diabetes_data.head())

# --- 1. Plot the raw time series ---
plt.figure(figsize=(12, 4.5))
plt.plot(diabetes_data["Glucose"], label="Glucose Level", color="blue")
plt.xlabel("Index"); plt.ylabel("Glucose Level")
plt.title("Time Series of Glucose Levels")
plt.legend()
plt.tight_layout()
plt.savefig("exp5c_timeseries_raw.png", dpi=130)
plt.show()

# --- 2. Decompose into trend, seasonality, residuals ---
decomposition = seasonal_decompose(diabetes_data["Glucose"], model="additive", period=30)
fig, axes = plt.subplots(3, 1, figsize=(12, 8))
decomposition.trend.plot(ax=axes[0], title="Trend Component")
decomposition.seasonal.plot(ax=axes[1], title="Seasonal Component")
decomposition.resid.plot(ax=axes[2], title="Residual Component")
plt.tight_layout()
plt.savefig("exp5c_timeseries_decomposition.png", dpi=130)
plt.show()

# --- 3. Moving-average smoothing ---
diabetes_data["Glucose_MA"] = diabetes_data["Glucose"].rolling(window=7).mean()
plt.figure(figsize=(12, 4.5))
plt.plot(diabetes_data["Glucose"], label="Original", alpha=0.5)
plt.plot(diabetes_data["Glucose_MA"], label="7-point Moving Average", color="red")
plt.legend(); plt.title("Moving Average Smoothing")
plt.tight_layout()
plt.savefig("exp5c_timeseries_moving_average.png", dpi=130)
plt.show()

# --- 4. ARIMA forecasting ---
train_size = int(len(diabetes_data) * 0.8)
train, test = diabetes_data["Glucose"][:train_size], diabetes_data["Glucose"][train_size:]
model = ARIMA(train, order=(5, 1, 0))
fitted_model = model.fit()
forecast = fitted_model.forecast(steps=len(test))

plt.figure(figsize=(12, 4.5))
plt.plot(range(len(test)), test, label="Actual", color="blue")
plt.plot(range(len(test)), forecast, label="Forecast", color="red")
plt.xlabel("Index"); plt.ylabel("Glucose Level")
plt.title("ARIMA Model Forecasting")
plt.legend()
plt.tight_layout()
plt.savefig("exp5c_timeseries_arima.png", dpi=130)
plt.show()

print("[RESULT] Time series analysis and ARIMA forecasting completed and visualized.")

# =============================================================================
# RESULT
# The Time Series Analysis identified trend and (period-30) seasonal patterns in
# glucose levels, moving-average smoothing reduced noise, and the ARIMA(5,1,0)
# model produced a forecast. The forecast flattens toward the series mean
# fairly quickly, indicating the sequence behaves close to random around a
# stable mean rather than following strong autocorrelated trends.
# =============================================================================
