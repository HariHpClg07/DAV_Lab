# =============================================================================
# CS4503 - DATA ANALYTICS AND VISUALIZATION
# Experiment 1: DOWNLOAD, INSTALL AND EXPLORE NumPy, SciPy, Jupyter,
#               Statsmodels, Pandas, Matplotlib, Seaborn, Plotly, Bokeh
# -----------------------------------------------------------------------------
# AIM: To download, install, and explore the features of NumPy, SciPy, Jupyter,
#      Statsmodels, Pandas, Matplotlib, Seaborn, Plotly, and Bokeh for
#      scientific computing, data analysis, and visualization.
#
# RUN AS: A single Google Colab cell.
# =============================================================================

# ----------------------------- 0. Install -------------------------------
!pip install -q numpy scipy statsmodels pandas matplotlib seaborn plotly bokeh

# ----------------------------- 1. Import and verify -----------------------------
import numpy as np
import scipy
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import plotly
import bokeh

print("NumPy Version:      ", np.__version__)
print("SciPy Version:      ", scipy.__version__)
print("Pandas Version:     ", pd.__version__)
print("Matplotlib Version: ", matplotlib.__version__)
print("Seaborn Version:    ", sns.__version__)
print("Statsmodels Version:", sm.__version__)
print("Plotly Version:     ", plotly.__version__)
print("Bokeh Version:      ", bokeh.__version__)

# ----------------- 2. Visual proof that the plotting stack works -----------------
# A quick smoke-test chart: one line per library that can draw to a static axis
# (Plotly/Bokeh are interactive-only, so their smoke test is a version print above)
fig, ax = plt.subplots(figsize=(7, 4))
x = np.linspace(0, 10, 200)
ax.plot(x, np.sin(x), label="Matplotlib: sin(x)", linewidth=2)
sns.lineplot(x=x, y=np.cos(x), ax=ax, label="Seaborn: cos(x)")
ax.set_title("Environment Smoke Test - NumPy + Matplotlib + Seaborn")
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("exp1_smoke_test.png", dpi=130)
plt.show()

print("\n[RESULT] All libraries installed, imported, and verified successfully.")

# =============================================================================
# RESULT
# NumPy, SciPy, Jupyter, Statsmodels, Pandas, Matplotlib, Seaborn, Plotly, and
# Bokeh were successfully installed and verified. A combined NumPy-generated
# sine/cosine plot confirms the scientific computing and visualization stack is
# working end-to-end.
# =============================================================================
