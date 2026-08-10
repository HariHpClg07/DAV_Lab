# =============================================================================
# Experiment 2A: WORKING WITH NUMPY ARRAYS - Operations and Array Manipulations
# -----------------------------------------------------------------------------
# AIM: Understand and implement NumPy array creation, indexing, slicing,
#      element-wise operations, aggregations, boolean masking, fancy indexing,
#      reshaping, and structured arrays - each backed by a visual.
# RUN AS: A single Google Colab cell.
# =============================================================================

!pip install -q numpy matplotlib seaborn

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("NumPy Version:", np.__version__)

# --- Creating different types of arrays ---
arr_1d = np.array([1, 2, 3, 4, 5])
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
arr_0d = np.array(42)
arr_ones = np.ones((3, 3))

# --- Indexing and Slicing ---
print("Element at index 2 in 1D array:", arr_1d[2])
print("Element at row 1, column 2 in 2D array:", arr_2d[1, 2])
print("Slice from 1D array:", arr_1d[1:4])
print("Slice row 1 from 2D array:", arr_2d[1, :])

# --- Element-wise operations ---
arr_a = np.array([10, 20, 30])
arr_b = np.array([1, 2, 3])
print("Addition:", arr_a + arr_b)
print("Subtraction:", arr_a - arr_b)
print("Multiplication:", arr_a * arr_b)
print("Division:", arr_a / arr_b)
print("Scalar Multiplication:", arr_a * 2)

# --- Aggregations ---
print("Sum:", np.sum(arr_a))
print("Mean:", np.mean(arr_a))
print("Standard Deviation:", np.std(arr_a))

# --- Boolean masking & fancy indexing ---
print("Elements greater than 15:", arr_a[arr_a > 15])
indices = [0, 2]
print("Selected elements (fancy indexing):", arr_a[indices])

# --- Reshape ---
reshaped_arr = arr_1d.reshape(5, 1)
print("Reshaped 1D array to 2D:\n", reshaped_arr)

# --- Structured array ---
structured_arr = np.array([(25, 90.5), (30, 85.2), (22, 76.4), (28, 88.9)],
                           dtype=[('age', 'i4'), ('score', 'f4')])
print("Structured array:", structured_arr)

# =============================== VISUALIZATION ===============================
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

# 1. Bar chart: element-wise arithmetic results
ops = ["Addition", "Subtraction", "Multiplication", "Division"]
results = [arr_a + arr_b, arr_a - arr_b, arr_a * arr_b, arr_a / arr_b]
x = np.arange(3)
width = 0.2
for i, (op, res) in enumerate(zip(ops, results)):
    axes[0, 0].bar(x + i * width, res, width, label=op)
axes[0, 0].set_title("Element-wise Arithmetic on arr_a & arr_b")
axes[0, 0].set_xticks(x + 1.5 * width)
axes[0, 0].set_xticklabels(["idx0", "idx1", "idx2"])
axes[0, 0].legend(fontsize=8)

# 2. Boolean mask highlighted on the array
colors = ["#4C72B0" if v > 15 else "#DD8452" for v in arr_a]
axes[0, 1].bar(range(len(arr_a)), arr_a, color=colors)
axes[0, 1].axhline(15, color="red", linestyle="--", label="threshold = 15")
axes[0, 1].set_title("Boolean Masking: arr_a > 15 (highlighted)")
axes[0, 1].legend()

# 3. Heatmap of the ones matrix + 2D array
sns.heatmap(arr_2d, annot=True, fmt="d", cmap="Blues", ax=axes[1, 0], cbar=False)
axes[1, 0].set_title("2D Array Heatmap")

# 4. Structured array scatter: age vs score
axes[1, 1].scatter(structured_arr['age'], structured_arr['score'],
                    s=100, color="#55A868")
axes[1, 1].set_title("Structured Array: Age vs Score")
axes[1, 1].set_xlabel("age")
axes[1, 1].set_ylabel("score")
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("exp2a_numpy_operations.png", dpi=130)
plt.show()

print("\n[RESULT] NumPy array operations demonstrated and visualized successfully.")

# =============================================================================
# RESULT
# NumPy array creation, indexing, slicing, arithmetic, aggregation, boolean
# masking, fancy indexing, reshaping, and structured arrays were successfully
# demonstrated, with each concept backed by a corresponding visualization.
# =============================================================================
