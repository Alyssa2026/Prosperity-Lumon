import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Symbols for symbolic computation
b1, b2 = sp.symbols('b1 b2', real=True)

# Uniform density over total valid domain [160–200] ∪ [250–320] => range = 40 + 70 = 110
density = 1 / 110

# For [160, 200]
E1 = ((b1 - 160) * (320 - b1) + (b2 - b1) * (320 - b2)) * density

# For [250, 320]
E2 = ((b1 - 250) * (320 - b1) + (b2 - b1) * (320 - b2)) * density

# Discretize search for bid pairs
def evaluate_region(region_start, region_end, expr):
    results = []
    for b1_val in np.linspace(region_start, region_end - 1, 50):
        for b2_val in np.linspace(b1_val + 1, region_end, 50):
            profit = expr.subs({b1: b1_val, b2: b2_val})
            results.append(((b1_val, b2_val), float(profit)))
    return results

# Evaluate both regions
results_region1 = evaluate_region(160, 200, E1)
results_region2 = evaluate_region(250, 320, E2)

# Combine results and find maximum
all_results = results_region1 + results_region2
optimal_pair, max_profit = max(all_results, key=lambda item: item[1])

print("Optimal Bid Pair (Two-Bid Strategy):")
print(f"  BidOne = {optimal_pair[0]:.2f}, BidTwo = {optimal_pair[1]:.2f}")
print(f"  Expected Profit = {max_profit:.2f} SeaShells")

# Visualization for one region (optional)
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')

r1_vals = np.linspace(250, 319, 40)
r2_vals = np.linspace(251, 320, 40)
R1, R2 = np.meshgrid(r1_vals, r2_vals)
Z = np.zeros_like(R1)

for i in range(R1.shape[0]):
    for j in range(R1.shape[1]):
        if R2[i, j] > R1[i, j]:
            Z[i, j] = float(E2.subs({b1: R1[i, j], b2: R2[i, j]}))
        else:
            Z[i, j] = np.nan

ax.plot_surface(R1, R2, Z, cmap='plasma')
ax.set_xlabel('BidOne (b1)')
ax.set_ylabel('BidTwo (b2)')
ax.set_zlabel('Expected Profit')
ax.set_title('Expected Profit Surface for Region [250, 320]')
plt.tight_layout()
plt.show()
