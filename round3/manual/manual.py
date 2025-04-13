import numpy as np
import matplotlib.pyplot as plt

# Density of the segmented uniform distribution
density = 1 / 110

# Compute cumulative probability from the custom CDF
def cumulative_prob(lower, upper, b1, b2):
    # Probabilities in each segment
    prob = 0.0
    if b1 < upper and b2 > lower:
        clipped_b1 = max(b1, lower)
        clipped_b2 = min(b2, upper)
        prob = max(0, clipped_b2 - clipped_b1)
    return prob * density

# Expected profit function
def expected_profit(b1, b2):
    if b2 <= b1:
        return 0

    # Part 1: reserve <= b1
    p1 = cumulative_prob(160, 200, 160, b1) + cumulative_prob(250, 320, 250, b1)
    profit1 = (320 - b1) * p1

    # Part 2: b1 < reserve <= b2
    p2 = cumulative_prob(160, 200, b1, b2) + cumulative_prob(250, 320, b1, b2)
    profit2 = (320 - b2) * p2

    return profit1 + profit2

# Create grid of bid pairs
b1_vals = np.linspace(160, 319, 80)
b2_vals = np.linspace(161, 320, 80)
B1, B2 = np.meshgrid(b1_vals, b2_vals)
Z = np.zeros_like(B1)

# Compute expected profits over the grid
for i in range(len(b1_vals)):
    for j in range(len(b2_vals)):
        b1 = B1[j, i]
        b2 = B2[j, i]
        Z[j, i] = expected_profit(b1, b2)

# Find optimal bid pair
max_idx = np.unravel_index(np.argmax(Z), Z.shape)
optimal_b1 = B1[max_idx]
optimal_b2 = B2[max_idx]
max_profit = Z[max_idx]

print(f"Optimal BidOne = {optimal_b1:.2f}, BidTwo = {optimal_b2:.2f}, Expected Profit = {max_profit:.2f}")

# Plotting
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(B1, B2, Z, cmap='viridis')
ax.set_xlabel('BidOne (b1)')
ax.set_ylabel('BidTwo (b2)')
ax.set_zlabel('Expected Profit')
ax.set_title('Expected Profit vs. (BidOne, BidTwo)')
plt.tight_layout()
plt.show()
