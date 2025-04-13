import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Define the symbolic variable
x = sp.Symbol('x', real=True)

# Total density factor for the uniform distribution over a total range of 110.
density = 1/110

# Define expected profit functions for the two allowed regions
# Region 1: bidOne in [160, 200]
E1 = (320 - x) * (x - 160) * density

# Region 2: bidOne in [250, 320]
E2 = (320 - x) * (x - 250) * density

# Compute derivatives for each region
dE1 = sp.diff(E1, x)
dE2 = sp.diff(E2, x)

# Find critical points for each region and filter those within the interval
crit_points_E1 = [pt for pt in sp.solve(dE1, x) if pt >= 160 and pt <= 200]
crit_points_E2 = [pt for pt in sp.solve(dE2, x) if pt >= 250 and pt <= 320]
# print(crit_points_E1, crit_points_E2)
# Prepare candidate bids: include boundaries and interior critical points
candidates = [160, 200, 250, 320]
candidates.extend(crit_points_E1)
candidates.extend(crit_points_E2)

# Evaluate the expected profit for each candidate
results = []
for candidate in candidates:
    candidate = sp.N(candidate)  # numerical value
    if 160 <= candidate <= 200:
        profit_val = E1.subs(x, candidate)
    elif 250 <= candidate <= 320:
        profit_val = E2.subs(x, candidate)
    results.append((candidate, sp.N(profit_val)))

# Print out each candidate bid and its expected profit
print("Candidate bids and their expected profit:")
for bid_val, profit_val in results:
    print(f"  BidOne = {bid_val:.2f}, Expected Profit = {profit_val:.2f}")

# Find the candidate with the maximum expected profit
optimal_bid, max_profit = max(results, key=lambda item: item[1])

print("\nOptimal BidOne:")
print(f"  BidOne = {optimal_bid:.2f} SeaShells, yielding an expected profit of {max_profit:.2f} SeaShells")

# Now, create a graph of the expected profit functions across the two regions.
# Generate data for region [160, 200]
x_vals_E1 = np.linspace(160, 200, 300)
E1_func = sp.lambdify(x, E1, 'numpy')
y_vals_E1 = E1_func(x_vals_E1)

# Generate data for region [250, 320]
x_vals_E2 = np.linspace(250, 320, 300)
E2_func = sp.lambdify(x, E2, 'numpy')
y_vals_E2 = E2_func(x_vals_E2)

plt.figure(figsize=(10, 6))
plt.plot(x_vals_E1, y_vals_E1, label='Expected Profit for Region [160, 200]')
plt.plot(x_vals_E2, y_vals_E2, label='Expected Profit for Region [250, 320]')

# Mark the optimal candidate with a red dot.
plt.scatter([optimal_bid], [max_profit], color='red', zorder=5, 
            label=f'Optimal Bid ({optimal_bid:.2f})')

plt.xlabel('bidOne (SeaShells)')
plt.ylabel('Expected Profit (SeaShells)')
plt.title('Expected Profit vs. bidOne')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
