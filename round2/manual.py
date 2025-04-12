import itertools
import heapq

from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import root_scalar
import plotly.graph_objects as go

# Containers: (treasure multiplier, number of inhabitants)
arr = [(73, 4), (10, 1), (80, 6), (37, 3), (17, 1), (31, 2),
       (90, 10), (50, 4), (20, 2), (89, 8)]

# track max percentage of selection to profit a 2nd box 
max_perc = []
for m, h in arr:
    max_perc.append((m-5*h)/500)

print("max percentage of selection to profit a 2nd box ")
print(max_perc)
    
# Assume popularity depends on raw ratios (multiplier / inhabitants) 
ratios = [mult / inhab for (mult, inhab) in arr]
print("="*50)
print("ratios")
print(ratios)

# Elevate ratios to a power to play with different strength ratio
def compute_weighted_shares(exponent):
    powered = [r ** exponent for r in ratios]
    total = sum(powered)
    return [r / total for r in powered]

# Fee to open container
def fee(n):
    return 0 if n == 1 else -50000

# Payoff calculation
def payoff(selected_mults, inhabs, shares):
    total = 0
    for i in range(len(selected_mults)):
        total += selected_mults[i] / (inhabs[i] + 100 * shares[i])
    return 10000 * total + fee(len(selected_mults))

# Maximize payoff given shares for selecting top k containers
def maximize_prior_top(shares, k):
    container_combos = list(itertools.combinations(range(10), k))
    max_payoff = float('-inf')
    optimal_combination = None
    
    for containers in container_combos:
        selected_mults = [arr[container][0] for container in containers]
        selected_inhab = [arr[container][1] for container in containers]
        selected_shares = [shares[container] for container in containers]
        current_payoff = payoff(selected_mults, selected_inhab, selected_shares)

        if current_payoff > max_payoff:
            max_payoff = current_payoff
            optimal_combination = containers
    
    return max_payoff, optimal_combination

# Run for different exponents
for exp in range(0, 10):    
    shares = compute_weighted_shares(exp)
    
    print("\n" + "=" * 60)
    print(f"Exponent: {exp}")
    print("-" * 60)
    print("Shares:")
    for i, share in enumerate(shares):
        print(f"  Container {i + 1}: Share = {share:.4f}")
    
    # Best single container
    max_payoff_1, optimal_1 = maximize_prior_top(shares, 1)
    container_1 = optimal_1[0]
    print("\n>> Best Single Container Choice:")
    print(f"  Container Box: {container_1 + 1}")
    print(f"  Container Data: {arr[container_1]}")
    print(f"  Max Payoff    : {max_payoff_1}")
    
    # Best pair of containers
    max_payoff_2, optimal_2 = maximize_prior_top(shares, 2)
    print("\n>> Best Pair of Containers Choice:")
    print(f"  Container Boxes: {[i + 1 for i in optimal_2]}")
    print(f"  Container Data : {[arr[i] for i in optimal_2]}")
    print(f"  Max Payoff     : {max_payoff_2}")
    
    # Best overall
    best_payoff = max(max_payoff_1, max_payoff_2)
    best_choice = optimal_1 if max_payoff_1 > max_payoff_2 else optimal_2
    print("\n>> Best Overall Choice:")
    print(f"  Container Boxes: {[i + 1 for i in best_choice]}")
    print(f"  Container Data : {[arr[i] for i in best_choice]}")
    print(f"  Max Payoff     : {best_payoff}")

# Find the expected value     
data = np.array([
    (73, 4), (10, 1), (80, 6), (37, 3), (17, 1), (31, 2),
    (90, 10), (50, 4), (20, 2) , (89, 8)
])
M = data[:, 0]
H = data[:, 1]

def share_sum(K):
    shares = (M / K - H) / 100
    return np.sum(shares) - 1

result = root_scalar(share_sum, bracket=[0.01, 10000], method='brentq')

if result.converged:
    K = result.root
    shares = (M / K - H) / 100
    actual_payoff = 10000 * K

    print(f"\nEqualized Expected Payoff Multiplier (K): {K:.4f}")
    print(f"Final Actual Payoff from any container: {actual_payoff:.2f}\n")

    print("Container-wise Shares and Payoff:")
    for i in range(len(M)):
        payoff_val = 10000 * M[i] / (H[i] + 100 * shares[i])
        print(f"  Container {i+1}: Share = {shares[i]:.4f}, "
              f"Payoff = {payoff_val:.2f}")
else:
    print("Failed to converge to a solution.")

# ==================== INTERACTIVE PLOT ==================== #

import plotly.graph_objects as go

# Container combinations (1 or 2)
combos = list(itertools.chain(
    itertools.combinations(range(10), 1),
    itertools.combinations(range(10), 2)
))

# Store payoff curves
combo_payoff_by_exp = {combo: [] for combo in combos}
exponents = list(range(0, 11))

for exp in exponents:
    shares = compute_weighted_shares(exp)
    for combo in combos:
        mults = [arr[i][0] for i in combo]
        inhabs = [arr[i][1] for i in combo]
        share_vals = [shares[i] for i in combo]
        pf = payoff(mults, inhabs, share_vals)
        combo_payoff_by_exp[combo].append(pf)

# Create Plotly figure
fig = go.Figure()

for combo, y_vals in combo_payoff_by_exp.items():
    label = f"Containers {tuple(i+1 for i in combo)}"
    fig.add_trace(go.Scatter(
        x=exponents,
        y=y_vals,
        mode='lines',
        name=label,
        hovertemplate=f"{label}<br>Exponent: %{{x}}<br>Payoff: %{{y:.2f}}<extra></extra>"
    ))

fig.update_layout(
    title="Container Combo Payoff vs Exponent",
    xaxis_title="Exponent (Ratio^exponent)",
    yaxis_title="Payoff",
    hovermode="closest",
    width=1000,
    height=700
)

fig.show()
