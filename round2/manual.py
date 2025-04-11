import itertools
import heapq

from matplotlib import pyplot as plt
import numpy as np


# Containers: (treasure multiplier, number of inhabitants)
arr = [(73, 4), (10, 1), (80, 6), (37, 3), (17, 1), (31, 2),
       (90, 10), (50, 4), (20, 2), (89, 8)]

# Assume popularity depends on raw ratios (multiplier / inhabitants) 
ratios = [mult / hunt for (mult, hunt) in arr]
print("="*50)
print("ratios")
print(ratios)

# max_perc = []
# for m, h in arr:
#     print(m,h)

#     max_perc.append((m-5*h)/500)

# print(max_perc)


# Elevate ratios to a power to play with different strength ratio
def compute_weighted_shares(exponent):
    powered = [r ** exponent for r in ratios]
    total = sum(powered)
    return [r / total for r in powered]

# Fee to open container
def fee(n):
    if n == 1:
        return 0
    if n == 2:
        return -50000

# Payoff calculation
def payoff(selected_mults, hunts, shares):
    total = 0
    # between 1 and 2 continers 
    for i in range(len(selected_mults)):
        total += selected_mults[i] / (hunts[i] + 100 * shares[i])
    return 10000 * total + fee(len(selected_mults))

# Maximize payoff given shares for selecting top k containers
def maximize_prior_top(shares, k):
    # Generate all possible combinations of 1-2 containers
    container_combos = list(itertools.combinations(range(10), k))

    max_payoff = float('-inf')
    optimal_combination = None
    
    for containers in container_combos:
        selected_mults = [arr[container][0] for container in containers]
        selected_hunts = [arr[container][1] for container in containers]
        selected_shares = [shares[container] for container in containers]
        current_payoff = payoff(selected_mults, selected_hunts, selected_shares)

        if current_payoff > max_payoff:
            max_payoff = current_payoff
            optimal_combination = containers
    
    return max_payoff, optimal_combination

# Run for different exponents
for exp in range(0,10):  # Using exponent 0 to 1 as an example    
    shares = compute_weighted_shares(exp)
    
    print("\n" + "=" * 60)
    print(f"Exponent: {exp}")
    print("-" * 60)
    print("Shares:")
    for i, share in enumerate(shares):
        print(f"  Container {i + 1}: {share:.4f}")
    
    # Maximize for 1 container
    max_payoff_1, optimal_1 = maximize_prior_top( shares, 1)
    print("\n>> Best Single Container Choice:")
    print(f"  Container Index: {optimal_1}")
    print(f"  Max Payoff     : {max_payoff_1}")
    
    # Maximize for 2 containers
    max_payoff_2, optimal_2 = maximize_prior_top(shares, 2)
    print("\n>> Best Pair of Containers Choice:")
    print(f"  Container Indices: {optimal_2}")
    print(f"  Max Payoff       : {max_payoff_2}")
    
    # Final best choice
    best_payoff = max(max_payoff_1, max_payoff_2)
    best_choice = optimal_1 if max_payoff_1 > max_payoff_2 else optimal_2
    print("\n>> Best Overall Choice:")
    print(f"  Containers: {best_choice}")
    print(f"  Max Payoff     : {best_payoff}")



# import numpy as np
# from scipy.optimize import root_scalar

# Container data: (multiplier, inhabitants)
# data = np.array([
#     (10, 1), (80, 6), (37, 3), (17, 1), (31, 2),
#     (90, 10), (50, 4), (20, 2), (73, 4), (89, 8)
# ])

# M = data[:, 0]
# H = data[:, 1]

# # Function to compute total share sum for a given K
# def share_sum(K):
#     shares = (M / K - H) / 100
#     return np.sum(shares) - 1

# # Find K such that sum of shares = 1
# result = root_scalar(share_sum, bracket=[0.01, 10000], method='brentq')

# if result.converged:
#     K = result.root
#     shares = (M / K - H) / 100
#     actual_payoff = 10000 * K

#     print(f"\nEqualized Expected Payoff Multiplier (K): {K:.4f}")
#     print(f"Final Actual Payoff from any container: {actual_payoff:.2f}\n")

#     print("Container-wise Shares and Payoff:")
#     for i in range(len(M)):
#         payoff = 10000 * M[i] / (H[i] + 100 * shares[i])
#         print(f"  Container {i+1}: Share = {shares[i]:.4f}, "
#               f"Payoff = {payoff:.2f}")
# else:
#     print("Failed to converge to a solution.")
