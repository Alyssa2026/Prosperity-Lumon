import numpy as np

# Constants
density = 1 / 110

def cumulative_prob(lower, upper, b1, b2):
    if b1 < upper and b2 > lower:
        clipped_b1 = max(b1, lower)
        clipped_b2 = min(b2, upper)
        return max(0, clipped_b2 - clipped_b1) * density
    return 0.0

def expected_profit(b1, b2, avg_b2):
    if b2 <= b1:
        return 0
    p1 = cumulative_prob(160, 200, 160, b1) + cumulative_prob(250, 320, 250, b1)
    p2 = cumulative_prob(160, 200, b1, b2) + cumulative_prob(250, 320, b1, b2)
    scale = 1.0 if b2 >= avg_b2 else ((320 - avg_b2) / max(1e-8, 320 - b2)) ** 3
    return (320 - b1) * p1 + scale * (320 - b2) * p2

# Grid of values to search
b1_vals = np.arange(160, 320, 2)
b2_vals = np.arange(161, 321, 2)

def solve(avg_b2):
    max_profit = float('-inf')
    maximizers = []

    for b1 in b1_vals:
        for b2 in b2_vals:
            if b2 <= b1:
                continue
            profit = expected_profit(b1, b2, avg_b2)
            if np.isclose(profit, max_profit, rtol=1e-5):
                maximizers.append((b1, b2, profit))
            elif profit > max_profit:
                max_profit = profit
                maximizers = [(b1, b2, profit)]

    return maximizers, max_profit

# Sweep over average bid values
for avg_b2 in np.arange(270, 310.5, 0.5):
    maximizers, max_profit = solve(avg_b2)
    if len(maximizers) > 1:
        print(f"avg_b2: {avg_b2:.1f}  Maximizers:", maximizers, f" Profit: {max_profit:.2f}")
    else:
        b1, b2, profit = maximizers[0]
        print(f"avg_b2: {avg_b2:.1f}  Maximizer: (b1={b1}, b2={b2})  Profit: {profit:.2f}")
