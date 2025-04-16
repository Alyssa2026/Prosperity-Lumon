import numpy as np
import itertools
import math
import heapq
import pandas as pd

arr = np.array([[(80,6), (50,4), (83,7), (31,2), (60,4)],
                [(89,8), (10,1), (37,3), (70,4), (90,10)],
                [(17,1), (40,3), (73,4), (100,15),(20,2)],
                [(41,3), (79,5), (23,2), (47,3), (30,2)]])

'''
Compute min percentage to earn profit for second container
'''
thresh = np.zeros((4, 5))
for i in range(4):
    for j in range(5):
        mult, inhab = arr[i, j]
        thresh[i, j] = (mult - 5*inhab)/500
with np.printoptions(precision=3, suppress=True):
    print("Distribution second container")
    print(thresh)
'''
Compute min percentage to earn profit for third container
'''
thresh = np.zeros((4, 5))
for i in range(4):
    for j in range(5):
        mult, inhab = arr[i, j]
        thresh[i, j] = (mult - 10*inhab)/1000
with np.printoptions(precision=3, suppress=True):
    print("Distribution third container")
    print(thresh)

def maximin1():
    """Solves the maximin optimization problem for a single container (first free container).
    We are pessimistic and set p=1
    Returns
    -------
    argmax : list of tuple
        Maximizing (multiplier, contestants) pairs.
    max_val : float
        Corresponding maximal profit.
    """
    max_val = float('-inf')
    argmax = []
    for mult, inhab in arr.reshape((-1, 2)):
        val = 10000*mult / (inhab + 100)   # profit formula for 1st container
        if math.isclose(val, max_val):
            argmax.append((int(mult), int(inhab)))  # 🔁 cast here
        elif val > max_val:
            argmax = [(int(mult), int(inhab))]      # 🔁 and here
            max_val = val
    return argmax, float(max_val)  # ensure float return too
print("optimal one container")
print(maximin1())

def maximize2(grid_steps=50):
    """
    Solves the maximin optimization problem for two expeditions
    using grid search over p1 and p2, under constraint p1 + p2 ≤ 1.
    
    Returns
    -------
    argmax1 : tuple
        First expedition.
    argmax2 : tuple
        Second expedition.
    max_val : float
        Maximal profit.
    """
    max_val = float('-inf')
    max_mult1, max_inhab1 = None, None
    max_mult2, max_inhab2 = None, None

    # Flatten to list of (mult, hunt)
    all_containers = [tuple(map(int, x)) for x in arr.reshape(-1, 2)]

    # Try all unique pairs of destinations
    for (mult1, inhab1), (mult2, inhab2) in itertools.combinations(all_containers, 2):
        best_min = float('inf')

        # Grid search over (p1, p2) such that p1 + p2 <= 1
        for i in range(grid_steps + 1):
            for j in range(grid_steps + 1 - i):
                p1 = i / grid_steps
                p2 = j / grid_steps
                score = (
                    mult1 / (inhab1 + 100 * p1) +
                    mult2 / (inhab2 + 100 * p2)
                )
                if score < best_min:
                    best_min = score

        val = -50_000 + best_min * 10_000  # second container costs 50k

        if np.isclose(val, max_val):
            print("collision")
        if val > max_val:
            max_mult1, max_inhab1 = mult1, inhab1
            max_mult2, max_inhab2 = mult2, inhab2
            max_val = val

    return ((max_mult1, max_inhab1), (max_mult2, max_inhab2), float(max_val))
print("optimal two container")
print(maximize2())
def maximize3(grid_steps=20):
    """
    Solves the maximin optimization problem for three expeditions using grid search.
    
    Returns
    -------
    argmax1 : tuple
        First expedition (mult, hunt).
    argmax2 : tuple
        Second expedition (mult, hunt).
    argmax3 : tuple
        Third expedition (mult, hunt).
    max_val : float
        Maximal profit.
    """
    max_val = float('-inf')
    max_mult1 = max_mult2 = max_mult3 = None
    max_inhab1 = max_inhab2 = max_inhab4 = None

    all_locations = [tuple(map(int, x)) for x in arr.reshape(-1, 2)]

    for (m1, h1), (m2, h2), (m3, h3) in itertools.combinations(all_locations, 3):
        best_min = float('inf')

        # Grid search: p1 + p2 + p3 <= 1
        for i in range(grid_steps + 1):
            for j in range(grid_steps + 1 - i):
                for k in range(grid_steps + 1 - i - j):
                    p1 = i / grid_steps
                    p2 = j / grid_steps
                    p3 = k / grid_steps
                    score = (
                        m1 / (h1 + 100 * p1) +
                        m2 / (h2 + 100 * p2) +
                        m3 / (h3 + 100 * p3)
                    )
                    best_min = min(best_min, score)

        val = -150_000 + best_min * 10_000  # third expedition total cost

        if np.isclose(val, max_val):
            print("collision")
        if val > max_val:
            max_mult1, max_inhab1 = m1, h1
            max_mult2, max_inhab2 = m2, h2
            max_mult3, max_inhab3 = m3, h3
            max_val = val

    return (
        (max_mult1, max_inhab1),
        (max_mult2, max_inhab2),
        (max_mult3, max_inhab1),
        float(max_val)
    )
print("optimal two container")
print(maximize3())

#########################################################################################
# optimization using prior
#########################################################################################
import itertools
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import root_scalar


# New container group: (multiplier, inhabitants)
arr = [(80, 6), (50, 4), (83, 7), (31, 2), (60, 4),
       (89, 8), (10, 1), (37, 3), (70, 4), (90, 10),
       (17, 1), (40, 3), (73, 4), (100, 15), (20, 2),
       (41, 3), (79, 5), (23, 2), (47, 3), (30, 2)]

# Track max percentage of selection to profit a 2nd box 
max_perc = [(m - 5*h)/500 for m, h in arr]

# Compute popularity ratios
ratios = [m / h for (m, h) in arr]

def compute_weighted_shares(exponent):
    powered = [r ** exponent for r in ratios]
    total = sum(powered)
    return [r / total for r in powered]

# Fee schedule: 0 for first box, -50k for second, -100k total for three
def fee(n):
    return 0 if n == 1 else -50000 * (n - 1)

def payoff(selected_mults, inhabs, shares):
    total = 0
    for i in range(len(selected_mults)):
        total += selected_mults[i] / (inhabs[i] + 100 * shares[i])
    return 10000 * total + fee(len(selected_mults))

# Maximize payoff for top-k container picks
def maximize_prior_top(shares, k):
    container_combos = list(itertools.combinations(range(len(arr)), k))
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

# Run for different exponents and k = 1, 2, 3
results = []
for exp in range(0, 11):
    shares = compute_weighted_shares(exp)
    max1, opt1 = maximize_prior_top(shares, 1)
    max2, opt2 = maximize_prior_top(shares, 2)
    max3, opt3 = maximize_prior_top(shares, 3)
    results.append((exp, max1, opt1, max2, opt2, max3, opt3))

# Generate payoff curves for all container combinations (1, 2, or 3 boxes)
combos = list(itertools.chain(
    itertools.combinations(range(len(arr)), 1),
    itertools.combinations(range(len(arr)), 2),
    itertools.combinations(range(len(arr)), 3)
))

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

# # Create Plotly figure
# fig = go.Figure()

# for combo, y_vals in combo_payoff_by_exp.items():
#     label = f"Containers {tuple(i+1 for i in combo)}"
#     fig.add_trace(go.Scatter(
#         x=exponents,
#         y=y_vals,
#         mode='lines',
#         name=label,
#         hovertemplate=f"{label}<br>Exponent: %{{x}}<br>Payoff: %{{y:.2f}}<extra></extra>"
#     ))

# fig.update_layout(
#     title="Payoff vs Popularity Exponent for 1, 2, or 3 Container Picks",
#     xaxis_title="Exponent (Ratio^exponent)",
#     yaxis_title="Payoff",
#     hovermode="closest",
#     width=1100,
#     height=700
# )

# annotations = []
# for i, (m, h) in enumerate(arr):
#     annotations.append(
#         dict(
#             xref="paper", yref="paper",
#             x=1.01, y=1 - i * 0.045,
#             showarrow=False,
#             text=f"{i+1}: mult={m}, inhab={h}",
#             font=dict(size=12),
#             align="left"
#         )
#     )

# fig.update_layout(
#     annotations=annotations,
#     margin=dict(r=200)  # give space for annotations
# )
# fig.show()

# only display 2 box selection
from plotly.subplots import make_subplots

# Only include 1-box and 2-box combos
combos_1_and_2 = list(itertools.chain(
    itertools.combinations(range(len(arr)), 1),
    itertools.combinations(range(len(arr)), 2)
))

combo_payoff_by_exp_filtered = {combo: combo_payoff_by_exp[combo] for combo in combos_1_and_2}

fig = make_subplots(
    rows=2, cols=1,
    row_heights=[0.3, 0.7],
    vertical_spacing=0.1,
    specs=[[{"type": "table"}],
           [{"type": "scatter"}]]
)

fig.add_trace(go.Table(
    header=dict(values=["Container #", "Multiplier", "Inhabitants"],
                fill_color='paleturquoise',
                align='left'),
    cells=dict(
        values=[
            [i + 1 for i in range(len(arr))],
            [m for m, h in arr],
            [h for m, h in arr]
        ],
        fill_color='lavender',
        align='left'
    )
), row=1, col=1)

for combo, y_vals in combo_payoff_by_exp_filtered.items():
    label = f"Containers {tuple(i+1 for i in combo)}"
    fig.add_trace(go.Scatter(
        x=exponents,
        y=y_vals,
        mode='lines',
        name=label,
        hovertemplate=f"{label}<br>Exponent: %{{x}}<br>Payoff: %{{y:.2f}}<extra></extra>"
    ), row=2, col=1)

fig.update_layout(
    title="Payoff vs Popularity Exponent (Only 1 and 2 Container Choices)",
    xaxis_title="Exponent (Ratio^exponent)",
    yaxis_title="Payoff",
    hovermode="closest",
    height=900,
    width=1100
)

fig.show()
