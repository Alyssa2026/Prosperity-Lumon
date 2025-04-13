import math
import numpy as np

def F(x):
    """Cumulative probability that a turtle's reserve price is ≤ x.
    The reserve price distribution is uniform on [160,200] ∪ [250,320]."""
    if x < 160:
        return 0
    elif x < 200:
        return (x - 160) / 110  # x covers a portion of the lower interval.
    elif x < 250:
        # Once x is between 200 and 250, only the lower interval counts.
        return 40 / 110
    elif x <= 320:
        # For x in [250,320], add full lower interval (length 40) and a fraction of the upper interval.
        return (40 + (x - 250)) / 110
    else:
        return 1

def expected_profit(l, h, p_avg):
    """Compute the expected profit when first bid is l and second bid is h,
    given the average second bid p_avg.
    
    Parameters:
      l: first bid (must be between 160 and 320)
      h: second bid (l <= h <= 320)
      p_avg: the average second bid submitted by other traders.
    
    Returns:
      Expected profit.
    """
    # Profit if trading occurs at the first bid.
    first_term = (320 - l) * F(l)
    
    # Profit if trading occurs at the second bid.
    # Only turtles with reserve in (l, h] are in consideration.
    second_term = (320 - h) * (F(h) - F(l))
    
    # Adjust the profit from the second bid by the probability factor.
    if h >= p_avg:
        factor = 1
    else:
        factor = ((320 - p_avg) / (320 - h)) ** 3
        
    return first_term + second_term * factor

def solve(p_avg):
    """Searches for the pair of bids (l, h) that maximizes expected profit.
    
    Parameters:
      p_avg: Average second bid from other traders.
    
    Returns:
      argmax: A list of tuples (l, h, profit) that maximize expected profit.
      val_max: The maximum expected profit.
    """
    val_max = -float('inf')
    argmax = []
    
    # Search over possible bids in the range [160, 320].
    for l in range(160, 321):
        for h in range(l, 321):
            profit = expected_profit(l, h, p_avg)
            # Check if profit is (nearly) equal to current maximum.
            if math.isclose(profit, val_max, rel_tol=1e-9):
                argmax.append((l, h, profit))
            # If a new maximum is found, update.
            if profit > val_max:
                val_max = profit
                argmax = [(l, h, profit)]
    return argmax, val_max

# Example: Assess optimal bids for various values of p_avg.
# Here we assume p_avg (the average second bid) might vary near the top of the range.
for p_avg in np.arange(160, 319.5, 0.5):
    maximizers, max_profit = solve(p_avg)
    if len(maximizers) > 1:
        print("p_avg:", p_avg, "  Maximizers:", maximizers, " Profit: {:.2f}".format(max_profit))
    else:
        print("p_avg:", p_avg, "  Maximizer:", maximizers[0], " Profit: {:.2f}".format(max_profit))
