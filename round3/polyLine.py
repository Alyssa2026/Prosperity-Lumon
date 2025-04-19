import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from statistics import NormalDist
from scipy.optimize import brentq
from scipy.stats import zscore

# Load the CSV
csv_path = "./data/prices_round_3_day_0.csv"
df = pd.read_csv(csv_path, sep=";")

# Define the strikes
voucher_strikes = {
    "VOLCANIC_ROCK_VOUCHER_9500": 9500,
    "VOLCANIC_ROCK_VOUCHER_9750": 9750,
    "VOLCANIC_ROCK_VOUCHER_10000": 10000,
    "VOLCANIC_ROCK_VOUCHER_10250": 10250,
    "VOLCANIC_ROCK_VOUCHER_10500": 10500,
}

# Black-Scholes call option pricing
def bs_call_price(S, K, T, sigma, r=0.0):
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return float("nan")
    d1 = (math.log(S / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * NormalDist().cdf(d1) - K * math.exp(-r * T) * NormalDist().cdf(d2)

# Implied volatility calculation
def implied_volatility(S, K, T, market_price):
    try:
        return brentq(lambda sigma: bs_call_price(S, K, T, sigma) - market_price, 1e-5, 5)
    except ValueError:
        return float("nan")

# Parameters
EXPIRY_DAY = 8
CURRENT_ROUND = 0

# Collect m and IV pairs
m_iv_pairs = []

for timestamp in df["timestamp"].unique():
    tick_df = df[df["timestamp"] == timestamp]
    rock_row = tick_df[tick_df["product"] == "VOLCANIC_ROCK"]
    if rock_row.empty:
        continue
    spot = rock_row["mid_price"].values[0]
    fractional_day = CURRENT_ROUND + (timestamp / 1_000_000)
    days_to_expiry = max(0.0, EXPIRY_DAY - fractional_day)
    TTE = days_to_expiry / 365
    if TTE <= 0:
        continue

    for symbol, strike in voucher_strikes.items():
        row = tick_df[tick_df["product"] == symbol]
        if row.empty:
            continue
        mid_price = row["mid_price"].values[0]
        m = math.log(strike / spot) / math.sqrt(TTE)
        iv = implied_volatility(spot, strike, TTE, mid_price)
        if not math.isnan(iv) and not math.isinf(iv):
            m_iv_pairs.append((m, iv))

# Convert to array
m_iv_array = np.array(m_iv_pairs)

# Unpack final cleaned data
ms = m_iv_array[:, 0]
ivs = m_iv_array[:, 1]

# Fit parabola
a, b, c = np.polyfit(ms, ivs, 2)

# Print coefficients
print(f"\nFitted parabola coefficients after filtering:")
print(f"a = {a:.6f}")
print(f"b = {b:.6f}")
print(f"c = {c:.6f}")

# Plot
m_vals = np.linspace(min(ms), max(ms), 200)
ivs_fit = a * m_vals**2 + b * m_vals + c

plt.figure(figsize=(10, 5))
plt.scatter(ms, ivs, alpha=0.6, label="Filtered IV")
plt.plot(m_vals, ivs_fit, color="orange", linestyle="--", label="Fitted IV Curve")
plt.xlabel("m = log(K/S) / sqrt(T)")
plt.ylabel("Implied Volatility")
plt.title("Implied Volatility vs m (Outliers Removed)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
