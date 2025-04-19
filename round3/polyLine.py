import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from statistics import NormalDist
from scipy.optimize import brentq
from scipy.stats import zscore

# Black‑Scholes call option pricing
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
EXPIRY_DAY    = 8
CURRENT_ROUND = 0

# Define your strikes
voucher_strikes = {
    "VOLCANIC_ROCK_VOUCHER_9500": 9500,
    "VOLCANIC_ROCK_VOUCHER_9750": 9750,
    "VOLCANIC_ROCK_VOUCHER_10000": 10000,
    "VOLCANIC_ROCK_VOUCHER_10250": 10250,
    "VOLCANIC_ROCK_VOUCHER_10500": 10500,
}

# Load & shift all three days
df_list = []
for day in (0, 1, 2):
    path = f"./round3/data/prices_round_3_day_{day}.csv"
    day_df = pd.read_csv(path, sep=";")
    day_df["timestamp"] += day * 100_000
    df_list.append(day_df)
df = pd.concat(df_list, ignore_index=True)

# Collect (m, iv) pairs
m_iv = []
for ts in df["timestamp"].unique():
    tick = df[df["timestamp"] == ts]

    # Recompute spot from VOLCANIC_ROCK bids/asks:
    rock = tick[tick["product"] == "VOLCANIC_ROCK"]
    if rock.empty:
        continue
    rock_row = rock.iloc[0]
    spot_bid = rock_row["bid_price_1"]
    spot_ask = rock_row["ask_price_1"]
    spot = (spot_bid + spot_ask) / 2

    # Compute time‑to‑expiry as before…
    fractional = CURRENT_ROUND + (ts / 1_000_000)  # adjust units if needed
    TTE = max(0.0, EXPIRY_DAY - fractional) / 365
    if TTE <= 0:
        continue

    # For each voucher, do the same:
    for sym, K in voucher_strikes.items():
        row = tick[tick["product"] == sym]
        if row.empty:
            continue
        vrow = row.iloc[0]
        bid = vrow["bid_price_1"]
        ask = vrow["ask_price_1"]
        mid = (bid + ask) / 2

        intrinsic = max(0.0, spot - K)
        if mid <= intrinsic + 1e-8:
            continue
    
        m = math.log(K / spot) / math.sqrt(TTE)
        iv = implied_volatility(spot, K, TTE, mid)
        if not math.isnan(iv) and not math.isinf(iv):
            m_iv.append((m, iv))

m_iv = np.array(m_iv)
ivs = m_iv[:, 1]

# Fit parabola to combined data
ms = m_iv[:, 0]
ivs = m_iv[:, 1]
a, b, c = np.polyfit(ms, ivs, 2)
print(f"Combined fit coefficients: A={a:.6f}\nB={b:.6f}\nC={c:.6f}")

# Plot
m_grid = np.linspace(ms.min(), ms.max(), 300)
iv_fit = a*m_grid**2 + b*m_grid + c

plt.figure(figsize=(10,5))
plt.scatter(ms, ivs, alpha=0.5, label="Cleaned IV points")
plt.plot(m_grid, iv_fit, "--", label="Fitted parabola", linewidth=2)
plt.xlabel("m = log(K/S) / sqrt(T)")
plt.ylabel("Implied Volatility")
plt.title("Combined IV vs m (Days 0–2, outliers removed)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
