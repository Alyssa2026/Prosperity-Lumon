import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from statistics import NormalDist
from scipy.optimize import brentq

# List the CSV files for day_0, day_1, and day_2
csv_files = [
    "data/prices_round_3_day_0.csv",
    "data/prices_round_3_day_1.csv",
    "data/prices_round_3_day_2.csv"
]

# Read each file and assign a day offset based on the file name
df_list = []
for file in csv_files:
    # Extract the day number from the file name (assumes format "..._day_X.csv")
    day_str = file.split("_")[-1].split(".")[0]  # e.g., "0" from "day_0.csv"
    day_offset = int(day_str)
    temp_df = pd.read_csv(file, sep=";")
    temp_df["day_offset"] = day_offset
    df_list.append(temp_df)

# Combine all DataFrames
df = pd.concat(df_list, ignore_index=True)

# Define the strikes for each voucher
voucher_strikes = {
    "VOLCANIC_ROCK_VOUCHER_9500": 9500,
    "VOLCANIC_ROCK_VOUCHER_9750": 9750,
    "VOLCANIC_ROCK_VOUCHER_10000": 10000,
    "VOLCANIC_ROCK_VOUCHER_10250": 10250,
    "VOLCANIC_ROCK_VOUCHER_10500": 10500,
}

# Black-Scholes call option pricing function
def bs_call_price(S, K, T, sigma, r=0.0):
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return float("nan")
    d1 = (math.log(S / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * NormalDist().cdf(d1) - K * math.exp(-r * T) * NormalDist().cdf(d2)

# Implied volatility calculation using Brent's method
def implied_volatility(S, K, T, market_price):
    try:
        return brentq(lambda sigma: bs_call_price(S, K, T, sigma) - market_price, 1e-5, 5)
    except ValueError:
        return float("nan")

# Parameters
EXPIRY_DAY = 8   # Total days to expiry
CURRENT_ROUND = 0  # Base round offset (could be adjusted if needed)

# Create a dictionary to store (m, iv) points for each voucher
voucher_data = {voucher: [] for voucher in voucher_strikes}

# Group the data by both timestamp and day_offset. Each group now corresponds to a 
# unique moment in time (within that day) and a particular day.
for (ts, day_offset), tick_df in df.groupby(["timestamp", "day_offset"]):
    # Compute the fractional day by adding day_offset to the timestamp fraction
    fractional_day = day_offset + (ts / 1_000_000)  # Adjust the divisor as needed
    days_to_expiry = max(0.0, EXPIRY_DAY - fractional_day)
    TTE = days_to_expiry / 365  # Time to expiry in years
    
    if TTE <= 0:
        continue

    # Get the spot (price) for VOLCANIC_ROCK in this timestamp
    rock_row = tick_df[tick_df["product"] == "VOLCANIC_ROCK"]
    if rock_row.empty:
        continue
    spot = rock_row["mid_price"].values[0]

    # Process each voucher for this group
    for symbol, strike in voucher_strikes.items():
        row = tick_df[tick_df["product"] == symbol]
        if row.empty:
            continue
        mid_price = row["mid_price"].values[0]
        m = math.log(strike / spot) / math.sqrt(TTE)
        iv = implied_volatility(spot, strike, TTE, mid_price)
        # Only use points with iv >= 0.01
        if not math.isnan(iv) and not math.isinf(iv) and iv >= 0.01:
            voucher_data[symbol].append((m, iv))

# Combine all points (with voucher labels) into one list for curve fitting
all_points = []
for voucher, points in voucher_data.items():
    for (m, iv) in points:
        all_points.append((voucher, m, iv))

# Prepare data for parabola fitting (using all filtered points)
fit_ms = np.array([m for (_, m, _) in all_points])
fit_ivs = np.array([iv for (_, _, iv) in all_points])
a, b, c = np.polyfit(fit_ms, fit_ivs, 2)

print(f"\nFitted parabola coefficients after filtering:")
print(f"a = {a:.6f}")
print(f"b = {b:.6f}")
print(f"c = {c:.6f}")

# Generate the fitted IV curve
m_vals = np.linspace(min(fit_ms), max(fit_ms), 200)
ivs_fit = a * m_vals**2 + b * m_vals + c

# Define custom colors for each voucher
colors = {
    "VOLCANIC_ROCK_VOUCHER_9500": "blue",
    "VOLCANIC_ROCK_VOUCHER_9750": "green",
    "VOLCANIC_ROCK_VOUCHER_10000": "red",
    "VOLCANIC_ROCK_VOUCHER_10250": "purple",
    "VOLCANIC_ROCK_VOUCHER_10500": "orange",
}

plt.figure(figsize=(10, 5))

# Plot each voucher's points in its designated color
for voucher, color in colors.items():
    voucher_points = [(m, iv) for (v, m, iv) in all_points if v == voucher]
    if voucher_points:
        voucher_ms = [pt[0] for pt in voucher_points]
        voucher_ivs = [pt[1] for pt in voucher_points]
        plt.scatter(voucher_ms, voucher_ivs, alpha=0.6, label=voucher, color=color)

# Plot the fitted IV curve
plt.plot(m_vals, ivs_fit, color="black", linestyle="--", label="Fitted IV Curve")
plt.xlabel("m = log(K/S) / sqrt(T)")
plt.ylabel("Implied Volatility")
plt.title("Implied Volatility vs m (Points with IV >= 0.01, Combined Days)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
