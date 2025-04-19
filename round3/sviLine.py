import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from statistics import NormalDist
from scipy.optimize import brentq, least_squares

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
        return brentq(lambda σ: bs_call_price(S, K, T, σ) - market_price, 1e-5, 5)
    except ValueError:
        return float("nan")

# SVI model: total variance w(m) = a + b[ρ(m−μ) + sqrt((m−μ)^2 + σ²)]
def svi_total_var(params, m):
    a, b, rho, mu, sigma = params
    return a + b * (rho*(m-mu) + np.sqrt((m-mu)**2 + sigma**2))

# load & prepare data
EXPIRY_DAY, CURRENT_ROUND = 8, 0
voucher_strikes = {
    "VOLCANIC_ROCK_VOUCHER_9500": 9500,
    "VOLCANIC_ROCK_VOUCHER_9750": 9750,
    "VOLCANIC_ROCK_VOUCHER_10000": 10000,
    "VOLCANIC_ROCK_VOUCHER_10250": 10250,
    "VOLCANIC_ROCK_VOUCHER_10500": 10500,
}

df_list = []
for day in (0,1,2):
    d = pd.read_csv(f"./round3/data/prices_round_3_day_{day}.csv", sep=";")
    d["timestamp"] += day * 1000000
    df_list.append(d)
df = pd.concat(df_list, ignore_index=True)

# collect (m, iv) pairs
data = []
for ts in df["timestamp"].unique():
    tick = df[df["timestamp"] == ts]
    rock = tick[tick["product"]=="VOLCANIC_ROCK"]
    if rock.empty: continue
    r = rock.iloc[0]
    # only use this tick if both bid1 and ask1 exist and are valid
    if not {"bid_price_1","ask_price_1"}.issubset(r.index) \
       or pd.isna(r["bid_price_1"]) or pd.isna(r["ask_price_1"]):
        continue
    spot = (r["bid_price_1"] + r["ask_price_1"]) / 2

    # convert ts → days → fractional days
    days_passed = (ts / 1e6) / 86400
    fractional  = CURRENT_ROUND + days_passed
    TTE = max(0, EXPIRY_DAY - fractional) / 365
    if TTE <= 0: continue

    for sym, K in voucher_strikes.items():
        v = tick[tick["product"]==sym]
        if v.empty: continue
        v = v.iloc[0]
        # skip if we don’t have a valid bid1/ask1
        if not {"bid_price_1","ask_price_1"}.issubset(v.index) \
           or pd.isna(v["bid_price_1"]) or pd.isna(v["ask_price_1"]):
            continue
        mid = (v["bid_price_1"] + v["ask_price_1"]) / 2

        m  = math.log(K/spot) / math.sqrt(TTE)
        iv = implied_volatility(spot, K, TTE, mid)
        
        intrinsic = max(0.0, spot - K)
        if mid <= intrinsic:  
            continue
        
        if not math.isnan(iv) and not math.isinf(iv):
            data.append((m, iv))

data = np.array(data)
ms, ivs = data[:,0], data[:,1]

# observed total variances
w_obs = ivs**2

# SVI calibration via least squares
def residuals(params):
    return svi_total_var(params, ms) - w_obs

# init & bounds: a>0, b>0, |ρ|<1, σ>0
init   = [0.02, 0.1, 0.0, 0.0, 0.1]
lower  = [1e-8, 1e-8, -0.999, np.min(ms), 1e-8]
upper  = [1.0, 1.0,  0.999, np.max(ms), 1.0]
res    = least_squares(residuals, init, bounds=(lower, upper))
a, b, rho, mu, sigma = res.x
print("SVI params:")
print(f"a_svi     = {a:.5f}")
print(f"b_svi     = {b:.5f}")
print(f"rho_svi   = {rho:.5f}")
print(f"mu_svi    = {mu:.5f}")
print(f"sigma_svi = {sigma:.5f}")

# build fit curve
m_grid = np.linspace(ms.min(), ms.max(), 300)
w_fit  = svi_total_var(res.x, m_grid)
iv_fit = np.sqrt(np.clip(w_fit, 0, None))

# plot
plt.figure(figsize=(10,5))
plt.scatter(ms, ivs, alpha=0.3, label="Observed IV")
plt.plot(m_grid, iv_fit, 'r--', linewidth=2, label="SVI fit")
plt.xlabel("m = log(K/S)/√T")
plt.ylabel("Implied Volatility")
plt.title("SVI‐fitted Vol Smile")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
