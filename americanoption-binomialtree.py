# nvda_options_demo.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import norm
from datetime import datetime

# ---------------------------------------------------
# 1. Download NVDA historical data
# ---------------------------------------------------
# Download historical price data for NVIDIA from Yahoo Finance
# (Jan 1, 2023 to Oct 31, 2025)
# auto_adjust=True to ensure prices are modified for splits and dividends
nvda = yf.download("NVDA", start="2023-01-01", end="2025-10-31", auto_adjust=True)
nvda_close = nvda['Close']
nvda_close.plot(title="NVDA Closing Prices (2023-2025)")
plt.show()

# ---------------------------------------------------
# 2. Estimate historical volatility
# ---------------------------------------------------
# Compute daily log returns from the adjusted closing prices
# Formula: log(P_t / P_{t-1}) to get percentage changes
log_returns = np.log(nvda_close / nvda_close.shift(1)).dropna()

# Calculate the annualized standard deviation of log returns
# np.std(..., axis=0) computes the sample standard deviation
# Extract scalar from Series using .iloc[0]
# Multiply by sqrt(252) to annualize
hist_vol = np.std(log_returns, axis=0).iloc[0] * np.sqrt(252)
print(f"Historical Volatility: {hist_vol:.4f}")

# ---------------------------------------------------
# 3. Set risk-free rate
# ---------------------------------------------------
# 4.5% annualized rate
r = 0.045

# ---------------------------------------------------
# 4. Binomial tree pricer for American options
# ---------------------------------------------------
# Function to price American Call or Put options using a binomial tree
def binomial_american_option(S, K, T, r, sigma, N, option_type="call"):
    # Calculate time step size in years
    dt = T / N

    # Compute the up and down factors (volatility and time step)
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u

    # Handle edge case where u == d so there is no division by 0
    if u == d:
        q = 0.5
    else:
        # Computes the risk-neutral probability of an up move
        q = (np.exp(r * dt) - d) / (u - d)

    # Use vectorized computing to produce terminal asset prices at maturity
    # asset_prices = S * d**np.arange(N, -1, -1) * u**np.arange(0, N+1)
    asset_prices = S * (d ** np.arange(N, -1, -1)) * (u ** np.arange(0, N+1))

    # Determine option payoffs at maturity according to type
    if option_type == "call":
        option_values = np.maximum(asset_prices - K, 0)
    else:
        option_values = np.maximum(K - asset_prices, 0)

    # To calculate the option value at earlier nodes, use backward induction
    for i in range(N-1, -1, -1):
        # From the following time step, discount the projected option value
        option_values = np.exp(-r * dt) * (q * option_values[1:] + (1 - q) * option_values[:-1])
        # At the current node level, recalculate asset prices
        asset_prices = S * (d ** np.arange(i, -1, -1)) * (u ** np.arange(0, i+1))
        # For American alternatives, apply the early exercise condition
        if option_type == "call":
            option_values = np.maximum(option_values, asset_prices - K)
        else:
            option_values = np.maximum(option_values, K - asset_prices)

    # At time 0, return the option value at the root node
    return option_values[0]

# ---------------------------------------------------
# 5. Price example options
# ---------------------------------------------------
# Determine the current spot price by extracting the most recent NVDA closing price
S0 = nvda_close.iloc[-1].item()

# Use the ATM option to set the strike price at the current spot price
K = S0

# Determine the maturity period in years
# 7 trading days = 7/252 of a year
T = 7 / 252

# Determine the binomial tree's time step count
N = 200

# Use the binomial tree model to determine the cost of an American call option
call_price = binomial_american_option(S0, K, T, r, hist_vol, N, "call")
# Use the binomial tree model to determine the cost of an American put option
put_price = binomial_american_option(S0, K, T, r, hist_vol, N, "put")

print(f"American Call Price: {call_price:.2f}")
print(f"American Put Price: {put_price:.2f}")

# ---------------------------------------------------
# 6. Greeks via finite differences
# ---------------------------------------------------
def delta(S, K, T, r, sigma, N, option_type):
    # Little change in the price of the underlying asset
    h = 0.1
    # Use a little higher spot price (S + h) to price the option
    up = binomial_american_option(S + h, K, T, r, sigma, N, option_type)
    # Use a little lower spot price (S - h)
    down = binomial_american_option(S - h, K, T, r, sigma, N, option_type)
    # Determine the symmetric difference quotient, or delta
    return (up - down) / (2 * h)

# Same concept as Delta but for Gamma
def gamma(S, K, T, r, sigma, N, option_type):
    h = 0.1
    up = binomial_american_option(S + h, K, T, r, sigma, N, option_type)
    center = binomial_american_option(S, K, T, r, sigma, N, option_type)
    down = binomial_american_option(S - h, K, T, r, sigma, N, option_type)
    # Determine the second-order central difference, or gamma
    return (up - 2 * center + down) / (h ** 2)

print("Call Delta:", delta(S0, K, T, r, hist_vol, N, "call"))
print("Call Gamma:", gamma(S0, K, T, r, hist_vol, N, "call"))

# ---------------------------------------------------
# 7. Monte Carlo simulation
# ---------------------------------------------------
# Function that uses Monte Carlo simulation to estimate the cost of an American option
def monte_carlo_american_option(S, K, T, r, sigma, option_type="call", paths=10000, steps=100):
    # Determine the time increase for each simulation step
    dt = T / steps
    # Determine the present value adjustment discount factor
    discount = np.exp(-r * T)
    payoff = []

    # Create "paths" for a variety of price trajectories
    for _ in range(paths):
        # Begin every path at the starting location price
        prices = [S]
        # Use geometric Brownian motion to create a single pricing route
        for _ in range(steps):
            z = np.random.normal()
            prices.append(prices[-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z))

        # Determine the payoff at maturity using the option type
        if option_type == "call":
            payoff.append(max(prices[-1] - K, 0))
        else:
            payoff.append(max(K - prices[-1], 0))

    # Return the expected option price based on the discounted average payment
    return discount * np.mean(payoff)

mc_call = monte_carlo_american_option(S0, K, T, r, hist_vol, "call")
mc_put = monte_carlo_american_option(S0, K, T, r, hist_vol, "put")

print(f"Monte Carlo Call Price: {mc_call:.2f}")
print(f"Monte Carlo Put Price: {mc_put:.2f}")

# ---------------------------------------------------
# 8. Hedging simulation
# ---------------------------------------------------
# Take the past five NVDA closing prices and turn them into a NumPy array
prices = nvda_close[-5:].values

# At each time step, create empty lists to hold option values and deltas
deltas = []
option_vals = []

# Go over every pricing during the five-day period
for i, S in enumerate(prices):
    # Calculate the years till adulthood, decreasing by one day (e.g., 4/252, 3/252, ..., 0/252)
    T_i = (4 - i) / 252
    # Determine the American put option's value at this time and price
    val = binomial_american_option(S, K, T_i, r, hist_vol, N, "put")
    # Determine the put option's delta at this time and price
    d = delta(S, K, T_i, r, hist_vol, N, "put")
    # Save the delta and option value for this time step
    option_vals.append(val)
    deltas.append(d)

# Create a blank list to hold the hedged profit and loss (PnL)
pnl = []
# To calculate daily PnL, loop through each day (beginning at the second)
for i in range(1, len(prices)):
    # Modification of the option value from the day before to the day now
    dp = option_vals[i] - option_vals[i-1]
    # Hedge position PnL: negative delta * underlying price change
    hedge = -deltas[i-1] * (prices[i] - prices[i-1])
    # Total hedged PnL = hedge adjustment + option value change
    pnl.append(dp + hedge)

print("Hedged PnL:", pnl)

# ---------------------------------------------------
# 9. Market comparison
# ---------------------------------------------------
# Making a Ticker object so that NVDA can access its option info
nvda_ticker = yf.Ticker("NVDA")
print("Available expirations:", nvda_ticker.options)

# Obtain the current date to filter for expirations in the future
today = datetime.today().date()

# After today, automatically choose the closest expiration date
valid_expiration = next(
    date for date in nvda_ticker.options
    if datetime.strptime(date, "%Y-%m-%d").date() > today
)

print("Using expiration:", valid_expiration)

# Get the whole option chain, including puts and calls, for the chosen expiration
chain = nvda_ticker.option_chain(valid_expiration)
calls = chain.calls
puts = chain.puts

# Determine each call and put option's mid-price by averaging the bid and ask prices
calls['mid'] = (calls['bid'] + calls['ask']) / 2
puts['mid'] = (puts['bid'] + puts['ask']) / 2

# Filter call and put options whose strikes are within ±5 of the spot price as of right now
atm_calls = calls[np.abs(calls['strike'] - S0) < 5]
atm_puts = puts[np.abs(puts['strike'] - S0) < 5]

print("Market Call Options:")
print(atm_calls[['strike', 'mid']])
print("Market Put Options:")
print(atm_puts[['strike', 'mid']])
