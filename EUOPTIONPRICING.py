import numpy as np
import matplotlib.pyplot as plt

def euro_call_mc(S0, K, r, sigma, T, N, seed=None):
    if seed is not None:
        np.random.seed(seed)
    Z  = np.random.normal(size=N)                    
    #recall the Black-Scholes Formula, here we use some properties of Brownian Motion term W_T
    #By def. of Brownian Motion, var(W_T)=T, so the volatility of stock return is just sqrt(T)
    #Since Z ~ N(0,1) then sqrt(T)*Z gives W_T ~ N(0,T)
    ST = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z) 
    payoff = np.maximum(ST - K, 0.0) #You can tweak the payoff function here to get European put option pricing algo
    disc_payoff = np.exp(-r*T) * payoff
    price = disc_payoff.mean()
    stderr = disc_payoff.std(ddof=1) / np.sqrt(N)    
    return price, stderr, ST

# Parameters
#Some comments on volatility: 
#Vol of Large-cap equity (e.g. S&P 500) = 0.15-0.25
#of Growth/Tech Stock = 0.25-0.40
#of Commodities(oil,gold) = 0.25-0.60
#of FX pairs = 0.05-0.15
#of Crypto = 0.8-1.5
S0, K, r, sigma, T, N = 104,114,0.0395,0.25,0.5,10000

price, stderr, ST = euro_call_mc(S0, K, r, sigma, T, N, seed=42)
print(f"MC Price = {price:.4f} ± {stderr:.4f}")

# Plot histogram of terminal stock prices distribution
plt.figure(figsize=(8,5))
plt.hist(ST, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
plt.title(f"Distribution of Simulated $S_T$ (N={N}, σ={sigma})")
plt.xlabel("Terminal Stock Price $S_T$")
plt.ylabel("Frequency")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

#Ploting sample paths
# Path settings
M = 182        # number of time steps
P = 20      # number of paths to plot
seed_paths = 0 # set to None for fresh randomness

if seed_paths is not None:
    np.random.seed(seed_paths)

dt = T / M
t = np.linspace(0.0, T, M+1)

# simulate P GBM paths (rows=time, cols=paths)
S_paths = np.empty((M+1, P), dtype=float)
S_paths[0] = S0

Z = np.random.normal(size=(M, P))             # iid N(0,1)
drift = (r - 0.5 * sigma**2) * dt
vol   = sigma * np.sqrt(dt)

for m in range(M):
    S_paths[m+1] = S_paths[m] * np.exp(drift + vol * Z[m])

# plot
plt.figure(figsize=(9,5))
plt.plot(t, S_paths)                           # one line per path
plt.plot(t, S0 * np.exp(r * t), '--', linewidth=2, label='E[S_t]=S0·e^{rt}')
plt.title(f'{P} simulated GBM paths (σ={sigma}, r={r}, T={T})')
plt.xlabel('Time (years)')
plt.ylabel('Stock price')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
