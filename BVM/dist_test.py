import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm

# Parameters
n = 100          # initial population size
lambda_ = 1.0     # branching rate
T = 2          # time to simulate
num_simulations = 10000

def simulate_yule_process(n, lambda_, T):
    """Simulates the Yule process for n individuals up to time T."""
    population = n
    t = 0
    while True:
        split_times = np.random.exponential(1 / (population * lambda_))
        next_split_time = t + split_times
        if next_split_time > T:
            break
        population += 1
        t = next_split_time
    return population

# Simulate the process
Xt_samples = np.array([simulate_yule_process(n, lambda_, T) for _ in range(num_simulations)])

print(Xt_samples)

# Theoretical expected value and variance
E_Xt = n * np.exp(lambda_ * T)
print(E_Xt)
Var_Xt = n * np.exp(lambda_ * T) * (np.exp(lambda_ * T) - 1)

# Centered samples
Xt_centered = Xt_samples - E_Xt

# Plot histogram
plt.figure(figsize=(10, 6))
counts, bins, _ = plt.hist(Xt_centered, bins=100, density=True, alpha=0.6, label='Simulated $X_T - \\mathbb{E}[X_T]$')

# Normal distribution overlay
x = np.linspace(bins[0], bins[-1], 1000)
normal_pdf = norm.pdf(x, 0, np.sqrt(Var_Xt))
plt.plot(x, normal_pdf, label='Normal Approximation', linewidth=2)

# Log-normal distribution overlay
# Since X_T ≈ LogNormal(μ, σ^2) with μ = ln(n) + λT, σ^2 = λT/n (from earlier analysis)
mu_lognorm = np.log(n) + lambda_ * T
sigma_lognorm = np.sqrt(lambda_ * T / n)
lognorm_pdf = lognorm.pdf(x + E_Xt, s=sigma_lognorm, scale=np.exp(mu_lognorm))
plt.plot(x, lognorm_pdf, label='Log-normal Approximation', linewidth=2)

plt.title(f'Distribution of $X_T - \\mathbb{{E}}[X_T]$ (T = {T}, n = {n})')
plt.xlabel('$X_T - \\mathbb{E}[X_T]$')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()

