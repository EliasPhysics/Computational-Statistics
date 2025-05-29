import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, norm, cauchy

# Parameters
m = 2  # mean
ns = [50, 1000, 10000]  # sample sizes

for n in ns:
    # Generate random Poisson numbers
    data = np.random.poisson(m, size=n)

    # Create histogram
    plt.figure(figsize=(10, 5))
    counts, bins, _ = plt.hist(data, bins=np.arange(-0.5, 10.5, 1), density=True, alpha=0.7, label=f'Histogram (n={n})')

    # Plot analytical Poisson PMF
    x = np.arange(0, 10)
    plt.plot(x, poisson.pmf(x, m), 'ro-', label='Poisson PMF')

    # Fit Gaussian
    mu, std = norm.fit(data)
    x_fit = np.linspace(-1, 10, 100)
    plt.plot(x_fit, norm.pdf(x_fit, mu, std), 'k-', linewidth=2, label=f'Gaussian fit (μ={mu:.2f}, σ={std:.2f})')

    plt.title(f'Poisson Distribution (m=2) vs Gaussian Fit (n={n})')
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.savefig(f"plot_a_n={n}")

#

for n in ns:
    # Generate random Cauchy numbers
    data = np.random.standard_cauchy(size=n)
    # Trim extreme values for visualization
    data = data[(data > -10) & (data < 10)]

    plt.figure(figsize=(10, 5))
    counts, bins, _ = plt.hist(data, bins=50, density=True, alpha=0.7, label=f'Histogram (n={n})')

    # Plot analytical Cauchy PDF
    x = np.linspace(-10, 10, 1000)
    plt.plot(x, cauchy.pdf(x), 'r-', label='Cauchy PDF')

    # Attempt Gaussian fit
    mu, std = norm.fit(data)
    plt.plot(x, norm.pdf(x, mu, std), 'k-', linewidth=2, label=f'Gaussian fit (μ={mu:.2f}, σ={std:.2f})')

    plt.title(f'Cauchy Distribution vs Gaussian Fit (n={n})')
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.savefig(f"plot_b_n={n}")

n = 100
means = [1, 5, 20]

for m in means:
    data = np.random.poisson(m, size=n)

    plt.figure(figsize=(10, 5))
    counts, bins, _ = plt.hist(data, bins=np.arange(-0.5, m * 3 + 0.5, 1), density=True, alpha=0.7, label='Histogram')

    # Poisson PMF
    x = np.arange(0, m * 3)
    plt.plot(x, poisson.pmf(x, m), 'ro-', label='Poisson PMF')

    # Gaussian fit
    mu, std = norm.fit(data)
    x_fit = np.linspace(min(bins), max(bins), 100)
    plt.plot(x_fit, norm.pdf(x_fit, mu, std), 'k-', label=f'Gaussian fit (μ={mu:.2f}, σ={std:.2f})')

    plt.title(f'Poisson (m={m}, n=100) vs Gaussian Fit')
    plt.legend()
    plt.savefig(f"plot_c_n={n}")

n = 100
locations = [0, 2, 5]

for loc in locations:
    data = np.random.standard_cauchy(size=n) + loc
    data = data[(data > loc - 10) & (data < loc + 10)]

    plt.figure(figsize=(10, 5))
    counts, bins, _ = plt.hist(data, bins=50, density=True, alpha=0.7, label='Histogram')

    # Cauchy PDF
    x = np.linspace(loc - 10, loc + 10, 1000)
    plt.plot(x, cauchy.pdf(x - loc), 'r-', label='Cauchy PDF')

    # Gaussian fit
    mu, std = norm.fit(data)
    plt.plot(x, norm.pdf(x, mu, std), 'k-', label=f'Gaussian fit (μ={mu:.2f}, σ={std:.2f})')

    plt.title(f'Cauchy (location={loc}, n=100) vs Gaussian Fit')
    plt.legend()
    plt.savefig(f"plot_d_n={n}")