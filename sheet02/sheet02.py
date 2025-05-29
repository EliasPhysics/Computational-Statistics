import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def gaussian(x, mu, sigma):
    """Gaussian function for fitting."""
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


# Sample sizes to iterate through
sample_sizes = [100, 1000, 10000, 100000]

for n in sample_sizes:
    # Generate random data from a normal distribution (mean=0, std=1)
    data = np.random.normal(loc=0, scale=1, size=n)

    # Create histogram
    plt.figure(figsize=(8, 5))
    counts, bins, _ = plt.hist(data, bins=30, density=True, alpha=0.6, color='blue', label='Sample Data')

    # Calculate bin centers for fitting
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Initial parameter guesses (sample mean and std)
    mu_initial = np.mean(data)
    sigma_initial = np.std(data)

    # Fit the Gaussian function to the histogram data
    try:
        popt, pcov = curve_fit(gaussian, bin_centers, counts, p0=[mu_initial, sigma_initial])
        mu_fit, sigma_fit = popt
    except RuntimeError:
        print(f"Fit failed for n={n}. Skipping plot.")
        plt.close()
        continue

    # Generate points for the fitted and true Gaussian curves
    x_fit = np.linspace(-4, 4, 1000)
    y_fit = gaussian(x_fit, mu_fit, sigma_fit)
    y_true = gaussian(x_fit, 0, 1)  # True Gaussian (mean=0, sigma=1)

    # Plot the fitted and true Gaussians
    plt.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'Fitted Gaussian ($\mu=${mu_fit:.2f}, $\sigma=${sigma_fit:.2f})')
    plt.plot(x_fit, y_true, 'k--', linewidth=2, label='True Gaussian ($\mu=0$, $\sigma=1$)')

    # Configure plot appearance
    plt.xlim(-4, 4)
    plt.title(f'Normal Distribution Fit with n={n} Samples')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"sheet02_nsample_size={n}.png")
