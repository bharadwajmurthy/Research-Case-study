from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
# Parameters for the mixture model
mu1, sigma1 = -2, 0.5  # Parameters for the first normal distribution
mu2, sigma2 = 2, 1.5   # Parameters for the second normal distribution
mixture_proportions = [0.3, 0.7]  # Proportions of each mixture component

# Generate synthetic data from a mixture of two normal distributions
np.random.seed(0)
data_mixture = np.concatenate([
    np.random.normal(mu1, sigma1, int(10000 * mixture_proportions[0])),
    np.random.normal(mu2, sigma2, int(10000 * mixture_proportions[1]))
])

# PLE: Fit a Gaussian mixture model (we use this as PLE here for simplicity)
# This is actually a full likelihood approach for mixture models, 
# but we can consider it a PLE since we maximize the likelihood for each component separately
gmm = GaussianMixture(n_components=2, covariance_type='full')
gmm.fit(data_mixture.reshape(-1, 1))

# QLE: Calculate the sample mean and variance (not using a specialized QLE method here, for simplicity)
qle_mean = np.mean(data_mixture)
qle_variance = np.var(data_mixture)

# Sort the mixture components by mean for easier comparison
sorted_indices = np.argsort(gmm.means_.ravel())
ple_means = gmm.means_.ravel()[sorted_indices]
ple_variances = gmm.covariances_.ravel()[sorted_indices]

# Plot the histogram of the mixture data
plt.figure(figsize=(12, 6))
plt.hist(data_mixture, bins=50, alpha=0.6, color='g', label='Mixture Data')
plt.axvline(x=qle_mean, color='b', linestyle='--', label='QLE Mean')
plt.title('Histogram of Mixture Data and QLE Estimate')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Plot for comparison of PLE vs QLE
plt.figure(figsize=(12, 6))

# True parameters
plt.scatter([mu1, mu2], [sigma1, sigma2], color='r', zorder=5, label='True Parameters')

# PLE Estimates
plt.scatter(ple_means, np.sqrt(ple_variances), color='g', label='PLE Estimates')

# QLE Estimate (only one mean and variance since QLE does not consider the mixture)
plt.scatter(qle_mean, np.sqrt(qle_variance), color='b', label='QLE Estimate')

plt.title('Comparison of Parameter Estimates for Mixture Data')
plt.xlabel('Mean Estimate')
plt.ylabel('Standard Deviation Estimate')
plt.legend()
plt.grid(True)
plt.show()

# Return the estimates for comparison
(ple_means, np.sqrt(ple_variances)), (qle_mean, np.sqrt(qle_variance))
