from scipy.stats import lognorm
import matplotlib.pyplot as plt
# Parameters for the true log-normal model (mu and sigma of the underlying normal distribution)
true_lognormal_mu = 0
true_lognormal_sigma = 0.5  # Smaller sigma to make the skewness more noticeable

# Generate synthetic data from a log-normal distribution
lognormal_data = np.random.lognormal(true_lognormal_mu, true_lognormal_sigma, 1000)

# MLE assuming normal distribution
mle_result_normal = minimize(lambda params: negative_log_likelihood(params, lognormal_data), 
                             [np.mean(lognormal_data), np.std(lognormal_data)], 
                             bounds=[(-np.inf, np.inf), (0.001, np.inf)])

mle_normal_mu, mle_normal_sigma = mle_result_normal.x

# QLE using median and IQR (as a proxy for a robust estimation method)
qle_median = np.median(lognormal_data)
qle_iqr = np.subtract(*np.percentile(lognormal_data, [75, 25]))

# For log-normal distribution, the median is exp(mu), and the IQR relates to the sigma
# We can transform these back to approximate the true parameters
qle_approx_mu = np.log(qle_median)
qle_approx_sigma = qle_iqr / (2 * norm.ppf(0.75))  # Approximation assuming a normal distribution

(mle_normal_mu, mle_normal_sigma), (qle_approx_mu, qle_approx_sigma)

# Plot the histogram of the lognormal data
plt.figure(figsize=(12, 6))
plt.hist(lognormal_data, bins=30, alpha=0.6, color='g', label='Log-normal Data')
plt.axvline(x=qle_median, color='b', linestyle='--', label='QLE Median (Approx. Mu)')
plt.axvline(x=np.exp(true_lognormal_mu), color='r', linestyle='-', label='True Mu')
plt.title('Histogram of Log-normal Data and Estimates')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Plot for comparison of MLE vs QLE
plt.figure(figsize=(12, 6))

# True parameters
plt.scatter(true_lognormal_mu, true_lognormal_sigma, color='r', zorder=5, label='True Parameters')

# MLE Estimates
plt.scatter(mle_normal_mu, mle_normal_sigma, color='g', label='MLE Estimates (Assuming Normality)')

# QLE Estimates
plt.scatter(qle_approx_mu, qle_approx_sigma, color='b', label='QLE Estimates')

plt.title('Comparison of Parameter Estimates')
plt.xlabel('Mu Estimate')
plt.ylabel('Sigma Estimate')
plt.legend()
plt.grid(True)
plt.show()
