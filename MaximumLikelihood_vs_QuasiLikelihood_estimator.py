
```Python
import matplotlib.pyplot as plt

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

```
