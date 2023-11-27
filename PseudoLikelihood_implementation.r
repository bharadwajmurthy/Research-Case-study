# Load required libraries
install.packages("EBImage")
library(EBImage)

# Generate a synthetic grayscale image
set.seed(123)
image_size <- 128  # Size of the image (e.g., 128x128 pixels)
original_image <- matrix(rnorm(image_size^2, mean = 0.5, sd = 0.1), ncol = image_size)

# Add Gaussian noise to create a noisy image
sigma_noise <- 0.1
noisy_image <- original_image + rnorm(image_size^2, mean = 0, sd = sigma_noise)

# Display the noisy image
display(noisy_image, method = "raster", title = "Noisy Image")

# Define the Likelihood Function
likelihood <- function(Y, mu, sigma) {
  -sum(dnorm(Y, mean = mu, sd = sigma, log = TRUE))
}

# Define the Pseudo-Likelihood Function
pseudo_likelihood <- function(Y, mu, sigma) {
  # Calculate the pseudo-likelihood based on pixel dependencies
  # For simplicity, we assume that each pixel depends on its four immediate neighbors
  N <- nrow(Y)
  M <- ncol(Y)
  
  log_pseudo_likelihood <- 0
  for (i in 2:(N - 1)) {
    for (j in 2:(M - 1)) {
      log_pseudo_likelihood <- log_pseudo_likelihood +
        dnorm(Y[i, j], mean = mu[i, j], sd = sigma, log = TRUE) +
        dnorm(Y[i - 1, j], mean = mu[i, j], sd = sigma, log = TRUE) +
        dnorm(Y[i + 1, j], mean = mu[i, j], sd = sigma, log = TRUE) +
        dnorm(Y[i, j - 1], mean = mu[i, j], sd = sigma, log = TRUE) +
        dnorm(Y[i, j + 1], mean = mu[i, j], sd = sigma, log = TRUE)
    }
  }
  return(log_pseudo_likelihood)
}

# Initialize parameter values (true pixel values)
mu_estimate <- noisy_image  # Initialize with noisy image

# Optimization: Find the optimal mu_estimate using pseudo-likelihood
# For simplicity, we use a basic grid search here
# In practice, you would use more advanced optimization methods
for (iteration in 1:100) {
  for (i in 2:(N - 1)) {
    for (j in 2:(M - 1)) {
      # Calculate the optimal mu_estimate for each pixel
      # You may use optimization techniques like gradient descent here
      # For simplicity, we assume that mu_estimate is the average of neighboring pixels
      mu_estimate[i, j] <- mean(c(mu_estimate[i - 1, j], mu_estimate[i + 1, j], 
                                  mu_estimate[i, j - 1], mu_estimate[i, j + 1]))
    }
  }
}

# Generate the denoised image
denoised_image <- mu_estimate

# Display the denoised image
display(denoised_image, method = "raster", title = "Denoised Image")

# Calculate Mean Squared Error (MSE) as an evaluation metric
mse <- sum((original_image - denoised_image)^2) / (image_size^2)
cat("Mean Squared Error (MSE):", mse, "\n")
