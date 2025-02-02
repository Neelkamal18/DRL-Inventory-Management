import numpy as np
from scipy.optimize import minimize

# Demand data (ensure it's a NumPy array)
demand_data = np.array([10, 15, 20, 25, 30])

# Log-likelihood function for Discrete Weibull (DWeibull) distribution
def dweibull_loglikelihood(params, data):
    """ Computes the negative log-likelihood for the Discrete Weibull distribution. """
    k, lam = params
    
    # Ensure parameters are positive to avoid invalid calculations
    if k <= 0 or lam <= 0:
        return np.inf  # Return infinity for invalid parameter values
    
    # Compute the log-likelihood
    log_likelihood = np.sum(np.log(k / lam * (data / lam)**(k - 1) * np.exp(-(data / lam)**k)))
    
    return -log_likelihood  # Negative log-likelihood (for minimization)

# Initial parameter guess (shape k, scale lambda)
initial_params = [1.0, 1.0]

# Perform Maximum Likelihood Estimation (MLE)
result = minimize(dweibull_loglikelihood, initial_params, args=(demand_data,), method='Nelder-Mead')

# Check if optimization was successful
if result.success:
    k_param, lam_param = result.x  # Extract estimated parameters
    print("MLE Estimation Successful!")
    print(f"Estimated Shape Parameter (k): {k_param:.4f}")
    print(f"Estimated Scale Parameter (λ): {lam_param:.4f}")
else:
    print("MLE Optimization Failed:", result.message)
