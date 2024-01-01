import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the coherence time function
def coherence_time_func(time, beta, T2):
    return np.exp(-(((2*time)/T2)**beta))

# Time range
timespace = np.linspace(0, 1, 201)

# Series parameters
series_params = [
    {"beta": 0.482, "T2": 0.093},  # Series 1
    {"beta": 2.152, "T2": 0.611},   # Series 2
    # {"beta": 2.185, "T2": 0.249},  # Series 1
    # {"beta": 3.539, "T2": 1.028},   # Series 2
    {"beta": 0.221, "T2": 144881},  # Series 3
    {"beta": 0.213, "T2": 241023},  # Series 4
]

# Calculate coherence for each series and store it
coherences = []
for params in series_params:
    coherence = coherence_time_func(timespace, params["beta"], params["T2"])
    coherences.append(coherence)



################################################
# Plotting
################################################
image_path = "VOTPP folder/Results/Plots/"
################################################

# First plot: Plot each coherence curve
plt.figure(figsize=(10, 6))

# Plotting each series
for i, coherence in enumerate(coherences, 1):
    plt.plot(timespace, coherence, label=f"Series {i} (Beta={series_params[i-1]['beta']}, T2={series_params[i-1]['T2']})")

plt.title("Coherence vs Time for Each Series")
plt.xlabel("Time")
plt.ylabel("Coherence")
plt.legend()
plt.tight_layout()
plt.savefig(image_path + "coherence_vs_time_for_each_series.png", dpi=300)
plt.show()

# Second plot: Product of coherences
product_coherence = np.prod(coherences, axis=0)

# Perform curve fitting on the product coherence curve
initial_guesses = [1, 0.5]  # Initial guesses for beta and T2
fitted_params, covariance = curve_fit(coherence_time_func, timespace, product_coherence, p0=initial_guesses)

# Extract the fitted beta and T2 values
fitted_beta, fitted_T2 = fitted_params

# Plot with fitted curve
plt.figure(figsize=(10, 6))
plt.plot(timespace, product_coherence, label="Product of Coherences")
plt.plot(timespace, coherence_time_func(timespace, fitted_beta, fitted_T2), '--', label=f"Fit: Beta={fitted_beta:.3f}, T2={fitted_T2:.3f}")
plt.title("Product of Coherences vs Time with Fit")
plt.xlabel("Time")
plt.ylabel("Product of Coherence")
plt.legend()
plt.tight_layout()
plt.savefig(image_path + "product_of_coherences_with_fit.png", dpi=300)
plt.show()
