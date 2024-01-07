import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

def T2_product_rule(Ta, Tb):
    return 1 / ((1/Ta) + (1/Tb))


def load_fit_results(path, pickle_filename):
    with open(os.path.join(path, pickle_filename), 'rb') as f:
        fit_results = pickle.load(f)
    return fit_results

def magnitude_of_field(field_tuple):
    """Compute the magnitude of the magnetic field from its tuple representation."""
    return np.sqrt(sum([x**2 for x in field_tuple]))

def extract_float_from_key(key):
    """Extract the float from the string of the key"""
    match = re.search(r'Value (\d+\.?\d*)', key)
    if match:
        return float(match.group(1))
    else:
        raise ValueError(f"Could not extract float from key: {key}")

def plot_T2_vs_field_combined(e_pickle_filename, n_pickle_filename, data_range=None):
    path = "VOTPP folder/Results/Pickle files/Fits/"

    # Load fit results for both datasets
    e_fit_results = load_fit_results(path, e_pickle_filename)
    n_fit_results = load_fit_results(path, n_pickle_filename)

    # Assuming the keys are consistent across both datasets
    keys = e_fit_results.keys()

    # Extract x_values from the keys; modify this line if x_values are different
    x_values = [key[1] for key in keys]

    # Extract T2 values for both datasets
    T2_values_e = [e_fit_results[key]['T2'] for key in keys]
    T2_values_n = [n_fit_results[key]['T2'] for key in keys]

    # Convert T2 from ms to microseconds
    T2_values_e = [T2 * 1e3 for T2 in T2_values_e]
    T2_values_n = [T2 * 1e3 for T2 in T2_values_n]

    # Calculate T2 product rule values
    T2_product_values = [T2_product_rule(Ta, Tb) for Ta, Tb in zip(T2_values_e, T2_values_n)]

    # Apply data range if provided
    if data_range is not None:
        start, end = data_range
        x_values = x_values[start:end]
        T2_values_e = T2_values_e[start:end]
        T2_values_n = T2_values_n[start:end]
        T2_product_values = T2_product_values[start:end]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, T2_values_e, marker='o', label='e Dataset')
    plt.plot(x_values, T2_values_n, marker='o', label='n Dataset')
    plt.plot(x_values, T2_product_values, marker='o', label='T2 Product Rule')
    plt.title("T2 vs Numerical Key")
    plt.xlabel("Numerical Key")
    plt.ylabel(r"$T_2$ ($\mu$s)")
    plt.ylim(bottom=0)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # Save and show plot
    output_filename = os.path.join(path, "T2_vs_Numerical_Key_Combined.png")
    plt.savefig(output_filename, dpi=300)
    plt.show()

# Example usage
plot_T2_vs_field_combined('[n-e]-(e).pkl_fit_results.pkl', '[n-e]-(n).pkl_fit_results.pkl')
