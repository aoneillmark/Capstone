import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import re  # Regular expressions

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

# def plot_T2_vs_field(pickle_filename, data_range=None):
#     path = "VOTPP folder/Results/Pickle files/Fits/"
    
#     # Load fit results
#     fit_results = load_fit_results(path, pickle_filename)
    
#     # Define a mapping of filenames to x-axis labels and titles
#     file_to_labels = {
#         "magnetic": ("B (T)", r"$T_2$ vs Magnetic Field Magnitude"),
#         "cell_size": ("Cell Size", r"$T_2$ vs Cell Size"),
#         "r_bath": ("Bath Radius", r"$T_2$ vs Bath Radius"),
#         "r_dipole": ("Dipole Radius", r"$T_2$ vs Dipole Radius"),
#         "alphabeta_combination": ("Alpha-Beta Index", r"$T_2$ vs Alpha-Beta Index"),
#         # ... add more mappings as needed
#     }

#     for key_part, (xlabel, title) in file_to_labels.items():
#         if key_part in pickle_filename:
#             break
#     else:
#         raise ValueError(f"Unrecognized pickle filename: {pickle_filename}")

#     if "magnetic" in pickle_filename:
#         x_values = [magnitude_of_field(tuple(map(float, key.split(', ')))) for key in fit_results.keys()]
#         # Convert magnetic field from Gauss to Tesla
#         x_values = [field*1e-4 for field in x_values]
#     else:
#         x_values = [extract_float_from_key(key) for key in fit_results.keys()]

#     T2_values = [fit_results[key]['T2'] for key in fit_results.keys()]
    
#     # Convert T2 from ms to microseconds
#     T2_values = [(T2*1e3) for T2 in T2_values]

#     # Apply data range if provided
#     if data_range is not None:
#         start, end = data_range
#         x_values = x_values[start:end]
#         T2_values = T2_values[start:end]

#     # Plotting
#     plt.figure(figsize=(10,6))
#     plt.semilogy(x_values, T2_values, marker='o')
#     plt.title(title)
#     plt.xlabel(xlabel)
#     plt.ylabel(r"$T_2$ ($\mu$s)")
#     plt.grid(True)
#     # plt.xticks(np.arange(0.04, 0.36, 0.04))
#     # plt.xlim(0.04, 0.34)
#     plt.tight_layout()
    
#     # Save and show plot
#     output_filename = os.path.join(path, f"T2_vs_{xlabel.replace(' ', '_').replace('(', '').replace(')', '')}.png")
#     plt.savefig(output_filename, dpi=300)
#     plt.show()

def plot_T2_vs_field(pickle_filename, data_range=None):
    path = "VOTPP folder/Results/Pickle files/Fits/"
    
    # Load fit results
    fit_results = load_fit_results(path, pickle_filename)

    # Since the keys are tuples, we directly extract the first element as the numerical key
    x_values = [key[0] for key in fit_results.keys()]

    # If you want to plot against the magnetic field magnitude, you would calculate it here
    # If the second element in the tuple is the magnetic field magnitude, you could use:
    # x_values = [magnitude_of_field(key[1:4]) for key in fit_results.keys()]

    T2_values = [fit_results[key]['T2'] for key in fit_results.keys()]

    # Convert T2 from ms to microseconds
    T2_values = [(T2*1e3) for T2 in T2_values]

    # Apply data range if provided
    if data_range is not None:
        start, end = data_range
        x_values = x_values[start:end]
        T2_values = T2_values[start:end]

    # Plotting
    plt.figure(figsize=(10,6))
    # plt.semilogy(x_values, T2_values, marker='o')
    plt.plot(x_values, T2_values, marker='o')
    plt.title("T2 vs Numerical Key")
    plt.xlabel("Numerical Key")
    plt.ylabel(r"$T_2$ ($\mu$s)")
    plt.grid(True)
    plt.tight_layout()
    
    # Save and show plot
    output_filename = os.path.join(path, "T2_vs_Numerical_Key.png")
    plt.savefig(output_filename, dpi=300)
    plt.show()

# Call the function with the name of your pickle files
# plot_T2_vs_field('magnetic_fit_results.pkl')
# plot_T2_vs_field('cell_size_fit_results.pkl')
# plot_T2_vs_field('r_bath_fit_results.pkl', data_range=(1, None))
# plot_T2_vs_field('r_dipole_fit_results.pkl', data_range=(1, None))
plot_T2_vs_field('alphabeta_combination_fit_results.pkl')