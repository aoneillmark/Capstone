import matplotlib.pyplot as plt
import os
import numpy as np
import pickle
import re  # Import the re module if not already imported

# Reuse the previously defined functions
def T2_product_rule(Ta, Tb, Tc, Td):
    return 1 / ((1/Ta) + (1/Tb) + (1/Tc) + (1/Td))

def load_fit_results(path, pickle_filename):
    with open(os.path.join(path, pickle_filename), 'rb') as f:
        fit_results = pickle.load(f)
    return fit_results

def plot_T2_product_rule_combined(e_pickle_filename, C_pickle_filename, h_pickle_filename, N_pickle_filename, AB, data_range=None):
    path = "VOTPP folder/Results/Pickle files/Fits/"

    # Load fit results for all datasets
    e_fit_results = load_fit_results(path, e_pickle_filename)
    C_fit_results = load_fit_results(path, C_pickle_filename)
    h_fit_results = load_fit_results(path, h_pickle_filename)
    N_fit_results = load_fit_results(path, N_pickle_filename)

    # Assuming the keys are consistent across all datasets
    keys = e_fit_results.keys()

    # Extract x_values from the keys
    x_values = [key[3] for key in keys]

    # Extract T2 values for all datasets
    T2_values_e = [e_fit_results[key]['T2'] for key in keys]
    T2_values_C = [C_fit_results[key]['T2'] for key in keys]
    T2_values_h = [h_fit_results[key]['T2'] for key in keys]
    T2_values_N = [N_fit_results[key]['T2'] for key in keys]

    # Convert T2 from ms to microseconds for all datasets
    T2_values_e = [T2 * 1e3 for T2 in T2_values_e]
    T2_values_C = [T2 * 1e3 for T2 in T2_values_C]
    T2_values_h = [T2 * 1e3 for T2 in T2_values_h]
    T2_values_N = [T2 * 1e3 for T2 in T2_values_N]

    # Calculate T2 product rule values
    T2_product_values = [T2_product_rule(Ta, Tb, Tc, Td) for Ta, Tb, Tc, Td in zip(T2_values_e, T2_values_C, T2_values_h, T2_values_N)]

    # Apply data range if provided
    if data_range is not None:
        start, end = data_range
        x_values = x_values[start:end]
        T2_product_values = T2_product_values[start:end]

    # Plotting the combined T2 product rule values
    plt.figure(figsize=(8, 4))
    plt.plot(x_values, T2_product_values, marker='o', label='Product Rule Result')
    # plt.title("T2 Product Rule Combined vs Numerical Key")
    plt.xlabel(r"$B_0$ (Gauss)")
    plt.ylabel(r"$T_2$ ($\mu$s)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # Save and show plot
    output_filename = os.path.join(path, f"T2_vs_B0_Product_Result_{AB}.png")
    plt.savefig(output_filename, dpi=300)
    plt.show()


def plot_T2_vs_field_combined(e_pickle_filename, C_pickle_filename, h_pickle_filename, N_pickle_filename, AB, data_range=None):
    path = "VOTPP folder/Results/Pickle files/Fits/"

    # Load fit results for all datasets
    e_fit_results = load_fit_results(path, e_pickle_filename)
    C_fit_results = load_fit_results(path, C_pickle_filename)
    h_fit_results = load_fit_results(path, h_pickle_filename)
    N_fit_results = load_fit_results(path, N_pickle_filename)

    # Assuming the keys are consistent across all datasets
    keys = e_fit_results.keys()

    # Extract x_values from the keys
    x_values = [key[3] for key in keys]

    # Extract T2 values for all datasets
    T2_values_e = [e_fit_results[key]['T2'] for key in keys]
    T2_values_C = [C_fit_results[key]['T2'] for key in keys]
    T2_values_h = [h_fit_results[key]['T2'] for key in keys]
    T2_values_N = [N_fit_results[key]['T2'] for key in keys]

    # Convert T2 from ms to microseconds for all datasets
    T2_values_e = [T2 * 1e3 for T2 in T2_values_e]
    T2_values_C = [T2 * 1e3 for T2 in T2_values_C]
    T2_values_h = [T2 * 1e3 for T2 in T2_values_h]
    T2_values_N = [T2 * 1e3 for T2 in T2_values_N]

    # Calculate T2 product rule values
    T2_product_values = [T2_product_rule(Ta, Tb, Tc, Td) for Ta, Tb, Tc, Td in zip(T2_values_e, T2_values_C, T2_values_h, T2_values_N)]

    # Apply data range if provided
    if data_range is not None:
        start, end = data_range
        x_values = x_values[start:end]
        T2_values_e = T2_values_e[start:end]
        T2_values_C = T2_values_C[start:end]
        T2_values_h = T2_values_h[start:end]
        T2_values_N = T2_values_N[start:end]
        T2_product_values = T2_product_values[start:end]

    # Plotting
    plt.figure(figsize=(8, 4))
    plt.semilogy(x_values, T2_values_e, marker='o', label='E Bath')
    plt.semilogy(x_values, T2_values_C, marker='o', label='C Bath')
    plt.semilogy(x_values, T2_values_h, marker='o', label='H Bath')
    plt.semilogy(x_values, T2_values_N, marker='o', label='N Bath')
    plt.semilogy(x_values, T2_product_values, marker='o', label='Product Rule Result')
    # plt.title(r"T2 vs Numerical Key Combined")
    plt.xlabel(r"$B_0$ (Gauss)")
    plt.ylabel(r"$T_2$ ($\mu$s)")
    plt.ylim(bottom=0)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # Save and show plot
    output_filename = os.path.join(path, f"T2_vs_B0_Plotted_Individually_{AB}.png")
    plt.savefig(output_filename, dpi=300)
    plt.show()

AB_list = ['AB1', 'AB2', 'AB3', 'AB4', 'AB5', 'AB6', 'AB7']  # Add all your AB values here

for AB in AB_list:
    e_filename = f'[n-e]-(e)_{AB}.pkl_fit_results.pkl'
    C_filename = f'[n-e]-(n)_C_{AB}.pkl_fit_results.pkl'
    h_filename = f'[n-e]-(n)_H_{AB}.pkl_fit_results.pkl'
    N_filename = f'[n-e]-(n)_N_{AB}.pkl_fit_results.pkl'

    # Call the function to plot individual datasets and combined T2 product rule on semilog scale
    plot_T2_vs_field_combined(e_filename, C_filename, h_filename, N_filename, AB)

    # Call the function to plot only the combined T2 product rule on a normal scale
    plot_T2_product_rule_combined(e_filename, C_filename, h_filename, N_filename, AB)