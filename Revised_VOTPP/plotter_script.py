import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import pickle

def coherence_time_func(time, beta, T2):
    """Coherence decay function for curve fitting."""
    return np.exp(-((2 * time) / T2) ** beta)

def get_colorblind_friendly_colors():
    # Custom color palette with shades from blue to red
    return [
        "#1f77b4",  # Muted blue
        "#ff7f0e",  # Safety orange
        "#2ca02c",  # Cooked asparagus green
        "#d62728",  # Brick red
        "#9467bd",  # Muted purple
        "#8c564b",  # Chestnut brown
        "#e377c2",  # Raspberry yogurt pink
        "#7f7f7f",  # Middle gray
        "#bcbd22",  # Curry yellow-green
        "#17becf"   # Blue-teal
    ]

def plot_individual_with_fit(loaded_results, variable_name, image_path, data_range=slice(None), ylim=None, show=False, filename_suffix=''):
    """
    Plot individual simulation results with curve fitting.

    Parameters:
        loaded_results: dict, loaded simulation results for individual variables.
        variable_name: str, the name of the variable being analyzed (e.g., 'r_bath', 'magnetic_field').
        image_path: str, the path where the plots should be saved.
        data_range: slice, the range of data points to consider for plotting and fitting.
        ylim: tuple, limits for the y-axis.
        show: bool, whether to display the plots.
        filename_suffix: str, additional text to be appended to the filename for saving the plot.
    """
    for key, results in loaded_results.items():
        # Create a new figure for each set of results
        plt.figure(figsize=(8, 4))

        times = np.linspace(0, 0.1, len(results))  # !!! To do!
        coherences = np.array(results)

        # Apply data range for fitting and plotting
        times = times[data_range]
        coherences = coherences[data_range]

        try:
            # Perform curve fitting
            params, cov = curve_fit(coherence_time_func, times, coherences, p0=[1, 0.1], bounds=(0, [2., 10.]))
            fitted_curve = coherence_time_func(times, *params)
            beta = params[0]
            T2_fit = params[1]
            beta_err, T2_err = np.sqrt(np.diag(cov))  # Calculate the standard deviation
            # Cap the error
            T2_err_capped = min(T2_err, 50e-3)  # Cap the error at 50 μs

            # Plot original data and fitted curve
            plt.plot(times, coherences, 'o', label=f'Original Data ({variable_name} = {key})')
            # plt.plot(times, fitted_curve, '-', label=f'Fitted Curve (beta={params[0]:.2f}, T2={params[1]:.2f})')

            plt.plot(times, fitted_curve, '--', label=f'Fit: T_2 = {(T2_fit*1e3):.0f}±{(T2_err*1e3):.0f} μs')

        except RuntimeError as e:
            print(f"Curve fitting failed for {variable_name} = {key}: {e}")
            # Plot original data even if fitting fails
            plt.plot(times, coherences, 'o', label=f'Original Data ({variable_name} = {key})')

        # Finalize the plot
        plt.title(f'{variable_name.capitalize()} = {key}')
        plt.xlabel('Time (ms)')
        plt.ylabel('Coherence')
        plt.legend()
        plt.tight_layout()
        plt.ylim(ylim)

        # Save the plot
        output_filename = os.path.join(image_path, f"{variable_name}_{key}_with_fit{filename_suffix}.png")
        plt.savefig(output_filename, dpi=300)
        if show:
            plt.show()
        plt.close()  # Close the plot to free up memory

def plot_combined(loaded_results, variable_name, image_path, ylim=None, show=False, filename=None):
    """
    Plot combined results from loaded simulation data.
    
    Parameters:
        loaded_results: dict, simulation results loaded from pickle file.
        variable_name: str, name of the variable being varied (e.g., 'r_bath', 'magnetic_field').
        image_path: str, path to save the generated plot.
        ylim: tuple, y-axis limits for the plot.
        show: bool, whether to display the plot.
        filename: str, name of the output file (without extension).
    """
    colors = get_colorblind_friendly_colors()
    plt.figure(figsize=(8, 4))

    color_index = 0  # Initialize color index for plotting
    for key, results in loaded_results.items():
        times = np.linspace(0, 0.1, len(results))  # !!! need to change this to take timespace from the results or the other script
        coherences = results

        color = colors[color_index % len(colors)]
        plt.plot(times, coherences, '-', label=f"{variable_name} = {key}", color=color, linewidth=1, alpha=0.75)
        plt.plot(times, coherences, 'o', color=color, markersize=3)  # Smaller points

        color_index += 1  # Increment color index for next variable value

    plt.xlabel('Time (ms)')
    plt.ylabel('Coherence')
    plt.legend()
    plt.tight_layout()
    plt.ylim(ylim)

    final_filename = os.path.join(image_path, f"{filename or variable_name}.png")
    plt.savefig(final_filename, dpi=300)
    if show:
        plt.show()
    else:
        plt.close()


def load_data_from_file(pickle_filename, pickle_path):
    """
    Load simulation data from a pickle file.
    
    Parameters:
        pickle_filename: str, name of the pickle file to load.
        pickle_path: str, path to the directory containing the pickle file.
    
    Returns:
        dict: Loaded simulation results.
    """
    with open(os.path.join(pickle_path, pickle_filename), 'rb') as f:
        loaded_results = pickle.load(f)
    return loaded_results

########################################################################################
# Usage example
pickle_path = "./Revised_VOTPP/Results_revised/Pickle_files/Convergence/Final"
pickle_filename = "electronic_r_bath.pkl"  # Example filename
loaded_results = load_data_from_file(pickle_filename, pickle_path)

plot_combined(loaded_results, "r_bath", "./Revised_VOTPP/Results_revised/Plots/Convergence", ylim=(0, 1), show=True, filename="electronic_r_bath_combined")
plot_individual_with_fit(loaded_results, "r_bath", "./Revised_VOTPP/Results_revised/Plots/Convergence", data_range=slice(0, 100), ylim=(0, 1), show=True, filename_suffix="_100points")

# Another example
pickle_path = "./Revised_VOTPP/Results_revised/Pickle_files/Simulation"
pickle_filename = "electronic_AB1_alpha2_beta3.pkl"
loaded_results = load_data_from_file(pickle_filename, pickle_path)

plot_combined(loaded_results, "magnetic_field", "./Revised_VOTPP/Results_revised/Plots/Simulation", ylim=(0, 1), show=True, filename="electronic_AB1_alpha2_beta3_combined")
plot_individual_with_fit(loaded_results, "magnetic_field", "./Revised_VOTPP/Results_revised/Plots/Simulation", data_range=slice(0, 100), ylim=(0, 1), show=True, filename_suffix="_100points")

# To do for Mark: need to make it so that this script gets the timespace which is used in the runner script 
# To do for Mark: Make sure the formatting of the plots are nice and consistent
# To do for Mark: Save the fit results for later analysis of T_2 vs Magnetic Field