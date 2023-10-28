import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pickle
import os

def coherence_time_func(time, beta, T2):
    y = np.exp(-(((time)/T2)**beta))
    return y

def load_data_from_file(path, pickle_filename):
    with open(os.path.join(path, pickle_filename), 'rb') as f:
        loaded_results = pickle.load(f)
    
    # Convert all coherence values to their absolute values
    for key in loaded_results:
        for sub_key in loaded_results[key]:
            loaded_results[key][sub_key][0] = np.abs(loaded_results[key][sub_key][0])
            
    return loaded_results

def plot_combined(loaded_results, variable_name, image_path):
    plt.figure(figsize=(10,6))
    for v_key, df in loaded_results[0].items():
        label_str = ', '.join(map(str, v_key)) if isinstance(v_key, tuple) else f"Value {v_key}"
        plt.plot(df.index, df[0], label=label_str)

    plt.title(f"All {variable_name.capitalize()} Results")
    plt.xlabel('Time')
    plt.ylabel('Coherence')
    plt.legend()
    plt.tight_layout()
    
    output_filename = os.path.join(image_path, f"{variable_name}_all_results.png")
    plt.savefig(output_filename, dpi=300)
    plt.show()

def plot_individual_with_fit(loaded_results, variable_name, image_path):
    for v_key, df in loaded_results[0].items():
        plt.figure(figsize=(10,6))
        ydata = df[df.index <= 0.1][0]
        label_str = ', '.join(map(str, v_key)) if isinstance(v_key, tuple) else f"Value {v_key}"
        
        try:
            params, _ = curve_fit(coherence_time_func, df.index, ydata, maxfev=5000, bounds=([0, -np.inf], [np.inf, np.inf]))
            beta_fit, T2_fit = params
            plt.plot(df.index, ydata, 'o', label=f'Data for {label_str}')
            plt.plot(df.index, coherence_time_func(df.index, *params), '--', 
                     label=f'Fit: Beta={beta_fit:.3f}, T2={T2_fit:.3f}')
        except RuntimeError as e:
            print(f"Fit for label {label_str} failed: {e}")

        plt.title(f"{variable_name.capitalize()} Result for {label_str}")
        plt.xlabel('Time')
        plt.ylabel('Coherence')
        plt.legend()
        plt.tight_layout()
        
        output_filename = os.path.join(image_path, f"{variable_name}_{label_str}_with_fit.png")
        plt.savefig(output_filename, dpi=300)
        plt.show()

def plot_from_file(pickle_filenames):
    path = "VOTPP folder/Results/Pickle files/"
    image_path = "VOTPP folder/Results/Plots/"

    for pickle_filename in pickle_filenames:
        variable_name = pickle_filename.split('_results.pkl')[0]
        loaded_results = load_data_from_file(path, pickle_filename)
        
        # Plot all results combined without fit
        plot_combined(loaded_results, variable_name, image_path)
        
        # Plot individual results with fit
        plot_individual_with_fit(loaded_results, variable_name, image_path)

# Provide the filenames you want to process as a list
# plot_from_file(['magnetic_results.pkl', 'r_bath_results.pkl', 'r_dipole_results.pkl'])
plot_from_file(['magnetic_results.pkl',])
