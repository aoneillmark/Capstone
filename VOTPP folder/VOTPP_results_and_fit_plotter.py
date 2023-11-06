import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pickle
import os

def coherence_time_func(time, beta, T2):
    y = np.exp(-(((2*time)/T2)**beta)) # Is there a scaling issue here?
    return y

def load_data_from_file(path, pickle_filename):
    with open(os.path.join(path, pickle_filename), 'rb') as f:
        loaded_results = pickle.load(f)
    
    # Print the entire dictionary of loaded results
    print(loaded_results.keys())

    # Convert all coherence values to their absolute values
    for key in loaded_results:
        for sub_key in loaded_results[key]:
            loaded_results[key][sub_key][0] = np.abs(loaded_results[key][sub_key][0])
    
    return loaded_results

def plot_combined(loaded_results, variable_name, image_path):
    plt.figure(figsize=(10,6))
    
    for outer_key in loaded_results.keys():  # Loop over each key in the loaded_results
        for v_key, df in loaded_results[outer_key].items():  # Then loop over the items under this key
            label_str = ', '.join(map(str, v_key)) if isinstance(v_key, tuple) else f"Value {v_key}"
            plt.plot(df.index, df[0], label=label_str)

    plt.title(f"All {variable_name.capitalize()} Results")
    plt.xlabel('Time (ms)')
    plt.ylabel('Coherence')
    plt.legend()
    plt.tight_layout()
    plt.ylim(0,1)
    
    output_filename = os.path.join(image_path, f"{variable_name}_all_results.png")
    plt.savefig(output_filename, dpi=300)
    plt.show()

def save_fit_results(fit_results, result_name, save_path):
    """
    Save the fit results to a pickle file.
    """
    output_filename = os.path.join(save_path, "Fits/", f"{result_name}_fit_results.pkl")
    with open(output_filename, 'wb') as f:
        pickle.dump(fit_results, f)
    print("Saved fit results to file:")
    print(output_filename)

def plot_individual_with_fit(loaded_results, variable_name, image_path, pickle_path):
    fit_results = {}  # Dictionary to store fit parameters for all results
    
    for outer_key in loaded_results.keys():  # Loop over each key in the loaded_results
        for v_key, df in loaded_results[outer_key].items():  # Then loop over the items under this key
            plt.figure(figsize=(10,6))
            ydata = df[0]
            label_str = ', '.join(map(str, v_key)) if isinstance(v_key, tuple) else f"Value {v_key}"
            
            try:
                params, _ = curve_fit(coherence_time_func, df.index, ydata, maxfev=5000, bounds=([0, -np.inf], [np.inf, np.inf]), p0=[1,0.025])
                beta_fit, T2_fit = params
                
                fit_key = (outer_key,) + v_key if isinstance(v_key, tuple) else (outer_key, v_key)
                # Save the results in the dictionary
                fit_results[fit_key] = {'beta': beta_fit, 'T2': T2_fit} 
                
                plt.plot(df.index, ydata, 'o', label=f'Data for {label_str}')
                plt.plot(df.index, coherence_time_func(df.index, *params), '--', 
                         label=f'Fit: Beta={beta_fit:.3f}, T2={T2_fit:.3f}')
            except RuntimeError as e:
                print(f"Fit for label {label_str} failed: {e}")

            plt.title(f"{variable_name.capitalize()} Result for {label_str}")
            plt.xlabel('Time (ms)')
            plt.ylabel('Coherence')
            plt.legend()
            plt.tight_layout()
            plt.ylim(0,1)
            
            output_filename = os.path.join(image_path, f"{variable_name}_{label_str}_with_fit.png")
            plt.savefig(output_filename, dpi=300)
            plt.show()

    save_fit_results(fit_results, variable_name, pickle_path)  # Save the fit results once all fits are done


def plot_from_file(pickle_filenames):
    pickle_path = "VOTPP folder/Results/Pickle files/"
    image_path = "VOTPP folder/Results/Plots/"

    for pickle_filename in pickle_filenames:
        variable_name = pickle_filename.split('_results.pkl')[0]
        loaded_results = load_data_from_file(pickle_path, pickle_filename)
        
        # Plot all results combined without fit
        plot_combined(loaded_results, variable_name, image_path)
        
        # Plot individual results with fit
        plot_individual_with_fit(loaded_results, variable_name, image_path, pickle_path)

# Provide the filenames you want to process as a list
# plot_from_file(['magnetic_results.pkl',])

# plot_from_file(['magnetic_results_e_n.pkl',])
# plot_from_file(['magnetic_results_n_n.pkl',])
plot_from_file(['magnetic_results_[n-e]_e.pkl',])

# plot_from_file(['alphabeta_results_0.pkl', 'alphabeta_results_1.pkl', 'alphabeta_results_2.pkl', 'alphabeta_results_3.pkl', ])
# plot_from_file(['alphabeta_results.pkl',])
# plot_from_file(['alphabeta_order2_results.pkl',])
# plot_from_file(['alphabeta_combination_results.pkl',])

# plot_from_file(['magnetic_results.pkl', 'r_bath_results.pkl', 'r_dipole_results.pkl'])
# plot_from_file(['r_bath_results.pkl'])
# plot_from_file(['r_dipole_results.pkl'])
# plot_from_file(['cell_size_results.pkl'])

# plot_from_file(['r_bath_results.pkl'])
# plot_from_file(['r_dipole_results.pkl'])
# plot_from_file(['cell_size_results.pkl'])