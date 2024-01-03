import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pickle
import os
import pandas as pd


def check_pickle_structure(pickle_path, pickle_filename):
    with open(os.path.join(pickle_path, pickle_filename), 'rb') as f:
        loaded_results = pickle.load(f)

    print("Keys in loaded results:", loaded_results.keys())

    for variable, seeds_data in loaded_results.items():
        print(f"\nVariable: {variable}")
        for seed, data in seeds_data.items():
            print(f"  Seed: {seed}")
            for sub_key, df in data.items():
                print(f"    Sub_key: {sub_key}, Type: {type(df)}, Structure: {type(df[0]) if isinstance(df, (list, tuple)) else 'N/A'}")


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

def print_series_before_averaging(loaded_results, filename="full_series_data.txt"):
    with open(filename, 'w') as file:
        for seed, data in loaded_results.items():
            file.write(f"Seed: {seed}\n")
            for sub_key, series in data.items():
                file.write(f"  Sub_key: {sub_key}, Series data:\n{series.to_string()}\n\n")

def average_results_by_seed(loaded_results, debug=False):
    averaged_results = {}

    # Print series to a file if debug is True
    if debug:
        print_series_before_averaging(loaded_results)

    for seed, data in loaded_results.items():
        for sub_key, series in data.items():
            if sub_key not in averaged_results:
                averaged_results[sub_key] = []

            # Replace '--' with NaN
            cleaned_series = series.replace('--', np.nan).astype(float)
            averaged_results[sub_key].append(cleaned_series)

    for sub_key, series_list in averaged_results.items():
        if all(len(s) == len(series_list[0]) for s in series_list):
            df = pd.concat(series_list, axis=1)
            averaged_results[sub_key] = df.mean(axis=1)  # This should be a Series
        else:
            print(f"Data length mismatch for sub_key {sub_key}")
            averaged_results[sub_key] = series_list[0]

    return averaged_results



def plot_combined(loaded_results, variable_name, image_path, ylim=None):
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
    plt.ylim(ylim)
    
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

def plot_individual_with_fit(loaded_results, variable_name, image_path, pickle_path, data_range=slice(None), ylim=None):
    fit_results = {}

    for outer_key in loaded_results.keys():
        for v_key, df in loaded_results[outer_key].items():
            plt.figure(figsize=(10, 6))
            ydata_series = pd.Series(df[0].iloc[data_range]).replace('--', np.nan).astype(float)

            # Create a mask for valid data points (not NaN and finite)
            valid_mask = ~ydata_series.isna() & np.isfinite(ydata_series)

            # Data for plotting and fitting
            ydata_plot = ydata_series[valid_mask]
            time_data = df.index[data_range][valid_mask]

            label_str = ', '.join(map(str, v_key)) if isinstance(v_key, tuple) else f"Value {v_key}"

            try:
                params, _ = curve_fit(coherence_time_func, time_data, ydata_plot, maxfev=5000, bounds=([0, -np.inf], [np.inf, np.inf]), p0=[1, 0.025])
                beta_fit, T2_fit = params

                fit_key = (outer_key,) + v_key if isinstance(v_key, tuple) else (outer_key, v_key)
                fit_results[fit_key] = {'beta': beta_fit, 'T2': T2_fit}

                plt.plot(time_data, ydata_plot, 'o', label=f'Data for {label_str}')
                plt.plot(time_data, coherence_time_func(time_data, *params), '--', label=f'Fit: Beta={beta_fit:.3f}, T2={T2_fit:.3f}')
            except RuntimeError as e:
                print(f"Fit for label {label_str} failed: {e}")

            plt.title(f"{variable_name.capitalize()} Result for {label_str}")
            plt.xlabel('Time (ms)')
            plt.ylabel('Coherence')
            plt.legend()
            plt.tight_layout()
            plt.ylim(ylim)

            output_filename = os.path.join(image_path, f"{variable_name}_{label_str}_with_fit.png")
            plt.savefig(output_filename, dpi=300)
            plt.show()

    save_fit_results(fit_results, variable_name, pickle_path)


    save_fit_results(fit_results, variable_name, pickle_path)

def plot_individual_with_fit_average(averaged_results, variable_name, image_path, pickle_path, data_range=slice(None), ylim=None):
    fit_results = {}  # Dictionary to store fit parameters for all results
    
    for sub_key, series in averaged_results.items():  # Now iterating over averaged results
        plt.figure(figsize=(10,6))
        # Interpolate to fill NaN values
        ydata = series.interpolate().iloc[data_range]  
        time_data = series.index[data_range]
        
        label_str = ', '.join(map(str, sub_key)) if isinstance(sub_key, tuple) else f"Value {sub_key}"
        
        try:
            params, _ = curve_fit(coherence_time_func, time_data, ydata, maxfev=5000, bounds=([0, -np.inf], [np.inf, np.inf]), p0=[0.5,0.025])
            beta_fit, T2_fit = params
            
            fit_key = sub_key
            fit_results[fit_key] = {'beta': beta_fit, 'T2': T2_fit}
            
            plt.plot(time_data, ydata, 'o', label=f'Data for {label_str}')
            plt.plot(time_data, coherence_time_func(time_data, *params), '--', 
                     label=f'Fit: Beta={beta_fit:.3f}, T2={T2_fit:.3f}')
        except RuntimeError as e:
            print(f"Fit for label {label_str} failed: {e}")

        plt.title(f"{variable_name.capitalize()} Result for {label_str}")
        plt.xlabel('Time (ms)')
        plt.ylabel('Coherence')
        plt.legend()
        plt.tight_layout()
        plt.ylim(ylim)
        
        output_filename = os.path.join(image_path, f"{variable_name}_{label_str}_with_fit.png")
        plt.savefig(output_filename, dpi=300)
        plt.show()

    save_fit_results(fit_results, variable_name, pickle_path)



def plot_combined_average(averaged_results, variable_name, image_path, ylim=None):
    plt.figure(figsize=(10,6))
    
    for sub_key, series in averaged_results.items():
        label_str = ', '.join(map(str, sub_key)) if isinstance(sub_key, tuple) else f"Value {sub_key}"
        plt.plot(series.index, series.values, label=label_str)  # series is a pandas Series

    plt.title(f"All {variable_name.capitalize()} Results")
    plt.xlabel('Time (ms)')
    plt.ylabel('Coherence')
    plt.legend()
    plt.tight_layout()
    plt.ylim(ylim)
    
    output_filename = os.path.join(image_path, f"{variable_name}_all_results.png")
    plt.savefig(output_filename, dpi=300)
    plt.show()


def plot_from_file(pickle_filenames, data_range=slice(None), ylim=None):
    pickle_path = "VOTPP folder/Results/Pickle files/"
    image_path = "VOTPP folder/Results/Plots/"

    for pickle_filename in pickle_filenames:
        variable_name = pickle_filename.split('_results.pkl')[0]
        loaded_results = load_data_from_file(pickle_path, pickle_filename)
        
        # Plot all results combined without fit
        plot_combined(loaded_results, variable_name, image_path, ylim=ylim)
        
        # Plot individual results with fit
        plot_individual_with_fit(loaded_results, variable_name, image_path, pickle_path, data_range=data_range, ylim=ylim)

def plot_from_file_average(pickle_filenames, data_range=slice(None), ylim=None):
    pickle_path = "VOTPP folder/Results/Pickle files/"
    image_path = "VOTPP folder/Results/Plots/"

    for pickle_filename in pickle_filenames:
        variable_name = pickle_filename.split('_results.pkl')[0]
        loaded_results = load_data_from_file(pickle_path, pickle_filename)

        # Average results by seed
        averaged_results = average_results_by_seed(loaded_results, debug=False)

        # Plot all results combined without fit, using the plot_combined_average function
        plot_combined_average(averaged_results, variable_name, image_path, ylim=ylim)

        # Plot individual results with fit
        plot_individual_with_fit_average(averaged_results, variable_name, image_path, pickle_path, data_range=data_range, ylim=ylim)



# Provide the filenames you want to process as a list
# plot_from_file(['magnetic_results.pkl',])
# plot_from_file(['magnetic_results.pkl'], data_range=slice(0, 40))

# # Modify the path and filename to match your file's location
# check_pickle_structure("VOTPP folder/Results/Pickle files/", "magnetic_results.pkl")

# plot_combined(load_data_from_file("VOTPP folder/Results/Pickle files/", "magnetic_results.pkl"), "magnetic", "VOTPP folder/Results/Plots/")

# For plotting with averaging
# plot_from_file_average(['magnetic_results.pkl'])#)#, data_range=slice(0, 100))


# plot_from_file(['magnetic_results_e_n.pkl',])
# plot_from_file(['magnetic_results_n_n.pkl',])
# plot_from_file(['magnetic_results_[n-e]_e.pkl',])
# plot_from_file(['magnetic_results_e_e.pkl',])
# plot_from_file(['magnetic_results_n_e.pkl',])

# plot_from_file(['magnetic_results_[n-e]_e_N_trans.pkl',]) #!!!!

# plot_from_file(['magnetic_results_[n-e]_e_E_trans.pkl',])
# plot_from_file(['[n-e]-(e).pkl',])
# plot_from_file(['[n-e]-(n).pkl',])
plot_from_file(['magnetic_results_0.pkl'])
# plot_from_file(['[n-e]-(e)_r_dipole_results.pkl'])
# plot_from_file(['[n-e]-(e)_r_bath_results.pkl'])
# plot_from_file_average(['[n-e]-(e).pkl'])

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
        
# plot_from_file(['magnetic_results_combined_bath.pkl'], data_range=slice(0, 100))