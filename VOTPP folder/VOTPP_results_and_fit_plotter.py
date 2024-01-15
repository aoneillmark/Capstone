import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pickle
import os
import pandas as pd

def coherence_time_func(time, beta, T2):
    y = np.exp(-(((2*time)/T2)**1)) # Is there a scaling issue here?
    return y


def get_colorblind_friendly_colors():
    # Custom color palette with shades from blue to red
    colors = [
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
    return colors



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

def load_data_from_file(pickle_filename):
    path = "VOTPP folder/Results/Pickle files 2/Simulation results/"
    with open(os.path.join(path, pickle_filename), 'rb') as f:
        loaded_results = pickle.load(f)
    
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



def plot_combined(loaded_results, variable_name, image_path, ylim=None, show=False):
    colors = get_colorblind_friendly_colors()  # Get colorblind-friendly colors
    plt.figure(figsize=(8,4))
    
    color_index = 0  # Initialize a separate color index
    for outer_key, data in loaded_results.items():
        for v_key, df in data.items():
            color = colors[color_index % len(colors)]  # Cycle through colors using color_index
            label_str = ', '.join(map(str, v_key)) if isinstance(v_key, tuple) else f"Value {v_key}"
            plt.plot(df.index, df[0], label=f"B_0 = {v_key[2]}", color=color)
            color_index += 1  # Increment color_index for each v_key

    plt.title(f"All {variable_name.capitalize()} Results")
    plt.xlabel(r'2$\tau$ (ms)')
    plt.ylabel('Coherence')
    plt.legend()
    plt.tight_layout()
    plt.ylim(ylim)
    
    output_filename = os.path.join(image_path, f"{variable_name}_all_results.png")
    plt.savefig(output_filename, dpi=300)
    if show == True:
        plt.show()
    else:
        plt.close()

def save_fit_results(fit_results, result_name):
    """
    Save the fit results to a pickle file.
    """
    save_path = "VOTPP folder/Results/Pickle files 2/Fits/"
    output_filename = os.path.join(save_path, f"{result_name}_fit_results.pkl")
    with open(output_filename, 'wb') as f:
        pickle.dump(fit_results, f)
    print("Saved fit results to file:", output_filename)


def plot_individual_with_fit(loaded_results, variable_name, image_path, data_range=slice(None), ylim=None, show=False):
    fit_results = {}

    for outer_key in loaded_results.keys():
        for v_key, df in loaded_results[outer_key].items():
            plt.figure(figsize=(8,4))
            # plt.figure(figsize=(4, 3))
            ydata_series = pd.Series(df[0].iloc[data_range]).replace('--', np.nan).astype(float)
            valid_mask = ~ydata_series.isna() & np.isfinite(ydata_series)
            ydata_plot = ydata_series[valid_mask]
            time_data = df.index[data_range][valid_mask]

            label_str = ', '.join(map(str, v_key)) if isinstance(v_key, tuple) else f"Value {v_key}"

            try:
                params, cov = curve_fit(coherence_time_func, time_data, ydata_plot, maxfev=5000, bounds=([0, -np.inf], [np.inf, np.inf]), p0=[1, 0.060])
                beta_fit, T2_fit = params
                beta_err, T2_err = np.sqrt(np.diag(cov))  # Standard deviation (error) of the fit parameters

                fit_key = (outer_key,) + v_key if isinstance(v_key, tuple) else (outer_key, v_key)
                fit_results[fit_key] = {'beta': beta_fit, 'T2': T2_fit, 'beta_err': beta_err, 'T2_err': T2_err}

                # plt.plot(time_data, ydata_plot, 'o', label=f'Data for B_0 = {v_key[2]} G')
                # plt.plot(time_data, coherence_time_func(time_data, *params), '--', label=f'Fit: T_2 = {T2_fit:.3f}±{T2_err:.3f} ms')
                plt.plot(time_data, ydata_plot, 'o', label=f'Data for B_0 = {(v_key[2] * 1e-4):.2f} T')
                plt.plot(time_data, coherence_time_func(time_data, *params), '--', label=f'Fit: T_2 = {(T2_fit*1e3):.0f}±{(T2_err*1e3):.0f} μs')
            except RuntimeError as e:
                print(f"Fit for label {label_str} failed: {e}")

            # plt.title(f"{variable_name.capitalize()} Result for {label_str}")
            plt.xlabel(r'2$\tau$ (ms)')
            plt.ylabel('Coherence')
            plt.legend()
            plt.tight_layout()
            plt.ylim(ylim)

            output_filename = os.path.join(image_path, f"{variable_name}_{label_str}_with_fit.png")
            plt.savefig(output_filename, dpi=300)
            if show == True:
                plt.show()
            else:
                plt.close()

    save_fit_results(fit_results, variable_name,)


def plot_individual_with_fit_average(averaged_results, variable_name, image_path, pickle_path, data_range=slice(None), ylim=None, show=False):
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
        if show == True:
            plt.show()
        else:
            plt.close()

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
    image_path = "VOTPP folder/Results/Plots 2/"

    for pickle_filename in pickle_filenames:
        variable_name = pickle_filename.split('_results.pkl')[0]
        loaded_results = load_data_from_file(pickle_filename)

        # Plot all results combined without fit
        plot_combined(loaded_results, variable_name, image_path, ylim=ylim)
        
        # Plot individual results with fit
        plot_individual_with_fit(loaded_results, variable_name, image_path, data_range=data_range, ylim=ylim)

def plot_from_file_average(pickle_filenames, data_range=slice(None), ylim=None):
    image_path = "VOTPP folder/Results/Plots 2/"

    for pickle_filename in pickle_filenames:
        variable_name = pickle_filename.split('_results.pkl')[0]
        loaded_results = load_data_from_file(pickle_filename)

        # Average results by seed
        averaged_results = average_results_by_seed(loaded_results, debug=False)

        # Plot all results combined without fit, using the plot_combined_average function
        plot_combined_average(averaged_results, variable_name, image_path, ylim=ylim)

        # Plot individual results with fit
        plot_individual_with_fit_average(averaged_results, variable_name, image_path, pickle_path, data_range=data_range, ylim=ylim)


##############################################################################################################
# Provide the filenames you want to process as a list
# plot_from_file(['magnetic_results.pkl',])
# plot_from_file(['magnetic_results.pkl'], data_range=slice(0, 40))

# # Modify the path and filename to match your file's location
# check_pickle_structure("VOTPP folder/Results/Pickle files/", "magnetic_results.pkl")

# plot_combined(load_data_from_file("VOTPP folder/Results/Pickle files/", "magnetic_results.pkl"), "magnetic", "VOTPP folder/Results/Plots/")

# For plotting with averaging
# plot_from_file_average(['magnetic_results.pkl'])#)#, data_range=slice(0, 100))
##############################################################################################################



# plot_from_file(['[n-e]-(e).pkl',],ylim=(-0.01,None))#, data_range=slice(0, 225), )
# plot_from_file(['[n-e]-(n).pkl',])

# AB_list = ['AB1', 'AB2', 'AB3', 'AB4', 'AB5', 'AB6', 'AB7']
# for AB in AB_list:
#     plot_from_file([f'[n-e]-(e)_{AB}.pkl',], ylim=(-0.01,None))
# # plot_from_file(['[n-e]-(e)_AB1.pkl',])
# # plot_from_file(['[n-e]-(e)_AB2.pkl',])
# # plot_from_file(['[n-e]-(e)_AB3.pkl',])
# # plot_from_file(['[n-e]-(e)_AB4.pkl',])
# # plot_from_file(['[n-e]-(e)_AB5.pkl',])
# # plot_from_file(['[n-e]-(e)_AB6.pkl',])
# # plot_from_file(['[n-e]-(e)_AB7.pkl',])

# bath_type_list = ['C', 'N', 'H']
# AB_list = ['AB1', 'AB2', 'AB3', 'AB4', 'AB5', 'AB6', 'AB7']
# for bath_type in bath_type_list:
#     for AB in AB_list:
#         plot_from_file([f'[n-e]-(n)_{bath_type}_{AB}.pkl',], ylim=(-0.01,None))
#     # plot_from_file([f'[n-e]-(n)_{bath_type}_AB1.pkl',], ylim=(-0.01,None))
#     # plot_from_file([f'[n-e]-(n)_{bath_type}_AB2.pkl',], ylim=(-0.01,None))
#     # plot_from_file([f'[n-e]-(n)_{bath_type}_AB3.pkl',], ylim=(-0.01,None))
#     # plot_from_file([f'[n-e]-(n)_{bath_type}_AB4.pkl',], ylim=(-0.01,None))
#     # plot_from_file([f'[n-e]-(n)_{bath_type}_AB5.pkl',], ylim=(-0.01,None))
#     # plot_from_file([f'[n-e]-(n)_{bath_type}_AB6.pkl',], ylim=(-0.01,None))
#     # plot_from_file([f'[n-e]-(n)_{bath_type}_AB7.pkl',], ylim=(-0.01,None))

# plot_from_file(['[n-e]-(n)_C_AB3.pkl',], ylim=(-0.01,None))
# plot_from_file(['[n-e]-(n)_N_AB3.pkl',], ylim=(-0.01,None))
# plot_from_file(['[n-e]-(n)_H_AB3.pkl',], ylim=(-0.01,None))
plot_from_file(['[n-e]-(e)_AB3.pkl',], ylim=(-0.01,None),)

##############################################################################################################
# H bath convergence
# plot_from_file(['[n-e]-(n)_r_bath_order2_H_results.pkl'], ylim=(-0.01,None))
# plot_from_file(['[n-e]-(n)_r_dipole_order2_H_results.pkl'], ylim=(-0.01,None))

# C bath convergence
# plot_from_file(['[n-e]-(n)_r_bath_order2_C_results.pkl'], ylim=(-0.01,1.01))
# plot_from_file(['[n-e]-(n)_r_dipole_order2_C_results.pkl'], ylim=(-0.01,1.01))
# plot_from_file(['[n-e]-(n)_order_C_results.pkl'], ylim=(-0.01,1.01))

# N bath convergence 
# plot_from_file(['[n-e]-(n)_r_bath_order2_N_results.pkl'], ylim=(-0.01,None))
# plot_from_file(['[n-e]-(n)_r_dipole_order2_N_results.pkl'], ylim=(-0.01,None))
# plot_from_file(['[n-e]-(n)_order_N_results.pkl'],)# ylim=(-0.01,1.01))
##############################################################################################################

# plot_from_file(['[n-e]-(e)_r_dipole_results.pkl'])
        
# plot_from_file(['[n-e]-(e)_r_dipole_order3_results.pkl'])
# plot_from_file(['[n-e]-(e)_r_bath_order3_results.pkl'])

# plot_from_file(['[n-e]-(e)_r_dipole_order2_results.pkl'])
# plot_from_file(['[n-e]-(e)_r_bath_order2_results.pkl'])

# plot_from_file(['[n-e]-(n)_r_bath_order2_results.pkl'])
# plot_from_file(['[n-e]-(n)_r_dipole_order2_results.pkl'])
# plot_from_file(['[n-e]-(n)_order_results.pkl'])

# plot_from_file(['[n-e]-(e)_r_bath_results.pkl'])
# plot_from_file_average(['[n-e]-(e).pkl'])

