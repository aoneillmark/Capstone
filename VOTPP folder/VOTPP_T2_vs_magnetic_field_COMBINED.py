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

def plot_T2_product_rule_combined(e_pickle_filename, C_pickle_filename, h_pickle_filename, N_pickle_filename, AB, data_range=None, show=False):
    path = "VOTPP folder/Results/Pickle files 2/Fits/"

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
    T2_values_e = [e_fit_results[key]['T2'] *1e3 for key in keys]
    T2_values_C = [C_fit_results[key]['T2'] *1e3 for key in keys]
    T2_values_h = [h_fit_results[key]['T2'] *1e3 for key in keys]
    T2_values_N = [N_fit_results[key]['T2'] *1e3 for key in keys]

    

    # Extract T2 errors for all datasets
    T2_errors_e = [e_fit_results[key]['T2_err'] * 1e3 for key in keys]  # Convert error from ms to microseconds
    T2_errors_C = [C_fit_results[key]['T2_err'] * 1e3 for key in keys]
    T2_errors_h = [h_fit_results[key]['T2_err'] * 1e3 for key in keys]
    T2_errors_N = [N_fit_results[key]['T2_err'] * 1e3 for key in keys]

    # print("T2 values e:", T2_values_e)
    # print("T2 values C:", T2_values_C)
    # print("T2 values h:", T2_values_h)
    # print("T2 values N:", T2_values_N)


    # print("T2 errors e:", T2_errors_e)
    # print("T2 errors C:", T2_errors_C)
    # print("T2 errors H:", T2_errors_h)
    # print("T2 errors N:", T2_errors_N)


    # Calculate T2 product rule values and errors
    T2_product_values = []
    T2_product_errors = []
    for Ta, Tb, Tc, Td, err_a, err_b, err_c, err_d in zip(T2_values_e, T2_values_C, T2_values_h, T2_values_N, T2_errors_e, T2_errors_C, T2_errors_h, T2_errors_N):
        T2_product = T2_product_rule(Ta, Tb, Tc, Td)
        T2_product_values.append(T2_product)
        T2_product_error = T2_product**2 * np.sqrt((err_a/Ta**2)**2 + (err_b/Tb**2)**2 + (err_c/Tc**2)**2 + (err_d/Td**2)**2)
        T2_product_errors.append(T2_product_error)

    # T2_product_values = [i *1e3 for i in T2_product_values]

    # # T2_product_errors = [i *1e-3 for i in T2_product_errors]
    # print(T2_product_errors)

    # Apply data range if provided
    if data_range is not None:
        start, end = data_range
        x_values = x_values[start:end]
        T2_product_values = T2_product_values[start:end]

    # Plotting the combined T2 product rule values
    plt.figure(figsize=(8, 4))
    x_values = [i*1e-4 for i in x_values]
    plt.errorbar(x_values, T2_product_values, yerr=T2_product_errors, fmt='-o', label='Product Rule Result', 
                 ecolor='black', elinewidth=0.5, capsize=2)
    # plt.title("T2 Product Rule Combined vs Numerical Key")
    plt.xlabel(r"$B_0$ (T)")
    plt.ylabel(r"$T_2$ ($\mu$s)")
    plt.ylim(bottom=0)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # Save plot in the "Results/T2_vs_B" folder
    save_path = "VOTPP folder/Results/T2_vs_B/"
    output_filename = os.path.join(save_path, f"T2_vs_B0_Product_Result_{AB}.png")
    plt.savefig(output_filename, dpi=300)
    if show == True:
        plt.show()
    else:
        plt.close()


def plot_T2_vs_field_combined(e_pickle_filename, C_pickle_filename, h_pickle_filename, N_pickle_filename, AB, data_range=None, show=False):
    path = "VOTPP folder/Results/Pickle files 2/Fits/"

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

    # Extract T2 errors for all datasets
    T2_errors_e = [e_fit_results[key]['T2_err'] * 1e3 for key in keys]  # Convert error from ms to microseconds
    T2_errors_C = [C_fit_results[key]['T2_err'] * 1e3 for key in keys]
    T2_errors_h = [h_fit_results[key]['T2_err'] * 1e3 for key in keys]
    T2_errors_N = [N_fit_results[key]['T2_err'] * 1e3 for key in keys]

    # Calculate T2 product rule values and errors
    T2_product_values = []
    T2_product_errors = []
    for Ta, Tb, Tc, Td, err_a, err_b, err_c, err_d in zip(T2_values_e, T2_values_C, T2_values_h, T2_values_N, T2_errors_e, T2_errors_C, T2_errors_h, T2_errors_N):
        T2_product = T2_product_rule(Ta, Tb, Tc, Td)
        T2_product_values.append(T2_product)
        T2_product_error = T2_product**2 * np.sqrt((err_a/Ta**2)**2 + (err_b/Tb**2)**2 + (err_c/Tc**2)**2 + (err_d/Td**2)**2)
        T2_product_errors.append(T2_product_error)


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
        T2_errors_e = T2_errors_e[start:end]
        T2_errors_C = T2_errors_C[start:end]
        T2_errors_h = T2_errors_h[start:end]
        T2_errors_N = T2_errors_N[start:end]
        T2_product_values = T2_product_values[start:end]

    # Plotting with error bars
    plt.figure(figsize=(8, 4))
    x_values = [i*1e-4 for i in x_values]
    plt.errorbar(x_values, T2_values_e, yerr=T2_errors_e, fmt='-o', color='#3577b4', label='E Bath', 
                 ecolor='black', elinewidth=0.5, capsize=2)
    plt.errorbar(x_values, T2_values_C, yerr=T2_errors_C, fmt='-o', color='#d72f26', label='C Bath', 
                 ecolor='black', elinewidth=0.5, capsize=2)
    plt.errorbar(x_values, T2_values_h, yerr=T2_errors_h, fmt='-o',color='#499e2f', label='H Bath', 
                 ecolor='black', elinewidth=0.5, capsize=2)
    plt.errorbar(x_values, T2_values_N, yerr=T2_errors_N, fmt='-o', color='#9364bc', label='N Bath', 
                 ecolor='black', elinewidth=0.5, capsize=2)

    # Plotting the T2 product rule values on a semi-logarithmic scale
    plt.semilogy(x_values, T2_product_values, '-o', color='#ee7c30', label='Product Rule Result')  # Using semilogy for the product rule

    # plt.title(r"T2 vs Numerical Key Combined")
    plt.xlabel(r"$B_0$ (T)")
    plt.ylabel(r"$T_2$ ($\mu$s)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save plot in the "Results/T2_vs_B" folder
    save_path = "VOTPP folder/Results/T2_vs_B/"
    output_filename = os.path.join(save_path, f"T2_vs_B0_Plotted_Individually_{AB}.png")
    plt.savefig(output_filename, dpi=300)
    if show == True:
        plt.show()
    else:
        plt.close()




def plot_T2_vs_B_single(pickle_filename, bath_type, identifier, data_range=None, show=False):
    path = "VOTPP folder/Results/Pickle files 2/Fits/"

    # Load fit results for the dataset
    fit_results = load_fit_results(path, pickle_filename)

    # Assuming the keys are consistent
    keys = fit_results.keys()

    # Extract x_values (B field values) from the keys
    x_values = [key[3] for key in keys]

    # Extract T2 values and errors
    T2_values = [fit_results[key]['T2'] *1e3 for key in keys]  # Convert from ms to microseconds
    T2_errors = [fit_results[key]['T2_err'] *1e3 for key in keys]

    # Apply data range if provided
    if data_range is not None:
        start, end = data_range
        x_values = x_values[start:end]
        T2_values = T2_values[start:end]
        T2_errors = T2_errors[start:end]

    # Plotting with error bars
    plt.figure(figsize=(8, 4))
    x_values = [i*1e-4 for i in x_values]
    plt.errorbar(x_values, T2_values, yerr=T2_errors, fmt='-o', ecolor='black', elinewidth=0.5, capsize=2, label=f'{bath_type} Bath T2 vs B')

    plt.xlabel(r"$B_0$ (T)")
    plt.ylabel(r"$T_2$ ($\mu$s)")
    plt.ylim(bottom=0)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save plot in the "Results/T2_vs_B" folder with a descriptive filename
    save_path = "VOTPP folder/Results/T2_vs_B/"
    output_filename = os.path.join(save_path, f"{bath_type}_bath_{identifier}_T2_vs_B.png")
    plt.savefig(output_filename, dpi=300)
    if show == True:
        plt.show()
    else:
        plt.close()





def plot_all_T2_product_rules_combined(file_label_dict, marker_color_dict, data_range=None, show=False):
    path = "VOTPP folder/Results/Pickle files 2/Fits/"
    plt.figure(figsize=(6/1.1, 4/1.1))

    for AB, file_list in file_label_dict.items():
        T2_product_values_all = []
        T2_product_errors_all = []
        x_values_all = []
        for pickle_filename in file_list:
            fit_results = load_fit_results(path, pickle_filename)
            keys = fit_results.keys()
            x_values = [key[3] for key in keys]

            # Extract T2 values and errors for each bath
            T2_values = [fit_results[key]['T2']*1e3 for key in keys]  # Convert from ms to microseconds
            T2_errors = [fit_results[key]['T2_err']*1e3 for key in keys]

            # Append to all values list
            T2_product_values_all.append(T2_values)
            T2_product_errors_all.append(T2_errors)
            x_values_all.append(x_values)

        # Calculate T2 product rule values and errors for each AB
        T2_product_values = []
        T2_product_errors = []
        for Ta, Tb, Tc, Td, err_a, err_b, err_c, err_d in zip(*T2_product_values_all, *T2_product_errors_all):
            T2_product = T2_product_rule(Ta, Tb, Tc, Td)
            T2_product_values.append(T2_product)
            T2_product_error = T2_product**2 * np.sqrt((err_a/Ta**2)**2 + (err_b/Tb**2)**2 + (err_c/Tc**2)**2 + (err_d/Td**2)**2)
            T2_product_errors.append(np.sqrt(T2_product_error))

        # Apply data range if provided
        if data_range is not None:
            start, end = data_range
            x_values = x_values_all[0][start:end]  # Assuming x_values are the same for all files
            T2_product_values = T2_product_values[start:end]
            T2_product_errors = T2_product_errors[start:end]

        # # Plotting without lines
        color, marker = marker_color_dict[AB]
        x_values = [i*1e-4 for i in x_values]
        # plt.errorbar(x_values, T2_product_values, yerr=T2_product_errors, fmt=marker, color=color, ecolor='black', elinewidth=0.5, capsize=2, label=f'{AB}')
        
        # Plotting with connected points
        color, marker = marker_color_dict[AB]
        # Plot lines separately with black color
        plt.plot(x_values, T2_product_values, color=color, linewidth=0.5, alpha=0.5, linestyle='-')
        # Plot markers separately with the specified color
        plt.errorbar(x_values, T2_product_values, yerr=T2_product_errors, fmt=marker, color=color,
                    ecolor='black', elinewidth=0.5, capsize=2, label=f'{AB}')

    plt.xlabel(r"$B_0$ (T)")
    plt.ylabel(r"$T_2$ ($\mu$s)")
    # plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.ylim(bottom=-5)
    # plt.ylim(-3,70)
    plt.xlim(0.04,0.32)

    # Save plot in the "Results/T2_vs_B" folder
    save_path = "VOTPP folder/Results/T2_vs_B/"
    output_filename = os.path.join(save_path, "Combined_T2_Product_Rules.png")
    plt.savefig(output_filename, dpi=300)
    if show == True:
        plt.show()
    else:
        plt.close()



AB_list = ['AB1', 'AB2', 'AB3', 'AB4', 'AB5', 'AB6', 'AB7']  # Add all your AB values here
bath_list = ['E', 'C', 'H', 'N']  



# For plotting combined T2 vs B on semilog scale, and Product Rule T2 vs B on normal scale for each AB
for AB in AB_list:
    e_filename = f'[n-e]-(e)_HPC_{AB}.pkl_fit_results.pkl'
    C_filename = f'[n-e]-(n)_C_{AB}.pkl_fit_results.pkl'
    h_filename = f'[n-e]-(n)_H_{AB}.pkl_fit_results.pkl'
    N_filename = f'[n-e]-(n)_N_{AB}.pkl_fit_results.pkl'

    # Call the function to plot individual datasets and combined T2 product rule on semilog scale
    plot_T2_vs_field_combined(e_filename, C_filename, h_filename, N_filename, AB)

    # Call the function to plot only the combined T2 product rule on a normal scale
    plot_T2_product_rule_combined(e_filename, C_filename, h_filename, N_filename, AB)


# For plotting individual bath T2 vs B (e.g E bath AB1; T2 vs B)
for AB in AB_list:
    # Loop through each AB and plot for each bath
    e_filename = f'[n-e]-(e)_HPC_{AB}.pkl_fit_results.pkl'
    plot_T2_vs_B_single(e_filename, 'E', AB)

    C_filename = f'[n-e]-(n)_C_{AB}.pkl_fit_results.pkl'
    plot_T2_vs_B_single(C_filename, 'C', AB)

    h_filename = f'[n-e]-(n)_H_{AB}.pkl_fit_results.pkl'
    plot_T2_vs_B_single(h_filename, 'H', AB)

    N_filename = f'[n-e]-(n)_N_{AB}.pkl_fit_results.pkl'
    plot_T2_vs_B_single(N_filename, 'N', AB)



# Re-creating Chicco T2 vs B
file_label_dict = {
    'AB1': [
        '[n-e]-(e)_HPC_AB1.pkl_fit_results.pkl', 
        '[n-e]-(n)_C_AB1.pkl_fit_results.pkl', 
        '[n-e]-(n)_H_AB1.pkl_fit_results.pkl', 
        '[n-e]-(n)_N_AB1.pkl_fit_results.pkl'
    ],
    'AB2': [
        '[n-e]-(e)_HPC_AB2.pkl_fit_results.pkl', 
        '[n-e]-(n)_C_AB2.pkl_fit_results.pkl',
        '[n-e]-(n)_H_AB2.pkl_fit_results.pkl',
        '[n-e]-(n)_N_AB2.pkl_fit_results.pkl'
    ],
    'AB3': [
        '[n-e]-(e)_HPC_AB3.pkl_fit_results.pkl', 
        '[n-e]-(n)_C_AB3.pkl_fit_results.pkl',
        '[n-e]-(n)_H_AB3.pkl_fit_results.pkl',
        '[n-e]-(n)_N_AB3.pkl_fit_results.pkl'
    ],
    'AB4': [
        '[n-e]-(e)_HPC_AB4.pkl_fit_results.pkl', 
        '[n-e]-(n)_C_AB4.pkl_fit_results.pkl',
        '[n-e]-(n)_H_AB4.pkl_fit_results.pkl',
        '[n-e]-(n)_N_AB4.pkl_fit_results.pkl'
    ],
    'AB5': [
        '[n-e]-(e)_HPC_AB5.pkl_fit_results.pkl', 
        '[n-e]-(n)_C_AB5.pkl_fit_results.pkl',
        '[n-e]-(n)_H_AB5.pkl_fit_results.pkl',
        '[n-e]-(n)_N_AB5.pkl_fit_results.pkl'
    ],
    'AB6': [
        '[n-e]-(e)_HPC_AB6.pkl_fit_results.pkl', 
        '[n-e]-(n)_C_AB6.pkl_fit_results.pkl',
        '[n-e]-(n)_H_AB6.pkl_fit_results.pkl',
        '[n-e]-(n)_N_AB6.pkl_fit_results.pkl'
    ],
    'AB7': [
        '[n-e]-(e)_HPC_AB7.pkl_fit_results.pkl', 
        '[n-e]-(n)_C_AB7.pkl_fit_results.pkl',
        '[n-e]-(n)_H_AB7.pkl_fit_results.pkl',
        '[n-e]-(n)_N_AB7.pkl_fit_results.pkl'
    ]

}

marker_color_dict = {
    'AB1': ('#D273FF', '-o'),  # Light purple color, circle marker
    'AB2': ('#AC39FF', '-s'),  # Purple color, square marker
    'AB3': ('#3F2DFF', '-^'),  # Vibrant blue color, triangle_up marker
    'AB4': ('#0000AC', '-v'),  # Dark blue color, triangle_down marker
    'AB5': ('#4D0000', '-d'),  # Dark brown color, diamond marker
    'AB6': ('#FF0C10', '-<'),  # Red color, triangle_left marker
    'AB7': ('#FF6508', '->')   # Orange color, triangle_right marker
}


marker_color_dict = {
    'AB1': ('#D273FF', 'o'),  # Light purple color, circle marker
    'AB2': ('#AC39FF', 's'),  # Purple color, square marker
    'AB3': ('#3F2DFF', '^'),  # Vibrant blue color, triangle_up marker
    'AB4': ('#0000AC', 'v'),  # Dark blue color, triangle_down marker
    'AB5': ('#4D0000', 'd'),  # Dark brown color, diamond marker
    'AB6': ('#FF0C10', '<'),  # Red color, triangle_left marker
    'AB7': ('#FF6508', '>')   # Orange color, triangle_right marker
}


plot_all_T2_product_rules_combined(file_label_dict, marker_color_dict)













#################################
# Function to plot T2 vs B for H, C, and N datasets on a semilog plot
def plot_T2_vs_B_semilog(C_pickle_filename, h_pickle_filename, N_pickle_filename, AB, data_range=None, show=False):
    path = "VOTPP folder/Results/Pickle files 2/Fits/"

    # Load fit results for H, C, and N datasets
    C_fit_results = load_fit_results(path, C_pickle_filename)
    h_fit_results = load_fit_results(path, h_pickle_filename)
    N_fit_results = load_fit_results(path, N_pickle_filename)

    # Assuming the keys are consistent across all datasets
    keys = C_fit_results.keys()

    # Extract x_values from the keys
    x_values = [key[3] for key in keys]

    # Extract T2 values and errors for C, H, and N datasets
    T2_values_C = [C_fit_results[key]['T2'] * 1e3 for key in keys]  # Convert from ms to microseconds
    T2_values_h = [h_fit_results[key]['T2'] * 1e3 for key in keys]
    T2_values_N = [N_fit_results[key]['T2'] * 1e3 for key in keys]

    T2_errors_C = [C_fit_results[key]['T2_err'] * 1e3 for key in keys]
    T2_errors_h = [h_fit_results[key]['T2_err'] * 1e3 for key in keys]
    T2_errors_N = [N_fit_results[key]['T2_err'] * 1e3 for key in keys]

    # Apply data range if provided
    if data_range is not None:
        start, end = data_range
        x_values = x_values[start:end]
        T2_values_C = T2_values_C[start:end]
        T2_values_h = T2_values_h[start:end]
        T2_values_N = T2_values_N[start:end]
        T2_errors_C = T2_errors_C[start:end]
        T2_errors_h = T2_errors_h[start:end]
        T2_errors_N = T2_errors_N[start:end]

    # Plotting with error bars on a semilog scale
    plt.figure(figsize=(8, 4))
    x_values = [i*1e-4 for i in x_values]
    #     plt.errorbar(x_values, T2_values_e, yerr=T2_errors_e, fmt='-o', color='#3577b4', label='E Bath', 
    #              ecolor='black', elinewidth=0.5, capsize=2)
    # plt.errorbar(x_values, T2_values_C, yerr=T2_errors_C, fmt='-o', color='#d72f26', label='C Bath', 
    #              ecolor='black', elinewidth=0.5, capsize=2)
    # plt.errorbar(x_values, T2_values_h, yerr=T2_errors_h, fmt='-o',color='#499e2f', label='H Bath', 
    #              ecolor='black', elinewidth=0.5, capsize=2)
    # plt.errorbar(x_values, T2_values_N, yerr=T2_errors_N, fmt='-o', color='#9364bc', label='N Bath', 
    #              ecolor='black', elinewidth=0.5, capsize=2)
    plt.errorbar(x_values, T2_values_C, yerr=T2_errors_C, fmt='-o', label='C Bath', color='#d72f26', ecolor='black', elinewidth=0.5, capsize=2)
    plt.errorbar(x_values, T2_values_h, yerr=T2_errors_h, fmt='-o', label='H Bath', color='#499e2f', ecolor='black', elinewidth=0.5, capsize=2)
    plt.errorbar(x_values, T2_values_N, yerr=T2_errors_N, fmt='-o', label='N Bath', color='#9364bc', ecolor='black', elinewidth=0.5, capsize=2)

    plt.semilogy(x_values, T2_values_C, '-o', color='tab:blue')
    plt.semilogy(x_values, T2_values_h, '-o', color='tab:orange')
    plt.semilogy(x_values, T2_values_N, '-o', color='tab:green')

    plt.xlabel(r"$B_0$ (T)")
    plt.ylabel(r"$T_2$ ($\mu$s)")
    plt.grid(True, which="both", ls="--")
    # plt.ylim(bottom=1)
    plt.legend()
    plt.tight_layout()

    # Save plot in the "Results/T2_vs_B" folder
    save_path = "VOTPP folder/Results/T2_vs_B/"
    output_filename = os.path.join(save_path, f"T2_vs_B_{AB}_without_e.png")
    plt.savefig(output_filename, dpi=300)
    if show:
        plt.show()
    else:
        plt.close()
    


C_filename = f'[n-e]-(n)_C_AB3.pkl_fit_results.pkl'
h_filename = f'[n-e]-(n)_H_AB3.pkl_fit_results.pkl'
N_filename = f'[n-e]-(n)_N_AB3.pkl_fit_results.pkl'

# Call the function to plot individual datasets and combined T2 product rule on semilog scale
plot_T2_vs_B_semilog(C_filename, h_filename, N_filename, 'AB3')

# Call the function to plot only the combined T2 product rule on a normal scale
plot_T2_vs_B_semilog(C_filename, h_filename, N_filename, 'AB3')