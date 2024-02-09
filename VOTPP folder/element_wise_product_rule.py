import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# def inspect_pickle_file_structure(pickle_filenames, pickle_path):
#     for filename in pickle_filenames:
#         full_path = os.path.join(pickle_path, filename)
#         print(f"\nInspecting file: {full_path}")
#         with open(full_path, 'rb') as file:
#             data = pickle.load(file)
#             if not data:
#                 print(f"No data found in file: {filename}")
#                 continue

#             for variable, seeds_data in data.items():
#                 print(f"Variable: {variable}, Type: {type(seeds_data)}")
#                 for seed, datasets in seeds_data.items():
#                     print(f"  Seed: {seed}, Type: {type(datasets)}")
#                     for sub_key, content in datasets.items():
#                         print(f"    Sub_key: {sub_key}, Content Type: {type(content)}, Example Data: {content if isinstance(content, (int, float, str)) else str(content)[:100]}")


def get_colorblind_friendly_colors():
    # Custom color palette with shades from blue to red
    colors = [
        "#1f77b4",  # Muted blue
        # "#ff7f0e",  # Safety orange
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

def extract_coherence_and_timespace(pickle_filenames, pickle_path, target_seed=(0, 0, 3000)):
    data_collections = {}
    for filename in pickle_filenames:
        full_path = os.path.join(pickle_path, filename)
        with open(full_path, 'rb') as file:
            data = pickle.load(file)
            for variable, seeds_data in data.items():
                if target_seed in seeds_data:
                    df = seeds_data[target_seed]
                    if 0 in df:
                        # Convert complex numbers to their magnitudes
                        # Replace '--' with NaN and convert to float
                        coherence_series = df[0].apply(lambda x: np.nan if x == '--' else abs(x))
                        key = f"{filename}_{variable}_{target_seed}"
                        data_collections[key] = coherence_series
                        print(f"Data extracted for key: {key}")
                    else:
                        print(f"No coherence data found for seed {target_seed} in file: {filename}")
                else:
                    print(f"Target seed {target_seed} not found in file: {filename}")
    return data_collections







# def calculate_y_values_product(data_collections):
#     product = None
#     for key, series in data_collections.items():
#         if product is None:
#             product = series.copy()
#         else:
#             product *= series
#     return product

def interpolate_datasets(data_collections, start=0, end=0.1, num_points=201):
    new_index = np.linspace(start, end, num_points)
    interpolated_collections = {}

    for key, series in data_collections.items():
        if not series.empty:
            # Replace zeros with NaN for interpolation
            series_replaced_zeros = series.replace(0, np.nan)

            # Drop NaNs for spline fitting
            valid_series = series_replaced_zeros.dropna()

            # Fit a cubic spline
            if len(valid_series) > 3:  # Cubic spline requires at least 4 points
                spline = CubicSpline(valid_series.index, valid_series)
                series_interpolated = pd.Series(spline(new_index), index=new_index)
            else:
                # Fallback to linear interpolation if not enough points for cubic spline
                series_interpolated = series_replaced_zeros.reindex(series_replaced_zeros.index.union(new_index))
                series_interpolated.interpolate(method='linear', inplace=True)
                series_interpolated = series_interpolated.loc[new_index]

            # Apply absolute value to ensure all values are non-negative
            series_interpolated = series_interpolated.abs()

            interpolated_collections[key] = series_interpolated
            print(f"Extended interpolated data for {key}:\n", series_interpolated.head(20))
        else:
            print(f"Data for {key} is empty, skipping.")
    
    return interpolated_collections






def calculate_y_values_product(data_collections):
    product = None
    for key, series in data_collections.items():
        if product is None:
            product = series.copy()
        else:
            product = product.combine(series, lambda x1, x2: x1 * x2 if pd.notna(x2) else x1)
        print(f"Product after including {key} - first 5 values:\n", product.head())
    return product

def plot_and_save_combined(data_collections, product, save_path):
    plt.figure(figsize=(8, 4))
    colors = get_colorblind_friendly_colors()

    if not data_collections:
        print("No data in data_collections to plot.")
        return

    label_list = ['E bath', 'H bath', 'C bath', 'N bath']

    for i, (key, series) in enumerate(data_collections.items()):
        if series.empty:
            print(f"Data for {key} is empty, skipping.")
            continue

        if isinstance(series, pd.Series):
            series = series.apply(pd.to_numeric, errors='coerce')  # Ensure numeric type
            # set the dot size in the scatter plot to be smaller
            plt.scatter(series.index, series.values, label=f"{label_list[i]}", color=colors[i % len(colors)], s=10)
            # plt.scatter(series.index, series.values, label=f"{label_list[i]}", color=colors[i % len(colors)])

        else:
            print(f"Data for {key} is not in expected format (pandas Series), skipping.")

    if product is not None and not product.empty:
        product = product.apply(pd.to_numeric, errors='coerce')
        plt.scatter(product.index, product.values, label="Product of Coherences", color="#ff7f0e", s=10)
        plt.plot(product.index, product.values, color="#ff7f0e", linewidth=1)

    plt.xlabel(r"2$\tau$ (ms)")
    plt.ylabel("Coherence")
    plt.legend()
    plt.tight_layout()
    plt.xlim(-0.001,0.1)
    # plt.ylim(-0.001,1)

    output_filename = os.path.join(save_path, "combined_coherence_plot.png")
    plt.savefig(output_filename, dpi=300)
    plt.show()
    plt.close()
    print(f"Plot saved to {output_filename}")


# Usage
pickle_path = "VOTPP folder/Results/Pickle files 2/Simulation Results/"
save_path = "VOTPP folder/Results/Plots 2/"
pickle_filenames = ["[n-e]-(e)_HPC_AB3.pkl", "[n-e]-(n)_H_AB3.pkl", "[n-e]-(n)_C_AB3.pkl", "[n-e]-(n)_N_AB3.pkl"]

data_collections = extract_coherence_and_timespace(pickle_filenames, pickle_path)
interpolated_data_collections = interpolate_datasets(data_collections)

# Now use 'interpolated_data_collections' for further processing and plotting
product_of_y_values = calculate_y_values_product(interpolated_data_collections)
plot_and_save_combined(interpolated_data_collections, product_of_y_values, save_path)

# # Usage
# pickle_path = "VOTPP folder/Results/Pickle files 2/Simulation Results/"
# save_path = "VOTPP folder/Results/Plots 2/"
# pickle_filenames = ["[n-e]-(e)_HPC_AB3.pkl", "[n-e]-(n)_H_AB3.pkl", "[n-e]-(n)_C_AB3.pkl", "[n-e]-(n)_N_AB3.pkl"]

# data_collections = extract_coherence_and_timespace(pickle_filenames, pickle_path)

# product_of_y_values = calculate_y_values_product(data_collections)

# plot_and_save_combined(data_collections, product_of_y_values, save_path)
    
