import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
                        coherence_series = df[0].abs()  # Absolute values for complex numbers
                        key = f"{filename}_{variable}_{target_seed}"
                        data_collections[key] = coherence_series
                        print(f"Data extracted for key: {key}")
                    else:
                        print(f"No coherence data found for seed {target_seed} in file: {filename}")
                else:
                    print(f"Target seed {target_seed} not found in file: {filename}")
    return data_collections



def calculate_y_values_product(data_collections):
    product = None
    for key, series in data_collections.items():
        if product is None:
            product = series.copy()
        else:
            product *= series
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
            plt.plot(series.index, series.values, label=f"{label_list[i]}", color=colors[(i) % len(colors)])
        else:
            print(f"Data for {key} is not in expected format (pandas Series), skipping.")

    if product is not None and not product.empty:
        plt.plot(product.index, product.values, label="Product of Coherences",color="#ff7f0e", linestyle="--", linewidth=2)
    else:
        print("Product data is empty or not provided.")

    # plt.title("Coherence vs Timespace")
    plt.xlabel(r"2$\tau$ (ms)")
    plt.ylabel("Coherence")
    plt.legend()
    plt.tight_layout()

    # Save the plot
    output_filename = os.path.join(save_path, "combined_coherence_plot.png")
    plt.savefig(output_filename, dpi=300)
    plt.show()
    plt.close()
    print(f"Plot saved to {output_filename}")

# Usage
pickle_path = "VOTPP folder/Results/Pickle files/"
save_path = "VOTPP folder/Results/Plots/"
pickle_filenames = ["[n-e]-(e)_AB3.pkl", "[n-e]-(n)_H_AB3.pkl", "[n-e]-(n)_C_AB3.pkl", "[n-e]-(n)_N_AB3.pkl"]

data_collections = extract_coherence_and_timespace(pickle_filenames, pickle_path)

product_of_y_values = calculate_y_values_product(data_collections)

plot_and_save_combined(data_collections, product_of_y_values, save_path)
    
