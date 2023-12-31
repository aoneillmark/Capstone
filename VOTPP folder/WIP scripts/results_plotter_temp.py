import matplotlib.pyplot as plt
import pickle
import pandas as pd
import os

def plot_from_file(pickle_filenames):
    path = "VOTPP folder/Results/Pickle files/"
    image_path = "VOTPP folder/Results/Plots/"

    for pickle_filename in pickle_filenames:
        # Extract the variable name from the pickle file name
        variable_name = pickle_filename.split('_results.pkl')[0]

        # Loading results from the pickle file
        with open(os.path.join(path, pickle_filename), 'rb') as f:
            loaded_results = pickle.load(f)

        print(loaded_results.keys())

        concentration_to_plot = 0
        plt.figure()  # Ensure a new figure for each plot
        for v_key, data in loaded_results[concentration_to_plot].items():
            print(f"Data for {v_key}:")
            print(data)
            
            # Check if the data is a DataFrame or Series
            if isinstance(data, pd.DataFrame):
                for column in data.columns:
                    plt.plot(data.index, data[column], label=f"{v_key}_{column}")
            elif isinstance(data, pd.Series):
                plt.plot(data.index, data, label=v_key)
            else:
                raise ValueError("Expected a DataFrame or Series.")

            # Check if the key is a tuple and handle the label accordingly
            if isinstance(v_key, tuple):
                label_str = ', '.join(map(str, v_key))
                label = f"Field: ({label_str})"
            else:
                label = f"Value {v_key}"

        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Coherence')
        plt.title(f"{variable_name.capitalize()} - Concentration: {concentration_to_plot}")

        # Modify the output_filename to include both variable_name and concentration_to_plot
        output_filename = os.path.join(image_path, f"{variable_name}_concentration_{concentration_to_plot}.png")
        plt.savefig(output_filename, dpi=300)  # dpi is dots per inch (resolution). You can adjust as needed.

        # If you want to show the plot, you can uncomment the next line
        plt.show()


# Provide the filenames you want to process as a list
plot_from_file(['Cathal_hogan_double_spin.pkl', 'magnetic_results.pkl', 'r_bath_results.pkl', 'r_dipole_results.pkl'])
