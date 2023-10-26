import matplotlib.pyplot as plt
import pickle
import pandas as pd
import os

path = "VOTPP folder/Results/Pickle files/"
pickle_filename = 'magnetic_results.pkl'

# Extract the variable name from the pickle file name
variable_name = pickle_filename.split('_results.pkl')[0]

# Loading results from the pickle file
with open(os.path.join(path, pickle_filename), 'rb') as f:
    loaded_results = pickle.load(f)

print(loaded_results.keys())

concentration_to_plot = 0
for v_key, df in loaded_results[concentration_to_plot].items():
    print(f"Data for {v_key}:")
    print(df)
    # Check if the key is a tuple and handle the label accordingly
    if isinstance(v_key, tuple):
        label_str = ', '.join(map(str, v_key))
        label = f"Field: ({label_str})"
    else:
        label = f"Value {v_key}"
    plt.plot(df.index, df[0], label=label, marker='o')

plt.legend()
plt.xlabel('Time')
plt.ylabel('Coherence')
plt.title(f"{variable_name.capitalize()} - Concentration: {concentration_to_plot}")

image_path = "VOTPP folder/Results/Plots/"

# Modify the output_filename to include both variable_name and concentration_to_plot
output_filename = os.path.join(image_path, f"{variable_name}_concentration_{concentration_to_plot}.png")

plt.savefig(output_filename, dpi=300)  # dpi is dots per inch (resolution). You can adjust as needed.

# If you want to show the plot, you can uncomment the next line
# plt.show()
