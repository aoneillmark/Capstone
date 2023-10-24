import matplotlib.pyplot as plt
import pickle
import pandas as pd

path = "VOTPP folder/Results/Pickle files/"
# Loading results in another script
with open((str(path) + 'order_results.pkl') , 'rb') as f:
    loaded_results = pickle.load(f)

concentration_to_plot = 0.05
for v_key, df in loaded_results[concentration_to_plot].items():
    # Check if the key is a tuple and handle the label accordingly
    if isinstance(v_key, tuple):
        label_str = ', '.join(map(str, v_key))
        label = f"Field: ({label_str})"
    else:
        label = f"Order {v_key}"
    plt.plot(df.index, df[0], label=label)

plt.legend()
plt.xlabel('Time')
plt.ylabel('Coherence')
plt.title(f"Concentration: {concentration_to_plot}")

image_path = "VOTPP folder/Results/Plots/"
# Save the figure to a file
output_filename = f"{image_path}Concentration_{concentration_to_plot}.png"
plt.savefig(output_filename, dpi=300)  # dpi is dots per inch (resolution). You can adjust as needed.

# If you want to show the plot, you can uncomment the next line
# plt.show()
