import matplotlib.pyplot as plt
import pickle
import pandas as pd

path = "VOTPP folder/Results/Pickle files/"
# Loading results in another script
with open((str(path) + 'order_results.pkl') , 'rb') as f:
    loaded_results = pickle.load(f)

concentration_to_plot = 0.05
for order_val, df in loaded_results[concentration_to_plot].items():
    plt.plot(df.index, df[0], label=f"Order {order_val}")

plt.legend()
plt.xlabel('Time')
plt.ylabel('Coherence')
plt.title(f"Concentration: {concentration_to_plot}")

image_path = "VOTPP folder/Results/Plots/"
# Save the figure to a file
output_filename = f"{image_path}Concentration_{concentration_to_plot}.png"
plt.savefig(output_filename, dpi=300)  # dpi is dots per inch (resolution). You can adjust as needed.

# If you still want to show the plot, you can uncomment the next line
plt.show()