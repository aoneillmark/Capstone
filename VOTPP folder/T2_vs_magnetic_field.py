import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

def load_fit_results(path, pickle_filename):
    with open(os.path.join(path, pickle_filename), 'rb') as f:
        fit_results = pickle.load(f)
    return fit_results

def magnitude_of_field(field_tuple):
    """Compute the magnitude of the magnetic field from its tuple representation."""
    return np.sqrt(sum([x**2 for x in field_tuple]))

def plot_T2_vs_field(pickle_filename):
    path = "VOTPP folder/Results/Pickle files/Fits/"
    
    # Load fit results
    fit_results = load_fit_results(path, pickle_filename)
    
    # Extract T2 values and magnetic field magnitudes
    magnetic_fields = [magnitude_of_field(tuple(map(float, key.split(', ')))) for key in fit_results.keys()]
    T2_values = [fit_results[key]['T2'] for key in fit_results.keys()]
    
    # Convert T2 from ms to microseconds
    T2_values = [(T2*1e3) for T2 in T2_values]
    # Convert magnetic field from Gauss to Tesla
    magnetic_fields = [field*1e-4 for field in magnetic_fields]

    # Plotting
    plt.figure(figsize=(10,6))
    plt.plot(magnetic_fields, T2_values, marker='o')
    plt.title(r"$T_2$ vs Magnetic Field Magnitude")
    plt.xlabel("B (T)")
    # ylabel in latex format T_2
    plt.ylabel(r"$T_2$ ($\mu$s)")
    plt.grid(True)
    # Start the tick markers at 0.04 and go up in increments of 0.04
    plt.xticks(np.arange(0.04, 0.36, 0.04))
    plt.xlim(0.04, 0.34)
    plt.ylim(0, 70)
    plt.tight_layout()
    
    # Save and show plot
    output_filename = os.path.join(path, "T2_vs_field_magnitude.png")
    plt.savefig(output_filename, dpi=300)
    plt.show()

# Call the function with the name of your pickle file
plot_T2_vs_field('magnetic_fit_results.pkl')
