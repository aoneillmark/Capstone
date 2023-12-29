import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import pandas as pd

# Define the curve fitting function
def coherence_time_func(time, beta, T2):
    return np.exp(-(((2*time)/T2)**beta))

concentration_list = [0.02, 0.05, 0.1, 0.3]
T2_values = []
path = "VOTPP folder/Results/Pickle files/"

for conc in concentration_list:
    conc_percentage = conc * 100
    with open(f'{path}results_combined_conc_{conc}.pkl', 'rb') as f:
        data = pickle.load(f)

    for key, df in data.items():
        plt.figure(figsize=(10, 6))
        coherence = df[0]  # Assuming df[0] is your data column
        time = df.index

        # Convert coherence values to their absolute values
        coherence = np.abs(coherence)

        # Replace '--' with 0 and convert to float
        coherence = pd.Series(coherence).replace('--', 0).astype(float)

        # Replace NaN and infinite values with zero
        coherence = np.nan_to_num(coherence)

        # Curve fitting and plotting logic
        try:
            popt, _ = curve_fit(coherence_time_func, time, coherence, maxfev=5000, bounds=([0, -np.inf], [np.inf, np.inf]), p0=[1, 0.1])
            beta_fit, T2_fit = popt
            T2_values.append(T2_fit)

            plt.plot(time, coherence, 'o', label=f'Conc {conc_percentage:.0f}% - Data')
            plt.plot(time, coherence_time_func(time, *popt), '--', label=f'Fit: Beta={beta_fit:.3f}, $T_2$={T2_fit:.3f} ms')
        except RuntimeError as e:
            print(f"Fit failed for concentration {conc}: {e}")

        plt.xlabel('Time (ms)')
        plt.ylabel('Coherence')
        plt.title(f'Coherence vs Time with Fitted Curves for Conc {conc_percentage:.0f}%')
        plt.legend()
        plt.tight_layout()
        output_filename = os.path.join(path, f"coherence_conc_{conc_percentage}_with_fit.png")
        plt.savefig(output_filename, dpi=300)
        plt.show()

plt.xlabel('Time (ms)')
plt.ylabel('Coherence')
plt.title('Coherence vs Time with Fitted Curves')
# plt.xlim(-0.01, 0.51)
plt.legend()
plt.tight_layout()
plt.savefig(f'{path}coherence_for_different_concentrations_(E_trans).png', dpi=300)
plt.show()

# Plot T2 vs concentration in a separate plot
plt.figure(figsize=(10, 6))
plt.plot(concentration_list, T2_values, 'o-')
plt.xlabel('Concentration of electron spins (%)')
plt.ylabel(f'$T_2$ (ms)')
plt.title(f'$T_2$ vs Concentration')
plt.tight_layout()
plt.savefig(f'{path}T2_vs_concentration_(combined).png', dpi=300)
plt.show()

# Save T2 values with respective concentrations to a pickle file for later use
with open(f'{path}T2_values.pkl', 'wb') as f:
    pickle.dump((concentration_list, T2_values), f)