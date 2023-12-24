import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the curve fitting function, e.g., an exponential decay for T2
def coherence_time_func(time, beta, T2):
    y = np.exp(-(((2*time)/T2)**beta)) # Is there a scaling issue here?
    return y

concentration_list = [0.02, 0.05,0.1,0.3,0.5]
T2_values = []
path = "VOTPP folder/Results/Pickle files/"

plt.figure(figsize=(10, 6))

for conc in concentration_list:
    conc_percentage = conc*100

    # Load results
    with open(f'{path}results_conc_{conc}.pkl', 'rb') as f:
        data = pickle.load(f)

    for key, df in data.items():
        time = df.index
        coherence = abs(df[0])  # Access the first (and only) column by index

        # Fit curve and extract T2
        try:
            popt, _ =  curve_fit(coherence_time_func, time, abs(coherence), maxfev=5000, bounds=([0, -np.inf], [np.inf, np.inf]), p0=[1,0.025])
            T2 = popt[0]
            T2_values.append(T2)

            # Plot raw data
            plt.plot(time, coherence, label=f'Conc {conc_percentage:.0f}% - Data')

            # Plot fitted curve
            plt.plot(time, coherence_time_func(time, *popt), linestyle='--', label=f'Conc {conc_percentage:.0f}% - Fit: $T_2$={T2:.2f} ms')
        except RuntimeError as e:
            print(f"Fit failed for concentration {conc}: {e}")

plt.xlabel('Time (ms)')
plt.ylabel('Coherence')
plt.title('Coherence vs Time with Fitted Curves')
plt.xlim(-0.01, 0.51)
plt.legend()
plt.tight_layout()
plt.savefig(f'{path}coherence_for_different_concentrations.png', dpi=300)
plt.show()

# Plot T2 vs concentration in a separate plot
plt.figure(figsize=(10, 6))
plt.plot(concentration_list, T2_values, 'o-')
plt.xlabel('Concentration of electron spins (%)')
plt.ylabel(f'$T_2$ (ms)')
plt.title(f'$T_2$ vs Concentration')
plt.tight_layout()
plt.savefig(f'{path}T2_vs_concentration.png', dpi=300)
plt.show()

# Save T2 values with respective concentrations to a pickle file for later use
with open(f'{path}T2_values.pkl', 'wb') as f:
    pickle.dump((concentration_list, T2_values), f)