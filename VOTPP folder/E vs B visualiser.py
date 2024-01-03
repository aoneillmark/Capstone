#importing functions 
import numpy as np
import matplotlib.pyplot as plt
import pycce as pc 
import pandas as pd
from scipy.optimize import curve_fit 
import mpi4py
from VOTPP_class import VOTPP_class

##############################################################################################################

# def convert_to_key(value):
#     if isinstance(value, list):
#         return tuple(value)
#     else:
#         return value


# def update_parameters(changing_variable, value, bath_parameters, simulator_parameters):
#     if changing_variable == 'cell_size':
#         bath_parameters['cell_size'] = value
#     elif changing_variable == 'r_bath':
#         simulator_parameters['r_bath'] = value
#     elif changing_variable == 'r_dipole':
#         simulator_parameters['r_dipole'] = value
#     elif changing_variable == 'order':
#         simulator_parameters['order'] = value
#     elif changing_variable == 'magnetic_field':
#         simulator_parameters['magnetic_field'] = value
#     elif changing_variable == 'pulses':
#         simulator_parameters['pulses'] = value
#     elif changing_variable == 'nbstates':
#         simulator_parameters['nbstates'] = value
    


# def get_changing_invalue(changing_variable, param_dicts):
#     for param_dict in param_dicts:
#         try:
#             return param_dict[changing_variable]
#         except KeyError:
#             continue
#     raise ValueError("'" + changing_variable + "' not found in any of the provided dictionaries.")


def setup_simulator(concentration_value, bath_parameters, simulator_parameters, num_spins, spin_type, alpha, beta):
    bath_parameters['concentration'] = concentration_value
    simulator = VOTPP_class(num_spins=num_spins, spin_type=spin_type, alpha=alpha, beta=beta, **bath_parameters)
    sim_original = simulator.setup_simulator(**simulator_parameters)
    return simulator, sim_original


# def run_single_simulation(concentration_value, bath_parameters, simulator_parameters, calc_parameters, num_spins, spin_type, alpha, beta, changing_variable, value, changing_variable2=None, variable_value2=None):
#     update_parameters(changing_variable, value, bath_parameters, simulator_parameters)
#     simulator, sim_original = setup_simulator(concentration_value, bath_parameters, simulator_parameters, num_spins, spin_type, alpha, beta)

#     # Print number of active nuclei
#     if rank == 0:
#         # bath = sim_original.bath
#         num_active_nuclei = simulator.get_number_of_active_nuclei(atoms=sim_original.bath, r_bath=simulator_parameters['r_bath'], central_spin_position=[0.50446035, 0.50446035, 0.55872939])
#         print(f"Number of active nuclei: {num_active_nuclei}")
#         print(f"Printing bath: ")
#         call = simulator.print_bath(sim_original)

#     # Run the simulation and return the result
#     return sim_original.compute(**calc_parameters)


# def process_results(simulation_results, calc_parameters):
#     results = {}
#     for v_key, result in simulation_results.items():
#         results[v_key] = pd.DataFrame([result], columns=calc_parameters['timespace']).T
#     return results


# def runner(concentration_value, changing_variable, variable_values, bath_parameters, simulator_parameters, calc_parameters, num_spins=2, spin_type=None, alpha=None, beta=None, changing_variable2=None, variable_values2=None):
#     changing_invalue = get_changing_invalue(changing_variable, [simulator_parameters, bath_parameters, calc_parameters])

#     simulation_results = {}
#     for v in variable_values:
#         if rank == 0:
#             print(f"Running simulation with {changing_variable}: {v}")

#         # Use a hashable key
#         v_key = convert_to_key(v)
#         simulation_results[v_key] = run_single_simulation(concentration_value, bath_parameters, simulator_parameters, calc_parameters, num_spins, spin_type, alpha, beta, changing_variable, v, changing_variable2, variable_values2)

#     results = process_results(simulation_results, calc_parameters)

#     # Reset the original values of the changing variables
#     if changing_variable in bath_parameters:
#         bath_parameters[changing_variable] = changing_invalue
#     elif changing_variable in simulator_parameters:
#         simulator_parameters[changing_variable] = changing_invalue

#     if changing_variable2 in calc_parameters:
#         calc_parameters[changing_variable2] = variable_values2[0] if variable_values2 else None

#     return results


timespace_absolute = np.linspace(0, 1, 101)

default_calc_parameters = {
    'timespace': timespace_absolute, # 7e-2
    'method': 'gcce',
    'pulses': [('x', ((2*np.pi)/3), timespace_absolute / 2), ('x', ((2*np.pi)/3), timespace_absolute / 2)], # Paper defines a Hahn-echo pulse sequence with 2pi/3 pulses?
    # 'pulses': [('x', np.pi, timespace_absolute / 2), ('x', np.pi, timespace_absolute / 2)], # Paper defines a Hahn-echo pulse sequence with 2pi/3 pulses?
    'nbstates': 30, #!
    'quantity': 'coherence',
    'parallel': True,
    'parallel_states': True,
}

default_bath_parameters = {
    'concentration': 0.02, #!
    'cell_size': 500, #!
    'seed': 8000
}

default_simulator_parameters = { ########## These should be greater when simulating with HPC
    'order': 3, #!
    'r_bath': 75, #16,
    'r_dipole': 60, #6,
    'magnetic_field': [3000, 0, 0], # Magnetic field in Gauss
}


simulator, sim_original = setup_simulator(0.02, default_bath_parameters, default_simulator_parameters, num_spins=2, spin_type='electron', alpha=4, beta=5)

##############################################################################################################

cen = simulator.cen

ens = []
ms = np.linspace(0, 3000, 51)  # applied magnetic field
for mf in ms:
    cen.generate_states([mf, 0, 0]) # APPLIED ALONG THE X DIRECTION
    ens.append(cen.energies)

ens = np.asarray(ens)

lowerdf = pd.DataFrame(ens[:, :10]/1e6, index=ms/1e4,
                       columns=np.arange(10))
higherdf = pd.DataFrame(ens[:, :10:-1]/1e6, index=ms/1e4,
                        columns=np.arange(11, ens.shape[1])[::-1])

fig, ax = plt.subplots(figsize=(3,5))

lowerdf.plot(ax=ax, cmap='viridis', legend=False, lw=1)
higherdf.plot(ax=ax, cmap='viridis', legend=False, lw=1)
lowerdf[4].plot(ax=ax, color='black', ls=':', lw=2)
lowerdf[5].plot(ax=ax, color='black', ls=':', lw=2)
ax.set(xlabel='B (T)', ylabel='E (GHz)', )
plt.show()