from VOTPP_class import VOTPP_class
from mpi4py import MPI
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import pickle

# Start timer
start = time.time()

# Get mpi rank
# rank = MPI.COMM_WORLD.Get_rank()
# size = MPI.COMM_WORLD.Get_size()

# Print the largest rank in the ranks
# print("Rank: {}".format(rank))


# Runner to converge parameters
def runner(concentration_value, 
           changing_variable, variable_values, 
           bath_parameters, simulator_parameters, calc_parameters,):
    
    # Attempt to retrieve the changing_variable from each dictionary
    changing_invalue = None
    for param_dict in [simulator_parameters, bath_parameters, calc_parameters]:
        try:
            changing_invalue = param_dict[changing_variable]
            break
        except KeyError:
            continue

    if changing_invalue is None:
        raise ValueError(f"'{changing_variable}' not found in any of the provided dictionaries.")

    bath_parameters['concentration'] = concentration_value # Set concentration value

    simulator = VOTPP_class(**bath_parameters) # Set up bath and atoms
    sim = simulator.setup_simulator(**simulator_parameters) # Set up simulator

    # ls = []
    results = {}
    for v in variable_values: # Iterate through variable values (e.g. order = 1, 2, 3)
        # Progress printing
        if rank == 0:
            print("Rank: {}, {} = {}".format(rank, changing_variable, v))

        setattr(simulator, changing_variable, v) # Set the variable in the simulator to the value (e.g. simulator['order'] = 1)
        l = sim.compute(**calc_parameters) # Run the simulation
        
        v_key = convert_to_key(v)
        results[v_key] = pd.DataFrame([l.real], columns=calc_parameters['timespace']).T
        # ls.append(l.real)

    simulator_parameters[changing_variable] = changing_invalue
    # ls = pd.DataFrame(ls, columns=calc_parameters['timespace'], index=variable_values).T
    return results


def convert_to_key(value):
    if isinstance(value, list):
        return tuple(value)
    else:
        return value


# concentration_list = [0, 0.02, 0.05, 0.1, 0.2, 0.5]
# concentration_list = [0.02, 0.05,]
concentration_list = [0]
order_list = [1, 2, 3, 4] 
# r_bath_list = [40, 80, 160, 220]
# r_dipole_list = [20, 40, 60, 100, 140, 180]
r_bath_list = [10,20,30, 40]
r_dipole_list = [8,10,12, 14]
# cell_size_list = [60,100,200]
cell_size_list = [60,100,200]
nbstates_list = [128,]
# nbstates_list = [16,32,64]

# r_bath_list = [10,]
# r_dipole_list = [8,]
# cell_size_list = [60,]


default_calc_parameters = {
    # 'timespace': np.linspace(0, 7e-2, 201),
    'timespace': np.linspace(0, 1e-3, 5), # 7e-2
    'method': 'cce',
    'pulses': 1,
    'nbstates': 128, #!
    'quantity': 'coherence',
    'parallel': True,
    'parallel_states': True,
}

default_bath_parameters = {
    'concentration': 0, #!
    'cell_size': 60, #!
    'displacement': 0, # Angstrom
    'seed': 8000
}

default_simulator_parameters = { ########## These should be greater when simulating with HPC
    'order': 3, #!
    'r_bath': 20, #!
    'r_dipole': 10, #!
    'pulses': 1, # N pulses in CPMG sequence
    'magnetic_field': [500, 0, 0], # Magnetic field in Gauss
}


simulator = VOTPP_class(**default_bath_parameters) # Set up bath and atoms
sim = simulator.setup_simulator(**default_simulator_parameters) # Set up simulator

active_positions = simulator.get_active_nuclei_positions(sim, default_simulator_parameters['r_bath'])
active_nuclei = simulator.get_number_of_active_nuclei(sim, default_simulator_parameters['r_bath'])
print("Active nuclei positions:")
print("     x         y         z")
print(active_positions)
print("")
print("Number of active nuclei:")
print(active_nuclei)

print("Position of central spin:")
print(simulator.qpos1)

simulator.visualize_cluster(sim)