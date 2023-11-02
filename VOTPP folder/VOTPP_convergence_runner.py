from VOTPP_class import VOTPP_class
from mpi4py import MPI
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import pickle
import pycce as pc
import copy

# Start timer
start = time.time()

# Get mpi rank
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

# Print the largest rank in the ranks
# print("Rank: {}".format(rank))


def runner(concentration_value, 
           changing_variable, variable_values, 
           bath_parameters, simulator_parameters, calc_parameters,
           num_spins=2, spin_type=None,
           changing_variable2=None, variable_values2=None,):
    
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

    simulator = VOTPP_class(num_spins=num_spins, spin_type=spin_type, **bath_parameters) # Set up bath and atoms
    sim_original = simulator.setup_simulator(**simulator_parameters) # Set up simulator

    results = {}
    for idx, v in enumerate(variable_values): # Iterate through variable values (e.g. order = 1, 2, 3)
        sim = copy.deepcopy(sim_original) # Create a deep copy of the original simulator object. This ensures that you always start with a fresh simulator object for each value of changing_variable, and it should prevent the error caused by reusing and potentially modifying the same simulator object across iterations.

        # Progress printing
        if rank == 0:
            print("Rank: {}, {} = {}".format(rank, changing_variable, v))

        # Need an if statement here to check if changing_variable or changing_variable2 is in calc_parameters
        if changing_variable2 in calc_parameters:
            calc_parameters[changing_variable2] = variable_values2[idx]

        setattr(sim, changing_variable, v)
        
        l = sim.compute(**calc_parameters) # Run the simulation
        
        v_key = convert_to_key(v)
        results[v_key] = pd.DataFrame([l], columns=calc_parameters['timespace']).T

    simulator_parameters[changing_variable] = changing_invalue

    if changing_variable2 in calc_parameters:
        calc_parameters[changing_variable2] = variable_values2[0]
    return results

def convert_to_key(value):
    if isinstance(value, list):
        return tuple(value)
    else:
        return value



# concentration_list = [0, 0.02, 0.05, 0.1, 0.2, 0.5]
# concentration_list = [0.02, 0.05,]
concentration_list = [0]
order_list = [1, 2, 3] 
# r_bath_list = [40, 80, 160, 220]
# r_dipole_list = [20, 40, 60, 100, 140, 180]
r_bath_list = [10, 15, 20, 25, 30, 35, 40, 45, 50]
r_dipole_list = [5, 10, 15, 20, 25, 30, 35, 40]
# cell_size_list = [60,100,200]
cell_size_list = [60,100,200]
nbstates_list = [128,]
# nbstates_list = [16,32,64]

# r_bath_list = [10,]
# r_dipole_list = [8,]
# cell_size_list = [60,]


default_calc_parameters = {
    # 'timespace': np.linspace(0, 7e-2, 201),
    'timespace': np.linspace(0, 4, 201), # 7e-2
    'method': 'cce',
    'pulses': [pc.Pulse('x', np.pi)], # Paper defines a Hahn-echo pulse sequence with 2pi/3 pulses?
    'nbstates': 4, #!
    'quantity': 'coherence',
    'parallel': True,
    'parallel_states': True,
}

default_bath_parameters = {
    'concentration': 0, #!
    'cell_size': 100, #!
    'seed': 8000
}

default_simulator_parameters = { ########## These should be greater when simulating with HPC
    'order': 2, #!
    'r_bath': 20, #35
    'r_dipole': 10, #20
    # 'pulses': 1, # N pulses in CPMG sequence (=1 is Hahn-echo, =0 is free induction decay)
    # 'pulses': [pc.Pulse('x', 2*(np.pi)/3)], # Paper defines a Hahn-echo pulse sequence with 2pi/3 pulses?
    # 'pulses': [pc.Pulse('x', np.pi), pc.Pulse('y', np.pi)],
    'magnetic_field': [1500, 0, 0], # Magnetic field in Gauss
}

# alpha_and_beta = {
#     'A1': {
#         'alpha': 0.5,
#         'beta': 0.025,
#     },
#     'A2': {
#         'alpha': 0.5,
#         'beta': 0.05,
#     },
# }

# magnetic_field_list = [[500,0,0], [800,0,0], [1200,0,0], [1500,0,0], [2000,0,0], [3000,0,0]]
# magnetic_field_list = [[0,0,500], [0,0,800], [0,0,1200], [0,0,1500], [0,0,2000], [0,0,3000]]
# timespace_list = [np.linspace(0, 5e-2, 201), np.linspace(0, 5e-2, 201), np.linspace(0, 5e-2, 201), np.linspace(0, 1e-1, 201), np.linspace(0, 1e-1, 201), np.linspace(0, 1e-1, 201), ]
# timespace_list = [np.linspace(0, 5e-2, 201), np.linspace(0, 5e-2, 201), np.linspace(0, 5e-2, 201), np.linspace(0, 7e-2, 201), np.linspace(0, 7e-2, 201), np.linspace(0, 7e-2, 201), ]

magnetic_field_list = [[1500,0,0],]
# magnetic_field_list = [[12000,0,0]]
# magnetic_field_list = [[3000,0,0]]
# timespace_list = [np.arange(0, 1e-2, 1e-5), np.arange(0, 2e-2, 2e-5), np.arange(0, 3e-2, 3e-5), np.arange(0, 4e-2, 4e-5), np.arange(0, 5e-2, 5e-5), np.arange(0, 6e-2, 6e-5), ]


# magnetic_field_list = [[500,0,0],[1000,0,0], [2000,0,0]]
# timespace_list = [np.linspace(0, 1e-1, 201),np.linspace(0, 1e-1, 201),np.linspace(0, 1e-1, 201),] # 2e-4

# magnetic_results = {}
# for conc in concentration_list:
#     magnetic_results[conc] = runner(
#                         concentration_value=conc,
#                         changing_variable='magnetic_field', variable_values=magnetic_field_list,
#                         num_spins=2,# spin_type='electron',
#                         bath_parameters=default_bath_parameters, simulator_parameters=default_simulator_parameters, calc_parameters=default_calc_parameters,
#                         # changing_variable2='timespace', variable_values2=timespace_list,
#                         )


# magnetic_nbstates_convergence = {}
# for conc in concentration_list:
#     magnetic_nbstates_convergence[conc] = runner(concentration_value=conc,
#                         changing_variable='nbstates', variable_values=nbstates_list,
#                         bath_parameters=default_bath_parameters, simulator_parameters=default_simulator_parameters, calc_parameters=default_calc_parameters,)


#####################################################################

# cell_size_results = {}
# for conc in concentration_list:
#     cell_size_results[conc] = runner(concentration_value=conc,
#                         changing_variable='cell_size', variable_values=cell_size_list,
#                         bath_parameters=default_bath_parameters, simulator_parameters=default_simulator_parameters, calc_parameters=default_calc_parameters,)

# if rank == 0:
#     print("cell_size results done")

order_results = {}
for conc in concentration_list:
    order_results[conc] = runner(
                        concentration_value=conc,
                        changing_variable='order', variable_values=order_list,
                        num_spins=2,# spin_type='electron',
                        bath_parameters=default_bath_parameters, simulator_parameters=default_simulator_parameters, calc_parameters=default_calc_parameters,
                        # changing_variable2='timespace', variable_values2=timespace_list,
                        )
    
print("Order results done")

# r_bath_results = {}
# for conc in concentration_list:
#     r_bath_results[conc] = runner(
#                         concentration_value=conc,
#                         changing_variable='r_bath', variable_values=r_bath_list,
#                         num_spins=2,# spin_type='nuclear',
#                         bath_parameters=default_bath_parameters, simulator_parameters=default_simulator_parameters, calc_parameters=default_calc_parameters,
#                         # changing_variable2='timespace', variable_values2=timespace_list,
#                         )

# if rank == 0:
#     print("r_bath results done")

# r_dipole_results = {}
# for conc in concentration_list:
#     r_dipole_results[conc] = runner(concentration_value=conc,
#                         changing_variable='r_dipole', variable_values=r_dipole_list,
#                         num_spins=2,# spin_type='nuclear',
#                         bath_parameters=default_bath_parameters, simulator_parameters=default_simulator_parameters, calc_parameters=default_calc_parameters,
#                         # changing_variable2='timespace', variable_values2=timespace_list,
#                         )

# if rank == 0:
#     print("r_dipole results done")

#####################################################################

# Print time taken
end = time.time()
path = "VOTPP folder/Results/Pickle files/"

# Save this data to an external file
if rank == 0:
    print("Time taken: {} seconds".format(end - start))

    # Saving results

    # with open((str(path) + 'magnetic_nbstates_convergence.pkl'), 'wb') as f:
    #     pickle.dump(magnetic_nbstates_convergence, f)

    # with open((str(path) + 'magnetic_results.pkl'), 'wb') as f:
    #     pickle.dump(magnetic_results, f)


    with open((str(path) + 'order_results.pkl'), 'wb') as f:
        pickle.dump(order_results, f)
    
    # with open((str(path) + 'cell_size_results.pkl'), 'wb') as f:
    #     pickle.dump(cell_size_results, f)

    # with open((str(path) + 'r_bath_results.pkl'), 'wb') as f:
    #     pickle.dump(r_bath_results, f)
    
    # with open((str(path) + 'r_dipole_results.pkl'), 'wb') as f:
    #     pickle.dump(r_dipole_results, f)




# if rank == 0:
#     # Plot these results
#     plt.figure()
#     plt.plot(order_results)
#     plt.legend(order_results.columns)
#     plt.xlabel('Time (ms)')
#     plt.ylabel('Coherence')
#     plt.title('Convergence test for order')
#     plt.show()
