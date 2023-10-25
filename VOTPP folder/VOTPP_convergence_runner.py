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
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

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

r_bath_list = [10,]
r_dipole_list = [8,]
cell_size_list = [60,]


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

# magnetic_field_list = [[500,0,0], [800,0,0], [1200,0,0], [1500,0,0], [2000,0,0], [2900,0,0]]
# magnetic_results = {}
# for conc in concentration_list:
#     magnetic_results[conc] = runner(concentration_value=conc,
#                         changing_variable='magnetic_field', variable_values=magnetic_field_list,
#                         bath_parameters=default_bath_parameters, simulator_parameters=default_simulator_parameters, calc_parameters=default_calc_parameters,)

# magnetic_nbstates_convergence = {}
# for conc in concentration_list:
#     magnetic_nbstates_convergence[conc] = runner(concentration_value=conc,
#                         changing_variable='nbstates', variable_values=nbstates_list,
#                         bath_parameters=default_bath_parameters, simulator_parameters=default_simulator_parameters, calc_parameters=default_calc_parameters,)


#####################################################################

# order_results = {}
# for conc in concentration_list:
#     order_results[conc] = runner(concentration_value=conc,
#                         changing_variable='order', variable_values=order_list,
#                         bath_parameters=default_bath_parameters, simulator_parameters=default_simulator_parameters, calc_parameters=default_calc_parameters,)

# print("Order results done")

r_bath_results = {}
for conc in concentration_list:
    r_bath_results[conc] = runner(concentration_value=conc,
                        changing_variable='r_bath', variable_values=r_bath_list,
                        bath_parameters=default_bath_parameters, simulator_parameters=default_simulator_parameters, calc_parameters=default_calc_parameters,)

print("r_bath results done")

r_dipole_results = {}
for conc in concentration_list:
    r_dipole_results[conc] = runner(concentration_value=conc,
                        changing_variable='r_dipole', variable_values=r_dipole_list,
                        bath_parameters=default_bath_parameters, simulator_parameters=default_simulator_parameters, calc_parameters=default_calc_parameters,)

print("r_dipole results done")

cell_size_results = {}
for conc in concentration_list:
    cell_size_results[conc] = runner(concentration_value=conc,
                        changing_variable='cell_size', variable_values=cell_size_list,
                        bath_parameters=default_bath_parameters, simulator_parameters=default_simulator_parameters, calc_parameters=default_calc_parameters,)

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

    # with open((str(path) + 'order_results.pkl'), 'wb') as f:
    #     pickle.dump(order_results, f)
    
    with open((str(path) + 'r_bath_results.pkl'), 'wb') as f:
        pickle.dump(r_bath_results, f)
    
    with open((str(path) + 'r_dipole_results.pkl'), 'wb') as f:
        pickle.dump(r_dipole_results, f)

    with open((str(path) + 'cell_size_results.pkl'), 'wb') as f:
        pickle.dump(cell_size_results, f)



# if rank == 0:
#     # Plot these results
#     plt.figure()
#     plt.plot(order_results)
#     plt.legend(order_results.columns)
#     plt.xlabel('Time (ms)')
#     plt.ylabel('Coherence')
#     plt.title('Convergence test for order')
#     plt.show()
