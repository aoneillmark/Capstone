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

print("Rank: {}".format(rank))


# Runner to converge parameters
def runner(concentration_value, 
           changing_variable, variable_values, 
           bath_parameters, simulator_parameters, calc_parameters,):
    
    changing_invalue = simulator_parameters[changing_variable] # Save this for later
    bath_parameters['concentration'] = concentration_value # Set concentration value

    simulator = VOTPP_class(**bath_parameters) # Set up bath and atoms
    sim = simulator.setup_simulator(**simulator_parameters) # Set up simulator

    # ls = []
    results = {}
    for v in variable_values: # Iterate through variable values (e.g. order = 1, 2, 3)
        setattr(simulator, changing_variable, v) # Set the variable in the simulator to the value (e.g. simulator['order'] = 1)
        l = sim.compute(**calc_parameters)
        
        # Printing for progress check, listing the variable and value
        print("{}: {}".format(changing_variable, v))
        
        results[v] = pd.DataFrame([l.real], columns=calc_parameters['timespace']).T
        # ls.append(l.real)

    simulator_parameters[changing_variable] = changing_invalue
    # ls = pd.DataFrame(ls, columns=calc_parameters['timespace'], index=variable_values).T
    return results

# concentration_list = [0, 0.02, 0.05, 0.1, 0.2, 0.5]
# concentration_list = [0.02, 0.05,]
concentration_list = [0]
order_list = [1, 2, 3, 4] 
# r_bath_list = [40, 80, 160, 220]
# r_dipole_list = [20, 40, 60, 100, 140, 180]
r_bath_list = [10,20,30]
r_dipole_list = [8,10,12]
# cell_size_list = [60,100,200]
cell_size_list = [60,100,200]
# nbstates_list = [32, 64, 128, 256]
nbstates_list = [16,32,64]

default_calc_parameters = {
    'timespace': np.linspace(0, 1e-2, 201),
    'method': 'cce',
    'pulses': 1,
    'nbstates': 16, #!
    'quantity': 'coherence',
    'parallel': True,
    'parallel_states': True,
}

default_bath_parameters = {
    'concentration': 0, #!
    'cell_size': 100, #!
    'displacement': 0, # Angstrom
    'seed': 8000
}

default_simulator_parameters = { ########## These should be greater when simulating with HPC
    'order': 3, #!
    'r_bath': 20, #!
    'r_dipole': 10, #!
    'pulses': 1, # N pulses in CPMG sequence
    'magnetic_field': [1000, 0, 0], # Magnetic field in Gauss
}

magnetic_field_list = [[500,0,0], [800,0,0], [1200,0,0], [1500,0,0], [2000,0,0], [2900,0,0]]
magnetic_results = {}
for conc in concentration_list:
    magnetic_results[conc] = runner(concentration_value=conc,
                        changing_variable='magnetic_field', variable_values=magnetic_field_list,
                        bath_parameters=default_bath_parameters, simulator_parameters=default_simulator_parameters, calc_parameters=default_calc_parameters,)

#####################################################################

# order_results = {}
# for conc in concentration_list:
#     order_results[conc] = runner(concentration_value=conc,
#                         changing_variable='order', variable_values=order_list,
#                         bath_parameters=default_bath_parameters, simulator_parameters=default_simulator_parameters, calc_parameters=default_calc_parameters,)

# print("Order results done")

# r_bath_results = {}
# for conc in concentration_list:
#     r_bath_results[conc] = runner(concentration_value=conc,
#                         changing_variable='r_bath', variable_values=order_list,
#                         bath_parameters=default_bath_parameters, simulator_parameters=default_simulator_parameters, calc_parameters=default_calc_parameters,)

# print("r_bath results done")

# r_dipole_results = {}
# for conc in concentration_list:
#     r_dipole_results[conc] = runner(concentration_value=conc,
#                         changing_variable='r_dipole', variable_values=order_list,
#                         bath_parameters=default_bath_parameters, simulator_parameters=default_simulator_parameters, calc_parameters=default_calc_parameters,)

# print("r_dipole results done")

#####################################################################

# Print time taken
end = time.time()
print("Time taken: {} seconds".format(end - start))

path = "VOTPP folder/Results/Pickle files/"

# Save this data to an external file
if rank == 0:
    # Saving results
    with open((str(path) + 'order_results.pkl'), 'wb') as f:
        pickle.dump(order_results, f)
    
    with open((str(path) + 'r_bath_results.pkl'), 'wb') as f:
        pickle.dump(r_bath_results, f)
    
    with open((str(path) + 'r_dipole_results.pkl'), 'wb') as f:
        pickle.dump(r_dipole_results, f)



# if rank == 0:
#     # Plot these results
#     plt.figure()
#     plt.plot(order_results)
#     plt.legend(order_results.columns)
#     plt.xlabel('Time (ms)')
#     plt.ylabel('Coherence')
#     plt.title('Convergence test for order')
#     plt.show()
