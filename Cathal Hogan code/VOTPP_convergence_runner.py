from VOTPP_class import VOTPP_class
from mpi4py import MPI
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# Start timer
start = time.time()

# Get mpi rank
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

concentration_list = [0, 0.02, 0.05, 0.1, 0.2, 0.5]
order_list = [1, 2, 3] 
r_bath_list = [40, 80, 160, 220]
r_dipole_list = [20, 40, 60, 100, 140, 180]
cell_size_list = [60,100,200]
nbstates_list = [32, 64, 128, 256]

default_calc_parameters = {
    'timespace': np.linspace(0, 1e-1, 201),
    'method': 'cce',
    'pulses': 1,
    'nbstates': 4, #!
    'quantity': 'coherence',
    'parallel': True,
    'parallel_states': True,
}

default_bath_parameters = {
    'concentration': 0, #!
    'cell_size': 100, #!
    'displacement': 3.75, # Angstrom
    'seed': 8000
}

default_simulator_parameters = { ########## These should be greater when simulating with HPC
    'order': 2, #!
    'r_bath': 20, #!
    'r_dipole': 10, #!
    'pulses': 2, # N pulses in CPMG sequence
    'magnetic_field': [0, 0, 10000], # Magnetic field in Gauss
}


# # default_calc_parameters.update(default_simulator_parameters)
# # print(default_calc_parameters)

# simulator = VOTPP_class(**default_bath_parameters)
# sim = simulator.setup_simulator(**default_simulator_parameters)
# result = sim.compute(**default_calc_parameters)


# if rank == 0:
#     simulator.plot_results(default_calc_parameters['timespace'], result)


# Runner to converge parameters
def runner(concentration_value, 
           changing_variable, variable_values, 
           bath_parameters, simulator_parameters, calc_parameters,):
    
    changing_invalue = simulator_parameters[changing_variable] # Save this for later
    bath_parameters['concentration'] = concentration_value # Set concentration value

    simulator = VOTPP_class(**bath_parameters) # Set up bath and atoms
    sim = simulator.setup_simulator(**simulator_parameters) # Set up simulator

    ls = []
    for v in variable_values: # Iterate through variable values (e.g. order = 1, 2, 3)
        setattr(simulator, changing_variable, v) # Set the variable in the simulator to the value (e.g. simulator['order'] = 1)
        l = sim.compute(**calc_parameters)
        
        # Printing for progress check, listing the variable and value
        print("{}: {}".format(changing_variable, v))
        
        ls.append(l.real)

    simulator_parameters[changing_variable] = changing_invalue
    ls = pd.DataFrame(ls, columns=calc_parameters['timespace'], index=variable_values).T
    return ls

order_results = runner(concentration_value=0,
                       changing_variable='order', variable_values=[1,2,3],
                       bath_parameters=default_bath_parameters, simulator_parameters=default_simulator_parameters, calc_parameters=default_calc_parameters,)

# Print time taken
end = time.time()
print("Time taken: {} seconds".format(end - start))

# Save this data to an external file
if rank == 0:
    order_results.to_csv('order_results.csv')

if rank == 0:
    # Plot these results
    plt.figure()
    plt.plot(order_results)
    plt.legend(order_results.columns)
    plt.xlabel('Time (ms)')
    plt.ylabel('Coherence')
    plt.title('Convergence test for order')
    plt.show()
