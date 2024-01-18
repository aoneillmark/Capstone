from VOTPP_class import VOTPP_class
from mpi4py import MPI
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import pickle
import pycce as pc
import copy
import pprint

# Start timer
start = time.time()

# Get mpi rank
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

# File path for pickle files
path = "VOTPP folder/Results/Pickle files/"

def convert_to_key(value):
    if isinstance(value, list):
        return tuple(value)
    else:
        return value


def update_parameters(changing_variable, value, bath_parameters, simulator_parameters):
    if changing_variable == 'cell_size':
        bath_parameters['cell_size'] = value
    elif changing_variable == 'r_bath':
        simulator_parameters['r_bath'] = value
    elif changing_variable == 'r_dipole':
        simulator_parameters['r_dipole'] = value
    elif changing_variable == 'order':
        simulator_parameters['order'] = value
    elif changing_variable == 'magnetic_field':
        simulator_parameters['magnetic_field'] = value
    elif changing_variable == 'pulses':
        simulator_parameters['pulses'] = value
    elif changing_variable == 'nbstates':
        simulator_parameters['nbstates'] = value
    


def get_changing_invalue(changing_variable, param_dicts):
    for param_dict in param_dicts:
        try:
            return param_dict[changing_variable]
        except KeyError:
            continue
    raise ValueError("'" + changing_variable + "' not found in any of the provided dictionaries.")


def setup_simulator(concentration_value, bath_parameters, simulator_parameters, num_spins, spin_type, alpha, beta):
    bath_parameters['concentration'] = concentration_value
    simulator = VOTPP_class(num_spins=num_spins, spin_type=spin_type, alpha=alpha, beta=beta, **bath_parameters)
    sim_original = simulator.setup_simulator(**simulator_parameters)
    if rank == 0:
        print(sim_original)
        # call = simulator.visualize_cluster(sim_original)
    return simulator, sim_original


def run_single_simulation(concentration_value, bath_parameters, simulator_parameters, calc_parameters, num_spins, spin_type, alpha, beta, changing_variable, value, changing_variable2=None, variable_value2=None):
    update_parameters(changing_variable, value, bath_parameters, simulator_parameters)
    simulator, sim_original = setup_simulator(concentration_value, bath_parameters, simulator_parameters, num_spins, spin_type, alpha, beta)

    # Print number of active nuclei
    if rank == 0:
        num_active_nuclei = simulator.get_number_of_active_nuclei(sim_original.bath, simulator_parameters['r_bath'])
        print(f"Number of active nuclei: {num_active_nuclei}")

        print("Bath: ")
        call = simulator.print_bath(sim_original)

        print("Bath SpinDict: ")
        for key, value in sim_original.bath.types.items():
            print(f"{value}")

    # Run the simulation and return the result
    return sim_original.compute(**calc_parameters)


def process_results(simulation_results, calc_parameters):
    results = {}
    for v_key, result in simulation_results.items():
        results[v_key] = pd.DataFrame([result], columns=calc_parameters['timespace']).T
    return results


def runner(concentration_value, changing_variable, variable_values, bath_parameters, simulator_parameters, calc_parameters, num_spins=2, spin_type=None, alpha=None, beta=None, changing_variable2=None, variable_values2=None):
    changing_invalue = get_changing_invalue(changing_variable, [simulator_parameters, bath_parameters, calc_parameters])

    simulation_results = {}
    for v in variable_values:
        if rank == 0:
            print(f"Running simulation with {changing_variable}: {v}")

        # Use a hashable key
        v_key = convert_to_key(v)
        simulation_results[v_key] = run_single_simulation(concentration_value, bath_parameters, simulator_parameters, calc_parameters, num_spins, spin_type, alpha, beta, changing_variable, v, changing_variable2, variable_values2)

    results = process_results(simulation_results, calc_parameters)

    # Reset the original values of the changing variables
    if changing_variable in bath_parameters:
        bath_parameters[changing_variable] = changing_invalue
    elif changing_variable in simulator_parameters:
        simulator_parameters[changing_variable] = changing_invalue

    if changing_variable2 in calc_parameters:
        calc_parameters[changing_variable2] = variable_values2[0] if variable_values2 else None

    return results


#####################################################################
# Define the parameters to be used in the simulation
#####################################################################

# concentration_list = [0, 0.02, 0.05, 0.1, 0.2, 0.5]
# concentration_list = [0.02, 0.05,]
# concentration_list = [0.02]
order_list = [1, 2, 3] 
# r_bath_list = [40, 80, 160, 220]
# r_dipole_list = [20, 40, 60, 100, 140, 180]
# r_bath_list = [10, 15, 20, 25, 30, 35, 40, 45]
r_bath_list = [5, 10, 15, 20]
# r_dipole_list = [5, 10, 15, 20, 25, 30]
r_dipole_list = [2, 5, 10, 15]
# cell_size_list = [60,100,200]
cell_size_list = [60,100,200]
nbstates_list = [128,]
# nbstates_list = [16,32,64]

# r_bath_list = [10,]
# r_dipole_list = [8,]
# cell_size_list = [60,]

timespace_absolute = np.linspace(0, 0.1, 201)

hahn_echo_sequence = pc.Sequence([ 
    pc.Pulse(axis='x', angle='2*pi/3', delay=np.zeros(timespace_absolute.size),),
    pc.Pulse(axis='x', angle='2*pi/3', delay=timespace_absolute/2,)])

default_calc_parameters = {
    'timespace': timespace_absolute, # 7e-2
    'method': 'gcce',
    # 'pulses': [('x', ((2*np.pi)/3), timespace_absolute / 2), ('x', ((2*np.pi)/3), timespace_absolute / 2)], # Paper defines a Hahn-echo pulse sequence with 2pi/3 pulses?
    # 'pulses': [('x', np.pi, timespace_absolute / 2), ('x', np.pi, timespace_absolute / 2)], # Paper defines a Hahn-echo pulse sequence with 2pi/3 pulses?
    'nbstates': 32, #!
    'quantity': 'coherence',
    'parallel': True,
    'parallel_states': True,
}

default_bath_parameters = {
    'concentration': 1111, #!
    'cell_size': 100, #!
    'seed': 8000
}

default_simulator_parameters = { ########## These should be greater when simulating with HPC
    'order': 2, #!
    'r_bath': 10, # C,
    'r_dipole': 10, # C,
    
    # 'r_bath': 10, # N,
    # 'r_dipole': 10, # N,

    # 'r_bath': 10, # H,
    # 'r_dipole': 10, # H,
    'magnetic_field': [0, 0, 3000], # Magnetic field in Gauss
    'pulses': hahn_echo_sequence,
}

# magnetic_field_list = [[0,0,3000]]
magnetic_field_list = [[0,0,500], [0,0,800], [0,0,1200], [0,0,1500], [0,0,2000], [0,0,3000]]
# bath_type_list = ['H']
bath_type_list = ['C', 'N', 'H']
for bath_type in bath_type_list:
    if bath_type == 'C':
        concentration_list = [1313,]
        default_bath_parameters['concentration'] = 1313
        r_bath_list = [20, 30, 40,50]
        r_dipole_list = [10, 20, 30,40]
        default_bath_parameters['cell_size'] = 500
        default_simulator_parameters['r_bath'] = 40
        default_simulator_parameters['r_dipole'] = 30

    elif bath_type == 'N':
        concentration_list = [1414,]
        default_bath_parameters['concentration'] = 1414
        r_bath_list = [10, 15,20]
        r_dipole_list = [5, 10, 15,20]
        default_bath_parameters['cell_size'] = 100
        default_simulator_parameters['r_bath'] = 15
        default_simulator_parameters['r_dipole'] = 5

    elif bath_type == 'H':
        concentration_list = [1111,]
        default_bath_parameters['concentration'] = 1111
        r_bath_list = [20, 25, 30]
        r_dipole_list = [15, 20, 25]
        default_bath_parameters['cell_size'] = 100
        default_simulator_parameters['r_bath'] = 10
        default_simulator_parameters['r_dipole'] = 10

#####################################################################
# Set up runner and run the simulation
#####################################################################


# # seed_list = [8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000]
# seed_list = [8000,]
# magnetic_results = {}
# for idx, seed in enumerate(seed_list):
#     # Change the value of the seed in the default parameters
#     default_bath_parameters['seed'] = seed # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#     if rank == 0:
#         print("Seed: {}".format(seed)) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#         # Start timer
#         start = time.time()

#     # # Change the value of the r_bath in the default parameters
#     # default_simulator_parameters['r_bath'] = r_bath_conc_list[idx]



#     magnetic_results[seed] = runner(
#                         concentration_value=default_bath_parameters['concentration'],
#                         # concentration_value=seed, # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#                         changing_variable='magnetic_field', variable_values=magnetic_field_list,
#                         num_spins=2, #spin_type='nuclear',
#                         alpha = 4, #
#                         beta = 5, #
#                         bath_parameters=default_bath_parameters, simulator_parameters=default_simulator_parameters, calc_parameters=default_calc_parameters,
#                         # changing_variable2='r_bath', variable_values2=r_bath_conc_list,
#                         )
    
#     if rank == 0:
#         # Print time
#         end = time.time()
#         print("Time taken: {} seconds".format(end - start))

#     # Save the current state of results
#     with open((str(path) + f'magnetic_results_AB3_{bath_type}_{idx}.pkl'), 'wb') as f:
#         pickle.dump(magnetic_results, f)

# if rank == 0:
#     with open((str(path) + f'[n-e]-(n)_{bath_type}_AB3.pkl'), 'wb') as f:
#         pickle.dump(magnetic_results, f)
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################

    # # Runner loop for single pulse sequence ##############################################################
    seed_list = [8000,]
    magnetic_results = {}
    for idx, seed in enumerate(seed_list):
        # Change the value of the seed in the default parameters
        default_bath_parameters['seed'] = seed # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if rank == 0:
            print("Seed: {}".format(seed)) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            print("AB1")

            # Start timer
            start = time.time()

        # # Change the value of the r_bath in the default parameters
        # default_simulator_parameters['r_bath'] = r_bath_conc_list[idx]



        magnetic_results[seed] = runner(
                            concentration_value=default_bath_parameters['concentration'],
                            # concentration_value=seed, # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                            changing_variable='magnetic_field', variable_values=magnetic_field_list,
                            num_spins=2, #spin_type='nuclear',
                            alpha = 2, #
                            beta = 3, #
                            bath_parameters=default_bath_parameters, simulator_parameters=default_simulator_parameters, calc_parameters=default_calc_parameters,
                            # changing_variable2='r_bath', variable_values2=r_bath_conc_list,
                            )
        
        if rank == 0:
            # Print time
            end = time.time()
            print("Time taken: {} seconds".format(end - start))

        # Save the current state of results
        with open((str(path) + f'magnetic_results_AB1_{bath_type}_{idx}.pkl'), 'wb') as f:
            pickle.dump(magnetic_results, f)

    if rank == 0:
        with open((str(path) + f'[n-e]-(n)_{bath_type}_AB1.pkl'), 'wb') as f:
            pickle.dump(magnetic_results, f)

    ########################################################################################################################################
    ########################################################################################################################################
    seed_list = [8000,]
    magnetic_results = {}
    for idx, seed in enumerate(seed_list):
        # Change the value of the seed in the default parameters
        default_bath_parameters['seed'] = seed # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if rank == 0:
            print("Seed: {}".format(seed)) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            print("AB2")

            # Start timer
            start = time.time()

        # # Change the value of the r_bath in the default parameters
        # default_simulator_parameters['r_bath'] = r_bath_conc_list[idx]



        magnetic_results[seed] = runner(
                            concentration_value=default_bath_parameters['concentration'],
                            # concentration_value=seed, # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                            changing_variable='magnetic_field', variable_values=magnetic_field_list,
                            num_spins=2, #spin_type='nuclear',
                            alpha = 3, #
                            beta = 4, #
                            bath_parameters=default_bath_parameters, simulator_parameters=default_simulator_parameters, calc_parameters=default_calc_parameters,
                            # changing_variable2='r_bath', variable_values2=r_bath_conc_list,
                            )
        
        if rank == 0:
            # Print time
            end = time.time()
            print("Time taken: {} seconds".format(end - start))

        # Save the current state of results
        with open((str(path) + f'magnetic_results_AB2_{bath_type}_{idx}.pkl'), 'wb') as f:
            pickle.dump(magnetic_results, f)

    if rank == 0:
        with open((str(path) + f'[n-e]-(n)_{bath_type}_AB2.pkl'), 'wb') as f:
            pickle.dump(magnetic_results, f)


    ########################################################################################################################################
    ########################################################################################################################################
    seed_list = [8000,]
    magnetic_results = {}
    for idx, seed in enumerate(seed_list):
        # Change the value of the seed in the default parameters
        default_bath_parameters['seed'] = seed # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if rank == 0:
            print("Seed: {}".format(seed)) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            print("AB3")

            # Start timer
            start = time.time()

        # # Change the value of the r_bath in the default parameters
        # default_simulator_parameters['r_bath'] = r_bath_conc_list[idx]



        magnetic_results[seed] = runner(
                            concentration_value=default_bath_parameters['concentration'],
                            # concentration_value=seed, # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                            changing_variable='magnetic_field', variable_values=magnetic_field_list,
                            num_spins=2, #spin_type='nuclear',
                            alpha = 4, #
                            beta = 5, #
                            bath_parameters=default_bath_parameters, simulator_parameters=default_simulator_parameters, calc_parameters=default_calc_parameters,
                            # changing_variable2='r_bath', variable_values2=r_bath_conc_list,
                            )
        
        if rank == 0:
            # Print time
            end = time.time()
            print("Time taken: {} seconds".format(end - start))

        # Save the current state of results
        with open((str(path) + f'magnetic_results_AB3_{bath_type}_{idx}.pkl'), 'wb') as f:
            pickle.dump(magnetic_results, f)

    if rank == 0:
        with open((str(path) + f'[n-e]-(n)_{bath_type}_AB3.pkl'), 'wb') as f:
            pickle.dump(magnetic_results, f)


    ########################################################################################################################################
    ########################################################################################################################################
    seed_list = [8000,]
    magnetic_results = {}
    for idx, seed in enumerate(seed_list):
        # Change the value of the seed in the default parameters
        default_bath_parameters['seed'] = seed # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if rank == 0:
            print("Seed: {}".format(seed)) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            print("AB4")

            # Start timer
            start = time.time()

        # # Change the value of the r_bath in the default parameters
        # default_simulator_parameters['r_bath'] = r_bath_conc_list[idx]



        magnetic_results[seed] = runner(
                            concentration_value=default_bath_parameters['concentration'],
                            # concentration_value=seed, # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                            changing_variable='magnetic_field', variable_values=magnetic_field_list,
                            num_spins=2, #spin_type='nuclear',
                            alpha = 5, #
                            beta = 6, #
                            bath_parameters=default_bath_parameters, simulator_parameters=default_simulator_parameters, calc_parameters=default_calc_parameters,
                            # changing_variable2='r_bath', variable_values2=r_bath_conc_list,
                            )
        
        if rank == 0:
            # Print time
            end = time.time()
            print("Time taken: {} seconds".format(end - start))

        # Save the current state of results
        with open((str(path) + f'magnetic_results_AB4_{bath_type}_{idx}.pkl'), 'wb') as f:
            pickle.dump(magnetic_results, f)

    if rank == 0:
        with open((str(path) + f'[n-e]-(n)_{bath_type}_AB4.pkl'), 'wb') as f:
            pickle.dump(magnetic_results, f)


    ########################################################################################################################################
    ########################################################################################################################################
    seed_list = [8000,]
    magnetic_results = {}
    for idx, seed in enumerate(seed_list):
        # Change the value of the seed in the default parameters
        default_bath_parameters['seed'] = seed # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if rank == 0:
            print("Seed: {}".format(seed)) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            print("AB5")

            # Start timer
            start = time.time()

        # # Change the value of the r_bath in the default parameters
        # default_simulator_parameters['r_bath'] = r_bath_conc_list[idx]



        magnetic_results[seed] = runner(
                            concentration_value=default_bath_parameters['concentration'],
                            # concentration_value=seed, # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                            changing_variable='magnetic_field', variable_values=magnetic_field_list,
                            num_spins=2, #spin_type='nuclear',
                            alpha = 9, #
                            beta = 10, #
                            bath_parameters=default_bath_parameters, simulator_parameters=default_simulator_parameters, calc_parameters=default_calc_parameters,
                            # changing_variable2='r_bath', variable_values2=r_bath_conc_list,
                            )
        
        if rank == 0:
            # Print time
            end = time.time()
            print("Time taken: {} seconds".format(end - start))

        # Save the current state of results
        with open((str(path) + f'magnetic_results_AB5_{bath_type}_{idx}.pkl'), 'wb') as f:
            pickle.dump(magnetic_results, f)

    if rank == 0:
        with open((str(path) + f'[n-e]-(n)_{bath_type}_AB5.pkl'), 'wb') as f:
            pickle.dump(magnetic_results, f)

    ########################################################################################################################################
    ########################################################################################################################################
    seed_list = [8000,]
    magnetic_results = {}
    for idx, seed in enumerate(seed_list):
        # Change the value of the seed in the default parameters
        default_bath_parameters['seed'] = seed # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if rank == 0:
            print("Seed: {}".format(seed)) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            print("AB6")

            # Start timer
            start = time.time()

        # # Change the value of the r_bath in the default parameters
        # default_simulator_parameters['r_bath'] = r_bath_conc_list[idx]



        magnetic_results[seed] = runner(
                            concentration_value=default_bath_parameters['concentration'],
                            # concentration_value=seed, # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                            changing_variable='magnetic_field', variable_values=magnetic_field_list,
                            num_spins=2, #spin_type='nuclear',
                            alpha = 11, #
                            beta = 12, #
                            bath_parameters=default_bath_parameters, simulator_parameters=default_simulator_parameters, calc_parameters=default_calc_parameters,
                            # changing_variable2='r_bath', variable_values2=r_bath_conc_list,
                            )
        
        if rank == 0:
            # Print time
            end = time.time()
            print("Time taken: {} seconds".format(end - start))

        # Save the current state of results
        with open((str(path) + f'magnetic_results_AB6_{bath_type}_{idx}.pkl'), 'wb') as f:
            pickle.dump(magnetic_results, f)

    if rank == 0:
        with open((str(path) + f'[n-e]-(n)_{bath_type}_AB6.pkl'), 'wb') as f:
            pickle.dump(magnetic_results, f)


    ########################################################################################################################################
    ########################################################################################################################################
    seed_list = [8000,]
    magnetic_results = {}
    for idx, seed in enumerate(seed_list):
        # Change the value of the seed in the default parameters
        default_bath_parameters['seed'] = seed # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if rank == 0:
            print("Seed: {}".format(seed)) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            print("AB7")

            # Start timer
            start = time.time()

        # # Change the value of the r_bath in the default parameters
        # default_simulator_parameters['r_bath'] = r_bath_conc_list[idx]



        magnetic_results[seed] = runner(
                            concentration_value=default_bath_parameters['concentration'],
                            # concentration_value=seed, # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                            changing_variable='magnetic_field', variable_values=magnetic_field_list,
                            num_spins=2, #spin_type='nuclear',
                            alpha = 12, #
                            beta = 13, #
                            bath_parameters=default_bath_parameters, simulator_parameters=default_simulator_parameters, calc_parameters=default_calc_parameters,
                            # changing_variable2='r_bath', variable_values2=r_bath_conc_list,
                            )
        
        if rank == 0:
            # Print time
            end = time.time()
            print("Time taken: {} seconds".format(end - start))

        # Save the current state of results
        with open((str(path) + f'magnetic_results_AB7_{bath_type}_{idx}.pkl'), 'wb') as f:
            pickle.dump(magnetic_results, f)

    if rank == 0:
        with open((str(path) + f'[n-e]-(n)_{bath_type}_AB7.pkl'), 'wb') as f:
            pickle.dump(magnetic_results, f)
    #####################################################################

############################################################################################################################
############################################################################################################################
############################################################################################################################
############################################################################################################################
############################################################################################################################

# cell_size_results = {}
# for idx, conc in enumerate(concentration_list):
#     cell_size_results[conc] = runner(
#                         concentration_value=conc,
#                         changing_variable='cell_size', variable_values=cell_size_list,
#                         num_spins=2, #spin_type='nuclear',
#                         alpha= [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
#                         beta=  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
#                         bath_parameters=default_bath_parameters, simulator_parameters=default_simulator_parameters, calc_parameters=default_calc_parameters,
#                         # changing_variable2='timespace', variable_values2=timespace_list,
#                         )
#     # Save the current state of results
#     with open((str(path) + f'cell_size_results_{idx}.pkl'), 'wb') as f:
#         pickle.dump(cell_size_results, f)

# if rank == 0:
#     print("cell_size results done")


    # r_bath_results = {}
    # for idx, conc in enumerate(concentration_list):
    #     if rank == 0:
    #         # Start timer
    #         start = time.time()

    #     r_bath_results[conc] = runner(
    #                         concentration_value=conc,
    #                         changing_variable='r_bath', variable_values=r_bath_list,
    #                         num_spins=2, #spin_type='nuclear',
    #                         alpha= 4,
    #                         beta=  5,
    #                         bath_parameters=default_bath_parameters, simulator_parameters=default_simulator_parameters, calc_parameters=default_calc_parameters,
    #                         # changing_variable2='timespace', variable_values2=timespace_list,
    #                         )
        
    #     if rank == 0:
    #         # Print time
    #         end = time.time()
    #         print("Time taken: {} seconds".format(end - start))
        
    #     # Save the current state of results
    #     with open((str(path) + f'r_bath_results_n_HPC_{bath_type}_{idx}.pkl'), 'wb') as f:
    #         pickle.dump(r_bath_results, f)

    # if rank == 0:
    #     with open((str(path) + f'[n-e]-(n)_r_bath_HPC_{bath_type}_results.pkl'), 'wb') as f:
    #         pickle.dump(r_bath_results, f)

    #     print("r_bath results done")
            



    # r_dipole_results = {}
    # for idx, conc in enumerate(concentration_list):
    #     if rank == 0:
    #         # Start timer
    #         start = time.time()

    #     r_dipole_results[conc] = runner(concentration_value=conc,
    #                         changing_variable='r_dipole', variable_values=r_dipole_list,
    #                         num_spins=2, #spin_type='nuclear',
    #                         alpha= 4,
    #                         beta=  5,
    #                         bath_parameters=default_bath_parameters, simulator_parameters=default_simulator_parameters, calc_parameters=default_calc_parameters,
    #                         # changing_variable2='timespace', variable_values2=timespace_list,
    #                         )
        
    #     if rank == 0:
    #         # Print time
    #         end = time.time()
    #         print("Time taken: {} seconds".format(end - start))

    #     # Save the current state of results
    #     with open((str(path) + f'r_dipole_results_n_HPC_{bath_type}_{idx}.pkl'), 'wb') as f:
    #         pickle.dump(r_dipole_results, f)

    # if rank == 0:
    #     with open((str(path) + f'[n-e]-(n)_r_dipole_HPC_{bath_type}_results.pkl'), 'wb') as f:
    #         pickle.dump(r_dipole_results, f)

    #     print("r_dipole results done")


# order_results = {}
# for idx, conc in enumerate(concentration_list):
#     if rank == 0:
#         # Start timer
#         start = time.time()

#     order_results[conc] = runner(
#                         concentration_value=conc,
#                         changing_variable='order', variable_values=order_list,
#                         num_spins=2,# spin_type='electron',
#                         alpha=4,
#                         beta=5,
#                         bath_parameters=default_bath_parameters, simulator_parameters=default_simulator_parameters, calc_parameters=default_calc_parameters,
#                         # changing_variable2='timespace', variable_values2=timespace_list,
#                         )
    
#     if rank == 0:
#         # Print time
#         end = time.time()
#         print("Time taken: {} seconds".format(end - start))
    
#     # Save the current state of results
#     with open((str(path) + f'order_results_n_{bath_type}_{idx}.pkl'), 'wb') as f:
#         pickle.dump(order_results, f)

# if rank == 0:
#     with open((str(path) + f'[n-e]-(n)_order_{bath_type}_results.pkl'), 'wb') as f:
#         pickle.dump(order_results, f)

#     print("order results done")

#####################################################################
# Save the results
#####################################################################

# Print time taken
end = time.time()

# Save this data to an external file
if rank == 0:
    print("Time taken: {} seconds".format(end - start))

    # Saving results

    # with open((str(path) + 'magnetic_nbstates_convergence.pkl'), 'wb') as f:
    #     pickle.dump(magnetic_nbstates_convergence, f)

    # with open((str(path) + '[n-e]-(n).pkl'), 'wb') as f:
    #     pickle.dump(magnetic_results, f)
    
    # with open((str(path) + 'alphabeta_order2_results.pkl'), 'wb') as f:
    #     pickle.dump(alphabeta_results, f)


    # with open((str(path) + 'order_results.pkl'), 'wb') as f:
    #     pickle.dump(order_results, f)
    
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
