from VOTPP_class_revised import BathSetup, CenterSetup, SimulatorSetup, RunCalc
from mpi4py import MPI
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import pickle
import pycce as pc
import copy

def get_default_parameters():
    """Returns a dictionary containing the default parameters for the calculation, bath, simulator, and center.
    This is useful for backing up the original parameters before changing them, and restoring them later."""
    return {
        'calc': copy.deepcopy(default_calc_parameters),
        'bath': copy.deepcopy(default_bath_parameters),
        'simulator': copy.deepcopy(default_simulator_parameters),
        'center': copy.deepcopy(default_center_parameters),
    }


def change_variable(variable_name, variable_value):
    """Changes the value of a given variable in the appropriate dictionary."""

    if variable_name == 'magnetic_field': # Special handling for magnetic field since PyCCE takes it as a list
        default_simulator_parameters[variable_name] = [0, 0, variable_value]
    elif variable_name in default_calc_parameters:
        default_calc_parameters[variable_name] = variable_value
    elif variable_name in default_bath_parameters:
        default_bath_parameters[variable_name] = variable_value
    elif variable_name in default_simulator_parameters:
        default_simulator_parameters[variable_name] = variable_value
    elif variable_name in default_center_parameters:
        default_center_parameters[variable_name] = variable_value
    else:
        print(f"Variable {variable_name} not found in default parameters.")


def runner_for_convergence(variable_name, variable_list):
    """Runs the calculation for a list of values of a given variable, 
    and saves all the results into a single file. Additionally, saves temporary results after each variable run."""

    global default_calc_parameters, default_bath_parameters, default_simulator_parameters, default_center_parameters

    # Backup original parameters
    original_parameters = get_default_parameters()

    # Get the current bath type
    bath_type = default_bath_parameters['bath_type']

    # Initialize a dictionary to store results for all values of the variable
    all_variable_results = {}

    for variable in variable_list:
        print(f"Running for {variable_name} = {variable} with {bath_type} bath")

        # Change the variable to the new value
        change_variable(variable_name, variable)

        bath = BathSetup(**default_bath_parameters)
        bath.create_bath()

        center = CenterSetup(qpos=bath.qpos, **default_center_parameters)
        cen = center.create_center()
        
        simulator = SimulatorSetup(center=cen, atoms=bath.atoms, **default_simulator_parameters)
        calc = simulator.setup_simulator()

        run = RunCalc(calc, **default_calc_parameters)
        result = run.run_calculation()

        # Store the result in the dictionary, keyed by the variable value
        all_variable_results[variable] = result

        # Save the results accumulated so far to a temporary file
        temp_filename = f'./Revised_VOTPP/Results_revised/Pickle_files/Convergence/temp/temp_{bath_type}_{variable_name}.pkl'
        with open(temp_filename, 'wb') as f:
            pickle.dump(all_variable_results, f)
        print(f"Saved temporary results for {variable_name} up to {variable} with {bath_type} bath to {temp_filename}")

    # After completing all runs, save all results to a single file
    final_filename = f'./Revised_VOTPP/Results_revised/Pickle_files/Convergence/final/{bath_type}_{variable_name}.pkl'
    with open(final_filename, 'wb') as f:
        pickle.dump(all_variable_results, f)
    print(f"Saved all results for {variable_name} with {bath_type} bath to {final_filename}")
    
    # Restore original parameters after completing all runs
    default_calc_parameters = original_parameters['calc']
    default_bath_parameters = original_parameters['bath']
    default_simulator_parameters = original_parameters['simulator']
    default_center_parameters = original_parameters['center']



def runner_for_alpha_beta_magnetic_fields(alpha_beta_pairs, magnetic_field_list):
    """Runs the calculation for a list of alpha-beta pairs and a list of magnetic fields,
    and saves the results to a file for each pair and each magnetic field."""

    global default_calc_parameters, default_bath_parameters, default_simulator_parameters, default_center_parameters

    # Backup original parameters
    original_parameters = get_default_parameters()

    # Get the current bath type
    bath_type = default_bath_parameters['bath_type']

    for pair_name, (alpha, beta) in alpha_beta_pairs.items():
        # Initialize a dictionary to store results for all magnetic fields for this alpha-beta pair
        all_magnetic_fields_results = {}

        # Set alpha and beta for this run
        default_center_parameters['alpha'] = alpha
        default_center_parameters['beta'] = beta

        for magnetic_field in magnetic_field_list:
            print(f"Running for {pair_name} with alpha={alpha}, beta={beta}, magnetic_field={magnetic_field}, and {bath_type} bath")
            
            # Change magnetic field to the new value
            change_variable('magnetic_field', magnetic_field)

            bath = BathSetup(**default_bath_parameters)
            bath.create_bath()

            center = CenterSetup(qpos=bath.qpos, **default_center_parameters)
            cen = center.create_center()

            simulator = SimulatorSetup(center=cen, atoms=bath.atoms, **default_simulator_parameters)
            calc = simulator.setup_simulator()

            run = RunCalc(calc, **default_calc_parameters)
            result = run.run_calculation()

            # Store the result in the dictionary, keyed by magnetic field
            all_magnetic_fields_results[f'mf{magnetic_field}'] = result

        # After completing all magnetic fields for this alpha-beta pair, save all results to a single file
        filename = f'./Revised_VOTPP/Results_revised/Pickle_files/Simulation/{bath_type}_{pair_name}_alpha{alpha}_beta{beta}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(all_magnetic_fields_results, f)
        print(f"Saved all results for {pair_name} to {filename}")

    # Restore original parameters after completing all runs
    default_calc_parameters = original_parameters['calc']
    default_bath_parameters = original_parameters['bath']
    default_simulator_parameters = original_parameters['simulator']
    default_center_parameters = original_parameters['center']



def adjust_parameters_for_bath_type():
    """This function changes the default values of r_bath and r_dipole based on the bath type,
    as well as the cell size and the range of values for r_bath and r_dipole we consider in convergence tests.
    This is because having r_bath=80 is already quite large for electronic bath, 
    but for nuclear bath, this will result in a massive and impossible simulation.
    These values below were the values used in the original script"""

    global default_bath_parameters, default_simulator_parameters

    bath_type = default_bath_parameters.get('bath_type', 'hydrogen')  # Default to 'hydrogen' if not set

    if bath_type == 'carbon':
        r_bath_list = [20, 30, 40, 50]
        r_dipole_list = [10, 20, 30, 40]
        default_bath_parameters['cell_size'] = 500
        default_simulator_parameters['r_bath'] = 40
        default_simulator_parameters['r_dipole'] = 30

    elif bath_type == 'nitrogen':
        r_bath_list = [10, 15, 20]
        r_dipole_list = [5, 10, 15, 20]
        default_bath_parameters['cell_size'] = 100
        default_simulator_parameters['r_bath'] = 15
        default_simulator_parameters['r_dipole'] = 5

    elif bath_type == 'hydrogen':
        r_bath_list = [20, 25, 30]
        r_dipole_list = [15, 20, 25]
        default_bath_parameters['cell_size'] = 100
        default_simulator_parameters['r_bath'] = 10
        default_simulator_parameters['r_dipole'] = 5

    elif bath_type == 'electronic':
        r_bath_list = [40, 60, 70, 75, 80, 85, 90, 95, 100, 105, 110]
        # r_bath_list = [40, 50, 55] # used for testing to make sure this script works
        r_dipole_list = [40, 50, 55, 60, 65, 70, 75, 80, 85]
        default_bath_parameters['cell_size'] = 1000
        default_simulator_parameters['r_bath'] = 80
        default_simulator_parameters['r_dipole'] = 65
        # default_simulator_parameters['r_bath'] = 40 # used for testing to make sure this script works
        # default_simulator_parameters['r_dipole'] = 20 # used for testing to make sure this script works
    
    # Return lists for potential external use
    return r_bath_list, r_dipole_list


# Define default parameters
timespace = np.linspace(0, 0.1, 201)

hahn_echo_sequence = pc.Sequence([ 
    pc.Pulse(axis='x', angle='2*pi/3', delay=np.zeros(timespace.size),
                bath_names=('51V',
                )), # 120° pulse around x-axis applied to bath spins),
    pc.Pulse(axis='x', angle='2*pi/3', delay=timespace/2,
                bath_names=('51V',
                )) ]) # 120° pulse around x-axis applied to bath spins ])

magnetic_field_list = [500, 800, 1200, 1500, 2000, 3000]

alpha_beta_pairs = {
    'AB1': [2, 3],
    'AB2': [3, 4],
    'AB3': [4, 5],
    'AB4': [5, 6],
    'AB5': [9, 10],
    'AB6': [11, 12],
    'AB7': [12, 13],
}

default_calc_parameters = {
    'timespace': timespace,
    'method': 'gcce',
    'nbstates': 1, 
    'quantity': 'coherence',
    'parallel': True,
    'parallel_states': True,
}

default_bath_parameters = {
    'filepath': './Revised_VOTPP/VOTPP_opt2.xyz',
    'bath_type': 'electronic', # choose between 'electronic', 'hydrogen', 'nitrogen', 'carbon'
    'concentration': 0.02, 
    'cell_size': 100, 
    'seed': 8000
}

default_simulator_parameters = {
    'order': 2,
    'r_bath': 40, 
    'r_dipole': 20,
    'magnetic_field': [0, 0, 3000], # Magnetic field in Gauss
    'pulses': hahn_echo_sequence,
}

default_center_parameters = {
    'atens_file_path': './Revised_VOTPP/VOTPP_opt.Atens',
    'gtens_file_path': './Revised_VOTPP/VOTPP_opt.gtens',
    'spin_type': 'both', # choose between 'electron', 'nuclear', 'both'
    # 'qpos': bath.qpos, we have to wait for the bath to be created :(
    'alpha': 4, 
    'beta': 5,
}

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

########################################################################################

# Example usage for simulating alpha-beta pairs over all magnetic fields for electronic bath
default_bath_parameters['bath_type'] = 'electronic'
r_bath_list, r_dipole_list = adjust_parameters_for_bath_type()
runner_for_alpha_beta_magnetic_fields(alpha_beta_pairs, magnetic_field_list)

# Example usage for simulating convergence tests for r_bath for electronic bath
default_bath_parameters['bath_type'] = 'electronic'
r_bath_list, r_dipole_list = adjust_parameters_for_bath_type()
runner_for_convergence('r_bath', r_bath_list)
# runner_for_convergence('r_dipole', r_dipole_list)
# runner_for_convergence('cell_size', [1000, 2000, 3000, 4000, 5000])
