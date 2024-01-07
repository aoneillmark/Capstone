# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 09:53:34 2023

@author: cchog
"""

#importing functions 
import numpy as np
import matplotlib.pyplot as plt
import pycce as pc 
import pandas as pd
from scipy.optimize import curve_fit 
from mpi4py import MPI

rank = MPI.COMM_WORLD.Get_rank()

def fit(x, T2, n):
    '''function to fit coherence curve'''
    y = np.exp(-(((x)/T2)**n))
    return y

# def get_active_nuclei_positions(atoms, r_bath):
#     central_spin_position = [0.50446035, 0.50446035, 0.55872939]
#     # Assuming 'atoms' is a BathArray object
#     # Extract positions and types of nuclei from the supercell
#     positions = atoms['xyz']  # Positions of bath spins
#     types = atoms['N']  # Types of bath spins

#     # Debug print
#     print(f"Central spin position: {central_spin_position}")

#     # Check for any atom overlapping with the central spin
#     for i, (pos, atom_type) in enumerate(zip(positions, types)):
#         if np.allclose(pos, central_spin_position, atol=1e-3):  # Increased atol
#             print(f"Warning: An atom of type {atom_type} at index {i} overlaps with the central spin.")

#     # Calculate distances from the central spin
#     distances = np.linalg.norm(positions - np.array(central_spin_position), axis=1)

#     # Find indices of nuclei within the r_bath radius
#     active_indices = np.where(distances <= r_bath)[0]

#     return positions[active_indices]

# def get_number_of_active_nuclei(atoms, r_bath):
#     # central_spin_position = [0.50446035, 0.50446035, 0.55872939]
#     active_positions = get_active_nuclei_positions(atoms, r_bath)
#     return len(active_positions)

def print_bath(calc):
    # with np.printoptions(threshold=np.inf):
        # print(np.array(calc.bath.N))
        # print(np.array(calc.bath))
    # from collections import Counter

    # element_counts = Counter(calc.bath.N)

    # # Now, element_counts is a dictionary with elements as keys and their counts as values
    # # To print them, you can iterate over this dictionary
    # for element, count in element_counts.items():
    #     print(f"{element}: {count}")
    print(calc)


#import xyz file - enter specific directory
uc = pd.read_csv('VOTPP_opt.xyz', skiprows=2, header=None, delimiter='      ', engine='python')
#seperate columns into numpy arrays
N = np.array(uc[0])
x = np.array(uc[1])
x = np.array(x)
y = np.array(uc[2])
z = np.array(uc[3])
#set up unit cell
sic = pc.BathCell(13.2613, 13.2613, 9.6851, 90, 90, 90, 'deg') #used optimized structure note there is also an x-ray structure with slightly different values
# z axis in cell coordinates
sic.zdir = [0, 0, 1]

#setting concentration
sic.isotopes['V']['51V'] = 0.02 #set to 2%

#populate unit cell with V
sic.add_atoms((N[76], [x[76], y[76], z[76]]), type='angstrom')

#assign position of qubit 
pos1 = x[76], y[76], z[76]
qpos = sic.to_cell(pos1)

#set up the center 
cen = pc.CenterArray(size=1, position=qpos, 
                     spin=7/2, D=-350, gyro=-7.05,  
                    #  alpha=[0,0,0,0,0,1,0,0], beta=[0,0,0,0,1,0,0,0]) #set to -3/2 to -1/2
                    alpha=4, beta=5) #set to -3/2 to -1/2

#parameters
nb = 10 # set to 0 because of issues using random bath sampling 
ts = np.linspace(0, 0.1, 201) #expected T2 of 0.138ms
cell = 500 #cell size 

#dictionary of parameters
# parameters = dict(
#     order=2, # CCE order
#     r_bath=200,  # Size of the bath in A
#     r_dipole=180,  # Cutoff of pairwise clusters in A
#     pulses = 1, # N pulses in CPMG sequence
#     magnetic_field=[0,0,10000] #set to 1T
# ) 

# pulse = pc.Pulse('x', np.pi)
pulse_bath = pc.Pulse(axis='z', angle=np.pi, delay=None,
                     bath_names='51V', bath_axes='z', bath_angles=np.pi)  # 120° pulse around x-axis applied to bath spins
hahn_echo_sequence = pc.Sequence([
                                # pulse_central, 
                                pulse_bath,
                                # pulse_central, 
                                # pulse_bath,
                                ])

parameters = {
    'order': 2, # CCE order
    'r_bath': 100,  # Size of the bath in A
    'r_dipole': 100,  # Cutoff of pairwise clusters in A
    'pulses': hahn_echo_sequence, # N pulses in CPMG sequence
    'magnetic_field': [3000,3000,0], # Magnetic field in Gauss
    }

#calculating T2 using 50 random seeds
n = 1 #counter to keep track
no_sims = 1 #set number of sims to average over
sims = []
for i in range(no_sims):
    
    #generte bath
    # rand = np.random.randint(0, 8000) # chooses random integer from 0 to 8000 
    rand = 8000
    atoms = sic.gen_supercell(cell, remove = [('V', qpos)], seed =rand) #generate supercell 
    #set spin |gyro | quadrupole 
    spin_types = ['51V',  1/2, -17608.59705]   #electronic bath
    atoms.add_type(*spin_types)

    #set up simulation
    calc = pc.Simulator(spin=cen, bath=atoms, **parameters)

    # print('Number of active nuclei:', get_number_of_active_nuclei(calc.bath, parameters['r_bath']))
    if rank == 0:
        print_bath(calc)

    #run simulation
    sim = calc.compute(ts, nbstates = nb,
                        method='gcce', quantity='coherence',
                        parallel=True,
                        parallel_states=True,
                        )
    
    #use and interate counter to keep track of progress
    #print(n, ' done') # uncomment to print progress
    n += 1
    
    if rank ==0:
        #plot this run and store it 
        plt.plot(ts, abs(sim.real))
        sims.append(abs(sim.real))

if rank == 0:
    #add labels and title to plot     
    plt.xlabel('Time [ms]')
    plt.ylabel('Coherence')
    plt.title('50 coherence curves for n-e 2% cell=500 [2:200:180]')
    plt.ylim(-0.1,1.1)
    plt.show()

    #use all 50 sims to calculate average curve 
    avg = np.zeros(201) #creates array of 0s same length as ts
    for i in range(201): 
        for j in range(no_sims):
            avg[i] += sims[j][i]   
    avg = avg/no_sims

    #plot the average coherence curve 
    plt.plot(ts, avg)
    plt.xlabel('Time [ms]')
    plt.ylabel('Coherence')
    plt.title('Averaged Coherence Curve n-e 2%')
    plt.ylim(-0.1,1.1)
    plt.show()

    #fit coherence curve 
    p_guess= [0.1, 2] #guessing T2 and the power respectively 
    par, cov = curve_fit(fit, ts, avg, p0=p_guess) #find best fitting parameters and covariance matrix
    err = np.sqrt(np.diag(cov)) #calculating the errors based on the fit 
    print('params =', par) # print params

    #plot fitted curve 
    label = 'Fit - T2=' + str(round(par[0]*1e3,3)) + ' \u00B1 ' + str(round(err[0]*1e3,3)) + ' us'
    plt.plot(ts, avg, label='average for 50')
    plt.plot(ts, fit(ts, *par), 'r-', label=label)
    plt.xlabel('Time [ms]')
    plt.ylabel('Coherence')
    plt.title('Averaged Coherence Curve n-e 2%')
    plt.ylim(-0.1,1.1)
    plt.legend()
    plt.show()