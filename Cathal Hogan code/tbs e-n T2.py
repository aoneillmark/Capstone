# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 09:56:20 2023

@author: cchog
"""

#importing functions 
import numpy as np
import matplotlib.pyplot as plt
import pycce as pc 
import pandas as pd
from scipy.optimize import curve_fit 

def fit(x, T2, n):
    '''function used to fit coherence curve'''
    y = np.exp(-(((2*x)/T2)**n))
    return y

def get_active_nuclei_positions(atoms, r_bath):
    central_spin_position = [0.50446035, 0.50446035, 0.55872939]
    # Assuming 'atoms' is a BathArray object
    # Extract positions and types of nuclei from the supercell
    positions = atoms['xyz']  # Positions of bath spins
    types = atoms['N']  # Types of bath spins

    # Debug print
    print(f"Central spin position: {central_spin_position}")

    # Check for any atom overlapping with the central spin
    for i, (pos, atom_type) in enumerate(zip(positions, types)):
        if np.allclose(pos, central_spin_position, atol=1e-3):  # Increased atol
            print(f"Warning: An atom of type {atom_type} at index {i} overlaps with the central spin.")

    # Calculate distances from the central spin
    distances = np.linalg.norm(positions - np.array(central_spin_position), axis=1)

    # Find indices of nuclei within the r_bath radius
    active_indices = np.where(distances <= r_bath)[0]

    return positions[active_indices]

def get_number_of_active_nuclei(atoms, r_bath):
    # central_spin_position = [0.50446035, 0.50446035, 0.55872939]
    active_positions = get_active_nuclei_positions(atoms, r_bath)
    return len(active_positions)

def print_bath(calc):
    # with np.printoptions(threshold=np.inf):
        # print(np.array(calc.bath.N))
        # print(np.array(calc.bath))
    from collections import Counter

    element_counts = Counter(calc.bath.N)

    # Now, element_counts is a dictionary with elements as keys and their counts as values
    # To print them, you can iterate over this dictionary
    for element, count in element_counts.items():
        print(f"{element}: {count}")

#import xyz file
# uc = pd.read_csv(r'C:\Users\cchog\OneDrive\Documents\Spin Decoherence - Summer Internship 23\VoTTP\VOTPP_opt.xyz', skiprows=2, header=None, delimiter='      ', engine='python') #enter file directory 
uc = pd.read_csv(r'VOTPP_opt.xyz', skiprows=2, header=None, delimiter='      ', engine='python') #enter file directory 
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

#set concentration
#sic.isotopes['H']['2H'] = 1 #uncomment for deuterium 

#populate cell
for i in range(len(N)):
    sic.add_atoms((N[i], [x[i], y[i], z[i]]), type='angstrom')

#assign position of qubit 
pos1 = x[76], y[76], z[76]
qpos = sic.to_cell(pos1)

#set up the center 
cen = pc.CenterArray(size=1, position=qpos, spin=1/2,
                     D=0, gyro=-17608.59705, alpha=[0,1], beta=[1,0])

#parameters
nb=25
ts = np.linspace(0, 0.1, 201) #expected T2 of 3.3e-2
cell = 60 #cell size 


#dictionary of parameters
parameters = dict(
    order=2, # CCE order
    r_bath=20,  # Size of the bath in A
    r_dipole=6,  # Cutoff of pairwise clusters in A
    pulses = 1, # N pulses in CPMG sequence
    magnetic_field=[0,0,10000] #set to 1T
) 

#calculating T2 using 50 random seeds
n = 1 #counter to keep track
no_sims = 1 # set number of sims to be run 
sims = []
for i in range(no_sims):
    
    #generte bath
    rand = np.random.randint(0, 8000) # chooses random integer from 0 to 8000 
    atoms = sic.gen_supercell(cell, remove = [('V', qpos)], seed=rand)
    #set spin |gyro | quadrupole 
    spin_types = [('C',  1 / 2,  6.72828),    
                  ('H', 1 / 2, 26.7522),
                  ('N', 1, 1.9331, 20.44 )
                  #('V', 7/2, 7.05, -350)  # not added for consistency between tests
              ]   
    atoms.add_type(*spin_types)

    #set up simulation
    calc = pc.Simulator(spin=cen, bath=atoms, **parameters)

    print('Number of active nuclei:', get_number_of_active_nuclei(calc.bath, parameters['r_bath']))
    print_bath(calc)

    #run simulation
    sim = calc.compute(ts, nbstates = nb,
                        method='cce', quantity='coherence')
    
    #use and interate counter to keep track of progress
    #print(n, ' done') #uncomment to keep track of how many done 
    n += 1
    
    #plot this run and store it 
    plt.plot(ts, sim.real)
    sims.append(sim.real)

#add labels and title to plot     
plt.xlabel('Time [ms]')
plt.ylabel('Coherence')
plt.title('50 coherence curves for e-n cell=60 [2:20:6]')
plt.show()

#using multiple sims to calculate average curve 
avg = np.zeros(201) #creates array of 0s same length as ts
for i in range(201): 
    for j in range(no_sims):
        avg[i] += sims[j][i]   
avg = avg/no_sims

#plot the average coherence curve 
plt.plot(ts, avg)
plt.xlabel('Time [ms]')
plt.ylabel('Coherence')
plt.title('Averaged Coherence Curve e-n')
plt.show()

#fit coherence curve 
p_guess= [0.03, 2] #guessing T2 and the power respectively 
par, cov = curve_fit(fit, ts, avg, p0=p_guess) #find best fitting parameters and covariance matrix
err = np.sqrt(np.diag(cov)) #calculating the errors based on the fit 
print('params =', par) # print params

#plot fitted curve 
label = 'Fit - T2=' + str(round(par[0]*1e3,3)) + ' \u00B1 ' + str(round(err[0]*1e3,3)) + ' us'
plt.plot(ts, avg, label='average for 50')
plt.plot(ts, fit(ts, *par), 'r-', label=label)
plt.xlabel('Time [ms]')
plt.ylabel('Coherence')
plt.title('Averaged Coherence Curve e-n%')
plt.legend()
plt.show()