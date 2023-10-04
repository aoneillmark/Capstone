# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 09:55:13 2023

@author: cchog
"""
#importing functions 
import numpy as np
import matplotlib.pyplot as plt
import pycce as pc 
import pandas as pd
from scipy.optimize import curve_fit 

#function to run convergence tests
def runner(variable, values):
    invalue = parameters[variable]
    calc = pc.Simulator(spin=cen, bath=atoms, **parameters)
    ls = []
    
    for v in values:
        setattr(calc, variable, v)
        l = calc.compute(ts, method='cce',
                         nbstates=nb,
                         quantity='coherence')
        
        
        ls.append(l.real)

    parameters[variable] = invalue
    ls = pd.DataFrame(ls, columns=ts, index=values).T
    return ls

#import xyz file - add specific directory 
uc = pd.read_csv('VOTPP_opt.xyz', skiprows=2, header=None, delimiter='      ', engine='python') #insert directory of file
#seperate columns into numpy arrays
N = np.array(uc[0])
x = np.array(uc[1])
y = np.array(uc[2])
z = np.array(uc[3])
#set up unit cell
sic = pc.BathCell(13.2613, 13.2613, 9.6851, 90, 90, 90, 'deg') #used optimized structure note there is also an x-ray structure with slightly different values
# z axis in cell coordinates
sic.zdir = [0, 0, 1]

#populate cell
for i in range(len(N)):
    sic.add_atoms((N[i], [x[i], y[i], z[i]]), type='angstrom')

#assign position of qubit 
pos1 = x[76], y[76], z[76]
qpos = sic.to_cell(pos1)

#generte supercell
cell = 60
atoms = sic.gen_supercell(cell, seed = 8000) #left fixed for convergence tests to avoid changes
#set spin |gyro | quadrupole 
spin_types = [('C',  1 / 2,  6.72828),    
              ('H', 1 / 2, 26.7522),
              ('N', 1, 1.9331, 20.44 )
              #('V', 7/2, 7.05, -350)  # not added for consistency between tests
              ]   
atoms.add_type(*spin_types)

#set up the center 
cen = pc.CenterArray(size=1, position=qpos, spin=1/2,
                     D=0, gyro=-17608.59705, alpha=[0,1], beta=[1,0])

#parameters
nb=25
ts = np.linspace(0, 0.1, 201) #expected T2 of 3.3e-2

#dictionary of parameters
parameters = dict(
    order=2, # CCE order
    r_bath=20,  # Size of the bath in A
    r_dipole=6,  # Cutoff of pairwise clusters in A
    pulses = 1, # N pulses in CPMG sequence
    magnetic_field=[0,0,10000] #set to 1T
) 

#converge for order 
orders = runner('order', [1,2,3])
orders.plot(title='convergence test for orders; cell=75; nbstates=0; [?:20:10]',
           xlabel='order', ylabel='Coherence')
#plt.show()  # show individually 

#converge for r_bath
rbs = runner('r_bath', [10,20,30])
orders.plot(title='convergence test for r_bath; cell=75; nbstates=0; [2:?:10]',
           xlabel='order', ylabel='Coherence')
#plt.show()  # show individually 

#converge for r_dipole
rds = runner('r_dipole', [4,6,8])
orders.plot(title='convergence test for r_dipole; cell=75; nbstates=0; [2:20:?]',
           xlabel='order', ylabel='Coherence')
#plt.show()  # show individually 

#plot together 
fig, axes = plt.subplots(1, 3, figsize=(12, 3))
orders.plot(ax=axes[0], title='order')
rbs.plot(ax=axes[1], title='r_bath')
rds.plot(ax=axes[2], title='r_dipole')
for ax in axes:    
    ax.set(xlabel='Time (ms)', ylabel='Coherence')
plt.suptitle('Convergence tests; cell = 75copies')
fig.tight_layout()

#converge for cell size 
ls_cell = [] #empty list to collect simulations 
cell_size = [40,60,100] #enter cell sizes to be simulated 
for i in cell_size:
    #generate cell
    atoms_cell = sic.gen_supercell(i, seed = 8000) #left fixed for convergence tests to avoid changes
    #set spin |gyro | quadrupole 
    spin_types = [('C',  1 / 2,  6.72828),    
                    ('H', 1 / 2, 26.7522),
                    ('N', 1, 1.9331, 20.44 )
                    #('V', 7/2, 7.05, -350)  # not added for consistency between tests
                      ]   
    atoms_cell.add_type(*spin_types)

    #set up calculation 
    calc = pc.Simulator(spin=cen, bath=atoms_cell, **parameters)

    #run simulation
    l = calc.compute(ts, method='cce',
                        nbstates=nb,
                        quantity='coherence')
        
    ls_cell.append(l.real)
    
ls_cell_frame = pd.DataFrame(ls_cell, columns=ts, index=cell_size).T
ls_cell_frame.plot(title='Convergence tests for cell size',
          xlabel='Time [ms]', ylabel='Coherence')

#converge nbstates
calc = pc.Simulator(spin=cen, bath=atoms, **parameters)
ls_nb = [] #empty list to collect sims
nbstates = [10,25,50] #enter nbstates to be simulated
for i in nbstates:
    l = calc.compute(ts, method='cce',
                        nbstates=i,
                        quantity='coherence')
    ls_nb.append(l.real)
        
ls_nb_frame = pd.DataFrame(ls_nb, columns=ts, index=nbstates).T
ls_nb_frame.plot(title='Convergence tests for nbstates',
          xlabel='Time [ms]', ylabel='Coherence') 
