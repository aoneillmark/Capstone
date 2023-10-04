# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 11:00:56 2023

@author: cchog
"""
#for 30% concentration 

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
        
        
        #print('a run done') # uncomment to keep track
        ls.append(l.real)

    parameters[variable] = invalue
    ls = pd.DataFrame(ls, columns=ts, index=values).T
    return ls

#import xyz file
uc = pd.read_csv('VOTPP_opt.xyz', skiprows=2, header=None, delimiter='      ', engine='python') # enter specific file directory
#seperate columns into numpy arrays
N = np.array(uc[0])
x = np.array(uc[1])
y = np.array(uc[2])
z = np.array(uc[3])
#set up unit cell
sic = pc.BathCell(13.2613, 13.2613, 9.6851, 90, 90, 90, 'deg') #used optimized structure note there is also an x-ray structure with slightly different values
# z axis in cell coordinates
sic.zdir = [0, 0, 1]

#setting concentration
sic.isotopes['V']['51V'] = 0.3 #set to 30%

#populate cell
for i in range(len(N)):
    sic.add_atoms((N[76], [x[76], y[76], z[76]]), type='angstrom')

#assign position of qubit 
pos1 = x[76], y[76], z[76]
qpos = sic.to_cell(pos1)

#generte supercell
cell = 60
atoms = sic.gen_supercell(cell, remove = [('V', qpos)], seed =8000)
#set spin |gyro | quadrupole 
spin_types = ['51V',  1/2, -17608.59705]   #electronic bath
atoms.add_type(*spin_types)

#set up the center 
cen = pc.CenterArray(size=1, position=qpos, spin=1/2,
                     D=0, gyro=-17608.59705, alpha=[0,1], beta=[1,0])

#parameters
nb=100 # no. of bath states
ts = np.linspace(0, 0.0003, 201) 

#dictionary of parameters
parameters = dict(
    order=3, # CCE order
    r_bath=40,  # Size of the bath in A
    r_dipole=40,  # Cutoff of pairwise clusters in A
    pulses = 1, # N pulses in CPMG sequence
    magnetic_field=[0,0,10000] #set to 1T
) 

#converge for order 
orders = runner('order', [1,2,3])
orders.plot(title='convergence test for orders; cell=60; nbstates=100; [?:40:40]',
           xlabel='order', ylabel='Coherence')
#plt.show() #uncomment to see individually

#converge for r_bath
rbs = runner('r_bath', [20,40,60])
orders.plot(title='convergence test for r_bath; cell=60; nbstates=100; [3:?:40]',
           xlabel='order', ylabel='Coherence')
#plt.show() #uncomment to see individually

#converge for r_dipole
rds = runner('r_dipole', [20,40,60])
orders.plot(title='convergence test for r_dipole; cell=600; nbstates=100; [3:40:?]',
           xlabel='order', ylabel='Coherence')
#plt.show() #uncomment to see individually

#plot together 
fig, axes = plt.subplots(1, 3, figsize=(12, 3))
orders.plot(ax=axes[0], title='order')
rbs.plot(ax=axes[1], title='r_bath')
rds.plot(ax=axes[2], title='r_dipole')
for ax in axes:    
    ax.set(xlabel='Time (ms)', ylabel='Coherence')
plt.suptitle('Convergence tests; cell = 60copies')
fig.tight_layout()


#converge for cell size 
ls_cell = [] #empty list to collect simulations 
cell_size = [60,100,200] #enter cell sizes to be simulated 
for i in cell_size:
    #generate cell
    atoms_cell = sic.gen_supercell(i, remove = [('V', qpos)], seed = 8000)
    #set spin |gyro | quadrupole 
    spin_types = ['51V',  1/2, -17608.59705]   #electronic bath
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
plt.show()

#converge nbstates
calc = pc.Simulator(spin=cen, bath=atoms, **parameters)
ls_nb = [] #empty list to collect sims
nbstates = [50,100,250] #enter nbstates to be simulated
for i in nbstates:
    l = calc.compute(ts, method='cce',
                        nbstates=i,
                        quantity='coherence')
    ls_nb.append(l.real)
        
ls_nb_frame = pd.DataFrame(ls_nb, columns=ts, index=nbstates).T
ls_nb_frame.plot(title='Convergence tests for nbstates',
          xlabel='Time [ms]', ylabel='Coherence') 
plt.show()