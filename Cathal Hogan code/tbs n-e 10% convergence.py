# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 11:42:12 2023

@author: cchog
"""
#10% concentration
#importing functions 
import numpy as np
import matplotlib.pyplot as plt
import pycce as pc 
import pandas as pd
from scipy.optimize import curve_fit 

#define function to converge parameters 
def runner(variable, values):
    invalue = parameters[variable]
    calc = pc.Simulator(spin=cen, bath=atoms, **parameters)
    ls = []
    
    for v in values:
        setattr(calc, variable, v)
        l = calc.compute(ts, method='cce',
                         nbstates=nb,
                         quantity='coherence')
        
        
        #print('a run done') # uncomment to print progress
        ls.append(l.real)

    parameters[variable] = invalue
    ls = pd.DataFrame(ls, columns=ts, index=values).T
    return ls

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
sic.isotopes['V']['51V'] = 0.1 #set to 10%

#populate cell
for i in range(len(N)):
    sic.add_atoms((N[76], [x[76], y[76], z[76]]), type='angstrom')

#assign position of qubit 
pos1 = x[76], y[76], z[76]
qpos = sic.to_cell(pos1)

#generte supercell
cell = 400
atoms = sic.gen_supercell(cell, remove = [('V', qpos)], seed =8000)
#set spin |gyro | quadrupole 
spin_types = ['51V',  1/2, -17608.59705]   #electronic bath
atoms.add_type(*spin_types)

#set up the center 
cen = pc.CenterArray(size=1, position=qpos, 
                     spin=7/2, D=-350, gyro=-7.05,  
                     alpha=[0,0,0,0,0,1,0,0], beta=[0,0,0,0,1,0,0,0]) #set to -3/2 to -1/2

#parameters
nb=0 # no. of bath states
ts = np.linspace(0, 1e-1, 201) #T2 =1us for 2% concentration

#dictionary of parameters
parameters = dict(
    order=2, # CCE order
    r_bath=200,  # Size of the bath in A
    r_dipole=180,  # Cutoff of pairwise clusters in A
    pulses = 1, # N pulses in CPMG sequence
    magnetic_field=[0,0,10000] #set to 1T
) 

#converge for order 
orders = runner('order', [1,2]) #problems with running order 3
orders.plot(title='convergence test for orders; cell=400; nbstates=0; [?:200:180]',
           xlabel='order', ylabel='Coherence')
#plt.show() # uncomment to show individually

#converge for r_bath
rbs = runner('r_bath', [100,200,300])
orders.plot(title='convergence test for r_bath; cell=400; nbstates=0; [2:?:180]',
           xlabel='order', ylabel='Coherence')
#plt.show() # uncomment to show individually

#converge for r_dipole
rds = runner('r_dipole', [100,180,200])
orders.plot(title='convergence test for r_dipole; cell=400; nbstates=0; [2:200:?]',
           xlabel='order', ylabel='Coherence')
#plt.show() # uncomment to show individually

#plot together 
fig, axes = plt.subplots(1, 3, figsize=(12, 3))
orders.plot(ax=axes[0], title='order')
rbs.plot(ax=axes[1], title='r_bath')
rds.plot(ax=axes[2], title='r_dipole')
for ax in axes:    
    ax.set(xlabel='Time (ms)', ylabel='Coherence')
plt.suptitle('Convergence tests; cell = 400copies')
fig.tight_layout()

#converge for cell size 
ls_cell = [] #empty list to collect simulations 
cell_size = [300,400,500] #enter cell sizes to be simulated 
for i in cell_size():
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
ls_cell.plot(title='Convergence tests for cell size',
          xlabel='Time [ms]', ylabel='Coherence')
plt.show()

#There was issues using the random bath sampling (nbstates) for this simulation so I left it out
#sim didn't follow predicted shape or pattern
#change nb=0 to a non-zero number if you want to use it and uncomment the next section to converge
'''
#converge nbstates
calc = pc.Simulator(spin=cen, bath=atoms, **parameters)
ls_nb = [] #empty list to collect sims
nbstates = [100,250,500] #enter nbstates to be simulated
for i in nbstates():
    l = calc.compute(ts, method='cce',
                        nbstates=i,
                        quantity='coherence')
    ls_nb.append(l.real)
        
ls_nb = pd.DataFrame(ls, columns=ts, index=values).T
ls_cell.plot(title='Convergence tests for nbstates',
          xlabel='Time [ms]', ylabel='Coherence') 
'''