# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 11:40:55 2023

@author: cchog
"""
#5% concentration

#importing functions 
import numpy as np
import matplotlib.pyplot as plt
import pycce as pc 
import pandas as pd
from scipy.optimize import curve_fit 

def fit(x, T2, n):
    '''function to fit coherence curve'''
    y = np.exp(-(((2*x)/T2)**n))
    return y

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
sic.isotopes['V']['51V'] = 0.05 #set to 5%

#populate unit cell with V
sic.add_atoms((N[76], [x[76], y[76], z[76]]), type='angstrom')

#assign position of qubit 
pos1 = x[76], y[76], z[76]
qpos = sic.to_cell(pos1)

#set up the center 
cen = pc.CenterArray(size=1, position=qpos, 
                     spin=7/2, D=-350, gyro=-7.05,  
                     alpha=[0,0,0,0,0,1,0,0], beta=[0,0,0,0,1,0,0,0]) #set to -3/2 to -1/2

#parameters
nb = 0 # set to 0 because of issues using random bath sampling 
ts = np.linspace(0, 1e-1, 201) #expected T2 of 3.8e-2
cell = 400 #cell size 

#dictionary of parameters
parameters = dict(
    order=2, # CCE order
    r_bath=200,  # Size of the bath in A
    r_dipole=180,  # Cutoff of pairwise clusters in A
    pulses = 1, # N pulses in CPMG sequence
    magnetic_field=[0,0,10000] #set to 1T
) 

#calculating T2 using 50 random seeds
n = 1 #counter to keep track
no_sims = 50 #set number of sims to average over
sims = []
for i in range(no_sims):
    
    #generte bath
    rand = np.random.randint(0, 8000) # chooses random integer from 0 to 8000 
    atoms = sic.gen_supercell(cell, remove = [('V', qpos)], seed =rand) #generate supercell 
    #set spin |gyro | quadrupole 
    spin_types = ['51V',  1/2, -17608.59705]   #electronic bath
    atoms.add_type(*spin_types)

    #set up simulation
    calc = pc.Simulator(spin=cen, bath=atoms, **parameters)

    #run simulation
    sim = calc.compute(ts, nbstates = nb,
                        method='cce', quantity='coherence')
    
    #use and interate counter to keep track of progress
    #print(n, ' done') # uncomment to print progress
    n += 1
    
    #plot this run and store it 
    plt.plot(ts, sim.real)
    sims.append(sim.real)

#add labels and title to plot     
plt.xlabel('Time [ms]')
plt.ylabel('Coherence')
plt.title('50 coherence curves for n-e 5% cell=400 [2:200:180]')
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
plt.title('Averaged Coherence Curve n-e 5%')
plt.show()

#fit coherence curve 
p_guess= [0.04, 2] #guessing T2 and the power respectively 
par, cov = curve_fit(fit, ts, avg, p0=p_guess) #find best fitting parameters and covariance matrix
err = np.sqrt(np.diag(cov)) #calculating the errors based on the fit 
print('params =', par) # print params

#plot fitted curve 
label = 'Fit - T2=' + str(round(par[0]*1e3,3)) + ' \u00B1 ' + str(round(err[0]*1e3,3)) + ' us'
plt.plot(ts, avg, label='average for 50')
plt.plot(ts, fit(ts, *par), 'r-', label=label)
plt.xlabel('Time [ms]')
plt.ylabel('Coherence')
plt.title('Averaged Coherence Curve n-e 5%')
plt.legend()
plt.show()