import numpy as np
import matplotlib.pyplot as plt
import pycce as pc
import pandas as pd
from scipy.optimize import curve_fit
from mpi4py import MPI

class VOTPP_class:
    def __init__(self, concentration, cell_size, displacement, seed=8000):
        self.concentration = concentration
        self.cell_size = cell_size
        self.displacement = displacement
        self.seed = seed
        self.atoms, self.qpos1, self.qpos2 = self.setup_bath()
        self.center_parameters = {
            'position': [self.qpos1, self.qpos2],
            'spin': [7/2, 1/2],
            'gyro': [-7.05,-17608.59705],
            'D': [ -350, 0],
            'alpha': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            'beta': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        }
        self.interaction_matrix = self.create_interaction_tensor()
        self.cen = self.setup_center(self.interaction_matrix)
        self.cen = self.create_interaction_tensor()

    def setup_bath(self):
        #import xyz file
        uc = pd.read_csv('VOTPP_opt.xyz', skiprows=2, header=None, delimiter='      ', engine='python') #enter specific directory
        #seperate columns into numpy arrays
        N = np.array(uc[0])
        x = np.array(uc[1])
        y = np.array(uc[2])
        z = np.array(uc[3])
        #set up unit cell
        sic = pc.BathCell(13.2613, 13.2613, 9.6851, 90, 90, 90, 'deg') #used optimized structure note there is also an x-ray structure with slightly different values
        # z axis in cell coordinates
        sic.zdir = [0, 0, 1]

        if self.concentration == 0:
            #populate cell
            for i in range(len(N)):
                sic.add_atoms((N[i], [x[i], y[i], z[i]]), type='angstrom')

            #assign position of qubit 
            pos2 = x[76], y[76], z[76] # Position of the nuclear spin
            pos1 = x[76], y[76], z[76] # Position of the electron spin (displacement is in angstroms)
            qpos1 = sic.to_cell(pos1)
            qpos2 = sic.to_cell(pos2)
            # print(qpos1)
            # print("")
            # print(qpos2)

            #generate supercell - nuclear bath 
            cell=self.cell_size
            atoms = sic.gen_supercell(cell, seed=self.seed,) #left fixed for convergence tests to avoid changes
            #set          spin | gyro | quadrupole 
            spin_types = [('C',  1 / 2,  6.72828),    
                        ('H', 1 / 2, 26.7522),
                        ('N', 1, 1.9331, 20.44 )
                        #('V', 7/2, 7.05, -350)  # not added for consistency between tests
                        ]   
            atoms.add_type(*spin_types)
        
        if self.concentration > 0:
            x = np.array(x)

            #setting concentration
            sic.isotopes['V']['51V'] = self.concentration

            #populate unit cell with V
            sic.add_atoms((N[76], [x[76], y[76], z[76]]), type='angstrom')

            #assign position of qubit 
            pos1 = x[76], y[76], z[76] # Position of the nuclear spin
            pos2 = x[76], y[76], z[76] # Position of the electron spin (displacement is in angstroms)
            qpos1 = sic.to_cell(pos1)
            qpos2 = sic.to_cell(pos2)
            # print(qpos1)
            # print("")
            # print(qpos2)

            #generate supercell - nuclear bath 
            cell=self.cell_size


            atoms = sic.gen_supercell(cell, seed=self.seed, remove=[('V', qpos1), ('V', qpos2)]) #generate supercell 
            #set          spin | gyro | quadrupole 
            spin_types = ['51V',  1/2, -17608.59705]   #electronic bath
            atoms.add_type(*spin_types)
    
        return atoms, qpos1, qpos2

    def setup_center(self, interaction_matrix):
        # set up the center
        cen = pc.CenterArray(
            size=2, 
            position=self.center_parameters['position'], 
            spin=self.center_parameters['spin'], 
            gyro=self.center_parameters['gyro'], 
            D=self.center_parameters['D'], 
            alpha=self.center_parameters['alpha'], 
            beta=self.center_parameters['beta'],
            imap = interaction_matrix
        )
        return cen

    def create_interaction_tensor(self, printing=False):
        # self.cen.point_dipole()

        # el_alpha = np.array([0,1])
        # el_beta = np.array([1,0])
        # n_alpha = np.array([0,0,0,0,0,1,0,0])
        # n_beta = np.array([0,0,0,0,1,0,0,0])

        # # cen[0].alpha = el_alpha
        # # cen[0].beta = el_beta
        # # cen[1].alpha = n_alpha
        # # cen[1].beta = n_beta
        # self.cen[0].alpha = n_alpha
        # self.cen[0].beta = n_beta
        # self.cen[1].alpha = el_alpha
        # self.cen[1].beta = el_beta

        # # Generate product state
        # # state = pc.normalize(np.kron(el_alpha + el_beta, n_alpha + n_beta)) # I'm kinda guessing the order here # This is performing a tensor product! (Kronecker product)
        # state = pc.normalize(np.kron(el_beta + el_alpha, n_beta + n_alpha)) # I'm kinda guessing the order here # This is performing a tensor product! (Kronecker product)
        # # state2 = pc.normalize(np.kron(n_beta, el_beta) + np.kron(n_alpha, el_alpha))
        # self.cen.state = state
        # if printing==True:
        #     print("Initial amplitudes in Sz x Sz basis:            ", np.abs(self.cen.state)) # Initial state
        #     # print("Initial amplitudes in Sz x Sz basis (Chatgpt v2)", np.abs(state2))
        #     print("Interaction tensor:")
        #     print(self.cen.imap[0, 1]) # in kHz

        with open('VOTPP_opt.Atens', 'r') as f:
            lines = f.readlines()

        # Extract numerical values from the string
        values = lines[0].strip().split()
        values = [float(value)*1e3 for value in values]

        # Create a 3x3 matrix from the list of data
        interaction_matrix = np.array(values).reshape((3, 3))

        return interaction_matrix # self.cen

    def setup_simulator(self, order, r_bath, r_dipole, pulses, magnetic_field):
        calc = pc.Simulator(spin=self.cen, bath=self.atoms, order=order, r_bath=r_bath, r_dipole=r_dipole, pulses=pulses, magnetic_field=magnetic_field)
        return calc

    def run_calculation(self, calc, timespace, method, pulses, nb_states, quantity, parallel, parallel_states):
        l = calc.compute(timespace=timespace, method=method, pulses=pulses, nbstates=nb_states, quantity=quantity, parallel=parallel, parallel_states=parallel_states)
        l_real = l.real
        l_abs = np.abs(l)
        return l_real

    def plot_results(self, timespace, result):
        plt.figure()
        plt.plot(timespace, result.real)
        plt.xlabel('Time (ms)')
        plt.ylabel('Coherence')
        plt.show()

# Usage:
# simulator = VOTPP_class(concentration=0.02, cell_size=500, displacement=3.75)
# sim = simulator.setup_simulator(order=2, r_bath=200, r_dipole=180, pulses=1, magnetic_field=[0,0,10000])
# result = sim.compute(timespace=np.linspace(0, 5e-3, 201), method='cce', pulses=1, nbstates=0, quantity='coherence', parallel=True, parallel_states=True)
# simulator.plot_results(np.linspace(0, 5e-3, 201), result)

# # Set up a runner to cycle through different orders to test for convergence for different concentrations
# concentration_list = [0, 0.02, 0.05, 0.1, 0.2, 0.5]
# order_list = [1, 2, 3]
# parameters = {
#     'order': 2,
#     'r_bath': 200,
#     'r_dipole': 180,
#     'pulses': 1,
#     'magnetic_field': [0,0,10000]
# }

# def runner(variables, values):
#     invalue = parameters[variable]
#     calc = pc.Simulator(spin=cen, bath=atoms, **parameters)
#     ls = []
    
#     for v in values:
#         setattr(calc, variable, v)
#         l = calc.compute(ts, method='cce',
#                          nbstates=nb,
#                          quantity='coherence')
        
        
#         #print('a run done') # uncomment to keep track
#         ls.append(l.real)

#     parameters[variable] = invalue
#     ls = pd.DataFrame(ls, columns=ts, index=values).T
#     return ls