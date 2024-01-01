import numpy as np
import matplotlib.pyplot as plt
import pycce as pc
import pandas as pd
from scipy.optimize import curve_fit
from mpi4py import MPI

class VOTPP_class:
    def __init__(self, concentration, cell_size, num_spins, alpha=None, beta=None, spin_type=None, seed=8000):
        self.concentration = concentration
        self.cell_size = cell_size
        self.seed = seed
        self.atoms, self.qpos1, self.qpos2 = self.setup_bath()

        if num_spins == 1:
            if spin_type == 'electron':
                self.center_parameters = {
                    'size': 1,
                    'position': [self.qpos1],
                    'spin': [1/2],
                    'gyro': [-17608.59705],
                    'D': [0],
                    # 'alpha': [0, 1],
                    # 'beta':  [1, 0],
                    'alpha': [1, 0],
                    'beta':  [0, 1],
                }
            elif spin_type == 'nuclear':
                self.center_parameters = {
                    'size': 1,
                    'position': [self.qpos1],
                    'spin': [7/2],
                    'gyro': [-7.05],
                    'D': [-350],
                    'alpha': 4,
                    'beta':  5,
                }
            elif num_spins==1 and spin_type==None:
                raise ValueError('spin_type must be "electron" or "nuclear" for num_spins=1')
            
            # Set up the center
            self.cen = self.setup_center()


        if num_spins == 2:
            self.center_parameters = {
                'size': 2,
                'position': [self.qpos1, self.qpos2],
                'spin': [7/2, 1/2],
                # 'gyro': [-7.05,-17608.59705], # isotropic
                'gyro': [self.get_nuclear_gyro(), self.get_electron_gyro()], # anisotropic (not isotropic! Mostly the electron isn't isotropic but PyCCE expects similar shapes for both elements, so I've converted nuclear to a 3x3 const tensor)
                'D': [-350, 0],
                'alpha': alpha,
                'beta': beta,
            }   

            # Get the interaction tensor
            self.interaction_matrix = self.get_interaction_tensor()
            # Set up the center
            self.cen = self.setup_center(interaction_matrix=self.interaction_matrix,)

        if num_spins != 1 and num_spins != 2:
            raise ValueError('num_spins must be 1 or 2')


    def setup_bath(self):
        #import xyz file
        uc = pd.read_csv('VOTPP folder/VOTPP_opt.xyz', skiprows=2, header=None, delimiter='      ', engine='python') #enter specific directory
        
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
            x = np.array(x)

            #populate cell
            for i in range(len(N)):
                sic.add_atoms((N[i], [x[i], y[i], z[i]]), type='angstrom')

            #assign position of qubit 
            pos2 = x[76], y[76], z[76] # Position of the nuclear spin
            pos1 = x[76], y[76], z[76] # Position of the electron spin
            qpos1 = sic.to_cell(pos1)
            qpos2 = sic.to_cell(pos2)

            sic.isotopes['C']['13C'] = 0.010700000000000001
            sic.isotopes['H']['1H'] = 0.999885
            sic.isotopes['H']['2H'] = 0.000115
            sic.isotopes['N']['14N'] = 0.9963200000000001
            sic.isotopes['N']['15N'] = 0.00368
            sic.isotopes['V']['50V'] = 0.0025
            sic.isotopes['V']['51V'] = 0



            #generate supercell - nuclear bath 
            cell=self.cell_size
            atoms = sic.gen_supercell(cell, seed=self.seed, remove=[('V', qpos1), ('V', qpos2)]) #left fixed for convergence tests to avoid changes
            #set          spin | gyro | quadrupole 
            spin_types = [('13C',  1 / 2,  6.72828),    
                        ('1H', 1 / 2, 26.7522),
                        ('N', 1, 1.9331, 20.44 ),
                        ('V', 7/2, 7.05, -350), # not added for consistency between tests
                        ]   
            atoms.add_type(*spin_types)
        
        if 0 < self.concentration <= 1:
            x = np.array(x)

            #setting concentration
            # sic.isotopes['V']['50V'] = 1 - self.concentration
            sic.isotopes['V']['51V'] = self.concentration

            #populate unit cell with V
            sic.add_atoms((N[76], [x[76], y[76], z[76]]), type='angstrom')

            #assign position of qubit 
            pos1 = x[76], y[76], z[76] # Position of the nuclear spin
            pos2 = x[76], y[76], z[76] # Position of the electron spin
            qpos1 = sic.to_cell(pos1)
            qpos2 = sic.to_cell(pos2)


            #generate supercell - nuclear bath 
            cell=self.cell_size


            atoms = sic.gen_supercell(cell, seed=self.seed, remove=[('V', qpos1), ('V', qpos2)]) #generate supercell 
            #set          spin | gyro | quadrupole 
            spin_types = [
                '51V',  1/2, -17608.59705
                # ('50V',   7/2,    7.05,       -350), # not added for consistency between tests
                ]   #electronic bath
            atoms.add_type(*spin_types)
        
        if self.concentration == 1111:
            x = np.array(x)

            #populate cell
            for i in range(len(N)):
                sic.add_atoms((N[i], [x[i], y[i], z[i]]), type='angstrom')

            #assign position of qubit 
            pos2 = x[76], y[76], z[76] # Position of the nuclear spin
            pos1 = x[76], y[76], z[76] # Position of the electron spin
            qpos1 = sic.to_cell(pos1)
            qpos2 = sic.to_cell(pos2)

            sic.isotopes['H']['1H'] = 0.999885
            sic.isotopes['H']['2H'] = 0.000115

            #generate supercell - nuclear bath 
            cell=self.cell_size
            atoms = sic.gen_supercell(cell, seed=self.seed, remove=[('V', qpos1), ('V', qpos2)]) #left fixed for convergence tests to avoid changes
            #set          spin | gyro | quadrupole 
            spin_types = [
                        '1H', 1 / 2, 26.7522
                        ]   
            atoms.add_type(*spin_types)
        
        if self.concentration == 1313:
            x = np.array(x)

            #populate cell
            for i in range(len(N)):
                sic.add_atoms((N[i], [x[i], y[i], z[i]]), type='angstrom')

            #assign position of qubit 
            pos2 = x[76], y[76], z[76] # Position of the nuclear spin
            pos1 = x[76], y[76], z[76] # Position of the electron spin
            qpos1 = sic.to_cell(pos1)
            qpos2 = sic.to_cell(pos2)

            sic.isotopes['C']['13C'] = 0.010700000000000001
            # sic.isotopes['C']['12C'] = 1- 0.010700000000000001 # We don't consider 12C because it has an effective spin of 0

            #generate supercell - nuclear bath 
            cell=self.cell_size
            atoms = sic.gen_supercell(cell, seed=self.seed, remove=[('V', qpos1), ('V', qpos2)]) #left fixed for convergence tests to avoid changes
            #set          spin | gyro | quadrupole 
            spin_types = ['13C',  1 / 2,  6.72828,    
                        ]   
            atoms.add_type(*spin_types)
    
        if self.concentration == 1414:
            x = np.array(x)

            #populate cell
            for i in range(len(N)):
                sic.add_atoms((N[i], [x[i], y[i], z[i]]), type='angstrom')

            #assign position of qubit 
            pos2 = x[76], y[76], z[76] # Position of the nuclear spin
            pos1 = x[76], y[76], z[76] # Position of the electron spin
            qpos1 = sic.to_cell(pos1)
            qpos2 = sic.to_cell(pos2)

            sic.isotopes['N']['14N'] = 0.9963200000000001
            sic.isotopes['N']['15N'] = 0.00368

            #generate supercell - nuclear bath 
            cell=self.cell_size
            atoms = sic.gen_supercell(cell, seed=self.seed, remove=[('V', qpos1), ('V', qpos2)]) #left fixed for convergence tests to avoid changes
            #set          spin | gyro | quadrupole 
            # spin_types = [('13C',  1 / 2,  6.72828),    
            #             ('1H', 1 / 2, 26.7522),
            #             ('N', 1, 1.9331, 20.44 ),
            #             ('V', 7/2, 7.05, -350), # not added for consistency between tests
            #             ]   
            # atoms.add_type(*spin_types)
    
        return atoms, qpos1, qpos2
        

    def setup_center(self, interaction_matrix=None):
        # set up the center
        cen = pc.CenterArray(
            size=self.center_parameters['size'],
            position=self.center_parameters['position'], 
            spin=self.center_parameters['spin'], 
            gyro=self.center_parameters['gyro'], 
            D=self.center_parameters['D'], 
            alpha=self.center_parameters['alpha'], 
            beta=self.center_parameters['beta'],
            imap = interaction_matrix
        )
        return cen

    def get_interaction_tensor(self, printing=False):
        with open(('VOTPP folder/VOTPP_opt.Atens'), 'r') as f:
            lines = f.readlines()

        # Extract numerical values from the string
        values = lines[0].strip().split()
        values = [float(value)*1e3 for value in values]

        # Create a 3x3 matrix from the list of data
        interaction_matrix = np.array(values).reshape((3, 3))


        return interaction_matrix # self.cen
    
    def get_electron_gyro(self):
        with open(('VOTPP folder/VOTPP_opt.gtens'), 'r') as f:
            lines = f.readlines()
        
        tensor = [float(x) for x in lines[0].split()]
        tensor_converted_by_factor = [i * 8794.10005384623 for i in tensor]

        # Reshape the tensor to a 3x3 matrix
        return np.array(tensor_converted_by_factor).reshape(3, 3)
        
    def get_nuclear_gyro(self):
        const = -7.05
        
        # Create a 3x3 matrix filled with the constant
        return np.full((3, 3), const)

    def setup_simulator(self, order, r_bath, r_dipole, magnetic_field):
        calc = pc.Simulator(spin=self.cen, bath=self.atoms, order=order, r_bath=r_bath, r_dipole=r_dipole, magnetic_field=magnetic_field)
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
    
    def get_active_nuclei_positions(self, atoms, r_bath):
        central_spin_position = self.qpos1
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

    def get_number_of_active_nuclei(self, atoms, r_bath):
        central_spin_position = self.qpos1
        active_positions = self.get_active_nuclei_positions(atoms, r_bath)
        return len(active_positions)

    def print_bath(self, calc):
        # with np.printoptions(threshold=np.inf):
            # print(np.array(calc.bath.N))
            # print(np.array(calc.bath))
        from collections import Counter

        element_counts = Counter(calc.bath.N)

        # Now, element_counts is a dictionary with elements as keys and their counts as values
        # To print them, you can iterate over this dictionary
        for element, count in element_counts.items():
            print(f"{element}: {count}")

        return

    def visualize_cluster(self, calc):
        central_spin_pos = self.qpos1

        # add 3D axis
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(projection='3d')

        # We want to visualize the smaller bath
        data = calc.bath

        # First plot the positions of the bath
        colors = np.abs(data.A[:,1,2]/data.A[:,1,2].max())
        ax.scatter3D(data.x, data.y, data.z, c=colors, cmap='viridis');

        ax.scatter3D(central_spin_pos[0], central_spin_pos[1], central_spin_pos[2], c='red', s=100, label='Central Spin');
        # Plot all pairs of nuclear spins, which are contained
        # in the calc.clusters dictionary under they key 2
        for c in calc.clusters[1]:
            ax.plot3D(data.x[c], data.y[c], data.z[c], color='grey')
        # # Plot all triplets of nuclear spins, which are contained
        # # in the calc.clusters dictionary under they key 3
        # for c in mock.clusters[3]:
        #     ax.plot3D(data.x[c], data.y[c], data.z[c], color='red', ls='--', lw=0.5)

        ax.set(xlabel='x (A)', ylabel='y (A)', zlabel='z (A)');
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