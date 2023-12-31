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
            'D': [-350, 0],
            'alpha': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # nuclear 1/2 to 3/2 for m_s = -1/2
            'beta': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        }
        self.interaction_matrix = self.create_interaction_tensor()
        self.cen = self.setup_center(interaction_matrix=self.interaction_matrix)

        # print("atoms.N:")
        # print(*self.atoms.N)

        # print("atoms.s:")
        # print(*self.atoms.s)
        # Count the number of atoms and spins


        num_atoms = len(self.atoms.N)
        num_spins = len(self.atoms.s)

        print(f"Number of atoms in the bath: {num_atoms}")
        print(f"Number of spins in the bath: {num_spins}")

        # print("Bath size:")
        # print(self.atoms.size)

        # # Display atom-spin pairings compactly
        # atom_spin_pairs = [f"{atom}: {spin}" for atom, spin in zip(self.atoms.N, self.atoms.s)]
        # compact_representation = ", ".join(atom_spin_pairs)
        # print(compact_representation)


    def setup_bath(self):
        def get_supercell_count_for_atom_type(atom_type, unit_cell_count, size, cell):
            # Calculate anumber, bnumber, and cnumber using the logic from gen_supercell
            axb = np.cross(cell[:, 0], cell[:, 1])
            bxc = np.cross(cell[:, 1], cell[:, 2])
            cxa = np.cross(cell[:, 2], cell[:, 0])
            anumber = int(size * np.linalg.norm(bxc) / (bxc @ cell[:, 0]) + 1)
            bnumber = int(size * np.linalg.norm(cxa) / (cxa @ cell[:, 1]) + 1)
            cnumber = int(size * np.linalg.norm(axb) / (axb @ cell[:, 2]) + 1)
            
            # Calculate the number of atomic sites for the atom_type in the supercell
            supercell_count = unit_cell_count * anumber * bnumber * cnumber
            
            return supercell_count


        #import xyz file
        uc = pd.read_csv('VOTPP folder/VOTPP_opt.xyz', skiprows=2, header=None, delimiter='      ', engine='python') #enter specific directory
        # print("xyz file (uc):")
        # print(uc)
        
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
            pos1 = x[76], y[76], z[76] # Position of the electron spin (displacement is in angstroms)
            qpos1 = sic.to_cell(pos1)
            qpos2 = sic.to_cell(pos2)
            # print(qpos1)
            # print("")
            # print(qpos2)

            print("sic.atoms:")
            # print(*sic.atoms, sep="\n")
            counts = {}
            for atom_type, coordinates in sic.atoms.items():
                counts[atom_type] = len(coordinates)
            print(counts)

            total_atoms = sum(len(coordinates) for coordinates in sic.atoms.values())
            print("Total number of atoms in a single cell:", total_atoms)


            # print("sic.isotopes:")
            # print(*sic.isotopes, sep="\n")


            #generate supercell - nuclear bath 
            cell=self.cell_size



            atoms = sic.gen_supercell(cell, seed=self.seed, remove=[('V', qpos1), ('V', qpos2)]) #left fixed for convergence tests to avoid changes

            # print("len of atoms.N in supercell:")
            # print(len(atoms.N))


            #set          spin | gyro | quadrupole 
            spin_types = [('C',  1 / 2,  6.72828),    
                        ('H', 1 / 2, 26.7522),
                        ('N', 1, 1.9331, 20.44 )
                        #('V', 7/2, 7.05, -350)  # not added for consistency between tests
                        ]   
            atoms.add_type(*spin_types)

            supercell_count = get_supercell_count_for_atom_type("13C", 44, self.cell_size, sic.cell)
            print("Number of 13C atom positions:", supercell_count)
            count_13C = sum(1 for atom in atoms.N if atom == '13C')
            print(f"Number of 13C spins:", count_13C)

            supercell_count = get_supercell_count_for_atom_type("1H", 32, self.cell_size, sic.cell)
            print("Number of 1H atom positions:", supercell_count)
            count_1H = sum(1 for atom in atoms.N if atom == '1H')
            print(f"Number of 1H spins:", count_1H)

            supercell_count = get_supercell_count_for_atom_type("14N", 4, self.cell_size, sic.cell)
            print("Number of 14N atom positions:", supercell_count)
            count_14N = sum(1 for atom in atoms.N if atom == '14N')
            print(f"Number of 14N spins:", count_14N)

        
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
        with open(('VOTPP folder/VOTPP_opt.Atens'), 'r') as f:
            lines = f.readlines()

        # Extract numerical values from the string
        values = lines[0].strip().split()
        values = [float(value)*1e3 for value in values]

        # Create a 3x3 matrix from the list of data
        interaction_matrix = np.array(values).reshape((3, 3))

        # print("Interaction tensor:")
        # print(interaction_matrix)

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
    
    def get_active_nuclei_positions(self, calc, r_bath):
        # Extracting positions from the setup
        uc = pd.read_csv('VOTPP folder/VOTPP_opt.xyz', skiprows=2, header=None, delimiter='      ', engine='python')
        positions = uc[[1, 2, 3]].values  # Extracting x, y, z columns as numpy array

        central_spin_position = np.array(self.qpos1)  # Using qpos1 as the central spin position
        distances = np.linalg.norm(positions - central_spin_position, axis=1)
        active_indices = np.where(distances <= r_bath)[0]
        return positions[active_indices]

    def get_number_of_active_nuclei(self, calc, r_bath):
        return len(self.get_active_nuclei_positions(calc, r_bath))

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