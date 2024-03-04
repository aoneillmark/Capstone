import numpy as np
import matplotlib.pyplot as plt
import pycce as pc
import pandas as pd
from mpi4py import MPI

class BathSetup:
    def __init__(self, filepath, concentration, cell_size, seed, bath_type):
        '''
        This function initializes a new instance of the BathSetup class 
        with parameters for filepath, concentration, cell size, seed, and bath type.
        '''
        self.filepath = filepath
        self.concentration = concentration
        self.cell_size = cell_size
        self.seed = seed
        self.sic = None
        self.atoms = None
        self.qpos = None
        self.bath_type = bath_type

    def create_bath(self):
        '''
        This is a parent function, collecting the next functions together, and running it as a single instance.
        
        In total, this function reads and processes the structure file, 
        sets up the unit cell, and populates the cell based on the specified bath type, 
        returning a list of atoms for the bath.
        '''
        # Read and process the structure file
        uc = self.read_structure_file()

        # Setup the unit cell
        self.setup_unit_cell()

        # Extract columns and set up the cell structure
        self.setup_cell_structure(uc)

        # Populate the cell based on the bath type
        N, x, y, z = self.extract_columns(uc)
        self.populate_cell(N, x, y, z, self.bath_type)

        return self.atoms

    def read_structure_file(self):
        '''
        This function reads and processes the structure file from the provided file path, 
        returning the unit cell information as a pandas DataFrame.
        '''
        # Read and process the file
        uc = pd.read_csv(self.filepath, skiprows=2, header=None, delimiter='      ', engine='python')
        return uc

    def setup_unit_cell(self):
        '''
        This function sets up the unit cell dimensions and angles based on predefined values.
        '''
        # Setup the unit cell
        self.sic = pc.BathCell(9.6851, 13.2613, 13.2613, 90, 90, 90, 'deg') #used optimized structure note there is also an x-ray structure with slightly different values
        # Additional setup based on self.concentration and other attributes

    def extract_columns(self, uc):
        '''
        This function extracts and returns the atomic numbers 
        and x, y, z positions from the unit cell DataFrame as numpy arrays.
        '''
        # Extract columns from the pandas DataFrame and return them
        N = np.array(uc[0])
        x = np.array(uc[1])
        y = np.array(uc[2])
        z = np.array(uc[3])
        return N, x, y, z
    
    def setup_cell_structure(self, uc):
        '''
        This function populates the cell structure 
        by separating columns from the DataFrame into numpy arrays.
        '''
        # Separate columns into numpy arrays and populate the cell
        # This could be a separate method
        self.N, self.x, self.y, self.z = self.extract_columns(uc)
        self.sic.zdir = [0, 0, 1]
        self.x = np.array(self.x)
        self.cell = self.cell_size
    
    def populate_cell(self, N, x, y, z, bath_type):
        ''' 
        Depending on the bath type, this function populates the unit cell with atoms, 
        setting isotope concentrations and spin types appropriately.
        '''
        
        if bath_type=='electronic':
            self.sic.isotopes['V']['51V'] = self.concentration
            #populate unit cell with V
            self.sic.add_atoms((N[76], [x[76], y[76], z[76]]), type='angstrom')
            # Set position
            self.pos = x[76], y[76], z[76]
            self.qpos = self.sic.to_cell(self.pos)
            # Generate supercell
            self.atoms = self.sic.gen_supercell(self.cell, seed=self.seed, remove=[('V', self.qpos)]) 
            # Set spin type 
            self.spin_types = [
                '51V',  1/2, -17608.59705
                ]   #electronic bath
            self.atoms.add_type(*self.spin_types)

        elif bath_type=='hydrogen' or bath_type=='nitrogen' or bath_type=='carbon':
            # Populate the cell with atoms
            for i in range(len(N)):
                self.sic.add_atoms((N[i], [x[i], y[i], z[i]]), type='angstrom')
            # Set position
            self.pos = x[76], y[76], z[76]
            self.qpos = self.sic.to_cell(self.pos)


            # Set spin baths conditionally
            if bath_type=='hydrogen':
                # Set isotope concentration
                self.sic.isotopes['H']['1H'] = 0.999885
                self.sic.isotopes['H']['2H'] = 0.000115
                # Set spin types
                self.spin_types = [
                            '1H', 1 / 2, 26.7522
                            ]  
                
                
            elif bath_type=='nitrogen':
                # Set isotope concentration
                self.sic.isotopes['N']['14N'] = 0.9963200000000001
                self.sic.isotopes['N']['15N'] = 0.00368
                # Set spin types
                self.spin_types = [
                            'N', 1, 1.9331, 20.44 
                            ]
                
            elif bath_type=='carbon':
                # Set isotope concentration
                self.sic.isotopes['C']['13C'] = 0.010700000000000001
                # Set spin types
                self.spin_types = [
                            '13C',  1 / 2,  6.72828,
                            ]

            # Generate supercell
            self.atoms = self.sic.gen_supercell(self.cell, seed=self.seed, remove=[('V', self.qpos)])
            self.atoms.add_type(*self.spin_types)

    def print_bath(self):
        '''
        This function prints out the counts of different elements present in the bath.
        Just a sanity check, really! :)
        '''
        from collections import Counter
        element_counts = Counter(self.atoms.N)

        for element, count in element_counts.items():
            print(f"{element}: {count}")
        return
        

############################################################################################################

class CenterSetup:
    def __init__(self, atens_file_path, gtens_file_path, spin_type, qpos, alpha=None, beta=None):
        '''
        This function initializes a new instance of the CenterSetup class 
        with parameters for A-tensor and G-tensor file paths, 
        spin type, qubit position, alpha and beta.
        '''
        self.atens_file_path = atens_file_path
        self.gtens_file_path = gtens_file_path
        self.spin_type = spin_type
        self.qpos = qpos
        self.alpha = alpha
        self.beta = beta
        self.interaction_matrix = None
        self.center_parameters = None

    # Overall function to set up the center
    def create_center(self):
        '''
        This is a parent function, which collects the next functions together,
        and runs it as a single instance.

        Overall, this function sets up the center's parameters, 
        and creates a center array based on the defined spin type and other attributes.'''
        # Set up the parameters
        self.setup_parameters()
        # Create the center array
        center = self.create_center_array()
        return center

    def setup_parameters(self):
        '''
        This function sets up the parameters for the central spin based on its spin type.
        If the spin type is 'both', it also gets the interaction tensor.
        '''
        if self.spin_type == 'electron':
            self.center_parameters = {
                'size': 1,
                'position': [self.qpos],
                'spin': [1/2],
                # 'gyro': [-17608.59705],
                'gyro': [self.get_electron_gyro()],
                'D': [0],
                'alpha': self.alpha,
                'beta':  self.beta,
            }
        elif self.spin_type == 'nuclear':
            self.center_parameters = {
                'size': 1,
                'position': [self.qpos],
                'spin': [7/2],
                'gyro': [-7.05],
                'D': [-350],
                'alpha': self.alpha,
                'beta':  self.beta,
            }
        elif self.spin_type == 'both':
            self.center_parameters = {
                'size': 2,
                'position': [self.qpos, self.qpos],
                'spin': [7/2, 1/2],
                'gyro': [self.get_nuclear_gyro(), self.get_electron_gyro()],
                'D': [-350, 0],
                'alpha': self.alpha,
                'beta': self.beta,
            }
            # Get the interaction tensor
            self.interaction_matrix = self.get_interaction_tensor()
        else:
            raise ValueError('Invalid spin type specified.')

    def create_center_array(self):
        '''
        This function creates a center array based on the defined center parameters.
        '''
        # set up the center
        cen = pc.CenterArray(
            size    = self.center_parameters['size'],
            position= self.center_parameters['position'], 
            spin    = self.center_parameters['spin'], 
            gyro    = self.center_parameters['gyro'], 
            D       = self.center_parameters['D'], 
            alpha   = self.center_parameters['alpha'], 
            beta    = self.center_parameters['beta'],
            imap    = self.interaction_matrix
        )
        return cen

    def get_interaction_tensor(self, printing=False):
        '''
        This function reads the interaction tensor from a file, 
        applies a scaling factor, and rotates the tensor, returning the final interaction matrix.
        The reason for rotating the tensor is outlined in the Methods section of Mark's Capstone report.
        '''
        # Read the tensor from the file: 'Revised_VOTPP/VOTPP_opt.Atens
        with open((self.atens_file_path), 'r') as f: 
            lines = f.readlines()
        # Convert the tensor to a list of floats and apply the scaling factor
        values = [float(value) * 1e3 for value in lines[0].strip().split()]
        # Reshape the tensor to a 3x3 matrix
        interaction_matrix = np.array(values).reshape((3, 3))
        
        # Define the rotation matrix to swap X and Z
        rotation_matrix = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0]
        ])
        # Rotate the tensor
        rotated_interaction_matrix = rotation_matrix @ interaction_matrix @ rotation_matrix.T

        return rotated_interaction_matrix

    def get_electron_gyro(self, printing=False):
        '''
        This function reads the gyromagnetic tensor from a file,
        applies a scaling factor, and rotates the tensor, returning the final G-tensor
        for use in defining the gyromagnetic ratio of the central electron spin.
        The reason for rotating the tensor is outlined in the Methods section of Mark's Capstone report.
        '''

        # Read the tensor from the file: 'Revised_VOTPP/VOTPP_opt.gtens'
        with open((self.gtens_file_path), 'r') as f:
            lines = f.readlines()
        # Convert the tensor to a list of floats
        tensor = [float(x) for x in lines[0].split()]
        # Multiply each element by the conversion factor
        tensor_converted_by_factor = [i * -8794.10005384623 for i in tensor]

        # Reshape the tensor to a 3x3 matrix
        tensor_matrix = np.array(tensor_converted_by_factor).reshape(3, 3)
        # Define the rotation matrix to swap X and Z
        rotation_matrix = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0]
        ])
        # Rotate the tensor
        rotated_tensor = rotation_matrix @ tensor_matrix @ rotation_matrix.T

        return rotated_tensor

    def get_nuclear_gyro(self):
        '''
        This function returns the gyromagnetic ratio of the central nuclear spin.
        '''
        const = -7.05
        # Create an identity matrix with the constant as the diagonal
        identity_matrix = np.identity(3) * const
        return identity_matrix

############################################################################################################

class SimulatorSetup:
    def __init__(self, center, atoms, order, r_bath, r_dipole, magnetic_field, pulses):
        '''
        This function initializes a new instance of the SimulatorSetup class 
        with parameters for the center, spin bath, simulation order, 
        bath radius and dipole radius, magnetic field, and pulse sequence.
        '''
        self.center = center
        self.atoms = atoms
        self.order = order
        self.r_bath = r_bath
        self.r_dipole = r_dipole
        self.magnetic_field = magnetic_field
        self.pulses = pulses

    def setup_simulator(self):
        '''
        This function sets up and returns a new Simulator object
        with the provided parameters for conducting simulations.
        '''
        calc = pc.Simulator(spin=self.center, 
                            bath=self.atoms, 
                            order=self.order, 
                            r_bath=self.r_bath, 
                            r_dipole=self.r_dipole, 
                            magnetic_field=self.magnetic_field, 
                            pulses=self.pulses)
        return calc

class RunCalc:
    def __init__(self, calc, timespace, method, nbstates, quantity, parallel, parallel_states):
        '''
        This function initializes a new instance of the RunCalc class 
        with parameters for the simulator object, timespace, method, 
        number of bath sampling states (Monte Carlo), 
        the quantity to be calculated (coherence), 
        and parallel computation settings.
        '''
        self.calc = calc
        self.timespace = timespace
        self.method = method
        self.nbstates = nbstates
        self.quantity = quantity
        self.parallel = parallel
        self.parallel_states = parallel_states

    def run_calculation(self):
        '''
        This function runs the simulation based on the provided parameters, 
        and returns the real part of the coherence for each timestep as a list.
        For example, this function returns the results in the form:
        l_real = [1, 0.95, 0.8, 0.6, 0.3, 0.1, 0, 0, 0, 0, ...]
        '''
        l = self.calc.compute(timespace=self.timespace, 
                        method=self.method, 
                        nbstates=self.nbstates, 
                        quantity=self.quantity, 
                        parallel=self.parallel, 
                        parallel_states=self.parallel_states)
        
        l_real = l.real
        return l_real

############################################################################################################

# # Example usage
# timespace_absolute = np.linspace(0, 2, 201)

# hahn_echo_sequence = pc.Sequence([ 
#     pc.Pulse(axis='x', angle='2*pi/3', delay=np.zeros(timespace_absolute.size),),
#     pc.Pulse(axis='x', angle='2*pi/3', delay=timespace_absolute/2,)])

# default_calc_parameters = {
#     'timespace': timespace_absolute,
#     'method': 'gcce',
#     'nbstates': 1, 
#     'quantity': 'coherence',
#     'parallel': True,
#     'parallel_states': True,
# }

# default_bath_parameters = {
#     'filepath': './Revised_VOTPP/VOTPP_opt2.xyz',
#     'bath_type': 'hydrogen', # choose between 'electronic', 'hydrogen', 'nitrogen', 'carbon'
#     'concentration': 0.02, 
#     'cell_size': 10, 
#     'seed': 8000
# }

# default_simulator_parameters = {
#     'order': 2,
#     'r_bath': 10, 
#     'r_dipole': 10,
#     'magnetic_field': [0, 0, 3000], # Magnetic field in Gauss
#     'pulses': hahn_echo_sequence,
# }

# default_center_parameters = {
#     'atens_file_path': './Revised_VOTPP/VOTPP_opt.Atens',
#     'gtens_file_path': './Revised_VOTPP/VOTPP_opt.gtens',
#     'spin_type': 'both', # choose between 'electron', 'nuclear', 'both'
#     # 'qpos': bath.qpos, we have to wait for the bath to be created :(
#     'alpha': 4, 
#     'beta': 5,
# }


# # bath = BathSetup(filepath='./Revised_VOTPP/VOTPP_opt2.xyz', concentration=0.02, cell_size=500, seed=8000, bath_type='electronic')
# bath = BathSetup(**default_bath_parameters)
# bath.create_bath()
# bath.print_bath()
# # print("Bath qpos: ",bath.qpos)
# # print(bath.atoms)

# # center = CenterSetup(atens_file_path='./Revised_VOTPP/VOTPP_opt.Atens', gtens_file_path='./Revised_VOTPP/VOTPP_opt.gtens', spin_type='both', qpos=bath.qpos, alpha=0, beta=0)
# center = CenterSetup(qpos=bath.qpos, **default_center_parameters)
# cen = center.create_center()
# print(cen)

# simulator = SimulatorSetup(center=cen, atoms=bath.atoms, **default_simulator_parameters)
# # calc = simulator.setup_simulator(order=2, r_bath=100, r_dipole=10, magnetic_field=[0, 0, 3000], pulses=None)
# calc = simulator.setup_simulator()
# print(calc)

# run = RunCalc(calc, **default_calc_parameters)
# result = run.run_calculation()

# plt.figure()
# plt.plot(timespace_absolute, result)
# plt.title('Example coherence plot, obviously not converged! :)')
# plt.xlabel('Time (ms)')
# plt.ylabel('Coherence')
# plt.show()