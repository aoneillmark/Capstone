import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import sys
import pycce as pc
import ase

from mpl_toolkits import mplot3d

seed = 8805
np.random.seed(seed)
np.set_printoptions(suppress=True, precision=5)

################### Generate nuclear spin bath
####### Build BathCell

from ase.build import bulk
# Generate unitcell from ase
diamond = bulk('C', 'diamond', cubic=True)
diamond = pc.read_ase(diamond)
print('Cell\n', diamond.cell)
print('\nAtoms\n', diamond.atoms)


####### Populate BathCell with isotopes
# Add types of isotopes
diamond.add_isotopes(('13C', 0.011))
# Isotopes may also be directly added to BathCell.isotopes. For example, below we are adding an isotope without the nuclear spin:
# diamond.isotopes['C']['14C'] = 0.001


####### Set z-direction of the bath (optional)
# set z direction of the defect
diamond.zdir = [1, 1, 1]
