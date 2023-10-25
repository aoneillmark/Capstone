import numpy as np
import matplotlib.pyplot as plt
import pycce as pc
import pandas as pd
from scipy.optimize import curve_fit
from mpi4py import MPI

uc = pd.read_csv('VOTPP folder/VOTPP_opt.xyz', skiprows=2, header=None, delimiter='      ', engine='python') #enter specific directory

# print(uc)

with open(('VOTPP folder/VOTPP_opt.Atens'), 'r') as f:
            lines = f.readlines()

# Extract numerical values from the string
values = lines[0].strip().split()
values = [float(value)*1e3 for value in values]

# Create a 3x3 matrix from the list of data
interaction_matrix = np.array(values).reshape((3, 3))

print(interaction_matrix)