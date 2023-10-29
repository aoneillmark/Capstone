import numpy as np
import os 

def get_electron_gyro():
    with open(('VOTPP folder/VOTPP_opt.gtens'), 'r') as f:
        lines = f.readlines()
    
    tensor = [float(x) for x in lines[0].split()]
    tensor_converted_by_factor = [i*8794.10005384623 for i in tensor]

    return tensor_converted_by_factor