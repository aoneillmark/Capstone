import numpy as np
import os 

def get_electron_gyro():
    with open(('VOTPP folder/VOTPP_opt.gtens'), 'r') as f:
        lines = f.readlines()
    
    tensor = [float(x) for x in lines[0].split()]
    tensor_converted_by_factor = [i*8794.10005384623 for i in tensor]

    return tensor_converted_by_factor

print(get_electron_gyro())
# outputs:
# [17445.875513011008, -0.1714849510500015, -0.3834227623476957, 0.17852023109307846, 17445.88078947104, 0.38166394233692635, -0.11520271070538561, 0.382543352342311, 17304.72581009675]

def get_nuclear_gyro():
    const = -7.05
    
    # Make a list with the same dimensions as electron tensor, but with const in every element
    # This needs to be a list
    const_list = [const]*9


    return const_list

print(get_nuclear_gyro())