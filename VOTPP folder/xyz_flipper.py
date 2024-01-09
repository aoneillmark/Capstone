import pandas as pd
import io
import numpy as np

# # Provided XYZ data as a string
# xyz_data = """
#  C        8.4470799999999997        9.1870999999999974        4.8570099999999998     
#  C        8.3630499999999994        10.620729999999998        4.8550599999999999     
#  ...
#  V        6.6898000000000000        6.6898000000000000        5.4113499999999997     
#  O        6.6897999999999973        6.6898000000000000        7.0411100000000006     
# """

# # Simulating reading from a file
# xyz_file = io.StringIO(xyz_data)

# Reading the XYZ data into a DataFrame
# uc = pd.read_csv('VOTPP folder/VOTPP_opt.xyz', skiprows=2, header=None, delimiter='      ', engine='python') #enter specific directory
uc = pd.read_csv('VOTPP folder/VOTPP_opt.xyz',skiprows=2, header=None, delim_whitespace=True, names=['Element', 'X', 'Y', 'Z'])

# Displaying the DataFrame to verify
print(uc.head())

# Applying the rotation matrix to the coordinates
rotation_matrix = np.array([
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0]
])

# Function to apply rotation to a set of coordinates
def apply_rotation(x, y, z, matrix):
    original_vector = np.array([x, y, z])
    rotated_vector = np.dot(matrix, original_vector)
    return rotated_vector

# Applying the rotation to each row in the DataFrame
uc[['X', 'Y', 'Z']] = uc.apply(lambda row: apply_rotation(row['X'], row['Y'], row['Z'], rotation_matrix), axis=1, result_type='expand')

print(uc.head())

print("Number of rows in uc:", len(uc))
print("First few rows in uc:")
print(uc.head())

rotated_xyz_file = "./VOTPP folder/VOTPP_opt2.xyz"
with open(rotated_xyz_file, 'w') as file:
    file.write(f"{len(uc)}\n\n")  # Number of atoms and a blank line
    for index, row in uc.iterrows():
        # print(f"Writing row {index}: {row['Element']}        {row['X']}        {row['Y']}        {row['Z']}")
        file.write(f"{row['Element']}        {row['X']}        {row['Y']}        {row['Z']}\n")

# Check the content of the file after writing
with open(rotated_xyz_file, 'r') as file:
    print("Content of rotated_xyz_file:")
    print(file.read())
