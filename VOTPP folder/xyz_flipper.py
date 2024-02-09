import pandas as pd
import io
import numpy as np

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
