import pandas as pd
import numpy as np

def filter_atoms_by_type_v2(file_path, atom_type):
    """
    Reads a .xyz file, filters out atoms of the specified type, and writes them to a new .xyz file using pandas.

    :param file_path: Path to the original .xyz file
    :param atom_type: Type of atom to filter (e.g., 'C' for carbon)
    :return: Path to the new .xyz file containing only the specified atoms
    """
    # Read the file skipping the first two lines
    uc = pd.read_csv(file_path, skiprows=2, header=None, delim_whitespace=True, engine='python')

    # Filter the DataFrame for the specified atom type
    filtered_df = uc[uc[0] == atom_type]

    # Prepare the new file name and path
    new_file_name = f"{atom_type}_atoms.xyz"
    new_file_path = f"VOTPP folder/{new_file_name}"

    # Write the filtered atoms to the new file
    with open(new_file_path, 'w') as new_file:
        # Writing the number of atoms and a blank line as per .xyz format
        new_file.write(f"{len(filtered_df)}\n\n")
        for index, row in filtered_df.iterrows():
            new_file.write(f"{row[0]} {row[1]} {row[2]} {row[3]}\n")

    return new_file_path

# Usage
atom_type = "H"  # Specify the atom type for filtering
file_path = "VOTPP folder/VOTPP_opt.xyz"  # Specify the path to the original .xyz file
new_file_path = filter_atoms_by_type_v2(file_path, atom_type)

atom_type = ""  # Specify the atom type for filtering
file_path = "VOTPP folder/VOTPP_opt.xyz"  # Specify the path to the original .xyz file
new_file_path = filter_atoms_by_type_v2(file_path, atom_type)

atom_type = "H"  # Specify the atom type for filtering
file_path = "VOTPP folder/VOTPP_opt.xyz"  # Specify the path to the original .xyz file
new_file_path = filter_atoms_by_type_v2(file_path, atom_type)

atom_type = "H"  # Specify the atom type for filtering
file_path = "VOTPP folder/VOTPP_opt.xyz"  # Specify the path to the original .xyz file
new_file_path = filter_atoms_by_type_v2(file_path, atom_type)
