import pycce as pc

# Assuming common_concentrations is accessible and part of the PyCCE module
common_concentrations = pc.common_concentrations

# Print out the abundances for Carbon (C), Hydrogen (H), and Nitrogen (N)
print("Carbon Isotope Abundances:", common_concentrations.get('C', 'Not available'))
print("Hydrogen Isotope Abundances:", common_concentrations.get('H', 'Not available'))
print("Nitrogen Isotope Abundances:", common_concentrations.get('N', 'Not available'))
print("Vanadium Isotope Abundances:", common_concentrations.get('V', 'Not available'))