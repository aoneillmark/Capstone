import pycce as pc

# Assuming common_concentrations is accessible and part of the PyCCE module
common_concentrations = pc.common_concentrations

# Print out the abundances for Carbon (C), Hydrogen (H), and Nitrogen (N)
print("Carbon Isotope Abundances:", common_concentrations.get('C', 'Not available'))
print("Carbon Isotope Abundances:", common_concentrations.get('C', 'Not available'))
print("Hydrogen Isotope Abundances:", common_concentrations.get('H', 'Not available'))
print("Nitrogen Isotope Abundances:", common_concentrations.get('N', 'Not available'))
print("Vanadium Isotope Abundances:", common_concentrations.get('V', 'Not available'))


###################################################################################################

# # Initialize PyCCE and get common concentrations
# common_concentrations = pc.common_concentrations

# # Elements of interest
# elements = ['C', 'H', 'N', 'V']

# # Print out abundances and properties for each isotope
# for element in elements:
#     print(f"{element} Isotope Abundances:", common_concentrations.get(element, 'Not available'))
#     print(f"\nProperties of {element} isotopes:")

#     # Accessing isotopes and their properties from the PyCCE module
#     isotopes = pc.common_isotopes.get(element, {})
#     for isotope in isotopes:
#         spin_type = isotopes[isotope]
#         if hasattr(spin_type, 'spin') and hasattr(spin_type, 'gyro') and hasattr(spin_type, 'q'):
#             print(f"{isotope}: Spin = {spin_type.spin}, Gyro = {spin_type.gyro}, Quadrupole = {spin_type.q}")
#         else:
#             print(f"{isotope}: Properties not available or not defined")
#     print("\n")
