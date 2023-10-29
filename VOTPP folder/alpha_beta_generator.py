# def transition_vector(m_I1, m_S1, m_I2, m_S2):
#     # Define the basis vectors for nuclear and electron spins
#     nuclear_basis = [7/2, 5/2, 3/2, 1/2, -1/2, -3/2, -5/2, -7/2]
#     electron_basis = [1/2, -1/2]
    
#     # Combine the nuclear and electron spins to create a joint basis
#     combined_basis = [(m_I, m_S) for m_I in nuclear_basis for m_S in electron_basis]
    
#     # Initialize the alpha and beta vectors as zero vectors of length 16
#     alpha = [0] * 16
#     beta = [0] * 16
    
#     # Set the corresponding entries for the initial and final states
#     alpha[combined_basis.index((m_I1, m_S1))] = 1
#     beta[combined_basis.index((m_I2, m_S2))] = 1
    
#     return alpha, beta

import numpy as np

def nuclear_vector(m_I):
    # Define the basis vectors for nuclear spin
    nuclear_basis = [7/2, 5/2, 3/2, 1/2, -1/2, -3/2, -5/2, -7/2]
    
    # Initialize the vector as a zero vector of length 8
    vector = [0] * len(nuclear_basis)
    
    # Set the corresponding entry for the given nuclear spin value
    vector[nuclear_basis.index(m_I)] = 1
    
    return vector

def electron_vector(m_S):
    # Define the basis vectors for electron spin
    electron_basis = [1/2, -1/2]
    
    # Initialize the vector as a zero vector of length 2
    vector = [0] * len(electron_basis)
    
    # Set the corresponding entry for the given electron spin value
    vector[electron_basis.index(m_S)] = 1
    
    return vector

def kronecker(m_I, m_S):
    nuclear_vec = nuclear_vector(m_I)
    electron_vec = electron_vector(m_S)
    return np.kron(nuclear_vec, electron_vec)

# Test
m_I1, m_S1 = 3/2, -1/2
alpha = (kronecker(m_I1, m_S1))

m_I2, m_S2 = 1/2, -1/2
beta = (kronecker(m_I2, m_S2))

print("Alpha:", alpha)
print("Beta:", beta)

print("Alpha:", ', '.join(map(str, alpha)))
print("Beta:", ', '.join(map(str, beta)))


# # Test
# m_I = 5/2
# vector = nuclear_vector(m_I)

# print("Vector:", vector)

# # Test
# m_I1, m_S1 = 3/2, -1/2
# m_I2, m_S2 = 1/2, -1/2
# alpha, beta = transition_vector(m_I1, m_S1, m_I2, m_S2)

# print("Alpha:", alpha)
# print("Beta:", beta)

# alpha2 = [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]
# beta2 = [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]

# # Check if alpha = alpha2 and beta = beta2
# print("alpha == alpha2:", alpha == alpha2)
# print("beta == beta2:", beta == beta2)
