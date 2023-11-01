import numpy as np

def nuclear_vector(m_I):
    # Define the basis vectors for nuclear spin
    # nuclear_basis = [7/2, 5/2, 3/2, 1/2, -1/2, -3/2, -5/2, -7/2]
    nuclear_basis = [-7/2, -5/2, -3/2, -1/2, 1/2, 3/2, 5/2, 7/2]
    # nuclear_basis = [7/2, 5/2, 3/2, 1/2, -7/2, -5/2, -3/2, -1/2]
    
    # Initialize the vector as a zero vector of length 8
    vector = [0] * len(nuclear_basis)
    
    # Set the corresponding entry for the given nuclear spin value
    vector[nuclear_basis.index(m_I)] = 1
    
    return vector

def electron_vector(m_S):
    # Define the basis vectors for electron spin
    electron_basis = [-1/2, 1/2]
    
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
m_I1, m_S1 = -7/2, -1/2
alpha = (kronecker(m_I1, m_S1))

m_I2, m_S2 = -7/2, 1/2
beta = (kronecker(m_I2, m_S2))

print("Kronecker product:")
print("Alpha:", list(alpha))
print("Beta: ", list(beta))

#####################################################################
#####################################################################

def state_to_index(m_s, m_e):
    # Lists of states for nuclear and electron spins
    nuclear_states = [-7/2, -5/2, -3/2, -1/2, 1/2, 3/2, 5/2, 7/2] 
    electron_states = [-1/2, 1/2]

    # Find the position of the states in the arrays
    nuclear_index = nuclear_states.index(m_s)
    electron_index = electron_states.index(m_e)

    # Calculate the index for the combined state
    index = 2 * nuclear_index + electron_index
    return index

def get_alpha_beta(initial_state, final_state):
    total_states = 16  # as there are 16 combined states

    # Getting indices of the initial and final states
    alpha_index = state_to_index(*initial_state)
    beta_index = state_to_index(*final_state)

    # Convert indices to array format
    alpha_array = [1 if i == alpha_index else 0 for i in range(total_states)]
    beta_array = [1 if i == beta_index else 0 for i in range(total_states)]

    return alpha_array, beta_array

# Example usage:
initial_state = (-7/2, -1/2)
final_state = (7/2, 1/2)
alpha1, beta1 = get_alpha_beta(initial_state, final_state)

print("Using initial ordering")
print(f"Alpha: {alpha1}")
print(f"Beta:  {beta1}")

#####################################################################
#####################################################################

def state_to_index2(m_s, m_e):
    combined_states = [
        # -1/2
        (-7/2, -1/2), (-5/2, -1/2), (-3/2, -1/2), (-1/2, -1/2),
        (1/2, -1/2), (3/2, -1/2), (5/2, -1/2), (7/2, -1/2),
        # +1/2
        (-7/2, 1/2), (-5/2, 1/2), (-3/2, 1/2), (-1/2, 1/2),
        (1/2, 1/2), (3/2, 1/2), (5/2, 1/2), (7/2, 1/2),
    ]

    return combined_states.index((m_s, m_e))

def get_alpha_beta2(initial_state, final_state):
    total_states = 16  # as there are 16 combined states

    # Getting indices of the initial and final states
    alpha_index = state_to_index2(*initial_state)
    beta_index = state_to_index2(*final_state)

    # Convert indices to array format
    alpha_array = [1 if i == alpha_index else 0 for i in range(total_states)]
    beta_array = [1 if i == beta_index else 0 for i in range(total_states)]

    return alpha_array, beta_array

# Example usage:
# initial_state = (1/2, -1/2)
# final_state = (3/2, -1/2)
initial_state = (-7/2, -1/2)
final_state = (-7/2, 1/2)
alpha, beta = get_alpha_beta2(initial_state, final_state)
print("Using other ordering method:")
print(f"Alpha: {alpha}")
print(f"Beta:  {beta}")

# print("Reverse the lists of alpha and beta")
# alpha.reverse()
# beta.reverse()
# print(f"Alpha: {alpha}")
# print(f"Beta:  {beta}")
