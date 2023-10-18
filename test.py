# import numpy as np
# import matplotlib.pyplot as plt
from mpi4py import MPI

# Get mpi rank
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

# Print test to see if mpi is working by outputting ranks and size
print('rank:', rank, ', size:', size)


# # Generate a dummy plot to test if matplotlib is working
# x = np.linspace(0, 2*np.pi, 100)
# y = np.sin(x)

# # Plot and save figure
# plt.figure()
# plt.plot(x, y)
# plt.savefig('test.png')

