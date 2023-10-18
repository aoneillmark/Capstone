import numpy as np
import matplotlib.pyplot as plt
import mpi4py

# Get mpi rank
rank = mpi4py.MPI.COMM_WORLD.Get_rank()
print(rank)
print("Output of test")

# Generate a dummy plot to test if matplotlib is working
x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)

# Plot and save figure
plt.figure()
plt.plot(x, y)
plt.savefig('test.png')

