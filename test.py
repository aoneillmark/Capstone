import numpy as np
import matplotlib.pyplot as plt


print("Output of test")

# Generate a dummy plot to test if matplotlib is working
x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)
plt.figure()
plt.plot(x, y)
plt.show()
