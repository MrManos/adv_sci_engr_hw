import numpy as np
import matplotlib.pyplot as plt

# Define the original function f(x) = sin(pi * x)
def f(x):
    return np.sin(np.pi * x)

# Define the mesh and the nodal values
x_nodes = np.array([0, 0.25, 0.5, 0.75, 1.0])
f_nodes = f(x_nodes)

# Create a fine grid for plotting the original function
x_fine = np.linspace(0, 1, 100)
f_fine = f(x_fine)

# Plot f(x)
plt.plot(x_fine, f_fine, label='$f(x) = \sin(\pi x)$', color='blue')

# Plot f_h(x) as a piecewise linear interpolant
plt.plot(x_nodes, f_nodes, label='$f_h(x)$', color='red', linestyle='--', marker='o')

# Plot the mesh points
plt.scatter(x_nodes, f_nodes, color='black', label='Mesh Points', zorder=5)

# Labels and title
plt.xlabel('x')
plt.ylabel('y')
plt.title('Function $f(x)$ and Finite Element Interpolant $f_h(x)$ Problem 1')
plt.legend()
plt.grid(True)
plt.show()
