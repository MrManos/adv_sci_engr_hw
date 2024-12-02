import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad
from mpl_toolkits.mplot3d import Axes3D

# Define the 1D Lagrange polynomials
def L1(x):
    return (1 - x) / 2

def L2(x):
    return (1 + x) / 2

# Define the 2D Lagrange polynomials
def L11(x, y):
    return L1(x) * L1(y)

def L12(x, y):
    return L1(x) * L2(y)

def L21(x, y):
    return L2(x) * L1(y)

def L22(x, y):
    return L2(x) * L2(y)

# Define the given function
def f(x, y):
    return 0.25 * (1 - x - y + x**2 * y)

# Interpolation using Lagrange polynomials
def f_interp(x, y):
    # Node values
    f_11 = f(-1, -1)
    f_12 = f(-1, 1)
    f_21 = f(1, -1)
    f_22 = f(1, 1)
    # Interpolation
    return f_11 * L11(x, y) + f_12 * L12(x, y) + f_21 * L21(x, y) + f_22 * L22(x, y)

# Numerical integration of the original function
integral_f, _ = dblquad(f, -1, 1, lambda x: -1, lambda x: 1)

# Numerical integration of the interpolated function
integral_f_interp, _ = dblquad(f_interp, -1, 1, lambda x: -1, lambda x: 1)

# Create grid for plotting
x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)
Z_interp = np.array([[f_interp(xi, yi) for xi, yi in zip(row_x, row_y)] for row_x, row_y in zip(X, Y)])

# Plot the original function (3D)
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
ax.set_title("Original Function (3D)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("f(x, y)")
plt.show()

# Plot the interpolated function (3D)
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z_interp, cmap='viridis', edgecolor='none', alpha=0.8)
ax.set_title("Interpolated Function (3D)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("Interpolated f(x, y)")
plt.show()

# Print the numerical integration results
print(f"Numerical integration of original function: {integral_f}")
print(f"Numerical integration of interpolated function: {integral_f_interp}")
