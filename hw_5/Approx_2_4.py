import numpy as np
import matplotlib.pyplot as plt

# function f(x) = -x^3/6 + x/6
def f(x):
    return -(x ** 3) / 6 + x / 6

# define the phi values using np.where for array compatibility
def phi_1(x):
    return np.where(x < 0.25, 4 * x, np.where((x >= 0.25) & (x <= 0.5), 2 - 4 * x, 0))

def phi_2(x):
    return np.where((x > 0.25) & (x < 0.5), 4 * x - 1, np.where((x >= 0.5) & (x <= 0.75), 3 - 4 * x, 0))

def phi_3(x):
    return np.where((x >= 0.5) & (x < 0.75), 4 * x - 2, np.where((x >= 0.75) & (x <= 1), 4 - 4 * x, 0))

# define f_h using the modified phi functions
def f_h(x):
    return ((5 / 128) * phi_1(x)) + ((1 / 16) * phi_2(x)) + ((7 / 128) * phi_3(x))

# Create grid for plots
x_fine = np.linspace(0, 1, 100)
f_fine = f(x_fine)
f_approx = f_h(x_fine)

# Plot f(x)
plt.plot(x_fine, f_fine, label='$f(x) = -\\frac{x^{3}}{6} + \\frac{x}{6}$', color='blue')

# Plot f_h(x) with appropriate labeling
plt.plot(x_fine, f_approx, label='$f_h(x) = \\frac{5}{128} \\phi_1(x) + \\frac{1}{16} \\phi_2(x) + \\frac{7}{128} \\phi_3(x)$', color='red', linestyle='--')

# Labels and title
plt.xlabel('x')
plt.ylabel('y')
plt.title('Function $f(x)$ and Finite Element Interpolant $f_h(x)$ Part 2.4')
plt.legend()
plt.grid(True)
plt.show()
