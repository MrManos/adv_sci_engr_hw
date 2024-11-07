import numpy as np
import matplotlib.pyplot as plt

# function f(x) = - x^3/6 + x/6
def f(x):
    return -(x ** 3)/6 + x/6

# define the y_h function
def f_h(x):
    return ((2/(np.pi ** 3)) * np.sin(np.pi * x)) - (1/(4*np.pi ** 3) * np.sin(2*np.pi * x)) + (2/(27*np.pi ** 3 ) * np.sin(3*np.pi*x))


# Create a fine grid for plotting the original function
x_fine = np.linspace(0, 1, 100)
f_fine = f(x_fine)
f_approx = f_h(x_fine)

# Plot f(x)
plt.plot(x_fine, f_fine, label='$f(x) = \\frac{x^{3}}{6} + \\frac{x}{6}$', color='blue')

# Plot f_h(x)
plt.plot(x_fine, f_approx, label='$f_h(x) = \\frac{2}{\\pi^3} \\sin(\\pi x) - \\frac{1}{4 \\pi^3} \\sin(2 \\pi x) + \\frac{2}{27 \\pi^3} \\sin(3 \\pi x)$'
, color='red', linestyle='--')

# Labels and title
plt.xlabel('x')
plt.ylabel('y')
plt.title('Function $f(x)$ and Finite Element Interpolant $f_h(x)$ Part 2.3')
plt.legend()
plt.grid(True)
plt.show()
