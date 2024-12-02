from scipy.integrate import dblquad

# Define the given function
def f(x, y):
    return 0.25 * (1 - x - y + x**2 * y)

# Define the interpolated function
def L1(x):
    return (1 - x) / 2

def L2(x):
    return (1 + x) / 2

def L11(x, y):
    return L1(x) * L1(y)

def L12(x, y):
    return L1(x) * L2(y)

def L21(x, y):
    return L2(x) * L1(y)

def L22(x, y):
    return L2(x) * L2(y)

def f_interp(x, y):
    # Node values
    f_11 = f(-1, -1)
    f_12 = f(-1, 1)
    f_21 = f(1, -1)
    f_22 = f(1, 1)
    # Interpolation
    return f_11 * L11(x, y) + f_12 * L12(x, y) + f_21 * L21(x, y) + f_22 * L22(x, y)

# Perform numerical integration of the original function
integral_f, _ = dblquad(f, -1, 1, lambda x: -1, lambda x: 1)

# Perform numerical integration of the interpolated function
integral_f_interp, _ = dblquad(f_interp, -1, 1, lambda x: -1, lambda x: 1)

# Print results
print(f"Numerical integration of the original function: {integral_f}")
print(f"Numerical integration of the interpolated function: {integral_f_interp}")
