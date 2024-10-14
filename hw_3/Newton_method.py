import numpy as np
import matplotlib.pyplot as plt

# Forward Euler Method
def forward_euler(alpha, dt, t_max):
    t_values = np.arange(0, t_max, dt)
    y_values = np.zeros_like(t_values)
    y_values[0] = 1  # Initial condition y(0) = 1
    
    for i in range(1, len(t_values)):
        y_values[i] = y_values[i-1] * (1 - alpha * dt)
    
    return t_values, y_values

# Backward Euler Method
def backward_euler(alpha, dt, t_max):
    t_values = np.arange(0, t_max, dt)
    y_values = np.zeros_like(t_values)
    y_values[0] = 1  # Initial condition y(0) = 1
    
    for i in range(1, len(t_values)):
        y_values[i] = y_values[i-1] / (1 + dt * alpha)
    
    return t_values, y_values

# Trapezoidal Method
def trapezoid_method(alpha, dt, t_max):
    t_values = np.arange(0, t_max, dt)
    y_values = np.zeros_like(t_values)
    y_values[0] = 1  # Initial condition y(0) = 1
    
    for i in range(1, len(t_values)):
        y_values[i] = y_values[i-1] * ((1 - (dt * alpha) / 2) / (1 + (dt * alpha) / 2))
    
    return t_values, y_values

# Exact Solution
def exact_solution(alpha, t_values):
    return np.exp(-alpha * t_values)

# Parameters
alpha = 1.0
t_max = 10.0
dt_stable = 0.1   # Stable time step
dt_stable_1 = 1.0 # Stable
dt_stable_2 = 1.5 # Stable
dt_unstable = 2.5 # Unstable time step

# Get results for Forward Euler
t_stable_fe, y_numerical_stable_fe = forward_euler(alpha, dt_stable, t_max)
t_stable_fe_1, y_numerical_stable_fe_1 = forward_euler(alpha, dt_stable_1, t_max)
t_stable_fe_2, y_numerical_stable_fe_2 = forward_euler(alpha, dt_stable_2, t_max)
t_unstable_fe, y_numerical_unstable_fe = forward_euler(alpha, dt_unstable, t_max)

# Get results for Backward Euler
t_stable_be, y_numerical_stable_be = backward_euler(alpha, dt_stable, t_max)
t_stable_be_1, y_numerical_stable_be_1 = backward_euler(alpha, dt_stable_1, t_max)
t_stable_be_2, y_numerical_stable_be_2 = backward_euler(alpha, dt_stable_2, t_max)
t_unstable_be, y_numerical_unstable_be = backward_euler(alpha, dt_unstable, t_max)

# Get results for Trapezoidal Method
t_stable_trap, y_numerical_stable_trap = trapezoid_method(alpha, dt_stable, t_max)
t_stable_trap_1, y_numerical_stable_trap_1 = trapezoid_method(alpha, dt_stable_1, t_max)
t_stable_trap_2, y_numerical_stable_trap_2 = trapezoid_method(alpha, dt_stable_2, t_max)
t_unstable_trap, y_numerical_unstable_trap = trapezoid_method(alpha, dt_unstable, t_max)

# Exact Solution for comparison
y_exact_stable = exact_solution(alpha, t_stable_fe)
y_exact_stable_1 = exact_solution(alpha, t_stable_fe_1)
y_exact_stable_2 = exact_solution(alpha, t_stable_fe_2)
y_exact_unstable = exact_solution(alpha, t_unstable_fe)

# Plot Forward Euler
plt.figure(figsize=(10, 6))
plt.plot(t_stable_fe, y_numerical_stable_fe, label="Forward Euler Numerical (dt = 0.1)", color='darkred', linestyle='--', marker='o')
plt.plot(t_stable_fe_1, y_numerical_stable_fe_1, label="Forward Euler Numerical (dt = 1.0)", color='darkgreen', linestyle='--', marker='x')
plt.plot(t_stable_fe_2, y_numerical_stable_fe_2, label="Forward Euler Numerical (dt = 1.5)", color='purple', linestyle='--', marker='s')
plt.plot(t_unstable_fe, y_numerical_unstable_fe, label="Forward Euler Numerical (dt = 2.5)", color='orange', linestyle='--', marker='d')
plt.plot(t_stable_fe, y_exact_stable, label="Exact (dt = 0.1)", color='blue', linestyle='-')
plt.title("Forward Euler Method: Numerical vs Exact Solution", fontsize=14)
plt.xlabel("Time (t)", fontsize=12)
plt.ylabel("y(t)", fontsize=12)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Plot Backward Euler
plt.figure(figsize=(10, 6))
plt.plot(t_stable_be, y_numerical_stable_be, label="Backward Euler Numerical (dt = 0.1)", color='darkred', linestyle='--', marker='o')
plt.plot(t_stable_be_1, y_numerical_stable_be_1, label="Backward Euler Numerical (dt = 1.0)", color='darkgreen', linestyle='--', marker='x')
plt.plot(t_stable_be_2, y_numerical_stable_be_2, label="Backward Euler Numerical (dt = 1.5)", color='purple', linestyle='--', marker='s')
plt.plot(t_unstable_be, y_numerical_unstable_be, label="Backward Euler Numerical (dt = 2.5)", color='orange', linestyle='--', marker='d')
plt.plot(t_stable_be, y_exact_stable, label="Exact (dt = 0.1)", color='blue', linestyle='-')

plt.title("Backward Euler Method: Numerical vs Exact Solution", fontsize=14)
plt.xlabel("Time (t)", fontsize=12)
plt.ylabel("y(t)", fontsize=12)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Plot Trapezoidal Method
plt.figure(figsize=(10, 6))
plt.plot(t_stable_trap, y_numerical_stable_trap, label="Trapezoidal Numerical (dt = 0.1)", color='darkred', linestyle='--', marker='o')
plt.plot(t_stable_trap_1, y_numerical_stable_trap_1, label="Trapezoidal Numerical (dt = 1.0)", color='darkgreen', linestyle='--', marker='x')
plt.plot(t_stable_trap_2, y_numerical_stable_trap_2, label="Trapezoidal Numerical (dt = 1.5)", color='purple', linestyle='--', marker='s')
plt.plot(t_unstable_trap, y_numerical_unstable_trap, label="Trapezoidal Numerical (dt = 2.5)", color='orange', linestyle='--', marker='d')
plt.plot(t_stable_trap, y_exact_stable, label="Exact (dt = 0.1)", color='blue', linestyle='-')
plt.title("Trapezoidal Method: Numerical vs Exact Solution", fontsize=14)
plt.xlabel("Time (t)", fontsize=12)
plt.ylabel("y(t)", fontsize=12)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
